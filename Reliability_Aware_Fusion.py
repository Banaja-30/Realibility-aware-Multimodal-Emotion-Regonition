import argparse
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from Multihead_attention_NN_GitHub_clean import (
    EmotionAnalysisModel, EMOTIONS, ROOT, load_data
)

SOURCE = ROOT / "results" / "github_clean_new" / "checkpoint.pt"

# New results folder
OUT = ROOT / "results" / "reliability_aware_balanced_fix"

# Load the existing successful reliability checkpoint
SAVED = (
    ROOT / "results"
    / "reliability_aware_new"
    / "reliability_checkpoint_new.pt"
)
SEED = 42


class ReliabilityHead(nn.Module):
    def __init__(self, pretrained_fc):
        super().__init__()

        # One learnable reliability scorer for each modality
        self.scorers = nn.ModuleList([
            nn.Linear(512, 1) for _ in range(3)
        ])

        # Start with equal modality weights
        for layer in self.scorers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        # Copy the pretrained classification head
        self.fc = copy.deepcopy(pretrained_fc)

        # IMPORTANT: the copied head must be trainable
        self.fc.requires_grad_(True)

        # Learn how much all-modality context should contribute
        self.pool_strength = nn.Parameter(torch.tensor(0.0))

    def forward(self, tokens, attention):
        # tokens: [batch, 3, 512]
        # Order: audio, text, video

        scores = torch.cat(
            [
                layer(tokens[:, i, :])
                for i, layer in enumerate(self.scorers)
            ],
            dim=1
        )

        # Reliability weights for each sample
        weights = torch.softmax(scores, dim=1)

        # Preserve feature scale when weights are equal
        weighted = tokens * (3.0 * weights.unsqueeze(-1))

        # Existing pretrained eight-head attention
        attended, _ = attention(
            weighted,
            weighted,
            weighted,
            need_weights=False
        )

        # Original GitHub output
        original_fused = attended[:, -1, :]

        # Reliability-weighted representation of all three modalities
        context = (
            attended * weights.unsqueeze(-1)
        ).sum(dim=1)

        # Learn whether additional context improves classification
        fused = (
            original_fused
            + torch.tanh(self.pool_strength) * context
        )

        return self.fc(fused), fused, weights

    


def load_pretrained(device):
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Existing GitHub checkpoint not found: {SOURCE}")
    state = torch.load(SOURCE, map_location="cpu", weights_only=False)
    model = EmotionAnalysisModel().to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval().requires_grad_(False)
    print(f"Using GitHub checkpoint at epoch {state['epoch']} (read-only)")
    return model, int(state["epoch"])


@torch.no_grad()
def make_tokens(model, features, device, batch_size=256):
    """Cache BiLSTM outputs ONCE; avoids repeated feature extraction/training."""
    output = []
    for start in range(0, len(features[0]), batch_size):
        block = [arr[start:start + batch_size].to(device) for arr in features]
        v1, v2, v3, v4, a2 = block
        voice = (v1 * model.alpha1 + model.project_V3(v3) * model.alpha2
                 + model.project_V4(v4) * model.alpha3)
        audio_token, _ = model.bilstm_voice(voice.unsqueeze(1))
        text_token, _ = model.bilstm_text(model.project_V2(v2).unsqueeze(1))
        video_token, _ = model.bilstm_video(a2.unsqueeze(1))
        output.append(torch.cat([audio_token, text_token, video_token], dim=1).cpu())
    return torch.cat(output, dim=0).contiguous()


@torch.no_grad()
def validate(adapter, attention, tokens, labels, indices, device, batch_size):
    adapter.eval()
    all_pred = []
    for start in range(0, len(indices), batch_size):
        chosen = indices[start:start + batch_size]
        logits, _, _ = adapter(tokens[chosen].to(device), attention)
        all_pred.append(logits.argmax(dim=1).cpu().numpy())
    pred = np.concatenate(all_pred)
    actual = labels[indices]
    return (accuracy_score(actual, pred),
            f1_score(actual, pred, average="weighted", zero_division=0),
            f1_score(actual, pred, average="macro", zero_division=0))


def train_adapter(adapter, attention, tokens, labels, train_idx, args, device):
    fit_idx, val_idx = train_test_split(
        train_idx, test_size=0.1, stratify=labels[train_idx], random_state=SEED
    )
    ds = TensorDataset(tokens[fit_idx], torch.from_numpy(labels[fit_idx]).long())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam([
    {
        "params": adapter.scorers.parameters(),
        "lr": 3e-4
    },
    {
        "params": adapter.fc.parameters(),
        "lr": 1e-4
    },
    {
        "params": [adapter.pool_strength],
        "lr": 1e-2
    }
])
    loss_fn = nn.CrossEntropyLoss()
    best_score = (-1., -1., -1.)
    best_state, best_epoch = None, 0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        adapter.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _, _ = adapter(x, attention)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        score = validate(adapter, attention, tokens, labels, val_idx, device, args.batch_size)
        print(f"Epoch {epoch:02d} | train_loss={total_loss / len(fit_idx):.4f} | "
              f"val_acc={score[0]:.4f} | val_weighted_f1={score[1]:.4f} | "
              f"val_macro_f1={score[2]:.4f}", flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(adapter.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping; restoring best head")
                break
    adapter.load_state_dict(best_state)
    return best_epoch, best_score


@torch.no_grad()
def extract(adapter, attention, tokens, device, batch_size):
    adapter.eval()
    embeddings, weights = [], []
    for start in range(0, len(tokens), batch_size):
        _, fused, reliability = adapter(tokens[start:start + batch_size].to(device), attention)
        embeddings.append(fused.cpu().numpy())
        weights.append(reliability.cpu().numpy())
    return np.concatenate(embeddings), np.concatenate(weights)


@torch.no_grad()
def original_embeddings(attention, tokens, device, batch_size):
    result = []
    for start in range(0, len(tokens), batch_size):
        x = tokens[start:start + batch_size].to(device)
        attended, _ = attention(x, x, x, need_weights=False)
        result.append(attended[:, -1, :].cpu().numpy())
    return np.concatenate(result)


def select_reliability_svm(embeddings, labels, train_idx):
    # Never tune on the final SVM evaluation subset.
    fit_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.15,
        stratify=labels[train_idx],
        random_state=SEED
    )

    def class_weights(indices, power):
        classes, counts = np.unique(
            labels[indices], return_counts=True
        )
        balanced = len(indices) / (len(classes) * counts)
        return {
            int(cls): float(weight ** power)
            for cls, weight in zip(classes, balanced)
        }

    def evaluate(C, power):
        svm = SVC(
            kernel="rbf",
            C=C,
            gamma="scale",
            class_weight=class_weights(fit_idx, power)
        )
        svm.fit(embeddings[fit_idx], labels[fit_idx])
        pred = svm.predict(embeddings[val_idx])
        actual = labels[val_idx]

        return (
            accuracy_score(actual, pred),
            f1_score(actual, pred, average="weighted",
                     zero_division=0),
            f1_score(actual, pred, average="macro",
                     zero_division=0)
        )

    # Original configuration: C=1, fully balanced weights.
    reference = evaluate(1.0, 1.0)
    best_score = reference
    best_C, best_power = 1.0, 1.0

    print("Original internal SVM validation:", reference)

    for power in (1.0, 0.75, 0.5, 0.25, 0.0):
        for C in (1.0, 3.0, 10.0):
            score = evaluate(C, power)

            print(
                f"C={C}, balance_power={power}: "
                f"acc={score[0]:.4f}, "
                f"weighted_f1={score[1]:.4f}, "
                f"macro_f1={score[2]:.4f}"
            )

            # Improve accuracy while protecting both F1 metrics
            # compared with the original internal validation model.
            if (
                score[0] > best_score[0]
                and score[1] >= reference[1]
                and score[2] >= reference[2]
            ):
                best_score = score
                best_C, best_power = C, power

    print(
        f"Selected C={best_C}, "
        f"balance_power={best_power}, "
        f"validation_metrics={best_score}"
    )

    # Retrain on all available SVM training samples.
    final_svm = SVC(
        kernel="rbf",
        C=best_C,
        gamma="scale",
        class_weight=class_weights(train_idx, best_power)
    )
    final_svm.fit(
        embeddings[train_idx],
        labels[train_idx]
    )

    return final_svm

def fit_svm(embeddings, labels, ids, train_idx, test_idx, prefix, save_model=False):

    if prefix == "reliability":

        # Select and fit the best SVM using internal validation
        svm = select_reliability_svm(
            embeddings,
            labels,
            train_idx
        )

    else:

        # Keep the original GitHub SVM baseline
        svm = SVC(
            kernel="rbf",
            C=1.0,
            class_weight="balanced"
        )

        svm.fit(
            embeddings[train_idx],
            labels[train_idx]
        )

    pred = svm.predict(embeddings[test_idx])
    true = labels[test_idx]

    metrics = {
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, pred, average="weighted", zero_division=0)),
        "protocol": "GitHub random SVM split; PRETRAINED BACKBONE SAW ALL LABELS; NOT independent test",
    }
    (OUT / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"sample_id": ids[test_idx], "actual": true, "predicted": pred}).to_csv(
        OUT / f"{prefix}_predictions.csv", index=False
    )
    cm = confusion_matrix(true, pred, labels=np.arange(7))
    pd.DataFrame(cm, index=EMOTIONS, columns=EMOTIONS).to_csv(
        OUT / f"{prefix}_confusion_matrix.csv"
    )
    if save_model:
        joblib.dump(svm, OUT / "svm_reliability.joblib")
    print(f"{prefix}: accuracy={metrics['accuracy']:.4f}, "
          f"macro_f1={metrics['macro_f1']:.4f}, weighted_f1={metrics['weighted_f1']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    features, target, sample_ids = load_data()
    labels = target.numpy()
    train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=0.2, random_state=SEED)
    base, source_epoch = load_pretrained(device)
    tokens = make_tokens(base, features, device, args.batch_size)
    adapter = ReliabilityHead(base.fc).to(device)

    if args.eval_only:
        if not SAVED.is_file():
            raise FileNotFoundError(f"Train adapter first: {SAVED}")
        saved = torch.load(SAVED, map_location="cpu", weights_only=False)
        adapter.load_state_dict(saved["model"], strict=True)
        print("Loaded reliability checkpoint from epoch", saved["epoch"])
    else:
        best_epoch, best_score = train_adapter(adapter, base.multihead_attention,
                                              tokens, labels, train_idx, args, device)
        torch.save({"model": adapter.state_dict(), "epoch": best_epoch,
                    "validation_metrics": best_score, "source_epoch": source_epoch}, SAVED)
        print("Saved reliability checkpoint:", SAVED)

    fused, weights = extract(adapter, base.multihead_attention, tokens, device, args.batch_size)
    np.savez_compressed(OUT / "reliability_embeddings.npz", embeddings=fused,
                        labels=labels, sample_ids=np.asarray(sample_ids, dtype=str),
                        weights=weights)
    pd.DataFrame({"sample_id": sample_ids,
                  "audio_weight": weights[:, 0],
                  "text_weight": weights[:, 1],
                  "video_weight": weights[:, 2]}).to_csv(
        OUT / "modality_weights.csv", index=False
    )

    base_fused = original_embeddings(base.multihead_attention, tokens, device, args.batch_size)
    baseline = fit_svm(base_fused, labels, sample_ids, train_idx, test_idx,
                       "baseline_same_checkpoint")
    proposed = fit_svm(fused, labels, sample_ids, train_idx, test_idx,
                       "reliability", save_model=True)
    print(f"Accuracy change vs SAME-checkpoint baseline: "
          f"{(proposed['accuracy'] - baseline['accuracy']) * 100:+.2f} percentage points")
    print("Results:", OUT)
    print("WARNING: Scores are repository-protocol results, NOT independent test scores.")


if __name__ == "__main__":
    main()
