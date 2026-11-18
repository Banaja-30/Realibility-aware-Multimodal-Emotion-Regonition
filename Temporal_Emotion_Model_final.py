import argparse
import copy
import json
from pyexpat import model
import re
from pathlib import Path
from xml.parsers.expat import model

import joblib
from matplotlib.pyplot import clf
import numpy as np
import pandas as pd
from sklearn import svm
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.svm import SVC
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

ROOT = Path(r"D:\MELD_Project\repo_style_final")
FEATURES = ROOT / "results" / "reliability_aware_balanced_fix" / "reliability_embeddings.npz"
EMOTIONS = ("anger", "disgust", "sadness", "joy", "neutral", "surprise", "fear")
SEED = 42
ID_RX = re.compile(r"^([^:]+):dia(\d+)_utt(\d+)$")


def load_embeddings(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing saved embeddings: {path}")
    with np.load(path, allow_pickle=False) as z:
        x = z["embeddings"].astype(np.float32)
        y = z["labels"].astype(np.int64)
        ids = z["sample_ids"].astype(str)
    if x.ndim != 2 or x.shape[1] != 512 or len(x) != len(y) or len(ids) != len(y):
        raise ValueError(f"Invalid embeddings/labels/IDs shapes: {x.shape}, {y.shape}, {ids.shape}")
    if not np.isfinite(x).all() or not np.isin(y, range(7)).all() or len(set(ids)) != len(ids):
        raise ValueError("Invalid embeddings, labels, or duplicate utterance IDs")
    return x, y, ids


def dialogue_groups(ids):
    grouped, key_for_row = {}, []
    for row, sid in enumerate(ids):
        m = ID_RX.fullmatch(sid)
        if not m:
            raise ValueError(f"Invalid sample ID {sid!r}: expected train:dia0_utt0")
        key = (m[1], int(m[2]))
        key_for_row.append(f"{key[0]}:dia{key[1]}")
        grouped.setdefault(key, []).append((int(m[3]), row))
    sequences = []
    for key in sorted(grouped):
        pairs = sorted(grouped[key])
        if len(set(u for u, _ in pairs)) != len(pairs):
            raise ValueError(f"Duplicate Utterance_ID in dialogue {key}")
        sequences.append(np.array([i for _, i in pairs], dtype=np.int64))
    return sequences, np.asarray(key_for_row)


def make_splits(y, dialogue_keys, protocol):
    all_rows = np.arange(len(y))
    if protocol == "github":
        train, test = train_test_split(all_rows, test_size=.2, random_state=SEED)
        fit, val = train_test_split(train, test_size=.15,
                                   stratify=y[train], random_state=SEED)
    else:
        outer = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED)
        tr, te = next(outer.split(all_rows, y, dialogue_keys))
        train, test = all_rows[tr], all_rows[te]
        inner = GroupShuffleSplit(n_splits=1, test_size=.15, random_state=SEED+1)
        fi, va = next(inner.split(train, y[train], dialogue_keys[train]))
        fit, val = train[fi], train[va]
        assert not (set(dialogue_keys[fit]) & set(dialogue_keys[val]))
        assert not (set(dialogue_keys[fit]) & set(dialogue_keys[test]))
        assert not (set(dialogue_keys[val]) & set(dialogue_keys[test]))
    return np.asarray(fit), np.asarray(val), np.asarray(test)


def subset_groups(groups, chosen, size):
    mask = np.zeros(size, dtype=bool)
    mask[chosen] = True
    return [part for seq in groups if len(part := seq[mask[seq]])]


class DialogueData(Dataset):
    def __init__(self, x, y, groups):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.groups = groups

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        idx = torch.from_numpy(self.groups[i])
        return self.x[idx], self.y[idx], idx


def collate(rows):
    a, b, idx = zip(*rows)
    lengths = torch.tensor([len(x) for x in a], dtype=torch.long)
    return (pad_sequence(a, batch_first=True),
            pad_sequence(b, batch_first=True, padding_value=-100),
            pad_sequence(idx, batch_first=True, padding_value=-1), lengths)


def loader(x, y, groups, batch, shuffle=False):
    return DataLoader(DialogueData(x, y, groups), batch_size=batch,
                      shuffle=shuffle, collate_fn=collate)


class TemporalGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(512, 256, batch_first=True, bidirectional=True)
        self.strength = nn.Parameter(torch.tensor(0.1))
        self.fc = nn.Linear(512, 7)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        result, _ = self.gru(packed)
        context, _ = pad_packed_sequence(result, batch_first=True,
                                         total_length=x.shape[1])
        fused = x + self.strength * context
        return self.fc(fused), fused



@torch.no_grad()
def extract(model, x, y, groups, chosen, device, batch):
    model.eval()

    result = np.empty((len(chosen), 512), dtype=np.float32)

    positions = {
        int(row): i for i, row in enumerate(chosen)
    }

    touched = np.zeros(len(chosen), dtype=bool)

    # Preserve complete dialogues instead of splitting them
    # into separate training, validation and test fragments.
    chosen_mask = np.zeros(len(y), dtype=bool)
    chosen_mask[chosen] = True

    seqs = [
        seq for seq in groups
        if chosen_mask[seq].any()
    ]

    for inputs, _, row_ids, lengths in loader(
        x, y, seqs, batch
    ):
        _, contextual = model(
            inputs.to(device),
            lengths
        )

        contextual = contextual.cpu().numpy()

        for b, length in enumerate(lengths.tolist()):
            for k in range(length):

                utterance_id = int(row_ids[b, k])

                # Keep representations only for requested rows.
                if utterance_id not in positions:
                    continue

                index = positions[utterance_id]

                if touched[index]:
                    raise RuntimeError(
                        "Duplicate utterance in temporal extraction"
                    )

                result[index] = contextual[b, k]
                touched[index] = True

    if not touched.all():
        raise RuntimeError(
            "Missing utterance in temporal extraction"
        )

    return result



def metrics(actual, pred):
    return (float(accuracy_score(actual, pred)),
            float(f1_score(actual, pred, average="weighted", zero_division=0)),
            float(f1_score(actual, pred, average="macro", zero_division=0)))



def svm_for(y, C, power, gamma="scale"):
    weights = None

    if power:
        classes, counts = np.unique(
            y, return_counts=True
        )
        weights = {
            int(label): float(
                (len(y) / (len(classes) * count)) ** power
            )
            for label, count in zip(classes, counts)
        }

    return SVC(
    kernel="rbf",
    C=C,
    gamma=gamma,
    class_weight=weights
)


def svm_gamma(features, fitting_rows, multiplier):
    # Compute gamma using training features only.
    # Multiplier 1.0 reproduces sklearn's gamma="scale".
    if multiplier == 1.0:
        return "scale"

    variance = float(
        np.var(features[fitting_rows], dtype=np.float64)
    )

    return multiplier / max(
        features.shape[1] * variance,
        1e-12
    )



def train(model, x, y, groups, fit, val, args, device, out):

    # Identify utterances whose labels may be used for training.
    fit_mask = np.zeros(len(y), dtype=bool)
    fit_mask[fit] = True

    # Keep the complete dialogue whenever it contains
    # at least one training utterance.
    training_groups = [
        seq for seq in groups
        if fit_mask[seq].any()
    ]

    dl = loader(
        x,
        y,
        training_groups,
        args.batch,
        shuffle=True
    )

    fit_mask_tensor = torch.from_numpy(
        fit_mask
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4
    )

    loss_fn = nn.CrossEntropyLoss()

    best = (-1, -1, -1)
    best_epoch = 0
    best_state = None
    stale = 0
    history = []

    for epoch in range(1, args.epochs + 1):

        model.train()

        total_loss = 0.0
        total_items = 0

        for inputs, targets, row_ids, lengths in dl:

            inputs = inputs.to(device)
            targets = targets.to(device)

            logits, _ = model(inputs, lengths)

            
            selected = fit_mask_tensor[
                row_ids.clamp(min=0).to(device)
            ]

            valid = (targets != -100) & selected

            if not valid.any():
                continue

            loss = loss_fn(
                logits[valid],
                targets[valid]
            )

            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            n = int(valid.sum().item())

            total_loss += loss.item() * n
            total_items += n

        
        # CHECKPOINT SELECTION MATCHES FINAL SVM
        # Use the complete-dialogue temporal representations.

        contextual = x.copy()

        contextual[fit] = extract(
            model, x, y, groups,
            fit, device, args.batch
        )

        contextual[val] = extract(
            model, x, y, groups,
            val, device, args.batch
        )

        # Original reliability 512-D + temporal 512-D.
        # The same feature design used by final evaluation.
        candidate_features = make_combined_features(
            x,
            contextual,
            fit,
            gain=0.25
        )

        # Match the successful validation configuration:
        # C=10, gamma_multiplier=0.25, uniform weights.
        gamma = svm_gamma(
            candidate_features,
            fit,
            0.25
        )

        clf = svm_for(
            y[fit],
            C=10.0,
            power=0.0,
            gamma=gamma
        )

        clf.fit(
            candidate_features[fit],
            y[fit]
        )

        score = metrics(
            y[val],
            clf.predict(candidate_features[val])
        )


        row = {
            "epoch": epoch,
            "loss": total_loss / total_items,
            "val_accuracy": score[0],
            "val_weighted_f1": score[1],
            "val_macro_f1": score[2]
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"loss={row['loss']:.4f} | "
            f"SVM val_acc={score[0]:.4f} "
            f"weighted_f1={score[1]:.4f} "
            f"macro_f1={score[2]:.4f}",
            flush=True
        )

        if score > best:
            best = score
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )
            stale = 0

        else:
            stale += 1

            if stale >= args.patience:
                print(
                    "Early stopping; restoring best "
                    "SVM-validation checkpoint"
                )
                break

    model.load_state_dict(best_state)

    pd.DataFrame(history).to_csv(
        out / "training_history.csv",
        index=False
    )

    return best_epoch




def make_combined_features(base, contextual, calibration_idx, gain):
    """
    Preserve the original 512-D reliability embedding and
    append a separately scaled 512-D temporal residual.

    gain=0 returns the original reliability embedding unchanged.

    Scaling statistics are calculated only from calibration_idx,
    never from validation or test samples.
    """
    if gain == 0.0:
        return base

    delta = contextual - base

    train_base = base[calibration_idx]
    train_delta = delta[calibration_idx]

    base_scale = float(np.sqrt(
        np.mean(np.var(train_base, axis=0, dtype=np.float64))
    ))

    delta_scale = float(np.sqrt(
        np.mean(np.var(train_delta, axis=0, dtype=np.float64))
    ))

    scale = base_scale / max(delta_scale, 1e-8)
    scale = min(scale, 100.0)

    combined = np.concatenate(
        [base, gain * scale * delta],
        axis=1
    )

    return combined.astype(np.float32)
# Order: anger, disgust, sadness, joy,
# neutral, surprise, fear
CLASS_BIAS = np.array(
    [-0.08, 0.0, 0.08, 0.03, 0.0, 0.0, 0.0]
)

def corrected_prediction(clf, features):
    probabilities = clf.predict_proba(features)

    scores = np.log(
        np.maximum(probabilities, 1e-12)
    ) + CLASS_BIAS

    return clf.classes_[scores.argmax(axis=1)]

def choose_svm(base, contextual, y, fit, val):

    choices = []

    for gain, C in [(0.0, 30.0), (0.25, 10.0)]:

        vectors = make_combined_features(
            base, contextual, fit, gain
        )

        gamma = svm_gamma(vectors, fit, 0.25)

        clf = svm_for(
            y[fit],
            C=C,
            power=0.0,
            gamma=gamma
        )

        if gain > 0:
            clf.set_params(
                probability=True,
                random_state=42
            )

        clf.fit(vectors[fit], y[fit])

        if gain > 0:
            pred = corrected_prediction(
            clf, vectors[val]
    )
        else:
            pred = clf.predict(vectors[val])
        choice = (
            gain,
            C,
            0.0,
            0.25,
            metrics(y[val], pred)
        )

        choices.append(choice)
        print("Validation:", choice, flush=True)

    baseline, temporal = choices

    improved = (
        temporal[4][0] > baseline[4][0]
        and temporal[4][1] >= baseline[4][1]
        and temporal[4][2] >= baseline[4][2]
    )

    if improved:
        print("SELECT: temporal SVM")
        return baseline, temporal

    print("SELECT: tuned reliability SVM")
    return baseline, baseline


def evaluate(label, vectors, y, ids, train, test,
             C, power, output, save=False,
             gamma_multiplier=1.0):

    gamma = svm_gamma(
        vectors,
        train,
        gamma_multiplier
    )

    svm = svm_for(
        y[train],
        C=C,
        power=power,
        gamma=gamma
    )
    if label == "reliability_temporal":
        svm.set_params(probability=True, random_state=42)

    svm.fit(
        vectors[train],
        y[train]
    )

    if label == "reliability_temporal":
        pred = corrected_prediction(
        svm, vectors[test]
    )
    else:
        pred = svm.predict(vectors[test])
    acc, weighted, macro = metrics(
        y[test],
        pred
    )

    report = {
        "accuracy": acc,
        "weighted_f1": weighted,
        "macro_f1": macro,
        "C": C,
        "balance_power": power,
        "gamma_multiplier": gamma_multiplier,
        "warning": (
            "Pretrained backbone saw all 9989 labels; "
            "NOT independent test accuracy"
        )
    }

    (output / f"{label}_metrics.json").write_text(
        json.dumps(report, indent=2)
    )

    pd.DataFrame({
        "sample_id": ids[test],
        "actual": y[test],
        "predicted": pred
    }).to_csv(
        output / f"{label}_predictions.csv",
        index=False
    )

    pd.DataFrame(
        confusion_matrix(
            y[test],
            pred,
            labels=range(7)
        ),
        index=EMOTIONS,
        columns=EMOTIONS
    ).to_csv(
        output / f"{label}_confusion_matrix.csv"
    )

    if save:
        joblib.dump(
            svm,
            output / "svm_temporal.joblib"
        )

    print(
        f"{label}: "
        f"accuracy={acc:.4f}, "
        f"macro_f1={macro:.4f}, "
        f"weighted_f1={weighted:.4f}"
    )

    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--protocol", choices=("github", "grouped"), default="github")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--input", type=Path, default=FEATURES)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    output = (
    args.out
    or ROOT / "results" / f"temporal_svm_tuned_{args.protocol}"
)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, y, ids = load_embeddings(args.input)
    groups, dialogue_keys = dialogue_groups(ids)
    fit, val, test = make_splits(y, dialogue_keys, args.protocol)
    train_idx = np.concatenate([fit, val])
    print(f"Device={device} | protocol={args.protocol} | utterances={len(y)} "
          f"| dialogues={len(groups)} | fit={len(fit)} val={len(val)} test={len(test)}")
    model = TemporalGRU().to(device)
    
    
    checkpoint = output / "temporal_checkpoint.pt"

    if args.eval_only:
        checkpoint = (
            ROOT / "results"
        / "temporal_svm_tuned_github"
        / "temporal_checkpoint.pt"
    )

        state = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False
        )

        if state["protocol"] != args.protocol:
            raise ValueError(
                "Checkpoint protocol does not match"
            )

        if not np.array_equal(state["sample_ids"], ids):
            raise ValueError(
                "Checkpoint sample IDs do not match"
            )

        model.load_state_dict(
            state["model"],
            strict=True
        )

        print(
            "Loaded temporal checkpoint epoch:",
            state["epoch"]
        )

    else:

        # Preserve the original warm-start behavior.
        if args.protocol == "github":

            head_file = (
                ROOT / "results"
                / "reliability_aware_new"
                / "reliability_checkpoint_new.pt"
            )

            if head_file.is_file():

                trained_head = torch.load(
                    head_file,
                    map_location="cpu",
                    weights_only=False
                )["model"]

                if (
                    "fc.weight" in trained_head
                    and "fc.bias" in trained_head
                ):
                    with torch.no_grad():

                        model.fc.weight.copy_(
                            trained_head["fc.weight"].to(device)
                        )

                        model.fc.bias.copy_(
                            trained_head["fc.bias"].to(device)
                        )

        best_epoch = train(
            model,
            x,
            y,
            groups,
            fit,
            val,
            args,
            device,
            output
        )

        torch.save(
            {
                "model": model.state_dict(),
                "epoch": best_epoch,
                "protocol": args.protocol,
                "sample_ids": ids
            },
            checkpoint
        )

        print("Saved temporal checkpoint:", checkpoint)
    contextual = np.empty_like(x)

    for subset in (fit, val, test):
        contextual[subset] = extract(
            model, x, y, groups,
            subset, device, args.batch
    )

    np.savez_compressed(
        output / "temporal_embeddings.npz",
        embeddings=contextual,
        labels=y,
        sample_ids=ids
)
    
    base_choice, temporal_choice = choose_svm(
        x, contextual, y, fit, val
    )

    print(
        "Best baseline (gain, C, power, "
        "gamma_multiplier, metrics):",
        base_choice
    )

    print(
        "Chosen model (gain, C, power, "
        "gamma_multiplier, metrics):",
        temporal_choice
    )

    gain, C, power, gamma_multiplier, _ = temporal_choice

    vectors = make_combined_features(
        x,
        contextual,
        fit,
        gain
    )

    # Preserve original 90.84% reference configuration.
    fixed_reference = evaluate(
        "reliability_C10",
        x, y, ids, train_idx, test,
        10.0, 0.0, output
    )

    # Baseline with independently tuned SVM parameters.
    tuned_reference = evaluate(
        "reliability_tuned",
        x, y, ids, train_idx, test,
        base_choice[1],
        base_choice[2],
        output,
        gamma_multiplier=base_choice[3]
    )

    # Evaluate the validation-selected temporal candidate,
    # or the fallback if no temporal candidate qualified.
    temporal = evaluate(
        "reliability_temporal",
        vectors, y, ids, train_idx, test,
        C, power, output,
        save=True,
        gamma_multiplier=gamma_multiplier
    )

    (output / "selection.json").write_text(
        json.dumps(
            {
                "gain": gain,
                "C": C,
                "power": power,
                "gamma_multiplier": gamma_multiplier,
                "best_validation": temporal_choice[4],
                "fixed_reference": fixed_reference,
                "tuned_reference": tuned_reference,
                "temporal": temporal,
                "protocol": args.protocol
            },
            indent=2
        )
    )

    print(
        "Temporal change vs fixed C10 baseline: "
        f"{100 * (temporal['accuracy'] - fixed_reference['accuracy']):+.2f} "
        "percentage points"
    )

    print(
        "Temporal change vs tuned baseline: "
        f"{100 * (temporal['accuracy'] - tuned_reference['accuracy']):+.2f} "
        "percentage points"
    )



if __name__ == "__main__":
    main()
