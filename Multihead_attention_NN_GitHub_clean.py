import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_curve, auc)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
import joblib

ROOT = Path(r"D:\MELD_Project\repo_style_final")
FEATURES = ROOT / "features"
OUT = ROOT / "results" / "github_clean_new"  
EMOTIONS = ("anger", "disgust", "sadness", "joy", "neutral", "surprise", "fear")
LABEL_ID = {name: i for i, name in enumerate(EMOTIONS)}
DIMENSIONS = {"V1": 512, "V2": 768, "V3": 25, "V4": 25, "A2": 512}
CHECKPOINT = OUT / "checkpoint.pt"


def load_data():
    """Join existing files by utterance ID; never regenerate embeddings."""
    sources = {
        "audio": (FEATURES / "audio_features_github.xlsx", ["V1", "V3", "V4"]),
        "text": (FEATURES / "text_features.xlsx", ["V2"]),
        "video": (FEATURES / "video_features.xlsx", ["A2"]),
    }
    tables = {}
    for name, (path, fields) in sources.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        df = pd.read_excel(path, usecols=["sample_id", "Emotion", *fields])
        if df["sample_id"].isna().any() or df["sample_id"].duplicated().any():
            raise ValueError(f"Duplicate or missing sample_id in {path}")
        df["sample_id"] = df["sample_id"].astype(str)
        df["Emotion"] = df["Emotion"].astype(str).str.strip().str.lower()
        tables[name] = df.set_index("sample_id")

    base = tables["text"]
    if len(base) != 9989:
        raise ValueError(f"Expected 9989 train utterances; got {len(base)}")
    for name in ("audio", "video"):
        if set(base.index) != set(tables[name].index):
            raise ValueError(f"{name} sample IDs do not match text")
        tables[name] = tables[name].reindex(base.index)
        if not tables[name]["Emotion"].equals(base["Emotion"]):
            raise ValueError(f"{name} emotion labels do not match text")

    def vectors(series, dim):
        result = np.stack([np.fromstring(str(s).strip().strip("[]"), sep=",", dtype=np.float32)
                           for s in series])
        if result.shape != (9989, dim) or not np.isfinite(result).all():
            raise ValueError(f"Invalid {series.name} features: {result.shape}, expected (9989,{dim})")
        return torch.from_numpy(result)

    x = [vectors(tables[source][field], DIMENSIONS[field]) for field, source in
         (("V1", "audio"), ("V2", "text"), ("V3", "audio"),
          ("V4", "audio"), ("A2", "video"))]
    if not base["Emotion"].isin(LABEL_ID).all():
        raise ValueError("Unknown emotion labels")
    y = torch.tensor(base["Emotion"].map(LABEL_ID).to_numpy(), dtype=torch.long)
    print("Loaded and aligned", len(y), "samples | V1=512 V2=768 V3=25 V4=25 A2=512", flush=True)
    return x, y, base.index.to_numpy()


class EmotionAnalysisModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha1 = nn.Parameter(torch.rand(512))
        self.alpha2 = nn.Parameter(torch.rand(512))
        self.alpha3 = nn.Parameter(torch.rand(512))
        self.bilstm_voice = nn.LSTM(512, 256, 1, batch_first=True, bidirectional=True)
        self.bilstm_text = nn.LSTM(512, 256, 1, batch_first=True, bidirectional=True)
        self.bilstm_video = nn.LSTM(512, 256, 1, batch_first=True, bidirectional=True)
        self.multihead_attention = nn.MultiheadAttention(512, 8, batch_first=True)
        self.fc = nn.Linear(512, 7)
        self.project_V3 = nn.Linear(25, 512)
        self.project_V4 = nn.Linear(25, 512)
        self.project_V2 = nn.Linear(768, 512)  # Required for your RoBERTa vectors.

    def forward(self, V1, V2, V3, V4, A2, extract=False):
        voice = V1 * self.alpha1 + self.project_V3(V3) * self.alpha2 + self.project_V4(V4) * self.alpha3
        F_voice, _ = self.bilstm_voice(voice.unsqueeze(1))
        F_text, _ = self.bilstm_text(self.project_V2(V2).unsqueeze(1))
        F_video, _ = self.bilstm_video(A2.unsqueeze(1))
        combined = torch.cat((F_voice, F_text, F_video), dim=1)
        attended, _ = self.multihead_attention(combined, combined, combined)
        fused = attended[:, -1, :]  # GitHub uses LAST token, not mean pooling.
        return fused if extract else self.fc(fused)


def save_checkpoint(model, optimizer, epoch, losses):
    temp = OUT / "checkpoint.tmp"
    torch.save({"epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "losses": losses}, temp)
    os.replace(temp, CHECKPOINT)


def train(model, x, y, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)  
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  
    start, losses = 1, []
    if CHECKPOINT.is_file():
        state = torch.load(CHECKPOINT, map_location=y.device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        losses = state["losses"]
        start = state["epoch"] + 1
        print(f"Resuming at epoch {start} from existing checkpoint", flush=True)
    if start > epochs:
        print(f"Checkpoint already at epoch {start - 1}; skipping training", flush=True)
        return losses
    model.train()
    for epoch in range(start, epochs + 1):
        optimizer.zero_grad()
        logits = model(*x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # GitHub uses gradient clipping.
        optimizer.step()  # ONE full-dataset update per epoch, exactly as GitHub.
        losses.append(float(loss.detach().cpu()))
        if epoch == start or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs} | loss={losses[-1]:.6f}", flush=True)
        if epoch % 25 == 0 or epoch == epochs:
            save_checkpoint(model, optimizer, epoch, losses)
    return losses
@torch.inference_mode()
def extract_features(model, x):
    model.eval()
    return model(*x, extract=True).cpu().numpy()


def evaluate_svm(fused, y, ids):
    # GitHub SPLIT: occurs AFTER supervised training on every sample.
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)
    svm = SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True)
    svm.fit(fused[train_idx], y[train_idx])
    actual = y[test_idx]
    predicted = svm.predict(fused[test_idx])
    probabilities = svm.predict_proba(fused[test_idx])
    scores = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_precision": float(precision_score(actual, predicted, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(actual, predicted, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(actual, predicted, average="weighted", zero_division=0)),
        "evaluation": "GitHub procedure: neural model trained on all SVM test labels; NOT independent test accuracy",
    }
    (OUT / "svm_metrics.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    pd.DataFrame({"sample_id": ids[test_idx], "actual": actual,
                  "predicted": predicted}).to_csv(OUT / "svm_predictions.csv", index=False)
    joblib.dump(svm, OUT / "svm_model.joblib")

    cm = confusion_matrix(actual, predicted, labels=range(7))
    pd.DataFrame(cm, index=EMOTIONS, columns=EMOTIONS).to_csv(OUT / "svm_confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(cm, cmap="Blues")
    for i in range(7):
        for j in range(7):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    ax.set_xticks(range(7), EMOTIONS, rotation=45, ha="right")
    ax.set_yticks(range(7), EMOTIONS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrix.png", dpi=160); plt.close(fig)

    binary = label_binarize(actual, classes=range(7))
    fig, ax = plt.subplots(figsize=(9, 6))
    for k in range(7):
        if np.unique(binary[:, k]).size < 2:
            continue
        column = int(np.where(svm.classes_ == k)[0][0])
        fpr, tpr, _ = roc_curve(binary[:, k], probabilities[:, column])
        ax.plot(fpr, tpr, label=f"{EMOTIONS[k]} AUC={auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(OUT / "svm_roc.png", dpi=160); plt.close(fig)


    for name in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"):
        print(f"  {name}: {scores[name]:.4f}", flush=True)
    print("Saved:", OUT, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=9988)
    parser.add_argument("--eval-only", action="store_true", help="Score saved checkpoint now; no training")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    features, labels, sample_ids = load_data()
    features = [v.to(device) for v in features]
    labels = labels.to(device)
    model = EmotionAnalysisModel().to(device)
    if args.eval_only:
        if not CHECKPOINT.is_file():
            raise FileNotFoundError(f"No checkpoint to evaluate: {CHECKPOINT}")
        state = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        losses = state["losses"]
        print(f"Evaluating saved neural epoch {state['epoch']}", flush=True)
    else:
        losses = train(model, features, labels, args.epochs)
    pd.DataFrame({"epoch": np.arange(1, len(losses) + 1), "loss": losses}).to_csv(
        OUT / "neural_training_loss.csv", index=False)
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(losses) + 1), losses)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Training loss")
    fig.tight_layout(); fig.savefig(OUT / "neural_training_loss.png", dpi=160); plt.close(fig)
    fused = extract_features(model, features)
    torch.save(model.state_dict(), OUT / "neural_model_latest.pth")
    evaluate_svm(fused, labels.cpu().numpy(), sample_ids)


if __name__ == "__main__":
    main()
