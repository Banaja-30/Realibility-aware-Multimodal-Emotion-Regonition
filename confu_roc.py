
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from Temporal_Emotion_Model_final import (
    load_embeddings, dialogue_groups, make_splits, make_combined_features
)

ROOT = Path(r"D:\MELD_Project\repo_style_final")
RESULTS = ROOT / "results" / "temporal_class_corrected"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(exist_ok=True, parents=True)

EMOTIONS = [
    "anger", "disgust", "sadness",
    "joy", "neutral", "surprise", "fear"
]

# CONFUSION MATRIX — PROPOSED MODEL

predictions = pd.read_csv(
    RESULTS / "reliability_temporal_predictions.csv"
)

y_true = predictions["actual"].to_numpy(dtype=int)
y_pred = predictions["predicted"].to_numpy(dtype=int)

cm = confusion_matrix(y_true, y_pred, labels=range(7))

fig, ax = plt.subplots(figsize=(9, 7))
image = ax.imshow(cm, cmap="Blues")
fig.colorbar(image, ax=ax, label="Utterances")

ax.set(
    xticks=range(7),
    yticks=range(7),
    xticklabels=EMOTIONS,
    yticklabels=EMOTIONS,
    xlabel="Predicted Emotion",
    ylabel="Actual Emotion",
    title="Proposed Model — Confusion Matrix"
)

plt.setp(ax.get_xticklabels(), rotation=40, ha="right")

for i in range(7):
    for j in range(7):
        ax.text(
            j, i, cm[i, j],
            ha="center", va="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black",
            fontsize=9
        )

fig.tight_layout()
fig.savefig(
    FIGURES / "Proposed_Model_Confusion_Matrix.png",
    dpi=300
)
plt.close(fig)

# ROC CURVES — PROPOSED MODEL

x, y, sample_ids = load_embeddings(
    ROOT / "results" / "reliability_aware_balanced_fix"
    / "reliability_embeddings.npz"
)

with np.load(RESULTS / "temporal_embeddings.npz") as z:
    contextual = z["embeddings"]
    assert np.array_equal(
        z["sample_ids"].astype(str), sample_ids
    )

fit, val, test = make_splits(
    y, dialogue_groups(sample_ids)[1], "github"
)

assert np.array_equal(
    predictions["sample_id"].astype(str),
    sample_ids[test]
)
assert np.array_equal(y_true, y[test])

gain = json.loads(
    (RESULTS / "selection.json").read_text()
)["gain"]

vectors = make_combined_features(
    x, contextual, fit, gain
)

proposed_model = joblib.load(
    RESULTS / "svm_temporal.joblib"
)

proba = proposed_model.predict_proba(vectors[test])

assert np.array_equal(
    proposed_model.classes_, np.arange(7)
)

fig, ax = plt.subplots(figsize=(9, 7))

for i, emotion in enumerate(EMOTIONS):
    binary_labels = (y_true == i).astype(int)

    fpr, tpr, _ = roc_curve(
        binary_labels, proba[:, i]
    )

    ax.plot(
        fpr, tpr,
        linewidth=2,
        label=f"{emotion.title()} (AUC = {auc(fpr, tpr):.3f})"
    )

ax.plot(
    [0, 1], [0, 1], "k--",
    linewidth=1, label="Chance"
)

ax.set(
    xlim=(0, 1),
    ylim=(0, 1.01),
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="Proposed Model — ROC Curves (7 Emotions)"
)

ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.2)

fig.tight_layout()
fig.savefig(
    FIGURES / "Proposed_Model_ROC_Curves.png",
    dpi=300
)
plt.close(fig)

print("Saved:")
print(FIGURES / "Proposed_Model_Confusion_Matrix.png")
print(FIGURES / "Proposed_Model_ROC_Curves.png")
