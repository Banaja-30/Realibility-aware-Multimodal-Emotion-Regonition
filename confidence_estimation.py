"""Read-only inference audit for the saved MELD temporal classifier.

Reconstructs the exact 1024-D temporal concatenation from the uploaded
Temporal_Emotion_Model_final.py design. Does not retrain or alter checkpoints.

Usage in Windows CMD (from D:\\MELD_Project\\repo_style_final):
    python confidence_estimation.py

Outputs an evaluation-protocol confidence-score CSV and summary JSON in a new
confidence_estimation_outputs folder, after verifying saved predictions match.
Scores are NOT independently calibrated probabilities, and the GitHub-protocol
results are NOT an independent held-out evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(r"D:\MELD_Project\repo_style_final")
EMOTIONS = ("anger", "disgust", "sadness", "joy", "neutral", "surprise", "fear")
# Must match corrected_prediction() in Temporal_Emotion_Model_final(3).py.
CLASS_BIAS = np.array([-0.08, 0.0, 0.08, 0.03, 0.0, 0.0, 0.0], dtype=np.float64)
SEED = 42


def load_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Required feature file was not found: {path}")
    with np.load(path, allow_pickle=False) as z:
        for key in ("embeddings", "labels", "sample_ids"):
            if key not in z.files:
                raise ValueError(f"{path.name} lacks required array: {key}")
        x = z["embeddings"].astype(np.float32)
        y = z["labels"].astype(np.int64)
        ids = z["sample_ids"].astype(str)
    if x.ndim != 2 or x.shape[1] != 512 or len(x) != len(y) or len(x) != len(ids):
        raise ValueError(f"Invalid saved feature shapes in {path}: {x.shape}, {y.shape}, {ids.shape}")
    if not np.isfinite(x).all() or not np.isin(y, np.arange(7)).all():
        raise ValueError(f"Invalid numerical values or emotion labels in {path}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate sample IDs in {path}")
    return x, y, ids


def make_combined_features(base: np.ndarray, contextual: np.ndarray,
                           fit_idx: np.ndarray, gain: float) -> np.ndarray:
    """Exact feature formula used by the uploaded temporal model."""
    if gain == 0:
        return base
    delta = contextual - base
    train_base = base[fit_idx]
    train_delta = delta[fit_idx]
    base_scale = float(np.sqrt(np.mean(np.var(train_base, axis=0, dtype=np.float64))))
    delta_scale = float(np.sqrt(np.mean(np.var(train_delta, axis=0, dtype=np.float64))))
    scale = min(base_scale / max(delta_scale, 1e-8), 100.0)
    return np.concatenate([base, gain * scale * delta], axis=1).astype(np.float32)


def run(root: Path) -> None:
    result_dir = root / "results" / "temporal_class_corrected"
    base_path = root / "results" / "reliability_aware_balanced_fix" / "reliability_embeddings.npz"
    temporal_path = result_dir / "temporal_embeddings.npz"
    selection_path = result_dir / "selection.json"
    model_path = result_dir / "svm_temporal.joblib"
    prior_predictions_path = result_dir / "reliability_temporal_predictions.csv"

    required = [base_path, temporal_path, selection_path, model_path, prior_predictions_path]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required saved result missing: {path}\n"
                "No changes were made. Check whether this is the same model run."
            )

    base, y, ids = load_embeddings(base_path)
    contextual, y_context, ids_context = load_embeddings(temporal_path)
    if not np.array_equal(ids, ids_context) or not np.array_equal(y, y_context):
        raise RuntimeError("Base and temporal embeddings have different sample IDs or labels. Stopping.")

    # Recreate the exact GitHub-protocol split from the uploaded source.
    if len(y) != 9989:
        raise ValueError(f"Expected 9,989 original-train utterances, found {len(y)}")
    all_idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(all_idx, test_size=0.2, random_state=SEED)
    fit_idx, _val_idx = train_test_split(
        train_idx, test_size=0.15, stratify=y[train_idx], random_state=SEED
    )

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("protocol") != "github":
        raise ValueError(f"Unexpected protocol: {selection.get('protocol')!r}")
    gain = float(selection["gain"])
    if not np.isclose(gain, 0.25):
        raise ValueError(f"Expected temporal gain 0.25 from verified run, found {gain}")

    x = make_combined_features(base, contextual, fit_idx, gain)
    clf = joblib.load(model_path)  # Load only your own trusted checkpoint.
    if not getattr(clf, "probability", False):
        raise ValueError("Saved classifier lacks probability=True")
    if int(clf.n_features_in_) != 1024 or x.shape[1] != 1024:
        raise ValueError(f"Expected 1024-D input, model={clf.n_features_in_}, features={x.shape[1]}")
    if not np.array_equal(clf.classes_, np.arange(7)):
        raise ValueError(f"Unexpected emotion label order in model: {clf.classes_}")

    original_proba = clf.predict_proba(x[test_idx]).astype(np.float64)
    if original_proba.shape != (len(test_idx), 7) or not np.isfinite(original_proba).all():
        raise ValueError("Invalid probability output")

    # The original final prediction applies additive class biases to log(SVC probabilities).
    biased_logits = np.log(np.maximum(original_proba, 1e-12)) + CLASS_BIAS[None, :]
    # Normalize these biased scores for display, without claiming calibration.
    stabilized = np.exp(biased_logits - biased_logits.max(axis=1, keepdims=True))
    adjusted_scores = stabilized / stabilized.sum(axis=1, keepdims=True)
    pred = clf.classes_[biased_logits.argmax(axis=1)].astype(np.int64)

    # Fail safely if the chosen run or its stored features do not reproduce its predictions.
    previous = pd.read_csv(prior_predictions_path)
    for col in ("sample_id", "actual", "predicted"):
        if col not in previous.columns:
            raise ValueError(f"Saved predictions missing column: {col}")
    matches = (
        len(previous) == len(test_idx)
        and np.array_equal(previous["sample_id"].astype(str).to_numpy(), ids[test_idx])
        and np.array_equal(previous["actual"].to_numpy(dtype=np.int64), y[test_idx])
        and np.array_equal(previous["predicted"].to_numpy(dtype=np.int64), pred)
    )
    if not matches:
        raise RuntimeError(
            "Saved predictions do NOT match reconstructed results. "
            "Do not use these confidence scores: verify source embeddings/checkpoint first. "
            "No files were written."
        )

    predicted_score = adjusted_scores[np.arange(len(test_idx)), pred]
    two_highest = np.partition(adjusted_scores, kth=5, axis=1)[:, -2:]
    margin = two_highest.max(axis=1) - two_highest.min(axis=1)
    entropy = -(adjusted_scores * np.log(np.maximum(adjusted_scores, 1e-12))).sum(axis=1) / np.log(7)
    # Heuristic only: these cutoffs have NOT been tuned or validated for deployment.
    needs_confirmation = (predicted_score < 0.60) | (margin < 0.15)

    result = pd.DataFrame({
        "sample_id": ids[test_idx],
        "actual_emotion": [EMOTIONS[i] for i in y[test_idx]],
        "predicted_emotion": [EMOTIONS[i] for i in pred],
        "uncalibrated_confidence_score": np.round(predicted_score, 6),
        "top_two_score_margin": np.round(margin, 6),
        "normalized_entropy": np.round(entropy, 6),
        "needs_confirmation_heuristic": needs_confirmation,
    })
    for i, emotion in enumerate(EMOTIONS):
        result[f"adjusted_score_{emotion}"] = np.round(adjusted_scores[:, i], 6)
        result[f"raw_svc_probability_{emotion}"] = np.round(original_proba[:, i], 6)

    report = {
        "model_folder": str(result_dir),
        "data_protocol": "github (original MELD train-only pool, internal split)",
        "evaluated_rows": len(test_idx),
        "prediction_match_with_saved_file": True,
        "accuracy_reproduced": float(np.mean(pred == y[test_idx])),
        "mean_uncalibrated_confidence_score": float(predicted_score.mean()),
        "needs_confirmation_heuristic_fraction": float(needs_confirmation.mean()),
        "notes": [
            "Confidence scores are based on existing SVC probabilities with the original class bias.",
            "They are not independently calibrated or validated on unseen data.",
            "The original supervised backbone saw all 9989 labels; this is NOT independent test accuracy.",
            "The 0.60 confidence and 0.15 margin cutoffs are demonstration-only heuristics.",
            "No existing model, feature, or result file was changed.",
        ],
    }

    out = result_dir / "confidence_estimation_outputs"
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "confidence_predictions.csv", index=False)
    (out / "confidence_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("MODEL VERIFIED: temporal_class_corrected")
    print(f"Prediction audit: PASS ({len(test_idx)} rows match saved predictions)")
    print(f"Accuracy reproduced: {report['accuracy_reproduced'] * 100:.2f}%")
    print("Confidence: uncalibrated demonstration scores (NOT independent test assessment)")
    print("Output:", out / "confidence_predictions.csv")
    print("Output:", out / "confidence_summary.json")
    print("\nFirst five results:")
    print(result[["sample_id", "predicted_emotion", "uncalibrated_confidence_score",
                  "needs_confirmation_heuristic"]].head().to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit saved temporal model confidence scores")
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="Path to repo_style_final folder")
    args = parser.parse_args()
    run(args.root)


if __name__ == "__main__":
    main()
