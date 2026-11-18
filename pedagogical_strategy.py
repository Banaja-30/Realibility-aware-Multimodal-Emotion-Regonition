"""Pedagogical strategy demo for the verified temporal_class_corrected run.

This adds a downstream, rule-based stage. Does not train or modify any
existing multimodal model or checkpoint. The source CSV is an *offline*, randomly
selected 1,998-row evaluation subset: it cannot establish chronological learner
session trends. Consequently offline rows use temporal_state='unavailable'.

Run from D:\\MELD_Project\\repo_style_final:
    python pedagogical_strategy.py

Optional for an alternate project location:
    python pedagogical_strategy.py --root D:\\MELD_Project\\repo_style_final
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\MELD_Project\repo_style_final")
VALID_EMOTIONS = {"anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"}
NEGATIVE_SIGNALS = {"anger", "fear", "sadness"}

# Prototype *pedagogical* policies, NOT learned from MELD or clinically validated.
POLICIES = {
    "anger": ("Simplify and offer choice", "Break the explanation into smaller steps; offer a different example."),
    "disgust": ("Change explanation format", "Try a different example or presentation style; do not assume why."),
    "fear": ("Lower-pressure guidance", "Start with an easy example and allow the learner to control the pace."),
    "joy": ("Positive reinforcement", "Reinforce progress, then offer an optional follow-up challenge."),
    "neutral": ("Clear step-by-step teaching", "Answer normally with one worked example and a check for understanding."),
    "sadness": ("Supportive pacing", "Use manageable steps and offer optional encouragement without presuming feelings."),
    "surprise": ("Clarify unexpected ideas", "Explain surprising outcomes and provide a clear comparison or example."),
}


def temporal_state_from_session(recent_predictions: list[str]) -> str:
    """Simple optional *online session* trend, never used on the offline CSV.

    Call only with chronologically ordered predictions from a real learner
    session, not arbitrary MELD test rows. This is a prototype heuristic.
    """
    recent = [str(x).strip().lower() for x in recent_predictions]
    if len(recent) < 2:
        return "insufficient_history"
    window = recent[-3:]
    if sum(label in NEGATIVE_SIGNALS for label in window) >= 2:
        return "repeated_negative_signals"
    if len(recent) >= 3 and recent[-1] in {"joy", "neutral"} and any(
        label in NEGATIVE_SIGNALS for label in recent[-3:-1]
    ):
        return "possibly_improving"
    return "mixed_or_stable"


def select_strategy(
    emotion: str | None,
    needs_confirmation_heuristic: bool = False,
    temporal_state: str = "unavailable",
) -> dict[str, str]:
    """Return the suggested teaching policy, not a claim of learner emotion."""
    label = "" if emotion is None else str(emotion).strip().lower()
    
    if needs_confirmation_heuristic or label not in VALID_EMOTIONS:

        return {
            "strategy": "Supportive explanation with optional check-in",
            "instructions": (
                "Answer the learner's academic question directly first. "
                "Use simple, supportive language and explain the topic "
                "in manageable steps. Do not assume the learner's "
                "emotional state. Ask at most one optional question "
                "after providing the explanation."
        ),
        "reason": "uncertain_prediction_or_unavailable_emotion",
    }
    strategy, instructions = POLICIES[label]
    if temporal_state == "repeated_negative_signals":
        strategy = "Slow down and offer a simpler route"
        instructions = (
            "Offer a shorter explanation with one example, then ask whether "
            "the learner wants to continue or change approach. Do not label their mood."
        )
    return {"strategy": strategy, "instructions": instructions, "reason": "rule_based_prototype",
             }


def checked_flag(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ValueError(f"Unexpected needs_confirmation_heuristic value: {value!r}")


def run(root: Path) -> None:
    input_dir = root / "results" / "temporal_class_corrected" / "confidence_estimation_outputs"
    source = input_dir / "confidence_predictions.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Run confidence_estimation.py first; missing: {source}")

    df = pd.read_csv(source, dtype={"sample_id": str})
    required = {"sample_id", "predicted_emotion", "uncalibrated_confidence_score",
                "needs_confirmation_heuristic"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Confidence CSV missing columns: {sorted(missing)}")
    if len(df) == 0 or df["sample_id"].isna().any() or df["sample_id"].duplicated().any():
        raise ValueError("Missing/duplicate IDs or empty input")
    if not df["predicted_emotion"].isin(VALID_EMOTIONS).all():
        raise ValueError("Unknown predicted emotion in input")
    scores = pd.to_numeric(df["uncalibrated_confidence_score"], errors="coerce")
    if scores.isna().any() or not scores.between(0.0, 1.0).all():
        raise ValueError("Invalid uncalibrated confidence score")

    flags = [checked_flag(v) for v in df["needs_confirmation_heuristic"]]
    # These are sampled evaluation rows, NOT complete chronological sessions.
    # Do not invent temporal learner-state estimates from this file.
    policies = [select_strategy(emotion, flag, temporal_state="unavailable")
                for emotion, flag in zip(df["predicted_emotion"], flags)]
    result = pd.DataFrame({
        "sample_id": df["sample_id"],
        "predicted_emotion": df["predicted_emotion"],
        "uncalibrated_confidence_score": scores,
        "needs_confirmation_heuristic": flags,
        "temporal_state": "unavailable_offline_subset",
        "pedagogical_strategy": [p["strategy"] for p in policies],
        "strategy_instructions": [p["instructions"] for p in policies],
        "strategy_reason": [p["reason"] for p in policies],
    })
    output_dir = root / "results" / "temporal_class_corrected" / "pedagogical_strategy_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / "strategy_predictions.csv"
    result.to_csv(dest, index=False)
    summary = {
        "rows": len(result),
        "source": str(source),
        "strategy_counts": dict(Counter(result["pedagogical_strategy"])),
        "uncertainty_check_in_rows": int(sum(flags)),
        "temporal_state": "unavailable: 1,998 rows are an offline nonconsecutive evaluation subset",
        "warning": (
            "Rule-based demonstration only; MELD has no pedagogical ground truth. "
            "Confidence flags are heuristic and scores are not independently calibrated. "
            "No upstream model or saved checkpoints modified."
        ),
    }
    summary_file = output_dir / "strategy_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"STRATEGY SELECTION: PASS ({len(result)} rows)")
    print("No retraining and no existing model files changed.")
    print("Temporal state: unavailable for offline sampled rows (not a full live session).")
    print("Output:", dest)
    print("Output:", summary_file)
    print("\nFirst five strategies:")
    print(result[["sample_id", "predicted_emotion", "pedagogical_strategy"]].head().to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Add rule-based teaching strategies to MELD confidence output")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.root)


if __name__ == "__main__":
    main()
