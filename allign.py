
import pandas as pd
from pathlib import Path

FEATURE_DIR = Path(r"D:\MELD_Project\repo_style_final\features")

AUDIO_FILE = FEATURE_DIR / "audio_features_github.xlsx"
TEXT_FILE = FEATURE_DIR / "text_features.xlsx"
VIDEO_FILE = FEATURE_DIR / "video_features.xlsx"

files = {
    "audio": AUDIO_FILE,
    "text": TEXT_FILE,
    "video": VIDEO_FILE
}

features = {}

# 1. Load and validate each modality
for modality, path in files.items():
    df = pd.read_excel(path)

    required = {"sample_id", "Emotion"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{modality}: missing columns {missing}"
        )

    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    df["Emotion"] = (
        df["Emotion"].astype(str).str.strip().str.lower()
    )

    if df["sample_id"].duplicated().any():
        raise ValueError(f"{modality}: duplicate sample IDs")

    if df["sample_id"].isna().any():
        raise ValueError(f"{modality}: missing sample IDs")

    features[modality] = df

    print(
        f"{modality}: {len(df)} rows, "
        f"{df['sample_id'].nunique()} unique IDs"
    )

# 2. Compare IDs across modalities
audio_ids = set(features["audio"]["sample_id"])
text_ids = set(features["text"]["sample_id"])
video_ids = set(features["video"]["sample_id"])

print("\n--- ALIGNMENT REPORT ---")

for name, ids in [
    ("Text", text_ids),
    ("Video", video_ids)
]:
    print(f"\nMissing from {name} compared with audio:",
          len(audio_ids - ids))
    print(f"Extra in {name} compared with audio:",
          len(ids - audio_ids))

if not (audio_ids == text_ids == video_ids):
    raise ValueError(
        "ALIGNMENT FAILED: IDs differ across modalities. "
        "Do not train the fusion model yet."
    )

print("\nAll three modalities contain matching IDs.")

# 3. Check emotion labels
audio_labels = (
    features["audio"].set_index("sample_id")["Emotion"]
)

for modality in ["text", "video"]:
    labels = (
        features[modality]
        .set_index("sample_id")["Emotion"]
    )

    labels = labels.reindex(audio_labels.index)

    mismatches = (
        audio_labels != labels
    )

    if mismatches.any():
        print(
            f"\n{modality}: label mismatches:",
            int(mismatches.sum())
        )
        print(
            audio_labels[mismatches].head()
        )
        raise ValueError(
            f"Emotion labels do not match in {modality}"
        )

print("All emotion labels match.")

print("\nALIGNMENT CHECK PASSED.")
