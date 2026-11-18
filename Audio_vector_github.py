
from __future__ import annotations

import ast
import json
import shutil
import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from torchaudio.transforms import MFCC
from transformers import Wav2Vec2Model, Wav2Vec2Processor



ROOT = Path(r"D:\MELD_Project")

RAW_DIR = ROOT / "raw" / "MELD.Raw"

CSV_FILES = {
    "train": RAW_DIR / "train_sent_emo.csv",
    "dev": RAW_DIR / "dev_sent_emo.csv",
    "test": RAW_DIR / "test_sent_emo.csv",
}

AUDIO_CACHE = ROOT / "cached_audio"

FEATURE_DIR = ROOT / "repo_style_final" / "features"

OUT_EXCEL = FEATURE_DIR / "audio_features_github.xlsx"

BACKUP_EXCEL = FEATURE_DIR / "audio_features_train_backup.xlsx"

REMOVED_IDS_CSV = FEATURE_DIR / "audio_missing_sample_ids.csv"

BATCH_CACHE = FEATURE_DIR / "audio_github_merged_batches"

MODEL_ID = "facebook/wav2vec2-base-960h"

SAMPLE_RATE = 16000
LIBROSA_DEFAULT_SR = 22050

ROWS_PER_BATCH = 25

EXPECTED_ROWS = {
    "train": 9989,
    "dev": 1109,
    "test": 2610,
}

VERSION = "repo_audio_merged_v2_npy16k"



def load_waveform(name: str, split: str) -> np.ndarray:

    path = AUDIO_CACHE / split / name.replace(".mp4", ".npy")

    if not path.is_file():
        raise FileNotFoundError(f"Missing cached audio file: {path}")

    y = np.load(path, allow_pickle=False)

    if (
        y.ndim != 1
        or y.size < 400
        or not np.issubdtype(y.dtype, np.number)
    ):
        raise ValueError(f"Invalid cached waveform: {path}: {y.shape}")

    if not np.isfinite(y).all():
        raise ValueError(f"Non-finite audio values: {path}")

    return np.asarray(y, dtype=np.float32)



@torch.inference_mode()
def extract_voice_embeddings(y, processor, model, device):

    input_values = processor(
        y,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    ).input_values

    output = model(input_values.to(device)).last_hidden_state

    v1 = output.mean(dim=1).squeeze(0).float().cpu().numpy()[:512]

    if v1.shape != (512,) or not np.isfinite(v1).all():
        raise ValueError("Invalid V1 vector")

    return v1



def extract_mfccs(y, transform):

    waveform = torch.from_numpy(y).unsqueeze(0)

    mfcc = transform(waveform).mean(dim=2).squeeze(0)

    v3 = mfcc.cpu().numpy().astype(np.float32, copy=False)[:25]

    if v3.shape != (25,) or not np.isfinite(v3).all():
        raise ValueError("Invalid V3 vector")

    return v3



def extract_combined_features(y):

    y22 = librosa.resample(
        y,
        orig_sr=SAMPLE_RATE,
        target_sr=LIBROSA_DEFAULT_SR
    )

    sr = LIBROSA_DEFAULT_SR

    centroid = float(
        librosa.feature.spectral_centroid(y=y22, sr=sr).mean()
    )

    zcr = float(
        librosa.feature.zero_crossing_rate(y=y22).mean()
    )

    rms = float(
        librosa.feature.rms(y=y22).mean()
    )

    tempo, _ = librosa.beat.beat_track(y=y22, sr=sr)

    tempo_value = float(np.asarray(tempo).reshape(-1)[0])

    pitches, _magnitudes = librosa.piptrack(y=y22, sr=sr)

    positives = pitches[pitches > 0]

    pitch = float(positives.mean()) if positives.size else 0.0

    values = np.array(
        [centroid, zcr, rms, tempo_value, pitch],
        dtype=np.float32
    )

    v4 = np.tile(values, 5)[:25]

    if v4.shape != (25,) or not np.isfinite(v4).all():
        raise ValueError("Invalid V4 vector")

    return v4



def load_split(split):

    csv_path = CSV_FILES[split]

    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    required = {
        "Dialogue_ID",
        "Utterance_ID",
        "Emotion"
    }

    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns in {csv_path}")

    if len(df) != EXPECTED_ROWS[split]:
        raise ValueError(
            f"{split}: expected {EXPECTED_ROWS[split]} rows, "
            f"found {len(df)}"
        )

    df = df.copy()

    df["File Name"] = [
        f"dia{int(d)}_utt{int(u)}.mp4"
        for d, u in zip(df["Dialogue_ID"], df["Utterance_ID"])
    ]

    df["sample_id"] = [
        f"{split}:{name[:-4]}"
        for name in df["File Name"]
    ]

    df["Emotion"] = (
        df["Emotion"].astype(str).str.strip().str.lower()
    )

    if df["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate IDs in {split} CSV")

    return df




def parse_vector(value, expected_dim):

    try:
        vector = np.asarray(ast.literal_eval(str(value)), dtype=np.float32)

        if vector.shape != (expected_dim,):
            return None

        if not np.isfinite(vector).all():
            return None

        return vector

    except Exception:
        return None


def validate_existing_train(existing):

    required_columns = {
        "sample_id",
        "File Name",
        "Emotion",
        "V1",
        "V3",
        "V4"
    }

    missing = required_columns - set(existing.columns)

    if missing:
        raise ValueError(
            f"Existing audio Excel missing columns: {missing}"
        )

    if existing["sample_id"].duplicated().any():
        raise ValueError("Duplicate IDs in existing audio Excel")

    if not existing["sample_id"].astype(str).str.startswith("train:").all():
        raise ValueError(
            "Existing audio Excel contains non-train IDs. "
            "Stopping to protect your data."
        )

    valid_rows = []
    removed_ids = []

    for _, row in existing.iterrows():

        v1 = parse_vector(row["V1"], 512)
        v3 = parse_vector(row["V3"], 25)
        v4 = parse_vector(row["V4"], 25)

        if v1 is None or v3 is None or v4 is None:
            removed_ids.append(str(row["sample_id"]))
            continue

        valid_rows.append(row.to_dict())

    valid_train = pd.DataFrame(valid_rows)

    valid_train["Emotion"] = (
        valid_train["Emotion"].astype(str).str.strip().str.lower()
    )

    print("Existing train feature rows:", len(existing))
    print("Valid train audio rows:", len(valid_train))
    print("Invalid train rows:", len(removed_ids))

    return valid_train, removed_ids




def extract_one(row, split, processor, model, device, mfcc_transform):

    name = row["File Name"]

    y = load_waveform(name, split)

    v1 = extract_voice_embeddings(y, processor, model, device)

    v3 = extract_mfccs(y, mfcc_transform)

    v4 = extract_combined_features(y)

    return {
        "sample_id": row["sample_id"],
        "File Name": name,
        "Emotion": row["Emotion"],
        "V1": str(v1.tolist()),
        "V3": str(v3.tolist()),
        "V4": str(v4.tolist()),
    }



def extract_new_split(
    df,
    split,
    processor,
    model,
    device,
    mfcc_transform,
    removed_ids
):

    extracted_rows = []

    total = len(df)

    for index, (_, row) in enumerate(df.iterrows(), start=1):

        try:

            features = extract_one(
                row,
                split,
                processor,
                model,
                device,
                mfcc_transform
            )

            extracted_rows.append(features)

        except Exception as exc:

            sample_id = str(row["sample_id"])

            removed_ids.append(sample_id)

            print(
                f"[SKIPPED] {split} | {sample_id} | "
                f"{type(exc).__name__}: {exc}",
                flush=True
            )

        if index % 25 == 0 or index == total:

            print(
                f"{split}: processed {index}/{total} | "
                f"valid={len(extracted_rows)} | "
                f"removed={len(removed_ids)}",
                flush=True
            )

    return pd.DataFrame(extracted_rows)



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Test only two dev samples; save nothing"
    )

    args = parser.parse_args()

    

    if not OUT_EXCEL.is_file():

        raise FileNotFoundError(
            "Existing train audio Excel not found. "
            "This script will not regenerate train features:\n"
            f"{OUT_EXCEL}"
        )

    existing = pd.read_excel(OUT_EXCEL)

    if len(existing) != EXPECTED_ROWS["train"]:

        raise ValueError(
            f"Expected {EXPECTED_ROWS['train']} existing train rows, "
            f"found {len(existing)}. Stopping to protect your data."
        )


    if not BACKUP_EXCEL.exists():

        shutil.copy2(OUT_EXCEL, BACKUP_EXCEL)

        print("Training backup created:", BACKUP_EXCEL)

    else:

        print("Existing backup retained:", BACKUP_EXCEL)

    # --------------------------------------------------------
    # 3. VALIDATE EXISTING TRAIN FEATURES
    # --------------------------------------------------------

    train_features, removed_ids = validate_existing_train(existing)

    # --------------------------------------------------------
    # 4. LOAD DEV AND TEST CSV
    # --------------------------------------------------------

    dev_df = load_split("dev")
    test_df = load_split("test")

    print("Existing train rows:", len(existing))
    print("Dev rows to process:", len(dev_df))
    print("Test rows to process:", len(test_df))

    # --------------------------------------------------------
    # 5. DEVICE
    # --------------------------------------------------------

    if args.device == "auto":

        device_name = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    else:

        device_name = args.device

    if device_name == "cuda" and not torch.cuda.is_available():

        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(device_name)

    print("Device:", device, flush=True)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
            flush=True
        )

    # --------------------------------------------------------
    # 6. LOAD WAV2VEC2
    # --------------------------------------------------------

    print("Loading Wav2Vec2:", MODEL_ID, flush=True)

    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)

    model = Wav2Vec2Model.from_pretrained(MODEL_ID)

    model = model.to(device).eval()

    if model.config.hidden_size != 768:

        raise ValueError(
            f"Unexpected hidden size: {model.config.hidden_size}"
        )

    mfcc_transform = MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=25
    )

    # --------------------------------------------------------
    # 7. OPTIONAL TEST MODE
    # --------------------------------------------------------

    if args.test:

        extract_new_split(
            dev_df.iloc[:2],
            "dev",
            processor,
            model,
            device,
            mfcc_transform,
            []
        )

        print("TEST COMPLETE. No Excel files were changed.")

        return

    # --------------------------------------------------------
    # 8. EXTRACT DEV AND TEST ONLY
    # --------------------------------------------------------

    dev_features = extract_new_split(
        dev_df,
        "dev",
        processor,
        model,
        device,
        mfcc_transform,
        removed_ids
    )

    test_features = extract_new_split(
        test_df,
        "test",
        processor,
        model,
        device,
        mfcc_transform,
        removed_ids
    )

    # --------------------------------------------------------
    # 9. COMBINE TRAIN + DEV + TEST
    # --------------------------------------------------------

    combined = pd.concat(
        [
            train_features,
            dev_features,
            test_features
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # 10. VALIDATE SAMPLE IDS
    # --------------------------------------------------------

    if combined["sample_id"].duplicated().any():

        duplicates = combined.loc[
            combined["sample_id"].duplicated(),
            "sample_id"
        ].tolist()

        raise ValueError(
            f"Duplicate IDs found after merging: {duplicates[:10]}"
        )

    # --------------------------------------------------------
    # 11. SAVE REMOVAL LIST FOR ALL MODALITIES
    # --------------------------------------------------------

    removed_ids = sorted(set(removed_ids))

    pd.DataFrame({
        "sample_id": removed_ids
    }).to_csv(
        REMOVED_IDS_CSV,
        index=False
    )

    # --------------------------------------------------------
    # 12. SAVE COMBINED AUDIO EXCEL
    # --------------------------------------------------------

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    temp_excel = OUT_EXCEL.with_name(
        "audio_features_github_combined_temp.xlsx"
    )

    combined.to_excel(temp_excel, index=False)

    # Verify temporary file before replacing original.
    check = pd.read_excel(temp_excel)

    if check["sample_id"].duplicated().any():

        raise RuntimeError(
            "Temporary Excel has duplicate IDs. "
            "Original Excel has not been replaced."
        )

    if len(check) != len(combined):

        raise RuntimeError(
            "Temporary Excel row count mismatch. "
            "Original Excel has not been replaced."
        )

    temp_excel.replace(OUT_EXCEL)

    # --------------------------------------------------------
    # 13. SUMMARY
    # --------------------------------------------------------

    print("\n========================================")
    print("AUDIO MERGED FEATURE EXTRACTION COMPLETE")
    print("========================================")

    print("Original train rows:", EXPECTED_ROWS["train"])
    print("Valid train rows:", len(train_features))
    print("Dev feature rows:", len(dev_features))
    print("Test feature rows:", len(test_features))
    print("Final audio feature rows:", len(combined))
    print("Removed sample IDs:", len(removed_ids))

    print("\nUpdated audio Excel:")
    print(OUT_EXCEL)

    print("\nBackup of original training features:")
    print(BACKUP_EXCEL)

    print("\nRemoval list for text/video:")
    print(REMOVED_IDS_CSV)

    print("\nExisting neural checkpoints were not modified.")


if __name__ == "__main__":
    main()
