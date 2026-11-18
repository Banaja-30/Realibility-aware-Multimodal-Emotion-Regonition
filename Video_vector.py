

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import models, transforms

from Audio_vector_github import ROOT

RAW_DIR = ROOT / "raw" / "MELD.Raw"

CSV_FILES = {
    "train": RAW_DIR / "train_sent_emo.csv",
    "dev": RAW_DIR / "dev_sent_emo.csv",
    "test": RAW_DIR / "test_sent_emo.csv",
}

VIDEO_DIRS = {
    "train": RAW_DIR / "train_splits",
    "dev": RAW_DIR / "dev_splits_complete",
    "test": RAW_DIR / "output_repeated_splits_test",
}

OUT_DIR = ROOT / "repo_style_final" / "features"
OUT_EXCEL = OUT_DIR / "video_features.xlsx"

BACKUP_EXCEL = OUT_DIR / "video_features_train_backup.xlsx"
REMOVED_IDS_CSV = OUT_DIR / "video_missing_sample_ids.csv"

BATCH_DIR = OUT_DIR / "video_batches_github_uniform_merged"

EXPECTED_ROWS = {
    "train": 9989,
    "dev": 1109,
    "test": 2610,
}
# New directory: do NOT mix with batches from the previous entropy-frame script.
BATCH_DIR = OUT_DIR / "video_batches_github_uniform"
NUM_FRAMES = 10
BATCH_ROWS = 50
SEED = 42

SETTINGS = {
    "version": "github_original_uniform_10_resnet50_lstm_fc_v1",
    "sample_rate": "video_frames",
    "frame_selection": "linspace_seek_10",
    "preprocess": "BGR_ToPIL_Resize256_CenterCrop224_ImageNetNormalize",
    "resnet50_weights": "torchvision_DEFAULT",
    "lstm_input_size": 2048,
    "lstm_hidden_size": 512,
    "lstm_num_layers": 1,
    "linear_output": 512,
    "seed": SEED,
}
SIGNATURE = hashlib.sha256(json.dumps(SETTINGS, sort_keys=True).encode()).hexdigest()

PREPROCESS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class LSTMModel(nn.Module):
    """Exactly the LSTM + fully connected shape of the supplied GitHub code."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2048, hidden_size=512,
                            num_layers=1, batch_first=True)
        self.fc = nn.Linear(512, 512)

    def forward(self, frame_vectors):
        # PyTorch defaults to zero h0/c0 when no initial state is supplied.
        output, _ = self.lstm(frame_vectors)
        return self.fc(output[:, -1, :])


def extract_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = []
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            # Original GitHub technique: uniformly spaced frame indices.
            indices = np.linspace(0, frame_count - 1, NUM_FRAMES, dtype=int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if ok and frame is not None:
                    frames.append(frame)
        else:
            # Fallback only when the video container cannot report its length.
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            if frames:
                indices = np.linspace(0, len(frames) - 1,
                                      NUM_FRAMES, dtype=int)
                frames = [frames[i] for i in indices]
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"No decodable frames: {video_path}")
    # Some MP4s have seek failures. Keep the expected 10-frame tensor shape.
    while len(frames) < NUM_FRAMES:
        frames.append(frames[-1])
    return frames[:NUM_FRAMES]


def build_models(device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    print("Loading pretrained ResNet50...", flush=True)
    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    backbone = nn.Sequential(*list(backbone.children())[:-1])
    backbone.to(device).eval()
    lstm_model = LSTMModel().to(device).eval()
    for model in (backbone, lstm_model):
        for param in model.parameters():
            param.requires_grad_(False)
    return backbone, lstm_model


def process_video(path, backbone, lstm_model, device):
    frames = extract_frames(path)
    # Intentionally pass OpenCV BGR frames directly, matching supplied GitHub code.
    batch = torch.stack([PREPROCESS(frame) for frame in frames], dim=0).to(device)
    with torch.inference_mode():
        frame_features = backbone(batch).flatten(1)  # [10, 2048]
        features = lstm_model(frame_features.unsqueeze(0))  # [1, 512]
        result = features[0].float().cpu().numpy()
    if result.shape != (512,) or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid 512-D feature: {path}")
    return result


def load_meld(split):

    csv_path = CSV_FILES[split]

    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    cols = {"Dialogue_ID", "Utterance_ID", "Emotion"}

    if len(df) != EXPECTED_ROWS[split] or not cols.issubset(df.columns):
        raise ValueError(
            f"Unexpected row count or columns in {split} CSV"
        )

    names = [
        f"dia{int(d)}_utt{int(u)}.mp4"
        for d, u in zip(df.Dialogue_ID, df.Utterance_ID)
    ]

    df = df.copy()

    df["File Name"] = names

    df["sample_id"] = [
        f"{split}:{n[:-4]}" for n in names
    ]

    df["Emotion"] = (
        df["Emotion"].astype(str).str.strip().str.lower()
    )

    if df["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample IDs in {split}")

    return df, names

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test",
        action="store_true",
        help="Check first two dev clips only"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Preserve existing train feature Excel
    # --------------------------------------------------------

    if not OUT_EXCEL.is_file():
        raise FileNotFoundError(
            f"Existing training video features not found: {OUT_EXCEL}"
        )

    existing = pd.read_excel(OUT_EXCEL)

    required_columns = {
        "sample_id", "File Name", "A2", "Emotion"
    }

    if not required_columns.issubset(existing.columns):
        raise ValueError("Existing video Excel has missing columns")

    if len(existing) != EXPECTED_ROWS["train"]:
        raise ValueError(
            f"Expected {EXPECTED_ROWS['train']} training rows, "
            f"found {len(existing)}. Stopping to protect train data."
        )

    if existing["sample_id"].duplicated().any():
        raise ValueError("Duplicate IDs in existing training Excel")

    if not existing["sample_id"].astype(str).str.startswith("train:").all():
        raise ValueError("Existing Excel does not contain only train IDs")

    existing["Emotion"] = (
        existing["Emotion"].astype(str).str.strip().str.lower()
    )

    # --------------------------------------------------------
    # Backup original train Excel
    # --------------------------------------------------------

    if not BACKUP_EXCEL.exists():
        import shutil
        shutil.copy2(OUT_EXCEL, BACKUP_EXCEL)

    print("Existing train features preserved:", len(existing))

    # --------------------------------------------------------
    # Load dev/test CSVs only
    # --------------------------------------------------------

    dev_df, dev_names = load_meld("dev")
    test_df, test_names = load_meld("test")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device, flush=True)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    backbone, lstm_model = build_models(device)

    if args.test:

        vectors = [
            process_video(
                VIDEO_DIRS["dev"] / name,
                backbone,
                lstm_model,
                device
            )
            for name in dev_names[:2]
        ]

        print(
            "TEST SUCCESS | samples:", len(vectors),
            "| A2:", np.stack(vectors).shape
        )

        print("No Excel files were changed.")
        return

    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Process dev/test only
    # --------------------------------------------------------

    new_rows = []
    removed_ids = []

    for split, df, names in [
        ("dev", dev_df, dev_names),
        ("test", test_df, test_names),
    ]:

        video_dir = VIDEO_DIRS[split]

        print(f"\nProcessing {split}: {len(df)} rows")

        for i, (_, row) in enumerate(df.iterrows()):

            name = names[i]
            sample_id = row["sample_id"]
            video_path = video_dir / name

            try:

                if not video_path.is_file():
                    raise FileNotFoundError(video_path)

                vector = process_video(
                    video_path,
                    backbone,
                    lstm_model,
                    device
                )

                new_rows.append({
                    "sample_id": sample_id,
                    "File Name": name,
                    "A2": str(vector.tolist()),
                    "Emotion": row["Emotion"],
                })

            except Exception as exc:

                removed_ids.append(sample_id)

                print(
                    f"[SKIPPED] {sample_id} | {exc}",
                    flush=True
                )

            if (i + 1) % 25 == 0 or (i + 1) == len(df):

                print(
                    f"{split}: {i + 1}/{len(df)} | "
                    f"valid={len(new_rows)} | "
                    f"removed={len(removed_ids)}",
                    flush=True
                )

    # --------------------------------------------------------
    # Append valid dev/test features to existing train features
    # --------------------------------------------------------

    new_features = pd.DataFrame(
        new_rows,
        columns=["sample_id", "File Name", "A2", "Emotion"]
    )

    combined = pd.concat(
        [existing, new_features],
        ignore_index=True
    )

    if combined["sample_id"].duplicated().any():
        raise RuntimeError("Duplicate IDs after merging video features")

    # --------------------------------------------------------
    # Save removal IDs for consistent text/audio cleanup
    # --------------------------------------------------------

    pd.DataFrame({
        "sample_id": sorted(set(removed_ids))
    }).to_csv(REMOVED_IDS_CSV, index=False)

    # --------------------------------------------------------
    # Save combined Excel safely
    # --------------------------------------------------------

    partial = OUT_EXCEL.with_name(
        "video_features_combined_temp.xlsx"
    )

    combined.to_excel(partial, index=False)

    check = pd.read_excel(partial)

    if len(check) != len(combined):
        raise RuntimeError("Temporary Excel row count mismatch")

    if check["sample_id"].duplicated().any():
        raise RuntimeError("Duplicate IDs in temporary Excel")

    partial.replace(OUT_EXCEL)

    print("\nVIDEO FEATURE MERGING COMPLETE")
    print("Original train rows:", EXPECTED_ROWS["train"])
    print("Valid dev/test rows:", len(new_features))
    print("Removed samples:", len(removed_ids))
    print("Final video feature rows:", len(combined))
    print("Saved:", OUT_EXCEL)
    print("Backup:", BACKUP_EXCEL)
    print("Removal IDs:", REMOVED_IDS_CSV)
if __name__ == "__main__":
    main()
