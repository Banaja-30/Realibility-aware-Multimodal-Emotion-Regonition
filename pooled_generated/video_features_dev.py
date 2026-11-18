

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

ROOT = Path(r"D:\MELD_Project")
CSV = Path(r"D:\MELD_Project\raw\MELD.Raw\dev_sent_emo.csv")
VIDEO_DIR = Path(r"D:\MELD_Project\raw\MELD.Raw\dev_splits_complete")
OUT_DIR = ROOT / "repo_style_final" / "features"
OUT_EXCEL = Path(r"D:\MELD_Project\repo_style_final\features\video_features_dev.xlsx")

# New directory: do NOT mix with batches from the previous entropy-frame script.
BATCH_DIR = Path(r"D:\MELD_Project\repo_style_final\features\video_batches_github_uniform_dev")
NUM_FRAMES = 10
BATCH_ROWS = 50
SEED = 42

SETTINGS = {
    "version": "github_original_uniform_10_resnet50_lstm_fc_v2_known_missing_dev",
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


KNOWN_MISSING_VIDEOS = {'dia110_utt7.mp4'}

def process_video_or_missing(path, backbone, lstm_model, device):
    if path.is_file():
        return process_video(path, backbone, lstm_model, device)
    if path.name not in KNOWN_MISSING_VIDEOS:
        raise FileNotFoundError(f"Unexpected missing video: {path}")
    print(f"MELD known missing video: {path.name}; using documented zero vector", flush=True)
    return np.zeros(512, dtype=np.float32)


def load_meld():
    if not CSV.is_file():
        raise FileNotFoundError(f"Missing CSV: {CSV}")
    df = pd.read_csv(CSV)
    cols = {"Dialogue_ID", "Utterance_ID", "Emotion"}
    if len(df) != 1109 or not cols.issubset(df.columns):
        raise ValueError("Expected the original MELD train CSV with 1109 rows")
    names = [f"dia{d}_utt{u}.mp4" for d, u in zip(df.Dialogue_ID, df.Utterance_ID)]
    if len(set(names)) != 1109:
        raise ValueError("Duplicate filename in MELD training CSV")
    missing = [n for n in names if not (VIDEO_DIR / n).is_file()]
    unexpected = [n for n in missing if n not in KNOWN_MISSING_VIDEOS]
    if unexpected:
        raise FileNotFoundError(f"Unexpected missing videos: {unexpected[:3]}")
    return df, names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Check first two clips only")
    args = parser.parse_args()

    df, names = load_meld()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Original MELD training MP4 files:", len(names), flush=True)
    backbone, lstm_model = build_models(device)

    if args.test:
        vectors = [process_video_or_missing(VIDEO_DIR / n, backbone, lstm_model, device)
                   for n in names[:2]]
        print("TEST SUCCESS | samples:", len(vectors), "| A2:",
              np.stack(vectors).shape, flush=True)
        print("No Excel or batch files were written in test mode.", flush=True)
        return

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for start in range(0, len(df), BATCH_ROWS):
        stop = min(start + BATCH_ROWS, len(df))
        path = BATCH_DIR / f"batch_{start:05d}_{stop:05d}.npz"
        ids = np.array(["dev:" + n[:-4] for n in names[start:stop]])
        if path.is_file():
            with np.load(path, allow_pickle=False) as data:
                if (data["A2"].shape != (stop - start, 512)
                        or not np.array_equal(data["sample_id"], ids)
                        or str(data["signature"].item()) != SIGNATURE
                        or not np.isfinite(data["A2"]).all()):
                    raise RuntimeError(f"Saved batch incompatible: {path}")
            print(f"Reusing saved video: {stop}/{len(df)}", flush=True)
        else:
            vectors = [process_video_or_missing(VIDEO_DIR / n, backbone, lstm_model, device)
                       for n in names[start:stop]]
            arr = np.stack(vectors).astype(np.float32)
            tmp = path.with_suffix(".tmp")
            with tmp.open("wb") as out:
                np.savez_compressed(out, sample_id=ids, A2=arr,
                                    signature=np.array(SIGNATURE))
            tmp.replace(path)
            print(f"Processed and saved: {stop}/{len(df)}", flush=True)
        paths.append(path)

    print("Combining completed batches into Excel...", flush=True)
    all_vectors = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            all_vectors.extend(str(vec.tolist()) for vec in data["A2"])
    if len(all_vectors) != len(df):
        raise RuntimeError("Feature count mismatch")
    result = pd.DataFrame({
        "sample_id": ["dev:" + n[:-4] for n in names],
        "File Name": names,
        "A2": all_vectors,
        "Emotion": df["Emotion"].astype(str),
    })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    partial = OUT_EXCEL.with_name("video_features.partial.xlsx")
    result.to_excel(partial, index=False)
    partial.replace(OUT_EXCEL)
    print("Saved:", OUT_EXCEL, flush=True)
    print("Rows:", len(result), "| A2 dimension: 512", flush=True)
    print("Video feature extraction complete.", flush=True)


if __name__ == "__main__":
    main()
