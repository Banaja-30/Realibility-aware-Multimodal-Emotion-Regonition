from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from torchaudio.transforms import MFCC
from transformers import Wav2Vec2Model, Wav2Vec2Processor

ROOT = Path(r"D:\MELD_Project")
TRAIN_CSV = Path(r"D:\MELD_Project\raw\MELD.Raw\dev_sent_emo.csv")
AUDIO_CACHE = Path(r"D:\MELD_Project\cached_audio\dev")
FEATURE_DIR = ROOT / "repo_style_final" / "features"
OUT_EXCEL = Path(r"D:\MELD_Project\repo_style_final\features\audio_features_github_dev.xlsx")
BATCH_CACHE = Path(r"D:\MELD_Project\repo_style_final\features\audio_github_batches_dev")
MODEL_ID = "facebook/wav2vec2-base-960h"
SAMPLE_RATE = 16000
LIBROSA_DEFAULT_SR = 22050
ROWS_PER_BATCH = 25
EXPECTED_ROWS = 1109
VERSION = "repo_audio_original_v1_npy16k"


def load_waveform(name: str) -> np.ndarray:
    path = AUDIO_CACHE / name.replace(".mp4", ".npy")
    if not path.is_file():
        raise FileNotFoundError(f"Missing cached audio file: {path}")
    y = np.load(path, allow_pickle=False)
    if y.ndim != 1 or y.size < 400 or not np.issubdtype(y.dtype, np.number):
        raise ValueError(f"Invalid cached waveform shape/type: {path}: {y.shape}")
    if not np.isfinite(y).all():
        raise ValueError(f"Non-finite values in audio: {path}")
    return np.asarray(y, dtype=np.float32)


@torch.inference_mode()
def extract_voice_embeddings(y, processor, model, device):
    
    input_values = processor(y, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values
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

    y22 = librosa.resample(y, orig_sr=SAMPLE_RATE, target_sr=LIBROSA_DEFAULT_SR)
    sr = LIBROSA_DEFAULT_SR
    centroid = float(librosa.feature.spectral_centroid(y=y22, sr=sr).mean())
    zcr = float(librosa.feature.zero_crossing_rate(y=y22).mean())
    rms = float(librosa.feature.rms(y=y22).mean())
    tempo, _ = librosa.beat.beat_track(y=y22, sr=sr)
    # In recent librosa versions tempo can be a 1-element ndarray.
    tempo_value = float(np.asarray(tempo).reshape(-1)[0])
    pitches, _magnitudes = librosa.piptrack(y=y22, sr=sr)
    positives = pitches[pitches > 0]
    
    pitch = float(positives.mean()) if positives.size else 0.0

    values = np.array([centroid, zcr, rms, tempo_value, pitch], dtype=np.float32)
    v4 = np.tile(values, 5)[:25]
    if v4.shape != (25,) or not np.isfinite(v4).all():
        raise ValueError("Invalid V4 vector")
    return v4


def extract_rows(rows, processor, model, device, mfcc_transform):
    a, b, c = [], [], []
    for row in rows.itertuples(index=False):
        name = f"dia{int(row.Dialogue_ID)}_utt{int(row.Utterance_ID)}.mp4"
        y = load_waveform(name)
        a.append(extract_voice_embeddings(y, processor, model, device))
        b.append(extract_mfccs(y, mfcc_transform))
        c.append(extract_combined_features(y))
    return np.stack(a), np.stack(b), np.stack(c)


def csv_and_filenames():
    if not TRAIN_CSV.is_file():
        raise FileNotFoundError(TRAIN_CSV)
    df = pd.read_csv(TRAIN_CSV)
    required = {"Dialogue_ID", "Utterance_ID", "Emotion"}
    if len(df) != EXPECTED_ROWS or not required.issubset(df.columns):
        raise ValueError("Expected the original 9,989-row MELD training CSV")
    names = [f"dia{int(d)}_utt{int(u)}.mp4" for d, u in zip(df.Dialogue_ID, df.Utterance_ID)]
    if len(set(names)) != EXPECTED_ROWS:
        raise ValueError("Duplicate MELD audio filenames")
    missing = [n for n in names if not (AUDIO_CACHE / n.replace(".mp4", ".npy")).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} audio cache files; first: {missing[0]}")
    return df, names


def save_batch(path, start, stop, names, v1, v3, v4):
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, start=np.array(start), stop=np.array(stop),
                            filenames=np.asarray(names, dtype=str),
                            V1=v1, V3=v3, V4=v4)
    tmp.replace(path)


def check_batch(path, start, stop, names):
    with np.load(path, allow_pickle=False) as data:
        if (int(data["start"]) != start or int(data["stop"]) != stop
                or not np.array_equal(data["filenames"], np.asarray(names))):
            raise ValueError(f"Batch row identities differ: {path}")
        for key, dim in (("V1", 512), ("V3", 25), ("V4", 25)):
            array = data[key]
            if array.shape != (stop - start, dim) or not np.isfinite(array).all():
                raise ValueError(f"Bad batch {key}: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="Run on two samples; save nothing")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="Compute device: auto (default), cpu, or cuda")
    parser.add_argument("--batch-rows", type=int, default=ROWS_PER_BATCH,
                        help="Saved rows per restartable batch; default 25")
    args = parser.parse_args()
    if args.batch_rows < 1:
        parser.error("--batch-rows must be positive")

    df, names = csv_and_filenames()
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    print("Device:", device, flush=True)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Loading GitHub Wav2Vec2:", MODEL_ID, flush=True)
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2Model.from_pretrained(MODEL_ID).to(device).eval()
    if model.config.hidden_size != 768:
        raise ValueError(f"Unexpected Wav2Vec2 hidden size: {model.config.hidden_size}")
    mfcc_transform = MFCC(sample_rate=SAMPLE_RATE, n_mfcc=25)

    if args.test:
        v1, v3, v4 = extract_rows(df.iloc[:2], processor, model, device, mfcc_transform)
        print(f"TEST SUCCESS | V1={v1.shape} | V3={v3.shape} | V4={v4.shape}", flush=True)
        print("No batch or final Excel files were written.", flush=True)
        return

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_CACHE.mkdir(parents=True, exist_ok=True)
    manifest_path = BATCH_CACHE / "metadata.json"
    metadata = {"source_csv": str(TRAIN_CSV), "source_rows": EXPECTED_ROWS,
                "model_id": MODEL_ID, "sample_rate": SAMPLE_RATE,
                "version": VERSION, "batch_rows": args.batch_rows}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old != metadata:
            raise RuntimeError("Existing GitHub audio batch metadata differs; do not mix batches.")
    else:
        manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    batches = []
    for start in range(0, len(df), args.batch_rows):
        stop = min(start + args.batch_rows, len(df))
        path = BATCH_CACHE / f"batch_{start:05d}_{stop:05d}.npz"
        these_names = names[start:stop]
        if path.is_file():
            check_batch(path, start, stop, these_names)
            print(f"Reusing GitHub audio batch: {stop}/{len(df)}", flush=True)
        else:
            v1, v3, v4 = extract_rows(df.iloc[start:stop], processor, model, device, mfcc_transform)
            save_batch(path, start, stop, these_names, v1, v3, v4)
            print(f"Processed and saved: {stop}/{len(df)}", flush=True)
        batches.append((path, start, stop, these_names))

    print("Combining GitHub audio features into a separate Excel...", flush=True)
    output1, output3, output4 = [], [], []
    for path, start, stop, these_names in batches:
        check_batch(path, start, stop, these_names)
        with np.load(path, allow_pickle=False) as data:
            output1.extend(data["V1"].tolist())
            output3.extend(data["V3"].tolist())
            output4.extend(data["V4"].tolist())
    output = pd.DataFrame({
        "sample_id": ["dev:" + n[:-4] for n in names],
        "File Name": names,
        "Emotion": df["Emotion"].astype(str).str.strip().str.lower(),
        "V1": [str(vec) for vec in output1],
        "V3": [str(vec) for vec in output3],
        "V4": [str(vec) for vec in output4],
    })
    if len(output) != EXPECTED_ROWS or output["sample_id"].duplicated().any():
        raise RuntimeError("Invalid final row count or duplicate IDs")
    temp = OUT_EXCEL.with_name("audio_features_github.partial.xlsx")
    output.to_excel(temp, index=False)
    temp.replace(OUT_EXCEL)
    print(f"AUDIO GITHUB COMPLETE | rows={len(output)} | V1=512 | V3=25 | V4=25", flush=True)
    print(f"Saved new file: {OUT_EXCEL}", flush=True)
    print("Existing audio features and neural checkpoints were NOT modified.", flush=True)


if __name__ == "__main__":
    main()
