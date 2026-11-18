from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

# Paths and settings
PROJECT = Path(r"D:\MELD_Project")
INPUT_CSV = Path(r"D:\MELD_Project\raw\MELD.Raw\dev_sent_emo.csv")
OUTPUT_EXCEL = Path(r"D:\MELD_Project\repo_style_final\features\text_features_dev.xlsx")
MODEL_NAME = "FacebookAI/roberta-base"
BATCH_SIZE = 16
MAX_LENGTH = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# Load the original MELD training CSV
if not INPUT_CSV.is_file():
    raise FileNotFoundError(f"Training CSV not found: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)
required = {"Dialogue_ID", "Utterance_ID", "Utterance", "Emotion"}
if not required.issubset(df.columns):
    raise ValueError(f"Missing CSV columns: {sorted(required - set(df.columns))}")
if len(df) != 1109:
    raise ValueError(f"Expected 1109 MELD training rows, got {len(df)}")

names = [f"dia{d}_utt{u}.mp4" for d, u in zip(df["Dialogue_ID"], df["Utterance_ID"])]
ids = ["dev:" + name[:-4] for name in names]
if len(ids) != len(set(ids)):
    raise ValueError("Duplicate sample IDs in the training CSV")

# Frozen pretrained RoBERTa: one 768-D feature per utterance
print("Loading RoBERTa...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
texts = df["Utterance"].fillna("").astype(str).tolist()
all_vectors = []

print(f"Extracting text vectors for {len(texts)} utterances...")
with torch.inference_mode():
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=MAX_LENGTH, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        hidden = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        vectors = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        all_vectors.extend(vectors.float().cpu().tolist())
        if (start // BATCH_SIZE + 1) % 50 == 0:
            print(f"Processed {min(start + BATCH_SIZE, len(texts))}/{len(texts)}")

if len(all_vectors) != 1109 or any(len(v) != 768 for v in all_vectors):
    raise RuntimeError("RoBERTa feature count/dimension mismatch")

# Store list-strings in Excel, matching the GitHub pipeline's input format
OUTPUT_EXCEL.parent.mkdir(parents=True, exist_ok=True)
output = pd.DataFrame({
    "sample_id": ids,
    "File Name": names,
    "V2": [str(vector) for vector in all_vectors],
    "Emotion": df["Emotion"].astype(str),
})
output.to_excel(OUTPUT_EXCEL, index=False)
print("Saved:", OUTPUT_EXCEL)
print("Rows:", len(output), "| Text dimension: 768")
print("Text feature extraction complete.")
