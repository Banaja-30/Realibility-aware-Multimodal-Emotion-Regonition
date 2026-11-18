
from pathlib import Path
import shutil
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

# ============================================================
# PATHS AND SETTINGS
# ============================================================

PROJECT = Path(r"D:\MELD_Project")

RAW_DIR = PROJECT / "raw" / "MELD.Raw"
FEATURE_DIR = PROJECT / "repo_style_final" / "features"

TRAIN_CSV = RAW_DIR / "train_sent_emo.csv"
DEV_CSV = RAW_DIR / "dev_sent_emo.csv"
TEST_CSV = RAW_DIR / "test_sent_emo.csv"

OUTPUT_EXCEL = FEATURE_DIR / "text_features.xlsx"
BACKUP_EXCEL = FEATURE_DIR / "text_features_train_backup.xlsx"
TEMP_EXCEL = FEATURE_DIR / "text_features_combined_temp.xlsx"

MODEL_NAME = "FacebookAI/roberta-base"
BATCH_SIZE = 16
MAX_LENGTH = 128

EXPECTED_TRAIN = 9989
EXPECTED_DEV = 1109
EXPECTED_TEST = 2610
EXPECTED_TOTAL = 13708

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# ============================================================
# 1. CHECK EXISTING TRAINING FEATURES
#    DO NOT RE-EXTRACT TRAINING DATA
# ============================================================

if not OUTPUT_EXCEL.exists():
    raise FileNotFoundError(
        f"Existing training feature file not found:\n{OUTPUT_EXCEL}"
    )

existing = pd.read_excel(OUTPUT_EXCEL)

required_columns = {
    "sample_id",
    "File Name",
    "V2",
    "Emotion"
}

missing_columns = required_columns - set(existing.columns)

if missing_columns:
    raise ValueError(
        f"Existing Excel is missing columns: {missing_columns}"
    )

if len(existing) != EXPECTED_TRAIN:
    raise ValueError(
        f"Expected {EXPECTED_TRAIN} existing training rows, "
        f"but found {len(existing)}. Stopping to protect your data."
    )

if existing["sample_id"].duplicated().any():
    raise ValueError("Duplicate sample_id found in existing training Excel.")

if not existing["sample_id"].astype(str).str.startswith("train:").all():
    raise ValueError(
        "Existing Excel contains IDs that are not all train-prefixed. "
        "Stopping to avoid modifying the wrong data."
    )

# Normalize existing training labels without re-extracting features.
existing["Emotion"] = (
    existing["Emotion"].astype(str).str.strip().str.lower()
)

print(f"Existing training features preserved: {len(existing)}")

# ============================================================
# 2. BACKUP THE EXISTING TRAINING EXCEL
# ============================================================

if not BACKUP_EXCEL.exists():
    shutil.copy2(OUTPUT_EXCEL, BACKUP_EXCEL)
    print("Training backup created:", BACKUP_EXCEL)
else:
    print("Existing backup retained:", BACKUP_EXCEL)

# ============================================================
# 3. LOAD ONLY DEV AND TEST CSV FILES
# ============================================================

def load_split(csv_path, split_name, expected_rows):
    df = pd.read_csv(csv_path)

    required = {
        "Dialogue_ID",
        "Utterance_ID",
        "Utterance",
        "Emotion"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{csv_path.name} is missing columns: {missing}"
        )

    if len(df) != expected_rows:
        raise ValueError(
            f"{split_name}: expected {expected_rows} rows, "
            f"found {len(df)}"
        )

    df = df.copy()

    df["File Name"] = [
        f"dia{d}_utt{u}.mp4"
        for d, u in zip(df["Dialogue_ID"], df["Utterance_ID"])
    ]

    df["sample_id"] = [
        f"{split_name}:{name[:-4]}"
        for name in df["File Name"]
    ]

    df["Emotion"] = (
        df["Emotion"].astype(str).str.strip().str.lower()
    )

    return df


dev_df = load_split(DEV_CSV, "dev", EXPECTED_DEV)
test_df = load_split(TEST_CSV, "test", EXPECTED_TEST)

new_df = pd.concat(
    [dev_df, test_df],
    ignore_index=True
)

if len(new_df) != EXPECTED_DEV + EXPECTED_TEST:
    raise ValueError("Unexpected number of dev/test samples.")

# Ensure the new IDs do not overlap with training IDs.
existing_ids = set(existing["sample_id"].astype(str))
new_ids = set(new_df["sample_id"].astype(str))

overlap = existing_ids.intersection(new_ids)

if overlap:
    raise ValueError(
        f"Dev/test IDs overlap with training IDs. Examples: "
        f"{list(overlap)[:5]}"
    )

if new_df["sample_id"].duplicated().any():
    raise ValueError("Duplicate sample IDs found among dev/test rows.")

print("Dev rows:", len(dev_df))
print("Test rows:", len(test_df))
print("New rows to extract:", len(new_df))

# ============================================================
# 4. LOAD ROBERTA
#    ONLY DEV/TEST TEXTS WILL BE PROCESSED
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModel.from_pretrained(MODEL_NAME)
model = model.to(device)
model.eval()

texts = new_df["Utterance"].fillna("").astype(str).tolist()

all_vectors = []

# ============================================================
# 5. EXTRACT DEV/TEST FEATURES ONLY
# ============================================================

with torch.no_grad():

    for start in range(0, len(texts), BATCH_SIZE):

        batch_texts = texts[start:start + BATCH_SIZE]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        outputs = model(**inputs)

        hidden = outputs.last_hidden_state

        mask = inputs["attention_mask"].unsqueeze(-1)

        vectors = (
            (hidden * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1)
        )

        vectors = vectors.cpu().numpy()

        all_vectors.extend(vectors)

        print(
            f"Processed {min(start + BATCH_SIZE, len(texts))}"
            f"/{len(texts)} dev/test samples"
        )

# ============================================================
# 6. CREATE DEV/TEST FEATURE ROWS
# ============================================================

new_features = pd.DataFrame({
    "sample_id": new_df["sample_id"].tolist(),
    "File Name": new_df["File Name"].tolist(),
    "V2": [str(vector.tolist()) for vector in all_vectors],
    "Emotion": new_df["Emotion"].tolist()
})

if len(new_features) != EXPECTED_DEV + EXPECTED_TEST:
    raise ValueError("Dev/test feature extraction row count mismatch.")

# ============================================================
# 7. APPEND NEW FEATURES TO EXISTING TRAIN FEATURES
# ============================================================

combined = pd.concat(
    [existing, new_features],
    ignore_index=True
)

if len(combined) != EXPECTED_TOTAL:
    raise ValueError(
        f"Expected {EXPECTED_TOTAL} combined rows, "
        f"but got {len(combined)}"
    )

if combined["sample_id"].duplicated().any():
    raise ValueError("Duplicate IDs found after merging.")

# Ensure all expected split prefixes exist.
for split in ["train", "dev", "test"]:
    count = combined["sample_id"].astype(str).str.startswith(
        f"{split}:"
    ).sum()

    print(f"{split} rows:", count)

# ============================================================
# 8. SAVE COMBINED FILE SAFELY
# ============================================================

combined.to_excel(TEMP_EXCEL, index=False)

# Read back the temporary file before replacing the original.
check = pd.read_excel(TEMP_EXCEL)

if len(check) != EXPECTED_TOTAL:
    raise ValueError(
        "Temporary Excel validation failed. "
        "Original Excel has not been replaced."
    )

if check["sample_id"].duplicated().any():
    raise ValueError(
        "Temporary Excel contains duplicate IDs. "
        "Original Excel has not been replaced."
    )

# Replace the original only after validation.
TEMP_EXCEL.replace(OUTPUT_EXCEL)

print("\nSUCCESS")
print("Original train features preserved:", EXPECTED_TRAIN)
print("New dev/test features appended:", EXPECTED_DEV + EXPECTED_TEST)
print("Final combined rows:", len(combined))
print("Updated Excel:", OUTPUT_EXCEL)
print("Training backup:", BACKUP_EXCEL)
