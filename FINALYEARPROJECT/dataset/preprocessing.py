import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

# === PATHS ===
RAW_CSV = r"C:\Users\CHARLES EKANEM\Documents\FINALYEARPROJECT\dataset\faces\balanced_faces_temp.csv"
RAW_FACE_DIR = r"C:\Users\CHARLES EKANEM\Documents\FINALYEARPROJECT\dataset\faces"
OUTPUT_DIR = r"C:\Users\CHARLES EKANEM\Documents\FINALYEARPROJECT\dataset\processed_faces"
CLEAN_CSV = os.path.join(OUTPUT_DIR, "processed_faces.csv")

# === CREATE OUTPUT FOLDERS ===
real_out = os.path.join(OUTPUT_DIR, "real")
fake_out = os.path.join(OUTPUT_DIR, "fake")
os.makedirs(real_out, exist_ok=True)
os.makedirs(fake_out, exist_ok=True)

# === IMAGE SETTINGS ===
TARGET_SIZE = (256, 256)

# === LOAD CSV ===
df = pd.read_csv(RAW_CSV)
cleaned_data = []

print("[INFO] Starting preprocessing...")
for i, row in tqdm(df.iterrows(), total=len(df)):
    filename = row["name"]
    label = int(row["label"])

    source_dir = "real" if label == 1 else "fake"
    input_path = os.path.join(RAW_FACE_DIR, source_dir, filename)
    output_path = os.path.join(real_out if label == 1 else fake_out, filename)

    try:
        with Image.open(input_path) as img:
            img = img.convert("RGB")
            img = img.resize(TARGET_SIZE)
            img.save(output_path)
            cleaned_data.append({"name": filename, "label": label})
    except Exception as e:
        print(f"[SKIP] Failed to process {input_path}: {e}")

# === SAVE CLEAN CSV ===
cleaned_df = pd.DataFrame(cleaned_data)
cleaned_df.to_csv(CLEAN_CSV, index=False)
print(f"[INFO] Saved cleaned CSV: {CLEAN_CSV}")
print(f"[INFO] Real: {len(cleaned_df[cleaned_df['label'] == 1])}, Fake: {len(cleaned_df[cleaned_df['label'] == 0])}")
