import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from shutil import move

# === CONFIGURATION ===
REAL_DIR = r"C:\Users\CHARLES EKANEM\Documents\FINALYEARPROJECT\dataset\faces\real"
FAKE_DIR = r"C:\Users\CHARLES EKANEM\Documents\FINALYEARPROJECT\dataset\faces\fake"
CSV_PATH = r"C:\Users\CHARLES EKANEM\Documents\FINALYEARPROJECT\dataset\faces\balanced_faces_temp.csv"
BAR_PATH = r"C:\Users\CHARLES EKANEM\Documents\FINALYEARPROJECT\dataset\faces"
PIE_PATH = r"C:\Users\CHARLES EKANEM\Documents\FINALYEARPROJECT\dataset\faces"

# === VALID EXTENSIONS ===
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

# === HELPER ===
def get_image_list(folder):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(VALID_EXTENSIONS)])

# === COLLECT IMAGES ===
real_images = get_image_list(REAL_DIR)
fake_images = get_image_list(FAKE_DIR)
num_real = len(real_images)
num_fake = len(fake_images)

print(f"[INFO] Total Real Faces: {num_real}")
print(f"[INFO] Total Fake Faces: {num_fake}")

# === BALANCE ===
if num_real > num_fake:
    delete_count = num_real - num_fake
    print(f"[INFO] Deleting {delete_count} excess real images...")
    random.shuffle(real_images)
    for img_name in real_images[:delete_count]:
        path = os.path.join(REAL_DIR, img_name)
        try:
            os.remove(path)
        except Exception as e:
            print(f"[WARN] Couldn't delete {path}: {e}")

# === RECOUNT ===
real_images = get_image_list(REAL_DIR)
fake_images = get_image_list(FAKE_DIR)

# === SAVE CSV ===
print("[INFO] Rebuilding faces.csv ...")
data = []
for img in fake_images:
    data.append({"name": img, "label": 0})  # 0 = fake
for img in real_images:
    data.append({"name": img, "label": 1})  # 1 = real
df = pd.DataFrame(data)
df.to_csv(CSV_PATH, index=False)
print(f"[INFO] Saved updated CSV: {CSV_PATH}")

# === PLOTTING ===
def plot_distribution(real_count, fake_count):
    labels = ["Fake", "Real"]
    counts = [fake_count, real_count]
    colors = ["red", "green"]

    # Bar chart
    plt.figure(figsize=(6, 4))
    plt.bar(labels, counts, color=colors)
    plt.title("Balanced Face Dataset Distribution (Bar Chart)")
    plt.ylabel("Image Count")
    plt.savefig(BAR_PATH)
    print(f"[INFO] Saved bar chart: {BAR_PATH}")
    plt.close()

    # Pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    plt.title("Balanced Face Dataset Distribution (Pie Chart)")
    plt.axis("equal")
    plt.savefig(PIE_PATH)
    print(f"[INFO] Saved pie chart: {PIE_PATH}")
    plt.close()

# === FINAL STATS ===
print(f"[FINAL] Real: {len(real_images)}, Fake: {len(fake_images)}")
plot_distribution(len(real_images), len(fake_images))
print("[INFO] Distribution plots generated successfully.")