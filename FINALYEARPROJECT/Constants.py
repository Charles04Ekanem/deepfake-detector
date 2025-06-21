import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FOLDER = os.path.join(BASE_DIR, "dataset")
ORIGINAL_VIDEOS_FOLDER = os.path.join(DATA_FOLDER, "original_sequences", "youtube", "raw", "videos")
FAKE_VIDEOS_FOLDER = os.path.join(DATA_FOLDER, "manipulated_sequences", "DeepFakeDetection", "raw", "videos")
FACES_FOLDER = os.path.join(DATA_FOLDER, "faces")
FACES_REAL = os.path.join(FACES_FOLDER, "real")
FACES_FAKE = os.path.join(FACES_FOLDER, "fake")
FACES_CSV = os.path.join(FACES_FOLDER, "balanced_faces_temp.csv")

print("Original videos:", ORIGINAL_VIDEOS_FOLDER)
print("Fake videos:", FAKE_VIDEOS_FOLDER)
print("Faces will be saved in:", FACES_REAL, "and", FACES_FAKE)
print("Metadata CSV will be:", FACES_CSV)

