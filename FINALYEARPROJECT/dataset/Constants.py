import os
DATA_FOLDER = r"C:\Users\CHARLES EKANEM\Documents\FINALYEARPROJECT\dataset"
ORIGINAL_VIDEOS_FOLDER = r"C:\Users\CHARLES EKANEM\Documents\FINALYEARPROJECT\dataset\Real_videos"
FAKE_VIDEOS_FOLDER = r"C:\Users\CHARLES EKANEM\Documents\FINALYEARPROJECT\dataset\Fake_videos"
FACES_FOLDER = r"C:\Users\CHARLES EKANEM\Documents\FINALYEARPROJECT\dataset\faces"
FACES_REAL = r"C:\Users\CHARLES EKANEM\Documents\FINALYEARPROJECT\dataset\faces\real"
FACES_FAKE = r"C:\Users\CHARLES EKANEM\Documents\FINALYEARPROJECT\dataset\faces\fake"
FACES_CSV = os.path.join(FACES_FOLDER, "balanced_faces_temp.csv")

print("Original videos:", ORIGINAL_VIDEOS_FOLDER)
print("Fake videos:", FAKE_VIDEOS_FOLDER)
print("Faces will be saved in:", FACES_REAL, "and", FACES_FAKE)
print("Metadata CSV will be:", FACES_CSV)
