import torch
from facenet_pytorch import InceptionResnetV1
import os
import gdown

device = torch.device("cpu")
MODEL_PATH = "best_model.pth"
DRIVE_FILE_ID = "1-Uc3_jm0-_LkV0otAn9osBD3ra44hC0T"  # <-- Your model file ID

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

def load_model():
    download_model()
    model = InceptionResnetV1(classify=True, num_classes=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model
