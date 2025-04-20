import torch
from facenet_pytorch import InceptionResnetV1
import os
import gdown

device = torch.device("cpu")

def download_model(model_path: str, file_id: str):
    if not os.path.exists(model_path):
        print("[INFO] Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)

def load_model(model_path="best_model.pth", file_id="1-Uc3_jm0-_LkV0otAn9osBD3ra44hC0T"):
    download_model(model_path, file_id)
    model = InceptionResnetV1(classify=True, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
