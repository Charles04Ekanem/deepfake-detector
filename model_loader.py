import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import gdown
import os

device = torch.device('cpu')

MODEL_PATH = "best_model.pth"
DRIVE_FILE_ID = "1-Uc3_jm0-_LkV0otAn9osBD3ra44hC0T"

def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

def load_model():
    download_model_if_needed()
    model = InceptionResnetV1(classify=True, num_classes=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

def predict_image(model, image):
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    img = transform(image).unsqueeze(0).to(device)
    output = model(img)
    _, predicted = torch.max(output, 1)
    return "Real" if predicted.item() == 0 else "Fake"

