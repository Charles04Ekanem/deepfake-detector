import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import classification_report, accuracy_score
from facenet_pytorch import InceptionResnetV1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")
WILD_DIR = os.path.join(BASE_DIR, "dataset", "DeepfakeTIMIT")
IMG_EXTS = (".jpg", ".jpeg", ".png")

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

class WildDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_paths = []
        self.labels = []

        for label_name in os.listdir(image_folder):
            label_path = os.path.join(image_folder, label_name)
            if os.path.isdir(label_path):
                label = 0 if label_name.lower() == "real" else 1
                for img_file in os.listdir(label_path):
                    if img_file.lower().endswith(IMG_EXTS):
                        self.image_paths.append(os.path.join(label_path, img_file))
                        self.labels.append(label)

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

def load_model(path):
    model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=2)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def evaluate():
    print("[INFO] Loading model...")
    model = load_model(MODEL_PATH)

    print("[INFO] Preparing dataset...")
    dataset = WildDataset(WILD_DIR, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["Real", "Fake"])

    print(f"[RESULT] Accuracy: {acc:.4f}")
    print("[RESULT] Classification Report:\n")
    print(report)

if __name__ == "__main__":
    evaluate()
