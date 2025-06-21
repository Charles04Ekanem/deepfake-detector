import os
import torch
import torch.utils.data as data_utils
from torch import nn, optim
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
import numpy as np
from dataset.Faces import Faces
import json

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACES_FOLDER = os.path.join(BASE_DIR, "dataset", "faces")
FACES_CSV = os.path.join(FACES_FOLDER, "balanced_faces_temp.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = {0: "fake", 1: "real"}
SPLIT_RATIO = 0.7
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "training_progress.json")

CONFIG = dict(
    epochs=20,
    batch_size=32,
    learning_rate=0.01,
    save_path=os.path.join(OUTPUT_DIR, "best_model.pth")
)

# Data Loaders for Validation and Training
def get_data_loaders():
    print("[INFO] Loading dataset...")
    dataset = Faces(root_dir=FACES_FOLDER, csv_file=FACES_CSV, transform=True)
    indices = torch.randperm(len(dataset))
    split_idx = int(SPLIT_RATIO * len(dataset))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    train_loader = data_utils.DataLoader(data_utils.Subset(dataset, train_idx),
                                         batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = data_utils.DataLoader(data_utils.Subset(dataset, val_idx),
                                       batch_size=CONFIG['batch_size'], shuffle=False)

    print(f"[INFO] Training set size: {len(train_idx)}")
    print(f"[INFO] Validation set size: {len(val_idx)}")
    return train_loader, val_loader

# Managing Training/Validation Progress
def save_progress(epoch):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({"last_completed_epoch": epoch}, f)

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f).get("last_completed_epoch", -1)
    return -1

# Training process
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0
    best_model_state = None
    start_epoch = load_progress() + 1

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            train_loss, train_acc = 0, 0
            for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                out = model(x).squeeze()
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                preds = (torch.sigmoid(out) > 0.5).float()
                train_loss += loss.item()
                train_acc += (preds == y).float().mean().item()

            model.eval()
            val_loss, val_acc = 0, 0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for x, y in tqdm(val_loader, desc="Validating"):
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    out = model(x).squeeze()
                    loss = criterion(out, y)
                    preds = (torch.sigmoid(out) > 0.5).float()
                    val_loss += loss.item()
                    val_acc += (preds == y).float().mean().item()
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_train_acc = train_acc / len(train_loader)
            avg_val_acc = val_acc / len(val_loader)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["train_acc"].append(avg_train_acc)
            history["val_acc"].append(avg_val_acc)

            print(f"[INFO] Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Acc={avg_train_acc:.4f} | Val Loss={avg_val_loss:.4f}, Acc={avg_val_acc:.4f}")

            # Saves best model
            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                best_model_state = model.state_dict()
                torch.save(best_model_state, CONFIG["save_path"])
                print(f"[INFO] New best model saved at epoch {epoch+1}")

            # Checkpointing
            checkpoint_path = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            save_progress(epoch)
            print(f"[INFO] Checkpoint saved: {checkpoint_path}")

    except KeyboardInterrupt:
        print("[WARNING] Training interrupted. Saving model...")
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "interrupted_model.pth"))
        save_progress(epoch - 1)
        print("[INFO] Progress saved. You can resume training later.")
        return history, all_preds, all_labels

    # Final save
    torch.save(best_model_state, CONFIG["save_path"])
    print(f"[INFO] Best model saved to {CONFIG['save_path']}")
    return history, all_preds, all_labels

# Visualization
def plot_metrics(history, all_preds, all_labels):
    # Loss vs Epoch
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.title("Loss vs Epoch")
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_vs_epoch.png"))
    plt.close()

    # Accuracy vs Epoch
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.title("Accuracy vs Epoch")
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_vs_epoch.png"))
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()

    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=["Fake", "Real"])
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)
    print("[INFO] Classification report saved.")

# Execution
if __name__ == "__main__":
    model = InceptionResnetV1(pretrained="vggface2", classify=True, num_classes=1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    train_loader, val_loader = get_data_loaders()
    history, all_preds, all_labels = train_model(model, train_loader, val_loader, criterion, optimizer, CONFIG['epochs'])
    plot_metrics(history, all_preds, all_labels)
    print("[INFO] Training complete.")
