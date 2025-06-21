import os
import uuid
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from Constants import ORIGINAL_VIDEOS_FOLDER, FAKE_VIDEOS_FOLDER, FACES_FOLDER, FACES_CSV

class Faces(Dataset):
    def __init__(self, csv_file, root_dir, split="training", transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        if transform is True:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
        elif transform is False or transform is None:
            self.transform = None
        else:
            transform = transform
        split="training"
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = int(idx)
        img_path = self.data.iloc[idx]['name']
        label = int(self.data.iloc[idx]['label'])

        # Determine if the image is real or fake
        if label == 1:
            folder = os.path.join("dataset", "faces", "real")
        else:
            folder = os.path.join("dataset", "faces", "fake")

        full_path = os.path.join(folder, img_path)
        image = Image.open(full_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label).float()

class FaceExtractor:
    def __init__(self, real_dir, fake_dir, csv_path, max_total_faces=20000, faces_per_video=10):
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.csv_path = csv_path
        self.max_total_faces = max_total_faces
        self.faces_per_video = faces_per_video

        os.makedirs(self.real_dir, exist_ok=True)
        os.makedirs(self.fake_dir, exist_ok=True)

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.data = []

    def extract_faces_from_video(self, video_path, label, output_dir, max_faces):
        cap = cv2.VideoCapture(video_path)
        count = 0

        while cap.isOpened() and count < max_faces:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                filename = f"{Path(video_path).stem}_{uuid.uuid4().hex}.jpg"
                path = os.path.join(output_dir, filename)
                cv2.imwrite(path, face_img)
                self.data.append({"name": filename, "label": label})
                count += 1
                if count >= max_faces:
                    break
        cap.release()

    def run(self):
        print(f"Original videos: {ORIGINAL_VIDEOS_FOLDER}")
        print(f"Fake videos: {FAKE_VIDEOS_FOLDER}")
        print(f"Faces will be saved in: {self.real_dir} and {self.fake_dir}")
        print(f"Metadata CSV will be: {self.csv_path}")

        real_videos = [f for f in os.listdir(ORIGINAL_VIDEOS_FOLDER) if f.endswith(".mp4")]
        fake_videos = [f for f in os.listdir(FAKE_VIDEOS_FOLDER) if f.endswith(".mp4")]

        total_real_videos = len(real_videos)
        total_fake_videos = len(fake_videos)
        print(f"[INFO] Total real videos: {total_real_videos}")
        print(f"[INFO] Total fake videos: {total_fake_videos}")

        # Calculate proportional fake faces
        total_real_faces = total_real_videos * self.faces_per_video
        fake_faces_per_video = max(1, (total_real_faces // total_fake_videos))

        print(f"[INFO] Extracting {self.faces_per_video} faces per real video.")
        print(f"[INFO] Extracting {fake_faces_per_video} faces per fake video (to balance).")

        # === Extract faces ===
        for video in real_videos:
            print(f"[REAL] Processing {video}")
            self.extract_faces_from_video(
                os.path.join(ORIGINAL_VIDEOS_FOLDER, video),
                label=1,  # 1 = real
                output_dir=self.real_dir,
                max_faces=self.faces_per_video
            )

        for video in fake_videos:
            print(f"[FAKE] Processing {video}")
            self.extract_faces_from_video(
                os.path.join(FAKE_VIDEOS_FOLDER, video),
                label=0,  # 0 = fake
                output_dir=self.fake_dir,
                max_faces=fake_faces_per_video
            )

        # === Add pre-existing images ===
        for folder, label in [(self.real_dir, 1), (self.fake_dir, 0)]:
            for file in os.listdir(folder):
                if file.endswith(('.jpg', '.png')):
                    self.data.append({"name": file, "label": label})

        # === Save to CSV ===
        df = pd.DataFrame(self.data).drop_duplicates(subset=["name"])
        df.to_csv(self.csv_path, index=False)
        print(f"[INFO] Metadata saved to {self.csv_path}")

        # === Visualizations ===
        self.visualize_counts(df)
        self.visualize_examples(df)

    def visualize_counts(self, df):
        counts = df["label"].value_counts().to_dict()
        num_fake = counts.get(0, 0)
        num_real = counts.get(1, 0)
        print(f"[INFO] Total Real Faces: {num_real}")
        print(f"[INFO] Total Fake Faces: {num_fake}")

        # Bar chart
        plt.figure(figsize=(6, 4))
        plt.bar(['Real', 'Fake'], [num_real, num_fake], color=['green', 'red'])
        plt.title("Face Class Distribution")
        plt.ylabel("Number of Faces")
        plt.tight_layout()
        plt.savefig(os.path.join(FACES_FOLDER, "class_distribution_bar.png"))
        plt.show()

        # Pie chart
        plt.figure(figsize=(6, 4))
        plt.pie([num_real, num_fake], labels=['Real', 'Fake'], autopct='%1.1f%%', colors=['green', 'red'])
        plt.title("Face Class Proportions")
        plt.tight_layout()
        plt.savefig(os.path.join(FACES_FOLDER, "class_distribution_pie.png"))
        plt.show()

    def visualize_examples(self, df, max_per_class=3):
        for label, label_name in [(1, "real"), (0, "fake")]:
            samples = df[df["label"] == label]["name"].sample(n=min(max_per_class, df["label"].value_counts().get(label, 0)))
            for name in samples:
                folder = self.real_dir if label == 1 else self.fake_dir
                path = os.path.join(folder, name)
                if os.path.exists(path):
                    img = cv2.imread(path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    plt.imshow(img)
                    plt.title(f"{label_name.upper()} - {name}")
                    plt.axis("off")
                    plt.show()

# === RUN SCRIPT ===
if __name__ == "__main__":
    extractor = FaceExtractor(
        real_dir=os.path.join(FACES_FOLDER, "real"),
        fake_dir=os.path.join(FACES_FOLDER, "fake"),
        csv_path=FACES_CSV
    )
    extractor.run()
