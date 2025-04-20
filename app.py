import streamlit as st
import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import numpy as np
import gdown
import os
import torchvision.transforms as transforms

# === SETUP PAGE ===
st.set_page_config(page_title="Deepfake Detector", page_icon="🕵️", layout="centered")
st.markdown("""
    <style>
        .reportview-container {
            background: linear-gradient(135deg, #1f1c2c, #928dab);
            color: white;
        }
        .stButton>button {
            color: white;
            background-color: #4CAF50;
            border-radius: 5px;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# === MODEL SETUP ===
MODEL_PATH = "best_model.pth"
DRIVE_FILE_ID = "1-Uc3_jm0-_LkV0otAn9osBD3ra44hC0T"

@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⏳ Downloading model..."):
            url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
    return MODEL_PATH

@st.cache_resource
def load_model(model_path):
    model = InceptionResnetV1(classify=True, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# === IMAGE TRANSFORM ===
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === HEADER ===
st.title("🕵️‍♂️ Deepfake Detection Web App")
st.subheader("Upload a face image to determine if it's **Real** or **Deepfake**.")

# === FILE UPLOAD ===
uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="👁 Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0)
    model_path = download_model()
    model = load_model(model_path)

    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1).squeeze()[prediction].item()

    label = "🟢 Real" if prediction == 0 else "🔴 Deepfake"
    st.markdown(f"## 🎯 Prediction: **{label}**")
    st.markdown(f"### 🔍 Confidence: **{confidence:.2%}**")
