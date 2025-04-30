import streamlit as st
import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import numpy as np
import gdown
import os
import torchvision.transforms as transforms

# === PAGE CONFIG & STYLING ===
st.set_page_config(page_title="DEEPFAKE DETECTION SYSTEM", page_icon="🕵️", layout="centered")

st.markdown("""
<style>
    footer {visibility: hidden;}
    .reportview-container {
        background: linear-gradient(135deg, #1f1c2c, #928dab);
        color: white;
    }
    .css-1d391kg {
        background-color: #262730 !important;
        color: white !important;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border-radius: 5px;
        padding: 0.4em 1em;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# === SIDEBAR ===
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3064/3064197.png", width=100)
st.sidebar.title("INFO")
st.sidebar.markdown("""
HELLO, This system detects whether a face image is **REAL** or a **DEEPFAKE**.

**HOW TO USE**:
1. Upload a face image (JPG/JPEG/PNG).
2. Get a prediction.
3. Please take note of the disclaimer below

**MODEL AND DATASET**:
- INCEPTIONRESNETV1
- Trained on FACEFORENSICS++ Dataset

[📂 Link to Github repo](https://github.com/Charles04Ekanem/deepfake-detector)
""")
st.sidebar.info("🧠 Powered by PyTorch + Streamlit", icon="ℹ️")

# === MODEL CONFIG ===
MODEL_PATH = "best_model.pth"
DRIVE_FILE_ID = "1-Uc3_jm0-_LkV0otAn9osBD3ra44hC0T" 

@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model..."):
            url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}" 
            gdown.download(url, MODEL_PATH, quiet=False)
    return MODEL_PATH

@st.cache_resource
def load_model(model_path):
    model = InceptionResnetV1(classify=True, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# === TRANSFORM ===
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === MAIN UI ===
st.title("🕵️ DEEPFAKE DETECTION SYSTEM")
st.markdown("Upload a **FACE IMAGE** to check if it's **REAL or DEEPFAKE**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0)

    model_path = download_model()
    model = load_model(model_path)

    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1).squeeze()[prediction].item()

    label = "Real" if prediction == 0 else "Deepfake"
    st.markdown(f"### 🔍 Prediction: **{label}**")
st.markdown("""---""")
st.markdown("""
### ⚠️ Disclaimer  
**This Deepfake Detection System is still under development and is meant for research and educational use.**  
It does **NOT GUARANTEE 100% ACCURACY** and should **NOT be used as a sole decision-making tool** in sensitive or critical scenarios.  
Please use the results carefully and combine these predictions with your own judgment or expert advice.
""")
st.markdown("© 2025 Charles Ekanem. All rights reserved.")

