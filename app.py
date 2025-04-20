import streamlit as st
import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import numpy as np
import gdown
import os
import torchvision.transforms as transforms

# SETUP MODEL PATH & DOWNLOAD FROM GOOGLE DRIVE 
MODEL_PATH = "best_model.pth"
DRIVE_FILE_ID = "1-Uc3_jm0-_LkV0otAn9osBD3ra44hC0T"  

@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            url = f"https://drive.google.com/uc?id=1-Uc3_jm0-_LkV0otAn9osBD3ra44hC0T"
            gdown.download(url, MODEL_PATH, quiet=False)
    return MODEL_PATH

# LOAD MODEL 
@st.cache_resource
def load_model(model_path):
    model = InceptionResnetV1(classify=True, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# DEFINE PREPROCESSING 
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === UI ===
st.title("DEEPFAKE DETECTION SYSTEM, ANIE-AKAN")
st.write("Upload a face image to check if it's REAL or DEEPFAKE")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Load model
    model_path = download_model()
    model = load_model(model_path)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1).squeeze()[prediction].item()

    # Display result
    label = "Real" if prediction == 0 else "Deepfake"
    st.markdown(f"### Prediction: **{label}**")
    st.markdown(f"**Confidence:** {confidence:.2%}")
