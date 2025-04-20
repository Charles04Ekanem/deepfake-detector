import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from model_loader import load_model

st.set_page_config(page_title="Deepfake Detector", layout="centered")
st.markdown("<style>footer {visibility: hidden;}</style>", unsafe_allow_html=True)

st.title("🔍 DEEPFAKE DETECTION SYSTEM")
st.write("Upload a face image to check if it's **REAL** or **DEEPFAKE**!")

uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    # Load the model (safe download inside)
    model = load_model()

    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1).squeeze()[prediction].item()

    label = "Real" if prediction == 0 else "Deepfake"
    st.markdown(f"### 🧠 prediction: **{label}**")
    st.markdown(f"### 🔒 confidence: **{confidence:.2%}**")
