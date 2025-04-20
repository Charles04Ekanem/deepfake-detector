import streamlit as st
from PIL import Image
from model_loader import load_model, predict_image

st.title("🕵️ DEEPFAKE DETECTION SYSTEM")
st.write("UPLOAD A FACE IMAGE TO FIND OUT IF IT'S **REAL** OR **FAKE**!")

model_path = "best_model.pth"
model = load_model(model_path)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect"):
        prediction = predict_image(model, image)
        st.success(f"Prediction: **{prediction}**")
