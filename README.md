# deepfake-detector
FINAL YEAR PROJECT ON DEEPFAKE DETECTION WITH CNN (USING PYTORCH)

The rise of deepfakes presents a significant threat to digital media authenticity
and trust. This project presents a deepfake detection pipeline based on
InceptionResNetV1, pre-trained on VGGFace2 and fine-tuned on a balanced
dataset of 206,348 preprocessed facial images extracted from the Celeb-DF, Faceforensics and Openforensics datasets with a stratified 70/30 split ensuring
fair training and validation distribution. Despite being trained on CPU for only
20 epochs, the model achieved a final validation accuracy of 97.15% and a
validation loss of 0.0671, indicating strong generalization and low overfitting. The study highlights the viability of using computationally efficient training
setups alongside proper data preparation for high-accuracy deepfake detection. Future work will explore video-level analysis, ensemble learning and real-time
deployment strategies to expand the model’s applicability.

![image](https://github.com/user-attachments/assets/761be6fb-25e1-4e0e-95b5-f0801986a684)
                                **Recommended Directory Structure**

# Model Architecture
![image](https://github.com/user-attachments/assets/b9a75cc8-1900-4ef5-993b-3e95402f0ef6)

# Tools and Technologies
1. Python, Programming Language.
2. PyTorch, ML Framework.
3. Datasets: Ensemble Dataset of FaceForensics, OpenForensics and Celeb-DF V2.
4. Visualization: Matplotlib.
5. Deployment: Streamlit App.

# Project setup
git clone https://github.com/Charles04Ekanem/deepfake-detector.git
cd deepfake-detector
pip install -r requirements.txt

# Author
Charles Ekanem
B.Eng Student, Computer Engineering
[LinkedIn](https://www.linkedin.com/in/charles-ekanem-34b6402a6/) • [Twitter](https://twitter.com/charles3ekanem) • [Website](https://charles04ekanem.github.io/CharlesEkanem/)
