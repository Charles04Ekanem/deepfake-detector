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

**Recommended Directory Structure**
![image](https://github.com/user-attachments/assets/761be6fb-25e1-4e0e-95b5-f0801986a684)
