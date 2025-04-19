import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image

device = torch.device('cpu')

# Load model
def load_model(model_path):
    model = InceptionResnetV1(classify=True, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Predict image
def predict_image(model, image):
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    img = transform(image).unsqueeze(0).to(device)
    output = model(img)
    _, predicted = torch.max(output, 1)
    return "Real" if predicted.item() == 0 else "Fake"
