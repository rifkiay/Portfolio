import torch
from torchvision import models, transforms
from PIL import Image
from safetensors.torch import load_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['comedonica', 'conglobota', 'papulopusta']
num_classes = len(class_names)

# Transformasi sama seperti training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load model sekali saat pertama
model_path = "model/acne_classifier.safetensors"
model = models.resnet50()
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 1024),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(1024, num_classes)
)
model.load_state_dict(load_file(model_path))
model = model.to(device)
model.eval()

def predict_image(img_path):
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.nn.functional.softmax(output, dim=1)
        confidence, pred = torch.max(prob, 1)
        return class_names[pred.item()], confidence.item() * 100
