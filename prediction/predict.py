import torch
from torchvision import models, transforms
from PIL import Image
from safetensors.torch import load_file
import os

# Konfigurasi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['comedonica', 'conglobota', 'papulopusta']
num_classes = len(class_names)

# Transformasi harus sama dengan saat training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load model hanya sekali
model_path = "model/acne_classifier.safetensors"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

# Load backbone
model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # Freezing backbone if needed

# Replace fc (head)
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 1024),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(1024, num_classes)
)

# Load hanya fc weights
fc_state = load_file(model_path)
model.fc.load_state_dict(fc_state)

# Move to device & eval mode
model = model.to(device)
model.eval()

# Fungsi prediksi
def predict_image(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Gambar tidak ditemukan: {img_path}")
    
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Error saat membuka gambar: {e}")
    
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.nn.functional.softmax(output, dim=1)
        confidence, pred = torch.max(prob, 1)
        return class_names[pred.item()], confidence.item() * 100
