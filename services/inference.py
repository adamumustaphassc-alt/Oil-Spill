import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import os

# --- MODEL ARCHITECTURE ---
# You MUST define build_model or import it. 
# If you have a separate file for the model, use: from model_def import build_model
def build_model():
    """
    Ensure this matches the architecture used during training.
    If 'shallownet' is a specific class you wrote, define it here.
    """
    # Placeholder: Replace this with your actual model class/initialization
    # model = MyShallowNetClass() 
    # return model
    pass 

# --- CONFIGURATION ---
with open("model/class_to_idx.json", "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}

# Define the same transforms you used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Adjust to your model's input size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- CORE FUNCTIONS ---

def load_model():
    model_path = "model/shallownet.pth"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"CRITICAL: {model_path} not found. Check Git LFS status.")

    model = build_model() 
    if model is None:
        raise ValueError("build_model() returned None. Define your architecture in inference.py")
        
    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )
    model.eval()
    return model

def predict(model, image_bytes):
    """
    Takes raw image bytes from Streamlit's file_uploader, 
    transforms them, and returns the prediction.
    """
    # 1. Preprocess the image
    img = Image.open(image_bytes).convert('RGB')
    tensor = transform(img).unsqueeze(0) # Add batch dimension
    
    # 2. Run Inference
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)
        label = idx_to_class[pred.item()]
        
    return label, confidence.item()