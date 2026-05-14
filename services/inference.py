import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import os
import gdown
from model.model import build_model

# Google Drive download
def download_model():
    # Use a relative path so it stays inside your app folder
    save_path = "model/shallownet.pth" 
    
    if not os.path.exists(save_path):
        os.makedirs("model", exist_ok=True)
        print("Downloading model from Google Drive...")
        gdown.download(
            id="1VuE0IRwpnHnnAL3MS1QZ9_ASR0mrFWFw",
            output=save_path, # Path matches the exists check
            quiet=False
        )
# def download_model():
#     if not os.path.exists("shallownet.pth"):
#         os.makedirs("model", exist_ok=True)
#         print("Downloading model from Google Drive...")
#         gdown.download(
#             id="1VuE0IRwpnHnnAL3MS1QZ9_ASR0mrFWFw",  # Google Drive file ID
#             output="shallownet.pth",
#             quiet=False
#         )
#         print("Model downloaded successfully.")


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
    download_model()  # This saves the file to "/shallownet.pth"
    
    # 1. Initialize the architecture
    model = build_model() 
    if model is None:
        raise ValueError("build_model() returned None.")
        
    # 2. FIX: Load the FILE PATH, not the model object
    # Use the same path defined in your download_model function
    model_path = "model/shallownet.pth" 
    
    state_dict = torch.load(model_path, map_location="cpu")
    
    # 3. Load the weights into the architecture
    model.load_state_dict(state_dict)
    
    model.eval()
    return model
    
# def load_model():
#     download_model()  # downloads only if file doesn't exist
#     model = build_model()

#     model = build_model() 
#     if model is None:
#         raise ValueError("build_model() returned None. Define your architecture in inference.py")
        
#     model.load_state_dict(
#         torch.load(model, map_location="cpu")
#     )
#     model.eval()
#     return model

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
