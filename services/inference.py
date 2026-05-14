import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import os
import gdown
from model.model import build_model

# 1. Google Drive download (Keep this logic for dynamic fetching)
def download_model():
    # Use relative path consistent with the rest of the project
    save_path = "model/shallownet.pth"
    if not os.path.exists(save_path):
        os.makedirs("model", exist_ok=True)
        print("Downloading model from Google Drive...")
        gdown.download(
            id="1VuE0IRwpnHnnAL3MS1QZ9_ASR0mrFWFw",
            output=save_path,
            quiet=False
        )
        print("Model downloaded successfully.")

# --- CONFIGURATION ---
with open("model/class_to_idx.json", "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}

# --- CORE FUNCTIONS ---

def load_model():
    download_model() 
    
    # Initialize the architecture
    model = build_model() 
    if model is None:
        raise ValueError("build_model() returned None. Check model/model.py")
    
    # FIX: Use the file path string, NOT the model object itself
    model_path = "model/shallownet.pth"
    
    # Load state dict from the file
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    
    model.eval()
    return model

def predict(model, tensor):
    """
    Takes a pre-processed tensor (already handled by app.py) 
    and returns the prediction.
    """
    # We remove the Image.open logic here because your app.py 
    # is already doing the preprocessing!
    
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)
        label = idx_to_class[pred.item()]
        
    return label, confidence.item()