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
    if not os.path.exists("/shallownet.pth"):
        os.makedirs("model", exist_ok=True)
        print("Downloading model from Google Drive...")
        gdown.download(
            id="1VuE0IRwpnHnnAL3MS1QZ9_ASR0mrFWFw",  # Google Drive file ID
            output="/shallownet.pth",
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

# --- CORE FUNCTIONS ---
def load_model():
    download_model()  # downloads only if file doesn't exist
    model = build_model()

    model = build_model() 
    if model is None:
        raise ValueError("build_model() returned None. Define your architecture in inference.py")
        
    model.load_state_dict(
        torch.load(model, map_location="cpu")
    )
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
