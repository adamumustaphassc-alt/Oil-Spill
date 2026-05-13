import torch
import torch.nn.functional as F
import json
import os

# Load class mapping
# This assumes you have a 'model' folder in your repo containing this file
with open("model/class_to_idx.json", "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}

# Load model
def load_model():
    # DIRECT LOCAL PATH - No more gdown or Google Drive
    model_path = "model/shallownet.pth"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Verify it is uploaded to GitHub.")

    # build_model() must be defined or imported in this file
    model = build_model() 
    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )
    model.eval()
    return model

# Predict
def predict(model, tensor):
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)
        label = idx_to_class[pred.item()]
    return label, confidence.item()