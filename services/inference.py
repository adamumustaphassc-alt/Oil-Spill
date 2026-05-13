import torch
import torch.nn.functional as F
import json
import os

# 1. Load class mapping locally
with open("model/class_to_idx.json", "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}

# 2. Load the model from the local folder
def load_model():
    model_path = "model/shallownet.pth"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"CRITICAL: {model_path} not found. Check Git LFS status.")

    # Note: Ensure build_model() is defined in your script or imported
    model = build_model() 
    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )
    model.eval()
    return model

# 3. Prediction Logic
def predict(model, tensor):
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)
        label = idx_to_class[pred.item()]
    return label, confidence.item()