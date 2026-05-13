import torch
import torch.nn.functional as F
import json
import os

# Load class mapping
# Ensure class_to_idx.json is in your 'model' folder on GitHub
with open("model/class_to_idx.json", "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}

# Load model
def load_model():
    # We no longer call download_model() because the file is 
    # already included in the repo via Git LFS.
    
    model_path = "model/shallownet.pth"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Check your LFS upload.")

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