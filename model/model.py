import torch.nn as nn

def build_model():
    height = 224
    width = 224

    # Layers must be passed as arguments inside the parentheses
    model = nn.Sequential(
        # Layer 0: converts a 3D image to 1D vector
        nn.Flatten(), 

        # Layer 1: maps flattened input to 512 features
        nn.Linear(3 * height * width, 512), 
        
        # Layer 2: Activation function
        nn.ReLU(), 
        
        # Layer 3: reduces features from 512 to 128
        nn.Linear(512, 128),  
        
        # Layer 4: Activation function
        nn.ReLU(), 
        
        # Layer 5: final output for 2 classes (Oil Spill vs No Oil Spill)
        nn.Linear(128, 2)
    )

    return model
