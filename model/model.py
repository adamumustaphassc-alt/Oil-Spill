import torch.nn as nn

def build_model():
    height = 224
    width = 224

    model = nn.Sequential()

    #converts a 3D image to 1D vector
    nn.Flatten(), 

    nn.Linear(3 * height * width, 512), #takes the flattened input &maps it to 512 features(neurons)
    nn.ReLU(), #Activation function which introduce non-linearity and allow model to learn complex patterns
    nn.Linear(512, 128),  #reduces features from 512 input to 128(neurons)
    nn.ReLU(), #Adds non-linearity again, prevents model from becoming just linear math
    nn.Linear(128, 2)

    return model
