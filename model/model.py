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


    # # Conv Block 1
    # model.append(nn.Conv2d(3, 16, kernel_size=3, padding=1))
    # model.append(nn.ReLU())
    # model.append(nn.MaxPool2d(2, 2))

    # # Conv Block 2
    # model.append(nn.Conv2d(16, 32, kernel_size=3, padding=1))
    # model.append(nn.ReLU())
    # model.append(nn.MaxPool2d(2, 2))

    # # Conv Block 3
    # model.append(nn.Conv2d(32, 64, kernel_size=3, padding=1))
    # model.append(nn.ReLU())
    # model.append(nn.MaxPool2d(2))

    # # Classifier
    # model.append(nn.Flatten())
    # model.append(nn.Dropout())

    # model.append(nn.Linear(50176, 500))
    # model.append(nn.ReLU())
    # model.append(nn.Dropout())

    # model.append(nn.Linear(500, 8))
