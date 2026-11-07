import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes, pretrained=True):
    """
    Loads a ResNet18 model and modifies the final layer for our specific number of classes.
    We use ResNet18 because it is lightweight, fast to train, and sufficient for this task.
    """
    # 1. Load the pre-trained ResNet18 backbone. 
    # 'pretrained=True' means it has already learned basic features (edges, textures) from ImageNet.
    # This is crucial for Transfer Learning.
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    
    # 2. Get the number of input features for the final fully connected (fc) layer.
    num_ftrs = model.fc.in_features
    
    # 3. Replace the final layer.
    # The original ResNet18 outputs 1000 classes (for ImageNet).
    # We replace it with a new layer that outputs our specific number of classes (e.g., 4).
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def freeze_backbone(model):
    """
    Freezes all layers except the final fully connected layer.
    Used during initial Fine-Tuning to prevent destroying the pre-trained weights.
    """
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze the final layer
    for param in model.fc.parameters():
        param.requires_grad = True
        
    return model

def unfreeze_all(model):
    """
    Unfreezes all layers for full Fine-Tuning.
    """
    for param in model.parameters():
        param.requires_grad = True
    return model
