import torch
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def generate_gradcam(model, input_tensor, original_image, target_class=None):
    """
    Generates a Grad-CAM heatmap overlay for a given image.
    
    Args:
        model: The trained PyTorch model.
        input_tensor: The preprocessed image tensor (1, C, H, W).
        original_image: The original image as a numpy array (H, W, 3) in [0, 1] range.
        target_class: The class index to generate the CAM for. If None, uses the highest scoring class.
        
    Returns:
        A visualization image combining the original image and the heatmap.
    """
    # For ResNet18, the last convolutional layer is layer4[-1]
    target_layers = [model.layer4[-1]]
    
    # Construct the CAM object once
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Define target
    targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
    
    # Generate the heatmap
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    
    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    
    # Resize original image to match CAM dimensions if necessary (it shouldn't be for this pipeline)
    if original_image.shape[:2] != grayscale_cam.shape:
        original_image = cv2.resize(original_image, (grayscale_cam.shape[1], grayscale_cam.shape[0]))
    
    # Ensure original image is in [0,1]
    if original_image.max() > 1.0:
        original_image = original_image / 255.0
        
    # Overlay heatmap on image
    visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True, image_weight=0.6)
    
    return visualization
