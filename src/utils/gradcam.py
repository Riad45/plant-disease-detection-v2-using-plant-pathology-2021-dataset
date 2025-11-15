"""
Grad-CAM implementation for XAI (Explainable AI)
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch model
            target_layer: Layer to compute gradients from
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, input_image, target_class=None):
        """
        Generate CAM for input image
        
        Args:
            input_image: Input tensor (1, C, H, W)
            target_class: Target class index (if None, use predicted class)
            
        Returns:
            cam: Class activation map (H, W)
            prediction: Predicted class
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_image)
        
        # Get predicted class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # Generate CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=0)  # (H, W)
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy(), target_class
    
    def visualize_cam(self, original_image, cam, alpha=0.5):
        """
        Overlay CAM on original image
        
        Args:
            original_image: Original PIL Image or numpy array
            cam: Class activation map (H, W)
            alpha: Transparency of overlay
            
        Returns:
            visualization: PIL Image with CAM overlay
        """
        # Convert original image to numpy
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        
        # Resize CAM to match image size
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Convert CAM to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap on original image
        visualization = heatmap * alpha + original_image * (1 - alpha)
        visualization = np.uint8(visualization)
        
        return Image.fromarray(visualization)


def get_target_layer(model, model_name):
    """
    Get the target layer for Grad-CAM based on model architecture
    
    Args:
        model: PyTorch model
        model_name: Name of model architecture
        
    Returns:
        target_layer: Target layer for Grad-CAM
    """
    if 'convnext' in model_name:
        # ConvNeXt: Last layer of last stage
        target_layer = model.stages[-1].blocks[-1]
    elif 'swin' in model_name:
        # Swin: Last layer of last stage
        target_layer = model.layers[-1].blocks[-1]
    elif 'deit' in model_name or 'vit' in model_name:
        # ViT/DeiT: Last block
        target_layer = model.blocks[-1].norm1
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")
    
    return target_layer


if __name__ == "__main__":
    print("✅ GradCAM module loaded successfully!")