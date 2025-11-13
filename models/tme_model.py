"""
Combined model for ADIOS-TME training.
Includes student encoder with TME head.
"""

import torch
import torch.nn as nn


class TMEModel(nn.Module):
    """
    Student model for ADIOS-TME training.
    
    Args:
        backbone: Vision Transformer backbone
        tme_head: TME projection head
    """
    def __init__(self, backbone, tme_head):
        super().__init__()
        
        # Remove any classification heads
        if hasattr(backbone, 'fc'):
            backbone.fc = nn.Identity()
        if hasattr(backbone, 'head'):
            backbone.head = nn.Identity()
            
        self.backbone = backbone
        self.tme_head = tme_head
    
    def set_grad_checkpointing(self, enable=True):
        """Enable gradient checkpointing."""
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(enable)
    
    def forward(self, x):
        """
        Forward pass for TME embeddings.
        
        Args:
            x: Input images [B, C, H, W] or list of images
            
        Returns:
            TME embeddings [B, D] or list of embeddings
        """
        if isinstance(x, list):
            # Process multiple images
            embeddings = []
            for img in x:
                features = self.backbone(img)
                if isinstance(features, dict):
                    features = features['clstoken']
                emb = self.tme_head(features)
                embeddings.append(emb)
            return embeddings
        else:
            # Single image
            features = self.backbone(x)
            if isinstance(features, dict):
                features = features['clstoken']
            return self.tme_head(features)