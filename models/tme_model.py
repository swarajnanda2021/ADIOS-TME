"""
Combined model for ADIOS-TME training.
Includes student encoder with TME head - OPTIMIZED VERSION.
"""

import torch
import torch.nn as nn


class TMEModel(nn.Module):
    """
    Student model for ADIOS-TME training with efficient batched forward passes.
    
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
        Forward pass for TME embeddings with efficient batching.
        
        Args:
            x: Input images [B, C, H, W] or list of images
            
        Returns:
            TME embeddings [B, D] or list of embeddings
        """
        if isinstance(x, list):
            # EFFICIENT: Concatenate all images into single batch
            batch_sizes = [img.shape[0] for img in x]
            all_images = torch.cat(x, dim=0)
            
            # SINGLE forward through backbone (not N separate forwards!)
            features = self.backbone(all_images)
            if isinstance(features, dict):
                features = features['clstoken']
            
            # SINGLE forward through head
            all_embeddings = self.tme_head(features)
            
            # Split back into list maintaining original structure
            embeddings = []
            start_idx = 0
            for bs in batch_sizes:
                embeddings.append(all_embeddings[start_idx:start_idx + bs])
                start_idx += bs
            
            return embeddings
        else:
            # Single image
            features = self.backbone(x)
            if isinstance(features, dict):
                features = features['clstoken']
            return self.tme_head(features)