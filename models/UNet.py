"""
ADIOS Mask Model - Exact implementation from Table 8 of ADIOS paper.
Reference: Shi et al. "Adversarial Masking for Self-Supervised Learning" (ICML 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ADIOSMaskModel(nn.Module):
    """
    U-Net based mask generator from ADIOS paper.
    
    Architecture from Table 8:
    - Downsampling path: 5 conv layers with GroupNorm
    - MLP bottleneck: 3 FC layers
    - Upsampling path: 5 conv layers with GroupNorm
    - Output head: 1x1 conv + softmax
    
    Args:
        num_masks: Number of masks to generate (N in paper)
        img_size: Input image size (default: 224)
    """
    def __init__(self, num_masks=4, img_size=224):
        super().__init__()
        self.num_masks = num_masks
        self.img_size = img_size
        
        # ============ Downsampling Path ============
        # Input: [B, 3, 224, 224]
        
        # Block 1: 3 -> 8 channels
        self.down_conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.down_gn1 = nn.GroupNorm(4, 8)  # 4 groups for 8 channels
        
        self.down_conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.down_gn2 = nn.GroupNorm(4, 8)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 224 -> 112
        
        # Block 2: 8 -> 16 channels
        self.down_conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.down_gn3 = nn.GroupNorm(8, 16)  # 8 groups for 16 channels
        
        self.down_conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.down_gn4 = nn.GroupNorm(8, 16)
        
        self.down_conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.down_gn5 = nn.GroupNorm(8, 16)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 112 -> 56
        
        # ============ MLP Bottleneck ============
        # Input: [B, 16, 56, 56] -> [B, 16*56*56] = [B, 50176]
        self.flatten_size = 16 * (img_size // 4) * (img_size // 4)  # 16 * 56 * 56
        
        self.mlp_fc1 = nn.Linear(self.flatten_size, 128)
        self.mlp_fc2 = nn.Linear(128, 128)
        self.mlp_fc3 = nn.Linear(128, 256)
        
        # ============ Upsampling Path ============
        # We need to go back to spatial dimensions
        # 256 -> reshape to [B, 16, 4, 4] then upsample
        self.mlp_to_spatial = nn.Linear(256, 16 * 4 * 4)  # Small spatial size to start
        self.upsample_size = 4
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 4 -> 8
        
        # Block 1: 16 channels
        self.up_conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.up_gn1 = nn.GroupNorm(8, 16)
        
        self.up_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.up_gn2 = nn.GroupNorm(8, 16)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 8 -> 16
        
        # Block 2: 16 -> 8 channels
        self.up_conv3 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
        self.up_gn3 = nn.GroupNorm(4, 8)
        
        self.up_conv4 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.up_gn4 = nn.GroupNorm(4, 8)
        
        self.up_conv5 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.up_gn5 = nn.GroupNorm(4, 8)
        
        # Final upsampling to match input size
        # Currently at 16x16, need to go to 224x224
        self.final_upsample = nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=False)
        
        # ============ Occlusion Head ============
        self.head = nn.Conv2d(8, num_masks, kernel_size=1, stride=1, padding=0)
        
        # Softmax is applied in forward (across mask dimension)
        
    def forward(self, x):
        """
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            Dictionary with 'masks': [B, num_masks, H, W] with softmax applied
        """
        batch_size = x.size(0)
        
        # ============ Downsampling ============
        # Block 1
        x = self.down_conv1(x)
        x = self.down_gn1(x)
        x = F.relu(x)
        
        x = self.down_conv2(x)
        x = self.down_gn2(x)
        x = F.relu(x)
        
        x = self.pool1(x)  # [B, 8, 112, 112]
        
        # Block 2
        x = self.down_conv3(x)
        x = self.down_gn3(x)
        x = F.relu(x)
        
        x = self.down_conv4(x)
        x = self.down_gn4(x)
        x = F.relu(x)
        
        x = self.down_conv5(x)
        x = self.down_gn5(x)
        x = F.relu(x)
        
        x = self.pool2(x)  # [B, 16, 56, 56]
        
        # ============ MLP Bottleneck ============
        x = x.view(batch_size, -1)  # [B, 16*56*56]
        
        x = self.mlp_fc1(x)
        x = F.relu(x)
        
        x = self.mlp_fc2(x)
        x = F.relu(x)
        
        x = self.mlp_fc3(x)
        x = F.relu(x)  # [B, 256]
        
        # ============ Upsampling ============
        # Back to spatial
        x = self.mlp_to_spatial(x)  # [B, 16*4*4]
        x = x.view(batch_size, 16, self.upsample_size, self.upsample_size)  # [B, 16, 4, 4]
        
        x = self.up1(x)  # [B, 16, 8, 8]
        
        # Block 1
        x = self.up_conv1(x)
        x = self.up_gn1(x)
        x = F.relu(x)
        
        x = self.up_conv2(x)
        x = self.up_gn2(x)
        x = F.relu(x)
        
        x = self.up2(x)  # [B, 16, 16, 16]
        
        # Block 2
        x = self.up_conv3(x)
        x = self.up_gn3(x)
        x = F.relu(x)
        
        x = self.up_conv4(x)
        x = self.up_gn4(x)
        x = F.relu(x)
        
        x = self.up_conv5(x)
        x = self.up_gn5(x)
        x = F.relu(x)  # [B, 8, 16, 16]
        
        # Upsample to full resolution
        x = self.final_upsample(x)  # [B, 8, 224, 224]
        
        # ============ Occlusion Head ============
        masks = self.head(x)  # [B, num_masks, 224, 224]
        
        # Apply softmax across mask dimension (ensures sum to 1 per pixel)
        masks = F.softmax(masks, dim=1)
        
        return {"masks": masks}
    
    def get_num_params(self):
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters())