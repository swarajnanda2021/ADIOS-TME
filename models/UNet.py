"""
ADIOS Mask Model - Exact implementation from Table 8 of ADIOS paper.
Reference: Shi et al. "Adversarial Masking for Self-Supervised Learning" (ICML 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



class ADIOSMaskModel(nn.Module):
    """
    ADIOS mask model - UNet with skip connections operating on RGB images.
    Based on: Shi et al., "Adversarial Masking for Self-Supervised Learning", ICML 2022
    
    Matches YugeTen's architecture from src/utils/unet.py
    """
    def __init__(self, num_masks=3, img_size=224, filter_start=32, norm='gn'):
        super().__init__()
        
        self.num_masks = num_masks
        self.img_size = img_size
        
        # Number of blocks based on image size (YugeTen formula)
        num_blocks = int(np.log2(img_size) - 1)  # 6 for 224x224
        self.num_blocks = num_blocks
        
        c = filter_start
        
        # Select normalization-aware conv block
        if norm == 'in':
            conv_block = ConvINReLU
        elif norm == 'gn':
            conv_block = ConvGNReLU
        else:
            conv_block = ConvReLU
        
        # Channel configurations based on num_blocks (from YugeTen)
        if num_blocks == 4:
            enc_in = [3, c, 2*c, 2*c]
            enc_out = [c, 2*c, 2*c, 2*c]
            dec_in = [4*c, 4*c, 4*c, 2*c]
            dec_out = [2*c, 2*c, c, c]
        elif num_blocks == 5:
            enc_in = [3, c, c, 2*c, 2*c]
            enc_out = [c, c, 2*c, 2*c, 2*c]
            dec_in = [4*c, 4*c, 4*c, 2*c, 2*c]
            dec_out = [2*c, 2*c, c, c, c]
        elif num_blocks == 6:
            enc_in = [3, c, c, c, 2*c, 2*c]
            enc_out = [c, c, c, 2*c, 2*c, 2*c]
            dec_in = [4*c, 4*c, 4*c, 2*c, 2*c, 2*c]
            dec_out = [2*c, 2*c, c, c, c, c]
        elif num_blocks == 7:
            enc_in = [3, c, c, c, c, 2*c, 2*c]
            enc_out = [c, c, c, c, 2*c, 2*c, 2*c]
            dec_in = [4*c, 4*c, 4*c, 4*c, 2*c, 2*c, 2*c]
            dec_out = [2*c, 2*c, 2*c, c, c, c, c]
        else:
            raise ValueError(f"num_blocks={num_blocks} not supported. Use 4, 5, 6, or 7.")
        
        # Build encoder (down) blocks
        self.down = nn.ModuleList()
        for i, o in zip(enc_in, enc_out):
            self.down.append(conv_block(i, o, 3, 1, 1))
        
        # Build decoder (up) blocks
        self.up = nn.ModuleList()
        for i, o in zip(dec_in, dec_out):
            self.up.append(conv_block(i, o, 3, 1, 1))
        
        # MLP bottleneck
        self.featuremap_size = img_size // (2 ** (num_blocks - 1))
        bottleneck_dim = 2 * c * self.featuremap_size * self.featuremap_size
        
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(bottleneck_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, bottleneck_dim),
            nn.ReLU(inplace=True)
        )
        
        # Final 1x1 conv to produce num_masks channels
        self.final_conv = nn.Conv2d(dec_out[-1], num_masks, kernel_size=1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.GroupNorm, nn.InstanceNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, images):
        """
        Args:
            images: RGB images [B, 3, H, W]
            
        Returns:
            dict with 'masks': [B, num_masks, H, W] with pixel-wise softmax
        """
        batch_size = images.size(0)
        
        x_down = [images]
        skip = []
        
        # Encoder path with skip connections
        for i, block in enumerate(self.down):
            act = block(x_down[-1])
            skip.append(act)
            if i < len(self.down) - 1:
                act = F.interpolate(act, scale_factor=0.5, mode='nearest')
            x_down.append(act)
        
        # MLP bottleneck
        x_up = self.mlp(x_down[-1])
        x_up = x_up.view(batch_size, -1, self.featuremap_size, self.featuremap_size)
        
        # Decoder path with skip connections
        for i, block in enumerate(self.up):
            # Concatenate with corresponding skip connection
            features = torch.cat([x_up, skip[-1 - i]], dim=1)
            x_up = block(features)
            if i < len(self.up) - 1:
                x_up = F.interpolate(x_up, scale_factor=2.0, mode='nearest')
        
        # Final conv to get mask logits
        logits = self.final_conv(x_up)
        
        # Pixel-wise softmax across masks
        masks = F.softmax(logits, dim=1)
        
        return {"masks": masks}


class ConvReLU(nn.Sequential):
    def __init__(self, nin, nout, kernel, stride=1, padding=0):
        super(ConvReLU, self).__init__(
            nn.Conv2d(nin, nout, kernel, stride, padding),
            nn.ReLU(inplace=True)
        )

class ConvINReLU(nn.Sequential):
    def __init__(self, nin, nout, kernel, stride=1, padding=0):
        super(ConvINReLU, self).__init__(
            nn.Conv2d(nin, nout, kernel, stride, padding, bias=False),
            nn.InstanceNorm2d(nout, affine=True),
            nn.ReLU(inplace=True)
        )

class ConvGNReLU(nn.Sequential):
    def __init__(self, nin, nout, kernel, stride=1, padding=0, groups=8):
        super(ConvGNReLU, self).__init__(
            nn.Conv2d(nin, nout, kernel, stride, padding, bias=False),
            nn.GroupNorm(groups, nout),
            nn.ReLU(inplace=True)
        )