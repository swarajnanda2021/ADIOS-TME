
import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import trunc_normal_
from collections import OrderedDict
import numpy as np
import torch.nn.utils.spectral_norm as spectral_norm


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        # Clustering layers inspired by SwAV
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class TMEHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, bottleneck_dim=256, nlayers=3, use_bn=False):
        super().__init__()
        nlayers = max(nlayers, 1)
        layers = [nn.Linear(in_dim, hidden_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        # x = nn.functional.normalize(x, dim=-1, p=2) # We do not need this normalization layer
        return x




###################################################################################################


class Conv2DBlock(nn.Module):
    """Conv2DBlock with convolution followed by GroupNorm, ReLU activation and dropout"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
                bias=False,
            ),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class Deconv2DBlock(nn.Module):
    """Deconvolution block with ConvTranspose2d followed by Conv2d, GroupNorm, ReLU and dropout"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
                bias=False,
            ),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)    




class MaskModel(nn.Module):
    """ViT-UNet Mask Model with GroupNorm (single decoder for all masks)."""
    
    def __init__(self, encoder, num_masks, encoder_dim=768, drop_rate=0.1):
        super().__init__()

        self.encoder = encoder
        self.num_masks = num_masks
        self.embed_dim = encoder_dim
        self.drop_rate = drop_rate

        # Set dimensions based on encoder size
        if self.embed_dim < 512:
            self.skip_dim_11 = 256
            self.skip_dim_12 = 128
            self.bottleneck_dim = 256
        elif self.embed_dim < 1024:
            self.skip_dim_11 = 512
            self.skip_dim_12 = 256
            self.bottleneck_dim = 512
        else:
            self.skip_dim_11 = 768
            self.skip_dim_12 = 384
            self.bottleneck_dim = 768

        # Shared decoder blocks
        self.decoder0 = nn.Sequential(
            Conv2DBlock(3, 32, 3, dropout=self.drop_rate),
            Conv2DBlock(32, 64, 3, dropout=self.drop_rate),
        )
        self.decoder1 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, self.skip_dim_12, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_12, 128, dropout=self.drop_rate),
        )
        self.decoder2 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, 256, dropout=self.drop_rate),
        )
        self.decoder3 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.bottleneck_dim, dropout=self.drop_rate)
        )

        # Single decoder for all masks
        self.mask_decoder = self._create_upsampling_branch(num_masks)
        
        self._initialize_weights()

    def set_grad_checkpointing(self, enable=True):
        """Enable or disable gradient checkpointing in the encoder."""
        if hasattr(self.encoder, 'set_grad_checkpointing'):
            self.encoder.set_grad_checkpointing(enable)

    def _initialize_weights(self):
        """Initialize weights with Kaiming for conv layers."""
        def init_fn(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.decoder0.apply(init_fn)
        self.decoder1.apply(init_fn)
        self.decoder2.apply(init_fn)
        self.decoder3.apply(init_fn)
        self.mask_decoder.apply(init_fn)

    def _create_upsampling_branch(self, num_classes: int) -> nn.Module:
        """Create upsampling branch for mask generation."""
        from collections import OrderedDict
        
        bottleneck_upsampler = nn.ConvTranspose2d(
            in_channels=self.embed_dim,
            out_channels=self.bottleneck_dim,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )
        
        decoder3_upsampler = nn.Sequential(
            Conv2DBlock(self.bottleneck_dim * 2, self.bottleneck_dim, dropout=self.drop_rate),
            Conv2DBlock(self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate),
            Conv2DBlock(self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=self.bottleneck_dim,
                out_channels=256,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        
        decoder2_upsampler = nn.Sequential(
            Conv2DBlock(256 * 2, 256, dropout=self.drop_rate),
            Conv2DBlock(256, 256, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        
        decoder1_upsampler = nn.Sequential(
            Conv2DBlock(128 * 2, 128, dropout=self.drop_rate),
            Conv2DBlock(128, 128, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        
        decoder0_header = nn.Sequential(
            Conv2DBlock(64 * 2, 64, dropout=self.drop_rate),
            Conv2DBlock(64, 64, dropout=self.drop_rate),
            nn.Conv2d(
                in_channels=64,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        return nn.Sequential(
            OrderedDict([
                ("bottleneck_upsampler", bottleneck_upsampler),
                ("decoder3_upsampler", decoder3_upsampler),
                ("decoder2_upsampler", decoder2_upsampler),
                ("decoder1_upsampler", decoder1_upsampler),
                ("decoder0_header", decoder0_header),
            ])
        )

    def _forward_upsample(
        self,
        images: torch.Tensor,
        f1: torch.Tensor,
        f2: torch.Tensor,
        f3: torch.Tensor,
        f4: torch.Tensor,
    ) -> torch.Tensor:
        """Forward through decoder with skip connections."""
        b4 = self.mask_decoder.bottleneck_upsampler(f4)
        b3 = self.decoder3(f3)
        b3 = self.mask_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))
        
        b2 = self.decoder2(f2)
        b2 = self.mask_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))
        
        b1 = self.decoder1(f1)
        b1 = self.mask_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))
        
        b0 = self.decoder0(images)
        output = self.mask_decoder.decoder0_header(torch.cat([b0, b1], dim=1))
        
        return output

    def forward(self, images):
        """Forward pass generating soft masks."""
        # Get features from encoder
        features = self.encoder.get_intermediate_layers(images)
        f1, f2, f3, f4 = features

        # Reshape features from [B, N, D] to [B, D, H, W]
        num_patches = f1.shape[1] - (4 + 1)  # 4 register tokens + 1 cls token
        feature_size = int(np.sqrt(num_patches))

        f1 = f1[:, 5:, :].permute(0, 2, 1).view(f1.shape[0], -1, feature_size, feature_size).contiguous()
        f2 = f2[:, 5:, :].permute(0, 2, 1).view(f2.shape[0], -1, feature_size, feature_size).contiguous()
        f3 = f3[:, 5:, :].permute(0, 2, 1).view(f3.shape[0], -1, feature_size, feature_size).contiguous()
        f4 = f4[:, 5:, :].permute(0, 2, 1).view(f4.shape[0], -1, feature_size, feature_size).contiguous()

        # Single forward pass through decoder
        logits = self._forward_upsample(images, f1, f2, f3, f4)
        masks = torch.softmax(logits, dim=1)

        return {"masks": masks}