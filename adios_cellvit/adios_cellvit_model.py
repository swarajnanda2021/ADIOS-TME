"""End-to-end nuclei counter: ADIOS mask model + selector + CellViT heads.

The MASK MODEL (192-dim ViT-Tiny encoder + UNet decoder) is the asset of
interest from ADIOS training; the original student encoder is not used.

Stage 1: only the selector is trained. The mask model is frozen.
Stage 2: the mask model (encoder + UNet decoder) is fine-tuned at LR 1e-6,
         the HoVer + NC heads are trained at LR 1e-4, the selector is
         parameter-frozen but still called (its softmax weights are part
         of the differentiable path).
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cellvit.models import CellViT

from .channel_selector import (
    ChannelSelector,
    collapse_channels_argmax,
    collapse_channels_soft,
)


class ADIOSCellViT(nn.Module):
    """Wrap ADIOS mask model + selector + CellViT into a single model.

    Components:
      * ``mask_model``   — ADIOS ViT-UNet mask model (192-dim ViT-Tiny encoder
                          + UNet decoder). Stage 1: frozen.
                          Stage 2: trainable at LR 1e-6 (encoder + decoder).
      * ``selector``     — ChannelSelector. Stage 1: trainable. Stage 2:
                          parameter-frozen but still called (its softmax
                          weights are part of the differentiable path).
      * ``cellvit``      — provides ``hv_map_decoder`` and
                          ``nuclei_type_map_decoder``. Constructed with
                          ``encoder=mask_model.encoder`` and
                          ``encoder_dim=192``; CellViT's small-encoder
                          decoder dimensioning kicks in automatically
                          (``embed_dim < 512`` branch).

    Forward returns a dict with:
      ``masks``           ``[B, 1, H, W]``  NP map from ADIOS+selector collapse.
      ``distances``       ``[B, 2, H, W]``  raw H/V logits.
      ``nuclei_types``    ``[B, K, H, W]``  raw NC class logits.
      ``channel_weights`` ``[B, N]``        selector softmax weights (logging).
    """

    def __init__(
        self,
        mask_model: nn.Module,
        selector: ChannelSelector,
        num_classes: int = 5,
        drop_rate: float = 0.1,
        inference_mode: str = 'soft',
    ):
        super().__init__()
        assert inference_mode in ('soft', 'argmax')

        self.mask_model = mask_model
        self.selector = selector
        self.inference_mode = inference_mode

        # CellViT consumes features from the 192-dim mask encoder.
        # encoder_dim=192 triggers CellViT's small-encoder decoder dimensioning
        # (skip_dim_11=256, skip_dim_12=128, bottleneck_dim=256).
        self.cellvit = CellViT(
            encoder=mask_model.encoder,
            encoder_dim=192,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )

    # ---- Stage-control helpers ---------------------------------------------

    def set_inference_mode(self, mode: str):
        assert mode in ('soft', 'argmax')
        self.inference_mode = mode

    def freeze_selector(self):
        for p in self.selector.parameters():
            p.requires_grad = False
        self.selector.eval()

    def unfreeze_mask_model(self):
        """Unfreeze the entire mask model for stage 2 fine-tuning.

        Unfreezes BOTH the 192-dim ViT-Tiny encoder (``mask_model.encoder``)
        and the UNet decoder. Both train at the lower LR group (1e-6).
        """
        for p in self.mask_model.parameters():
            p.requires_grad = True
        self.mask_model.train()

    # ---- Forward ------------------------------------------------------------

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 3-channel mask output from ADIOS mask model.
        # This call internally runs mask_model.encoder + UNet decoder.
        mask_output = self.mask_model(images)['masks']

        channel_logits = self.selector(mask_output)

        if self.training or self.inference_mode == 'soft':
            np_map = collapse_channels_soft(mask_output, channel_logits)
        else:
            np_map = collapse_channels_argmax(mask_output, channel_logits)

        # Features for HoVer + NC heads come from the 192-dim mask encoder.
        # TODO(perf): cache the mask_model encoder features from the mask_model
        # forward pass and reuse here. Requires modifying MaskModel.forward to
        # optionally return intermediates. Defer to post-validation.
        features = self.mask_model.encoder.get_intermediate_layers(images)
        f1, f2, f3, f4 = features
        num_registers = 4
        num_patches = f1.shape[1] - (num_registers + 1)
        feature_size = int(np.sqrt(num_patches))

        def _reshape(f):
            return (
                f[:, (num_registers + 1):, :]
                .permute(0, 2, 1)
                .reshape(f.shape[0], -1, feature_size, feature_size)
                .contiguous()
            )

        f1, f2, f3, f4 = (_reshape(f) for f in (f1, f2, f3, f4))

        hv = self.cellvit._forward_upsample(
            images, f1, f2, f3, f4, self.cellvit.hv_map_decoder
        )
        nc = self.cellvit._forward_upsample(
            images, f1, f2, f3, f4, self.cellvit.nuclei_type_map_decoder
        )

        return {
            'masks': np_map,
            'distances': hv,
            'nuclei_types': nc,
            'channel_weights': F.softmax(channel_logits.detach(), dim=1),
        }
