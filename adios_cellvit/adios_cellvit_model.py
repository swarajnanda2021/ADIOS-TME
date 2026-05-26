"""End-to-end nuclei counter: ADIOS backbone + selector + CellViT heads.

The NP branch from CellViT is overridden by the ADIOS mask decoder collapsed
through ``ChannelSelector``. The HV and NC branches come from CellViT.
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
    """Wrap ADIOS + selector + CellViT into a single model.

    Components:
      * ``encoder``      — ViT, frozen in both stages.
      * ``mask_decoder`` — ADIOS ViT-UNet decoder. Stage 1: frozen.
                          Stage 2: trainable at LR 1e-6.
      * ``selector``     — ChannelSelector. Stage 1: trainable. Stage 2:
                          parameter-frozen but still called (its softmax
                          weights are part of the differentiable path).
      * ``cellvit``      — provides ``hv_map_decoder`` and
                          ``nuclei_type_map_decoder``. Its own NP branch is
                          left in place (so unchanged upstream code still
                          works) but is not called by us.

    Forward returns a dict with:
      ``masks``           ``[B, 1, H, W]``  NP map from ADIOS+selector collapse.
      ``distances``       ``[B, 2, H, W]``  raw H/V logits.
      ``nuclei_types``    ``[B, K, H, W]``  raw NC class logits.
      ``channel_weights`` ``[B, N]``        selector softmax weights (logging).
    """

    def __init__(
        self,
        encoder: nn.Module,
        mask_decoder: nn.Module,
        selector: ChannelSelector,
        encoder_dim: int = 768,
        num_classes: int = 5,
        drop_rate: float = 0.1,
        inference_mode: str = 'soft',
    ):
        super().__init__()
        assert inference_mode in ('soft', 'argmax')

        self.encoder = encoder
        self.mask_decoder = mask_decoder
        self.selector = selector
        self.inference_mode = inference_mode

        # CellViT freezes self.encoder.parameters() in its __init__ — which is
        # what we want — and the encoder reference is shared, so its features
        # are computed once per forward.
        self.cellvit = CellViT(
            encoder=encoder,
            encoder_dim=encoder_dim,
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

    def unfreeze_mask_decoder(self):
        for p in self.mask_decoder.parameters():
            p.requires_grad = True
        self.mask_decoder.train()

    # ---- Forward ------------------------------------------------------------

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        mask_output = self.mask_decoder(images)['masks']

        channel_logits = self.selector(mask_output)

        if self.training or self.inference_mode == 'soft':
            np_map = collapse_channels_soft(mask_output, channel_logits)
        else:
            np_map = collapse_channels_argmax(mask_output, channel_logits)

        features = self.encoder.get_intermediate_layers(images)
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
