"""ADIOS-TME backbone + ViT-UNet mask decoder loader.

Loads the frozen encoder and mask decoder from an ADIOS-TME training
checkpoint. This branch is pinned to ``mask_model_type='vit_unet'`` with
``mask_encoder_dim=192`` and ``num_masks=3``. The UNet variant
(``mask_model_type='adios'``) is not supported here.
"""

import warnings
from typing import Tuple

import torch

from models.vision_transformer.modern_vit import VisionTransformer
from models.vision_transformer.auxiliary_models import MaskModel


def load_adios_backbone_and_decoder(
    checkpoint_path: str,
    device: str = 'cuda',
) -> Tuple[VisionTransformer, MaskModel]:
    """Load ADIOS encoder + ViT-UNet mask decoder from a checkpoint.

    Both the encoder and the mask decoder are returned in ``eval()`` mode with
    every parameter frozen (``requires_grad=False``). Stage 2 can later call
    ``model.unfreeze_mask_decoder()`` to allow fine-tuning.

    Args:
        checkpoint_path: path to ADIOS-TME checkpoint (e.g. iter_94000).
        device: 'cuda' or 'cpu'.

    Returns:
        (encoder, mask_decoder) — both frozen and in eval mode.

    Raises:
        ValueError: if the checkpoint was trained with a non-vit_unet mask
            model architecture.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    args = checkpoint['args']

    if getattr(args, 'mask_model_type', None) != 'vit_unet':
        raise ValueError(
            f"adios_backbone only supports mask_model_type='vit_unet', "
            f"got {getattr(args, 'mask_model_type', None)!r}"
        )

    mask_encoder_dim = args.mask_encoder_dim
    num_masks = args.num_masks
    if mask_encoder_dim != 192:
        warnings.warn(
            f"mask_encoder_dim={mask_encoder_dim} (expected 192 for this branch)"
        )
    if num_masks != 3:
        warnings.warn(
            f"num_masks={num_masks} (expected 3 for this branch)"
        )

    encoder = VisionTransformer(
        img_size=224,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.4,
        num_register_tokens=4,
    )

    backbone_state = {}
    for k, v in checkpoint['student'].items():
        if k.startswith('module.backbone.'):
            backbone_state[k[len('module.backbone.'):]] = v
        elif k.startswith('backbone.'):
            backbone_state[k[len('backbone.'):]] = v
    missing, unexpected = encoder.load_state_dict(backbone_state, strict=False)
    if missing:
        print(f"[adios_backbone] encoder missing keys: {missing}")
    if unexpected:
        print(f"[adios_backbone] encoder unexpected keys: {unexpected}")

    mask_encoder = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=mask_encoder_dim,
        depth=args.mask_encoder_depth,
        num_heads=max(mask_encoder_dim // 64, 3),
        mlp_ratio=4.0,
        drop_path_rate=0.1,
        num_register_tokens=4,
    )

    mask_decoder = MaskModel(
        encoder=mask_encoder,
        num_masks=num_masks,
        encoder_dim=mask_encoder_dim,
        drop_rate=0.0,
    )

    mask_state = {}
    for k, v in checkpoint['mask_model'].items():
        if k.startswith('module.'):
            mask_state[k[len('module.'):]] = v
        else:
            mask_state[k] = v
    missing, unexpected = mask_decoder.load_state_dict(mask_state, strict=False)
    if missing:
        print(f"[adios_backbone] mask_decoder missing keys: {missing}")
    if unexpected:
        print(f"[adios_backbone] mask_decoder unexpected keys: {unexpected}")

    encoder.to(device).eval()
    mask_decoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    for p in mask_decoder.parameters():
        p.requires_grad = False

    return encoder, mask_decoder
