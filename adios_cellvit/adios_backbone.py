"""ADIOS mask model loader.

NOTE: Unlike the original ADIOS paper (Shi et al. 2022), this project keeps
the MASK MODEL and discards the STUDENT encoder. The user trains general
representation encoders separately by other methods; the mask model is the
asset of interest here.

The function loads only the mask model. The 768-dim student encoder present
in the checkpoint is intentionally ignored.
"""

import warnings

import torch

from models.vision_transformer.modern_vit import VisionTransformer
from models.vision_transformer.auxiliary_models import MaskModel


def load_adios_mask_model(
    checkpoint_path: str,
    device: str = 'cuda',
) -> MaskModel:
    """Load the ADIOS-TME mask model from a training checkpoint.

    The ADIOS-TME checkpoint contains both a 768-dim student encoder and the
    mask model. In this project we DO NOT use the student encoder (the user
    trains larger encoders separately via other methods). We only use the
    mask model:

      * ``MaskModel.encoder``  -- 192-dim ViT-Tiny, source of features for
                                  downstream HoVer and NC heads.
      * ``MaskModel.<UNet>``   -- 3-channel mask decoder, source of the
                                  channel-selector input.

    Pinned for this branch: ``mask_model_type='vit_unet'``,
    ``mask_encoder_dim=192``, ``num_masks=3``. Raises ``ValueError`` if the
    checkpoint args disagree.

    The returned mask model is in ``eval()`` mode with all parameters frozen
    (``requires_grad=False``). Stage 2 may call
    ``ADIOSCellViT.unfreeze_mask_model()`` to enable fine-tuning.

    Args:
        checkpoint_path: path to ADIOS-TME checkpoint.pth.
        device: 'cuda' or 'cpu'.

    Returns:
        mask_model: ``MaskModel`` (frozen, eval mode).
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

    mask_model = MaskModel(
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
    missing, unexpected = mask_model.load_state_dict(mask_state, strict=False)
    if missing:
        print(f"[adios_backbone] mask_model missing keys: {missing}")
    if unexpected:
        print(f"[adios_backbone] mask_model unexpected keys: {unexpected}")

    mask_model.to(device).eval()
    for p in mask_model.parameters():
        p.requires_grad = False

    return mask_model
