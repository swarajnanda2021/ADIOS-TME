"""ViT-B encoder loader for FMC_ViT-B_<recipe> checkpoints.

The FMC fork (sibling DINOv2 codebase under
``/data1/vanderbc/test_dinov2_swaraj/FMC_ViT-B_*``) stores its training args
under non-standard names: ``embeddingdim``, ``vitdepth``, ``vitheads``. This
loader reads those, builds a stock ``VisionTransformer``, and loads the student
weights with the standard ``module.backbone.`` / ``backbone.`` prefix probe.

The encoder is returned **unfrozen**; the caller (train_vitb.py) decides
parameter groups and LRs.
"""

from typing import Tuple

import torch
import torch.nn as nn

from models.vision_transformer.modern_vit import VisionTransformer


def load_vitb_encoder(
    checkpoint_path: str,
    device: str = 'cuda',
) -> Tuple[nn.Module, int, int, int]:
    """Load a ViT-B encoder from an FMC_ViT-B_<recipe> checkpoint.

    Args:
        checkpoint_path: path to checkpoint.pth (e.g. ``checkpoint_iter_00150000.pth``).
        device: 'cuda' or 'cpu'.

    Returns:
        ``(encoder, patch_size, num_registers, embed_dim)``.
        ``encoder`` is on ``device`` with all parameters in their default
        trainable state (caller may freeze).
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    args = checkpoint['args']

    # FMC fork field names (see handoff §"NEW: vitb_backbone.py"). If any of
    # these are missing the recipe is non-standard; stop and ask rather than
    # guess a fallback.
    try:
        patch_size = args.patch_size
        embed_dim = args.embeddingdim
        depth = args.vitdepth
        num_heads = args.vitheads
    except AttributeError as e:
        raise RuntimeError(
            f"Checkpoint args missing expected FMC field ({e}). The FMC fork uses "
            "'embeddingdim'/'vitdepth'/'vitheads'; some recipes may differ. Inspect "
            f"checkpoint['args'] and update load_vitb_encoder accordingly."
        ) from e
    num_registers = getattr(args, 'num_register_tokens', 4)

    encoder = VisionTransformer(
        img_size=224,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        drop_path_rate=0.1,
        pre_norm=False,
        num_register_tokens=num_registers,
    )

    # Probe student state for the backbone prefix.
    student = checkpoint['student']
    backbone_state = {}
    used_prefix = None
    for prefix in ('module.backbone.', 'backbone.'):
        candidate = {
            k[len(prefix):]: v for k, v in student.items() if k.startswith(prefix)
        }
        if candidate:
            backbone_state = candidate
            used_prefix = prefix
            break
    if not backbone_state:
        raise RuntimeError(
            "Could not extract encoder weights from checkpoint['student']: "
            "neither 'module.backbone.' nor 'backbone.' prefix matched any keys."
        )
    print(f"[vitb_backbone] extracted backbone using prefix '{used_prefix}' "
          f"({len(backbone_state)} tensors)")

    missing, unexpected = encoder.load_state_dict(backbone_state, strict=False)
    print(f"[vitb_backbone] load_state_dict: "
          f"{len(missing)} missing, {len(unexpected)} unexpected keys")
    if missing:
        print(f"[vitb_backbone] missing (first 5): {list(missing)[:5]}")
    if unexpected:
        print(f"[vitb_backbone] unexpected (first 5): {list(unexpected)[:5]}")

    encoder.to(device)
    return encoder, patch_size, num_registers, embed_dim
