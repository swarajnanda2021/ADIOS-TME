"""ViT-B + three-branch CellViT + optional ADIOS prior (training-only).

Inference graph
---------------
ViT-B encoder (loaded via ``load_vitb_encoder``) feeds four intermediate
levels into a three-branch CellViT (NP / HV / NC) reused from the existing
post-PHASE-D ``cellvit/models.py`` patch. The NP branch here comes from
CellViT itself (2-channel softmax → channel 0), NOT from the ADIOS mask
model. So this is structurally different from Path Z's ``ADIOSCellViT``.

Training graph
--------------
Optionally, ``set_adios_prior(mask_model, selector)`` attaches a frozen
ADIOS mask model + selector. When the model is in ``train()`` mode and a
prior is attached, ``forward`` also returns ``adios_fg``/``adios_bg`` for
the consistency BCE term in ``train_vitb.py``'s loss.

The prior is intentionally NOT registered as an ``nn.Module`` submodule
(``SimpleNamespace`` wrapper, plain attribute) so its parameters do not
appear in ``self.state_dict()`` and ``self.to(device)`` does not touch them.
The caller is responsible for placing the prior models on the right device
before ``set_adios_prior``.
"""

from types import SimpleNamespace
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cellvit.models import CellViT


class ViTBCellViT(nn.Module):
    """ViT-B + three-branch CellViT, optional frozen ADIOS prior for training."""

    def __init__(
        self,
        encoder: nn.Module,
        patch_size: int,
        num_registers: int,
        encoder_dim: int = 768,
        num_classes: int = 6,
        drop_rate: float = 0.1,
        image_size: int = 224,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_registers = num_registers
        self.image_size = image_size

        # Three-branch CellViT (NP, HV, NC) from the assemble_cluster.sh
        # PHASE D patch.
        self.cellvit = CellViT(
            encoder=encoder,
            encoder_dim=encoder_dim,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )

        # CellViT.__init__ freezes the encoder. Path B-2 wants the encoder
        # trainable at a low LR; the training script will build the param
        # groups. Undo the freeze here.
        for p in self.cellvit.encoder.parameters():
            p.requires_grad = True

    # ---- prior attachment --------------------------------------------------

    def set_adios_prior(self, mask_model: nn.Module, selector: nn.Module) -> None:
        """Attach a frozen ADIOS prior for the training consistency loss.

        Wrapped in ``SimpleNamespace`` so the prior models bypass
        ``nn.Module.__setattr__``'s submodule registration. Their parameters
        won't appear in ``self.state_dict()`` and ``self.to(device)`` won't
        move them — the caller must place them on the right device.
        """
        for p in mask_model.parameters():
            p.requires_grad = False
        for p in selector.parameters():
            p.requires_grad = False
        mask_model.eval()
        selector.eval()
        self.adios_prior = SimpleNamespace(mask_model=mask_model, selector=selector)

    @property
    def has_adios_prior(self) -> bool:
        return getattr(self, 'adios_prior', None) is not None

    # ---- forward -----------------------------------------------------------

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        # ViT features at 1/4, 1/2, 3/4, full depth. Layout per token dim:
        # [cls, register_tokens..., patches...]  →  1 + num_registers + N
        features = self.cellvit.encoder.get_intermediate_layers(images)
        f1, f2, f3, f4 = features

        nr = self.num_registers
        num_patches = f1.shape[1] - (nr + 1)
        feature_size = int(np.sqrt(num_patches))

        def _reshape(f: torch.Tensor) -> torch.Tensor:
            return (
                f[:, (nr + 1):, :]
                .permute(0, 2, 1)
                .reshape(f.shape[0], -1, feature_size, feature_size)
                .contiguous()
            )

        f1, f2, f3, f4 = (_reshape(f) for f in (f1, f2, f3, f4))

        # Three CellViT branches.
        np_logits = self.cellvit._forward_upsample(
            images, f1, f2, f3, f4, self.cellvit.nuclei_binary_map_decoder
        )
        hv = self.cellvit._forward_upsample(
            images, f1, f2, f3, f4, self.cellvit.hv_map_decoder
        )
        nc = self.cellvit._forward_upsample(
            images, f1, f2, f3, f4, self.cellvit.nuclei_type_map_decoder
        )

        # Native decoder output is ``feature_size * 16``: 224 for patch_size=16,
        # 256 for patch_size=14. Interpolate to image_size unconditionally so
        # downstream loss + eval code stays size-agnostic.
        if np_logits.shape[-1] != self.image_size:
            np_logits = F.interpolate(
                np_logits, size=self.image_size, mode='bilinear', align_corners=False,
            )
            hv = F.interpolate(
                hv, size=self.image_size, mode='bilinear', align_corners=False,
            )
            nc = F.interpolate(
                nc, size=self.image_size, mode='bilinear', align_corners=False,
            )

        # NP is 2-channel logits → softmax → channel 0 (foreground probability).
        pred_np = F.softmax(np_logits, dim=1)[:, 0:1, :, :]

        out: Dict[str, torch.Tensor] = {
            'masks': pred_np,
            'distances': hv,
            'nuclei_types': nc,
        }

        # Optional ADIOS prior outputs — training only.
        if self.training and self.has_adios_prior:
            adios_fg, adios_bg = self._adios_prior_targets(images)
            out['adios_fg'] = adios_fg
            out['adios_bg'] = adios_bg

        return out

    # ---- prior plumbing ----------------------------------------------------

    def _adios_prior_targets(
        self, images: torch.Tensor
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Run the frozen ADIOS prior; return (adios_fg, adios_bg) at image_size.

        ``adios_fg`` is the selector-chosen channel (per-sample argmax).
        ``adios_bg`` is the per-pixel max over the two non-chosen channels.
        Both shaped ``[B, 1, image_size, image_size]``.
        """
        with torch.no_grad():
            mask_output = self.adios_prior.mask_model(images)['masks']
            channel_logits = self.adios_prior.selector(mask_output)
            B, N, H, W = mask_output.shape
            chosen = channel_logits.argmax(dim=1)                          # [B]

            # adios_fg: gather chosen channel.
            idx = chosen.view(B, 1, 1, 1).expand(B, 1, H, W)
            adios_fg = torch.gather(mask_output, dim=1, index=idx)         # [B, 1, H, W]

            # adios_bg: max over the two non-chosen channels.
            chosen_mask = torch.zeros(B, N, dtype=torch.bool, device=mask_output.device)
            chosen_mask.scatter_(1, chosen.unsqueeze(1), True)
            non_chosen = mask_output.masked_fill(
                chosen_mask.view(B, N, 1, 1), float('-inf')
            )
            adios_bg = non_chosen.max(dim=1, keepdim=True).values          # [B, 1, H, W]

        if adios_fg.shape[-1] != self.image_size:
            adios_fg = F.interpolate(
                adios_fg, size=self.image_size, mode='bilinear', align_corners=False,
            )
            adios_bg = F.interpolate(
                adios_bg, size=self.image_size, mode='bilinear', align_corners=False,
            )
        return adios_fg, adios_bg
