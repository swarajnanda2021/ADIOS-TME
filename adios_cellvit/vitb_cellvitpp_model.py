"""ViT-B + NP/HV CellViT branches + per-cell MLP classifier (CellViT++ style).

Diverges from ``ViTBCellViT`` in one place: the per-pixel NC decoder branch
is replaced by a lightweight MLP that consumes per-cell pooled tokens
from the encoder's last block. The original CellViT NC decoder branch
still exists inside ``self.cellvit`` (created by the PHASE D patch) but
is frozen and never called — keeping the underlying ``CellViT`` class
unmodified so the assemble_cluster.sh pipeline is unchanged.

Token pooling
-------------
For each instance in an input instance mask:
1.  Build a binary [H, W] pixel mask of that instance.
2.  Max-pool with kernel=stride=patch_size → [fs, fs] binary patch-overlap
    mask (a patch is "touched" if any of its pixels belong to the
    instance). For ViT-B/16 @ 224 this is a 14×14 grid.
3.  Average the encoder's last-block patch tokens at the touched patches
    → single [D] cell embedding.

This is the original CellViT paper's pooling, reused in CellViT++ for
its lightweight cell-classification module.

Optional ADIOS prior
--------------------
The same NP-prior consistency hook from ``ViTBCellViT`` is preserved
(adios_fg / adios_bg emitted during training when a prior is attached).
Kept knob-compatible so we can run with-prior and no-prior ablations
from the same training script — see HANDOFF_cellvitpp.md.
"""

from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cellvit.models import CellViT


class CellTokenClassifier(nn.Module):
    """Per-cell MLP. One hidden layer + ReLU + dropout, as in CellViT++."""

    def __init__(self, embed_dim: int = 768, hidden_dim: int = 384,
                 num_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, cell_embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(cell_embeddings)


class ViTBCellViTPP(nn.Module):
    """ViT-B encoder + CellViT NP/HV decoders + per-cell MLP classifier."""

    def __init__(
        self,
        encoder: nn.Module,
        patch_size: int,
        num_registers: int,
        encoder_dim: int = 768,
        num_cell_classes: int = 5,
        classifier_hidden_dim: int = 384,
        classifier_dropout: float = 0.1,
        drop_rate: float = 0.1,
        image_size: int = 224,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_registers = num_registers
        self.image_size = image_size
        self.num_cell_classes = num_cell_classes

        # The post-PHASE-D CellViT exposes nuclei_binary_map_decoder (NP),
        # hv_map_decoder (HV), and nuclei_type_map_decoder (NC). We reuse
        # the first two; the NC branch is frozen and never called from
        # forward(). We keep it in the module tree because removing it
        # would require patching CellViT.__init__, which would drift from
        # the assemble_cluster.sh PHASE D patch and break re-assembly.
        # num_classes here only sizes the unused NC branch; pick a small
        # value (2) to minimize wasted parameters.
        self.cellvit = CellViT(
            encoder=encoder,
            encoder_dim=encoder_dim,
            num_classes=2,
            drop_rate=drop_rate,
        )
        # CellViT.__init__ freezes the encoder; undo (training script
        # decides param groups — encoder typically at a low LR).
        for p in self.cellvit.encoder.parameters():
            p.requires_grad = True
        # Freeze the unused NC decoder (it would otherwise accumulate
        # gradients from any accidental call site, and waste optimizer
        # state if grouped).
        for p in self.cellvit.nuclei_type_map_decoder.parameters():
            p.requires_grad = False
        self.cellvit.nuclei_type_map_decoder.eval()

        # The actual classifier.
        self.cell_classifier = CellTokenClassifier(
            embed_dim=encoder_dim,
            hidden_dim=classifier_hidden_dim,
            num_classes=num_cell_classes,
            dropout=classifier_dropout,
        )

    # ---- ADIOS prior plumbing (identical to ViTBCellViT) -------------------

    def set_adios_prior(self, mask_model: nn.Module, selector: nn.Module) -> None:
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

    # ---- cell token pooling ------------------------------------------------

    def classify_cells_from_features(
        self,
        last_tokens: torch.Tensor,   # [B, 1 + nr + N, D]
        instance_mask: torch.Tensor, # [B, 1, H, W] long  (foreground IDs > 0, bg = 0)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pool last-block tokens per foreground instance and classify.

        Returns:
            cell_logits:    [N_cells, num_cell_classes]
            cell_batch_idx: [N_cells] long   — which batch element each cell belongs to
            cell_inst_id:   [N_cells] long   — instance ID within the source image

        Cells whose foreground pixels don't overlap any patch (sub-patch
        nuclei or numerical edge cases) are skipped. ``N_cells`` can be
        smaller than the number of unique instance IDs across the batch.
        """
        B = last_tokens.shape[0]
        nr = self.num_registers
        patch_tokens = last_tokens[:, (nr + 1):, :]   # [B, N, D]

        inst = instance_mask.squeeze(1).long()        # [B, H, W]
        ps = self.patch_size

        embeddings = []
        batch_idx_list = []
        inst_id_list = []

        for b in range(B):
            uniq = inst[b].unique()
            uniq = uniq[uniq > 0]
            if uniq.numel() == 0:
                continue
            for k in uniq:
                pix_mask = (inst[b] == k).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                patch_mask = F.max_pool2d(pix_mask, kernel_size=ps, stride=ps)
                patch_flat = patch_mask.view(-1) > 0  # [N] bool
                if not patch_flat.any():
                    # Sub-patch nucleus: no patch is touched by any of its
                    # pixels after max-pool. Skip; happens rarely with
                    # patch_size=16 on PanNuke.
                    continue
                emb = patch_tokens[b, patch_flat, :].mean(dim=0)  # [D]
                embeddings.append(emb)
                batch_idx_list.append(b)
                inst_id_list.append(int(k.item()))

        if not embeddings:
            empty_logits = torch.empty(
                0, self.num_cell_classes,
                device=last_tokens.device, dtype=last_tokens.dtype,
            )
            empty_idx = torch.empty(0, dtype=torch.long, device=last_tokens.device)
            return empty_logits, empty_idx, empty_idx.clone()

        cell_embeddings = torch.stack(embeddings, dim=0)    # [N_cells, D]
        cell_logits = self.cell_classifier(cell_embeddings) # [N_cells, K]
        cell_batch_idx = torch.tensor(
            batch_idx_list, dtype=torch.long, device=last_tokens.device,
        )
        cell_inst_id = torch.tensor(
            inst_id_list, dtype=torch.long, device=last_tokens.device,
        )
        return cell_logits, cell_batch_idx, cell_inst_id

    # ---- forward -----------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        instance_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encoder + NP + HV; optionally per-cell classification on instance_mask.

        Always emits ``encoder_tokens`` — the last-block tokens — so the
        eval caller can run watershed first and then re-invoke
        ``classify_cells_from_features`` without redoing the encoder pass.
        """
        # 4 intermediate ViT levels, ordered shallow → deep.
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

        f1_s, f2_s, f3_s, f4_s = (_reshape(f) for f in (f1, f2, f3, f4))

        np_logits = self.cellvit._forward_upsample(
            images, f1_s, f2_s, f3_s, f4_s, self.cellvit.nuclei_binary_map_decoder,
        )
        hv = self.cellvit._forward_upsample(
            images, f1_s, f2_s, f3_s, f4_s, self.cellvit.hv_map_decoder,
        )

        if np_logits.shape[-1] != self.image_size:
            np_logits = F.interpolate(
                np_logits, size=self.image_size, mode='bilinear', align_corners=False,
            )
            hv = F.interpolate(
                hv, size=self.image_size, mode='bilinear', align_corners=False,
            )

        pred_np = F.softmax(np_logits, dim=1)[:, 0:1, :, :]

        out: Dict[str, torch.Tensor] = {
            'masks': pred_np,
            'distances': hv,
            'encoder_tokens': f4,   # [B, 1+nr+N, D] — last block, for cell pooling
        }

        # Per-cell classification (training pass with GT, or eval second
        # pass with watershed-predicted instances).
        if instance_mask is not None:
            cell_logits, cell_batch_idx, cell_inst_id = (
                self.classify_cells_from_features(f4, instance_mask)
            )
            out['cell_logits'] = cell_logits
            out['cell_batch_idx'] = cell_batch_idx
            out['cell_inst_id'] = cell_inst_id

        # ADIOS prior outputs — training only.
        if self.training and self.has_adios_prior:
            adios_fg, adios_bg = self._adios_prior_targets(images)
            out['adios_fg'] = adios_fg
            out['adios_bg'] = adios_bg

        return out

    # ---- ADIOS prior forward (verbatim from ViTBCellViT) -------------------

    def _adios_prior_targets(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            mask_output = self.adios_prior.mask_model(images)['masks']
            channel_logits = self.adios_prior.selector(mask_output)
            B, N, H, W = mask_output.shape
            chosen = channel_logits.argmax(dim=1)

            idx = chosen.view(B, 1, 1, 1).expand(B, 1, H, W)
            adios_fg = torch.gather(mask_output, dim=1, index=idx)

            chosen_mask = torch.zeros(B, N, dtype=torch.bool, device=mask_output.device)
            chosen_mask.scatter_(1, chosen.unsqueeze(1), True)
            non_chosen = mask_output.masked_fill(
                chosen_mask.view(B, N, 1, 1), float('-inf'),
            )
            adios_bg = non_chosen.max(dim=1, keepdim=True).values

        if adios_fg.shape[-1] != self.image_size:
            adios_fg = F.interpolate(
                adios_fg, size=self.image_size, mode='bilinear', align_corners=False,
            )
            adios_bg = F.interpolate(
                adios_bg, size=self.image_size, mode='bilinear', align_corners=False,
            )
        return adios_fg, adios_bg
