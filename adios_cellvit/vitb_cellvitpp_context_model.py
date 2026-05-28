"""Stream A: ViT-B + NP/HV CellViT branches + cell-context-attention classifier.

Subclasses ``ViTBCellViTPP`` (Path B-3) and changes exactly one thing: the
per-cell NC head. B-3's ``CellTokenClassifier`` (per-cell pooled-token MLP)
is replaced by ``CellContextClassifier`` (a transformer that lets cells in
the same image attend to each other, with a 2D positional encoding from
cell centroids). Everything else — encoder, NP/HV decoders, pooling, the
ADIOS-prior plumbing, the forward() contract — is inherited unchanged, so
this is a clean single-variable comparison against B-3.

Why a subclass and not an in-place edit of ``vitb_cellvitpp_model.py``:
keeps B-3 fully runnable on the same branch and gives this experiment a
distinct class name in the saved checkpoint config for traceability.

The override of ``classify_cells_from_features`` is the B-3 pooling loop
plus one addition: it computes each surviving cell's normalized centroid
and feeds (embeddings, centroids, batch_idx) to the context classifier.
The returned 3-tuple ``(cell_logits, cell_batch_idx, cell_inst_id)`` is
identical in shape and ordering to B-3, so the inherited ``forward``, the
training loss, and the eval second pass need no changes.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from adios_cellvit.cell_context_attention import CellContextClassifier
from adios_cellvit.vitb_cellvitpp_model import ViTBCellViTPP


class ViTBCellViTPPContext(ViTBCellViTPP):
    """B-3 architecture with the per-cell MLP swapped for cell-context attention."""

    def __init__(
        self,
        encoder: nn.Module,
        patch_size: int,
        num_registers: int,
        encoder_dim: int = 768,
        num_cell_classes: int = 5,
        classifier_hidden_dim: int = 384,
        classifier_dropout: float = 0.1,
        context_num_layers: int = 2,
        context_num_heads: int = 8,
        context_dim_feedforward: int = 2048,
        drop_rate: float = 0.1,
        image_size: int = 224,
    ):
        super().__init__(
            encoder=encoder,
            patch_size=patch_size,
            num_registers=num_registers,
            encoder_dim=encoder_dim,
            num_cell_classes=num_cell_classes,
            classifier_hidden_dim=classifier_hidden_dim,
            classifier_dropout=classifier_dropout,
            drop_rate=drop_rate,
            image_size=image_size,
        )
        # Reassigning the registered submodule replaces the B-3 MLP built by
        # super().__init__() in the module tree (nn.Module.__setattr__ drops
        # the old one), so no dead parameters are left behind.
        self.cell_classifier = CellContextClassifier(
            embed_dim=encoder_dim,
            num_classes=num_cell_classes,
            num_layers=context_num_layers,
            num_heads=context_num_heads,
            dim_feedforward=context_dim_feedforward,
            hidden_dim=classifier_hidden_dim,
            dropout=classifier_dropout,
        )

    def classify_cells_from_features(
        self,
        last_tokens: torch.Tensor,    # [B, 1 + nr + N, D]
        instance_mask: torch.Tensor,  # [B, 1, H, W] long  (foreground IDs > 0, bg = 0)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """B-3 pooling + per-cell centroid, then cell-context classification.

        Same return contract as the parent: ``(cell_logits [N_cells, K],
        cell_batch_idx [N_cells], cell_inst_id [N_cells])`` with rows in the
        order cells were pooled. Sub-patch cells (no touched patch) are skipped.
        """
        B = last_tokens.shape[0]
        nr = self.num_registers
        patch_tokens = last_tokens[:, (nr + 1):, :]   # [B, N, D]

        inst = instance_mask.squeeze(1).long()        # [B, H, W]
        H, W = inst.shape[1], inst.shape[2]
        ps = self.patch_size

        embeddings = []
        centroids = []
        batch_idx_list = []
        inst_id_list = []

        for b in range(B):
            uniq = inst[b].unique()
            uniq = uniq[uniq > 0]
            if uniq.numel() == 0:
                continue
            for k in uniq:
                pix = (inst[b] == k)                                  # [H, W] bool
                pix_mask = pix.float().unsqueeze(0).unsqueeze(0)      # [1, 1, H, W]
                patch_mask = F.max_pool2d(pix_mask, kernel_size=ps, stride=ps)
                patch_flat = patch_mask.view(-1) > 0                 # [N] bool
                if not patch_flat.any():
                    # Sub-patch nucleus: skip (matches B-3).
                    continue
                emb = patch_tokens[b, patch_flat, :].mean(dim=0)     # [D]
                ys, xs = torch.nonzero(pix, as_tuple=True)
                cy = ys.float().mean() / H
                cx = xs.float().mean() / W
                embeddings.append(emb)
                centroids.append(torch.stack([cy, cx]))
                batch_idx_list.append(b)
                inst_id_list.append(int(k.item()))

        if not embeddings:
            empty_logits = torch.empty(
                0, self.num_cell_classes,
                device=last_tokens.device, dtype=last_tokens.dtype,
            )
            empty_idx = torch.empty(0, dtype=torch.long, device=last_tokens.device)
            return empty_logits, empty_idx, empty_idx.clone()

        cell_embeddings = torch.stack(embeddings, dim=0)             # [N_cells, D]
        cell_centroids = torch.stack(centroids, dim=0).to(cell_embeddings.dtype)  # [N_cells, 2]
        cell_batch_idx = torch.tensor(
            batch_idx_list, dtype=torch.long, device=last_tokens.device,
        )
        cell_inst_id = torch.tensor(
            inst_id_list, dtype=torch.long, device=last_tokens.device,
        )
        cell_logits = self.cell_classifier(
            cell_embeddings, cell_centroids, cell_batch_idx,
        )                                                            # [N_cells, K]
        return cell_logits, cell_batch_idx, cell_inst_id
