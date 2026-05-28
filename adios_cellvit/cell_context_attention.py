"""Cell-context attention classifier (Path B-3 → Stream A).

The B-3 per-cell MLP (``CellTokenClassifier``) classifies each cell from
its own pooled token alone, discarding inter-cell spatial context. The
B-3 ablation showed this loses to the B-2 per-pixel decoder specifically
on **connective** (recall 0.646 → 0.539) — the class whose identity most
depends on neighbours (near vessels, in stroma). See HANDOFF_pathB3_results.md.

``CellContextClassifier`` lets cells *see each other*: pooled cell tokens
become a length-``N`` sequence (one sequence per source image), a 2D
sinusoidal positional encoding derived from each cell's centroid is added,
and a small ``nn.TransformerEncoder`` mixes information across cells before
a per-cell MLP head emits the class logits.

Contract
--------
``forward(cell_embeddings, cell_centroids, cell_batch_idx)`` returns
``[N_cells, num_classes]`` logits in the **same row order** as the inputs,
so the caller (``ViTBCellViTPPContext.classify_cells_from_features``) and
everything downstream — ``compute_cell_labels``, the per-cell CE loss, the
eval second pass — are byte-for-byte unchanged vs B-3. Attention is scoped
*within* an image via ``cell_batch_idx``; cells from different images in a
batch never attend to one another.
"""

import math

import torch
import torch.nn as nn


def sinusoidal_2d_pos_enc(
    centroids: torch.Tensor,
    dim: int,
    temperature: float = 10000.0,
    scale: float = 2.0 * math.pi,
) -> torch.Tensor:
    """DETR-style fixed 2D sin-cos positional encoding from (y, x) centroids.

    Args:
        centroids: [N, 2] float, normalized (y, x) in [0, 1].
        dim:       output channels. Must be divisible by 4 (split y/x, sin/cos).

    Returns:
        [N, dim] positional encoding, parameter-free (no overfitting on the
        ~20-cell sequences PanNuke produces).
    """
    if dim % 4 != 0:
        raise ValueError(f"sinusoidal_2d_pos_enc: dim must be divisible by 4, got {dim}")
    half = dim // 2                      # channels for y, channels for x
    num_freqs = half // 2                # sin+cos pair per frequency
    device = centroids.device

    freq_idx = torch.arange(num_freqs, device=device, dtype=torch.float32)
    dim_t = temperature ** (2.0 * freq_idx / num_freqs)   # [num_freqs]

    y = centroids[:, 0:1] * scale        # [N, 1]
    x = centroids[:, 1:2] * scale
    py = y / dim_t[None, :]              # [N, num_freqs]
    px = x / dim_t[None, :]
    return torch.cat([py.sin(), py.cos(), px.sin(), px.cos()], dim=1)  # [N, dim]


class CellContextClassifier(nn.Module):
    """Transformer over per-cell pooled tokens + 2D centroid pos-enc + MLP head."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 5,
        num_layers: int = 2,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        hidden_dim: int = 384,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        cell_embeddings: torch.Tensor,   # [N, D]
        cell_centroids: torch.Tensor,    # [N, 2] normalized (y, x) in [0, 1]
        cell_batch_idx: torch.Tensor,    # [N] long — source image index per cell
    ) -> torch.Tensor:
        """Returns [N, num_classes] logits in the same row order as inputs."""
        N = cell_embeddings.shape[0]
        if N == 0:
            return cell_embeddings.new_zeros(0, self.num_classes)

        device = cell_embeddings.device
        tokens = cell_embeddings + sinusoidal_2d_pos_enc(cell_centroids, self.embed_dim)

        # Group cells by source image into a padded [G, L_max, D] batch so the
        # transformer attends only within an image. G is small (<= batch_size).
        groups = torch.unique(cell_batch_idx)
        G = int(groups.shape[0])
        sel_per_group = [
            (cell_batch_idx == g).nonzero(as_tuple=False).squeeze(1) for g in groups
        ]
        L_max = max(int(s.numel()) for s in sel_per_group)

        batched = cell_embeddings.new_zeros(G, L_max, self.embed_dim)
        key_padding_mask = torch.ones(G, L_max, dtype=torch.bool, device=device)  # True = pad
        for gi, sel in enumerate(sel_per_group):
            L = int(sel.numel())
            batched[gi, :L] = tokens[sel]
            key_padding_mask[gi, :L] = False

        encoded = self.transformer(batched, src_key_padding_mask=key_padding_mask)  # [G, L_max, D]
        logits_padded = self.head(encoded)                                          # [G, L_max, K]

        # Scatter back to flat [N, K] preserving the caller's row order.
        logits = cell_embeddings.new_zeros(N, self.num_classes)
        for gi, sel in enumerate(sel_per_group):
            L = int(sel.numel())
            logits[sel] = logits_padded[gi, :L]
        return logits
