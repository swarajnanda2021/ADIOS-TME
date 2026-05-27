"""Channel selector: pick the nuclei channel from ADIOS mask outputs.

The ADIOS mask decoder emits ``num_masks`` soft channels; the nuclei channel's
index drifts between images. ``ChannelSelector`` predicts per-image channel
logits, which are turned into selection weights (softmax) at training time and
into a single-channel argmax at deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelSelector(nn.Module):
    """Per-image channel-index classifier over ADIOS mask outputs.

    Input:  ``mask_output`` ``[B, N, H, W]`` — pixel-wise softmax probabilities
            over channels (as emitted by ``MaskModel``).
    Output: ``channel_logits`` ``[B, N]`` — raw logits (no softmax applied).

    Architecture: 3 strided conv blocks → GAP → 2 FC layers (~10K params).
    """

    def __init__(self, num_masks: int = 3):
        super().__init__()
        self.num_masks = num_masks

        # BEEFIER_SELECTOR_V2: deeper, wider stack (~250K params vs ~14K before).
        self.features = nn.Sequential(
            nn.Conv2d(num_masks, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_masks),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, mask_output: torch.Tensor) -> torch.Tensor:
        x = self.features(mask_output)
        x = self.pool(x)
        return self.classifier(x)


def collapse_channels_soft(
    mask_output: torch.Tensor,
    channel_logits: torch.Tensor,
) -> torch.Tensor:
    """Differentiable softmax-weighted sum over channels.

    Used during stage 1 (selector trains end-to-end) and during stage 2 (the
    selector is parameter-frozen but gradients flow through the weighted sum
    into the ADIOS decoder).

    Args:
        mask_output:    ``[B, N, H, W]``.
        channel_logits: ``[B, N]``.

    Returns:
        ``np_map`` ``[B, 1, H, W]``.
    """
    B, N, _, _ = mask_output.shape
    weights = F.softmax(channel_logits, dim=1).view(B, N, 1, 1)
    return (mask_output * weights).sum(dim=1, keepdim=True)


def collapse_channels_argmax(
    mask_output: torch.Tensor,
    channel_logits: torch.Tensor,
) -> torch.Tensor:
    """Non-differentiable single-channel selection by argmax.

    Used only at deployment (``model.set_inference_mode('argmax')``).
    """
    B, _, H, W = mask_output.shape
    chosen = channel_logits.argmax(dim=1)
    idx = chosen.view(B, 1, 1, 1).expand(B, 1, H, W)
    return torch.gather(mask_output, dim=1, index=idx)


def compute_best_channel_target(
    mask_output: torch.Tensor,
    gt_binary: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Stage 1 target: the channel whose thresholded mask best matches GT (IoU).

    Args:
        mask_output: ``[B, N, H, W]`` post-softmax probabilities.
        gt_binary:   ``[B, H, W]`` in {0, 1}.
        threshold:   binarization threshold for ``mask_output``.

    Returns:
        ``LongTensor [B]`` in ``{0, ..., N-1}``.
    """
    gt = gt_binary.unsqueeze(1)
    pred_bin = (mask_output > threshold).float()
    intersection = (pred_bin * gt).sum(dim=(-1, -2))
    union = (pred_bin + gt - pred_bin * gt).sum(dim=(-1, -2)).clamp(min=1e-6)
    iou = intersection / union
    return iou.argmax(dim=1).long()



def compute_soft_channel_target(
    mask_output: torch.Tensor,
    gt_binary: torch.Tensor,
    threshold: float = 0.5,
    fallback_uniform_when_iou_below: float = 0.05,
) -> torch.Tensor:
    """Stage 1 target as a normalized IoU distribution over channels.

    Returns a probability distribution per sample rather than a single index.
    This honestly represents ambiguity: if two channels both have IoU 0.5, the
    target says ``[0.5, 0.5, 0.0]`` rather than committing to one.

    Args:
        mask_output: ``[B, N, H, W]`` post-softmax probabilities.
        gt_binary:   ``[B, H, W]`` in {0, 1}.
        threshold:   binarization threshold for ``mask_output``.
        fallback_uniform_when_iou_below:
            if every channel has IoU below this value, the patch is essentially
            background-only; return a uniform distribution instead of dividing
            by ~0.

    Returns:
        ``FloatTensor [B, N]`` rows summing to 1.
    """
    gt = gt_binary.unsqueeze(1)
    pred_bin = (mask_output > threshold).float()
    intersection = (pred_bin * gt).sum(dim=(-1, -2))
    union = (pred_bin + gt - pred_bin * gt).sum(dim=(-1, -2)).clamp(min=1e-6)
    iou = intersection / union  # [B, N]

    total = iou.sum(dim=1, keepdim=True)
    has_signal = (iou.max(dim=1, keepdim=True).values > fallback_uniform_when_iou_below).float()

    # Where signal exists: normalize IoUs to a distribution.
    # Where signal is absent: use uniform.
    N = iou.shape[1]
    safe_total = total.clamp(min=1e-6)
    normalized = iou / safe_total
    uniform = torch.full_like(iou, 1.0 / N)
    return has_signal * normalized + (1.0 - has_signal) * uniform
