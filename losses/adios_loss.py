"""
ADIOS Loss Implementation - YugeTen Style with Multi-Crop and Distributed Support
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all processes with gradient support."""
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


def gather(x, dim=0):
    """Gathers tensors from all processes, supporting backward propagation."""
    if dist.is_initialized() and dist.get_world_size() > 1:
        return torch.cat(GatherLayer.apply(x), dim=dim)
    return x


class ADIOSLoss(nn.Module):
    """
    ADIOS loss following YugeTen's approach:
    - N separate SimCLR losses (one per mask/crop)
    - Sparsity regularization with inverse_sin or sinh_squared
    - Distributed training support
    
    Args:
        alpha_sparsity: Weight for sparsity penalty
        img_size: Image size for computing sparsity
        temperature: Temperature for contrastive loss (fixed)
        sparsity_penalty_type: Type of sparsity penalty ('inverse_sin' or 'sinh_squared')
    """
    def __init__(
        self, 
        alpha_sparsity=0.1, 
        img_size=224,
        temperature=0.1,
        sparsity_penalty_type='inverse_sin'
    ):
        super().__init__()
        self.alpha_sparsity = alpha_sparsity
        self.img_size = img_size
        self.temperature = temperature
        self.sparsity_penalty_type = sparsity_penalty_type
        
        print(f"ADIOSLoss initialized (YugeTen style):")
        print(f"  - Temperature: {self.temperature} (fixed)")
        print(f"  - Sparsity penalty: {self.sparsity_penalty_type}")
        print(f"  - Alpha sparsity: {self.alpha_sparsity}")

    def simclr_loss_func(self, z1, z2):
        """
        Standard SimCLR loss for a pair of embeddings with distributed support.
        Matches YugeTen's implementation in src/losses/simclr.py
        
        Args:
            z1: Embeddings from view 1 [B, D] (e.g., original)
            z2: Embeddings from view 2 [B, D] (e.g., masked)
            
        Returns:
            SimCLR loss scalar
        """
        device = z1.device
        
        # Normalize before gathering
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        # Gather across GPUs for larger effective batch
        z1 = gather(z1)
        z2 = gather(z2)
        
        b = z1.size(0)
        
        # Concatenate [z1; z2] -> [2B, D]
        z = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix [2B, 2B]
        logits = torch.einsum("if, jf -> ij", z, z) / self.temperature
        
        # Numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        # Positive mask: (i, B+i) and (B+i, i) pairs
        pos_mask = torch.zeros((2 * b, 2 * b), dtype=torch.bool, device=device)
        pos_mask[:b, b:].fill_diagonal_(True)  # z1[i] <-> z2[i]
        pos_mask[b:, :b].fill_diagonal_(True)  # z2[i] <-> z1[i]
        
        # Mask out self-similarity by setting diagonal to -inf (won't contribute to logsumexp)
        logits_masked = logits.clone()
        logits_masked.fill_diagonal_(-float('inf'))

        # Numerically stable logsumexp (handles max-subtraction internally)
        log_sum_exp = torch.logsumexp(logits_masked, dim=1, keepdim=True)

        # Log probabilities: logits - log(sum(exp(logits_masked)))
        log_prob = logits - log_sum_exp
        
        # Mean log probability over positives
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
        
        # Loss (negative log likelihood)
        loss = -mean_log_prob_pos.mean()
        
        return loss

    def sparsity_penalty_inverse_sin(self, masks):
        """
        YugeTen's sparsity penalty: 1 / sin(activation * π)
        Encourages ~50% mask activation.
        """
        penalty = 0
        for i in range(masks.shape[1]):
            h, w = masks[:, i].shape[-2:]
            mean_activation = masks[:, i].sum(dim=(-1, -2)) / (h * w)
            sin_term = torch.sin(mean_activation * math.pi) + 1e-10
            penalty += (1 / sin_term).mean()
        
        return penalty #/ masks.shape[1]

    def sparsity_penalty_sinh_squared(self, masks):
        """
        Alternative sparsity penalty: sinh²(|activation - 0.5| * π)
        """
        penalty = 0
        for i in range(masks.shape[1]):
            h, w = masks[:, i].shape[-2:]
            mean_activation = masks[:, i].sum(dim=(-1, -2)) / (h * w)
            centered_x = mean_activation - 0.5
            penalty += (torch.sinh(torch.abs(centered_x) * math.pi) ** 2).mean()
        
        return penalty #/ masks.shape[1]

    def sparsity_penalty(self, masks):
        """Compute sparsity penalty based on configured type."""
        if self.sparsity_penalty_type == 'inverse_sin':
            return self.sparsity_penalty_inverse_sin(masks)
        elif self.sparsity_penalty_type == 'sinh_squared':
            return self.sparsity_penalty_sinh_squared(masks)
        else:
            raise ValueError(f"Unknown sparsity penalty type: {self.sparsity_penalty_type}")

    def forward(self, original_emb, masked_embs, masks=None, iteration=0, 
                forward_type='student', num_base_masks=3, K=0):
        """
        Forward pass computing N separate SimCLR losses (YugeTen style).
        
        Args:
            original_emb: Original embeddings [B, D]
            masked_embs: List of masked embeddings 
                         Layout: [mask0_full, ..., maskN_full, mask0_crop0, mask0_crop1, ..., maskN_cropK-1]
            masks: Mask tensors [B, num_masks, H, W] (only needed for 'mask' forward type)
            iteration: Current iteration (unused, kept for API compatibility)
            forward_type: 'student' or 'mask'
            num_base_masks: Number of base masks (default: 3)
            K: Number of crops per mask (default: 0)
        
        Returns:
            loss: Total loss value
            metrics: Dictionary of metrics
        """
        similarities = []
        
        # Full-size masked images: indices 0 to num_base_masks-1
        for i in range(num_base_masks):
            loss_i = self.simclr_loss_func(original_emb, masked_embs[i])
            similarities.append(loss_i)
        
        # Cropped masked images: indices num_base_masks onwards
        # Layout: [mask0_crop0, mask0_crop1, ..., mask0_cropK-1, mask1_crop0, ...]
        if K > 0:
            crop_start_idx = num_base_masks
            for m in range(num_base_masks):
                for k in range(K):
                    crop_idx = crop_start_idx + m * K + k
                    if crop_idx < len(masked_embs):
                        loss_crop = self.simclr_loss_func(original_emb, masked_embs[crop_idx])
                        similarities.append(loss_crop)
        
        # Sum all SimCLR losses (YugeTen style)
        total_contrastive = torch.stack(similarities).sum()
        
        metrics = {
            'similarity': total_contrastive.item(),
            'temperature': self.temperature,
            'num_pairs': len(similarities)
        }
        
        total_loss = total_contrastive
        
        # Add sparsity penalty only for mask model training
        if forward_type == 'mask' and masks is not None:
            sparsity = self.sparsity_penalty(masks)
            # Mask model objective: maximize contrastive (adversarial) + sparsity penalty
            # YugeTen does: manual_backward(-m_loss) where m_loss includes similarity
            # Equivalent: return -contrastive + alpha * sparsity
            total_loss = -total_contrastive + self.alpha_sparsity * sparsity
            metrics['sparsity'] = sparsity.item()
            metrics['adversarial_loss'] = (-total_contrastive).item()
        
        return total_loss, metrics