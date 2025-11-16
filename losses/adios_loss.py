"""
COMPLETE ADIOS Loss Implementation with Multi-Crop Support and Configurable Sparsity Penalties
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


class ADIOSLoss(nn.Module):
    """
    ADIOS loss with contrastive learning and configurable sparsity regularization.
    Supports multi-crop and distributed training.
    
    Args:
        alpha_sparsity: Weight for sparsity penalty
        img_size: Image size for computing sparsity
        temperature: Temperature for contrastive loss (fixed, no schedule)
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
        
        print(f"ADIOSLoss initialized with:")
        print(f"  - Temperature: {self.temperature} (fixed)")
        print(f"  - Sparsity penalty: {self.sparsity_penalty_type}")
        print(f"  - Alpha sparsity: {self.alpha_sparsity}")

    def multi_mask_contrastive_loss_with_crops(
        self, 
        masked_embeddings, 
        original_embeddings, 
        num_base_masks=3, 
        K=0
    ):
        """
        Modified contrastive loss that handles both full-size and cropped masks.
        
        Args:
            masked_embeddings: List of embeddings [mask1_full, ..., mask3_full, 
                                                   mask1_crop1, ..., mask3_cropK]
            original_embeddings: Original image embeddings [B, D]
            num_base_masks: Number of base masks (3 in your case)
            K: Number of crops per mask (0 if no crops)
        """
        device = original_embeddings.device
        batch_size = original_embeddings.shape[0]
        
        # Total masked embeddings = full masks + cropped masks
        num_full_masks = num_base_masks
        num_crop_masks = num_base_masks * K if K > 0 else 0
        total_masked = num_full_masks + num_crop_masks
        
        # Verify we have the right number of embeddings
        expected_views = total_masked
        actual_views = len(masked_embeddings)
        if actual_views != expected_views:
            print(f"WARNING: Expected {expected_views} masked views but got {actual_views}")
            print(f"  num_base_masks={num_base_masks}, K={K}")
            total_masked = actual_views
        
        # Normalize all embeddings
        original_embeddings = F.normalize(original_embeddings, p=2, dim=1)
        masked_embeddings = [F.normalize(emb, p=2, dim=1) for emb in masked_embeddings]
        
        # Concatenate: [original, full_masks, cropped_masks]
        all_embeddings = torch.cat([original_embeddings] + masked_embeddings, dim=0)
        
        # Gather from all GPUs (critical for distributed training!)
        if dist.is_initialized() and dist.get_world_size() > 1:
            all_embeddings_gathered = torch.cat(GatherLayer.apply(all_embeddings), dim=0)
            world_size = dist.get_world_size()
        else:
            # Single GPU case
            all_embeddings_gathered = all_embeddings
            world_size = 1
        
        total_batch_size = batch_size * world_size
        total_size = (1 + total_masked) * total_batch_size
        
        # Create positive mask matrix
        positive_mask = torch.zeros(total_size, total_size, device=device)
        
        # For each GPU
        for gpu in range(world_size):
            # Calculate offset for this GPU's embeddings
            gpu_offset = gpu * (1 + total_masked) * batch_size
            
            for i in range(batch_size):
                # Original image index for this sample
                orig_idx = gpu_offset + i
                
                # Full mask indices
                for m in range(num_full_masks):
                    mask_idx = gpu_offset + batch_size * (1 + m) + i
                    if mask_idx < total_size:
                        positive_mask[orig_idx, mask_idx] = 1.0
                        positive_mask[mask_idx, orig_idx] = 1.0
                
                # Cropped mask indices
                if K > 0:
                    for m in range(num_base_masks):
                        for k in range(K):
                            crop_idx = gpu_offset + batch_size * (1 + num_full_masks + m * K + k) + i
                            if crop_idx < total_size:
                                positive_mask[orig_idx, crop_idx] = 1.0
                                positive_mask[crop_idx, orig_idx] = 1.0
                    
                    # Optional: Add connections between full mask and its crops
                    for m in range(num_base_masks):
                        full_mask_idx = gpu_offset + batch_size * (1 + m) + i
                        for k in range(K):
                            crop_idx = gpu_offset + batch_size * (1 + num_full_masks + m * K + k) + i
                            if crop_idx < total_size and full_mask_idx < total_size:
                                positive_mask[full_mask_idx, crop_idx] = 1.0
                                positive_mask[crop_idx, full_mask_idx] = 1.0
        
        # Compute similarity matrix with fixed temperature
        sim_matrix = torch.matmul(all_embeddings_gathered, all_embeddings_gathered.T) / self.temperature
        
        # Numerical stability
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()
        
        # Compute exp similarities
        exp_sim = torch.exp(sim_matrix)
        self_mask = torch.eye(total_size, device=device)
        exp_sim = exp_sim * (1 - self_mask)
        
        # Compute log probabilities
        denominator = exp_sim.sum(dim=1, keepdim=True)
        log_prob = sim_matrix - torch.log(denominator + 1e-7)
        
        # Compute loss with positive pairs
        positive_pairs_per_row = positive_mask.sum(dim=1)
        mean_log_prob = (positive_mask * log_prob).sum(dim=1) / positive_pairs_per_row.clamp(min=1e-7)
        
        # Extract loss only for original embeddings
        original_indices = []
        for gpu in range(world_size):
            gpu_offset = gpu * (1 + total_masked) * batch_size
            for i in range(batch_size):
                original_indices.append(gpu_offset + i)
        
        original_indices = torch.tensor(original_indices, device=device)
        original_rows = mean_log_prob[original_indices]
        
        return -original_rows.mean()

    def sparsity_penalty_inverse_sin(self, masks):
        """
        YugeTen's sparsity penalty: 1 / sin(activation * π)
        
        Encourages masks to have ~50% activation.
        Very steep penalty near 0 and 1, gentle near 0.5.
        
        Args:
            masks: [B, num_masks, H, W]
        """
        penalty = 0
        for i in range(masks.shape[1]):
            h, w = masks[:, i].shape[-2:]
            mean_activation = masks[:, i].sum(dim=(-1, -2)) / (h * w)
            
            # Add small epsilon to prevent division by zero
            sin_term = torch.sin(mean_activation * math.pi) + 1e-10
            penalty += (1 / sin_term).mean()
        
        return penalty / masks.shape[1]

    def sparsity_penalty_sinh_squared(self, masks):
        """
        Your sparsity penalty: sinh²(|activation - 0.5| * π)
        
        Encourages masks to have ~50% activation.
        Smoother near 0.5, explodes at extremes.
        
        Args:
            masks: [B, num_masks, H, W]
        """
        penalty = 0
        for i in range(masks.shape[1]):
            h, w = masks[:, i].shape[-2:]
            mean_activation = masks[:, i].sum(dim=(-1, -2)) / (h * w)
            
            centered_x = mean_activation - 0.5
            penalty += (torch.sinh(torch.abs(centered_x) * math.pi) ** 2).mean()
        
        return penalty / masks.shape[1]

    def sparsity_penalty(self, masks):
        """
        Compute sparsity penalty based on configured type.
        
        Args:
            masks: [B, num_masks, H, W]
        
        Returns:
            Sparsity penalty scalar
        """
        if self.sparsity_penalty_type == 'inverse_sin':
            return self.sparsity_penalty_inverse_sin(masks)
        elif self.sparsity_penalty_type == 'sinh_squared':
            return self.sparsity_penalty_sinh_squared(masks)
        else:
            raise ValueError(f"Unknown sparsity penalty type: {self.sparsity_penalty_type}")

    def forward(self, original_emb, masked_embs, masks=None, iteration=0, 
                forward_type='student', num_base_masks=3, K=0):
        """
        Forward pass with support for multi-crop.
        
        Args:
            original_emb: Original embeddings [B, D]
            masked_embs: List of masked embeddings (full + crops)
            masks: Mask tensors (only needed for 'mask' forward type)
            iteration: Current iteration (kept for API compatibility, not used)
            forward_type: 'student' or 'mask'
            num_base_masks: Number of base masks (default: 3)
            K: Number of crops per mask (default: 0)
        
        Returns:
            loss: Total loss value
            metrics: Dictionary of metrics
        """
        # Use fixed temperature (no schedule)
        contrastive = self.multi_mask_contrastive_loss_with_crops(
            masked_embs, 
            original_emb,
            num_base_masks=num_base_masks,
            K=K
        )
        
        metrics = {
            'similarity': contrastive.item(),
            'temperature': self.temperature
        }
        
        total_loss = contrastive
        
        # Add sparsity penalty only for mask model training
        if forward_type == 'mask' and masks is not None:
            sparsity = self.sparsity_penalty(masks)
            # Mask model: maximize contrastive loss (adversarial) + sparsity penalty
            total_loss = -contrastive + self.alpha_sparsity * sparsity
            metrics['sparsity'] = sparsity.item()
            metrics['adversarial_loss'] = (-contrastive).item()
        
        return total_loss, metrics