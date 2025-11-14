"""
Feature correspondence loss for semantic grounding.
Aligns mask-selected features with template features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureCorrespondenceLoss(nn.Module):
    """
    Semantic grounding via feature alignment.
    
    Args:
        temperature: Temperature for contrastive loss
        top_k: Number of top patches to select per mask
        diversity_weight: Weight for inter-mask diversity
        sparsity_weight: Weight for sparsity regularization
    """
    def __init__(
        self,
        temperature=0.07,
        top_k=20,
        diversity_weight=0.1,
        sparsity_weight=0.1,
    ):
        super().__init__()
        self.temperature = temperature
        self.top_k = top_k
        self.diversity_weight = diversity_weight
        self.sparsity_weight = sparsity_weight
    
    def select_top_k_patches(self, features, mask):
        """
        Select top-k patches by mask value.
        
        Args:
            features: [B, D, H, W] feature maps
            mask: [B, H, W] mask values
            
        Returns:
            selected: [B*K, D] selected patch features
            weights: [B*K] selection weights (mask values)
        """
        B, D, H, W = features.shape
        
        # Flatten spatial
        features_flat = features.reshape(B, D, H*W).permute(0, 2, 1)  # [B, HW, D]
        mask_flat = mask.reshape(B, H*W)  # [B, HW]
        
        # Get top-k indices
        topk_values, topk_indices = torch.topk(
            mask_flat, 
            k=min(self.top_k, H*W), 
            dim=-1
        )  # [B, K]
        
        # Gather features
        selected_features = torch.gather(
            features_flat,
            dim=1,
            index=topk_indices.unsqueeze(-1).expand(-1, -1, D)
        )  # [B, K, D]
        
        # Flatten batch dimension
        selected_features = selected_features.reshape(B * self.top_k, D)
        topk_values = topk_values.reshape(B * self.top_k)
        
        return selected_features, topk_values
    
    def contrastive_alignment(self, selected_features, weights,
                             positive_bank, negative_bank):
        """
        Contrastive loss: selected features should match positives, not negatives.
        
        Args:
            selected_features: [M, D] mask-selected features
            weights: [M] selection weights
            positive_bank: [N_pos, D] template features (e.g., nuclei)
            negative_bank: [N_neg, D] template features (e.g., background)
            
        Returns:
            loss: Weighted contrastive loss
        """
        # Normalize
        selected_norm = F.normalize(selected_features, dim=-1)
        pos_norm = F.normalize(positive_bank, dim=-1)
        neg_norm = F.normalize(negative_bank, dim=-1)
        
        # Compute similarities
        pos_sim = torch.matmul(selected_norm, pos_norm.T) / self.temperature  # [M, N_pos]
        neg_sim = torch.matmul(selected_norm, neg_norm.T) / self.temperature  # [M, N_neg]
        
        # For each selected patch, find max similarity to positives
        max_pos_sim = pos_sim.max(dim=-1)[0]  # [M]
        
        # Combine with negatives for contrastive learning
        logits = torch.cat([pos_sim, neg_sim], dim=-1)  # [M, N_pos + N_neg]
        
        # Soft labels: highest weight on best positive match
        labels = pos_sim.argmax(dim=-1)  # [M]
        
        # Weighted cross-entropy (weight by mask confidence)
        loss = F.cross_entropy(logits, labels, reduction='none')
        weighted_loss = (loss * weights).sum() / (weights.sum() + 1e-8)
        
        return weighted_loss
    
    def diversity_loss(self, mask1, mask2):
        """
        Encourage different masks to focus on different regions.
        
        Args:
            mask1: [B, H, W]
            mask2: [B, H, W]
            
        Returns:
            loss: Negative correlation (minimize overlap)
        """
        # Flatten
        m1 = mask1.reshape(mask1.shape[0], -1)
        m2 = mask2.reshape(mask2.shape[0], -1)
        
        # Pearson correlation
        m1_centered = m1 - m1.mean(dim=-1, keepdim=True)
        m2_centered = m2 - m2.mean(dim=-1, keepdim=True)
        
        correlation = (m1_centered * m2_centered).sum(dim=-1) / (
            torch.sqrt((m1_centered**2).sum(dim=-1) * (m2_centered**2).sum(dim=-1)) + 1e-8
        )
        
        # We want LOW correlation (high diversity)
        return correlation.mean()
    
    def sparsity_loss(self, masks):
        """
        Encourage ~50% activation per mask (from ADIOS).
        
        Args:
            masks: [B, num_masks, H, W]
            
        Returns:
            loss: Sparsity penalty
        """
        import math
        
        penalty = 0
        for i in range(masks.shape[1]):
            h, w = masks[:, i].shape[-2:]
            mean_activation = masks[:, i].sum(dim=(-1, -2)) / (h * w)
            centered = mean_activation - 0.5
            penalty += (torch.sinh(torch.abs(centered) * math.pi) ** 2).mean()
        
        return penalty / masks.shape[1]
    
    def forward(self, features, masks, nuclei_bank, background_bank):
        """
        Compute correspondence loss.
        
        Args:
            features: [B, D, H, W] from frozen encoder
            masks: [B, 3, H_mask, W_mask] from mask model
            nuclei_bank: [N_nuclei, D] template nuclei features
            background_bank: [N_bg, D] template background features
            
        Returns:
            loss: Total loss
            metrics: Dict of loss components
        """
        B, D, H, W = features.shape
        
        # Upsample or downsample masks to match feature resolution
        if masks.shape[-2:] != (H, W):
            masks_resized = F.interpolate(
                masks,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
        else:
            masks_resized = masks
        
        # Extract individual masks
        nuclei_mask = masks_resized[:, 0]      # [B, H, W]
        background_mask = masks_resized[:, 1]  # [B, H, W]
        
        # Select patches
        nuclei_features, nuclei_weights = self.select_top_k_patches(
            features, nuclei_mask
        )
        bg_features, bg_weights = self.select_top_k_patches(
            features, background_mask
        )
        
        # Alignment losses (contrastive)
        L_nuclei = self.contrastive_alignment(
            nuclei_features, nuclei_weights,
            nuclei_bank, background_bank
        )
        
        L_background = self.contrastive_alignment(
            bg_features, bg_weights,
            background_bank, nuclei_bank
        )
        
        # Total loss
        total_loss = (
            L_nuclei +
            L_background
        )
        
        # Metrics
        metrics = {
            'nuclei_alignment': L_nuclei.item(),
            'background_alignment': L_background.item(),
        }
        
        return total_loss, metrics