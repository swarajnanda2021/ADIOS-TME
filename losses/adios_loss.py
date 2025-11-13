"""
ADIOS Loss for TME training.
Contrastive loss between original and masked embeddings with sparsity regularization.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class ADIOSLoss(nn.Module):
    """
    ADIOS loss with contrastive learning and sparsity regularization.
    
    Args:
        alpha_sparsity: Weight for sparsity penalty
        img_size: Image size for computing sparsity
        initial_temp: Initial temperature
        final_temp: Final temperature
        total_iters: Total training iterations
    """
    def __init__(
        self, 
        alpha_sparsity=0.1, 
        img_size=224,
        initial_temp=0.2, 
        final_temp=0.05, 
        total_iters=300000
    ):
        super().__init__()
        self.alpha_sparsity = alpha_sparsity
        self.img_size = img_size
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.total_iters = total_iters

    def get_temperature(self, iteration):
        """Cosine temperature schedule."""
        if iteration >= self.total_iters:
            return self.final_temp
        progress = iteration / self.total_iters
        return self.final_temp + 0.5 * (self.initial_temp - self.final_temp) * \
            (1 + math.cos(math.pi * progress))

    def contrastive_loss(self, original_emb, masked_embs, temperature):
        """
        Compute contrastive loss between original and masked embeddings.
        """
        batch_size = original_emb.shape[0]
        num_masks = len(masked_embs)
        
        # Normalize embeddings
        original_emb = F.normalize(original_emb, p=2, dim=1)
        masked_embs = [F.normalize(emb, p=2, dim=1) for emb in masked_embs]
        
        # Concatenate all embeddings
        all_embeddings = torch.cat([original_emb] + masked_embs, dim=0)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(all_embeddings, all_embeddings.T) / temperature
        
        # Create positive pair mask
        batch_size = original_emb.shape[0]
        positive_mask = torch.zeros(sim_matrix.shape, device=sim_matrix.device)
        
        # Original to each mask is positive
        for m in range(num_masks):
            for i in range(batch_size):
                orig_idx = i
                mask_idx = batch_size * (m + 1) + i
                positive_mask[orig_idx, mask_idx] = 1.0
                positive_mask[mask_idx, orig_idx] = 1.0
        
        # Remove self-similarities
        self_mask = torch.eye(sim_matrix.shape[0], device=sim_matrix.device)
        sim_matrix = sim_matrix * (1 - self_mask) - self_mask * 1e9
        
        # Compute loss
        exp_sim = torch.exp(sim_matrix)
        denominator = exp_sim.sum(dim=1, keepdim=True) - torch.exp(torch.diag(sim_matrix)).unsqueeze(1)
        
        loss = -torch.log(
            (positive_mask * exp_sim).sum(dim=1) / (denominator + 1e-8)
        )
        
        # Only compute loss for original embeddings
        return loss[:batch_size].mean()

    def sparsity_penalty(self, masks):
        """
        Encourage masks to have ~50% activation.
        Using sinh² for smooth gradients.
        """
        penalty = 0
        for i in range(masks.shape[1]):
            h, w = masks[:, i].shape[-2:]
            mean_activation = masks[:, i].sum(dim=(-1, -2)) / (h * w)
            centered_x = mean_activation - 0.5
            penalty += (torch.sinh(torch.abs(centered_x) * math.pi) ** 2).mean()
        return penalty / masks.shape[1]

    def forward(self, original_emb, masked_embs, masks=None, iteration=0, 
                forward_type='student'):
        """
        Args:
            forward_type: 'student' or 'mask' to control behavior
        """
        temperature = self.get_temperature(iteration)
        
        # Contrastive loss
        contrastive = self.contrastive_loss(original_emb, masked_embs, temperature)
        
        metrics = {
            'contrastive': contrastive.item(),
            'temperature': temperature
        }
        
        if forward_type == 'student':
            # Student: minimize contrastive loss (wants similarity)
            total_loss = contrastive
            
        elif forward_type == 'mask':
            # Mask: MAXIMIZE contrastive loss (adversarial)
            sparsity = self.sparsity_penalty(masks)
            total_loss = -contrastive + self.alpha_sparsity * sparsity
            metrics['sparsity'] = sparsity.item()
            metrics['adversarial_loss'] = (-contrastive).item()
        
        return total_loss, metrics