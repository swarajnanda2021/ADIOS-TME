"""
STEGO correspondence distillation loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class STEGOLoss(nn.Module):
    """
    STEGO correspondence loss with self, KNN, and random image pairs.
    
    From STEGO paper (Hamilton et al., ICLR 2022):
    "We encourage features to form compact clusters while preserving 
    their relationships across the corpora."
    
    Args:
        lambda_self: Weight for self-correspondence (within image)
        lambda_knn: Weight for KNN correspondence (similar images)
        lambda_rand: Weight for random correspondence (dissimilar images)
        b_self: Threshold for self-correspondence
        b_knn: Threshold for KNN correspondence
        b_rand: Threshold for random correspondence
    """
    def __init__(
        self,
        lambda_self=1.0,
        lambda_knn=0.5,
        lambda_rand=1.0,
        b_self=0.12,
        b_knn=0.20,
        b_rand=1.0,
    ):
        super().__init__()
        self.lambda_self = lambda_self
        self.lambda_knn = lambda_knn
        self.lambda_rand = lambda_rand
        self.b_self = b_self
        self.b_knn = b_knn
        self.b_rand = b_rand
    
    def compute_correspondence(self, features1, features2):
        """
        Compute cosine similarity correspondence tensor.
        
        Args:
            features1: [B, C, H, W] first feature map
            features2: [B, C, H, W] second feature map
            
        Returns:
            correspondence: [B, H*W, H*W] cosine similarity matrix
        """
        B, C, H, W = features1.shape
        
        # Flatten spatial dimensions [B, C, H*W]
        f1 = features1.reshape(B, C, H * W)
        f2 = features2.reshape(B, C, H * W)
        
        # L2 normalize
        f1_norm = F.normalize(f1, dim=1, p=2)  # [B, C, H*W]
        f2_norm = F.normalize(f2, dim=1, p=2)  # [B, C, H*W]
        
        # Compute cosine similarity: [B, H*W, H*W]
        correspondence = torch.bmm(
            f1_norm.transpose(1, 2),  # [B, H*W, C]
            f2_norm                    # [B, C, H*W]
        )
        
        return correspondence
    
    def spatial_centering(self, correspondence):
        """
        Apply spatial centering to correspondence tensor (Equation 3).
        
        For each spatial position, subtract the mean similarity to all positions.
        This removes global bias and emphasizes relative similarities.
        
        Args:
            correspondence: [B, N, N] where N = H*W
            
        Returns:
            centered: [B, N, N] spatially centered correspondences
        """
        # Subtract row-wise mean
        # For each position (row), subtract average similarity to all positions
        mean_sim = correspondence.mean(dim=-1, keepdim=True)  # [B, N, 1]
        centered = correspondence - mean_sim
        
        return centered
    
    def correspondence_loss(self, F, S, b):
        """
        Core STEGO loss (Equation 4).
        
        L = -∑ (F^SC - b) * max(S, 0)
        
        Args:
            F: [B, N, N] feature correspondences (frozen, spatially centered)
            S: [B, N, N] segmentation correspondences (learned)
            b: Threshold hyperparameter
            
        Returns:
            loss: Scalar loss value
        """
        # Clamp S to positive values only
        S_clamped = torch.clamp(S, min=0.0)
        
        # Element-wise multiplication and sum
        loss = -((F - b) * S_clamped).sum()
        
        # Normalize by number of elements
        B, N, _ = F.shape
        loss = loss / (B * N * N)
        
        return loss
    
    def forward(self, features, seg_codes, knn_features=None, knn_seg_codes=None):
        """
        Compute full STEGO loss.
        
        Args:
            features: [B, D, H, W] frozen encoder features
            seg_codes: [B, K, H, W] learned segmentation codes
            knn_features: [B, D, H, W] features from KNN images (optional)
            knn_seg_codes: [B, K, H, W] seg codes from KNN images (optional)
            
        Returns:
            total_loss: Scalar loss
            metrics: Dict of loss components
        """
        B, D, H, W = features.shape
        
        # ========== Self-Correspondence (within same image) ==========
        F_self = self.compute_correspondence(features, features)
        S_self = self.compute_correspondence(seg_codes, seg_codes)
        
        F_self_centered = self.spatial_centering(F_self)
        
        L_self = self.correspondence_loss(F_self_centered, S_self, self.b_self)
        
        # ========== Random Correspondence (between shuffled images) ==========
        # Shuffle images within batch to create random pairs
        indices_rand = torch.randperm(B, device=features.device)
        features_rand = features[indices_rand]
        seg_codes_rand = seg_codes[indices_rand]
        
        F_rand = self.compute_correspondence(features, features_rand)
        S_rand = self.compute_correspondence(seg_codes, seg_codes_rand)
        
        F_rand_centered = self.spatial_centering(F_rand)
        
        L_rand = self.correspondence_loss(F_rand_centered, S_rand, self.b_rand)
        
        # ========== KNN Correspondence (optional, between similar images) ==========
        L_knn = torch.tensor(0.0, device=features.device)
        
        if knn_features is not None and knn_seg_codes is not None:
            F_knn = self.compute_correspondence(features, knn_features)
            S_knn = self.compute_correspondence(seg_codes, knn_seg_codes)
            
            F_knn_centered = self.spatial_centering(F_knn)
            
            L_knn = self.correspondence_loss(F_knn_centered, S_knn, self.b_knn)
        
        # ========== Total Loss ==========
        total_loss = (
            self.lambda_self * L_self +
            self.lambda_knn * L_knn +
            self.lambda_rand * L_rand
        )
        
        metrics = {
            'loss_self': L_self.item(),
            'loss_knn': L_knn.item(),
            'loss_rand': L_rand.item(),
            'total_loss': total_loss.item(),
        }
        
        return total_loss, metrics

