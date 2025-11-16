"""
K-Nearest Neighbor mining for STEGO training.

Pre-computes KNN graph for the dataset using frozen encoder features.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm


class KNNIndex:
    """
    Efficient KNN lookup for dataset.
    
    Usage:
        # Build index once
        knn_index = KNNIndex(k=7)
        knn_index.build(data_loader, encoder, save_path='knn_index.pkl')
        
        # During training, lookup KNN for batch
        knn_indices = knn_index.get_knn(batch_indices)
    """
    def __init__(self, k=7):
        self.k = k
        self.features = None
        self.index_to_dataset_idx = None
    
    def build(self, data_loader, encoder, save_path=None, use_five_crop=False):
        """
        Build KNN index by extracting features for entire dataset.
        
        Args:
            data_loader: DataLoader for dataset
            encoder: Frozen encoder model
            save_path: Path to save index (optional)
            use_five_crop: Whether to use 5-crop (like STEGO)
        """
        print("Building KNN index...")
        encoder.eval()
        
        all_features = []
        all_indices = []
        
        with torch.no_grad():
            for idx, images in enumerate(tqdm(data_loader, desc="Extracting features")):
                images = images.cuda()
                
                if use_five_crop:
                    # Extract features from 5 crops
                    crops = self._five_crop(images)
                    for crop in crops:
                        features = self._extract_global_features(encoder, crop)
                        all_features.append(features.cpu())
                        # Track which crops belong to which original image
                        batch_size = crop.shape[0]
                        for i in range(batch_size):
                            all_indices.append(idx * batch_size + i)
                else:
                    # Single forward pass
                    features = self._extract_global_features(encoder, images)
                    all_features.append(features.cpu())
                    batch_size = images.shape[0]
                    for i in range(batch_size):
                        all_indices.append(idx * batch_size + i)
        
        # Stack all features [N, D]
        self.features = torch.cat(all_features, dim=0)
        self.index_to_dataset_idx = np.array(all_indices)
        
        # Normalize for cosine similarity
        self.features = F.normalize(self.features, dim=1, p=2)
        
        print(f"Built KNN index with {self.features.shape[0]} entries")
        
        if save_path:
            self.save(save_path)
    
    def _extract_global_features(self, encoder, images):
        """
        Extract global features (CLS token or GAP).
        
        Args:
            encoder: ViT encoder
            images: [B, 3, H, W]
            
        Returns:
            features: [B, D] global feature vectors
        """
        output = encoder(images)
        
        if isinstance(output, dict):
            # Modern ViT returns dict with 'clstoken'
            features = output['clstoken']
        else:
            # Fallback: assume it's the CLS token
            features = output[:, 0]
        
        return features
    
    def _five_crop(self, images):
        """
        Create 5 crops: center + 4 corners (like STEGO).
        
        Args:
            images: [B, 3, H, W]
            
        Returns:
            crops: List of 5 tensors, each [B, 3, H/2, W/2]
        """
        B, C, H, W = images.shape
        crop_size = (H // 2, W // 2)
        
        # Center crop
        center = images[:, :, H//4:3*H//4, W//4:3*W//4]
        
        # Corner crops
        top_left = images[:, :, :H//2, :W//2]
        top_right = images[:, :, :H//2, W//2:]
        bottom_left = images[:, :, H//2:, :W//2]
        bottom_right = images[:, :, H//2:, W//2:]
        
        return [center, top_left, top_right, bottom_left, bottom_right]
    
    def get_knn(self, query_indices, exclude_self=True):
        """
        Get K-nearest neighbors for query indices.
        
        Args:
            query_indices: [B] dataset indices
            exclude_self: Whether to exclude the query itself from results
            
        Returns:
            knn_indices: [B, K] indices of nearest neighbors
        """
        # Get query features
        query_features = self.features[query_indices]  # [B, D]
        
        # Compute similarities to all features
        similarities = torch.mm(query_features, self.features.T)  # [B, N]
        
        # Get top-K
        if exclude_self:
            # Set self-similarity to -inf
            for i, idx in enumerate(query_indices):
                similarities[i, idx] = -float('inf')
            topk = self.k
        else:
            topk = self.k + 1
        
        _, knn_indices = similarities.topk(topk, dim=1)  # [B, K]
        
        if not exclude_self:
            knn_indices = knn_indices[:, 1:]  # Remove self
        
        return knn_indices.cpu().numpy()
    
    def save(self, path):
        """Save KNN index to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'features': self.features,
                'index_to_dataset_idx': self.index_to_dataset_idx,
                'k': self.k,
            }, f)
        print(f"Saved KNN index to {path}")
    
    def load(self, path):
        """Load KNN index from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.features = data['features']
        self.index_to_dataset_idx = data['index_to_dataset_idx']
        self.k = data['k']
        print(f"Loaded KNN index from {path}")