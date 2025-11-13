"""
Simplified dataset for ADIOS-TME training.
Only needs to return original images, no complex augmentation.
"""

import os
import torch
import random
import pickle
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import zipfile
import io


class ADIOSPathologyDataset(Dataset):
    """
    Simple dataset for ADIOS-TME training.
    Returns normalized pathology images.
    """
    def __init__(
        self,
        data_path,
        index_file="dataset_index.pkl",
        img_size=224,
        mean=(0.6816, 0.5640, 0.7232),
        std=(0.1617, 0.1714, 0.1389),
        max_samples=None,  # For debugging
    ):
        self.data_path = data_path
        self.img_size = img_size
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.max_samples = max_samples
        
        # Load dataset index
        self.samples = self._load_index(index_file)
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded {len(self.samples)} samples")
    
    def _load_index(self, index_file):
        """Load the dataset index."""
        index_path = os.path.join(self.data_path, index_file)
        
        if os.path.exists(index_path):
            with open(index_path, 'rb') as f:
                index_data = pickle.load(f)
            
            # Flatten the index into a list of (zip_path, image_name) tuples
            samples = []
            for zip_path, image_names in index_data:
                for img_name in image_names:
                    if '_448_' in img_name and '_224_' not in img_name:
                        samples.append((zip_path, img_name))
            
            return samples
        else:
            # Simple fallback: scan for zip files
            samples = []
            for zip_file in os.listdir(self.data_path):
                if zip_file.endswith('.zip'):
                    zip_path = os.path.join(self.data_path, zip_file)
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zf:
                            for name in zf.namelist():
                                if name.endswith('.png'):
                                    samples.append((zip_path, name))
                    except:
                        continue
            return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Load and preprocess a single image.
        
        Returns:
            Tensor: Normalized image [3, H, W]
        """
        zip_path, img_name = self.samples[idx]
        
        # Load image from zip
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                data = zf.read(img_name)
                img = Image.open(io.BytesIO(data)).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_name} from {zip_path}: {e}")
            # Return a random valid sample instead
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        # Resize to target size
        img = img.resize((self.img_size, self.img_size), Image.BICUBIC)
        
        # Convert to tensor
        img = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        
        # Normalize
        img = (img - self.mean) / self.std
        
        return img


class ADIOSDataTransform:
    """
    Simple augmentation for ADIOS training.
    Just basic augmentation, no multi-crop complexity.
    """
    def __init__(
        self,
        img_size=224,
        mean=(0.6816, 0.5640, 0.7232),
        std=(0.1617, 0.1714, 0.1389),
    ):
        import torchvision.transforms as T
        
        self.transform = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.4, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    
    def __call__(self, img):
        """Apply augmentation to PIL image."""
        return self.transform(img)