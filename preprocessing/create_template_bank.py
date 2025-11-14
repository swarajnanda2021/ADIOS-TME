"""
Create template feature bank from PanNuke dataset.

Usage:
    python preprocessing/create_template_bank.py \
        --pannuke_path /path/to/pannuke \
        --checkpoint_path /path/to/pretrained_vit.pth \
        --output_path templates/pannuke_features.pkl
"""

import torch
import pickle
from pathlib import Path
from tqdm import tqdm

# Would need PanNuke dataset loading code
# Extract features for nuclei and background patches
# Save as pickle file