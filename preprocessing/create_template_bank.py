"""
Create template feature bank from PanNuke dataset.

This script extracts feature templates from nuclei and background patches
using a frozen pretrained ViT encoder.

Usage:
    python preprocessing/create_template_bank.py \
        --pannuke_path /path/to/pannuke \
        --checkpoint_path /path/to/pretrained_vit.pth \
        --output_path ./templates/pannuke_features.pkl \
        --mask_threshold 0.5
"""

import sys
import os

# Add project root to path to allow imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2

from models.vision_transformer.modern_vit import VisionTransformer


def load_frozen_encoder(checkpoint_path, args):
    """Load and freeze ViT encoder."""
    print(f"Loading frozen encoder from {checkpoint_path}")
    
    encoder = VisionTransformer(
        img_size=224,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.1,
        num_register_tokens=4,
    )
    
    # Try loading checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'student' in checkpoint:
            state_dict = checkpoint['student']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        msg = encoder.load_state_dict(state_dict, strict=False)
        print(f"  Loaded checkpoint with message: {msg}")
    except FileNotFoundError:
        print(f"  WARNING: Checkpoint not found at {checkpoint_path}")
        print(f"  Using random initialization (placeholder)")
    
    # Freeze all parameters
    for param in encoder.parameters():
        param.requires_grad = False
    
    encoder.eval()
    encoder.cuda()
    
    print("  Encoder frozen and on GPU")
    return encoder


def load_pannuke_image_and_mask(image_path, mask_path):
    """
    Load image and corresponding binary mask.
    
    Returns:
        image: [H, W, 3] in range [0, 1]
        binary_mask: [H, W] in {0, 1}
    """
    # Load image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    
    # Load mask (NPY format with instance labels)
    mask = np.load(str(mask_path))
    
    # Create binary mask: nuclei > 0, background = 0
    # From HoverNetBasedDataset._process_mask_binary:
    # nuclei are labeled as (mask > 0 & mask < max_value)
    mask_single = np.squeeze(mask)
    binary_mask = np.zeros(mask_single.shape, dtype=np.float32)
    max_value = np.max(mask_single)
    
    # Nuclei: (mask > 0 & mask < max_value)
    binary_mask[(mask_single > 0) & (mask_single < max_value)] = 1.0
    
    return image, binary_mask


def extract_templates(encoder, image, binary_mask, mask_threshold=0.5):
    """
    Extract template features from image.
    
    For each spatial location in the feature map:
    - If nuclei coverage >= threshold: store as nuclei template
    - If background coverage >= threshold: store as background template
    
    Args:
        encoder: Frozen ViT encoder
        image: [H, W, 3] image in range [0, 1]
        binary_mask: [H, W] binary mask (1=nuclei, 0=background)
        mask_threshold: Minimum coverage to include patch (default 0.5)
        
    Returns:
        nuclei_templates: List of [D] feature vectors for nuclei
        background_templates: List of [D] feature vectors for background
    """
    # Prepare image tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    image_tensor = image_tensor.cuda()
    
    # Extract features
    with torch.no_grad():
        # Get intermediate layers (returns list of 4 feature maps)
        layer_features = encoder.get_intermediate_layers(image_tensor)
        
        # Use last layer features
        features = layer_features[-1]  # [1, N+5, D] (cls + 4 registers + patches)
        
        # Extract patch tokens only (remove cls and registers)
        patch_features = features[:, 5:, :]  # [1, N_patches, D]
        
        # Reshape to spatial
        B, N, D = patch_features.shape
        H_feat = W_feat = int(N ** 0.5)
        features_spatial = patch_features.reshape(B, H_feat, W_feat, D).permute(0, 3, 1, 2)
        # [B, D, H_feat, W_feat]
        
        features_spatial = features_spatial[0]  # [D, H_feat, W_feat]
    
    # Downsample mask to match feature spatial resolution
    mask_resized = cv2.resize(
        binary_mask.astype(np.float32),
        (W_feat, H_feat),
        interpolation=cv2.INTER_NEAREST
    )  # [H_feat, W_feat]
    
    nuclei_templates = []
    background_templates = []
    
    # Iterate over spatial locations
    for i in range(H_feat):
        for j in range(W_feat):
            # Coverage at this spatial location
            nuclei_coverage = mask_resized[i, j]
            bg_coverage = 1.0 - mask_resized[i, j]
            
            # Extract feature vector for this location
            feature_vec = features_spatial[:, i, j].cpu()  # [D]
            
            # Store if coverage threshold met
            if nuclei_coverage >= mask_threshold:
                nuclei_templates.append(feature_vec)
            
            if bg_coverage >= mask_threshold:
                background_templates.append(feature_vec)
    
    return nuclei_templates, background_templates


def create_template_bank(args):
    """Main function to create template bank from PanNuke."""
    
    # Setup paths
    pannuke_path = Path(args.pannuke_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load encoder
    encoder = load_frozen_encoder(args.checkpoint_path, args)
    
    # Get image directories
    image_dir = pannuke_path / "Training" / args.magnification / "tissue_images"
    mask_dir = pannuke_path / "Training" / args.magnification / "masks"
    
    if not image_dir.exists():
        raise ValueError(f"Image directory not found: {image_dir}")
    if not mask_dir.exists():
        raise ValueError(f"Mask directory not found: {mask_dir}")
    
    # Get all image files
    image_files = sorted([f for f in image_dir.iterdir() if f.suffix in ['.png', '.jpg']])
    print(f"Found {len(image_files)} images in {image_dir}")
    
    # Collect all templates
    all_nuclei_templates = []
    all_background_templates = []
    
    stats = {
        'total_images': len(image_files),
        'processed_images': 0,
        'skipped_images': 0,
        'total_nuclei_patches': 0,
        'total_background_patches': 0,
    }
    
    # Process each image
    for image_path in tqdm(image_files, desc="Extracting templates"):
        try:
            # Construct mask path
            mask_name = image_path.stem + '.npy'
            mask_path = mask_dir / mask_name
            
            if not mask_path.exists():
                stats['skipped_images'] += 1
                continue
            
            # Load image and mask
            image, binary_mask = load_pannuke_image_and_mask(image_path, mask_path)
            
            # Extract templates
            nuclei_templates, bg_templates = extract_templates(
                encoder, image, binary_mask, 
                mask_threshold=args.mask_threshold
            )
            
            all_nuclei_templates.extend(nuclei_templates)
            all_background_templates.extend(bg_templates)
            
            stats['processed_images'] += 1
            stats['total_nuclei_patches'] += len(nuclei_templates)
            stats['total_background_patches'] += len(bg_templates)
            
        except Exception as e:
            print(f"  Error processing {image_path.name}: {str(e)}")
            stats['skipped_images'] += 1
            continue
    
    # Stack into tensors
    if not all_nuclei_templates:
        raise RuntimeError("No nuclei templates extracted!")
    if not all_background_templates:
        raise RuntimeError("No background templates extracted!")
    
    nuclei_features = torch.stack(all_nuclei_templates)
    background_features = torch.stack(all_background_templates)
    
    print(f"\n{'='*80}")
    print(f"Template Extraction Complete")
    print(f"{'='*80}")
    print(f"Nuclei features shape: {nuclei_features.shape}")
    print(f"Background features shape: {background_features.shape}")
    print(f"Total images processed: {stats['processed_images']}")
    print(f"Images skipped: {stats['skipped_images']}")
    print(f"Total nuclei patches stored: {stats['total_nuclei_patches']}")
    print(f"Total background patches stored: {stats['total_background_patches']}")
    
    # Save to pickle
    template_bank = {
        'nuclei_features': nuclei_features,
        'background_features': background_features,
        'stats': stats,
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(template_bank, f)
    
    file_size_gb = output_path.stat().st_size / 1e9
    print(f"\nTemplate bank saved to {output_path}")
    print(f"File size: {file_size_gb:.3f} GB")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser("Create PanNuke Template Bank")
    
    # Path arguments
    parser.add_argument('--pannuke_path',
                       default='/data1/vanderbc/nandas1/Benchmarks/PanNuke_patches_unnormalized',
                       type=str, help='Path to PanNuke dataset')
    parser.add_argument('--checkpoint_path',
                       default='/data1/vanderbc/nandas1/TCGA_Dinov2_ViT-B_run2/logs/checkpoint.pth',
                       type=str, help='Path to pretrained ViT checkpoint')
    parser.add_argument('--output_path',
                       default='./templates/pannuke_features.pkl',
                       type=str, help='Output path for template bank pickle')
    
    # Processing arguments
    parser.add_argument('--magnification', default='40x', type=str,
                       choices=['20x', '40x'],
                       help='Magnification to use (default: 40x)')
    parser.add_argument('--mask_threshold', default=0.5, type=float,
                       help='Minimum coverage threshold for including patches (default: 0.5)')
    
    # Model architecture arguments
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--embed_dim', default=768, type=int,
                       help='384=small, 768=base, 1024=large')
    parser.add_argument('--num_heads', default=12, type=int,
                       help='6=small, 12=base, 16=large')
    parser.add_argument('--depth', default=12, type=int,
                       help='12=small/base, 24=large')
    
    args = parser.parse_args()
    
    print("="*80)
    print("PanNuke Template Bank Creation")
    print("="*80)
    print(f"PanNuke path: {args.pannuke_path}")
    print(f"Magnification: {args.magnification}")
    print(f"Mask threshold: {args.mask_threshold}")
    print(f"Output path: {args.output_path}")
    print(f"Model: ViT (embed_dim={args.embed_dim}, depth={args.depth}, heads={args.num_heads})")
    print("="*80)
    
    create_template_bank(args)


if __name__ == '__main__':
    main()