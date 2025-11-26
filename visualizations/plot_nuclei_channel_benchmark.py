#!/usr/bin/env python3
"""
Nuclei Channel Benchmark for ADIOS-TME Mask Models.

Evaluates mask model checkpoints on PanNuke and MonuSeg datasets,
identifying which channel best segments nuclei and computing mIoU.
"""

import os
import sys
import glob
import re
import json
import gc
import argparse
import importlib.util
import numpy as np
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Add ADIOS-UNet parent directory to path for model imports
script_dir = Path(__file__).parent.absolute()
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

# Import ADIOS-UNet models (these come from parent_dir/models/)
from models.UNet import ADIOSMaskModel
from models.vision_transformer.modern_vit import VisionTransformer
from models.vision_transformer.auxiliary_models import MaskModel

# Load datasets module directly from PostProc path (avoids path conflicts)
def load_datasets_module():
    datasets_path = "/data1/vanderbc/nandas1/PostProc/datasets.py"
    spec = importlib.util.spec_from_file_location("postproc_datasets", datasets_path)
    datasets_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(datasets_module)
    return datasets_module

datasets_module = load_datasets_module()
PanNukeDataset = datasets_module.PanNukeDataset
MonuSegDataset = datasets_module.MonuSegDataset
SynchronizedTransform = datasets_module.SynchronizedTransform


def find_checkpoints(logs_dir):
    """Find all checkpoint files and extract iteration numbers."""
    pattern = os.path.join(logs_dir, "checkpoint_iter_*.pth")
    checkpoint_files = glob.glob(pattern)
    
    checkpoints = []
    for ckpt_path in checkpoint_files:
        match = re.search(r'checkpoint_iter_(\d+)\.pth', os.path.basename(ckpt_path))
        if match:
            iteration = int(match.group(1))
            checkpoints.append((iteration, ckpt_path))
    
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def build_mask_model_from_args(args, device):
    """Build the correct mask model architecture based on checkpoint args."""
    
    mask_model_type = getattr(args, 'mask_model_type', 'vit_unet')
    num_masks = getattr(args, 'num_masks', 3)
    
    print(f"  Building mask model: type={mask_model_type}, num_masks={num_masks}")
    
    if mask_model_type == 'adios':
        from models.UNet import ADIOSMaskModel
        mask_model = ADIOSMaskModel(
            num_masks=num_masks,
            img_size=224,
        )
    else:
        from models.vision_transformer.modern_vit import VisionTransformer
        from models.vision_transformer.auxiliary_models import MaskModel
        
        mask_encoder_dim = getattr(args, 'mask_encoder_dim', 192)
        mask_encoder_depth = getattr(args, 'mask_encoder_depth', 12)
        mask_dropout = getattr(args, 'mask_dropout', 0.2)
        
        mask_encoder = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=mask_encoder_dim,
            depth=mask_encoder_depth,
            num_heads=max(mask_encoder_dim // 64, 3),
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_norm=False,
            dual_norm=False,
            drop_path_rate=0.1,
            pre_norm=False,
            num_register_tokens=4,
        )
        
        mask_model = MaskModel(
            encoder=mask_encoder,
            num_masks=num_masks,
            encoder_dim=mask_encoder_dim,
            drop_rate=mask_dropout
        )
    
    return mask_model.to(device)


def load_mask_model_weights(mask_model, checkpoint_path):
    """Load mask model weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'mask_model' not in checkpoint:
        raise KeyError(f"No 'mask_model' in checkpoint. Keys: {list(checkpoint.keys())}")
    
    mask_state_dict = checkpoint['mask_model']
    
    cleaned_state_dict = OrderedDict()
    for k, v in mask_state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        cleaned_state_dict[name] = v
    
    missing, unexpected = mask_model.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        print(f"    Missing keys: {len(missing)}")
    if unexpected:
        print(f"    Unexpected keys: {len(unexpected)}")
    
    return mask_model


def calculate_iou(pred_mask, gt_mask, threshold=0.5):
    """Calculate IoU between predicted and ground truth masks."""
    pred_binary = (pred_mask > threshold).float()
    gt_binary = gt_mask.float()
    
    intersection = (pred_binary * gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (intersection / union).item()


def evaluate_checkpoint(mask_model, dataset, device, batch_size=32):
    """Evaluate mask model on dataset, finding best nuclei channel per sample."""
    from torch.utils.data import DataLoader
    
    mask_model.eval()
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    all_best_ious = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, leave=False, desc="    Batches"):
            images, mask_2ch, _, _ = batch
            
            gt_nuclei = mask_2ch[:, 0]  # [B, H, W]
            
            images_gpu = images.to(device)
            if images_gpu.shape[-1] != 224:
                images_gpu = F.interpolate(images_gpu, size=(224, 224), mode='bilinear', align_corners=False)
            
            with torch.cuda.amp.autocast():
                mask_output = mask_model(images_gpu)
                predicted_masks = mask_output['masks']
            
            H_gt, W_gt = gt_nuclei.shape[1], gt_nuclei.shape[2]
            if predicted_masks.shape[-2:] != (H_gt, W_gt):
                predicted_masks = F.interpolate(predicted_masks, size=(H_gt, W_gt), mode='bilinear', align_corners=False)
            
            predicted_masks = predicted_masks.cpu()
            gt_nuclei = gt_nuclei.cpu()
            
            for b in range(predicted_masks.shape[0]):
                channel_ious = [calculate_iou(predicted_masks[b, ch], gt_nuclei[b]) 
                               for ch in range(predicted_masks.shape[1])]
                all_best_ious.append(max(channel_ious))
    
    return np.array(all_best_ious)


class CombinedDataset:
    """Combine train and test datasets."""
    def __init__(self, train_ds, test_ds):
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.train_len = len(train_ds)
    
    def __len__(self):
        return len(self.train_ds) + len(self.test_ds)
    
    def __getitem__(self, idx):
        if idx < self.train_len:
            return self.train_ds[idx]
        return self.test_ds[idx - self.train_len]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths (relative to script location)
    script_dir = Path(__file__).parent.absolute()
    logs_dir = script_dir.parent / 'logs'
    output_dir = script_dir
    
    pannuke_path = "/data1/vanderbc/nandas1/Benchmarks/PanNuke_patches_unnormalized"
    monuseg_path = "/data1/vanderbc/nandas1/Benchmarks/MonuSeg_patches_unnormalized"
    
    # Find checkpoints
    print(f"Searching for checkpoints in {logs_dir}...")
    checkpoints = find_checkpoints(str(logs_dir))
    
    if not checkpoints:
        print(f"No checkpoints found in {logs_dir}")
        print("Looking for checkpoint_iter_*.pth files")
        return
    
    print(f"Found {len(checkpoints)} checkpoints:")
    for iteration, path in checkpoints:
        print(f"  {iteration:>7d}: {os.path.basename(path)}")
    
    # Create transform (no augmentation for evaluation)
    transform_settings = {
        "normalize": {"mean": [0.6816, 0.5640, 0.7232], "std": [0.1617, 0.1714, 0.1389]},
        "RandomRotate90": {"p": 0}, "HorizontalFlip": {"p": 0}, "VerticalFlip": {"p": 0},
        "Downscale": {"scale": 0.5, "p": 0}, "Blur": {"blur_limit": 7, "p": 0},
        "ColorJitter": {"scale_setting": 0.25, "scale_color": 0.1, "p": 0}
    }
    transform = SynchronizedTransform(transform_settings, input_shape=96)
    
    # Load datasets
    print("\nLoading datasets...")
    pannuke = CombinedDataset(
        PanNukeDataset(pannuke_path, 'Training', '40x', transform),
        PanNukeDataset(pannuke_path, 'Test', '40x', transform)
    )
    monuseg = CombinedDataset(
        MonuSegDataset(monuseg_path, 'Training', '40x', transform),
        MonuSegDataset(monuseg_path, 'Test', '40x', transform)
    )
    print(f"  PanNuke: {len(pannuke)} samples")
    print(f"  MonuSeg: {len(monuseg)} samples")
    
    # Results storage
    results = {
        'iterations': [],
        'pannuke_mean': [], 'pannuke_std': [],
        'monuseg_mean': [], 'monuseg_std': []
    }
    
    # Evaluate each checkpoint
    for iteration, ckpt_path in checkpoints:
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}")
        print(f"{'='*60}")
        
        # Load checkpoint args
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        args = checkpoint.get('args', None)
        
        if args is None:
            print("  WARNING: No args in checkpoint, using defaults (adios, 3 masks)")
            import argparse
            args = argparse.Namespace(mask_model_type='adios', num_masks=3)
        
        # Build and load model
        mask_model = build_mask_model_from_args(args, device)
        mask_model = load_mask_model_weights(mask_model, ckpt_path)
        mask_model.eval()
        
        # Evaluate
        print("  Evaluating PanNuke...")
        pannuke_ious = evaluate_checkpoint(mask_model, pannuke, device)
        
        print("  Evaluating MonuSeg...")
        monuseg_ious = evaluate_checkpoint(mask_model, monuseg, device)
        
        # Store results
        results['iterations'].append(iteration)
        results['pannuke_mean'].append(float(pannuke_ious.mean()))
        results['pannuke_std'].append(float(pannuke_ious.std()))
        results['monuseg_mean'].append(float(monuseg_ious.mean()))
        results['monuseg_std'].append(float(monuseg_ious.std()))
        
        print(f"  PanNuke mIoU: {pannuke_ious.mean():.4f} ± {pannuke_ious.std():.4f}")
        print(f"  MonuSeg mIoU: {monuseg_ious.mean():.4f} ± {monuseg_ious.std():.4f}")
        
        # Cleanup
        del mask_model, checkpoint
        gc.collect()
        torch.cuda.empty_cache()
    
    # Convert to arrays for plotting
    iterations = np.array(results['iterations'])
    pannuke_mean = np.array(results['pannuke_mean'])
    pannuke_std = np.array(results['pannuke_std'])
    monuseg_mean = np.array(results['monuseg_mean'])
    monuseg_std = np.array(results['monuseg_std'])
    
    # Create plot
    print("\nGenerating plot...")
    plt.style.use('ggplot')
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    iterations_k = iterations / 1000  # Convert to thousands
    
    ax.errorbar(iterations_k, pannuke_mean, yerr=pannuke_std, 
                fmt='o-', capsize=4, capthick=1.5, linewidth=2, markersize=7,
                label='PanNuke', color='#E24A33', alpha=0.9)
    ax.errorbar(iterations_k, monuseg_mean, yerr=monuseg_std,
                fmt='s-', capsize=4, capthick=1.5, linewidth=2, markersize=7,
                label='MonuSeg', color='#348ABD', alpha=0.9)
    
    ax.set_xlabel('Iteration (k)', fontsize=14)
    ax.set_ylabel('mIoU', fontsize=14)
    ax.legend(fontsize=12, framealpha=0.9, loc='lower right')
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=11)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / 'nuclei_channel_benchmark.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Plot saved to {output_path}")
    
    # Save numerical results
    results_path = output_dir / 'nuclei_channel_benchmark.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for i, it in enumerate(results['iterations']):
        print(f"  {it:>6d}:  PanNuke={results['pannuke_mean'][i]:.4f}±{results['pannuke_std'][i]:.4f}  "
              f"MonuSeg={results['monuseg_mean'][i]:.4f}±{results['monuseg_std'][i]:.4f}")


if __name__ == '__main__':
    main()