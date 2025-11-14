"""
Helper functions for DINOv2 training.
Includes mask generation, visualization, and data loading utilities.
"""

import os
import gc
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


from models.vision_transformer.auxiliary_models import MaskModel_SpectralNorm




def save_iteration_masks_efficient(
    images, 
    masks, 
    iteration, 
    save_dir, 
    reconstructed_images=None, 
    num_samples=4, 
    timeout_seconds=30
):
    """
    Efficient mask visualization that prevents hanging.
    
    Args:
        images: Input images [B, C, H, W]
        masks: Generated masks [B, num_masks, H, W]
        iteration: Current iteration
        save_dir: Directory to save visualizations
        reconstructed_images: Optional reconstructed images
        num_samples: Number of samples to visualize
        timeout_seconds: Timeout for visualization (unused, kept for compatibility)
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        images_cpu = images.detach().cpu().float()
        masks_cpu = masks.detach().cpu().float()
        
        torch.cuda.empty_cache()
        
        batch_size = images_cpu.size(0)
        num_samples = min(num_samples, batch_size, 4)
        
        torch.manual_seed(42)
        if batch_size > num_samples:
            indices = torch.randperm(batch_size)[:num_samples]
            images_cpu = images_cpu[indices]
            masks_cpu = masks_cpu[indices]
        
        mean_cpu = torch.tensor([0.6816, 0.5640, 0.7232]).view(1, 3, 1, 1)
        std_cpu = torch.tensor([0.1617, 0.1714, 0.1389]).view(1, 3, 1, 1)
        
        images_norm = images_cpu * std_cpu + mean_cpu
        images_norm = torch.clamp(images_norm, 0, 1)
        
        num_masks = masks_cpu.size(1)
        
        cols = min(num_masks + 2, 5)
        fig, axes = plt.subplots(num_samples, cols, figsize=(3*cols, 3*num_samples))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        
        for i in range(num_samples):
            col_idx = 0
            
            img_np = images_norm[i].permute(1, 2, 0).numpy()
            axes[i, col_idx].imshow(img_np)
            axes[i, col_idx].axis('off')
            if i == 0:
                axes[i, col_idx].set_title('Original', fontsize=10)
            col_idx += 1
            
            for j in range(min(num_masks, cols - 2)):
                mask_np = masks_cpu[i, j].numpy()
                axes[i, col_idx].imshow(mask_np, cmap='viridis', vmin=0, vmax=1)
                axes[i, col_idx].axis('off')
                if i == 0:
                    axes[i, col_idx].set_title(f'Mask {j+1}', fontsize=10)
                col_idx += 1
            
            if num_masks >= 3:
                rgb_masks = torch.stack([
                    masks_cpu[i, 0], 
                    masks_cpu[i, 1], 
                    masks_cpu[i, 2]
                ], dim=0).permute(1, 2, 0).numpy()
                axes[i, col_idx].imshow(rgb_masks, vmin=0, vmax=1)
                title = 'RGB Combined'
            else:
                avg_mask = masks_cpu[i].mean(dim=0).numpy()
                axes[i, col_idx].imshow(avg_mask, cmap='viridis', vmin=0, vmax=1)
                title = 'Avg Masks'
            
            axes[i, col_idx].axis('off')
            if i == 0:
                axes[i, col_idx].set_title(title, fontsize=10)
        
        save_path = os.path.join(save_dir, f'iter_{iteration:06d}_masks.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=100, facecolor='white')
        
        plt.close(fig)
        plt.clf()
        
        del images_cpu, masks_cpu, images_norm
        gc.collect()
        
        print(f"Mask visualization saved to {save_path}")
        
    except Exception as e:
        print(f"Visualization failed: {str(e)}")
        plt.close('all')
        plt.clf()


def worker_init_fn(worker_id):
    """
    Initialize worker with proper random seeding.
    
    Args:
        worker_id: Worker ID
    """
    import numpy as np
    import torch
    from torch.utils.data import get_worker_info
    
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    
    if hasattr(dataset, 'base_dataset'):
        dataset.base_dataset.set_worker_info(worker_info.id, worker_info.num_workers)
        seed = dataset.base_dataset.seed
    else:
        dataset.worker_id = worker_info.id
        seed = dataset.seed
    
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def setup_ddp_model(model, args, find_unused=False):
    """
    Setup model for distributed data parallel training.
    
    Args:
        model: Model to wrap    
        args: Arguments with GPU info
        find_unused: Whether to find unused parameters
        
    Returns:
        DDP-wrapped model (or original model if single GPU)
    """
    # Enable gradient checkpointing BEFORE DDP wrapping
    if hasattr(args, 'grad_checkpointing') and args.grad_checkpointing:
        if hasattr(model, 'set_grad_checkpointing'):
            model.set_grad_checkpointing(True)
            print(f"Enabled gradient checkpointing before DDP wrapping")

    # Only wrap with DDP if using multiple GPUs
    if args.world_size > 1:
        ddp_model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=find_unused,
            broadcast_buffers=True
        )
        ddp_model._set_static_graph()
        return ddp_model
    else:
        # Single GPU - return model as-is
        print(f"Single GPU mode - skipping DDP wrapper")
        return model



def rotate_tensor(tensor, angle):
    """
    Rotate tensor by given angle in degrees.
    
    Args:
        tensor: Input tensor [B, C, H, W]
        angle: Rotation angle in degrees
    
    Returns:
        Rotated tensor
    """
    # Convert angle to radians
    angle_rad = angle * np.pi / 180
    
    # Create rotation matrix
    cos_val = np.cos(angle_rad)
    sin_val = np.sin(angle_rad)
    
    # Use grid_sample for rotation
    theta = torch.tensor([
        [cos_val, -sin_val, 0],
        [sin_val, cos_val, 0]
    ], dtype=tensor.dtype, device=tensor.device).unsqueeze(0).repeat(tensor.shape[0], 1, 1)
    
    grid = F.affine_grid(theta, tensor.size(), align_corners=False)
    rotated = F.grid_sample(tensor, grid, align_corners=False, padding_mode='zeros')
    
    return rotated




def rotate_tensor_batch(tensor, angle):
    """
    Rotate entire batch by given angle in degrees.
    
    Args:
        tensor: Input tensor [B, C, H, W]
        angle: Rotation angle in degrees (same for entire batch)
    
    Returns:
        Rotated tensor [B, C, H, W]
    """
    B = tensor.shape[0]
    
    # Convert angle to radians
    angle_rad = angle * np.pi / 180
    
    # Create rotation matrix
    cos_val = np.cos(angle_rad)
    sin_val = np.sin(angle_rad)
    
    # Use grid_sample for rotation (same rotation for entire batch)
    theta = torch.tensor([
        [cos_val, -sin_val, 0],
        [sin_val, cos_val, 0]
    ], dtype=tensor.dtype, device=tensor.device).unsqueeze(0).repeat(B, 1, 1)
    
    grid = F.affine_grid(theta, tensor.size(), align_corners=False)
    rotated = F.grid_sample(tensor, grid, align_corners=False, padding_mode='zeros')
    
    return rotated


