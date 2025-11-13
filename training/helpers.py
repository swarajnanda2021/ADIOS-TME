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


def load_pretrained_mask_model(checkpoint_path, num_masks=3):
    """
    Load pre-trained mask model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        num_masks: Number of masks
        
    Returns:
        Loaded mask model
    """
    print(f"Loading pre-trained mask model from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    from models.vision_transformer.modern_vit import VisionTransformer
    
    mask_encoder = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        dual_norm=False,
        drop_path_rate=0.1,
        pre_norm=False,
        num_register_tokens=4,
    )
    
    mask_model = MaskModel_SpectralNorm(
        encoder=mask_encoder,
        num_masks=num_masks,
        encoder_dim=192,
        drop_rate=0.2
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'mask_model' in checkpoint:
        mask_state_dict = checkpoint['mask_model']
    else:
        raise KeyError("No 'mask_model' found in checkpoint")
    
    cleaned_state_dict = {}
    for k, v in mask_state_dict.items():
        if k.startswith('module.'):
            cleaned_state_dict[k.replace('module.', '')] = v
        else:
            cleaned_state_dict[k] = v
    
    mask_model.load_state_dict(cleaned_state_dict, strict=False)
    
    print(f"Successfully loaded mask model weights")
    
    if 'iteration' in checkpoint:
        print(f"Checkpoint was saved at iteration: {checkpoint['iteration']}")
    
    return mask_model


def apply_masks_to_images(images, masks):
    """
    Apply masks to images.
    
    Args:
        images: [B, C, H, W]
        masks: [B, num_masks, H, W]
        
    Returns:
        List of [B, C, H, W] masked images
    """
    B, num_masks, H, W = masks.shape
    masked_images = []
    
    for i in range(num_masks):
        mask = masks[:, i:i+1, :, :]
        masked = images * (1 - mask)
        masked_images.append(masked)
    
    return masked_images


def extract_local_crops_from_masked(masked_img, n_crops, crop_size=96):
    """
    Extract n_crops local crops from a masked image.
    
    Args:
        masked_img: [B, C, H, W] masked image
        n_crops: Number of crops to extract
        crop_size: Size of each crop
        
    Returns:
        List of crops
    """
    B, C, H, W = masked_img.shape
    crops = []
    
    for _ in range(n_crops):
        crops_batch = []
        for b in range(B):
            top = random.randint(0, H - crop_size)
            left = random.randint(0, W - crop_size)
            crop = masked_img[b:b+1, :, top:top+crop_size, left:left+crop_size]
            crops_batch.append(crop)
        crops.append(torch.cat(crops_batch, dim=0))
    
    return crops


def generate_random_token_masks(batch_size, n_patches_h, n_patches_w, mask_ratio, device):
    """
    Generate random token masks for iBOT training.
    
    Args:
        batch_size: Batch size
        n_patches_h: Number of patches in height
        n_patches_w: Number of patches in width
        mask_ratio: Ratio of tokens to mask
        device: Device to create masks on
        
    Returns:
        Boolean mask where True indicates masked tokens [B, N]
    """
    n_patches = n_patches_h * n_patches_w
    token_masks = torch.bernoulli(
        torch.ones(batch_size, n_patches) * mask_ratio
    ).bool().to(device)
    return token_masks


def calculate_total_student_views(args):
    """
    Calculate total number of student views based on configuration.
    
    Args:
        args: Arguments namespace
        
    Returns:
        Total number of student views
    """
    if args.global_views == 0:
        total = args.num_masks
        total += args.n_standard_local_crops
        total += args.num_masks * args.crops_per_mask
    else:
        total = args.global_views
        total += args.num_masks
        total += args.n_standard_local_crops
        total += args.num_masks * args.crops_per_mask
    return total


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




def apply_crops_to_masked_images_batched(original_image, cached_masks, K):
    """
    Apply crops while maintaining batch structure.
    
    CRITICAL: Returns K crop variants per mask, each with shape [B, C, H, W].
    
    Args:
        original_image: Original images [B, C, H, W]
        cached_masks: Pre-computed masks [B, num_masks, H, W]
        K: Number of crops per mask
    
    Returns:
        List of [B, C, H, W] tensors, one per (mask, crop_position) combination
        Length: num_masks * K
    """
    B, C, H, W = original_image.shape
    num_masks = cached_masks.shape[1]
    
    all_crop_variants = []
    
    # For each mask
    for m in range(num_masks):
        mask = cached_masks[:, m:m+1, :, :]  # [B, 1, H, W]
        
        # Create masked image for this mask
        masked_img = original_image * (1 - mask)  # [B, C, H, W]
        
        # For each crop position
        for k in range(K):
            # Generate crop parameters for entire batch (same params for all samples)
            scale = np.random.uniform(0.4, 1.0)
            crop_size = int(224 * scale)
            
            # Random position
            max_pos = 224 - crop_size
            top = np.random.randint(0, max_pos + 1) if max_pos > 0 else 0
            left = np.random.randint(0, max_pos + 1) if max_pos > 0 else 0
            
            # Random rotation
            angle = np.random.uniform(-180, 180)
            
            # Apply crop to entire batch
            cropped = masked_img[:, :, top:top+crop_size, left:left+crop_size]
            
            # Resize to 224x224
            cropped_resized = F.interpolate(cropped, size=(224, 224),
                                           mode='bilinear', align_corners=False)
            
            # Apply rotation to entire batch
            cropped_rotated = rotate_tensor_batch(cropped_resized, angle)
            
            all_crop_variants.append(cropped_rotated)  # [B, C, 224, 224]
    
    return all_crop_variants


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


def process_student_with_cached_masks_and_crops(
    student, 
    cached_masks, 
    original_image, 
    crop_params,  # Not used in batched version
    K, 
    current_iteration,
    adios_loss, 
    num_masks=3
):
    """
    Efficient student forward with cached masks and multi-crop (BATCHED VERSION).
    
    This function:
    1. Uses pre-computed masks (no mask model forward)
    2. Generates masked images (batched)
    3. Generates cropped variants (batched) 
    4. Does ONE batched forward through student
    5. Computes loss with proper multi-crop parameters
    
    Args:
        student: Student model
        cached_masks: Pre-computed masks [B, num_masks, H, W]
        original_image: Original images [B, C, H, W]
        crop_params: Not used (kept for API compatibility)
        K: Number of crops per mask
        current_iteration: Current training iteration
        adios_loss: Loss function
        num_masks: Number of masks (default: 3)
    
    Returns:
        loss: Computed loss
        metrics: Loss metrics
    """
    # Create all images to process (all maintaining batch structure)
    all_images = [original_image]  # [B, C, H, W]
    
    # Add full-size masked images (each maintains batch structure)
    for i in range(num_masks):
        mask = cached_masks[:, i:i+1, :, :]  # [B, 1, H, W]
        masked_img = original_image * (1 - mask)  # [B, C, H, W]
        all_images.append(masked_img)
    
    # Generate and add cropped variants if K > 0
    if K > 0:
        crop_variants = apply_crops_to_masked_images_batched(
            original_image, cached_masks, K
        )
        # crop_variants is a list of num_masks * K tensors, each [B, C, H, W]
        all_images.extend(crop_variants)
    
    # Debug: Check we have the right number
    expected_views = 1 + num_masks + (num_masks * K)
    actual_views = len(all_images)
    if actual_views != expected_views:
        print(f"[ERROR] Expected {expected_views} views but got {actual_views}")
        print(f"  original: 1, full_masks: {num_masks}, crops: {num_masks * K}")
    
    # ✅ SINGLE batched forward pass through student with ALL images
    # all_images is a list of tensors, each [B, C, H, W]
    all_embeddings = student(all_images)
    # all_embeddings is a list of tensors, each [B, D]
    
    # Split the embeddings
    orig_emb = all_embeddings[0]  # [B, D]
    masked_embs = all_embeddings[1:num_masks+1]  # List of num_masks tensors, each [B, D]
    
    if K > 0:
        cropped_embs = all_embeddings[num_masks+1:]  # List of (num_masks*K) tensors, each [B, D]
        # Combine for loss computation
        all_masked_embeddings = masked_embs + cropped_embs
    else:
        all_masked_embeddings = masked_embs
    
    # Verify all embeddings have same batch size
    batch_size = orig_emb.shape[0]
    for i, emb in enumerate(all_masked_embeddings):
        if emb.shape[0] != batch_size:
            print(f"[ERROR] Embedding {i} has batch size {emb.shape[0]}, expected {batch_size}")
    
    # Compute loss with proper parameters
    loss, metrics = adios_loss(
        orig_emb,
        all_masked_embeddings,
        masks=None,
        iteration=current_iteration,
        forward_type='student',
        num_base_masks=num_masks,
        K=K
    )
    
    return loss, metrics

