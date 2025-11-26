"""
Helper functions for ADIOS-TME training.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def worker_init_fn(worker_id):
    """
    Initialize worker with proper random seeding.
    """
    from torch.utils.data import get_worker_info
    
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    
    if hasattr(dataset, 'base_dataset'):
        dataset.base_dataset.set_worker_info(worker_info.id, worker_info.num_workers)
        seed = dataset.base_dataset.seed
    else:
        dataset.worker_id = worker_info.id
        seed = getattr(dataset, 'seed', 42)
    
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def setup_ddp_model(model, args, find_unused=False):
    """
    Setup model for distributed data parallel training.
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
        return ddp_model
    else:
        print(f"Single GPU mode - skipping DDP wrapper")
        return model


def rotate_tensor_batch(tensor, angle):
    """
    Rotate entire batch by given angle in degrees.
    """
    B = tensor.shape[0]
    angle_rad = angle * np.pi / 180
    
    cos_val = np.cos(angle_rad)
    sin_val = np.sin(angle_rad)
    
    theta = torch.tensor([
        [cos_val, -sin_val, 0],
        [sin_val, cos_val, 0]
    ], dtype=tensor.dtype, device=tensor.device).unsqueeze(0).repeat(B, 1, 1)
    
    grid = F.affine_grid(theta, tensor.size(), align_corners=False)
    rotated = F.grid_sample(tensor, grid, align_corners=False, padding_mode='zeros')
    
    return rotated


def apply_crops_to_masked_images(original_image, cached_masks, K):
    """
    Apply crops while maintaining batch structure.
    
    Returns K crop variants per mask, each with shape [B, C, H, W].
    """
    B, C, H, W = original_image.shape
    num_masks = cached_masks.shape[1]
    
    all_crop_variants = []
    
    for m in range(num_masks):
        mask = cached_masks[:, m:m+1, :, :]
        masked_img = original_image * (1 - mask)
        
        for k in range(K):
            scale = np.random.uniform(0.4, 1.0)
            crop_size = int(224 * scale)
            
            max_pos = 224 - crop_size
            top = np.random.randint(0, max_pos + 1) if max_pos > 0 else 0
            left = np.random.randint(0, max_pos + 1) if max_pos > 0 else 0
            
            angle = np.random.uniform(-180, 180)
            
            cropped = masked_img[:, :, top:top+crop_size, left:left+crop_size]
            cropped_resized = F.interpolate(cropped, size=(224, 224),
                                           mode='bilinear', align_corners=False)
            cropped_rotated = rotate_tensor_batch(cropped_resized, angle)
            
            all_crop_variants.append(cropped_rotated)
    
    return all_crop_variants


def process_student_with_cached_masks_and_crops(
    student, 
    cached_masks, 
    original_image, 
    crop_params,
    K, 
    adios_loss, 
    num_masks=3
):
    """
    Efficient student forward with cached masks and multi-crop.
    
    1. Uses pre-computed masks (no mask model forward)
    2. Generates masked images (batched)
    3. Generates cropped variants (batched) 
    4. Does ONE batched forward through student
    5. Computes loss
    """
    all_images = [original_image]
    
    # Add full-size masked images
    for i in range(num_masks):
        mask = cached_masks[:, i:i+1, :, :]
        masked_img = original_image * (1 - mask)
        all_images.append(masked_img)
    
    # Generate and add cropped variants if K > 0
    if K > 0:
        crop_variants = apply_crops_to_masked_images(original_image, cached_masks, K)
        all_images.extend(crop_variants)
    
    # Single batched forward pass through student
    all_embeddings = student(all_images)
    
    # Split embeddings
    orig_emb = all_embeddings[0]
    masked_embs = all_embeddings[1:num_masks+1]
    
    if K > 0:
        cropped_embs = all_embeddings[num_masks+1:]
        all_masked_embeddings = masked_embs + cropped_embs
    else:
        all_masked_embeddings = masked_embs
    
    # Compute loss
    loss, metrics = adios_loss(
        orig_emb,
        all_masked_embeddings,
        masks=None,
        forward_type='student',
        num_base_masks=num_masks,
        K=K
    )
    
    return loss, metrics


def load_mask_encoder_from_student_checkpoint(mask_encoder, checkpoint_path, freeze=True):
    """
    Load mask encoder weights from a student checkpoint.
    """
    print(f"Loading mask encoder from student checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'student' not in checkpoint:
        raise KeyError("'student' key not found in checkpoint. Available keys: " + 
                      str(list(checkpoint.keys())))
    
    student_state = checkpoint['student']
    
    # Extract backbone weights from DDP-wrapped student
    encoder_state = {}
    
    # Try 'module.backbone.' prefix first (DDP)
    for k, v in student_state.items():
        if k.startswith('module.backbone.'):
            new_key = k.replace('module.backbone.', '')
            encoder_state[new_key] = v
    
    # Try 'backbone.' prefix (non-DDP)
    if not encoder_state:
        for k, v in student_state.items():
            if k.startswith('backbone.'):
                new_key = k.replace('backbone.', '')
                encoder_state[new_key] = v
    
    if not encoder_state:
        raise ValueError("Could not extract encoder weights from checkpoint.")
    
    # Load weights
    msg = mask_encoder.load_state_dict(encoder_state, strict=False)
    print(f"Loaded {len(encoder_state)} parameters into mask encoder")
    print(f"Load result: {msg}")
    
    if 'iteration' in checkpoint:
        print(f"Checkpoint was saved at iteration: {checkpoint['iteration']}")
    
    # Freeze encoder if requested
    if freeze:
        frozen_params = 0
        for param in mask_encoder.parameters():
            param.requires_grad = False
            frozen_params += 1
        print(f"Froze {frozen_params} parameters in mask encoder")
    
    return mask_encoder