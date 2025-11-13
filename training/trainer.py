"""
ADIOS-TME training loop.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from pathlib import Path
import json

import utils
from models import TMEModel, ModernViT, TMEHead, MaskModel
from models.vision_transformer.auxiliary_models import ReconstructorModel
from losses.adios_loss import ADIOSLoss
from data import DINOv2PathologyDataset
from .helpers import (
    save_iteration_masks_efficient,
    worker_init_fn,
    setup_ddp_model,
)


def train_adios_tme(args):
    """
    Main training function for ADIOS-TME.
    """
    # ============ Setup ============
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("Starting ADIOS-TME training")
    
    # ============ Create models ============
    
    # Student encoder
    student_backbone = ModernViT(
        img_size=224,
        patch_size=args.patch_size,
        embed_dim=args.embeddingdim,
        depth=args.vitdepth,
        num_heads=args.vitheads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.4,
        num_register_tokens=4,
    )
    
    # TME head
    tme_head = TMEHead(
        in_dim=args.embeddingdim,
        hidden_dim=2048,
        bottleneck_dim=256,
        use_bn=False
    )
    
    # Combined student model
    student = TMEModel(
        backbone=student_backbone,
        tme_head=tme_head
    )
    
    # Mask model (smaller encoder)
    mask_encoder = ModernViT(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        drop_path_rate=0.1,
        num_register_tokens=4,
    )
    
    mask_model = MaskModel(
        encoder=mask_encoder,
        num_masks=args.num_masks,
        encoder_dim=192,
        drop_rate=0.2
    )
    
    # Reconstructor
    reconstructor_encoder = ModernViT(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        drop_path_rate=0.1,
        num_register_tokens=4,
    )
    
    reconstructor = ReconstructorModel(
        encoder=reconstructor_encoder,
        encoder_dim=192,
        drop_rate=0.2
    )
    
    # Move to GPU
    student = student.cuda()
    mask_model = mask_model.cuda()
    reconstructor = reconstructor.cuda()
    
    # Setup DDP
    student = setup_ddp_model(student, args, find_unused=False)
    mask_model = setup_ddp_model(mask_model, args, find_unused=False)
    reconstructor = setup_ddp_model(reconstructor, args, find_unused=False)
    
    # ============ Create loss ============
    adios_loss = ADIOSLoss(
        alpha_sparsity=0.1,
        img_size=224,
        initial_temp=0.2,
        final_temp=0.05,
        total_iters=args.total_iterations,
    ).cuda()
    
    # ============ Create optimizers ============
    student_optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    mask_optimizer = torch.optim.AdamW(
        mask_model.parameters(),
        lr=args.lr * 0.1,
        weight_decay=args.weight_decay
    )
    
    reconstructor_optimizer = torch.optim.AdamW(
        reconstructor.parameters(),
        lr=args.lr * 0.1,
        weight_decay=args.weight_decay
    )
    
    # ============ Create dataset ============
    dataset = DINOv2PathologyDataset(
        base_dir=args.base_dir,
        index_file="dataset_index.pkl",
        n_standard_local_crops=0,
        global_views=2,
        worker_id=0,
        num_workers=args.num_workers,
        rank=args.gpu,
        world_size=dist.get_world_size(),
        seed=args.seed,
    )
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_init_fn
    )
    
    # ============ Training loop ============
    iteration = 0
    metric_logger = utils.IterationMetricLogger(total_iterations=args.total_iterations)
    
    for data in data_loader:
        if iteration >= args.total_iterations:
            break
            
        # Get original image (last in the batch)
        original_image = data[-1].cuda(non_blocking=True)
        
        # ========== Phase 1: Generate masks ==========
        with torch.no_grad():
            mask_output = mask_model(original_image)
            masks = mask_output['masks']
        
        # Apply masks
        masked_images = []
        for i in range(masks.shape[1]):
            mask = masks[:, i:i+1, :, :]
            masked_img = original_image * (1 - mask)
            masked_images.append(masked_img)
        
        # ========== Phase 2: Student training ==========
        student_optimizer.zero_grad()
        
        # Get embeddings
        original_emb = student(original_image)
        masked_embs = student(masked_images)
        
        # Compute ADIOS loss
        student_loss, metrics = adios_loss(
            original_emb, 
            masked_embs, 
            masks=None,  # Don't need sparsity for student
            iteration=iteration
        )
        
        student_loss.backward()
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip_grad)
        student_optimizer.step()
        
        # ========== Phase 3: Update mask model (every N iterations) ==========
        if iteration % args.mask_update_freq == 0:
            
            # Train reconstructor
            reconstructor_optimizer.zero_grad()
            
            # Create hybrid input
            hybrid_input = create_hybrid_input(original_image, masks)
            reconstructed = reconstructor(hybrid_input)
            
            # L1 loss only
            recon_loss = F.l1_loss(reconstructed, original_image)
            recon_loss.backward()
            reconstructor_optimizer.step()
            
            # Train mask model
            mask_optimizer.zero_grad()
            
            # Recompute embeddings with fresh masks
            mask_output_fresh = mask_model(original_image)
            masks_fresh = mask_output_fresh['masks']
            
            masked_images_fresh = []
            for i in range(masks_fresh.shape[1]):
                mask = masks_fresh[:, i:i+1, :, :]
                masked_img = original_image * (1 - mask)
                masked_images_fresh.append(masked_img)
            
            original_emb_fresh = student(original_image)
            masked_embs_fresh = student(masked_images_fresh)
            
            # ADIOS loss with sparsity
            mask_loss, _ = adios_loss(
                original_emb_fresh,
                masked_embs_fresh,
                masks=masks_fresh,
                iteration=iteration
            )
            
            # Add reconstruction reward
            with torch.no_grad():
                hybrid_test = create_hybrid_input(original_image, masks_fresh)
                reconstructed_test = reconstructor(hybrid_test)
                recon_error = F.l1_loss(reconstructed_test, original_image)
            
            reconstruction_reward = 1.0 / (1.0 + recon_error)
            mask_loss = mask_loss - 0.1 * reconstruction_reward
            
            mask_loss.backward()
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(mask_model.parameters(), args.clip_grad)
            mask_optimizer.step()
            
            metric_logger.update(mask_loss=mask_loss.item())
            metric_logger.update(recon_loss=recon_loss.item())
            metric_logger.update(recon_reward=reconstruction_reward.item())
        
        # ========== Logging ==========
        metric_logger.update(student_loss=student_loss.item())
        metric_logger.update(**metrics)
        
        if iteration % 10 == 0 and utils.is_main_process():
            print(f"Iteration {iteration}: {metric_logger}")
        
        # ========== Checkpoint ==========
        if iteration % args.save_checkpoint_freq == 0:
            save_dict = {
                'student': student.state_dict(),
                'mask_model': mask_model.state_dict(),
                'reconstructor': reconstructor.state_dict(),
                'iteration': iteration,
                'args': args,
            }
            utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        
        iteration += 1
    
    print("Training complete!")


def create_hybrid_input(images, masks):
    """Create hybrid input for reconstructor."""
    B, num_masks, H, W = masks.shape
    
    # Use first mask as content, others as guidance
    content_mask = masks[:, 0:1, :, :]
    
    hybrid = torch.zeros_like(images)
    hybrid[:, 0, :, :] = images[:, 0, :, :] * content_mask.squeeze(1)
    
    if num_masks > 1:
        guidance_mask = masks[:, 1:2, :, :]
        hybrid[:, 1, :, :] = (images[:, 1, :, :] * content_mask.squeeze(1) + 
                              guidance_mask.squeeze(1) * (1 - content_mask.squeeze(1)))
    
    if num_masks > 2:
        guidance_mask = masks[:, 2:3, :, :]
        hybrid[:, 2, :, :] = (images[:, 2, :, :] * content_mask.squeeze(1) + 
                              guidance_mask.squeeze(1) * (1 - content_mask.squeeze(1)))
    
    return hybrid