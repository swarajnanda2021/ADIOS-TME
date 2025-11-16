"""
ADIOS-TME training loop 

Key optimizations:
1. Proper fp16/bfloat16 with gradient scaler
2. Mask caching (compute once per iteration)
3. Multi-crop implementation
4. Batched forward passes
5. Memory-efficient crop processing
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from pathlib import Path
import json
import random

import utils
from models.tme_model import TMEModel
from models.vision_transformer.modern_vit import VisionTransformer
from models.vision_transformer.auxiliary_models import TMEHead, MaskModel_SpectralNorm, ReconstructorModel
from data.datasets import ADIOSPathologyDataset
from losses.adios_loss import ADIOSLoss

from visualizations import safe_visualization_wrapper

from .helpers import (
    save_iteration_masks_efficient,
    worker_init_fn,
    setup_ddp_model,
    apply_crops_to_masked_images,
    process_student_with_cached_masks_and_crops,
)


def train_adios_tme(args):
    """
    Main training function for ADIOS-TME with optimizations.
    """
    # ============ Setup ============
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("Starting ADIOS-TME training (OPTIMIZED)")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # ============ Create models ============
    
    # Student encoder
    student_backbone = VisionTransformer(
        img_size=224,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.4,
        num_register_tokens=4,
    )
    
    # TME head
    tme_head = TMEHead(
        in_dim=args.embed_dim,
        hidden_dim=2048,
        bottleneck_dim=256,
        use_bn=False
    )
    
    # Combined student model
    student = TMEModel(
        backbone=student_backbone,
        tme_head=tme_head
    )
    
    # Mask model (ViT-Tiny)
    mask_encoder = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        drop_path_rate=0.1,
        num_register_tokens=4,
    )
    
    mask_model = MaskModel_SpectralNorm(
        encoder=mask_encoder,
        num_masks=args.num_masks,
        encoder_dim=192,
        drop_rate=0.2
    )
    
    # Reconstructor (ViT-Tiny)
    reconstructor_encoder = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        drop_path_rate=0.1,
        num_register_tokens=4,
    )
    
    reconstructor = ReconstructorModel(
        encoder=reconstructor_encoder,
        encoder_dim=384,
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
    
    # Set static graph for efficiency
    student._set_static_graph()
    mask_model._set_static_graph()
    reconstructor._set_static_graph()
    
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

    lr_schedule = utils.cosine_scheduler(
        base_value=args.lr,
        final_value=args.min_lr,
        total_iters=args.total_iterations,
        warmup_iters=args.warmup_iterations,
    )
    
    # ============ Setup fp16/bfloat16 scaler ============
    fp16_scaler = None
    use_amp = args.use_fp16
    amp_dtype = torch.bfloat16  # Use bfloat16 for better stability
    
    if use_amp:
        fp16_scaler = torch.cuda.amp.GradScaler()
        print(f"Using automatic mixed precision with {amp_dtype}")
    
    # ============ Create dataset ============
    dataset = ADIOSPathologyDataset(
        data_path=args.data_path,
        index_file="dataset_index.pkl",
        img_size=args.img_size,
        max_samples=None,
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
    
    # Resume from checkpoint if exists
    to_restore = {"iteration": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        mask_model=mask_model,
        reconstructor=reconstructor,
        student_optimizer=student_optimizer,
        mask_optimizer=mask_optimizer,
        reconstructor_optimizer=reconstructor_optimizer,
        fp16_scaler=fp16_scaler,
    )
    start_iteration = to_restore["iteration"]

    # ============ Training loop ============
    iteration = start_iteration
    metric_logger = utils.IterationMetricLogger(total_iterations=args.total_iterations)
    
    # Multi-crop configuration
    crops_per_mask = getattr(args, 'crops_per_mask', 2)  # Default: 2 crops per mask
    print(f"Using {crops_per_mask} crops per mask for multi-scale training")
    
    for data in data_loader:
        if iteration >= args.total_iterations:
            break
        
        # Get original image
        original_image = data.cuda(non_blocking=True)
        batch_size = original_image.shape[0]

        # Update learning rate
        for param_group in student_optimizer.param_groups:
            param_group['lr'] = lr_schedule[iteration]
        
        # ============ CRITICAL OPTIMIZATION: Cache masks once per iteration ============
        # This dramatically reduces memory usage by avoiding redundant forward passes
        mask_model.eval()
        with torch.no_grad():
            # Generate masks ONCE and cache them
            cached_mask_output = mask_model(original_image)
            cached_masks = cached_mask_output["masks"].clone()  # Clone to ensure no graph retention
            
            crop_params = None
        
        # Clean up the original output to free memory
        del cached_mask_output
        torch.cuda.empty_cache()

        # ============ Phase 1: Train Student (Inpainter) ============
        student.train()
        student_optimizer.zero_grad()

        with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
            # Use the efficient cached masks + crops function
            student_loss, student_metrics = process_student_with_cached_masks_and_crops(
                student=student,
                cached_masks=cached_masks,
                original_image=original_image,
                crop_params=crop_params,
                K=crops_per_mask,
                current_iteration=iteration,
                adios_loss=adios_loss,
                num_masks=args.num_masks
            )
        
        # Backward with gradient scaling
        if fp16_scaler is not None:
            fp16_scaler.scale(student_loss).backward()
            
            if args.clip_grad:
                fp16_scaler.unscale_(student_optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip_grad)
            
            fp16_scaler.step(student_optimizer)
            fp16_scaler.update()
        else:
            student_loss.backward()
            
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip_grad)
            
            student_optimizer.step()

        # ============ Phase 2 & 3: Update reconstructor and mask model ============
        mask_loss = torch.tensor(0.0)
        mask_metrics = {'similarity': 0.0, 'sparsity': 0.0}
        
        if iteration % args.mask_update_freq == 0:
            # Phase 2: Train Reconstructor
            student.eval()
            reconstructor.train()
            reconstructor_optimizer.zero_grad()

            with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
                # Create hybrid input using cached masks
                content_mask = cached_masks[:, 0:1, :, :]
                guidance_mask_g = cached_masks[:, 1:2, :, :] if cached_masks.shape[1] > 1 else torch.zeros_like(content_mask)
                guidance_mask_b = cached_masks[:, 2:3, :, :] if cached_masks.shape[1] > 2 else torch.zeros_like(content_mask)

                hybrid_input = torch.zeros_like(original_image)
                hybrid_input[:, 0:1, :, :] = original_image[:, 0:1, :, :] * content_mask
                hybrid_input[:, 1:2, :, :] = (original_image[:, 1:2, :, :] * content_mask +
                                            guidance_mask_g * (1 - content_mask))
                hybrid_input[:, 2:3, :, :] = (original_image[:, 2:3, :, :] * content_mask +
                                            guidance_mask_b * (1 - content_mask))

                reconstructed = reconstructor(hybrid_input)
                recon_loss = F.l1_loss(reconstructed, original_image)

            if fp16_scaler is not None:
                fp16_scaler.scale(recon_loss).backward()
                if args.clip_grad:
                    fp16_scaler.unscale_(reconstructor_optimizer)
                    torch.nn.utils.clip_grad_norm_(reconstructor.parameters(), args.clip_grad)
                fp16_scaler.step(reconstructor_optimizer)
                fp16_scaler.update()
            else:
                recon_loss.backward()
                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(reconstructor.parameters(), args.clip_grad)
                reconstructor_optimizer.step()

            metric_logger.update(recon_loss=recon_loss.item())
            
            # ============ Phase 3: Update Mask Model ============
            student.eval()
            mask_model.train()
            reconstructor.eval()
            mask_optimizer.zero_grad()

            with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
                # Generate FRESH masks for mask model training
                mask_output = mask_model(original_image)
                fresh_masks = mask_output['masks']
                
                # Create masked images
                masked_images = []
                for i in range(args.num_masks):
                    mask = fresh_masks[:, i:i+1, :, :]
                    masked_img = original_image * (1 - mask)
                    masked_images.append(masked_img)
                
                # Forward through student
                all_embeddings = student([original_image] + masked_images)
                original_emb = all_embeddings[0]
                masked_embs = all_embeddings[1:]
                
                # Compute adversarial loss (mask tries to maximize contrastive loss)
                # NOTE: K=0 for mask model training (no crops in adversarial phase)
                mask_loss, mask_metrics = adios_loss(
                    original_emb,
                    masked_embs,
                    masks=fresh_masks,
                    iteration=iteration,
                    forward_type='mask',
                    num_base_masks=args.num_masks,  
                    K=0  
                )
                
                # Add reconstruction reward
                with torch.no_grad():
                    # Use same randomization approach for consistency
                    test_mask_indices = list(range(args.num_masks))
                    random.shuffle(test_mask_indices)
                    
                    hybrid_test = torch.zeros_like(original_image)
                    content_mask = fresh_masks[:, test_mask_indices[0]:test_mask_indices[0]+1, :, :]
                    hybrid_test[:, 0:1, :, :] = original_image[:, 0:1, :, :] * content_mask
                    if len(test_mask_indices) > 1:
                        guidance_mask_g = fresh_masks[:, test_mask_indices[1]:test_mask_indices[1]+1, :, :]
                        hybrid_test[:, 1:2, :, :] = (original_image[:, 1:2, :, :] * content_mask +
                                                    guidance_mask_g * (1 - content_mask))
                    if len(test_mask_indices) > 2:
                        guidance_mask_b = fresh_masks[:, test_mask_indices[2]:test_mask_indices[2]+1, :, :]
                        hybrid_test[:, 2:3, :, :] = (original_image[:, 2:3, :, :] * content_mask +
                                                    guidance_mask_b * (1 - content_mask))
                    
                    reconstructed_test = reconstructor(hybrid_test)
                    recon_error = F.l1_loss(reconstructed_test, original_image)
                    
                # Combine: adversarial 
                total_mask_loss = mask_loss 

            if fp16_scaler is not None:
                fp16_scaler.scale(total_mask_loss).backward()
                if args.clip_grad:
                    fp16_scaler.unscale_(mask_optimizer)
                    torch.nn.utils.clip_grad_norm_(mask_model.parameters(), args.clip_grad)
                fp16_scaler.step(mask_optimizer)
                fp16_scaler.update()
            else:
                total_mask_loss.backward()
                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(mask_model.parameters(), args.clip_grad)
                mask_optimizer.step()

            metric_logger.update(mask_adversarial_loss=mask_loss.item())
            metric_logger.update(mask_total_loss=total_mask_loss.item())

        # ============ Visualization ============
        if iteration % args.viz_freq == 0:
            sample_image = original_image[:1]
            with torch.no_grad():
                vis_masks = mask_model(sample_image)['masks']
                
                reconstructed_images = None
                if reconstructor is not None:
                    content_mask = vis_masks[:, 0:1, :, :]
                    hybrid_input = torch.zeros_like(sample_image)
                    hybrid_input[:, 0:1, :, :] = sample_image[:, 0:1, :, :] * content_mask
                    
                    reconstructed_images = reconstructor(hybrid_input)
                
                safe_visualization_wrapper(
                    sample_image,
                    vis_masks,
                    iteration,
                    os.path.join(args.output_dir, 'visualizations', 'masks'),
                    reconstructed_images
                )

        # Clean up cached masks
        del cached_masks
        if crop_params is not None:
            del crop_params
        torch.cuda.empty_cache()

        # ============ Logging ============
        metric_logger.update(student_loss=student_loss.item())
        metric_logger.update(mask_loss=mask_loss.item())
        metric_logger.update(**student_metrics)
        metric_logger.update(**mask_metrics)
        
        if iteration % 10 == 0 and utils.is_main_process():
            print(f"Iteration {iteration}: {metric_logger}")
        
        # ============ Checkpoint ============
        if iteration % args.save_freq == 0:
            save_dict = {
                'student': student.state_dict(),
                'mask_model': mask_model.state_dict(),
                'reconstructor': reconstructor.state_dict(),
                'student_optimizer': student_optimizer.state_dict(),
                'mask_optimizer': mask_optimizer.state_dict(),
                'reconstructor_optimizer': reconstructor_optimizer.state_dict(),
                'iteration': iteration,
                'args': args,
            }
            
            if fp16_scaler is not None:
                save_dict['fp16_scaler'] = fp16_scaler.state_dict()
                
            utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        
        iteration += 1
    
    print("Training complete!")