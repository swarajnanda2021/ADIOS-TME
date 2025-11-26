"""
ADIOS-TME training loop 

Key optimizations:
1. Proper fp16/bfloat16 with gradient scaler
2. Mask caching (compute once per iteration)
3. Multi-crop implementation
4. Batched forward passes
5. Memory-efficient crop processing
6. SGD + LARS support for faithful ADIOS replication
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
import datetime

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
from .lars import LARSWrapper


def create_optimizers(args, student, mask_model, reconstructor=None):
    """
    Create optimizers based on configuration.
    
    Supports:
    - AdamW (default, your approach)
    - SGD + LARS (ADIOS paper approach)
    
    Args:
        args: Configuration arguments
        student: Student model
        mask_model: Mask model
        reconstructor: Optional reconstructor model
    
    Returns:
        Tuple of (student_optimizer, mask_optimizer, reconstructor_optimizer or None)
    """
    # Calculate mask learning rate
    mask_lr = args.lr * args.mask_lr_ratio
    
    print(f"Creating optimizers:")
    print(f"  Type: {args.optimizer_type}")
    print(f"  Student LR: {args.lr}")
    print(f"  Mask LR: {mask_lr} (ratio: {args.mask_lr_ratio})")
    print(f"  Weight decay: {args.weight_decay}")
    if args.optimizer_type == 'sgd':
        print(f"  Momentum: {args.momentum}")
        print(f"  LARS: {args.use_lars}")
        if args.use_lars:
            print(f"  LARS eta: {args.lars_eta}")
    
    if args.optimizer_type == 'sgd':
        # SGD with momentum (ADIOS paper approach)
        student_optimizer = torch.optim.SGD(
            student.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        
        mask_optimizer = torch.optim.SGD(
            mask_model.parameters(),
            lr=mask_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        
        # Wrap with LARS if enabled
        if args.use_lars:
            student_optimizer = LARSWrapper(
                student_optimizer,
                eta=args.lars_eta,
                clip=True,
                exclude_bias_n_norm=args.exclude_bias_n_norm_lars
            )
            mask_optimizer = LARSWrapper(
                mask_optimizer,
                eta=args.lars_eta,
                clip=True,
                exclude_bias_n_norm=args.exclude_bias_n_norm_lars
            )
            print("  Applied LARS wrapper to optimizers")
        
        # Reconstructor optimizer (if used)
        reconstructor_optimizer = None
        if reconstructor is not None:
            reconstructor_optimizer = torch.optim.SGD(
                reconstructor.parameters(),
                lr=mask_lr,  # Same as mask model
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )
            if args.use_lars:
                reconstructor_optimizer = LARSWrapper(
                    reconstructor_optimizer,
                    eta=args.lars_eta,
                    clip=True,
                    exclude_bias_n_norm=args.exclude_bias_n_norm_lars
                )
    
    elif args.optimizer_type == 'adamw':
        # AdamW (your current approach)
        student_optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        mask_optimizer = torch.optim.AdamW(
            mask_model.parameters(),
            lr=mask_lr,
            weight_decay=args.weight_decay
        )
        
        # Reconstructor optimizer (if used)
        reconstructor_optimizer = None
        if reconstructor is not None:
            reconstructor_optimizer = torch.optim.AdamW(
                reconstructor.parameters(),
                lr=mask_lr,
                weight_decay=args.weight_decay
            )
    
    else:
        raise ValueError(f"Unknown optimizer type: {args.optimizer_type}")
    
    return student_optimizer, mask_optimizer, reconstructor_optimizer


def train_adios_tme(args):
    """
    Main training function for ADIOS-TME with optional reconstructor.
    """
    # ============ Setup ============
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    
    method_name = "ADIOS-TME (with reconstruction)" if args.use_reconstructor else "ADIOS-TME (pure adversarial)"
    print(f"Starting {method_name}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # ============ Create models ============
    
    # Student encoder (always created)
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
    
    tme_head = TMEHead(
        in_dim=args.embed_dim,
        hidden_dim=args.tme_hidden_dim,
        bottleneck_dim=args.tme_output_dim,
        use_bn=False
    )
    
    student = TMEModel(
        backbone=student_backbone,
        tme_head=tme_head
    )
    
    # Create mask model based on type
    if args.mask_model_type == 'vit_unet':
        # Your original approach: ViT encoder + U-Net decoder
        print("Using ViT-UNet mask model (your approach)")
        mask_encoder = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=args.mask_encoder_dim,
            depth=args.mask_encoder_depth,
            num_heads=args.mask_encoder_dim // 64,
            mlp_ratio=4.0,
            drop_path_rate=0.1,
            num_register_tokens=4,
        )
        
        # Load pretrained encoder if specified
        if hasattr(args, 'mask_encoder_checkpoint') and args.mask_encoder_checkpoint is not None:
            from .helpers import load_mask_encoder_from_student_checkpoint
            mask_encoder = load_mask_encoder_from_student_checkpoint(
                mask_encoder,
                args.mask_encoder_checkpoint,
                freeze=getattr(args, 'freeze_mask_encoder', True)
            )
        
        mask_model = MaskModel_SpectralNorm(
            encoder=mask_encoder,
            num_masks=args.num_masks,
            encoder_dim=args.mask_encoder_dim,
            drop_rate=0.2
        )
        
    elif args.mask_model_type == 'adios':
        # ADIOS approach: Simple U-Net on RGB images (no downsampling)
        print("Using ADIOS mask model (YugeTen et al. 2022)")
        from models.UNet import ADIOSMaskModel
        mask_model = ADIOSMaskModel(
            num_masks=args.num_masks,
            img_size=224,
            drop_rate=0.0  # ADIOS doesn't use dropout
        )
    
    # Reconstructor (conditional creation)
    reconstructor = None
    
    if args.use_reconstructor:
        print("Creating reconstructor model")
        reconstructor_encoder = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=args.reconstructor_encoder_dim,
            depth=args.reconstructor_encoder_depth,
            num_heads=args.reconstructor_encoder_dim // 64,
            mlp_ratio=4.0,
            drop_path_rate=0.1,
            num_register_tokens=4,
        )
        
        reconstructor = ReconstructorModel(
            encoder=reconstructor_encoder,
            encoder_dim=args.reconstructor_encoder_dim,
            drop_rate=0.2
        )
        
        reconstructor = reconstructor.cuda()
        reconstructor = setup_ddp_model(reconstructor, args, find_unused=False)
        reconstructor._set_static_graph()
    else:
        print("Skipping reconstructor (pure ADIOS method)")
    
    # Move to GPU and setup DDP
    student = student.cuda()
    mask_model = mask_model.cuda()
    
    student = setup_ddp_model(student, args, find_unused=False)
    mask_model = setup_ddp_model(mask_model, args, find_unused=False)
    
    student._set_static_graph()
    mask_model._set_static_graph()
    
    # ============ Create loss ============
    adios_loss = ADIOSLoss(
        alpha_sparsity=args.alpha_sparsity,
        img_size=224,
        temperature=args.initial_temp,  # Fixed temperature (no schedule)
        sparsity_penalty_type=args.sparsity_penalty_type,
    ).cuda()
    
    # ============ Create optimizers ============
    student_optimizer, mask_optimizer, reconstructor_optimizer = create_optimizers(
        args, student, mask_model, reconstructor
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
    amp_dtype = torch.bfloat16
    
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
    restore_dict = {
        'student': student,
        'mask_model': mask_model,
        'student_optimizer': student_optimizer,
        'mask_optimizer': mask_optimizer,
        'fp16_scaler': fp16_scaler,
    }
    
    # Add reconstructor to restoration if it exists
    if args.use_reconstructor:
        restore_dict['reconstructor'] = reconstructor
        restore_dict['reconstructor_optimizer'] = reconstructor_optimizer
    
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        **restore_dict
    )
    start_iteration = to_restore["iteration"]

    # ============ Training loop ============
    iteration = start_iteration
    metric_logger = utils.IterationMetricLogger(total_iterations=args.total_iterations)
    
    # Multi-crop configuration
    crops_per_mask = getattr(args, 'crops_per_mask', 0)  # Default to 0 (no crops)
    print(f"Using {crops_per_mask} crops per mask")
    
    for data in data_loader:
        if iteration >= args.total_iterations:
            break
        
        # Get original image
        original_image = data.cuda(non_blocking=True)
        batch_size = original_image.shape[0]

        # Update learning rate
        for param_group in student_optimizer.param_groups:
            param_group['lr'] = lr_schedule[iteration]
        
        # Scale mask optimizer LR proportionally
        mask_lr = lr_schedule[iteration] * args.mask_lr_ratio
        for param_group in mask_optimizer.param_groups:
            param_group['lr'] = mask_lr
        
        # ============ Cache masks once per iteration ============
        mask_model.eval()
        with torch.no_grad():
            cached_mask_output = mask_model(original_image)
            cached_masks = cached_mask_output["masks"].clone()
            crop_params = None
        
        del cached_mask_output
        torch.cuda.empty_cache()

        # ============ Phase 1: Train Student (ALWAYS) ============
        student.train()
        student_optimizer.zero_grad()

        with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
            student_loss, student_metrics = process_student_with_cached_masks_and_crops(
                student=student,
                cached_masks=cached_masks,
                original_image=original_image,
                crop_params=crop_params,
                K=crops_per_mask,
                adios_loss=adios_loss,
                num_masks=args.num_masks
            )
        
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

        # ============ Phase 2: Train Reconstructor (CONDITIONAL) ============
        recon_loss = torch.tensor(0.0).cuda()
        
        if args.use_reconstructor and iteration % args.reconstructor_update_freq == 0:
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
            
        # ============ Phase 3: Update Mask Model (ALWAYS, with conditional logic) ============
        mask_loss = torch.tensor(0.0).cuda()
        mask_metrics = {'similarity': 0.0, 'sparsity': 0.0}
        
        if iteration % args.mask_update_freq == 0:
            student.eval()
            mask_model.train()
            if args.use_reconstructor:
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
                
                # Compute adversarial loss (always computed)
                mask_loss, mask_metrics = adios_loss(
                    original_emb,
                    masked_embs,
                    masks=fresh_masks,
                    forward_type='mask',
                    num_base_masks=args.num_masks,  
                    K=args.crops_per_mask
                )
                
                # CONDITIONAL: Add reconstruction component
                if args.use_reconstructor:
                    with torch.no_grad():
                        # Use randomized mask selection for robustness
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
                        reconstruction_error = F.l1_loss(reconstructed_test, original_image)
                        
                    # Balanced objective: adversarial + sparsity + reconstruction
                    # mask_loss already contains (-adversarial + sparsity)
                    # Add reconstruction component
                    total_mask_loss = mask_loss + args.beta_reconstruction * reconstruction_error
                    
                    metric_logger.update(mask_recon_error=reconstruction_error.item())
                else:
                    # Pure adversarial (YugeTen's method)
                    # mask_loss already contains (-adversarial + sparsity)
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

        # ============ Visualization (handle reconstructor conditionally) ============
        if iteration % args.viz_freq == 0:
            sample_image = original_image[:1]
            with torch.no_grad():
                vis_masks = mask_model(sample_image)['masks']
                
                # Only compute reconstructed images if reconstructor exists
                reconstructed_images = None
                if args.use_reconstructor:
                    content_mask = vis_masks[:, 0:1, :, :]
                    hybrid_input = torch.zeros_like(sample_image)
                    hybrid_input[:, 0:1, :, :] = sample_image[:, 0:1, :, :] * content_mask
                    
                    reconstructed_images = reconstructor(hybrid_input)
                
                safe_visualization_wrapper(
                    sample_image,
                    vis_masks,
                    iteration,
                    os.path.join(args.output_dir, 'visualizations', 'masks'),
                    reconstructed_images  # Will be None if no reconstructor
                )

        # Clean up
        del cached_masks
        if crop_params is not None:
            del crop_params
        torch.cuda.empty_cache()

        # ============ Logging ============
        metric_logger.update(student_loss=student_loss.item())
        metric_logger.update(mask_loss=mask_loss.item())
        metric_logger.update(**student_metrics)
        metric_logger.update(**mask_metrics)

        if iteration % args.log_freq == 0 and utils.is_main_process():
            elapsed_time = time.time() - metric_logger.start_time
            iter_time = elapsed_time / max(iteration, 1)
            remaining_iters = args.total_iterations - iteration
            eta_seconds = iter_time * remaining_iters
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            
            space_fmt = len(str(args.total_iterations))
            progress_str = f"[{iteration:>{space_fmt}}/{args.total_iterations}]"
            
            if torch.cuda.is_available():
                memory_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                memory_str = f"max mem: {memory_mb:.0f} MB"
            else:
                memory_str = ""
            
            print(f"{progress_str}  "
                  f"eta: {eta_string}  "
                  f"{metric_logger}  "
                  f"time: {iter_time:.4f} s/it  "
                  f"{memory_str}")
        
        # ============ Checkpoint ============
        if iteration % args.save_freq == 0:
            save_dict = {
                'student': student.state_dict(),
                'mask_model': mask_model.state_dict(),
                'student_optimizer': student_optimizer.state_dict(),
                'mask_optimizer': mask_optimizer.state_dict(),
                'iteration': iteration,
                'args': args,
            }
            
            # Conditionally add reconstructor to checkpoint
            if args.use_reconstructor:
                save_dict['reconstructor'] = reconstructor.state_dict()
                save_dict['reconstructor_optimizer'] = reconstructor_optimizer.state_dict()
            
            if fp16_scaler is not None:
                save_dict['fp16_scaler'] = fp16_scaler.state_dict()
                
            utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        
        iteration += 1
    
    print("Training complete!")