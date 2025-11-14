"""
Semantic grounding trainer via feature correspondence.

Architecture:
    - Frozen ViT backbone for feature extraction
    - MaskModel (frozen encoder + trainable decoder)
    - Template feature bank (nuclei + background)

Training:
    - Extract features from unmasked images
    - Generate masks
    - Select patches by mask values
    - Align selected features with templates
"""

import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import utils
from models.vision_transformer.modern_vit import VisionTransformer
from models.vision_transformer.auxiliary_models import MaskModel
from data.datasets import ADIOSPathologyDataset
from losses.correspondence_loss import FeatureCorrespondenceLoss
from training.helpers import worker_init_fn, save_iteration_masks_efficient


def load_template_bank(template_path):
    """
    Load pre-computed template features.
    
    Expected pickle format:
    {
        'nuclei_features': [N_nuclei, D] tensor,
        'background_features': [N_bg, D] tensor,
    }
    """
    print(f"Loading template bank from {template_path}")
    with open(template_path, 'rb') as f:
        data = pickle.load(f)
    
    nuclei_features = data['nuclei_features'].cuda()
    background_features = data['background_features'].cuda()
    
    print(f"  Nuclei templates: {nuclei_features.shape}")
    print(f"  Background templates: {background_features.shape}")
    
    return nuclei_features, background_features


def create_frozen_encoder(checkpoint_path, args):
    """
    Create and freeze ViT encoder from checkpoint.
    
    Args:
        checkpoint_path: Path to pretrained checkpoint
        args: Training arguments
        
    Returns:
        frozen_encoder: Frozen ViT model
    """
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
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'student' in checkpoint:
        state_dict = checkpoint['student']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load weights (strict=False in case of head mismatches)
    msg = encoder.load_state_dict(state_dict, strict=False)
    print(f"  Loaded with message: {msg}")
    
    # Freeze all parameters
    for param in encoder.parameters():
        param.requires_grad = False
    
    encoder.eval()
    encoder.cuda()
    
    print("  Encoder frozen and ready")
    
    return encoder


def create_mask_model(frozen_encoder, args):
    """
    Create mask model with frozen encoder backbone.
    
    The encoder is shared (frozen), only decoder is trainable.
    
    Args:
        frozen_encoder: Pre-loaded frozen ViT
        args: Training arguments
        
    Returns:
        mask_model: MaskModel with trainable decoder
    """
    mask_model = MaskModel(
        encoder=frozen_encoder,
        num_masks=args.num_masks,
        encoder_dim=args.embed_dim,
        drop_rate=0.2
    )
    
    # Verify encoder is frozen
    for name, param in mask_model.encoder.parameters():
        assert not param.requires_grad, f"Encoder param {name} is not frozen!"
    
    # Count trainable parameters (should only be decoder)
    trainable = sum(p.numel() for p in mask_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in mask_model.parameters())
    print(f"Mask model: {trainable:,} trainable / {total:,} total parameters")
    
    mask_model.cuda()
    
    return mask_model


def extract_features(encoder, images):
    """
    Extract features from unmasked images.
    
    Args:
        encoder: Frozen ViT encoder
        images: [B, 3, H, W]
        
    Returns:
        features: [B, D, H_feat, W_feat]
    """
    with torch.no_grad():
        # Get intermediate layers
        layer_features = encoder.get_intermediate_layers(images)
        
        # Use last layer
        features = layer_features[-1]  # [B, N+5, D] (cls + 4 registers + patches)
        
        # Extract patch tokens only (remove cls and registers)
        patch_features = features[:, 5:, :]  # [B, N_patches, D]
        
        # Reshape to spatial
        B, N, D = patch_features.shape
        H_feat = W_feat = int(N ** 0.5)
        features_spatial = patch_features.reshape(B, H_feat, W_feat, D).permute(0, 3, 1, 2)
        # [B, D, H_feat, W_feat]
        
    return features_spatial


def train_semantic_grounding(args):
    """
    Main training function with semantic grounding.
    """
    # ============ Setup ============
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("Starting semantic grounding training")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # ============ Load template bank ============
    nuclei_bank, background_bank = load_template_bank(args.template_path)
    
    # ============ Create models ============
    
    # Frozen encoder (shared for both feature extraction and mask model)
    frozen_encoder = create_frozen_encoder(args.checkpoint_path, args)
    
    # Mask model (uses same frozen encoder + trainable decoder)
    mask_model = create_mask_model(frozen_encoder, args)
    
    # Wrap in DDP if multi-GPU
    if args.world_size > 1:
        mask_model = nn.parallel.DistributedDataParallel(
            mask_model,
            device_ids=[args.gpu],
            find_unused_parameters=False,
        )
        mask_model._set_static_graph()
    
    # ============ Create loss ============
    correspondence_loss = FeatureCorrespondenceLoss(
        temperature=args.temperature,
        top_k=args.top_k_patches,
        diversity_weight=args.diversity_weight,
        sparsity_weight=args.sparsity_weight,
    ).cuda()
    
    # ============ Create optimizer ============
    # Only decoder parameters are trainable
    optimizer = torch.optim.AdamW(
        [p for p in mask_model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Learning rate schedule
    lr_schedule = utils.cosine_scheduler(
        base_value=args.lr,
        final_value=args.min_lr,
        total_iters=args.total_iterations,
        warmup_iters=args.warmup_iterations,
    )
    
    # ============ Setup mixed precision ============
    use_amp = args.use_fp16
    amp_dtype = torch.bfloat16
    fp16_scaler = None
    
    if use_amp:
        fp16_scaler = torch.cuda.amp.GradScaler()
        print(f"Using mixed precision with {amp_dtype}")
    
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
        worker_init_fn=worker_init_fn,
    )
    
    # ============ Resume from checkpoint ============
    to_restore = {"iteration": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        mask_model=mask_model,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
    )
    start_iteration = to_restore["iteration"]
    
    # ============ Training loop ============
    iteration = start_iteration
    metric_logger = utils.IterationMetricLogger(total_iterations=args.total_iterations)
    
    print("Starting training loop...")
    
    for data in data_loader:
        if iteration >= args.total_iterations:
            break
        
        # Get images
        images = data.cuda(non_blocking=True)  # [B, 3, 224, 224]
        
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_schedule[iteration]
        
        # ============ Forward pass ============
        
        # Step 1: Extract features with frozen encoder (no masking)
        features = extract_features(frozen_encoder, images)  # [B, D, 14, 14]
        
        # Step 2: Generate masks
        mask_model.train()
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
            # Forward through mask model
            mask_output = mask_model(images)
            masks = mask_output['masks']  # [B, 3, 224, 224]
            
            # Compute correspondence loss
            loss, metrics = correspondence_loss(
                features=features,
                masks=masks,
                nuclei_bank=nuclei_bank,
                background_bank=background_bank,
            )
        
        # ============ Backward pass ============
        if fp16_scaler is not None:
            fp16_scaler.scale(loss).backward()
            
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in mask_model.parameters() if p.requires_grad],
                    args.clip_grad
                )
            
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            loss.backward()
            
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in mask_model.parameters() if p.requires_grad],
                    args.clip_grad
                )
            
            optimizer.step()
        
        # ============ Logging ============
        metric_logger.update(total_loss=loss.item())
        metric_logger.update(**metrics)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        
        if iteration % 10 == 0 and utils.is_main_process():
            print(f"Iteration {iteration}/{args.total_iterations}: {metric_logger}")
        
        # ============ Visualization ============
        if iteration % args.viz_freq == 0 and utils.is_main_process():
            with torch.no_grad():
                sample_image = images[:1]
                sample_masks = masks[:1]
                
                save_iteration_masks_efficient(
                    sample_image,
                    sample_masks,
                    iteration,
                    os.path.join(args.output_dir, 'visualizations', 'masks'),
                    reconstructed_images=None,
                    num_samples=1,
                )
        
        # ============ Checkpoint ============
        if iteration % args.save_freq == 0 and utils.is_main_process():
            save_dict = {
                'mask_model': mask_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': iteration,
                'args': args,
            }
            
            if fp16_scaler is not None:
                save_dict['fp16_scaler'] = fp16_scaler.state_dict()
            
            torch.save(
                save_dict,
                os.path.join(args.output_dir, 'checkpoint.pth')
            )
            
            print(f"Saved checkpoint at iteration {iteration}")
        
        iteration += 1
    
    print("Training complete!")
