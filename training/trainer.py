"""
STEGO-style semantic segmentation trainer.
"""

import os
import torch
import torch.nn as nn
from pathlib import Path

import utils
from models.vision_transformer.modern_vit import VisionTransformer
from models.vision_transformer.auxiliary_models import MaskModel
from data.datasets import ADIOSPathologyDataset
from losses.stego_loss import STEGOLossWithRegularizers
from training.helpers import worker_init_fn, save_iteration_masks_efficient
from utils_knn import KNNIndex


def create_frozen_encoder(checkpoint_path, args):
    """Create and freeze ViT encoder from checkpoint."""
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
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'student' in checkpoint:
        state_dict = checkpoint['student']
    else:
        state_dict = checkpoint
    
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Extract backbone if wrapped
    backbone_keys = [k for k in state_dict.keys() if k.startswith('backbone.')]
    if backbone_keys:
        print(f"  Detected wrapped model: extracting backbone weights")
        state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() 
                     if k.startswith('backbone.')}
    
    msg = encoder.load_state_dict(state_dict, strict=False)
    print(f"  Loaded with message: {msg}")
    
    # Freeze
    for param in encoder.parameters():
        param.requires_grad = False
    
    encoder.eval()
    encoder.cuda()
    
    return encoder


def create_mask_model(frozen_encoder, args):
    """Create mask model with frozen encoder backbone."""
    mask_model = MaskModel(
        encoder=frozen_encoder,
        num_masks=args.num_masks,
        encoder_dim=args.embed_dim,
        drop_rate=0.2
    )
    
    # Verify encoder is frozen
    for name, param in mask_model.encoder.named_parameters():
        assert not param.requires_grad, f"Encoder param {name} is not frozen!"
    
    trainable = sum(p.numel() for p in mask_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in mask_model.parameters())
    print(f"Mask model: {trainable:,} trainable / {total:,} total parameters")
    
    mask_model.cuda()
    
    return mask_model


def extract_features(encoder, images):
    """
    Extract dense spatial features from frozen encoder.
    
    Args:
        encoder: Frozen ViT
        images: [B, 3, H, W]
        
    Returns:
        features: [B, D, H_feat, W_feat]
    """
    with torch.no_grad():
        layer_features = encoder.get_intermediate_layers(images)
        features = layer_features[-1]  # [B, N+5, D]
        
        # Extract patch tokens (remove CLS + registers)
        patch_features = features[:, 5:, :]  # [B, N_patches, D]
        
        # Reshape to spatial
        B, N, D = patch_features.shape
        H_feat = W_feat = int(N ** 0.5)
        features_spatial = patch_features.reshape(B, H_feat, W_feat, D).permute(0, 3, 1, 2)
        
    return features_spatial


def extract_segmentation_codes(mask_model, images):
    """
    Extract learned segmentation codes (before softmax).
    
    The codes are what we want to cluster, not the masks themselves.
    
    Args:
        mask_model: MaskModel
        images: [B, 3, H, W]
        
    Returns:
        seg_codes: [B, K, H, W] continuous codes
        masks: [B, K, H, W] softmax-normalized masks
    """
    output = mask_model(images)
    masks = output['masks']  # [B, K, H, W] - already softmaxed
    
    # For STEGO, we need the codes BEFORE softmax
    # The masks are just for visualization and regularization
    # We'll use the masks as our "codes" since your MaskModel already applies softmax
    # Ideally, you'd modify MaskModel to return pre-softmax logits
    
    # WORKAROUND: Use masks as codes (they're already normalized)
    seg_codes = masks
    
    return seg_codes, masks


def train_semantic_grounding(args):
    """Main training function with STEGO loss."""
    # ============ Setup ============
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("Starting STEGO-style training")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # ============ Create models ============
    frozen_encoder = create_frozen_encoder(args.checkpoint_path, args)
    mask_model = create_mask_model(frozen_encoder, args)
    
    # Wrap in DDP if multi-GPU
    if args.world_size > 1:
        mask_model = nn.parallel.DistributedDataParallel(
            mask_model,
            device_ids=[args.gpu],
            find_unused_parameters=False,
        )
        mask_model._set_static_graph()
    
    # ============ Create STEGO loss ============
    stego_loss = STEGOLossWithRegularizers(
        lambda_self=args.lambda_self,
        lambda_knn=args.lambda_knn,
        lambda_rand=args.lambda_rand,
        lambda_diversity=args.lambda_diversity,
        lambda_sparsity=args.lambda_sparsity,
        b_self=args.b_self,
        b_knn=args.b_knn,
        b_rand=args.b_rand,
    ).cuda()
    
    # ============ Create optimizer ============
    optimizer = torch.optim.AdamW(
        [p for p in mask_model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
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
    
    # Use sampler for DDP
    if args.world_size > 1:
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
        )
    else:
        sampler = None
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size_per_gpu,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
        shuffle=(sampler is None),
    )
    
    # ============ Build KNN index (optional, can skip initially) ============
    knn_index = None
    if args.use_knn:
        knn_path = os.path.join(args.output_dir, 'knn_index.pkl')
        
        if os.path.exists(knn_path):
            print(f"Loading KNN index from {knn_path}")
            knn_index = KNNIndex(k=args.knn_k)
            knn_index.load(knn_path)
        else:
            print("Building KNN index (this will take a while)...")
            knn_index = KNNIndex(k=args.knn_k)
            knn_index.build(data_loader, frozen_encoder, save_path=knn_path)
    
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
    
    data_iter = iter(data_loader)
    
    while iteration < args.total_iterations:
        try:
            images = next(data_iter)
        except StopIteration:
            # Reset iterator
            data_iter = iter(data_loader)
            images = next(data_iter)
        
        images = images.cuda(non_blocking=True)
        
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_schedule[iteration]
        
        # ============ Forward pass ============
        mask_model.train()
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
            # Extract frozen features
            features = extract_features(frozen_encoder, images)
            
            # Extract learned segmentation codes
            seg_codes, masks = extract_segmentation_codes(mask_model, images)
            
            # Optionally get KNN pairs
            knn_features = None
            knn_seg_codes = None
            
            if knn_index is not None and args.use_knn:
                # Get KNN indices for this batch
                # Note: This is simplified - in practice you'd track dataset indices
                # For now, we'll skip KNN and just use self + random
                pass
            
            # Compute STEGO loss
            loss, metrics = stego_loss(
                features=features,
                seg_codes=seg_codes,
                masks=masks,
                knn_features=knn_features,
                knn_seg_codes=knn_seg_codes,
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
        metric_logger.update(**metrics)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        
        if iteration % 10 == 0 and utils.is_main_process():
            print(f"Iteration {iteration}/{args.total_iterations}: {metric_logger}")
        
        # ============ Visualization ============
        if iteration % args.viz_freq == 0 and utils.is_main_process():
            with torch.no_grad():
                save_iteration_masks_efficient(
                    images[:4],
                    masks[:4],
                    iteration,
                    os.path.join(args.output_dir, 'visualizations', 'masks'),
                    num_samples=4,
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