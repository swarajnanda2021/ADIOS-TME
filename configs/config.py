"""
Configuration for ADIOS-TME training.
Clean configuration without DINOv2 complexity.
"""

import argparse
import utils


def get_args_parser():
    """
    Create argument parser for ADIOS-TME training.
    """
    parser = argparse.ArgumentParser('ADIOS-TME Training', add_help=False)
    
    # ========== Model Architecture ==========
    parser.add_argument('--arch', default='vit_base', type=str,
                        choices=['vit_small', 'vit_base', 'vit_large'],
                        help='Architecture size')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='Patch size for vision transformer')
    parser.add_argument('--embed_dim', default=768, type=int,
                        help='Embedding dimension (384=small, 768=base, 1024=large)')
    parser.add_argument('--num_heads', default=12, type=int,
                        help='Number of attention heads (6=small, 12=base, 16=large)')
    parser.add_argument('--depth', default=12, type=int,
                        help='Number of transformer blocks (12=small/base, 24=large)')
    parser.add_argument('--mlp_ratio', default=4.0, type=float,
                        help='MLP hidden dim ratio')
    
    # ========== TME Head ==========
    parser.add_argument('--tme_hidden_dim', default=2048, type=int,
                        help='TME head hidden dimension')
    parser.add_argument('--tme_output_dim', default=256, type=int,
                        help='TME head output dimension')
    parser.add_argument('--tme_layers', default=3, type=int,
                        help='Number of layers in TME head')
    
    # ========== Mask Model ==========
    parser.add_argument('--num_masks', default=3, type=int,
                        help='Number of semantic masks to generate')
    parser.add_argument('--mask_encoder_dim', default=192, type=int,
                        help='Mask encoder embedding dimension')
    parser.add_argument('--mask_encoder_depth', default=12, type=int,
                        help='Mask encoder depth')
    parser.add_argument('--mask_update_freq', default=5, type=int,
                        help='Update mask model every N iterations')
    parser.add_argument('--mask_dropout', default=0.2, type=float,
                        help='Dropout rate in mask decoder')
    
    # ========== ADIOS Loss ==========
    parser.add_argument('--alpha_sparsity', default=0.1, type=float,
                        help='Weight for sparsity penalty on masks')
    parser.add_argument('--initial_temp', default=0.2, type=float,
                        help='Initial temperature for contrastive loss')
    parser.add_argument('--final_temp', default=0.05, type=float,
                        help='Final temperature for contrastive loss')
    parser.add_argument('--reconstruction_weight', default=0.1, type=float,
                        help='Weight for reconstruction reward')
    
    # ========== Training ==========
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--total_iterations', default=300000, type=int,
                        help='Total number of training iterations')
    parser.add_argument('--warmup_iterations', default=10000, type=int,
                        help='Number of warmup iterations')
    parser.add_argument('--lr', default=5e-5, type=float,
                        help='Base learning rate')
    parser.add_argument('--min_lr', default=1e-6, type=float,
                        help='Minimum learning rate')
    parser.add_argument('--weight_decay', default=0.04, type=float,
                        help='Weight decay')
    parser.add_argument('--clip_grad', default=1.0, type=float,
                        help='Gradient clipping value (0 = no clipping)')
    parser.add_argument('--use_fp16', default=True, type=utils.bool_flag,
                        help='Use mixed precision training')
    parser.add_argument('--grad_checkpointing', default=True, type=utils.bool_flag,
                        help='Enable gradient checkpointing')
    
    # ========== Data ==========
    parser.add_argument('--data_path', 
                        default='/data1/vanderbc/foundation_model_training_images/TCGA',
                        type=str, help='Path to training data')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of data loading workers per GPU')
    parser.add_argument('--img_size', default=224, type=int,
                        help='Input image size')
    
    # ========== Logging & Checkpointing ==========
    parser.add_argument('--output_dir', default='./output', type=str,
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--save_freq', default=2000, type=int,
                        help='Save checkpoint every N iterations')
    parser.add_argument('--log_freq', default=10, type=int,
                        help='Log metrics every N iterations')
    parser.add_argument('--viz_freq', default=500, type=int,
                    help='Visualize masks every N iterations')
    
    # ========== Distributed ==========
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--rank', default=0, type=int,
                        help='Global rank of the process')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed')
    parser.add_argument('--dist_url', default='env://', type=str,
                        help='URL for distributed training')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='Local rank (set automatically)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use')
    
    return parser