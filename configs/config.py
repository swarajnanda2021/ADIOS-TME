"""
Configuration for semantic grounding training.
"""

import argparse
import utils


def get_args_parser():
    parser = argparse.ArgumentParser('Semantic Grounding Training', add_help=False)
    
    # ========== Model Architecture ==========
    parser.add_argument('--arch', default='vit_base', type=str,
                        choices=['vit_small', 'vit_base', 'vit_large'])
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--embed_dim', default=768, type=int,
                        help='384=small, 768=base, 1024=large')
    parser.add_argument('--num_heads', default=12, type=int,
                        help='6=small, 12=base, 16=large')
    parser.add_argument('--depth', default=12, type=int,
                        help='12=small/base, 24=large')
    parser.add_argument('--mlp_ratio', default=4.0, type=float)
    
    # ========== Mask Model ==========
    parser.add_argument('--num_masks', default=3, type=int,
                        help='Number of semantic masks (nuclei, background, stroma)')
    parser.add_argument('--mask_dropout', default=0.2, type=float,
                        help='Dropout in mask decoder')
    
    # ========== Semantic Grounding ==========
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to pretrained ViT checkpoint')
    parser.add_argument('--template_path', type=str, required=True,
                        help='Path to template feature bank pickle')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for contrastive loss')
    parser.add_argument('--top_k_patches', type=int, default=20,
                        help='Number of patches to select per mask')
    parser.add_argument('--diversity_weight', type=float, default=0.1,
                        help='Weight for inter-mask diversity')
    parser.add_argument('--sparsity_weight', type=float, default=0.1,
                        help='Weight for sparsity regularization')
    
    # ========== Training ==========
    parser.add_argument('--batch_size_per_gpu', default=64, type=int)
    parser.add_argument('--total_iterations', default=100000, type=int)
    parser.add_argument('--warmup_iterations', default=5000, type=int)
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Learning rate (only trains decoder)')
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=0.04, type=float)
    parser.add_argument('--clip_grad', default=1.0, type=float)
    parser.add_argument('--use_fp16', default=True, type=utils.bool_flag)
    
    # ========== Data ==========
    parser.add_argument('--data_path', 
                        default='/data1/vanderbc/foundation_model_training_images/TCGA',
                        type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--img_size', default=224, type=int)
    
    # ========== Logging & Checkpointing ==========
    parser.add_argument('--output_dir', default='./output', type=str)
    parser.add_argument('--save_freq', default=2000, type=int)
    parser.add_argument('--log_freq', default=10, type=int)
    parser.add_argument('--viz_freq', default=500, type=int)
    
    # ========== Distributed ==========
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--dist_url', default='env://', type=str)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    
    return parser