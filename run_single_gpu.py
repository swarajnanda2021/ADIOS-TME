"""
Single GPU training script for ADIOS-TME (Semantic Grounding).
Run with: python run_single_gpu.py
"""

import os
import datetime
from pathlib import Path
from configs.config import get_args_parser
from training.trainer import train_semantic_grounding


def main():
    """Main single GPU training."""
    parser = get_args_parser()
    args = parser.parse_args()
    
    # ========== Output Directory ==========
    user = os.environ.get('USER', 'user')
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    args.output_dir = f"/data1/vanderbc/nandas1/STEGO-TME/logs"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # ========== Single GPU Setup ==========
    args.world_size = 1
    args.rank = 0
    args.gpu = 0
    args.dist_url = 'env://'
    
    # ========== Architecture Configuration ==========
    args.arch = 'vit_base'
    args.patch_size = 16
    args.embed_dim = 768
    args.num_heads = 12
    args.depth = 12
    
    # ========== Mask Model ==========
    args.num_masks = 2
    args.mask_dropout = 0.2
    
    # ========== Semantic Grounding ==========
    args.checkpoint_path = "/data1/vanderbc/nandas1/TCGA_TMEDinov3_ViT-B_B2_seqpacking/logs/checkpoint.pth"
    args.template_path = "/data1/vanderbc/nandas1/STEGO-TME/templates/pannuke_features.pkl"
    args.temperature = 0.07
    args.top_k_patches = 20
    args.diversity_weight = 0.1
    args.sparsity_weight = 0.1
    
    # ========== Training ==========
    args.batch_size_per_gpu = 128
    args.total_iterations = 100_000
    args.warmup_iterations = 5_000
    args.lr = 1e-4  # No scaling for single GPU
    args.min_lr = 1e-6
    args.weight_decay = 0.04
    args.clip_grad = 1.0
    args.use_fp16 = True
    
    # ========== Data ==========
    args.data_path = "/data1/vanderbc/foundation_model_training_images/TCGA"
    args.img_size = 224
    args.num_workers = 8
    
    # ========== Logging ==========
    args.save_freq = 2_000
    args.log_freq = 10
    args.viz_freq = 500
    
    # ========== Save Configuration ==========
    config_file = os.path.join(args.output_dir, "config.txt")
    with open(config_file, "w") as f:
        f.write("STEGO-TME Single GPU Training Configuration\n")
        f.write("=" * 80 + "\n\n")
        for arg, value in sorted(vars(args).items()):
            f.write(f"{arg}: {value}\n")
    
    # ========== Print Configuration ==========
    print("\n" + "=" * 80)
    print("STEGO-TME SINGLE GPU TRAINING")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print(f"Config file: {config_file}")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Architecture: {args.arch}")
    print(f"  Patch size: {args.patch_size}")
    print(f"  Embedding dim: {args.embed_dim}")
    print(f"  Num heads: {args.num_heads}")
    print(f"  Depth: {args.depth}")
    print()
    print(f"Semantic Grounding:")
    print(f"  Checkpoint: {args.checkpoint_path}")
    print(f"  Template bank: {args.template_path}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-K patches: {args.top_k_patches}")
    print(f"  Diversity weight: {args.diversity_weight}")
    print(f"  Sparsity weight: {args.sparsity_weight}")
    print()
    print(f"Mask Configuration:")
    print(f"  Num masks: {args.num_masks}")
    print()
    print(f"Training:")
    print(f"  Batch size: {args.batch_size_per_gpu}")
    print(f"  Learning rate: {args.lr:.2e}")
    print(f"  Total iterations: {args.total_iterations:,}")
    print(f"  Warmup iterations: {args.warmup_iterations:,}")
    print()
    print(f"Checkpointing:")
    print(f"  Save frequency: every {args.save_freq} iterations")
    print(f"  Visualization frequency: every {args.viz_freq} iterations")
    print("=" * 80 + "\n")
    
    # ========== Start Training ==========
    print("Starting training...")
    train_semantic_grounding(args)


if __name__ == "__main__":
    main()
