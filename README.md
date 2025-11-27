# ADIOS-TME: Adversarial Masking Implementation With Extension to Multi-Node and CellViT Decoder

A PyTorch implementation of adversarial masking for self-supervised representation learning in computational pathology. This codebase adapts the ADIOS (Adversarial Inference-Occlusion Self-supervision) framework from [Shi et al. (ICML 2022)](https://arxiv.org/abs/2201.13100) for histopathology image analysis.

## Overview

This framework learns visual representations by simultaneously training:

1. **Student Encoder**: A Vision Transformer that learns to produce consistent embeddings between original and masked images
2. **Mask Model**: A network that adversarially generates semantic masks to maximally disrupt the encoder's representations

The adversarial interplay encourages the encoder to learn robust, semantically meaningful features while the mask model discovers tissue structures relevant to downstream pathology tasks.

## Key Features

- **Modern ViT Architecture**: XFormers memory-efficient attention with SwiGLU MLP, register tokens, and sequence packing for efficient multi-crop training
- **Dual Mask Model Support**: Choice between the original ADIOS UNet architecture and a ViT-UNet variant with skip connections
- **Distributed Training**: Full support for multi-GPU and multi-node training via PyTorch DDP
- **Mixed Precision**: BFloat16 training with gradient scaling
- **Efficient Data Pipeline**: Memory-efficient iterable dataset with proper sharding, corruption handling, and zip archive support for large-scale pathology datasets
- **LARS Optimizer**: Optional Layer-wise Adaptive Rate Scaling for improved large-batch training

## Installation

### Requirements

```bash
pip install torch torchvision
pip install xformers
pip install timm
pip install matplotlib
pip install numpy
pip install pillow
```

### Hardware Requirements

- CUDA-capable GPU with at least 16GB VRAM (24GB+ recommended for ViT-Base)
- For multi-node training: InfiniBand or high-speed ethernet interconnect

## Project Structure

```
├── configs/
│   └── config.py              # Training configuration and argument parsing
├── data/
│   └── datasets.py            # Pathology dataset with zip archive support
├── losses/
│   └── adios_loss.py          # SimCLR-based adversarial loss with sparsity penalties
├── models/
│   ├── UNet.py                # Original ADIOS mask model architecture
│   ├── tme_model.py           # Student model wrapper with TME projection head
│   └── vision_transformer/
│       ├── modern_vit.py      # ViT with XFormers attention and sequence packing
│       └── auxiliary_models.py # TME head and ViT-UNet mask model
├── training/
│   ├── trainer.py             # Main training loop
│   ├── helpers.py             # Training utilities and multi-crop processing
│   └── lars.py                # LARS optimizer wrapper
├── visualizations/
│   ├── mask_viz.py            # Mask visualization utilities
│   ├── plot_and_save_loss.py  # Training curve visualization
│   └── plot_nuclei_channel_benchmark.py  # Nuclei segmentation evaluation
├── main_train.py              # Entry point for local training
├── run_with_submitit.py       # SLURM cluster submission script
└── utils.py                   # Distributed training and general utilities
```

## Quick Start

### Single GPU Training

```bash
python main_train.py \
    --data_path /path/to/your/dataset \
    --output_dir ./output \
    --batch_size_per_gpu 64 \
    --arch vit_base \
    --mask_model_type adios \
    --total_iterations 300000
```

### Multi-GPU Training (Local)

```bash
torchrun --nproc_per_node=4 main_train.py \
    --data_path /path/to/your/dataset \
    --output_dir ./output \
    --batch_size_per_gpu 64
```

### SLURM Cluster Training

Edit the configuration section in `run_with_submitit.py` and run:

```bash
python run_with_submitit.py
```

## Configuration

### Model Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--arch` | `vit_base` | Architecture size: `vit_small`, `vit_base`, `vit_large` |
| `--patch_size` | `16` | Patch size for vision transformer |
| `--embed_dim` | `768` | Embedding dimension (384/768/1024 for S/B/L) |
| `--depth` | `12` | Number of transformer blocks |
| `--num_heads` | `12` | Number of attention heads |

### Mask Model

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mask_model_type` | `vit_unet` | Mask architecture: `vit_unet` or `adios` |
| `--num_masks` | `3` | Number of semantic masks to generate |
| `--crops_per_mask` | `1` | Random crops per mask for multi-scale training |
| `--mask_update_freq` | `1` | Update mask model every N iterations |

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_size_per_gpu` | `64` | Batch size per GPU |
| `--total_iterations` | `300000` | Total training iterations |
| `--lr` | `5e-5` | Base learning rate |
| `--optimizer_type` | `adamw` | Optimizer: `adamw` or `sgd` |
| `--use_lars` | `False` | Enable LARS wrapper (recommended with SGD) |

### Loss Function

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--alpha_sparsity` | `0.1` | Weight for mask sparsity penalty |
| `--sparsity_penalty_type` | `inverse_sin` | Penalty type: `inverse_sin` or `sinh_squared` |
| `--initial_temp` | `0.2` | Initial contrastive temperature |
| `--final_temp` | `0.05` | Final contrastive temperature |

## Data Format

The dataset expects histopathology images organized in zip archives:

```
data_path/
├── slide_001.zip
│   ├── patch_001_448_x100_y200.png
│   ├── patch_002_448_x300_y400.png
│   └── ...
├── slide_002.zip
└── ...
```

A dataset index file (`dataset_index.pkl`) should be pre-generated listing all zip files and their contents. The dataset filters for 448px patches by default (configurable in `datasets.py`).

### Normalization

Default normalization values are tuned for H&E stained histopathology images:
- Mean: `(0.6816, 0.5640, 0.7232)`
- Std: `(0.1617, 0.1714, 0.1389)`

## Training Dynamics

The training alternates between two phases:

1. **Student Update**: The encoder minimizes the contrastive loss between original and masked image embeddings
2. **Mask Update**: The mask model maximizes the same contrastive loss (adversarial) plus a sparsity regularizer

The sparsity penalty encourages approximately 50% mask activation, preventing trivial solutions (all-zero or all-one masks).

## Checkpointing

Checkpoints are saved to `output_dir` with the following structure:

- `checkpoint.pth`: Latest checkpoint for resumption
- `checkpoint_iter_XXXXXXXX.pth`: Iteration-specific checkpoints

Each checkpoint contains:
- Student and mask model state dicts
- Optimizer states
- Training iteration
- Configuration arguments

## Visualization

### Mask Visualization

During training, mask visualizations are saved every `viz_freq` iterations to `output_dir/visualizations/masks/`.

### Training Curves

```bash
python visualizations/plot_and_save_loss.py --base_path ./logs/*_0_log.out
```

### Nuclei Channel Benchmark

Evaluate mask model checkpoints on nuclei segmentation:

```bash
python visualizations/plot_nuclei_channel_benchmark.py
```

## Citation

If you use this code, please cite the original ADIOS paper:

```bibtex
@inproceedings{shi2022adversarial,
  title={Adversarial Masking for Self-Supervised Learning},
  author={Shi, Yuge and Siddharth, N and Torr, Philip and Kosiorek, Adam R},
  booktitle={International Conference on Machine Learning},
  pages={20026--20040},
  year={2022},
  organization={PMLR}
}
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

This implementation builds upon:
- [ADIOS](https://github.com/YugeTen/adios) - Original adversarial masking framework
- [DINOv2](https://github.com/facebookresearch/dinov2) - Modern ViT architecture components
- [XFormers](https://github.com/facebookresearch/xformers) - Memory-efficient attention