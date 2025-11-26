"""
Submitit launcher for ADIOS-TME training on SLURM clusters.
"""

import argparse
import os
import uuid
import datetime
from pathlib import Path

import submitit

from training.trainer import train_adios_tme


def get_shared_folder() -> Path:
    """Get shared folder for logs and checkpoints."""
    p = Path(f"/data1/vanderbc/nandas1/ADIOS-TME/logs")
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_init_file():
    """Create unique init file for distributed training."""
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    """Wrapper class for submitit job."""
    def __init__(self, args):
        self.args = args

    def __call__(self):
        """Main training call."""
        self._setup_gpu_args()
        train_adios_tme(self.args)

    def checkpoint(self):
        """Checkpoint for job preemption."""
        self.args.dist_url = get_init_file().as_uri()
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        """Setup GPU arguments from submitit environment."""
        job_env = submitit.JobEnvironment()
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    """Main submitit launcher."""
    
    # ========== SLURM Configuration ==========
    ngpus = 4
    nodes = 1
    timeout_min = 10000
    partition = "vanderbc_gpu"
    constraint = "h100"
    
    # ========== Architecture Configuration ==========
    arch = 'vit_base'
    patch_size = 16
    embed_dim = 768
    num_heads = 12
    depth = 12
    mlp_ratio = 4.0

    # ========== TME Head Configuration ==========
    tme_hidden_dim = 2048
    tme_output_dim = 256
    tme_layers = 3

    # ========== Mask Model Configuration ==========
    mask_model_type = 'adios'  # <vit_unet preset> 'vit_unet'
    num_masks = 3
    crops_per_mask = 0
    mask_update_freq = 1
    mask_dropout = 0.0  # <vit_unet preset> 0.2
    mask_encoder_dim = 192  # <vit_unet preset> 384
    mask_encoder_depth = 12
    mask_encoder_checkpoint = None  # <vit_unet preset> '/data1/vanderbc/nandas1/TCGA_Dinov2_ViT-B_run2/logs/checkpoint.pth'
    freeze_mask_encoder = False  # <vit_unet preset> True

    # ========== Reconstructor Configuration ==========
    use_reconstructor = False  # <vit_unet preset> True
    reconstructor_encoder_dim = 192  # <vit_unet preset> 384
    reconstructor_encoder_depth = 12
    beta_reconstruction = 1.0
    reconstructor_update_freq = 1

    # ========== ADIOS Loss Configuration ==========
    sparsity_penalty_type = 'inverse_sin'  # <vit_unet preset> 'sinh_squared'
    alpha_sparsity = 0.1
    initial_temp = 0.1
    final_temp = 0.1
    reconstruction_weight = 0.1

    # ========== Optimizer Configuration ==========
    optimizer_type = 'sgd'  # <vit_unet preset> 'adamw'
    use_lars = True  # <vit_unet preset> False
    lars_eta = 0.02
    momentum = 0.9
    exclude_bias_n_norm_lars = True
    mask_lr_ratio = 1.0

    # ========== Training Configuration ==========
    batch_size_per_gpu = 64  # <vit_unet preset> 64
    total_iterations = 300_000
    warmup_iterations = 10_000
    
    effective_batch = batch_size_per_gpu * ngpus * nodes
    lr = 0.4 * (effective_batch / 256.0)  # <vit_unet preset> 5e-5 * (effective_batch / 256.0)
    min_lr = 1e-5  # <vit_unet preset> 1e-6
    weight_decay = 0.0  # <vit_unet preset> 0.04
    clip_grad = 1.0

    # ========== Training Setup ==========
    use_fp16 = True
    grad_checkpointing = True
    num_workers = 8

    # ========== Logging & Checkpointing ==========
    save_freq = 2_000
    log_freq = 10
    viz_freq = 500

    # ========== Dataset Configuration ==========
    data_path = "/data1/vanderbc/foundation_model_training_images/TCGA"
    img_size = 224

    # ========== Build args namespace ==========
    args = argparse.Namespace(
        # Architecture
        arch=arch,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth,
        mlp_ratio=mlp_ratio,
        # TME Head
        tme_hidden_dim=tme_hidden_dim,
        tme_output_dim=tme_output_dim,
        tme_layers=tme_layers,
        # Mask Model
        mask_model_type=mask_model_type,
        num_masks=num_masks,
        crops_per_mask=crops_per_mask,
        mask_update_freq=mask_update_freq,
        mask_dropout=mask_dropout,
        mask_encoder_dim=mask_encoder_dim,
        mask_encoder_depth=mask_encoder_depth,
        mask_encoder_checkpoint=mask_encoder_checkpoint,
        freeze_mask_encoder=freeze_mask_encoder,
        # Reconstructor
        use_reconstructor=use_reconstructor,
        reconstructor_encoder_dim=reconstructor_encoder_dim,
        reconstructor_encoder_depth=reconstructor_encoder_depth,
        beta_reconstruction=beta_reconstruction,
        reconstructor_update_freq=reconstructor_update_freq,
        # Loss
        sparsity_penalty_type=sparsity_penalty_type,
        alpha_sparsity=alpha_sparsity,
        initial_temp=initial_temp,
        final_temp=final_temp,
        reconstruction_weight=reconstruction_weight,
        # Optimizer
        optimizer_type=optimizer_type,
        use_lars=use_lars,
        lars_eta=lars_eta,
        momentum=momentum,
        exclude_bias_n_norm_lars=exclude_bias_n_norm_lars,
        mask_lr_ratio=mask_lr_ratio,
        # Training
        batch_size_per_gpu=batch_size_per_gpu,
        total_iterations=total_iterations,
        warmup_iterations=warmup_iterations,
        lr=lr,
        min_lr=min_lr,
        weight_decay=weight_decay,
        clip_grad=clip_grad,
        use_fp16=use_fp16,
        grad_checkpointing=grad_checkpointing,
        num_workers=num_workers,
        # Logging
        save_freq=save_freq,
        log_freq=log_freq,
        viz_freq=viz_freq,
        # Data
        data_path=data_path,
        img_size=img_size,
        # Distributed (populated later)
        world_size=ngpus * nodes,
        rank=0,
        seed=42,
        dist_url=None,
        local_rank=0,
        gpu=0,
        output_dir=None,
    )

    # ========== Setup output directory ==========
    args.output_dir = str(get_shared_folder())
    args.dist_url = get_init_file().as_uri()

    # ========== Setup executor ==========
    executor = submitit.AutoExecutor(folder=args.output_dir, slurm_max_num_timeout=30)

    job_name = f"adios_tme_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    executor.update_parameters(
        mem_gb=256,
        gpus_per_node=ngpus,
        tasks_per_node=ngpus,
        cpus_per_task=8,
        nodes=nodes,
        timeout_min=timeout_min,
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        slurm_gres=f'gpu:{ngpus}',
        slurm_constraint=constraint,
        slurm_setup=[
            f'export OMP_NUM_THREADS=8',
            f'export NCCL_SOCKET_IFNAME=ib,bond',
            f'export MASTER_PORT=23468',
            f'export WORLD_SIZE={ngpus * nodes}',
        ]
    )
    
    executor.update_parameters(name=job_name)

    # ========== Save configuration ==========
    config_file = os.path.join(args.output_dir, f"{job_name}_config.txt")
    with open(config_file, "w") as f:
        f.write("ADIOS-TME Training Configuration\n")
        f.write("=" * 80 + "\n\n")
        for arg, value in sorted(vars(args).items()):
            f.write(f"{arg}: {value}\n")

    # ========== Submit job ==========
    trainer = Trainer(args)
    job = executor.submit(trainer)

    # ========== Print summary ==========
    print("\n" + "=" * 80)
    print("JOB SUBMITTED")
    print("=" * 80)
    print(f"Job ID: {job.job_id}")
    print(f"Output: {args.output_dir}")
    print()
    print(f"Mask model: {mask_model_type}")
    print(f"Optimizer: {optimizer_type}" + (" + LARS" if use_lars else ""))
    print(f"Batch size: {batch_size_per_gpu} per GPU × {ngpus * nodes} GPUs = {effective_batch}")
    print(f"LR: {lr:.6f}")
    print(f"Reconstructor: {use_reconstructor}")
    print("=" * 80)
    print()
    print(f"Monitor: tail -f {args.output_dir}/{job.job_id}_0_log.out")
    print(f"Cancel:  scancel {job.job_id}")
    print()


if __name__ == "__main__":
    main()