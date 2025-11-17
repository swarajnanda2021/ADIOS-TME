"""
Submitit launcher for ADIOS-TME training on SLURM clusters.
"""

import argparse
import os
import uuid
import datetime
from pathlib import Path

import submitit

from configs.config import get_args_parser
from training.trainer import train_adios_tme


def parse_args():
    """Parse submitit and training arguments."""
    parser = argparse.ArgumentParser(
        "Submitit for ADIOS-TME", 
        parents=[get_args_parser()]
    )
    
    # Submitit specific arguments
    parser.add_argument("--ngpus", default=4, type=int, 
                        help="Number of GPUs per node")
    parser.add_argument("--nodes", default=1, type=int, 
                        help="Number of nodes")
    parser.add_argument("--timeout", default=10000, type=int, 
                        help="Job duration in minutes")
    parser.add_argument("--partition", default="vanderbc_gpu", type=str, 
                        help="Partition name")
    parser.add_argument("--constraint", default="h100", type=str,
                        help="GPU constraint (a100, h100, etc)")
    
    return parser.parse_args()


def get_shared_folder() -> Path:
    """Get shared folder for logs and checkpoints."""
    user = os.environ.get('USER', 'user')
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
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
    args = parse_args()
    
    # Set output directory
    args.output_dir = str(get_shared_folder())
    
    # Setup executor
    executor = submitit.AutoExecutor(folder=args.output_dir, slurm_max_num_timeout=30)

    # Job name with timestamp
    job_name = f"adios_tme_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Slurm parameters
    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    executor.update_parameters(
        mem_gb=256,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,
        cpus_per_task=8,
        nodes=nodes,
        timeout_min=timeout_min,
        slurm_partition=args.partition,
        slurm_signal_delay_s=120,
        slurm_gres=f'gpu:{args.ngpus}',
        slurm_constraint=args.constraint,
        slurm_setup=[
            f'export OMP_NUM_THREADS=8',
            f'export NCCL_SOCKET_IFNAME=ib,bond',
            f'export MASTER_PORT=23468',
            f'export WORLD_SIZE={num_gpus_per_node * nodes}',
        ]
    )
    
    executor.update_parameters(name=job_name)

    args.dist_url = get_init_file().as_uri()

    # ========== Architecture Configuration ==========
    args.arch = 'vit_base'
    args.patch_size = 16
    args.embed_dim = 768
    args.num_heads = 12 # please make heads * 64 = 768, good for Fast Attention methods
    args.depth = 12
    args.mlp_ratio = 4.0

    # ========== TME Head Configuration ==========
    args.tme_hidden_dim = 2048
    args.tme_output_dim = 256
    args.tme_layers = 3

    # ========== Mask Model Configuration ==========
    args.mask_model_type = 'adios'  # Ours: 'vit_unet', ADIOS:'adios'
    args.num_masks = 3
    args.mask_encoder_dim = 384
    args.mask_encoder_depth = 12
    args.mask_update_freq = 1
    args.mask_dropout = 0.2
    args.crops_per_mask = 0
    args.sparsity_penalty_type = 'sinh_squared'
    # In case you don't want to train the mask model backbone and have a checkpoint ready
    args.freeze_mask_encoder = True
    args.mask_encoder_checkpoint = '/data1/vanderbc/nandas1/TCGA_Dinov2_ViT-B_run2/logs/checkpoint.pth' 

    # ========== Reconstructor Model Configuration ==========
    args.use_reconstructor = True
    args.reconstructor_encoder_dim = args.mask_encoder_dim
    args.reconstructor_encoder_depth = args.mask_encoder_depth
    args.beta_reconstruction = 1.0
    args.reconstructor_update_freq = 1

    # ========== ADIOS Loss Configuration ==========
    args.alpha_sparsity = 0.1
    args.initial_temp = 0.2
    args.final_temp = 0.05
    args.reconstruction_weight = 0.1

    # ========== Training Configuration ==========
    args.batch_size_per_gpu = 128
    args.total_iterations = 300_000
    args.warmup_iterations = 10_000
    args.lr = 5e-5
    args.min_lr = 1e-6
    args.weight_decay = 0.0 # ADIOS: 0.0, Ours: 0.04
    args.clip_grad = 1.0

    # ========== Training Setup ==========
    args.use_fp16 = True
    args.grad_checkpointing = True
    args.num_workers = 8

    # ========== Logging & Checkpointing ==========
    args.save_freq = 2_000
    args.log_freq = 10
    args.viz_freq = 500

    # ========== Dataset Configuration ==========
    args.data_path = "/data1/vanderbc/foundation_model_training_images/TCGA"
    args.img_size = 224

    # ========== Calculate Effective Batch Size ==========
    effective_batch_size = args.batch_size_per_gpu * num_gpus_per_node * nodes
    
    # Scale learning rate with batch size
    args.lr = args.lr * (effective_batch_size / 256.0)
    
    # Save configuration
    config_file = os.path.join(args.output_dir, f"{job_name}_config.txt")
    with open(config_file, "w") as f:
        f.write("ADIOS-TME Training Configuration\n")
        f.write("=" * 80 + "\n\n")
        for arg, value in sorted(vars(args).items()):
            f.write(f"{arg}: {value}\n")

    # Create and submit trainer
    trainer = Trainer(args)
    job = executor.submit(trainer)

    # Print submission info
    print("\n" + "=" * 80)
    print("JOB SUBMITTED SUCCESSFULLY")
    print("=" * 80)
    print(f"Job ID: {job.job_id}")
    print(f"Job name: {job_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"Config file: {config_file}")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"Architecture: {args.arch}")
    print(f"  Patch size: {args.patch_size}")
    print(f"  Embedding dim: {args.embed_dim}")
    print(f"  Num heads: {args.num_heads}")
    print(f"  Depth: {args.depth}")
    print()
    print(f"Mask Configuration:")
    print(f"  Num masks: {args.num_masks}")
    print(f"  Mask update freq: {args.mask_update_freq}")
    print()
    print(f"Training:")
    print(f"  Batch size per GPU: {args.batch_size_per_gpu}")
    print(f"  Total GPUs: {num_gpus_per_node * nodes} ({nodes} node(s) × {num_gpus_per_node} GPUs)")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Learning rate: {args.lr:.2e} (scaled)")
    print(f"  Total iterations: {args.total_iterations:,}")
    print()
    print(f"Checkpointing:")
    print(f"  Save frequency: every {args.save_freq} iterations")
    print(f"  Visualization frequency: every {args.viz_freq} iterations")
    print("=" * 80 + "\n")
    
    print("Monitor job:")
    print(f"  tail -f {args.output_dir}/{job.job_id}_0_log.out")
    print(f"  squeue -u $USER | grep {job.job_id}")
    print()
    print("Cancel job:")
    print(f"  scancel {job.job_id}")
    print()


if __name__ == "__main__":
    main()
