#!/usr/bin/env python
"""
Submitit launcher for ADIOS-TME training on SLURM.
Simplified version without DINOv2 complexity.
"""

import argparse
import os
import datetime
import uuid
from pathlib import Path
import submitit

from configs.config import get_args_parser
from training.trainer import train_adios_tme


def get_shared_folder() -> Path:
    """Get folder for logs and checkpoints."""
    user = os.environ.get('USER', 'user')
    base_dir = f"/data1/vanderbc/{user}/adios_tme_outputs"
    
    # Create timestamp-based subfolder
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(base_dir) / f"run_{timestamp}"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    return output_dir


class Trainer:
    """Trainer object for submitit."""
    
    def __init__(self, args):
        self.args = args
    
    def __call__(self):
        self._setup_gpu_args()
        train_adios_tme(self.args)
    
    def checkpoint(self):
        """Handle preemption."""
        import submitit
        
        self.args.dist_url = get_init_file().as_uri()
        print(f"Requeuing job with args: {self.args}")
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)
    
    def _setup_gpu_args(self):
        """Setup GPU args from SLURM environment."""
        job_env = submitit.JobEnvironment()
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def get_init_file():
    """Create unique init file for distributed training."""
    shared_folder = get_shared_folder()
    init_file = shared_folder / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


def main():
    """Main submitit launcher."""
    
    # Parse base arguments
    parser = argparse.ArgumentParser('ADIOS-TME Submitit', parents=[get_args_parser()])
    
    # Add submitit-specific arguments
    parser.add_argument('--ngpus', default=4, type=int,
                        help='Number of GPUs per node')
    parser.add_argument('--nodes', default=1, type=int,
                        help='Number of nodes')
    parser.add_argument('--timeout', default=72, type=int,
                        help='Job duration in hours')
    parser.add_argument('--partition', default='gpu', type=str,
                        help='SLURM partition')
    parser.add_argument('--constraint', default='h100', type=str,
                        help='GPU constraint (a100, h100, etc)')
    parser.add_argument('--job_name', default='adios_tme', type=str,
                        help='Job name prefix')
    parser.add_argument('--mem_gb', default=256, type=int,
                        help='Memory per node in GB')
    parser.add_argument('--cpus_per_task', default=8, type=int,
                        help='CPUs per task')
    
    args = parser.parse_args()
    
    # Setup output directory
    args.output_dir = str(get_shared_folder())
    print(f"Output directory: {args.output_dir}")
    
    # Create executor
    executor = submitit.AutoExecutor(folder=args.output_dir, slurm_max_num_timeout=30)
    
    # Job configuration
    timeout_min = args.timeout * 60
    
    executor.update_parameters(
        mem_gb=args.mem_gb,
        gpus_per_node=args.ngpus,
        tasks_per_node=args.ngpus,
        cpus_per_task=args.cpus_per_task,
        nodes=args.nodes,
        timeout_min=timeout_min,
        slurm_partition=args.partition,
        slurm_signal_delay_s=120,
        slurm_job_name=f"{args.job_name}_{datetime.datetime.now():%Y%m%d_%H%M%S}",
    )
    
    # Add GPU constraint if specified
    if args.constraint:
        executor.update_parameters(slurm_constraint=args.constraint)
    
    # Set distributed training URL
    args.dist_url = get_init_file().as_uri()
    
    # ========== Configure training hyperparameters ==========
    
    # Architecture based on --arch flag
    if args.arch == 'vit_small':
        args.embed_dim = 384
        args.num_heads = 6
        args.depth = 12
    elif args.arch == 'vit_base':
        args.embed_dim = 768
        args.num_heads = 12
        args.depth = 12
    elif args.arch == 'vit_large':
        args.embed_dim = 1024
        args.num_heads = 16
        args.depth = 24
    
    # Calculate total batch size
    total_batch_size = args.batch_size_per_gpu * args.ngpus * args.nodes
    print(f"Total batch size: {total_batch_size}")
    
    # Scale learning rate with batch size
    args.lr = args.lr * (total_batch_size / 256.0)
    print(f"Scaled learning rate: {args.lr}")
    
    # Save configuration
    config_path = Path(args.output_dir) / "config.txt"
    with open(config_path, 'w') as f:
        f.write("ADIOS-TME Training Configuration\n")
        f.write("=" * 50 + "\n\n")
        for key, value in sorted(vars(args).items()):
            f.write(f"{key}: {value}\n")
    
    # Submit job
    trainer = Trainer(args)
    job = executor.submit(trainer)
    
    # Print job information
    print("\n" + "=" * 50)
    print("JOB SUBMITTED SUCCESSFULLY")
    print("=" * 50)
    print(f"Job ID: {job.job_id}")
    print(f"Output directory: {args.output_dir}")
    print(f"Logs: {args.output_dir}/%j_0_log.out")
    print(f"Config: {config_path}")
    print("=" * 50 + "\n")
    
    # Monitor command
    print("To monitor:")
    print(f"  tail -f {args.output_dir}/{job.job_id}_0_log.out")
    print(f"  squeue -u $USER | grep {job.job_id}")
    print("\nTo cancel:")
    print(f"  scancel {job.job_id}")


if __name__ == "__main__":
    main()