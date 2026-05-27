"""Submitit launcher for stage2 of the nuclei-counter pipeline."""

import argparse
import os
import subprocess
import uuid
import datetime
from pathlib import Path

import submitit

WORK_DIR = "/data1/vanderbc/test_dinov2_swaraj/ADIOS"
SHARED_LOG_DIR = Path(WORK_DIR) / "logs" / "stage2_submitit"


def get_shared_folder() -> Path:
    SHARED_LOG_DIR.mkdir(parents=True, exist_ok=True)
    return SHARED_LOG_DIR


def get_init_file() -> Path:
    SHARED_LOG_DIR.mkdir(parents=True, exist_ok=True)
    p = SHARED_LOG_DIR / f"{uuid.uuid4().hex}_init"
    if p.exists():
        p.unlink()
    return p


class Trainer:
    def __init__(self, args):
        self.args = args

    def __call__(self):
        self._setup_gpu_args()
        cmd = ["python", "train_stage2_cellvit.py", "--config", "configs.nuclei_counter"]
        print(f"Launching: {' '.join(cmd)}", flush=True)
        subprocess.check_call(cmd, cwd=WORK_DIR)

    def checkpoint(self):
        self.args.dist_url = get_init_file().as_uri()
        return submitit.helpers.DelayedSubmission(type(self)(self.args))

    def _setup_gpu_args(self):
        job_env = submitit.JobEnvironment()
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"SLURM env: tasks={job_env.num_tasks} rank={job_env.global_rank}", flush=True)


def main():
    args = argparse.Namespace(
        output_dir=str(get_shared_folder()),
        dist_url=get_init_file().as_uri(),
        world_size=1, rank=0, gpu=0,
    )
    executor = submitit.AutoExecutor(folder=args.output_dir, slurm_max_num_timeout=30)
    job_name = f"nuclei_stage2_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    executor.update_parameters(
        mem_gb=128, gpus_per_node=1, tasks_per_node=1, cpus_per_task=8, nodes=1,
        timeout_min=2880,
        slurm_partition="your-slurm-partition-name",
        slurm_signal_delay_s=120,
        slurm_gres="gpu:1",
        slurm_constraint="h100",
        slurm_setup=["export OMP_NUM_THREADS=8", "export WORLD_SIZE=1"],
    )
    executor.update_parameters(name=job_name)
    job = executor.submit(Trainer(args))
    print("=" * 80)
    print(f"JOB SUBMITTED — stage2")
    print("=" * 80)
    print(f"Job ID:  {job.job_id}")
    print(f"Monitor: tail -f {args.output_dir}/{job.job_id}_0_log.out")
    print(f"Cancel:  scancel {job.job_id}")


if __name__ == "__main__":
    main()
