"""Stage 1: train the channel selector only.

ADIOS encoder and mask decoder are frozen. The selector learns to identify
the nuclei channel given the ADIOS mask outputs, supervised by the
best-IoU-channel target derived from PanNuke binary masks.

Success criterion (HANDOFF §6.1): val accuracy >= 0.85.
"""

import argparse
import copy
import importlib
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler

from cellvit.datasets import SynchronizedTransform
from cellvit.utils import WarmupDecayScheduler, set_seed

from adios_cellvit.adios_backbone import load_adios_mask_model
from adios_cellvit.channel_selector import (
    ChannelSelector,
    compute_best_channel_target,
    compute_soft_channel_target,
)
from adios_cellvit.pannuke_dataset import ADIOSPanNukeDataset


def _load_config(config_path: str) -> dict:
    spec = importlib.util.spec_from_file_location('nuclei_counter_cfg', config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.STAGE1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/nuclei_counter.py')
    p.add_argument('--adios_checkpoint', default=None)
    p.add_argument('--pannuke_path', default=None)
    p.add_argument('--output_dir', default=None)
    return p.parse_args()


def build_loaders(config):
    train_transform_settings = {
        'normalize':      {'mean': config['normalize_mean'], 'std': config['normalize_std']},
        'RandomRotate90': {'p': 0.5},
        'HorizontalFlip': {'p': 0.5},
        'VerticalFlip':   {'p': 0.5},
        'Downscale':      {'p': 0.15, 'scale': 0.5},
        'Blur':           {'p': 0.2,  'blur_limit': 3},
        'ColorJitter':    {'p': 0.2},
    }
    val_transform_settings = {
        'normalize': {'mean': config['normalize_mean'], 'std': config['normalize_std']},
    }
    train_transform = SynchronizedTransform(train_transform_settings, input_shape=224)
    val_transform = SynchronizedTransform(val_transform_settings, input_shape=224)

    train_dataset = ADIOSPanNukeDataset(
        data_dir=config['pannuke_path'],
        split='Training',
        magnification=config['magnification'],
        transform=train_transform,
    )
    # Held out for after the run; kept for symmetry with stage 2.
    _ = ADIOSPanNukeDataset(
        data_dir=config['pannuke_path'],
        split='Test',
        magnification=config['magnification'],
        transform=val_transform,
    )

    val_size = int(len(train_dataset) * config['val_split'])
    val_indices = list(range(val_size))
    train_indices = list(range(val_size, len(train_dataset)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=SubsetRandomSampler(train_indices),
        num_workers=config['num_workers'],
        pin_memory=True,
    )
    val_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=SubsetRandomSampler(val_indices),
        num_workers=config['num_workers'],
        pin_memory=True,
    )
    return train_loader, val_loader


def run_epoch(selector, mask_model, loader, optimizer, device, train: bool):
    selector.train(mode=train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for image, mask_2ch, _, _, _ in loader:
            image = image.to(device, non_blocking=True)
            gt_binary = mask_2ch[:, 0].to(device, non_blocking=True).float()

            with torch.no_grad():
                mask_output = mask_model(image)['masks']

            # Channel scrambling: per-sample random permutation of the 3 mask
            # channels.  Forces the selector to learn nuclei-vs-not from channel
            # content rather than from channel index.  The target is computed on
            # the permuted tensor so it tracks the permutation automatically
            # (compute_best_channel_target is symmetric in channels).
            B = mask_output.shape[0]
            N = mask_output.shape[1]
            H, W = mask_output.shape[-2:]
            perms = torch.stack([torch.randperm(N) for _ in range(B)]).to(mask_output.device)
            perm_idx = perms.view(B, N, 1, 1).expand(B, N, H, W)
            mask_output = torch.gather(mask_output, dim=1, index=perm_idx)

            # SOFT_TARGET_KL: KL divergence between log-softmax(logits) and
            # the IoU-derived soft target.  Accuracy is reported against the
            # hardened soft target (argmax) for comparability with previous runs.
            soft_target = compute_soft_channel_target(mask_output, gt_binary)
            target = soft_target.argmax(dim=1)
            channel_logits = selector(mask_output)
            log_probs = F.log_softmax(channel_logits, dim=1)
            loss = F.kl_div(log_probs, soft_target, reduction='batchmean')

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(selector.parameters(), max_norm=1.0)
                optimizer.step()

            bs = image.size(0)
            total_loss += loss.item() * bs
            total_correct += (channel_logits.argmax(dim=1) == target).sum().item()
            total_count += bs

    return total_loss / total_count, total_correct / total_count


def main():
    args = parse_args()
    config = _load_config(args.config)
    for key in ('adios_checkpoint', 'pannuke_path', 'output_dir'):
        cli_val = getattr(args, key)
        if cli_val is not None:
            config[key] = cli_val

    set_seed(config['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(config['output_dir'], exist_ok=True)

    mask_model = load_adios_mask_model(config['adios_checkpoint'], device)

    selector = ChannelSelector(num_masks=3).to(device)

    train_loader, val_loader = build_loaders(config)

    optimizer = torch.optim.AdamW(
        selector.parameters(),
        lr=1e-8,
        weight_decay=config['weight_decay'],
    )
    scheduler = WarmupDecayScheduler(
        optimizer,
        warmup_epochs=config['warmup_epochs'],
        total_epochs=config['max_epochs'],
        base_lr=config['lr'],
        final_lr=config['lr'] / 10,
        warmup_start_lr=config['lr'] / 100,
    )

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_state = None
    epochs_since_improve = 0

    for epoch in range(config['max_epochs']):
        train_loss, train_acc = run_epoch(
            selector, mask_model, train_loader, optimizer, device, train=True,
        )
        val_loss, val_acc = run_epoch(
            selector, mask_model, val_loader, optimizer, device, train=False,
        )
        scheduler.step()

        print(
            f"[stage1] epoch {epoch+1}/{config['max_epochs']} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = copy.deepcopy(selector.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= config['early_stop_patience']:
                print(f"[stage1] early stop at epoch {epoch+1}")
                break

    ckpt_path = os.path.join(config['output_dir'], 'stage1_selector.pth')
    torch.save(
        {
            'selector_state_dict': best_state,
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'config': config,
        },
        ckpt_path,
    )
    print(f"[stage1] saved best to {ckpt_path}")
    print(f"[stage1] best_val_loss={best_val_loss:.4f} best_val_acc={best_val_acc:.3f}")
    if best_val_acc < 0.85:
        print("[stage1] WARNING: best_val_acc < 0.85 — do not proceed to stage 2.")


if __name__ == '__main__':
    main()
