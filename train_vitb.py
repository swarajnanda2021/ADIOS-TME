"""Path B-2 training: ViT-B + three-branch CellViT + optional ADIOS prior.

Single-GPU A100 interactive node. Modeled on ``train_stage2_cellvit.py`` —
same dataloader, same dense-supervision loss design, same scheduler — with
two additions:

  * Class-weighted NC cross-entropy (background + 5 foreground classes).
    Weights are computed from the PanNuke Training class_masks once and
    cached on disk; see ``compute_nc_class_weights``.
  * Optional ADIOS-prior consistency BCE on the NP map. When
    ``config['use_adios_consistency']`` is true, the frozen ADIOS mask
    model + Stage 1 selector are loaded and attached to the model. The
    model then emits ``adios_fg``/``adios_bg`` during training and the
    loss adds ``lambda_adios * (BCE(pred_np, adios_fg) + BCE(pred_np, 1 - adios_bg))``.

Three ablations all run from this script by toggling
``use_adios_consistency`` and (for the no-prior pure-supervised baseline)
``lambda_adios = 0.0``.  Path Z itself stays as the reference baseline run
via ``train_stage2_cellvit.py``.
"""

import argparse
import copy
import importlib
import json
import os
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler

from cellvit.datasets import SynchronizedTransform
from cellvit.utils import WarmupDecayScheduler, set_seed

from adios_cellvit.adios_backbone import load_adios_mask_model
from adios_cellvit.channel_selector import ChannelSelector
from adios_cellvit.pannuke_dataset import (
    ADIOSPanNukeDataset,
    FOREGROUND_CLASSES,
)
from adios_cellvit.vitb_backbone import load_vitb_encoder
from adios_cellvit.vitb_cellvit_model import ViTBCellViT


# ===========================================================================
# Loss
# ===========================================================================

class CombinedLossVitB(nn.Module):
    """Path Z combined loss + optional ADIOS-prior consistency BCE on NP."""

    def __init__(
        self,
        weights: dict,
        nc_class_weights: torch.Tensor,
        lambda_adios: float = 0.1,
    ):
        super().__init__()
        self.w = weights
        self.lambda_adios = lambda_adios
        # Move with the module via to(device); not a Parameter.
        self.register_buffer('nc_class_weights', nc_class_weights.float())

    # The three static helpers below are copied verbatim from
    # CombinedLossWithNC in train_stage2_cellvit.py.

    @staticmethod
    def _xentropy_2ch(prob_2ch, true_2ch):
        eps = 1e-7
        prob = prob_2ch.clamp(min=eps, max=1.0 - eps)
        return -(true_2ch * prob.log()).sum(dim=1).mean()

    @staticmethod
    def _dice_loss(prob_2ch, true_2ch):
        eps = 1e-6
        inter = (prob_2ch * true_2ch).sum(dim=(0, 2, 3))
        union = prob_2ch.sum(dim=(0, 2, 3)) + true_2ch.sum(dim=(0, 2, 3))
        return 1.0 - ((2.0 * inter + eps) / (union + eps)).mean()

    @staticmethod
    def _msge_loss(pred_dist, true_dist):
        kx = torch.tensor(
            [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
            device=pred_dist.device,
            dtype=pred_dist.dtype,
        ).view(1, 1, 3, 3) / 8.0
        ky = kx.transpose(-1, -2)
        C = pred_dist.shape[1]
        kx_full = kx.expand(C, 1, 3, 3)
        ky_full = ky.expand(C, 1, 3, 3)
        gx_p = F.conv2d(pred_dist, kx_full, padding=1, groups=C)
        gy_p = F.conv2d(pred_dist, ky_full, padding=1, groups=C)
        gx_t = F.conv2d(true_dist, kx_full, padding=1, groups=C)
        gy_t = F.conv2d(true_dist, ky_full, padding=1, groups=C)
        return ((gx_p - gx_t) ** 2 + (gy_p - gy_t) ** 2).mean()

    def forward(
        self,
        true_mask,         # [B, 2, H, W] one-hot
        pred_mask,         # [B, 1, H, W] probability in [0, 1]
        true_dist,         # [B, 2, H, W]
        pred_dist,         # [B, 2, H, W] raw
        true_class,        # [B, H, W] long in {0..5}
        pred_class,        # [B, 6, H, W] raw
        adios_fg=None,     # [B, 1, H, W] in [0, 1], optional
        adios_bg=None,     # [B, 1, H, W] in [0, 1], optional
    ):
        prob_fg = pred_mask.clamp(0.0, 1.0)
        prob_bg = 1.0 - prob_fg
        prob_2ch = torch.cat([prob_fg, prob_bg], dim=1)

        l_xent = self._xentropy_2ch(prob_2ch, true_mask)
        l_dice = self._dice_loss(prob_2ch, true_mask)
        l_mse = F.mse_loss(pred_dist, true_dist)
        l_msge = self._msge_loss(pred_dist, true_dist)
        l_nc = F.cross_entropy(pred_class, true_class, weight=self.nc_class_weights)

        total = (
            self.w['w_xentropy'] * l_xent
            + self.w['w_dice'] * l_dice
            + self.w['w_mse'] * l_mse
            + self.w['w_msge'] * l_msge
            + self.w['w_nc'] * l_nc
        )

        l_cons = torch.tensor(0.0, device=pred_mask.device)
        if adios_fg is not None and adios_bg is not None and self.lambda_adios > 0:
            eps = 1e-7
            p = prob_fg.clamp(min=eps, max=1.0 - eps)
            l_cons_fg = F.binary_cross_entropy(p, adios_fg.clamp(0.0, 1.0))
            l_cons_bg = F.binary_cross_entropy(p, (1.0 - adios_bg).clamp(0.0, 1.0))
            l_cons = l_cons_fg + l_cons_bg
            total = total + self.lambda_adios * l_cons

        return total, {
            'xent': l_xent.item(),
            'dice': l_dice.item(),
            'mse': l_mse.item(),
            'msge': l_msge.item(),
            'nc': l_nc.item(),
            'cons': l_cons.item() if isinstance(l_cons, torch.Tensor) else float(l_cons),
            'total': total.item(),
        }


# ===========================================================================
# NC class weights — count once, cache on disk
# ===========================================================================

def compute_nc_class_weights(
    pannuke_path: str,
    output_dir: str,
    split: str = 'Training',
    magnification: str = '40x',
    num_classes: int = 6,
    weight_cap: float = 20.0,
) -> torch.Tensor:
    """Inverse-frequency NC class weights, normalized to sum to ``num_classes``.

    Walks ``class_masks/<class>/*.png`` for each foreground class to count
    nonzero pixels per class; background pixel count is derived by
    subtraction. Caches the result to ``output_dir/nc_class_weights.json``
    keyed on ``(pannuke_path, split, magnification)``.

    Weights are capped at ``weight_cap`` (default 20) to avoid runaway
    weights on tiny classes — handoff §"Open questions" notes that
    weights >50 destabilize training. Pre- and post-cap distributions are
    printed.

    Returns a ``FloatTensor`` of length ``num_classes`` aligned with
    ``F.cross_entropy(weight=...)``. Index 0 is background; indices
    1..len(FOREGROUND_CLASSES) match ``FOREGROUND_CLASSES``.
    """
    cache_path = os.path.join(output_dir, 'nc_class_weights.json')
    cache_key = f"{pannuke_path}::{split}::{magnification}::K={num_classes}"

    if os.path.isfile(cache_path):
        try:
            with open(cache_path) as f:
                cache = json.load(f)
            if cache_key in cache:
                weights = torch.tensor(cache[cache_key], dtype=torch.float32)
                print(f"[nc_weights] reusing cached weights from {cache_path}: "
                      f"{weights.tolist()}")
                return weights
        except (json.JSONDecodeError, OSError):
            print(f"[nc_weights] could not read cache at {cache_path}; recomputing")

    print(f"[nc_weights] computing from {pannuke_path}/{split}/{magnification}")
    # We expect num_classes = 1 + len(FOREGROUND_CLASSES). If a caller passes a
    # different value, fall back to the foreground-class list and zero-pad
    # background only.
    counts_fg = [0] * len(FOREGROUND_CLASSES)
    total_pixels = 0

    class_root = os.path.join(pannuke_path, split, magnification, 'class_masks')
    if not os.path.isdir(class_root):
        raise FileNotFoundError(f"class_masks dir not found: {class_root}")

    for i, cls in enumerate(FOREGROUND_CLASSES):
        cls_dir = os.path.join(class_root, cls)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(f"class dir not found: {cls_dir}")
        files = sorted(f for f in os.listdir(cls_dir) if f.endswith('.png'))
        if i == 0:
            print(f"[nc_weights] {len(files)} patches per class; counting pixels...")
        for fname in files:
            arr = cv2.imread(os.path.join(cls_dir, fname), cv2.IMREAD_UNCHANGED)
            if arr is None:
                continue
            counts_fg[i] += int((arr > 0).sum())
            if i == 0:
                total_pixels += int(arr.size)
        print(f"[nc_weights]   {cls:>14s}: {counts_fg[i]:>14,d} px")

    count_bg = total_pixels - sum(counts_fg)
    counts = [count_bg] + counts_fg                # length 1 + 5 = 6 by default
    # If caller specified a different num_classes, pad/truncate.
    while len(counts) < num_classes:
        counts.append(0)
    counts = counts[:num_classes]

    counts_t = torch.tensor(counts, dtype=torch.float64).clamp(min=1.0)
    weights = total_pixels / (num_classes * counts_t)

    # Renormalize so weights sum to num_classes (standard convention).
    weights = weights * (num_classes / weights.sum())
    pre_cap = weights.clone()

    weights = weights.clamp(max=weight_cap)
    # Re-renormalize after capping.
    weights = weights * (num_classes / weights.sum())

    print(f"[nc_weights] pre-cap:  {[round(w, 3) for w in pre_cap.tolist()]}")
    print(f"[nc_weights] post-cap: {[round(w, 3) for w in weights.tolist()]} (cap={weight_cap})")

    weights = weights.float()

    os.makedirs(output_dir, exist_ok=True)
    cache = {}
    if os.path.isfile(cache_path):
        try:
            with open(cache_path) as f:
                cache = json.load(f)
        except (json.JSONDecodeError, OSError):
            cache = {}
    cache[cache_key] = weights.tolist()
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)
    print(f"[nc_weights] cached to {cache_path}")
    return weights


# ===========================================================================
# Config + dataloaders (mirrors train_stage2_cellvit.build_loaders)
# ===========================================================================

def _load_config(config_path: str) -> dict:
    spec = importlib.util.spec_from_file_location('nuclei_counter_cfg', config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.STAGE_VITB


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/nuclei_counter.py')
    p.add_argument('--vitb_checkpoint', default=None)
    p.add_argument('--adios_checkpoint', default=None)
    p.add_argument('--stage1_selector', default=None)
    p.add_argument('--pannuke_path', default=None)
    p.add_argument('--output_dir', default=None)
    p.add_argument('--use_adios_consistency', type=lambda s: s.lower() in ('1', 'true', 'yes'),
                   default=None, help='Override config use_adios_consistency.')
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


# ===========================================================================
# Train / val loop
# ===========================================================================

def run_epoch(model, criterion, loader, optimizer, device, train: bool):
    model.train(mode=train)
    # The ADIOS prior lives outside nn.Module's submodule tree, so
    # model.train() doesn't touch it. Belt-and-suspenders: explicitly
    # eval() each prior component when training to suppress any dropout/BN.
    if model.has_adios_prior:
        model.adios_prior.mask_model.eval()
        model.adios_prior.selector.eval()

    totals = {'xent': 0.0, 'dice': 0.0, 'mse': 0.0, 'msge': 0.0,
              'nc': 0.0, 'cons': 0.0, 'total': 0.0}
    n_seen = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for image, mask_2ch, distance_map, _instance, class_mask in loader:
            image = image.to(device, non_blocking=True)
            mask_2ch = mask_2ch.to(device, non_blocking=True).float()
            distance_map = distance_map.to(device, non_blocking=True).float()
            class_mask = class_mask.to(device, non_blocking=True).long()

            out = model(image)
            loss, log = criterion(
                true_mask=mask_2ch,
                pred_mask=out['masks'],
                true_dist=distance_map,
                pred_dist=out['distances'],
                true_class=class_mask,
                pred_class=out['nuclei_types'],
                adios_fg=out.get('adios_fg'),
                adios_bg=out.get('adios_bg'),
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=0.3,
                )
                optimizer.step()

            bs = image.size(0)
            n_seen += bs
            for k, v in log.items():
                totals[k] += v * bs

    return {k: v / n_seen for k, v in totals.items()}


# ===========================================================================
# main
# ===========================================================================

def main():
    args = parse_args()
    config = _load_config(args.config)
    for key in ('vitb_checkpoint', 'adios_checkpoint', 'stage1_selector',
                'pannuke_path', 'output_dir'):
        cli_val = getattr(args, key)
        if cli_val is not None:
            config[key] = cli_val
    if args.use_adios_consistency is not None:
        config['use_adios_consistency'] = args.use_adios_consistency

    set_seed(config['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(config['output_dir'], exist_ok=True)

    # ---- model -------------------------------------------------------------
    encoder, patch_size, num_registers, embed_dim = load_vitb_encoder(
        config['vitb_checkpoint'], device,
    )
    model = ViTBCellViT(
        encoder=encoder,
        patch_size=patch_size,
        num_registers=num_registers,
        encoder_dim=embed_dim,
        num_classes=config['num_classes'],
        drop_rate=0.1,
    ).to(device)

    if config.get('use_adios_consistency', False):
        prior_mask_model = load_adios_mask_model(config['adios_checkpoint'], device)
        prior_selector = ChannelSelector(num_masks=3).to(device)
        stage1_ckpt = torch.load(
            config['stage1_selector'], map_location=device, weights_only=False,
        )
        prior_selector.load_state_dict(stage1_ckpt['selector_state_dict'])
        model.set_adios_prior(prior_mask_model, prior_selector)
        print(f"[train_vitb] ADIOS prior attached "
              f"(lambda_adios={config['lambda_adios']})")
    else:
        print("[train_vitb] no ADIOS prior (pure-supervised ablation)")

    # ---- loss --------------------------------------------------------------
    nc_weights = compute_nc_class_weights(
        config['pannuke_path'],
        config['output_dir'],
        split='Training',
        magnification=config['magnification'],
        num_classes=config['num_classes'],
    )
    criterion = CombinedLossVitB(
        weights=config['loss_weights'],
        nc_class_weights=nc_weights,
        lambda_adios=config['lambda_adios'] if config.get('use_adios_consistency', False) else 0.0,
    ).to(device)

    # ---- optimizer ---------------------------------------------------------
    encoder_params = list(model.cellvit.encoder.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    heads_params = [
        p for p in model.cellvit.parameters()
        if id(p) not in encoder_ids and p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        [
            {'params': encoder_params, 'lr': config['encoder_lr']},
            {'params': heads_params,   'lr': config['heads_lr']},
        ],
        weight_decay=config['weight_decay'],
    )
    scheduler = WarmupDecayScheduler(
        optimizer,
        warmup_epochs=config['warmup_epochs'],
        total_epochs=config['epochs'],
        base_lr=config['heads_lr'],
        final_lr=config['heads_lr'] / 10,
        warmup_start_lr=config['heads_lr'] / 100,
    )

    train_loader, val_loader = build_loaders(config)

    # ---- loop --------------------------------------------------------------
    best_val_total = float('inf')
    best_state = None
    best_epoch = -1

    for epoch in range(config['epochs']):
        train_log = run_epoch(model, criterion, train_loader, optimizer, device, train=True)
        val_log = run_epoch(model, criterion, val_loader, optimizer, device, train=False)
        scheduler.step()

        lr_enc = optimizer.param_groups[0]['lr']
        lr_hds = optimizer.param_groups[1]['lr']
        print(
            f"[vitb] epoch {epoch+1}/{config['epochs']} "
            f"train_total={train_log['total']:.4f} val_total={val_log['total']:.4f} "
            f"val_nc={val_log['nc']:.4f} val_mse={val_log['mse']:.4f} "
            f"val_cons={val_log['cons']:.4f} "
            f"lr_enc={lr_enc:.2e} lr_heads={lr_hds:.2e}"
        )

        if val_log['total'] < best_val_total:
            best_val_total = val_log['total']
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

    ckpt_path = os.path.join(config['output_dir'], 'vitb_pathb2.pth')
    torch.save(
        {
            'model_state_dict': best_state,
            'epoch': best_epoch,
            'val_total': best_val_total,
            'config': config,
        },
        ckpt_path,
    )
    print(f"[vitb] saved best to {ckpt_path}  best_val_total={best_val_total:.4f} "
          f"@ epoch {best_epoch}")


if __name__ == '__main__':
    main()
