"""Path B-3 (CellViT++ style) training: ViT-B + NP/HV + per-cell MLP.

End-to-end fresh from the FMC ViT-B init — same training regime as
``train_vitb.py`` so the only experimental delta vs B-2 is the NC head
architecture (per-pixel decoder → per-cell pooled-token MLP).

Three configurations all run from this script by toggling
``use_adios_consistency`` and ``lambda_adios``:

  * with-prior   (use_adios_consistency=True,  lambda_adios=0.1)
  * no-prior     (use_adios_consistency=False)

NC loss is now per-instance, not per-pixel. Cell-level CE on K=5
foreground classes; targets are derived from the modal class inside
each GT instance. Class weights computed once by walking
``class_masks/<class>/*.png`` and counting distinct nonzero instance
IDs per class — i.e., cells-per-class, not pixels-per-class — which
is the appropriate denominator for an instance-level loss.
"""

import argparse
import copy
import importlib
import json
import os

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
from adios_cellvit.vitb_cellvitpp_model import ViTBCellViTPP


# ===========================================================================
# Loss
# ===========================================================================

class CombinedLossCellViTPP(nn.Module):
    """NP/HV losses identical to Path B-2; NC term is per-cell CE.

    The per-pixel ``xent`` / ``dice`` / ``mse`` / ``msge`` terms are
    byte-equivalent to ``train_vitb.py``'s CombinedLossVitB so the
    detection objective is unchanged. The NC term swaps from per-pixel
    CE over a 6-channel decoder to per-cell CE over K=5 foreground
    classes from the MLP. When no cells survived pooling (rare
    sub-patch-only batch) the NC term is zero and the gradient is
    well-defined.
    """

    def __init__(
        self,
        weights: dict,
        cell_class_weights: torch.Tensor,
        lambda_adios: float = 0.1,
    ):
        super().__init__()
        self.w = weights
        self.lambda_adios = lambda_adios
        self.register_buffer('cell_class_weights', cell_class_weights.float())

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
        true_mask,
        pred_mask,
        true_dist,
        pred_dist,
        cell_logits,
        cell_labels,
        adios_fg=None,
        adios_bg=None,
    ):
        prob_fg = pred_mask.clamp(0.0, 1.0)
        prob_bg = 1.0 - prob_fg
        prob_2ch = torch.cat([prob_fg, prob_bg], dim=1)

        l_xent = self._xentropy_2ch(prob_2ch, true_mask)
        l_dice = self._dice_loss(prob_2ch, true_mask)
        l_mse = F.mse_loss(pred_dist, true_dist)
        l_msge = self._msge_loss(pred_dist, true_dist)

        if cell_logits.numel() > 0:
            l_nc = F.cross_entropy(
                cell_logits, cell_labels, weight=self.cell_class_weights,
            )
        else:
            # No surviving cells this batch (extremely rare; tiny patches).
            # Use the classifier's parameters in a no-op to keep gradient
            # well-defined.  cross_entropy on empty input is not allowed.
            l_nc = pred_mask.sum() * 0.0

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
# Cell class weights — count cells per class, cache to disk
# ===========================================================================

def compute_cell_class_weights(
    pannuke_path: str,
    output_dir: str,
    split: str = 'Training',
    magnification: str = '40x',
    num_cell_classes: int = 5,
    weight_cap: float = 20.0,
) -> torch.Tensor:
    """Inverse-frequency cell-class weights, normalized to sum to K.

    For each foreground class, counts distinct nonzero instance IDs
    across every patch's ``class_masks/<class>/*.png``. Per-class PNGs
    are instance-labeled (uint8 with values > 0 indicating that pixel
    belongs to the class for that specific nucleus), so
    ``len(np.unique(arr)) - 1`` is the cell count per patch.

    The denominator here is *cells*, not pixels — relevant because
    large-area classes (e.g. epithelial) and small-area classes (dead)
    can have very different pixel/cell ratios; the per-cell CE loss
    sees one example per cell regardless of size.

    Caches to ``output_dir/cell_class_weights.json`` keyed on
    ``(pannuke_path, split, magnification, K)``.
    """
    cache_path = os.path.join(output_dir, 'cell_class_weights.json')
    cache_key = (
        f"{pannuke_path}::{split}::{magnification}::K={num_cell_classes}"
    )

    if os.path.isfile(cache_path):
        try:
            with open(cache_path) as f:
                cache = json.load(f)
            if cache_key in cache:
                weights = torch.tensor(cache[cache_key], dtype=torch.float32)
                print(f"[cell_weights] reusing cached weights from {cache_path}: "
                      f"{weights.tolist()}")
                return weights
        except (json.JSONDecodeError, OSError):
            print(f"[cell_weights] could not read cache at {cache_path}; recomputing")

    print(f"[cell_weights] computing from {pannuke_path}/{split}/{magnification}")
    counts = [0] * num_cell_classes

    class_root = os.path.join(pannuke_path, split, magnification, 'class_masks')
    if not os.path.isdir(class_root):
        raise FileNotFoundError(f"class_masks dir not found: {class_root}")

    for i, cls in enumerate(FOREGROUND_CLASSES[:num_cell_classes]):
        cls_dir = os.path.join(class_root, cls)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(f"class dir not found: {cls_dir}")
        files = sorted(f for f in os.listdir(cls_dir) if f.endswith('.png'))
        if i == 0:
            print(f"[cell_weights] {len(files)} patches per class; counting cells...")
        for fname in files:
            arr = cv2.imread(os.path.join(cls_dir, fname), cv2.IMREAD_UNCHANGED)
            if arr is None:
                continue
            uniq = np.unique(arr)
            # Distinct instance IDs > 0 are distinct cells of this class
            # in this patch.
            n = int((uniq > 0).sum())
            counts[i] += n
        print(f"[cell_weights]   {cls:>14s}: {counts[i]:>8d} cells")

    total = sum(counts)
    if total == 0:
        raise RuntimeError("compute_cell_class_weights: zero cells counted")

    counts_t = torch.tensor(counts, dtype=torch.float64).clamp(min=1.0)
    weights = total / (num_cell_classes * counts_t)
    weights = weights * (num_cell_classes / weights.sum())
    pre_cap = weights.clone()

    weights = weights.clamp(max=weight_cap)
    weights = weights * (num_cell_classes / weights.sum())

    print(f"[cell_weights] pre-cap:  {[round(w, 3) for w in pre_cap.tolist()]}")
    print(f"[cell_weights] post-cap: {[round(w, 3) for w in weights.tolist()]} (cap={weight_cap})")

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
    print(f"[cell_weights] cached to {cache_path}")
    return weights


# ===========================================================================
# Cell labels — derive per-cell class targets from (instance_mask, class_mask)
# ===========================================================================

def compute_cell_labels(
    instance_mask: torch.Tensor,
    class_mask: torch.Tensor,
    cell_batch_idx: torch.Tensor,
    cell_inst_id: torch.Tensor,
) -> torch.Tensor:
    """Modal foreground class per cell, shifted into {0..K-1} for CE.

    Args:
        instance_mask: [B, 1, H, W] long.
        class_mask:    [B, H, W] long, values in {0..K}, 0 = background.
        cell_batch_idx, cell_inst_id: [N_cells] long, returned by
            ``ViTBCellViTPP.classify_cells_from_features``.

    Returns:
        cell_labels: [N_cells] long in {0..K-1}, suitable for
        ``F.cross_entropy`` against logits of shape [N_cells, K].
    """
    inst = instance_mask.squeeze(1).long()
    labels = []
    for b, k in zip(cell_batch_idx.tolist(), cell_inst_id.tolist()):
        pix = (inst[b] == k)
        cls_pixels = class_mask[b][pix]
        if cls_pixels.numel() == 0:
            # Shouldn't happen — classify_cells only emits cells with
            # >0 pixels — but be defensive.
            labels.append(0)
            continue
        vals, counts = cls_pixels.unique(return_counts=True)
        modal = int(vals[counts.argmax()].item())
        # Modal class is in {1..K_class_mask}; shift to {0..K-1} for CE
        # over the K-channel cell classifier. (Background should never be
        # the modal class of a foreground instance, but if it is — e.g.
        # an instance pixel got rotated outside the labeled foreground
        # under augmentation — clamp to class 0 of foreground.)
        labels.append(max(modal - 1, 0))
    return torch.tensor(labels, dtype=torch.long, device=instance_mask.device)


# ===========================================================================
# Config + dataloaders (mirrors train_vitb.build_loaders)
# ===========================================================================

def _load_config(config_path: str) -> dict:
    spec = importlib.util.spec_from_file_location('nuclei_counter_cfg', config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.STAGE_CELLVITPP


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
    if model.has_adios_prior:
        model.adios_prior.mask_model.eval()
        model.adios_prior.selector.eval()
    # NC decoder is frozen but defensively keep in eval mode.
    model.cellvit.nuclei_type_map_decoder.eval()

    totals = {'xent': 0.0, 'dice': 0.0, 'mse': 0.0, 'msge': 0.0,
              'nc': 0.0, 'cons': 0.0, 'total': 0.0}
    n_seen = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for image, mask_2ch, distance_map, instance_mask, class_mask in loader:
            image = image.to(device, non_blocking=True)
            mask_2ch = mask_2ch.to(device, non_blocking=True).float()
            distance_map = distance_map.to(device, non_blocking=True).float()
            instance_mask = instance_mask.to(device, non_blocking=True)
            class_mask = class_mask.to(device, non_blocking=True).long()

            out = model(image, instance_mask=instance_mask)
            cell_labels = compute_cell_labels(
                instance_mask, class_mask,
                out['cell_batch_idx'], out['cell_inst_id'],
            )
            loss, log = criterion(
                true_mask=mask_2ch,
                pred_mask=out['masks'],
                true_dist=distance_map,
                pred_dist=out['distances'],
                cell_logits=out['cell_logits'],
                cell_labels=cell_labels,
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
    model = ViTBCellViTPP(
        encoder=encoder,
        patch_size=patch_size,
        num_registers=num_registers,
        encoder_dim=embed_dim,
        num_cell_classes=config['num_cell_classes'],
        classifier_hidden_dim=config['classifier_hidden_dim'],
        classifier_dropout=config['classifier_dropout'],
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
        print(f"[train_cellvitpp] ADIOS prior attached "
              f"(lambda_adios={config['lambda_adios']})")
    else:
        print("[train_cellvitpp] no ADIOS prior (no-consistency ablation)")

    # ---- loss --------------------------------------------------------------
    cell_weights = compute_cell_class_weights(
        config['pannuke_path'],
        config['output_dir'],
        split='Training',
        magnification=config['magnification'],
        num_cell_classes=config['num_cell_classes'],
    )
    criterion = CombinedLossCellViTPP(
        weights=config['loss_weights'],
        cell_class_weights=cell_weights,
        lambda_adios=config['lambda_adios'] if config.get('use_adios_consistency', False) else 0.0,
    ).to(device)

    # ---- optimizer ---------------------------------------------------------
    encoder_params = list(model.cellvit.encoder.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    # Everything else trainable: NP + HV branches via self.cellvit (minus the
    # frozen NC branch — its requires_grad=False excludes it naturally), plus
    # the cell classifier MLP.
    heads_params = [
        p for p in model.parameters()
        if id(p) not in encoder_ids and p.requires_grad
    ]

    # Param group ordering: heads at group 0 (scheduled), encoder at group 1
    # (constant — pinned around scheduler.step() below). Matches train_vitb.py.
    optimizer = torch.optim.AdamW(
        [
            {'params': heads_params,   'lr': config['heads_lr']},
            {'params': encoder_params, 'lr': config['encoder_lr']},
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
    # WarmupDecayScheduler ignores per-group base_lrs; pin encoder LR.
    optimizer.param_groups[1]['lr'] = config['encoder_lr']

    train_loader, val_loader = build_loaders(config)

    # ---- loop --------------------------------------------------------------
    best_val_total = float('inf')
    best_state = None
    best_epoch = -1

    for epoch in range(config['epochs']):
        train_log = run_epoch(model, criterion, train_loader, optimizer, device, train=True)
        val_log = run_epoch(model, criterion, val_loader, optimizer, device, train=False)
        scheduler.step()
        optimizer.param_groups[1]['lr'] = config['encoder_lr']

        lr_hds = optimizer.param_groups[0]['lr']
        lr_enc = optimizer.param_groups[1]['lr']
        print(
            f"[cellvitpp] epoch {epoch+1}/{config['epochs']} "
            f"train_total={train_log['total']:.4f} val_total={val_log['total']:.4f} "
            f"val_nc={val_log['nc']:.4f} val_mse={val_log['mse']:.4f} "
            f"train_cons={train_log['cons']:.4f} val_cons={val_log['cons']:.4f} "
            f"lr_enc={lr_enc:.2e} lr_heads={lr_hds:.2e}"
        )

        if val_log['total'] < best_val_total:
            best_val_total = val_log['total']
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

    ckpt_path = os.path.join(config['output_dir'], 'cellvitpp.pth')
    torch.save(
        {
            'model_state_dict': best_state,
            'epoch': best_epoch,
            'val_total': best_val_total,
            'config': config,
        },
        ckpt_path,
    )
    print(f"[cellvitpp] saved best to {ckpt_path}  best_val_total={best_val_total:.4f} "
          f"@ epoch {best_epoch}")


if __name__ == '__main__':
    main()
