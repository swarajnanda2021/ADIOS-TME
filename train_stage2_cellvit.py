"""Stage 2: train HoVer + NC heads, fine-tune ADIOS decoder at low LR.

Stage 1's selector is loaded and parameter-frozen, but still called so that
gradients flow through its softmax weights into the ADIOS mask decoder.
The ADIOS encoder stays frozen throughout.
"""

import argparse
import copy
import importlib
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler

from cellvit.datasets import PanNukeDataset, SynchronizedTransform
from cellvit.utils import WarmupDecayScheduler, set_seed

from adios_cellvit.adios_backbone import load_adios_backbone_and_decoder
from adios_cellvit.adios_cellvit_model import ADIOSCellViT
from adios_cellvit.channel_selector import ChannelSelector


# TODO(cluster-assembly): HANDOFF §6.2 — confirm HoverNetBasedDataset returns
# a 5th tensor `class_mask` [B, H, W] LongTensor with PanNuke class indices
# in {0, ..., 5} (0=background). If not, extend HoverNetBasedDataset to add
# this output. Until then, this script's dataloader unpack will fail at the
# 5-tuple destructure below.


class CombinedLossWithNC(nn.Module):
    """HoVer-Net style combined loss + a per-pixel NC cross-entropy term.

    The NP path here receives ``pred_mask`` as ``[B, 1, H, W]`` (already a
    probability in [0, 1] from the ADIOS+selector collapse), unlike the
    original CellViT loss which expected ``[B, 2, H, W]`` softmax logits.
    We rebuild a 2-channel form ``[fg, bg]`` and skip the softmax that the
    upstream xentropy/dice paths would otherwise apply.
    """

    def __init__(self, weights: dict):
        super().__init__()
        self.w = weights

    @staticmethod
    def _msge_loss(pred_dist: torch.Tensor, true_dist: torch.Tensor) -> torch.Tensor:
        # Squared error of spatial gradients of the H/V maps.
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

    @staticmethod
    def _dice_loss(prob_2ch: torch.Tensor, true_2ch: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        inter = (prob_2ch * true_2ch).sum(dim=(0, 2, 3))
        union = prob_2ch.sum(dim=(0, 2, 3)) + true_2ch.sum(dim=(0, 2, 3))
        return 1.0 - ((2.0 * inter + eps) / (union + eps)).mean()

    @staticmethod
    def _xentropy_2ch(prob_2ch: torch.Tensor, true_2ch: torch.Tensor) -> torch.Tensor:
        eps = 1e-7
        prob = prob_2ch.clamp(min=eps, max=1.0 - eps)
        return -(true_2ch * prob.log()).sum(dim=1).mean()

    def forward(
        self,
        true_mask: torch.Tensor,   # [B, 2, H, W] one-hot
        pred_mask: torch.Tensor,   # [B, 1, H, W] probability in [0, 1]
        true_dist: torch.Tensor,   # [B, 2, H, W]
        pred_dist: torch.Tensor,   # [B, 2, H, W] raw
        true_class: torch.Tensor,  # [B, H, W] long
        pred_class: torch.Tensor,  # [B, K, H, W] raw
    ):
        prob_fg = pred_mask.clamp(0.0, 1.0)
        prob_bg = 1.0 - prob_fg
        prob_2ch = torch.cat([prob_fg, prob_bg], dim=1)

        l_xent = self._xentropy_2ch(prob_2ch, true_mask)
        l_dice = self._dice_loss(prob_2ch, true_mask)
        l_mse = F.mse_loss(pred_dist, true_dist)
        l_msge = self._msge_loss(pred_dist, true_dist)
        l_nc = F.cross_entropy(pred_class, true_class)

        total = (
            self.w['w_xentropy'] * l_xent
            + self.w['w_dice'] * l_dice
            + self.w['w_mse'] * l_mse
            + self.w['w_msge'] * l_msge
            + self.w['w_nc'] * l_nc
        )
        return total, {
            'xent': l_xent.item(),
            'dice': l_dice.item(),
            'mse': l_mse.item(),
            'msge': l_msge.item(),
            'nc': l_nc.item(),
            'total': total.item(),
        }


def _load_config(config_path: str) -> dict:
    spec = importlib.util.spec_from_file_location('nuclei_counter_cfg', config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.STAGE2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/nuclei_counter.py')
    p.add_argument('--adios_checkpoint', default=None)
    p.add_argument('--stage1_selector', default=None)
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

    train_dataset = PanNukeDataset(
        data_dir=config['pannuke_path'],
        split='Training',
        magnification=config['magnification'],
        transform=train_transform,
    )
    _ = PanNukeDataset(
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


def run_epoch(model, criterion, loader, optimizer, device, train: bool):
    model.train(mode=train)
    # Safeguards: even when frozen, ensure no dropout/BN drift.
    model.encoder.eval()
    model.selector.eval()

    totals = {'xent': 0.0, 'dice': 0.0, 'mse': 0.0, 'msge': 0.0, 'nc': 0.0, 'total': 0.0}
    n_seen = 0

    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for image, mask_2ch, distance_map, _, class_mask in loader:
            image = image.to(device, non_blocking=True)
            mask_2ch = mask_2ch.to(device, non_blocking=True).float()
            distance_map = distance_map.to(device, non_blocking=True).float()
            class_mask = class_mask.to(device, non_blocking=True).long()

            output = model(image)
            loss, log = criterion(
                true_mask=mask_2ch,
                pred_mask=output['masks'],
                true_dist=distance_map,
                pred_dist=output['distances'],
                true_class=class_mask,
                pred_class=output['nuclei_types'],
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


def main():
    args = parse_args()
    config = _load_config(args.config)
    for key in ('adios_checkpoint', 'stage1_selector', 'pannuke_path', 'output_dir'):
        cli_val = getattr(args, key)
        if cli_val is not None:
            config[key] = cli_val

    set_seed(config['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(config['output_dir'], exist_ok=True)

    encoder, mask_decoder = load_adios_backbone_and_decoder(
        config['adios_checkpoint'], device,
    )
    selector = ChannelSelector(num_masks=3).to(device)
    stage1_ckpt = torch.load(config['stage1_selector'], map_location=device, weights_only=False)
    selector.load_state_dict(stage1_ckpt['selector_state_dict'])

    model = ADIOSCellViT(
        encoder=encoder,
        mask_decoder=mask_decoder,
        selector=selector,
        encoder_dim=768,
        num_classes=config['num_classes'],
        drop_rate=0.1,
        inference_mode='soft',
    ).to(device)

    model.freeze_selector()
    model.unfreeze_mask_decoder()

    criterion = CombinedLossWithNC(config['loss_weights'])

    heads_params = (
        list(model.cellvit.hv_map_decoder.parameters())
        + list(model.cellvit.nuclei_type_map_decoder.parameters())
        + list(model.cellvit.decoder0.parameters())
        + list(model.cellvit.decoder1.parameters())
        + list(model.cellvit.decoder2.parameters())
        + list(model.cellvit.decoder3.parameters())
    )
    adios_params = list(model.mask_decoder.parameters())

    optimizer = torch.optim.AdamW(
        [
            {'params': heads_params, 'lr': config['lr_heads']},
            {'params': adios_params, 'lr': config['lr_adios_decoder']},
        ],
        weight_decay=config['weight_decay'],
    )
    # Scheduler scales group 0 (heads); ADIOS decoder LR stays at 1e-6.
    scheduler = WarmupDecayScheduler(
        optimizer,
        warmup_epochs=config['warmup_epochs'],
        total_epochs=config['max_epochs'],
        base_lr=config['lr_heads'],
        final_lr=config['lr_heads'] / 10,
        warmup_start_lr=config['lr_heads'] / 100,
    )

    train_loader, val_loader = build_loaders(config)

    best_val_loss = float('inf')
    best_state = None
    epochs_since_improve = 0

    for epoch in range(config['max_epochs']):
        train_log = run_epoch(model, criterion, train_loader, optimizer, device, train=True)
        val_log = run_epoch(model, criterion, val_loader, optimizer, device, train=False)
        scheduler.step()

        print(
            f"[stage2] epoch {epoch+1}/{config['max_epochs']} "
            f"train_total={train_log['total']:.4f} val_total={val_log['total']:.4f} "
            f"val_nc={val_log['nc']:.4f} val_mse={val_log['mse']:.4f}"
        )

        if val_log['total'] < best_val_loss:
            best_val_loss = val_log['total']
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= config['early_stop_patience']:
                print(f"[stage2] early stop at epoch {epoch+1}")
                break

    ckpt_path = os.path.join(config['output_dir'], 'stage2_adios_cellvit.pth')
    torch.save(
        {'model_state_dict': best_state, 'best_val_loss': best_val_loss, 'config': config},
        ckpt_path,
    )
    print(f"[stage2] saved best to {ckpt_path}  best_val_loss={best_val_loss:.4f}")


if __name__ == '__main__':
    main()
