"""Diagnostic: NP confidence distributions for PanNuke-foreground vs
PanNuke-background pixels.  Tells us whether NP threshold sweeping has any
useful operating points before we invest in a full sweep with watershed.
"""

import argparse
import importlib.util

import numpy as np
import torch
from torch.utils.data import DataLoader

from cellvit.datasets import SynchronizedTransform
from adios_cellvit.adios_backbone import load_adios_mask_model
from adios_cellvit.adios_cellvit_model import ADIOSCellViT
from adios_cellvit.channel_selector import ChannelSelector
from adios_cellvit.pannuke_dataset import ADIOSPanNukeDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='./logs/stage2/stage2_adios_cellvit.pth')
    p.add_argument('--adios_checkpoint',
                   default='/data1/vanderbc/nandas1/ADIOS-CellViT/logs/checkpoint_iter_00094000.pth')
    p.add_argument('--pannuke_path',
                   default='/data1/vanderbc/test_dinov2_swaraj/ADIOS/data/pannuke')
    p.add_argument('--config', default='configs/nuclei_counter.py')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--max_patches', type=int, default=200)
    return p.parse_args()


def _load_stage2_config(config_path):
    spec = importlib.util.spec_from_file_location('cfg', config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.STAGE2


def main():
    args = parse_args()
    config = _load_stage2_config(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build model + load checkpoint
    mask_model = load_adios_mask_model(args.adios_checkpoint, device)
    selector = ChannelSelector(num_masks=3).to(device)
    model = ADIOSCellViT(
        mask_model=mask_model, selector=selector,
        num_classes=config['num_classes'], drop_rate=0.1,
        inference_mode='argmax',
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    model.set_inference_mode('argmax')

    val_transform = SynchronizedTransform(
        {'normalize': {'mean': config['normalize_mean'], 'std': config['normalize_std']}},
        input_shape=224,
    )
    test_dataset = ADIOSPanNukeDataset(
        data_dir=args.pannuke_path, split='Test',
        magnification=config['magnification'], transform=val_transform,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Pixel-level accumulators
    # 100-bin histogram from 0 to 1 (each bin width 0.01) so we can also report
    # the tail distributions accurately.
    n_bins = 100
    fg_hist = np.zeros(n_bins, dtype=np.int64)
    bg_hist = np.zeros(n_bins, dtype=np.int64)
    total_fg_px = 0
    total_bg_px = 0

    print(f'Running on up to {args.max_patches} patches...')
    seen = 0
    with torch.no_grad():
        for image, mask_2ch, _, instance_mask, _ in test_loader:
            if seen >= args.max_patches: break
            image = image.to(device, non_blocking=True)
            output = model(image)
            np_arr = output['masks'].cpu().numpy()       # [B, 1, H, W]
            gt_fg = (instance_mask > 0).numpy()           # [B, 1, H, W] bool

            B = image.size(0)
            for b in range(B):
                if seen >= args.max_patches: break
                seen += 1
                np_flat = np_arr[b, 0].ravel()
                gt_flat = gt_fg[b, 0].ravel()
                # Bin the NP probability into 100 bins.
                # Clip to [0, 1-epsilon] so 1.0 lands in bin 99 not out-of-range.
                bins = np.clip((np_flat * n_bins).astype(np.int64), 0, n_bins - 1)
                fg_pixels_this = gt_flat.sum()
                total_fg_px += fg_pixels_this
                total_bg_px += gt_flat.size - fg_pixels_this
                # Histogram for fg pixels and bg pixels separately
                np.add.at(fg_hist, bins[gt_flat], 1)
                np.add.at(bg_hist, bins[~gt_flat], 1)

    print(f'Processed {seen} patches.')
    print(f'Total foreground pixels: {total_fg_px:,}')
    print(f'Total background pixels: {total_bg_px:,}')
    print(f'Foreground fraction:     {total_fg_px / (total_fg_px + total_bg_px):.3f}')
    print()

    # ========================================================
    # ASCII histograms (compressed to 20 bins for readability)
    # ========================================================
    bin_edges = np.linspace(0, 1, 21)  # 20 bins
    fg_compressed = np.zeros(20, dtype=np.int64)
    bg_compressed = np.zeros(20, dtype=np.int64)
    for j in range(20):
        lo = j * 5
        hi = (j + 1) * 5
        fg_compressed[j] = fg_hist[lo:hi].sum()
        bg_compressed[j] = bg_hist[lo:hi].sum()

    # Normalize to fraction
    fg_frac = fg_compressed / max(total_fg_px, 1)
    bg_frac = bg_compressed / max(total_bg_px, 1)

    print('=' * 78)
    print('NP probability distribution: PanNuke-foreground vs PanNuke-background')
    print('=' * 78)
    print(f'{"bin":>12s}  {"FG frac":>10s}  {"BG frac":>10s}  {"FG / BG ratio":>14s}  visual')
    print('-' * 78)
    for j in range(20):
        lo = bin_edges[j]
        hi = bin_edges[j + 1]
        f = fg_frac[j]
        b = bg_frac[j]
        ratio = f / max(b, 1e-9)
        # Bar plot: blue (fg) + red (bg), each normalized
        fg_bar = '#' * int(40 * f)
        bg_bar = '.' * int(40 * b)
        print(f'  [{lo:.2f}-{hi:.2f})  {f:>10.4f}  {b:>10.4f}  {ratio:>14.2f}  {fg_bar}|{bg_bar}')

    print('-' * 78)
    print('  Legend: # = foreground fraction in bin, . = background fraction.')
    print()

    # ========================================================
    # Operating point table — threshold sweep on pixels only
    # ========================================================
    print('=' * 78)
    print('Operating points (pixel-level only, no watershed yet)')
    print('=' * 78)
    print(f'{"threshold":>10s}  {"TP retained":>13s}  {"FP retained":>13s}  '
          f'{"recall":>8s}  {"precision":>10s}  {"IoU":>8s}')
    print('-' * 78)
    for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]:
        # Sum bins where bin_lower >= t
        # bin j covers [j/100, (j+1)/100); pixel is "kept" iff probability >= t
        # so we keep bins j where j/100 >= t  →  j >= ceil(t*100)
        k_start = int(np.ceil(t * 100))
        tp = fg_hist[k_start:].sum()
        fp = bg_hist[k_start:].sum()
        fn = total_fg_px - tp
        recall = tp / max(total_fg_px, 1)
        precision = tp / max(tp + fp, 1)
        iou = tp / max(tp + fp + fn, 1)
        tp_retained = tp / max(total_fg_px, 1)
        fp_retained = fp / max(total_bg_px, 1)
        print(f'  {t:>8.2f}    {tp_retained:>13.4f}  {fp_retained:>13.4f}  '
              f'{recall:>8.4f}  {precision:>10.4f}  {iou:>8.4f}')

    print()
    print('Reading the table:')
    print('  TP retained: of PanNuke-foreground pixels, fraction with NP prob >= t.')
    print('  FP retained: of PanNuke-background pixels, fraction with NP prob >= t.')
    print('  A useful operating point has high TP retained, low FP retained.')
    print()
    print('If TP and FP retention drop at the SAME rate as we raise t, threshold')
    print('does NOTHING for us — the NP map is non-discriminative.  We need')
    print('Option 2 (retrain with corrected loss) in that case.')


if __name__ == '__main__':
    main()
