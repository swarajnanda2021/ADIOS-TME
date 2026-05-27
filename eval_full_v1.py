"""Combined Path A (AJI/PQ/count) + Path B (NP recall/precision, NC confusion,
extras) eval for ADIOSCellViT on PanNuke Test.

Writes:
- logs/eval/eval_report_v1.txt   (ASCII; cat-friendly)
- logs/eval/eval_results_v1.json (raw values)
"""

import argparse
import importlib.util
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from cellvit.datasets import SynchronizedTransform
from cellvit.postproc.benchmarking import (
    __proc_np_hv,
    aggregated_jaccard_index,
)

from adios_cellvit.adios_backbone import load_adios_mask_model
from adios_cellvit.adios_cellvit_model import ADIOSCellViT
from adios_cellvit.channel_selector import ChannelSelector
from adios_cellvit.pannuke_dataset import ADIOSPanNukeDataset


PANNUKE_CLASS_NAMES = {
    0: 'background', 1: 'neoplastic', 2: 'inflammatory',
    3: 'connective', 4: 'dead', 5: 'epithelial',
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='./logs/stage2/stage2_adios_cellvit.pth')
    p.add_argument('--adios_checkpoint',
                   default='/data1/vanderbc/nandas1/ADIOS-CellViT/logs/checkpoint_iter_00094000.pth')
    p.add_argument('--pannuke_path',
                   default='/data1/vanderbc/test_dinov2_swaraj/ADIOS/data/pannuke')
    p.add_argument('--config', default='configs/nuclei_counter.py')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--output_dir', default='./logs/eval')
    p.add_argument('--iou_match_threshold', type=float, default=0.5)
    p.add_argument('--max_patches', type=int, default=None,
                   help='Cap on number of patches (for quick sanity runs).')
    return p.parse_args()


def _load_stage2_config(config_path):
    spec = importlib.util.spec_from_file_location('cfg', config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.STAGE2


def match_instances_by_iou(pred_inst, gt_inst, iou_threshold=0.5):
    """Greedy IoU matching between predicted and GT instance maps.

    Args:
        pred_inst: [H, W] int array; 0 = background, 1..P = instance IDs.
        gt_inst:   [H, W] int array; 0 = background, 1..G = instance IDs.

    Returns:
        matches:  list of (pred_id, gt_id, iou)
        unmatched_pred:  list of pred_ids that did not match
        unmatched_gt:    list of gt_ids that did not match
    """
    pred_ids = sorted(np.unique(pred_inst).tolist())
    gt_ids = sorted(np.unique(gt_inst).tolist())
    pred_ids = [i for i in pred_ids if i > 0]
    gt_ids = [i for i in gt_ids if i > 0]

    # Build IoU matrix only for overlapping pairs.
    pred_masks = {p: (pred_inst == p) for p in pred_ids}
    gt_masks = {g: (gt_inst == g) for g in gt_ids}

    pairs = []
    for p in pred_ids:
        pm = pred_masks[p]
        for g in gt_ids:
            gm = gt_masks[g]
            inter = int((pm & gm).sum())
            if inter == 0:
                continue
            union = int((pm | gm).sum())
            iou = inter / union if union > 0 else 0.0
            if iou >= iou_threshold:
                pairs.append((iou, p, g))

    # Greedy: highest IoU first; each pred and each gt claimed at most once.
    pairs.sort(key=lambda x: -x[0])
    claimed_pred, claimed_gt = set(), set()
    matches = []
    for iou, p, g in pairs:
        if p in claimed_pred or g in claimed_gt:
            continue
        claimed_pred.add(p)
        claimed_gt.add(g)
        matches.append((p, g, iou))

    unmatched_pred = [p for p in pred_ids if p not in claimed_pred]
    unmatched_gt = [g for g in gt_ids if g not in claimed_gt]
    return matches, unmatched_pred, unmatched_gt


def instance_modal_class(inst_mask, class_map, instance_id):
    """Return the most common class label inside an instance."""
    pixels = class_map[inst_mask == instance_id]
    if pixels.size == 0:
        return 0
    vals, counts = np.unique(pixels, return_counts=True)
    return int(vals[counts.argmax()])


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    config = _load_stage2_config(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('=== Setup ===')
    mask_model = load_adios_mask_model(args.adios_checkpoint, device)
    selector = ChannelSelector(num_masks=3).to(device)
    model = ADIOSCellViT(
        mask_model=mask_model,
        selector=selector,
        num_classes=config['num_classes'],
        drop_rate=0.1,
        inference_mode='argmax',
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    model.set_inference_mode('argmax')
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Loaded checkpoint, {n_params:,} params, best_val_loss={state.get("best_val_loss", "?")}')

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
    n_test = len(test_dataset) if args.max_patches is None else min(args.max_patches, len(test_dataset))
    print(f'Test patches: {n_test} (of {len(test_dataset)} total), batch_size={args.batch_size}, device={device}')

    # ---- Accumulators ----
    # NP pixel-level (Path B Section 2)
    np_total_gt = 0
    np_total_pred = 0
    np_total_inter = 0
    np_total_pred_extra = 0
    np_total_gt_extra = 0

    # Instance-level (Section 3) — INSTANCE_PQ accumulators (global, not per-patch)
    aji_scores = []
    gt_counts = []
    pred_counts = []
    count_diffs = []  # signed: pred - gt
    # Global PQ accumulators
    pq_iou_sum = 0.0          # sum of IoU over matched pairs (dataset-wide)
    pq_tp = 0                 # matched pairs
    pq_fp = 0                 # unmatched predicted instances
    pq_fn = 0                 # unmatched GT instances
    # Per-class PQ accumulators (keys 1..5)
    pq_class_iou_sum = {c: 0.0 for c in [1, 2, 3, 4, 5]}
    pq_class_tp = {c: 0 for c in [1, 2, 3, 4, 5]}
    pq_class_fp = {c: 0 for c in [1, 2, 3, 4, 5]}
    pq_class_fn = {c: 0 for c in [1, 2, 3, 4, 5]}

    # Classification on matched pairs (Section 4)
    # confusion[(gt_class, pred_class)] = count
    confusion = Counter()
    n_matched = 0

    # Extras (Section 5)
    extras_class_counts = Counter()  # NC predicted class for unmatched preds
    extras_per_patch = []

    # ---- Inference loop ----
    seen = 0
    t0 = time.time()
    with torch.no_grad():
        for image, mask_2ch, distance_map, instance_mask, class_mask in test_loader:
            if args.max_patches is not None and seen >= args.max_patches:
                break
            image = image.to(device, non_blocking=True)
            output = model(image)

            np_arr = output['masks'].cpu().numpy()           # [B, 1, H, W]
            hv_arr = output['distances'].cpu().numpy()       # [B, 2, H, W]
            nc_arr = output['nuclei_types'].argmax(1).cpu().numpy()  # [B, H, W]

            B = image.size(0)
            for b in range(B):
                if args.max_patches is not None and seen >= args.max_patches:
                    break
                seen += 1

                # Build the per-patch HWC tensor for watershed: ch0=NP, ch1=H, ch2=V.
                pred_hwc = np.stack([np_arr[b, 0], hv_arr[b, 0], hv_arr[b, 1]], axis=-1)
                pred_inst = __proc_np_hv(pred_hwc, mask_threshold=0.5, overall_threshold=0.4)

                gt_inst = instance_mask[b, 0].cpu().numpy().astype(np.int64)
                gt_cls = class_mask[b].cpu().numpy().astype(np.int64)
                pred_cls = nc_arr[b]

                # --- NP pixel metrics ---
                gt_fg = (gt_inst > 0)
                pred_fg = (np_arr[b, 0] > 0.5)
                inter = int((gt_fg & pred_fg).sum())
                np_total_gt += int(gt_fg.sum())
                np_total_pred += int(pred_fg.sum())
                np_total_inter += inter
                np_total_pred_extra += int((pred_fg & ~gt_fg).sum())
                np_total_gt_extra += int((~pred_fg & gt_fg).sum())

                # --- Instance metrics ---
                gt_n = int(len(np.unique(gt_inst)) - (1 if 0 in gt_inst else 0))
                pred_n = int(len(np.unique(pred_inst)) - (1 if 0 in pred_inst else 0))
                gt_counts.append(gt_n)
                pred_counts.append(pred_n)
                count_diffs.append(pred_n - gt_n)

                aji_scores.append(float(aggregated_jaccard_index(gt_inst, pred_inst)))

                # --- Instance matching (used for PQ, classification, extras) ---
                matches, unmatched_pred, unmatched_gt = match_instances_by_iou(
                    pred_inst, gt_inst, iou_threshold=args.iou_match_threshold,
                )

                # --- Instance PQ accumulators (dataset-global) ---
                pq_tp += len(matches)
                pq_fp += len(unmatched_pred)
                pq_fn += len(unmatched_gt)
                for _p, _g, iou in matches:
                    pq_iou_sum += iou

                # --- Section 4: confusion + per-class PQ ---
                for p_id, g_id, iou in matches:
                    g_class = instance_modal_class(gt_inst, gt_cls, g_id)
                    p_class = instance_modal_class(pred_inst, pred_cls, p_id)
                    confusion[(g_class, p_class)] += 1
                    n_matched += 1
                    # Per-class PQ: a TP for class c requires GT class == c AND pred class == c.
                    if g_class == p_class and g_class in pq_class_tp:
                        pq_class_tp[g_class] += 1
                        pq_class_iou_sum[g_class] += iou
                    else:
                        # Same-instance match but class disagreement: FP for the
                        # predicted class, FN for the GT class (standard panoptic
                        # treatment of class-mismatched matches).
                        if p_class in pq_class_fp:
                            pq_class_fp[p_class] += 1
                        if g_class in pq_class_fn:
                            pq_class_fn[g_class] += 1
                # Unmatched predictions: FP for their predicted class.
                for p_id in unmatched_pred:
                    p_class = instance_modal_class(pred_inst, pred_cls, p_id)
                    if p_class in pq_class_fp:
                        pq_class_fp[p_class] += 1
                # Unmatched GTs: FN for their GT class.
                for g_id in unmatched_gt:
                    g_class = instance_modal_class(gt_inst, gt_cls, g_id)
                    if g_class in pq_class_fn:
                        pq_class_fn[g_class] += 1

                # --- Section 5: extras (unmatched preds) ---
                # NOTE: unmatched_pred from the matching above is reused here.
                # For per-patch extras count we re-iterate but don't re-match.
                extras_count_this_patch = 0
                for p_id in unmatched_pred:
                    p_class = instance_modal_class(pred_inst, pred_cls, p_id)
                    extras_class_counts[p_class] += 1
                    extras_count_this_patch += 1
                extras_per_patch.append(extras_count_this_patch)

            if seen % 200 == 0:
                dt = time.time() - t0
                print(f'  processed {seen}/{n_test} patches ({dt:.1f}s elapsed, {dt/max(seen,1):.2f}s/patch)')

    total_time = time.time() - t0
    print(f'\nLoop complete in {total_time:.1f}s ({total_time/max(seen,1):.2f}s/patch avg)')

    # ============================================================
    # AGGREGATE
    # ============================================================
    n_patches = seen

    # Section 2 — NP pixel-level
    np_iou = np_total_inter / max(np_total_inter + np_total_pred_extra + np_total_gt_extra, 1)
    np_recall = np_total_inter / max(np_total_gt, 1)
    np_precision = np_total_inter / max(np_total_pred, 1)
    np_extras_ratio = np_total_pred_extra / max(np_total_gt, 1)

    # Section 3 — Instance metrics
    aji_mean, aji_std = float(np.mean(aji_scores)), float(np.std(aji_scores))
    mae_count = float(np.mean([abs(d) for d in count_diffs]))
    mean_count_diff = float(np.mean(count_diffs))  # signed bias

    # Instance PQ (global, standard panoptic definition)
    pq_sq = pq_iou_sum / max(pq_tp, 1)
    pq_rq = pq_tp / max(pq_tp + 0.5 * pq_fp + 0.5 * pq_fn, 1)
    pq_overall = pq_sq * pq_rq

    pq_per_class_results = {}
    for c in [1, 2, 3, 4, 5]:
        tp = pq_class_tp[c]
        fp = pq_class_fp[c]
        fn = pq_class_fn[c]
        sq = pq_class_iou_sum[c] / max(tp, 1)
        rq = tp / max(tp + 0.5 * fp + 0.5 * fn, 1)
        pq_per_class_results[c] = {
            'tp': tp, 'fp': fp, 'fn': fn,
            'sq': sq, 'rq': rq, 'pq': sq * rq,
        }

    # Count-error histogram (signed bins)
    diff_arr = np.array(count_diffs)
    diff_bins = {
        '<=-5':      int((diff_arr <= -5).sum()),
        '-4..-2':    int(((diff_arr >= -4) & (diff_arr <= -2)).sum()),
        '-1':        int((diff_arr == -1).sum()),
        ' 0':        int((diff_arr == 0).sum()),
        '+1':        int((diff_arr == 1).sum()),
        '+2..+4':    int(((diff_arr >= 2) & (diff_arr <= 4)).sum()),
        '>=+5':      int((diff_arr >= 5).sum()),
    }

    # Section 4 — per-class P/R/F1 on matched-pair classifications
    fg_classes = [1, 2, 3, 4, 5]
    class_pr = {}
    for c in fg_classes:
        tp = confusion.get((c, c), 0)
        fp = sum(confusion.get((g, c), 0) for g in fg_classes if g != c)
        fn = sum(confusion.get((c, p), 0) for p in fg_classes if p != c)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        class_pr[c] = {'tp': tp, 'fp': fp, 'fn': fn,
                       'precision': prec, 'recall': rec, 'f1': f1}
    macro_f1 = float(np.mean([class_pr[c]['f1'] for c in fg_classes]))

    # Section 5 — extras distribution
    total_extras = sum(extras_class_counts.values())
    extras_dist = {
        f"class_{c}_{PANNUKE_CLASS_NAMES.get(c, '?')}":
            extras_class_counts.get(c, 0) / max(total_extras, 1)
        for c in range(6)
    }
    extras_per_patch_mean = float(np.mean(extras_per_patch))

    # ============================================================
    # WRITE REPORT
    # ============================================================
    lines = []
    L = lines.append

    L('=' * 78)
    L(f'  ADIOSCellViT Evaluation Report (v1)')
    L('=' * 78)
    L('')

    L('--- Section 1: Setup ---')
    L(f'  Checkpoint:       {args.checkpoint}')
    L(f'  Param count:      {n_params:,}')
    L(f'  Test patches:     {n_patches}')
    L(f'  Batch size:       {args.batch_size}')
    L(f'  Device:           {device}')
    L(f'  IoU match thresh: {args.iou_match_threshold}')
    L(f'  Eval time:        {total_time:.1f}s ({total_time/max(n_patches,1):.2f}s/patch)')
    L('')

    L('--- Section 2: NP pixel-level metrics (binary nuclei map vs PanNuke) ---')
    L(f'  GT-foreground pixels:        {np_total_gt:>12,d}')
    L(f'  Predicted-foreground pixels: {np_total_pred:>12,d}')
    L(f'  Intersection:                {np_total_inter:>12,d}')
    L(f'  Predicted-only (extras):     {np_total_pred_extra:>12,d}')
    L(f'  GT-only (missed):            {np_total_gt_extra:>12,d}')
    L('')
    L(f'  IoU:               {np_iou:.4f}')
    L(f'  Recall:            {np_recall:.4f}   (of PanNuke nuclei, fraction model finds)')
    L(f'  Precision:         {np_precision:.4f}   (of model predictions, fraction PanNuke labels)')
    L(f'  Extras-ratio:      {np_extras_ratio:.4f}   (extras px / GT px)')
    L('')

    L('--- Section 3: Instance metrics (post-watershed) ---')
    L(f'  Mean GT count/patch:    {float(np.mean(gt_counts)):.2f}   (std {float(np.std(gt_counts)):.2f})')
    L(f'  Mean pred count/patch:  {float(np.mean(pred_counts)):.2f}   (std {float(np.std(pred_counts)):.2f})')
    L(f'  Count MAE (|pred-gt|):  {mae_count:.2f}')
    L(f'  Count bias (pred-gt):   {mean_count_diff:+.2f}   (positive = over-predicting)')
    L(f'  AJI:                    {aji_mean:.4f}  +/- {aji_std:.4f}  (per-patch mean)')
    L('')
    L('  Instance PQ (global, IoU>=0.5 matching, standard panoptic definition):')
    L(f'    matched pairs (TP):    {pq_tp}')
    L(f'    unmatched pred (FP):   {pq_fp}')
    L(f'    unmatched GT   (FN):   {pq_fn}')
    L(f'    SQ (mean IoU on TPs):  {pq_sq:.4f}')
    L(f'    RQ (det F1-like):      {pq_rq:.4f}')
    L(f'    PQ (SQ x RQ):          {pq_overall:.4f}')
    L('')
    L('  Per-class instance PQ (class assignment via per-instance modal NC):')
    L(f'    {"class":>14s} {"TP":>5s} {"FP":>5s} {"FN":>5s}   {"SQ":>6s}   {"RQ":>6s}   {"PQ":>6s}')
    for c in [1, 2, 3, 4, 5]:
        d = pq_per_class_results[c]
        L(f'    {PANNUKE_CLASS_NAMES[c]:>14s} {d["tp"]:>5d} {d["fp"]:>5d} {d["fn"]:>5d}   '
          f'{d["sq"]:>6.3f}   {d["rq"]:>6.3f}   {d["pq"]:>6.3f}')
    L('')
    L('  Count-error histogram:')
    for lab, n in diff_bins.items():
        bar = '#' * int(50 * n / max(n_patches, 1))
        L(f'    {lab:>8s}: {n:>5d}  {bar}')
    L('')

    L('--- Section 4: Classification quality on matched instances (IoU > thresh) ---')
    L(f'  Matched pairs:  {n_matched}')
    L('')
    L('  Confusion matrix (rows=GT, cols=Pred; only foreground classes):')
    header_classes = [0] + fg_classes  # include "predicted as background" too
    L('              | ' + ' | '.join(f'pred {c}' for c in header_classes) + ' |')
    L('  ' + '-' * 76)
    for g in fg_classes:
        row = [confusion.get((g, p), 0) for p in header_classes]
        L(f'    gt {g} {PANNUKE_CLASS_NAMES[g][:7]:>7s} | ' +
          ' | '.join(f'{v:>6d}' for v in row) + ' |')
    L('')
    L('  Per-class P/R/F1:')
    L(f'    {"class":>14s} {"tp":>6s} {"fp":>6s} {"fn":>6s}    {"P":>6s}   {"R":>6s}   {"F1":>6s}')
    for c in fg_classes:
        d = class_pr[c]
        L(f'    {PANNUKE_CLASS_NAMES[c]:>14s} '
          f'{d["tp"]:>6d} {d["fp"]:>6d} {d["fn"]:>6d}    '
          f'{d["precision"]:>6.3f}   {d["recall"]:>6.3f}   {d["f1"]:>6.3f}')
    L(f'  Macro-F1 (foreground only): {macro_f1:.4f}')
    L('')

    L('--- Section 5: Extras (predicted instances with no GT match) ---')
    L(f'  Total extras across test set: {total_extras}')
    L(f'  Mean extras / patch:          {extras_per_patch_mean:.2f}')
    L('')
    L('  NC prediction on extras:')
    for c in range(6):
        n = extras_class_counts.get(c, 0)
        frac = n / max(total_extras, 1)
        bar = '#' * int(60 * frac)
        L(f'    class {c} {PANNUKE_CLASS_NAMES[c]:>13s}: {n:>5d}  ({frac:.3f})  {bar}')
    L('')
    L('  Interpretation: extras predicted as class 0 (background) will be')
    L('  surfaced to the user as "unknown nucleus" via the post-hoc rule.')
    L('  High class-0 share = the rule works; high other-class shares = the')
    L('  rule is less effective and fellows would see noisy classifications.')
    L('')

    L('--- Section 6: Summary ---')
    L(f'  Path A (PanNuke benchmark):  AJI={aji_mean:.3f}  PQ={pq_overall:.3f}  count_MAE={mae_count:.2f}')
    L(f'  Path B (product use case):   NP recall={np_recall:.3f}  NP precision={np_precision:.3f}  NC macro-F1={macro_f1:.3f}')
    L(f'  Extras:                      {total_extras} total ({extras_per_patch_mean:.2f}/patch); '
      f'{extras_class_counts.get(0, 0)/max(total_extras,1):.0%} predicted as background ("unknown")')
    L('')
    L('=' * 78)

    report_text = '\n'.join(lines)
    report_path = os.path.join(args.output_dir, 'eval_report_v1.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(report_text)
    print(f'\nReport written to: {report_path}')

    # JSON dump for programmatic use
    json_results = {
        'checkpoint': args.checkpoint,
        'n_patches': n_patches,
        'np_pixel': {
            'gt_fg': np_total_gt, 'pred_fg': np_total_pred, 'inter': np_total_inter,
            'pred_extra': np_total_pred_extra, 'gt_extra': np_total_gt_extra,
            'iou': np_iou, 'recall': np_recall, 'precision': np_precision,
            'extras_ratio': np_extras_ratio,
        },
        'instance': {
            'mean_gt_count': float(np.mean(gt_counts)),
            'mean_pred_count': float(np.mean(pred_counts)),
            'count_mae': mae_count,
            'count_bias': mean_count_diff,
            'aji_mean': aji_mean, 'aji_std': aji_std,
            'pq': {
                'tp': pq_tp, 'fp': pq_fp, 'fn': pq_fn,
                'sq': pq_sq, 'rq': pq_rq, 'pq': pq_overall,
            },
            'pq_per_class': {PANNUKE_CLASS_NAMES[c]: pq_per_class_results[c]
                             for c in [1, 2, 3, 4, 5]},
            'count_error_histogram': diff_bins,
        },
        'classification': {
            'n_matched': n_matched,
            'macro_f1': macro_f1,
            'per_class': {PANNUKE_CLASS_NAMES[c]: class_pr[c] for c in fg_classes},
            'confusion': {f'gt_{g}_pred_{p}': v for (g, p), v in confusion.items()},
        },
        'extras': {
            'total': total_extras,
            'mean_per_patch': extras_per_patch_mean,
            'class_distribution': {PANNUKE_CLASS_NAMES[c]: extras_class_counts.get(c, 0)
                                   for c in range(6)},
        },
    }
    json_path = os.path.join(args.output_dir, 'eval_results_v1.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f'JSON written to: {json_path}')


if __name__ == '__main__':
    main()
