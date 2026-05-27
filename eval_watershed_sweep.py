"""Watershed parameter sweep for ADIOSCellViT.

Runs inference once on the full PanNuke Test split, caches NP/HV outputs,
then sweeps (mask_threshold, overall_threshold) and recomputes instance
metrics for each combination.  Tells us whether under-prediction is a
post-processing problem or a model problem.
"""

import argparse
import importlib.util
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from cellvit.datasets import SynchronizedTransform
from cellvit.postproc.benchmarking import __proc_np_hv, aggregated_jaccard_index

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
    p.add_argument('--max_patches', type=int, default=None)
    return p.parse_args()


def _load_stage2_config(config_path):
    spec = importlib.util.spec_from_file_location('cfg', config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.STAGE2


def match_instances_by_iou(pred_inst, gt_inst, iou_threshold=0.5):
    """Greedy IoU matching. Returns matches, unmatched_pred, unmatched_gt."""
    pred_ids = [i for i in np.unique(pred_inst).tolist() if i > 0]
    gt_ids = [i for i in np.unique(gt_inst).tolist() if i > 0]
    if not pred_ids or not gt_ids:
        return [], pred_ids, gt_ids

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

    pairs.sort(key=lambda x: -x[0])
    claimed_pred, claimed_gt = set(), set()
    matches = []
    for iou, p, g in pairs:
        if p in claimed_pred or g in claimed_gt:
            continue
        claimed_pred.add(p); claimed_gt.add(g)
        matches.append((p, g, iou))

    return (matches,
            [p for p in pred_ids if p not in claimed_pred],
            [g for g in gt_ids if g not in claimed_gt])


def instance_modal_class(inst_mask, class_map, instance_id):
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

    # ============================================================
    # Build model
    # ============================================================
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
    n_test = len(test_dataset) if args.max_patches is None else min(args.max_patches, len(test_dataset))
    print(f'Test patches: {n_test}')

    # ============================================================
    # PHASE 1: cache NP/HV/NC predictions + GT for all patches
    # ============================================================
    print('Caching inference outputs...')
    t0 = time.time()
    cache = []   # list of dicts: {np_map, hv_map, nc_map, gt_inst, gt_cls}
    seen = 0
    with torch.no_grad():
        for image, mask_2ch, distance_map, instance_mask, class_mask in test_loader:
            if args.max_patches is not None and seen >= args.max_patches:
                break
            image = image.to(device, non_blocking=True)
            output = model(image)
            np_arr = output['masks'].cpu().numpy()              # [B, 1, H, W]
            hv_arr = output['distances'].cpu().numpy()          # [B, 2, H, W]
            nc_arr = output['nuclei_types'].argmax(1).cpu().numpy()  # [B, H, W]

            B = image.size(0)
            for b in range(B):
                if args.max_patches is not None and seen >= args.max_patches:
                    break
                seen += 1
                cache.append({
                    'np_map':  np_arr[b, 0].astype(np.float32),
                    'hv_map':  hv_arr[b].astype(np.float32),       # [2, H, W]
                    'nc_map':  nc_arr[b].astype(np.int64),
                    'gt_inst': instance_mask[b, 0].cpu().numpy().astype(np.int64),
                    'gt_cls':  class_mask[b].cpu().numpy().astype(np.int64),
                })
            if seen % 400 == 0:
                print(f'  cached {seen}/{n_test} ({time.time()-t0:.1f}s)')

    print(f'Cache complete: {len(cache)} patches in {time.time()-t0:.1f}s')

    # ============================================================
    # PHASE 2: sweep thresholds
    # ============================================================
    mask_thresholds = [0.3, 0.4, 0.5]
    overall_thresholds = [0.2, 0.3, 0.4, 0.5]
    combos = [(mt, ot) for mt in mask_thresholds for ot in overall_thresholds]

    # baseline (current settings) listed explicitly so it's always visible
    if (0.5, 0.4) not in combos:
        combos.append((0.5, 0.4))

    print(f'\nSweeping {len(combos)} (mask_thresh, overall_thresh) combinations')
    print('=' * 78)

    results = []  # list of dicts: one per combo
    for mt, ot in combos:
        t_start = time.time()
        # Per-combo accumulators
        aji_scores = []
        pq_iou_sum = 0.0
        pq_tp = pq_fp = pq_fn = 0
        gt_counts, pred_counts = [], []
        np_tp_px = np_fp_px = np_fn_px = 0
        # Per-class
        class_tp = {c: 0 for c in [1, 2, 3, 4, 5]}
        class_fp = {c: 0 for c in [1, 2, 3, 4, 5]}
        class_fn = {c: 0 for c in [1, 2, 3, 4, 5]}
        class_iou_sum = {c: 0.0 for c in [1, 2, 3, 4, 5]}

        for entry in cache:
            np_map = entry['np_map']
            hv_map = entry['hv_map']
            gt_inst = entry['gt_inst']
            gt_cls = entry['gt_cls']
            nc_map = entry['nc_map']

            # Watershed with current thresholds
            pred_hwc = np.stack([np_map, hv_map[0], hv_map[1]], axis=-1)
            try:
                pred_inst = __proc_np_hv(pred_hwc, mask_threshold=mt, overall_threshold=ot)
            except Exception as e:
                # If watershed fails (e.g. all-zero map at extreme thresholds), treat as no detections
                pred_inst = np.zeros_like(gt_inst)

            # NP pixel-level
            gt_fg = (gt_inst > 0)
            pred_fg = (np_map > mt)
            np_tp_px += int((gt_fg & pred_fg).sum())
            np_fp_px += int((pred_fg & ~gt_fg).sum())
            np_fn_px += int((~pred_fg & gt_fg).sum())

            # Instance-level
            gt_n = len(np.unique(gt_inst)) - (1 if 0 in gt_inst else 0)
            pred_n = len(np.unique(pred_inst)) - (1 if 0 in pred_inst else 0)
            gt_counts.append(gt_n); pred_counts.append(pred_n)

            aji_scores.append(float(aggregated_jaccard_index(gt_inst, pred_inst)))

            matches, unm_p, unm_g = match_instances_by_iou(
                pred_inst, gt_inst, iou_threshold=args.iou_match_threshold,
            )
            pq_tp += len(matches); pq_fp += len(unm_p); pq_fn += len(unm_g)
            for _p, _g, iou in matches:
                pq_iou_sum += iou

            # Per-class
            for p_id, g_id, iou in matches:
                g_class = instance_modal_class(gt_inst, gt_cls, g_id)
                p_class = instance_modal_class(pred_inst, nc_map, p_id)
                if g_class == p_class and g_class in class_tp:
                    class_tp[g_class] += 1
                    class_iou_sum[g_class] += iou
                else:
                    if p_class in class_fp: class_fp[p_class] += 1
                    if g_class in class_fn: class_fn[g_class] += 1
            for p_id in unm_p:
                p_class = instance_modal_class(pred_inst, nc_map, p_id)
                if p_class in class_fp: class_fp[p_class] += 1
            for g_id in unm_g:
                g_class = instance_modal_class(gt_inst, gt_cls, g_id)
                if g_class in class_fn: class_fn[g_class] += 1

        # Aggregate this combo
        aji = float(np.mean(aji_scores))
        sq = pq_iou_sum / max(pq_tp, 1)
        rq = pq_tp / max(pq_tp + 0.5 * pq_fp + 0.5 * pq_fn, 1)
        pq = sq * rq
        np_recall = np_tp_px / max(np_tp_px + np_fn_px, 1)
        np_precision = np_tp_px / max(np_tp_px + np_fp_px, 1)
        np_iou = np_tp_px / max(np_tp_px + np_fp_px + np_fn_px, 1)
        count_mae = float(np.mean(np.abs(np.array(pred_counts) - np.array(gt_counts))))
        count_bias = float(np.mean(np.array(pred_counts) - np.array(gt_counts)))

        per_class_rq = {}
        for c in [1, 2, 3, 4, 5]:
            tp, fp, fn = class_tp[c], class_fp[c], class_fn[c]
            per_class_rq[c] = tp / max(tp + 0.5 * fp + 0.5 * fn, 1)

        results.append({
            'mask_thresh': mt, 'overall_thresh': ot,
            'aji': aji, 'pq': pq, 'sq': sq, 'rq': rq,
            'np_recall': np_recall, 'np_precision': np_precision, 'np_iou': np_iou,
            'count_mae': count_mae, 'count_bias': count_bias,
            'pq_tp': pq_tp, 'pq_fp': pq_fp, 'pq_fn': pq_fn,
            'rq_neoplastic': per_class_rq[1],
            'rq_inflammatory': per_class_rq[2],
            'rq_connective': per_class_rq[3],
            'rq_dead': per_class_rq[4],
            'rq_epithelial': per_class_rq[5],
            'elapsed': time.time() - t_start,
        })
        print(f'  mt={mt:.2f}  ot={ot:.2f}  '
              f'AJI={aji:.3f}  PQ={pq:.3f}  R={np_recall:.3f}  P={np_precision:.3f}  '
              f'cMAE={count_mae:.2f}  ({results[-1]["elapsed"]:.1f}s)')

    # ============================================================
    # WRITE REPORT
    # ============================================================
    lines = []
    L = lines.append

    L('=' * 110)
    L('  Watershed Sweep Report')
    L('=' * 110)
    L(f'  Checkpoint:     {args.checkpoint}')
    L(f'  Test patches:   {len(cache)}')
    L(f'  IoU match:      {args.iou_match_threshold}')
    L('')

    L('--- Full sweep (sorted by PQ descending) ---')
    L(f'  {"mask":>5s}  {"over":>5s}  {"AJI":>6s}  {"PQ":>6s}  {"SQ":>6s}  {"RQ":>6s}  '
      f'{"NP R":>6s}  {"NP P":>6s}  {"NP IoU":>7s}  {"cMAE":>6s}  {"bias":>7s}')
    L('  ' + '-' * 100)
    sorted_by_pq = sorted(results, key=lambda r: -r['pq'])
    for r in sorted_by_pq:
        flag = ' (current)' if (r['mask_thresh'] == 0.5 and r['overall_thresh'] == 0.4) else ''
        L(f'  {r["mask_thresh"]:>5.2f}  {r["overall_thresh"]:>5.2f}  '
          f'{r["aji"]:>6.3f}  {r["pq"]:>6.3f}  {r["sq"]:>6.3f}  {r["rq"]:>6.3f}  '
          f'{r["np_recall"]:>6.3f}  {r["np_precision"]:>6.3f}  {r["np_iou"]:>7.3f}  '
          f'{r["count_mae"]:>6.2f}  {r["count_bias"]:>+7.2f}{flag}')
    L('')

    # Best per metric
    L('--- Best combos by criterion ---')
    best_pq = max(results, key=lambda r: r['pq'])
    best_aji = max(results, key=lambda r: r['aji'])
    best_count = min(results, key=lambda r: r['count_mae'])
    best_recall = max(results, key=lambda r: r['np_recall'])
    L(f'  Best PQ:        mt={best_pq["mask_thresh"]:.2f} ot={best_pq["overall_thresh"]:.2f}  → PQ={best_pq["pq"]:.4f}')
    L(f'  Best AJI:       mt={best_aji["mask_thresh"]:.2f} ot={best_aji["overall_thresh"]:.2f}  → AJI={best_aji["aji"]:.4f}')
    L(f'  Best count MAE: mt={best_count["mask_thresh"]:.2f} ot={best_count["overall_thresh"]:.2f}  → cMAE={best_count["count_mae"]:.2f}')
    L(f'  Best NP recall: mt={best_recall["mask_thresh"]:.2f} ot={best_recall["overall_thresh"]:.2f}  → recall={best_recall["np_recall"]:.4f}')
    L('')

    # Per-class RQ at the best PQ combo
    L(f'--- Per-class RQ at best-PQ combo (mt={best_pq["mask_thresh"]:.2f}, ot={best_pq["overall_thresh"]:.2f}) ---')
    L(f'  neoplastic     RQ = {best_pq["rq_neoplastic"]:.3f}')
    L(f'  inflammatory   RQ = {best_pq["rq_inflammatory"]:.3f}')
    L(f'  connective     RQ = {best_pq["rq_connective"]:.3f}')
    L(f'  dead           RQ = {best_pq["rq_dead"]:.3f}')
    L(f'  epithelial     RQ = {best_pq["rq_epithelial"]:.3f}')
    L('')

    # Current vs best comparison
    current = [r for r in results if r['mask_thresh'] == 0.5 and r['overall_thresh'] == 0.4][0]
    L('--- Current vs best PQ (delta) ---')
    L(f'  AJI:    {current["aji"]:.4f}  →  {best_pq["aji"]:.4f}   (Δ={best_pq["aji"]-current["aji"]:+.4f})')
    L(f'  PQ:     {current["pq"]:.4f}  →  {best_pq["pq"]:.4f}   (Δ={best_pq["pq"]-current["pq"]:+.4f})')
    L(f'  NP R:   {current["np_recall"]:.4f}  →  {best_pq["np_recall"]:.4f}')
    L(f'  NP P:   {current["np_precision"]:.4f}  →  {best_pq["np_precision"]:.4f}')
    L(f'  cMAE:   {current["count_mae"]:.2f}    →  {best_pq["count_mae"]:.2f}')
    L('')

    L('=' * 110)
    L('Interpretation:')
    L('  - If best PQ is < 1pt above current, the model is the bottleneck, not watershed.')
    L('  - If a (low_mt, low_ot) combo dominates, the model has signal we are clipping.')
    L('  - If best is at boundary of swept range, expand the range.')
    L('=' * 110)

    report_text = '\n'.join(lines)
    print('\n' + report_text)

    report_path = os.path.join(args.output_dir, 'eval_watershed_sweep.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f'\nReport written to: {report_path}')


if __name__ == '__main__':
    main()
