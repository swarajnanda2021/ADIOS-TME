"""PanNuke evaluation: AJI, panoptic quality, count MAE.

Loads a stage-2 ADIOSCellViT checkpoint, sets inference mode to ``argmax``
(single-channel deployment collapse), and runs over the PanNuke Test split.
"""

import argparse
import importlib
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from cellvit.datasets import PanNukeDataset, SynchronizedTransform
from cellvit.postproc.benchmarking import (
    __proc_np_hv,
    aggregated_jaccard_index,
    panoptic_quality_semantic,
)

from adios_cellvit.adios_backbone import load_adios_backbone_and_decoder
from adios_cellvit.adios_cellvit_model import ADIOSCellViT
from adios_cellvit.channel_selector import ChannelSelector


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--adios_checkpoint', required=True)
    p.add_argument('--pannuke_path', required=True)
    p.add_argument('--output_json', default='./eval_results.json')
    p.add_argument('--config', default='configs/nuclei_counter.py')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    return p.parse_args()


def _load_stage2_config(config_path: str) -> dict:
    spec = importlib.util.spec_from_file_location('nuclei_counter_cfg', config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.STAGE2


def main():
    args = parse_args()
    config = _load_stage2_config(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    encoder, mask_decoder = load_adios_backbone_and_decoder(args.adios_checkpoint, device)
    selector = ChannelSelector(num_masks=3).to(device)
    model = ADIOSCellViT(
        encoder=encoder,
        mask_decoder=mask_decoder,
        selector=selector,
        encoder_dim=768,
        num_classes=config['num_classes'],
        drop_rate=0.1,
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
    test_dataset = PanNukeDataset(
        data_dir=args.pannuke_path,
        split='Test',
        magnification=config['magnification'],
        transform=val_transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    aji_scores = []
    pq_scores = []
    pq_per_class = []
    count_aes = []

    with torch.no_grad():
        for image, _, _, instance_mask, class_mask in test_loader:
            image = image.to(device, non_blocking=True)
            output = model(image)

            np_np = output['masks'].cpu().numpy()
            hv_np = output['distances'].cpu().numpy()
            nc_np = output['nuclei_types'].argmax(1).cpu().numpy()

            B = image.size(0)
            for b in range(B):
                pred_hwc = np.stack(
                    [np_np[b, 0], hv_np[b, 0], hv_np[b, 1]], axis=-1
                )
                pred_instances = __proc_np_hv(
                    pred_hwc, mask_threshold=0.5, overall_threshold=0.4
                )

                gt_instances = instance_mask[b, 0].cpu().numpy().astype(np.int64)
                gt_class = class_mask[b].cpu().numpy().astype(np.int64)
                pred_class = nc_np[b]

                aji_scores.append(
                    aggregated_jaccard_index(gt_instances, pred_instances)
                )
                pq = panoptic_quality_semantic(
                    gt_class, pred_class, num_classes=6, ignore_classes=[0]
                )
                pq_scores.append(pq.get('pq', float('nan')))
                pq_per_class.append(pq)

                gt_count = len(np.unique(gt_instances)) - 1
                pred_count = len(np.unique(pred_instances)) - 1
                count_aes.append(abs(gt_count - pred_count))

    # Aggregate per-class PQ across samples (mean over present classes).
    per_class_keys = set()
    for d in pq_per_class:
        per_class_keys.update(k for k in d.keys() if k.startswith('pq_class_'))
    pq_per_class_agg = {}
    for k in per_class_keys:
        vals = [d[k] for d in pq_per_class if k in d and not np.isnan(d[k])]
        if vals:
            pq_per_class_agg[k] = float(np.mean(vals))

    results = {
        'AJI_mean': float(np.mean(aji_scores)),
        'AJI_std':  float(np.std(aji_scores)),
        'PQ_mean':  float(np.nanmean(pq_scores)),
        'PQ_std':   float(np.nanstd(pq_scores)),
        'PQ_per_class': pq_per_class_agg,
        'count_MAE': float(np.mean(count_aes)),
        'n_samples': len(aji_scores),
    }

    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
