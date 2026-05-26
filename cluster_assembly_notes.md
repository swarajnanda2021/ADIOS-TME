# Cluster assembly notes — `nuclei-counter` branch

Bridge document for whoever assembles this branch on the cluster. The Mac-side
work is complete; this file lists what's still required before training can run.

## 1. Place the `cellvit/` package from PostProc

Code imports from these modules (see HANDOFF §2):

| Module                                | Used by                                                       |
|---------------------------------------|---------------------------------------------------------------|
| `cellvit/datasets.py`                 | `train_stage1_selector.py`, `train_stage2_cellvit.py`, `evaluate_pannuke.py` |
| `cellvit/models.py`                   | `adios_cellvit/adios_cellvit_model.py`                        |
| `cellvit/utils.py`                    | `train_stage1_selector.py`, `train_stage2_cellvit.py`         |
| `cellvit/postproc/benchmarking.py`    | `evaluate_pannuke.py`                                         |

Copy these from the PostProc repo into a `cellvit/` package at the repo root.

Names imported (so the assembly step can verify the symbols match):

- `cellvit.datasets`: `PanNukeDataset`, `SynchronizedTransform`. (`MonuSegDataset`
  is mentioned in HANDOFF §2.1 but not used by this branch.)
- `cellvit.models`: `CellViT`. (`CellViTMultiClass` is unused.)
- `cellvit.utils`: `WarmupDecayScheduler`, `set_seed`.
- `cellvit.postproc.benchmarking`: `__proc_np_hv`, `aggregated_jaccard_index`,
  `panoptic_quality_semantic`.

## 1b. ADIOS convention inversion

This project uses the ADIOS MASK MODEL, not the student encoder.

Standard ADIOS (Shi et al. 2022) trains a student encoder adversarially against
a mask model, then keeps the student and discards the mask model. The user's
workflow is the inverse: the mask model is kept (it produces visually clean
nuclei masks across many experiments), the student is discarded.

Practically, this means:
- `load_adios_mask_model` does NOT load `checkpoint['student']`.
- The HoVer and NC heads consume features from the 192-dim ViT-Tiny encoder
  inside the mask model (`mask_model.encoder`), not from the 768-dim student.
- In stage 2, fine-tuning `mask_model.parameters()` includes both the 192-dim
  encoder and the UNet decoder. Both are trained at LR 1e-6.

If you're new to this codebase: do not be confused by the small encoder size.
The 192-dim ViT-Tiny is intentional — it's the encoder the user trusts for
this task.

## 2. Modify `cellvit/models.py:CellViT` (HANDOFF §4)

Add the NC (nuclei classification) decoder as a third branch. Concretely:

1. `__init__` signature gains `num_classes: int = 5`.
2. `__init__` body, right after `self.hv_map_decoder = self.create_upsampling_branch(2)`:
   ```python
   self.num_classes = num_classes
   self.nuclei_type_map_decoder = self.create_upsampling_branch(num_classes)
   ```
3. `initialize_weights`: extend the final-conv-init loop to include
   `self.nuclei_type_map_decoder`.
4. `forward`: after `out_dict["distances"] = ...`, add
   ```python
   out_dict["nuclei_types"] = self._forward_upsample(
       images, f1, f2, f3, f4, self.nuclei_type_map_decoder
   )
   ```
   (raw logits — softmax is applied inside the stage-2 loss).
5. Leave `out_dict["masks"]` (the softmax NP branch) intact — `ADIOSCellViT`
   overrides it but unchanged `train_CellViT.py` in PostProc still depends on it.

`ADIOSCellViT.forward` reaches into `self.cellvit._forward_upsample(...)` and
`self.cellvit.hv_map_decoder` / `self.cellvit.nuclei_type_map_decoder`
directly, so those names must exist after the modification.

## 3. Provide the unified PanNuke layout

The training and eval scripts read from a dedicated `ADIOSPanNukeDataset`
(in `adios_cellvit/pannuke_dataset.py`) — they do NOT use PostProc's
`HoverNetBasedDataset` / `PanNukeDataset`. The earlier idea of extending
`HoverNetBasedDataset` to return a 5th class-mask tensor was abandoned in
favor of this dedicated dataloader. `cellvit/datasets.py` only needs to
provide `SynchronizedTransform`; its `*Dataset` classes are not used.

The expected layout at `<pannuke_path>`:

```
<pannuke_path>/
├── Training/
│   └── 40x/
│       ├── tissue_images/<patch>.png
│       ├── instance_masks/<patch>.npy           # uint16 instance ID map
│       └── class_masks/
│           ├── neoplastic/<patch>.png           # uint8, per-class instance-labeled
│           ├── inflammatory/<patch>.png
│           ├── connective/<patch>.png
│           ├── dead/<patch>.png
│           └── epithelial/<patch>.png
└── Test/
    └── 40x/
        ├── tissue_images/
        ├── instance_masks/
        └── class_masks/
            └── ... (same 5 subdirs)
```

Notes:

- The `non_neoplastic` subdir from the wtypes folder is intentionally
  **excluded** — it's misleadingly named (actually the background mask), and
  background is implied by all 5 foreground class masks being zero.
- The class-mask folder is named `class_masks/` (not `masks/`) to disambiguate
  from the instance-mask folder.
- Each sample's class mask is built per-batch by stacking the 5 class PNGs
  and taking argmax with foreground offset (index 0 = background, 1..5 = the
  five classes in `FOREGROUND_CLASSES`).

The `ADIOSPanNukeDataset` returns a 5-tuple
`(image, mask_2ch, distance_map, instance_mask, class_mask)` matching the
loop unpack in `train_stage2_cellvit.py` and `evaluate_pannuke.py`. A
`TODO(cluster-test)` in the dataset file flags one thing to verify at first
cluster run: that `SynchronizedTransform` handles a 2-channel HWC mask
input (instance + class concatenated) without errors. If it doesn't, the
fix is to apply the geometric augmentation to the class mask manually.

## 4. Fill placeholder paths in `configs/nuclei_counter.py`

Both `STAGE1` and `STAGE2` dicts contain `'<FILL ON CLUSTER>'` placeholders
for:

- `adios_checkpoint` — path to the ADIOS-TME training checkpoint. The
  expected one is `/data1/vanderbc/nandas1/ADIOS-CellViT/logs/checkpoint_iter_00094000.pth`
  per HANDOFF §3.
- `pannuke_path`    — root directory of the PanNuke dataset.

`STAGE2['stage1_selector']` defaults to `./logs/stage1/stage1_selector.pth`
which is where stage 1 writes its best checkpoint.

## 5. PanNuke `panoptic_quality_semantic` return schema

The HANDOFF (§2.4) says this function returns a metrics dict but does not
specify the key names. `evaluate_pannuke.py` assumes `'pq'` for the aggregate
and `'pq_class_*'` for per-class scores, falling back to NaN when keys are
missing. If the actual schema differs, adjust the aggregation block in
`evaluate_pannuke.py` (the model-side code is unaffected).

## 6. Run order

```bash
# Stage 1
python train_stage1_selector.py --config configs/nuclei_counter.py

# Verify val accuracy >= 0.85 in the printed summary before proceeding.

# Stage 2
python train_stage2_cellvit.py --config configs/nuclei_counter.py

# Evaluation
python evaluate_pannuke.py \
  --checkpoint ./logs/stage2/stage2_adios_cellvit.pth \
  --adios_checkpoint /data1/vanderbc/nandas1/ADIOS-CellViT/logs/checkpoint_iter_00094000.pth \
  --pannuke_path /path/to/pannuke
```

## 7. Sanity (Mac-side, already done)

`scripts/sanity_imports.py` verifies that `adios_cellvit.channel_selector`
imports cleanly without `cellvit/*` being present. On the cluster, after the
above assembly, all modules including `adios_cellvit.adios_cellvit_model`
should import cleanly.

The mask model loader is `adios_cellvit.adios_backbone.load_adios_mask_model`
(see §1b for why; the original `load_adios_backbone_and_decoder` name was
replaced when the ADIOS-convention inversion was applied).
