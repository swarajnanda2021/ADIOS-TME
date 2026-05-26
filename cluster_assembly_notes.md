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

## 3. Dataloader must return a 5-tuple including per-pixel class labels (HANDOFF §6.2)

`train_stage2_cellvit.py` and `evaluate_pannuke.py` unpack each batch as

```python
image, mask_2ch, distance_map, instance_mask, class_mask = batch
```

`HoverNetBasedDataset` in PostProc currently returns 4 tensors and stops at
`instance_mask`. For stage 2 NC training we need a 5th tensor:

- `class_mask`: `LongTensor [H, W]` with values in `{0, 1, ..., 5}` where 0 is
  background and 1..5 are the PanNuke classes (neoplastic, inflammatory,
  connective, dead, epithelial).

Either confirm `HoverNetBasedDataset` already exposes this (and adjust the
import / unpack as needed), or extend `HoverNetBasedDataset.__getitem__` to
return it as the 5th item. A `TODO(cluster-assembly)` block at the top of
`train_stage2_cellvit.py` calls this out.

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
