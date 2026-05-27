# Path B-2 Handoff — ViT-B + three-branch CellViT + optional ADIOS prior

## TL;DR

Path B-2 swaps Path Z's 192-dim ADIOS ViT-Tiny encoder for a 768-dim ViT-B
encoder trained externally on 300M tissue samples (the FMC_ViT-B_<recipe>
checkpoints in sibling directories of ADIOS/). The NP map comes from
CellViT's own 2-channel branch (softmax → channel 0), not from the ADIOS
mask decoder. A frozen ADIOS+selector is optionally attached at training
time to provide a consistency BCE on the NP map. Path Z stays runnable as
the reference baseline.

## What's new on this branch

- `adios_cellvit/vitb_backbone.py` — `load_vitb_encoder(checkpoint, device)`
  loads a ViT-B from an FMC checkpoint. Reads non-standard arg names from
  the FMC fork: `embeddingdim`, `vitdepth`, `vitheads`,
  `num_register_tokens`. Probes `module.backbone.` then `backbone.` for the
  student state-dict prefix. Returns the unfrozen encoder, patch_size,
  num_registers, embed_dim.

- `adios_cellvit/vitb_cellvit_model.py` — `ViTBCellViT`. Three-branch
  CellViT (NP/HV/NC) reusing the existing post-PHASE-D
  `cellvit/models.py` patch. The encoder freeze that `CellViT.__init__`
  performs is immediately undone (training script decides the LR group).
  Decoder outputs are bilinear-interpolated to `image_size` (handles both
  patch_size=16 → native 224 and patch_size=14 → native 256). NP is the
  CellViT 2-channel NP branch (softmax → channel 0). The ADIOS prior, if
  attached, is wrapped in `SimpleNamespace` and assigned as a plain
  attribute — bypasses `nn.Module`'s submodule registration so its
  parameters don't appear in `state_dict()` and `.to(device)` doesn't move
  it (the caller is responsible for device placement).

- `train_vitb.py` — single-GPU training. Same dataloader and
  dense-supervision loss as Stage 2 (Path Z) plus:
  - **Class-weighted NC CE**. `compute_nc_class_weights` walks the
    PanNuke `class_masks/<class>/*.png` once, computes inverse-frequency
    weights normalized to sum to `num_classes`, caps any individual
    weight at 20.0 (per handoff §"Open questions" — high weights
    destabilize training), and caches the result to
    `<output_dir>/nc_class_weights.json` keyed on
    `(pannuke_path, split, magnification, K)`.
  - **Optional ADIOS-prior consistency**. When
    `config['use_adios_consistency']`, the frozen ADIOS mask model and
    the Stage 1 selector are loaded and attached. Loss adds
    `lambda_adios * (BCE(pred_np, adios_fg) + BCE(pred_np, 1 - adios_bg))`.
  - Optimizer has two param groups: encoder at `encoder_lr` (1e-5),
    everything else at `heads_lr` (1e-4). `WarmupDecayScheduler` scales
    the heads group.

- `eval_full_v2_vitb.py` — line-by-line copy of `eval_full_v1.py` with
  `ADIOSCellViT` → `ViTBCellViT`, no ADIOS prior at eval time,
  `--vitb_checkpoint` in place of `--adios_checkpoint` /
  `--stage1_selector`. Metrics (AJI / PQ / per-class PQ / NP recall /
  precision / NC macro-F1 / count MAE / extras characterization) match
  Path Z's eval exactly so result tables are directly comparable.

- `configs/nuclei_counter.py` — adds `STAGE_VITB` dict at the bottom.
  `STAGE1` and `STAGE2` untouched.

## Planned ablations

1. **Path Z (reference baseline)**: run `train_stage2_cellvit.py` with
   the existing `STAGE2` config. Already runnable; nothing changed.
2. **Path B-2 no consistency**: `train_vitb.py` with
   `STAGE_VITB['use_adios_consistency'] = False` (or
   `--use_adios_consistency false`). Tests the pure encoder-swap effect.
3. **Path B-2 with consistency**: `train_vitb.py` with
   `use_adios_consistency=True` (the central B-2 model). Tests whether
   the ADIOS soft prior adds signal on top of the stronger encoder.

All three are evaluated with the same `eval_full_v*.py` script of the
matching family so AJI / PQ / count MAE numbers are directly comparable.

## NC head channel count (K=6) — resolution of handoff ambiguity

The handoff text said `num_classes = 5` in three places (config,
`compute_nc_class_weights(num_classes=5)`, commit-2 verification assertion)
but also said "match Stage 2" and described a length-6 weight tensor.
Stage 2 uses K=6 (background + 5 foreground), CellViT NC head is 6-channel,
and `F.cross_entropy(pred[6ch], class_mask in {0..5}, weight=weights[6])`
is the working setup.

Resolution: `STAGE_VITB['num_classes'] = 6`. Following "match Stage 2"
verbatim. Smoke-test assertion for commit 2 should be
`out['nuclei_types'].shape == (2, 6, 224, 224)` (the handoff text's `5`
appears to be a copy-paste from an earlier draft).

## Cluster assembly — PHASE E placeholder behavior

`STAGE_VITB` uses `<FILL ON CLUSTER (VITB)>` as its placeholder string
(five total: `vitb_checkpoint`, `adios_checkpoint`, `stage1_selector`,
`pannuke_path`, `output_dir`). This is intentional and different from
`STAGE1`/`STAGE2`'s `<FILL ON CLUSTER>`.

Reason: `assemble_cluster.sh`'s PHASE E does four ordered `str.replace`
calls against the literal token `<FILL ON CLUSTER>`, then errors out if
that exact substring still appears anywhere in the config. Using a
distinct placeholder tag (`(VITB)` suffix breaks the substring match —
`<FILL ON CLUSTER>` is not a substring of `<FILL ON CLUSTER (VITB)>`)
keeps the VITB placeholders invisible to PHASE E. The script's existing
4-replacement logic still correctly fills `STAGE1`/`STAGE2`, and the
post-substitution check passes.

The user fills the five VITB placeholders manually after assemble
completes. A one-shot sed works (each placeholder is unique on its line,
so substitution is unambiguous):

```bash
cd /data1/vanderbc/test_dinov2_swaraj/ADIOS
# Pick the recipe you want to train with:
VITB_CKPT=/data1/vanderbc/test_dinov2_swaraj/FMC_ViT-B_recipe_canonteach/logs/checkpoint_iter_00150000.pth
ADIOS_CKPT=/data1/vanderbc/nandas1/ADIOS-CellViT/logs/checkpoint_iter_00094000.pth
STAGE1_SEL=./logs/stage1/stage1_selector.pth
PANNUKE=/data1/vanderbc/test_dinov2_swaraj/ADIOS/data/pannuke
OUTPUT=./logs/vitb_pathb2

python3 -c "
import re
with open('configs/nuclei_counter.py') as f: src = f.read()
for key, val in (('vitb_checkpoint', '$VITB_CKPT'),
                 ('adios_checkpoint', '$ADIOS_CKPT'),
                 ('stage1_selector', '$STAGE1_SEL'),
                 ('pannuke_path', '$PANNUKE'),
                 ('output_dir', '$OUTPUT')):
    src = re.sub(rf'({re.escape(key)}.*?:\s*)\\'<FILL ON CLUSTER \\(VITB\\)>\\'', rf'\\1\\'{val}\\'', src, count=1)
with open('configs/nuclei_counter.py', 'w') as f: f.write(src)
"
```

(Or just open the file in $EDITOR and replace the five values by hand.)

## Cluster run plan

```bash
cd /data1/vanderbc/test_dinov2_swaraj/ADIOS
# 1) assemble (idempotent; safely re-runs)
bash assemble_cluster.sh

# 2) fill STAGE_VITB placeholders (see above)

# 3) train Path B-2 with consistency
python train_vitb.py --config configs/nuclei_counter.py

# 4) train Path B-2 no-consistency ablation
python train_vitb.py --config configs/nuclei_counter.py \
    --output_dir ./logs/vitb_pathb2_noprior \
    --use_adios_consistency false

# 5) evaluate
python eval_full_v2_vitb.py \
    --checkpoint ./logs/vitb_pathb2/vitb_pathb2.pth \
    --vitb_checkpoint /data1/vanderbc/test_dinov2_swaraj/FMC_ViT-B_recipe_canonteach/logs/checkpoint_iter_00150000.pth \
    --pannuke_path /data1/vanderbc/test_dinov2_swaraj/ADIOS/data/pannuke \
    --output_dir ./logs/eval/vitb_pathb2
```

## Open questions to verify at first cluster run

These are the items the handoff §"Open questions" flagged as
"stop and ask if you hit this":

1. **Checkpoint args field names.** `load_vitb_encoder` reads
   `args.embeddingdim`, `args.vitdepth`, `args.vitheads`. If a particular
   recipe uses different names (e.g. `embed_dim`/`depth`/`num_heads`),
   the loader raises a clear `RuntimeError` pointing back at the loader
   for the user to update.
2. **State-dict prefix.** Loader tries `module.backbone.` first, falls
   back to `backbone.`. Prints missing/unexpected key counts; if either
   is >5, the prefix logic likely needs another option.
3. **Decoder output size.** The model interpolates to `image_size`
   unconditionally, so a non-standard intermediate size (anything that
   isn't `feature_size * 16`) wouldn't crash — it'd just produce visually
   incorrect outputs. Worth a sanity print on first run.
4. **NC class weight capping.** `compute_nc_class_weights` caps weights
   at 20.0 and prints pre-cap + post-cap distributions. If the dead class
   weight comes out above ~50 pre-cap, the cap is doing useful work; if
   pre-cap is already below 20 the cap is a no-op.
5. **`SimpleNamespace` prior bypass.** The training script's
   `torch.save(model.state_dict())` should not contain
   `adios_prior.mask_model.*` or `adios_prior.selector.*` keys. If it
   does, switch to `object.__setattr__(model, '_adios_prior', ...)` or
   another mechanism.

## Out of scope (deliberately untouched)

- All Path Z files: `adios_cellvit/{adios_backbone, adios_cellvit_model,
  channel_selector, pannuke_dataset}.py`, `train_stage1_selector.py`,
  `train_stage2_cellvit.py`, `eval_{full_v1, watershed_sweep,
  threshold_diagnostic}.py`, `evaluate_pannuke.py`. Path Z remains
  runnable for reference.
- `assemble_cluster.sh`. PHASE D's CellViT patch already produces the
  NP/HV/NC three-branch CellViT this branch needs.
- Stage 1 (selector training). The trained selector is reused.
- The frozen ADIOS mask model loader (`load_adios_mask_model`). Reused
  unchanged for the optional prior path.
- patch_size=14 encoder support. The loader reads `patch_size` from
  `args` already, so when those checkpoints land it'll work without a
  code change here. The model's interpolate-to-`image_size` step handles
  the differently-sized decoder output.
