# Path Z Handoff — ADIOS Nuclei Counter

## TL;DR

We built a CellViT-style nuclei counter on top of an ADIOS-pretrained ViT-Tiny
encoder + UNet mask decoder. After three architectural iterations, the model is
trained as **standard dense supervision with split encoder/decoder learning
rates** (encoder 1e-5, decoder 1e-4, heads 1e-4). Final test metrics:

- AJI 0.448, PQ 0.426 (panoptic instance metric, IoU>=0.5 matching).
- NP IoU 0.730, recall 0.858, precision 0.830.
- NC macro-F1 (matched instances) 0.626.
- Count MAE 5.95 / patch, biased toward under-counting (-5.85 mean).
- Per-class instance recall (RQ): neoplastic 0.52, inflammatory 0.55,
  connective 0.32, **dead 0.00**, epithelial 0.50.

The model is shippable for a fellow-feedback v1. Dead-class detection is
absent. Connective is weak.

## Project structure

- `adios_cellvit/` — model wrapping ADIOS + CellViT heads.
  - `adios_backbone.py` — `load_adios_mask_model(checkpoint, device)` loads
    the frozen MaskModel (ViT-Tiny encoder + UNet decoder, 14.6M params).
    Imports from the parent DINOv2 fork's `models/` directory (must be on
    `PYTHONPATH` or run from inside `ADIOS/`).
  - `channel_selector.py` — `ChannelSelector` (250K params, deeper than
    original). Stage 1 trains this to pick the nuclei channel from the
    mask model's 3-channel output.
  - `adios_cellvit_model.py` — `ADIOSCellViT` wraps mask model + selector +
    CellViT heads. Has `inference_mode='soft'|'argmax'`.
  - `pannuke_dataset.py` — `ADIOSPanNukeDataset`. Returns
    `(image, mask_2ch, distance_map, instance_mask, class_mask)`.
    Background convention is **standard (background=0)** — was
    non-standard in raw PanNuke (background=max-instance-id); we normalize.

- `cellvit/` — vendored from `/data1/vanderbc/nandas1/PostProc`. Not
  in git (assembled by `assemble_cluster.sh`).
  Two cluster-only patches required, baked into the assembly script:
  - `cellvit/datasets.py` line ~464: `[:, :, 3:4]` -> `[:, :, 3:]`
    (allows multi-channel masks through `SynchronizedTransform`).
  - `cellvit/postproc/benchmarking.py`: four bare imports rewritten to
    namespace-qualified `from cellvit.utils import ...` etc.

- `train_stage1_selector.py` — trains `ChannelSelector` to pick nuclei.
  Mask model frozen.
- `train_stage2_cellvit.py` — fine-tunes everything together.
  Selector frozen, mask encoder at 1e-5, mask decoder at 1e-4, heads at 1e-4.
- `configs/nuclei_counter.py` — STAGE1 / STAGE2 hyperparameters.
- `eval_full_v1.py` — Path A (AJI/PQ/count) + Path B (NP recall/precision,
  NC confusion, extras characterization) eval.
- `eval_threshold_diagnostic.py` — NP-map histogram (TP vs FP confidence
  distribution).
- `eval_watershed_sweep.py` — sweep (mask_threshold, overall_threshold).
- `run_stage{1,2}_with_submitit.py` — SLURM launchers (known bug: pass
  `configs/nuclei_counter.py` as file path, not `configs.nuclei_counter`).
- `assemble_cluster.sh` — six-phase idempotent cluster assembly.

## Project history (high-level)

The original ambition was "use ADIOS unsupervised pretraining to find
nuclei PanNuke didn't label, surface those as 'unknown class' to clinical
fellows." We measured ADIOS-vs-PanNuke agreement: precision 0.62,
recall 0.71, IoU 0.50. The ADIOS mask model genuinely produces structure
PanNuke ignores -- so the ambition was data-grounded.

### Stage 1 (selector)

Goal: pick which of ADIOS's 3 mask channels is the nuclei channel.
After two iterations:

- **v1**: small selector + hard CE on argmax-of-IoU target. Channel 0
  dominated PanNuke prior (99% of patches). Selector trivially picked 0.
- **v2**: bigger selector (250K params), **channel scrambling** at train
  time (per-sample random permutation of input channels), **soft target**
  (IoU-normalized distribution per sample), **KL divergence loss**.
- 60 epochs, val accuracy **0.977**. The selector is essentially solved
  for this checkpoint.

### Stage 2 v1 — Mode 3 (failed)

Loss design: NP/HV/MSGE/Dice fire only on PanNuke-foreground pixels;
NC uses `ignore_index=0`. Goal: never penalize "extras" so the
unsupervised discoveries survive. Mask model fine-tuned at LR 1e-6 (encoder
and decoder together).

**Result**: model collapsed to predicting foreground at every pixel.
NP precision 0.18 with 99.9% recall. The NP map became near-binary
saturated at ~0.99 confidence across most of every tissue patch.
Mode 3 had no background-supervision signal, so the model maximized the
loss by being confident everywhere. Stage 2 v1 is broken.

### Diagnostic: ADIOS-as-prior idea, then ruled out

We considered using ADIOS-ch0 as a background prior ("supervise NP=0 where
both PanNuke says background AND ADIOS says background"). A histogram of
ADIOS-ch0 across PanNuke-foreground vs background pixels showed:

- 99.9% of FG pixels score in `[0.95, 1.0)` (good).
- 60.6% of BG pixels also score in `[0.95, 1.0)` (bad).
- 18.96% of inside-PanNuke-nucleus pixels score in `[0, 0.05)` — these
  are intra-nucleus voids the ADIOS model produces.

ADIOS-ch0 thresholded at any value is only marginally discriminative.
ADIOS's negative signal is unreliable (intra-nucleus voids). Conclusion:
ADIOS soft masks are not informative enough to use as a prior.

### Stage 2 v2 — Path Z (current)

Drop the ADIOS-as-prior ambition entirely. Use ADIOS as
**initialization only**:

- Mask encoder LR: 1e-5 (modest adaptation, preserves features).
- Mask decoder LR: 1e-4 (full adaptation, decoder learns to be a clean
  NP producer instead of staying near its ADIOS shape).
- Heads (HV, NC): 1e-4 (unchanged).
- Selector: frozen (loaded from stage 1 checkpoint).
- Loss: dense BCE + Dice + MSE + MSGE + CE — no masking.

50 epochs, val_total descended from 2.49 to 0.58. Eval numbers above.

### Watershed sweep — bottleneck is model, not post-processing

`(mask_threshold, overall_threshold)` swept over 12 combos. Best PQ
matches the current defaults (mt=0.5, ot=0.4). Lower thresholds split
nuclei into pieces; higher thresholds make no difference. The
under-prediction of dead and connective is a **model limitation**, not
a tuneable post-processing artifact.

## What's pending for the next iteration

The user wants to revisit "ADIOS as prior" with a **much stronger encoder**.
They have trained ViT-B+ encoders via DINOv2 / modified DINOv2 on 300M
tissue samples. These are at sibling paths to `ADIOS/`, e.g.
`/data1/vanderbc/test_dinov2_swaraj/FMC_ViT-B_baseline/`,
`FMC_ViT-B_semantic_ibot_3ch_recipe_canonteach/`, etc. The DINOv2 fork's
`models/vision_transformer/modern_vit.py` defines the model class.

The plan for the next stage:

1. Swap the ADIOS ViT-Tiny encoder (192-dim) for a ViT-B+ encoder
   (768-dim) loaded from one of the DINOv2 checkpoints. Encoder shape
   change cascades into the decoder + heads — dimensions need to be
   recomputed throughout `ADIOSCellViT` and the CellViT decoder branches.
2. Decide what to do with the ADIOS mask decoder: keep it as
   initialization (and update its input dimension to 768) or replace
   with a fresh decoder. Currently undecided.
3. Re-explore ADIOS-as-prior under the new encoder. The hope: a stronger
   encoder produces cleaner feature maps and the mask decoder, when
   re-initialized or fine-tuned, doesn't produce the intra-nucleus voids
   we saw before.

## Class-balance ideas (orthogonal, low effort)

The dead class has 0 detections across 2722 test patches. Path Z used
unweighted CE. Two cheap fixes worth trying after the encoder swap:

- Inverse-frequency NC class weights (or focal loss).
- Drop xentropy from the NP loss entirely; rely on Dice alone (Dice
  handles imbalance better).

These changes alone would not have rescued v1 (where the failure was
structural, not class-balance), but they may help Path Z's dead and
connective recall.

## Known cluster-only fixes (already baked into assemble_cluster.sh)

1. `cellvit/datasets.py` line ~464: `mask = transformed_combined[:, :, 3:4]`
   → `mask = transformed_combined[:, :, 3:]`.
   Allows multi-channel masks through `SynchronizedTransform`.

2. `cellvit/postproc/benchmarking.py`: four bare imports near top of
   file rewritten to namespace-qualified form:
   - `from utils import ...` → `from cellvit.utils import ...`
   - `from datasets import (...)` → `from cellvit.datasets import (...)`
   - `from models import CellViT, CellViTMultiClass` → `from cellvit.models import ...`

3. `adios_cellvit/pannuke_dataset.py`: instance_mask returned from the
   dataloader has background=0 (foreground IDs = 1..N). PanNuke's
   internal convention is background=max-instance-id; we normalize.

4. `eval_watershed_sweep.py`: removed `+` from string format spec
   (`{"bias":>+7s}` → `{"bias":>7s}`). Cosmetic — positive bias values
   no longer show leading `+`. Future Claude Code work can re-add with
   correct syntax: `f'{val:+.2f}'`.

## Known issues to fix during the encoder swap

- `run_stage{1,2}_with_submitit.py`: passes `--config configs.nuclei_counter`
  (dotted name) but training scripts expect a file path. Use
  `python train_stage{1,2}_*.py` directly, or fix the submitit args.

- Albumentations `Downscale` deprecation warning: `scale_min` and
  `scale_max` are obsolete; new API is `scale=(min, max)`. Cosmetic.

- Dataloader produces 4 worker warning: "suggested max workers is 1".
  Cosmetic.

## How to resume work

Read this file. Read `train_stage2_cellvit.py` to see Path Z's training
setup. Read `eval_full_v1.py` to see the eval. Look at
`configs/nuclei_counter.py` for hyperparameters.

The first thing to do for the encoder swap is plan the dimension
cascade. ViT-Base has 768-dim features; the mask decoder, channel
selector, and CellViT skip connections need to handle this. The CellViT
class in `cellvit/models.py` takes `encoder_dim` as a constructor
arg — set it to 768. The mask decoder in the parent DINOv2 fork's
`models/UNet.py` will need analogous updates.

The user is the ideator and will discuss design at each step. Code
changes are routed through Claude Code on the user's Mac, not on the
cluster. Cluster is read-only for git: pull only, no push.
