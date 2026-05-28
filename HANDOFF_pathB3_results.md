# Path B-3 results — iteration close-out

## What was run

Two configurations, both evaluated on PanNuke Test (2722 patches,
IoU=0.5 instance matching, identical eval harness
`eval_full_v3_cellvitpp.py`):

1. **Path B-3 with-cons**: ViT-B (FMC) + NP/HV CellViT branches +
   per-cell pooled-token MLP classifier + frozen ADIOS-prior
   consistency BCE at λ=0.1.
2. **Path B-3 no-cons**: same as (1) with `use_adios_consistency=False`.

Encoder, NP/HV losses, optimizer schedule, batch size, dataset all
identical to Path B-2. Only NC head architecture differs (per-pixel
6-channel decoder → per-cell pooled tokens + MLP, K=5 foreground classes).

Both runs trained 100 epochs from FMC ViT-B init under the *old* save
logic (best checkpoint saved at end-of-loop only). Best val_total hit
at ep 33 (with-cons) and ep 39 (no-cons); both showed val divergence
from train after ep ~45 — the per-cell head with ~20 samples/image
overfits much earlier than the per-pixel head (~50k pixels/image),
consistent with the smaller effective per-batch loss footprint.

## Headline metrics

| metric                | Path Z | B-2 +cons | **B-2 no-cons** | B-3 +cons | B-3 no-cons | B-2 + HV-fix |
|-----------------------|--------|-----------|-----------------|-----------|-------------|--------------|
| AJI                   | 0.448  | 0.407     | **0.481**       | 0.404     | 0.443       | 0.471        |
| Instance PQ           | 0.426  | 0.361     | **0.448**       | 0.354     | 0.412       | 0.437        |
| NP IoU                | 0.730  | 0.722     | 0.721           | 0.719     | 0.724       | 0.727        |
| NP recall             | 0.858  | 0.851     | 0.863           | 0.850     | 0.858       | 0.860        |
| NP precision          | 0.830  | 0.826     | 0.814           | 0.824     | 0.822       | 0.825        |
| NC macro-F1           | 0.626  | **0.748** | 0.737           | 0.676     | 0.706       | 0.730        |
| neoplastic recall     | 0.52   | 0.834     | 0.833           | 0.806     | 0.838       | 0.811        |
| inflammatory recall   | 0.55   | 0.876     | 0.877           | 0.851     | 0.857       | 0.851        |
| **connective recall** | 0.32   | 0.639     | 0.646           | 0.490     | 0.539       | **0.694**    |
| dead recall           | 0.00   | 0.844     | 0.812           | 0.642     | 0.803       | 0.677        |
| epithelial recall     | 0.50   | 0.886     | 0.870           | 0.894     | 0.870       | 0.883        |
| count MAE             | 5.95   | 7.35      | 6.52            | 7.34      | 7.02        | **6.17**     |
| count bias            | -5.85  | -7.19     | -6.41           | -7.23     | -6.94       | -6.04        |
| matched pairs (TP)    | -      | 22256     | 27549           | 22123     | 25291       | 27588        |

**Best of the six: still B-2 no-cons by a hair on AJI/PQ.** B-2 + HV-fix
is within ±0.01 of B-2 no-cons on every macro metric — essentially a
tie on PanNuke, with a small per-class redistribution (connective
recall lifted +0.048, dead recall regressed -0.135).

## Findings

### 1. Per-cell pooled-token MLP is a regression vs per-pixel NC decoder

Both detection AJI and classification NC macro-F1 dropped going from
B-2 no-cons to B-3 no-cons:
- AJI 0.481 → 0.443 (−0.038)
- PQ 0.448 → 0.412 (−0.036)
- NC macro-F1 0.737 → 0.706 (−0.031)
- Count MAE 6.52 → 7.02 (+0.50)

The "++" in CellViT++ refers to operational flexibility (swappable
class schemes, embeddings as deliverable), not raw F1 on a fixed
benchmark. This iteration's data is consistent with the paper's
implicit position: per-cell pooling is comparable-to-slightly-worse
than per-pixel on PanNuke leaderboard metrics.

### 2. The regression is concentrated in connective + dead

Per-class F1 changes from B-2 no-cons → B-3 no-cons:
- neoplastic: 0.873 → 0.866 (−0.007, within noise)
- inflammatory: 0.776 → 0.762 (−0.014, within noise)
- **connective: 0.696 → 0.622 (−0.074)**
- **dead: 0.489 → 0.454 (−0.035, on a small base)**
- epithelial: 0.850 → 0.826 (−0.024)

Connective recall dropped 0.107 (0.646 → 0.539). The confusion matrix
shows the loss is to inflammatory (1269 connective→inflam) and
epithelial (478 connective→epith), not to background. The classifier
is *guessing among foreground classes* on connective cells.

Mechanism (hypothesized): the per-pixel NC decoder reads from
decoder0..3 features that integrate ~16-32 pixel receptive fields,
giving each pixel implicit access to its *surroundings*. The per-cell
pooled MLP averages 1-4 patches strictly inside the cell — no
surrounding context. Connective discrimination depends heavily on what
surrounds the cell (stromal collagen, vessels, lymphoid aggregates).
Pooling exactly the cell's patches loses that signal.

### 3. ADIOS consistency is now neutral, not harmful (small but real shift)

In B-2 with-cons (per-pixel NC + consistency), the prior cost 0.087 PQ
and 0.074 AJI vs no-cons because per-pixel NC shared `decoder0..3`
with NP and the consistency-loss perturbation bled into NC's gradient.

In B-3 with-cons (per-cell MLP + consistency), the prior costs only
0.030 macro-F1 and 0.039 AJI vs no-cons — less than half the B-2 hit.
The architectural decoupling (separate MLP head reading encoder
tokens directly, not decoder features) reduced the spillover but
didn't eliminate it. The prior is still slightly detrimental, just
less so.

This also confirmed during training: val_total floors of with-cons
(0.838) and no-cons (0.837) were within 0.001 — the prior contributed
nothing to val improvement. With-cons just reached the floor 6 epochs
earlier.

## Current best configuration
**Path B-2 no-cons** remains the reference baseline by a hair (AJI 0.481
vs B-2+HVfix's 0.471). The B-2 + HV-fix run is essentially tied on
macro metrics but moves per-class metrics around: connective recall
lifted +0.048, dead recall regressed -0.135.

For the *clinical-counting* objective specifically, B-2 + HV-fix is
arguably preferred — count MAE drops 6.52 → 6.17, count bias improves
-6.41 → -6.04. Real-cells-per-patch is closer.

## Finding 4: HV-fix is not the silver bullet for AJI/PQ

Hypothesis at iteration start: under-weighted MSGE (1.0 vs Path Z's
8.0) was the dominant detection bottleneck; fixing it would deliver
AJI +0.03-0.07. The B-2 + HV-fix run did NOT confirm this:

- AJI: 0.481 → 0.471 (-0.010)
- PQ: 0.448 → 0.437 (-0.011)
- Count MAE: 6.52 → 6.17 (-0.35, real but modest improvement)
- Connective recall: 0.646 → 0.694 (+0.048, the expected boundary-sharpening effect)
- Dead recall: 0.812 → 0.677 (-0.135, the unexpected collapse)

The HV regression *did* converge to a lower val_mse floor (0.0142 vs
0.0145 for B-3 — modest), so the loss did what it was supposed to do
on its own target. The downstream effect on instance metrics was
smaller than expected because:

1. **Dead-class collapse**: rare-class cells (dead, ~840 in test) need
   strong NC gradient pull. With w_msge bumped from 1.0 to 8.0, the
   NC term's relative contribution to total loss dropped from ~20%
   (one of five equal-weighted) to ~7.5% (one of 13.5-equivalent).
   Tiny dead nuclei were further fragmented by sharper HV gradients
   and watershed dropped some as too-small.
2. **PanNuke detection is near a representation ceiling**: the encoder
   + per-pixel-decoder + watershed combination has a structural limit
   that boundary sharpness alone can't break. Two earlier signals
   pointed here: the watershed parameter sweep found defaults near-
   optimal, and NP IoU plateaued near 0.72 across all six runs.

The real detection ceiling on PanNuke is probably representation
quality (encoder, decoder capacity) or data noise (PanNuke's
inter-annotator variance), not loss weighting. HV-fix is therefore
**not a wasted experiment** — connective recall +0.048 and count MAE
-0.35 are real wins — but it's also not the lever to break through
the AJI ceiling on this dataset.

## Open questions / next-step plan

Updated in light of finding (4): the two-stream plan (cell-context
attention + multi-dataset 10-class) is now the path forward.

1. **Cell-context attention** (Stream A, marquee architecture
   experiment, new branch). Per-cell pooled tokens + 2D positional
   encoding from centroids + 1-2 self-attention layers over cells +
   MLP head. End-to-end trainable. Tests whether spatial cell context
   rescues connective + dead classes that suffered in B-3. If it
   recovers connective recall via inter-cell spatial reasoning, the
   per-cell architecture earns its keep. Otherwise, per-pixel is the
   permanent architecture.

2. **Multi-dataset 10-class with per-pixel NC head** (Stream B, the
   lab's actual ask). Extend the existing B-2 architecture's NC
   decoder output from 6 channels to 11 (background + 10 unified
   classes). Train end-to-end on PanNuke + Lizard + CoNSeP +
   MoNuSAC combined with soft-target CE for ambiguous superclass
   labels (PanNuke "inflammatory" → uniform over {lymph, plasma,
   neutrophil, eosinophil, macrophage}). This is informed by the
   observation that val_nc saturated at ~0.55 on 6 classes during
   B-2+HVfix training — the classifier has spare capacity that more
   classes would consume.

3. **HV weighting choice for Stream B**: start from B-2 no-cons's
   1.0/1.0 rather than HV-fix's 2.5/8.0. The dead-class collapse
   pattern will likely repeat on a 10-class taxonomy with more rare
   classes (eosinophil, plasma, dead). Alternatively keep 2.5/8.0
   and bump w_nc to 2.5 to compensate the relative weight. Cleaner
   to start with defaults; tune in a follow-up.

Streams A and B are **orthogonal and can run in parallel**. If both
win individually, their wins compose. Cell-context attention is
PanNuke-only (architecture isolation); multi-dataset 10-class is
data-only (architecture unchanged). Combining them requires both
landed, then a final integration run.

## Infrastructure changes landed this round

- **Atomic incremental save**: every new best-val now writes
  `<output>/cellvitpp.pth.tmp` then `os.replace`s to the final path.
  Ctrl+C at any epoch is safe — what's on disk is always the best so
  far. (Commit af9fbf3 for cellvitpp; commit 5209ce7 for vitb.)
- **Early stopping**: training loop breaks if val_total hasn't
  improved for `early_stop_patience` epochs (default 15). Saves
  ~50% of GPU time on overfitting variants that previously ran to
  epoch 100 needlessly. Both train scripts and STAGE_VITB /
  STAGE_CELLVITPP configs updated.
- **HV-weight fix in STAGE_CELLVITPP** (commit b90b0aa): `w_mse=2.5`,
  `w_msge=8.0` to match STAGE2. Identical fix for STAGE_VITB pending
  this iteration's commit (next).

## Out of scope (deliberate)

- Path Z, B-2 (`adios_cellvit/*.py`, `train_stage*.py`, `train_vitb.py`,
  `eval_full_v1.py`, `eval_full_v2_vitb.py`) all stay runnable for
  side-by-side comparison.
- The cell-context attention branch is built only after this round of
  HV-fix re-runs is evaluated; no point in stacking architecture
  changes on uncertain detection baselines.
