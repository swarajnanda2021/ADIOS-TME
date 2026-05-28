# Path B-2 results — iteration close-out

## What was run

Three configurations, all evaluated on PanNuke Test (2722 patches,
IoU=0.5 instance matching, identical eval harness `eval_full_v2_vitb.py`):

1. Path Z (reference, prior iteration): ViT-Tiny (192-d) ADIOS encoder
   + ADIOS UNet decoder + selector-collapsed NP + CellViT HV/NC heads.
2. Path B-2 with consistency: ViT-B (768-d) FMC encoder + three-branch
   CellViT (NP/HV/NC) + frozen ADIOS-prior consistency BCE at λ=0.1.
3. Path B-2 no consistency: same as (2) with the consistency term off.

Encoder for B-2: `FMC_ViT-B_baseline/logs/checkpoint_iter_00150000.pth`
(patch_size 16, embed_dim 768, depth 12, heads 12, 4 register tokens,
loaded clean: 0 missing / 0 unexpected keys, 176 backbone tensors).

Stage 1 selector reused for the consistency prior: val_acc 0.976,
val_loss 0.0739 (60 epochs, batch 32). Matches the prior iteration's
0.977.

Training: 100 epochs, batch 16, encoder LR 1e-5 (constant, pinned),
heads LR 1e-4 (warmup 2 ep -> decay to 1e-5). Both B-2 runs hit best
val_total near epochs 41-50 and overfit thereafter; the best-val
checkpoint was saved and evaluated.

## Headline metrics

| metric                | Path Z | B-2 +cons (λ=0.1) | B-2 no-cons |
|-----------------------|--------|-------------------|-------------|
| AJI                   | 0.448  | 0.407             | 0.481       |
| Instance PQ           | 0.426  | 0.361             | 0.448       |
| NP IoU                | 0.730  | 0.722             | 0.721       |
| NP recall             | 0.858  | 0.851             | 0.863       |
| NP precision          | 0.830  | 0.826             | 0.814       |
| NC macro-F1 (matched) | 0.626  | 0.748             | 0.737       |
| neoplastic recall     | 0.52   | 0.834             | 0.833       |
| inflammatory recall   | 0.55   | 0.876             | 0.877       |
| connective recall     | 0.32   | 0.639             | 0.646       |
| dead recall           | 0.00   | 0.844             | 0.812       |
| epithelial recall     | 0.50   | 0.886             | 0.870       |
| count MAE             | 5.95   | 7.35              | 6.52        |
| count bias            | -5.85  | -7.19             | -6.41       |
| matched pairs (TP)    | -      | 22256             | 27549       |
| unmatched GT (FN)     | -      | 31742             | 26449       |

## Findings

### 1. The ViT-B encoder swap is the win
No-consistency B-2 beats Path Z on AJI (0.481 vs 0.448), PQ (0.448 vs
0.426), and NC macro-F1 (0.737 vs 0.626). The per-class recall jumps
are large and uniform across all five classes. The 768-d FMC encoder
plus dense PanNuke supervision is materially stronger than the
192-d ADIOS-pretrained ViT-Tiny it replaces.

### 2. Class-weighted NC cross-entropy solves the dead-class collapse
Path Z detected zero dead nuclei across the entire test set (recall
0.00). Inverse-frequency NC weights (dead weight ~5.5x, computed from
PanNuke train pixel counts, capped at 20 — cap did not bind) lift dead
recall to 0.81-0.84 in both B-2 runs. Dead precision is low (0.35-0.37):
the model now over-calls dead. Dead F1 ~0.49-0.51. The failure mode
flipped from "never predicts dead" to "over-predicts dead" — net
improvement, with precision now the bottleneck.

### 3. The ADIOS-prior consistency loss is a trade-off, not a clear win or loss
At λ=0.1, consistency **reduces instance metrics** (AJI 0.481->0.407,
PQ 0.448->0.361) while **marginally improving classification quality
on detected nuclei** (NC macro-F1 0.737->0.748).

The instance regression traces to under-detection: with consistency the
model produces 22256 matched pairs vs 27549 without, and misses 5293
more GT nuclei (FN 31742 vs 26449). Count bias worsens (-7.19 vs -6.41).
Mechanism (hypothesized): the ADIOS prior produces blob-like,
over-smooth nucleus masks; asking the NP map to agree with it encourages
merging adjacent nuclei, which watershed then fails to split, dropping
matched-instance IoU below 0.5 and converting true instances into FN+FP.

The classification edge is real but small and partly an artifact of the
matched-only denominator: the consistency model evaluates class on a
smaller, more conservative subset of detections (e.g. inflammatory
precision 0.725 vs 0.696 at identical recall ~0.877). Per-class instance
PQ — which folds detection and classification together — favors
no-consistency on **every class**.

Conclusion: for a nuclei *counter* (total per-class count accuracy is
the deliverable), no-consistency is the better configuration. The
consistency prior communicates "where" but degrades "how many," and
"how many" is the product metric. The hypothesis is not falsified for
all use cases — a precision-prioritizing deployment that only scores
detected nuclei could prefer the consistency variant — but it does not
help the counting objective at λ=0.1.

## Current best configuration
Path B-2 no-consistency: AJI 0.481, PQ 0.448, NC macro-F1 0.737, all
five classes detected at recall >= 0.65. This is the new reference
baseline, replacing Path Z.

## Open questions for the next iteration

1. **Loss formulation for instance separation.** Both B-2 runs overfit
   after epoch ~45 (val_total rises while train falls), and both
   under-count (bias -6 to -7 per patch). The bottleneck is instance
   *separation*, not pixel coverage (NP IoU is flat at ~0.72 across all
   three models). Candidate directions: stronger HV-map weighting
   (Path Z's STAGE2 used w_mse=2.5, w_msge=8.0; the B-2 runs used the
   default 1.0/1.0 — this is a real un-tuned knob), boundary-aware loss,
   or a dedicated instance-separation term. The HV-weight discrepancy
   between STAGE2 and STAGE_VITB is the single most likely lever and
   should be the first thing tried.

2. **Dead-class precision.** Recall is solved (0.81-0.84); precision is
   the problem (0.35-0.37). Try lowering the NC weight cap from 20 to
   ~5, or a per-class confidence threshold, or focal loss.

3. **Overfitting past epoch 45.** Both runs peaked early. Either
   early-stop on val, add regularization/augmentation, or shorten the
   schedule. The val set also currently uses train-time augmentation
   (inherited from Path Z's loader), which makes val_total a noisy
   estimator — worth fixing for cleaner early-stop signals.

4. **Low-λ consistency sweep (low priority).** If the classification
   edge is worth pursuing, λ in {0.01, 0.02, 0.05} might find a regime
   where the prior nudges class without merging instances. Not the
   highest-leverage next step given finding (1).

## Infrastructure issues surfaced this iteration

These are `assemble_cluster.sh` problems found and (partially) fixed:

- **FIXED (commit fc2cf8d):** PHASE C's benchmarking.py namespace patch
  used a literal-string anchor that drifted from PostProc source; it
  fell through with a warning instead of failing, leaving bare imports
  that crashed eval. Replaced with a regex rewrite that hard-fails if it
  makes zero changes.
- **NOT FIXED:** PHASE B unconditionally `git checkout nuclei-counter`,
  swapping the user off their working branch mid-assemble. Worked around
  manually each run (re-checkout + re-apply PHASE D patch + re-fill
  placeholders). Should accept a branch argument or detect/restore the
  current branch.
- **NOT FIXED:** PostProc source files are not pinned to a commit, so
  every `assemble_cluster.sh` literal-string anchor (PHASE C imports,
  PHASE D CellViT signature) is one PostProc edit away from drift.
  Consider AST-based patching or pinning PostProc to a known commit.
- **COSMETIC:** Albumentations `Downscale(scale_min, scale_max)`
  deprecation warning (inherited from Path Z's dataset transforms).
