# Path B-3 (CellViT++-style) handoff — per-cell MLP classifier on top of NP/HV

## TL;DR

Path B-3 replaces the per-pixel NC decoder branch with a lightweight
MLP that consumes pooled per-cell tokens from the ViT-B encoder's last
block, matching the cell-classification module from CellViT++. The NP
and HV branches stay exactly as in Path B-2. Two ablations
(use_adios_consistency=True/False) run from the same training script.
The Path B-2 scripts (train_vitb.py, eval_full_v2_vitb.py) and model
(ViTBCellViT) stay untouched so the B-2 result table remains the
reference.

## What's new on this branch

- `adios_cellvit/vitb_cellvitpp_model.py` — `ViTBCellViTPP` model and
  `CellTokenClassifier` MLP (1 hidden layer + ReLU + dropout). Reuses
  the post-PHASE-D CellViT for NP/HV; the per-pixel NC decoder is left
  in the module tree but frozen and never called (avoids changing
  PHASE D, keeps assemble_cluster.sh idempotent).
- `train_vitb_cellvitpp.py` — training mirror of `train_vitb.py`.
  Per-cell CE on K=5 foreground classes; class weights computed from
  cells-per-class (not pixels-per-class) and cached to
  `cell_class_weights.json`. ADIOS prior consistency knob preserved.
- `eval_full_v3_cellvitpp.py` — eval mirror of `eval_full_v2_vitb.py`.
  Same metric definitions; classification path runs the MLP on
  watershed-predicted instances after the first forward pass.
- `configs/nuclei_counter.py` — adds `STAGE_CELLVITPP` (placeholder tag
  `<FILL ON CLUSTER (CELLVITPP)>` to stay invisible to
  assemble_cluster.sh PHASE E).

## How this differs from B-2 — experimental delta

| | B-2 (per-pixel NC) | B-3 (per-cell MLP) |
|---|---|---|
| Encoder | ViT-B (FMC) | ViT-B (FMC) — identical init, identical schedule |
| NP head | CellViT NP decoder, 2-ch softmax | unchanged |
| HV head | CellViT HV decoder | unchanged |
| NC head | CellViT NC decoder, 6 channels, per-pixel CE | last-block tokens pooled per instance, MLP(768→384→5) |
| NC loss target | per-pixel class_mask ∈ {0..5} | per-cell modal class, shifted to {0..4} |
| NC class weights | inverse-frequency on pixels (length 6) | inverse-frequency on cells (length 5) |
| When instance identity enters | post-hoc, at eval (modal vote inside watershed instances) | at training (GT instances) and at eval (watershed instances) |
| Params (head) | ~5M (NC decoder branch) | ~590K (MLP) |
| All other hyperparameters | STAGE_VITB defaults | STAGE_CELLVITPP — same values |

End-to-end, fresh from FMC ViT-B init, same encoder_lr 1e-5 + heads_lr
1e-4, same warmup/decay schedule, same dataset, same NP/HV/MSGE weights
(1.0/1.0/1.0/1.0). Only the NC head architecture and loss differ.
This is the *isolating* experiment vs B-2 no-cons.

## Token pooling spec

For each foreground instance in an instance mask:

1. Build binary [H, W] pixel mask of that instance.
2. `F.max_pool2d(mask, kernel=stride=patch_size)` → [fs, fs] binary
   patch-overlap grid (14×14 for ViT-B/16 @ 224). A patch is "touched"
   iff any of its pixels belong to the instance.
3. Average the encoder's last-block patch tokens at the touched
   patches → single [D] cell embedding.
4. MLP → 5 class logits.

Sub-patch nuclei (no touched patches after max-pool) are skipped — at
training they're absent from CE targets, at eval they fall back to
class 1 (neoplastic). Rare in practice on PanNuke with patch_size=16.

## Three ablations planned (with reference rows)

Run from the same training script via `use_adios_consistency` flag:

| Run | Script | use_adios_consistency | NC head |
|---|---|---|---|
| Path Z baseline | `train_stage2_cellvit.py` (unchanged) | n/a (ADIOS decoder is the backbone) | per-pixel NC, 6-ch |
| B-2 no-cons (current best) | `train_vitb.py` | False | per-pixel NC, 6-ch |
| B-2 with-cons | `train_vitb.py` | True (λ=0.1) | per-pixel NC, 6-ch |
| **B-3 no-cons** | `train_vitb_cellvitpp.py` | False | per-cell MLP, 5 |
| **B-3 with-cons** | `train_vitb_cellvitpp.py` | True (λ=0.1) | per-cell MLP, 5 |

Per the conversation that led here: we run both with-cons and no-cons
for B-3 even though B-2 with-cons hurt detection — the question is
whether per-cell pooling changes how the ADIOS consistency signal
interacts with the encoder. We will tune the two together afterward.

## Hypotheses being tested

What we expect to move (relative to B-2 no-cons):

- **NC macro-F1 should improve.** Per-cell pooling avoids the
  pixel-vote-inside-watershed artifact that mixes neighbor pixels.
  Dead F1 in particular: currently bottlenecked by precision 0.35;
  classifying *whole cells* rather than pixels should help.
- **AJI / PQ / count MAE should be ~unchanged.** NP and HV branches
  are identical to B-2; the encoder gradient now flows through
  per-cell CE instead of per-pixel CE but the dominant detection
  bottleneck (HV-weight tuning, finding (1) of HANDOFF_pathB2_results)
  is untouched. If detection moves materially in either direction
  that's worth flagging.
- **with-cons vs no-cons gap might differ.** In B-2, +cons traded
  classification (+0.011 macro-F1) for detection (-0.087 PQ).
  Mechanism in B-2 was via the per-pixel NC decoder sharing
  decoder0..3 with the NP branch. With B-3 the NC head is a
  separate MLP; the consistency-loss → encoder → other-heads
  feedback path is different.

## Cluster assembly + run plan

`assemble_cluster.sh` is unmodified by this branch. PHASE E fills the
literal `<FILL ON CLUSTER>` placeholders (STAGE1/2 only); the
`<FILL ON CLUSTER (CELLVITPP)>` tags in STAGE_CELLVITPP are
substring-distinct so PHASE E ignores them and doesn't crash. The user
fills them manually after assemble completes — same pattern as
STAGE_VITB's `(VITB)` tag.

Cluster recipe (after `git pull` to this branch):

```bash
cd /data1/vanderbc/test_dinov2_swaraj/ADIOS
bash assemble_cluster.sh   # idempotent re-run; fills STAGE1/2 placeholders

# Fill the five STAGE_CELLVITPP placeholders (or sed):
#   vitb_checkpoint   -> .../FMC_ViT-B_baseline/logs/checkpoint_iter_00150000.pth
#   adios_checkpoint  -> same as STAGE2['adios_checkpoint']
#   stage1_selector   -> ./logs/stage1/stage1_selector.pth
#   pannuke_path      -> .../data/pannuke
#   output_dir        -> ./logs/cellvitpp     (or ./logs/cellvitpp_noprior for the ablation)

# B-3 with-cons:
python train_vitb_cellvitpp.py --config configs/nuclei_counter.py

# B-3 no-cons:
python train_vitb_cellvitpp.py --config configs/nuclei_counter.py \
    --output_dir ./logs/cellvitpp_noprior \
    --use_adios_consistency false

# Evaluate:
python eval_full_v3_cellvitpp.py \
    --checkpoint ./logs/cellvitpp/cellvitpp.pth \
    --vitb_checkpoint .../FMC_ViT-B_baseline/logs/checkpoint_iter_00150000.pth \
    --pannuke_path .../data/pannuke \
    --output_dir ./logs/eval/cellvitpp
```

## Open items to verify at first cluster run

1. **`cell_class_weights` distribution.** Counts cells (distinct
   instance IDs) per class across the Training class_masks. If the
   dead-class weight pre-cap is > 50 (very tiny class), the cap kicks
   in to 20.0 and is renormalized. The script prints pre-cap and
   post-cap distributions — eyeball them once.
2. **Sub-patch nuclei.** The training loop skips cells with zero
   touched patches; if this drops more than ~1% of GT cells per batch,
   patch_size=16 is too coarse and we'd want to log this number.
3. **Param count + state-dict shape.** ViTBCellViTPP keeps the unused
   per-pixel NC decoder around. Expect param count ~142M (same as B-2)
   plus ~590K for the new MLP — net ~143M. If much higher, something
   leaked into requires_grad.

## Out of scope

- Path Z files and Path B-2 files (`adios_cellvit/{adios_backbone,
  adios_cellvit_model, channel_selector, pannuke_dataset,
  vitb_backbone, vitb_cellvit_model}.py`, `train_stage*.py`,
  `train_vitb.py`, `eval_full_v{1,2}_vitb.py`, etc.) — untouched. B-2
  remains runnable for direct comparison.
- `assemble_cluster.sh` — untouched. PHASE D's CellViT NC-branch patch
  is reused as-is; the unused NC decoder doesn't hurt anything.
- HV-weight tuning (finding (1) from HANDOFF_pathB2_results.md) — that
  is a separate experiment, deliberately not changed here so this
  isolates the NC-head architecture variable.
