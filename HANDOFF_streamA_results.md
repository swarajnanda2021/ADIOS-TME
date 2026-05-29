# Stream A results — cell-context attention (close-out)

## What was run

Branch `cellvitpp-context`. The B-3 per-cell pooled-token MLP
(`CellTokenClassifier`) is replaced by `CellContextClassifier`: a transformer
over per-cell pooled tokens with a parameter-free 2D sinusoidal positional
encoding from cell centroids and a per-image key-padding mask, so cells in an
image attend to one another before classification. NP/HV decoders, pooling,
optimizer schedule, dataset all unchanged. Code:
`adios_cellvit/cell_context_attention.py`,
`adios_cellvit/vitb_cellvitpp_context_model.py`,
`train_vitb_cellvitpp_context.py`.

Two runs, both no-cons, PanNuke Test (2722 patches, IoU=0.5, `eval_full_v3_cellvitpp.py`):

1. `logs/cellvitpp_context` @ HV **2.5/8.0** — **CONFOUNDED** (see note).
   Report: `eval/eval_report_context_w2580_noprior.txt`.
2. `logs/cellvitpp_context_w11` @ HV **1.0/1.0** — the **clean, controlled** run.
   Report: `eval/eval_report_context_w11_noprior.txt`.

## Methodology note (why two runs)

The first run trained at HV weights 2.5/8.0 while its comparison baseline,
B-3 no-cons (`logs/cellvitpp_noprior`), trained at 1.0/1.0 — so head and
HV-weights both changed and the comparison could not isolate the attention
head. The 2.5/8.0 run's apparent connective lift (recall 0.610) and AJI bump
(0.476) were therefore uninterpretable. The clean test is run (2) at 1.0/1.0,
where the **only** delta vs B-3 no-cons is the classifier head. (Lesson: an
architecture ablation must match the baseline's HV weights; the 2.5/8.0
production default does not apply to a controlled comparison.)

## Clean head-only comparison (all 1.0/1.0, no-cons)

| metric              | B-2 no-cons (per-pixel, winner) | B-3 no-cons (per-cell MLP) | Stream A (per-cell + attn) | attn vs MLP |
|---------------------|---------------------------------|----------------------------|----------------------------|-------------|
| NC macro-F1         | 0.737                           | 0.706                      | **0.661**                  | **-0.045**  |
| connective F1       | 0.696                           | 0.622                      | 0.599                      | -0.023      |
| connective recall   | 0.646                           | 0.539                      | 0.582                      | +0.043      |
| connective precision| 0.753                           | 0.733                      | 0.617                      | **-0.116**  |
| inflammatory recall | 0.877                           | 0.857                      | 0.737                      | -0.120      |
| neoplastic F1       | 0.873                           | 0.866                      | 0.813                      | -0.053      |
| epithelial F1       | 0.850                           | 0.826                      | 0.762                      | -0.064      |
| dead recall         | 0.812                           | 0.803                      | 0.783                      | -0.020      |
| AJI                 | 0.481                           | 0.443                      | 0.438                      | -0.005      |
| Instance PQ         | 0.448                           | 0.412                      | 0.391                      | -0.021      |
| NP IoU              | 0.721                           | 0.724                      | 0.712                      | -0.012      |
| count MAE           | 6.52                            | 7.02                       | 5.48                       | -1.54*      |

\* count MAE improves only via over-segmentation (pred count 14.7 vs 12.9, FP
14193 vs 9827, PQ down) — not better detection.

## Findings

1. **Net regression vs the plain MLP.** At matched weights, attention scores
   macro-F1 0.661 — below the per-cell MLP (0.706) and the per-pixel winner
   (0.737) — and is worse on **every** per-class F1.

2. **Connective not recovered.** Recall rises (+0.043) but precision collapses
   (-0.116; connective false-positives doubled, 1043 -> 2115). Connective F1
   (0.599) is *below* the MLP (0.622) and far from per-pixel (0.696). Attention
   raises connective recall only by over-assigning the class.

3. **Over-smoothing is the mechanism.** Stroma/connective is spatially
   pervasive, so a cell's neighbors are disproportionately connective;
   neighbor-attention pulls cells toward connective. Hence connective recall up
   but inflammatory recall down -0.120 (inflam->connective 349 -> 772) and
   epithelial F1 down -0.064. The transformer's residual stream did not
   preserve enough of each cell's own identity.

4. **AJI coupling disproven.** At matched weights AJI is flat (0.438 vs B-3's
   0.443) and NP IoU is flat (0.712 vs 0.724). The 0.476 AJI in the 2.5/8.0 run
   was the HV-weight change, not the attention head. Confirms the classifier is
   upstream of detection: it does not influence AJI/count (the residual
   count/FP difference at matched weights is most likely training variance).

## Verdict

Pre-registered criterion: *recover the connective regression -> per-cell earns
its keep; otherwise per-cell is permanently abandoned.* Cell-context attention
did **not** recover connective (F1 down) and is a **net regression** vs even the
plain per-cell MLP. **Per-cell architecture is abandoned. B-2 no-cons
(per-pixel) remains the reference winner.**

Open option (user's call, goalpost-moving): a gated / single-layer attention
variant to curb over-smoothing (let a cell's own token dominate its neighbors),
re-tested at 1.0/1.0. Not pursued absent explicit direction.

## Artifacts

- Checkpoints (cluster): `logs/cellvitpp_context_w11` (clean 1.0/1.0),
  `logs/cellvitpp_context` (2.5/8.0, confounded).
- Eval reports: `eval/eval_report_context_w11_noprior.txt` (clean),
  `eval/eval_report_context_w2580_noprior.txt` (confounded).
- Config: `STAGE_CELLVITPP_CONTEXT` (now 1.0/1.0, matched to the B-3 baseline).
