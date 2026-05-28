#!/usr/bin/env bash
# =============================================================================
# assemble_cluster.sh  (v2 — unified data prep + code assembly)
#
# Phases:
#   A. Prepare PanNuke data into unified layout under <WORK_DIR>/data/pannuke/
#        - tissue images and instance masks   <- PanNuke_patches_unnormalized
#        - 5 foreground class PNGs (no non_neoplastic) <- ..._wtypes
#   B. Clone repo into <WORK_DIR>, check out 'nuclei-counter' branch.
#   C. Copy PostProc files into cellvit/.
#   D. Patch cellvit/models.py to add the NC decoder branch.
#   E. Fill config placeholders in configs/nuclei_counter.py.
#   F. Write submitit submission scripts for stage 1 and stage 2.
#
# Idempotent at every phase.  Re-running skips work that's already done.
#
# Flags:
#   --force        Wipe everything in WORK_DIR EXCEPT data/ before rebuilding.
#                  (Data prep is expensive; preserve it across code rebuilds.)
#   --force-data   Also wipe data/.  Combine with --force for a full clean rebuild.
#
# Usage:
#   bash assemble_cluster.sh
#   bash assemble_cluster.sh --force
#   bash assemble_cluster.sh --force --force-data
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — edit paths if your cluster layout differs
# ---------------------------------------------------------------------------
WORK_DIR="/data1/vanderbc/test_dinov2_swaraj/ADIOS"
REPO_URL="https://github.com/swarajnanda2021/ADIOS-TME.git"
BRANCH="nuclei-counter"

POSTPROC="/data1/vanderbc/nandas1/PostProc"

# Source PanNuke directories
PANNUKE_PLAIN="/data1/vanderbc/nandas1/Benchmarks/PanNuke_patches_unnormalized"
PANNUKE_WTYPES="/data1/vanderbc/nandas1/Benchmarks/PanNuke_patches_unnormalized_wtypes"

# Unified PanNuke target (under WORK_DIR; the dataloader points here)
PANNUKE_UNIFIED_REL="data/pannuke"        # relative to WORK_DIR
ADIOS_CHECKPOINT="/data1/vanderbc/nandas1/ADIOS-CellViT/logs/checkpoint_iter_00094000.pth"

# Splits and magnification we copy.  PanNuke has both 20x and 40x; we use 40x.
SPLITS=(Training Test)
MAGNIFICATION="40x"
FG_CLASSES=(neoplastic inflammatory connective dead epithelial)
# Intentionally NOT in FG_CLASSES: non_neoplastic.  In the wtypes folder it's
# the background mask, not a class.

# SLURM defaults (edit to match your cluster)
SLURM_PARTITION="your-slurm-partition-name"
SLURM_CONSTRAINT="h100"
NGPUS=1
NODES=1
TIMEOUT_MIN=2880

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()   { echo "[assemble] $*"; }
die()   { echo "[assemble] ERROR: $*" >&2; exit 1; }
skip()  { log "SKIP: $*"; }
ok()    { log "OK:   $*"; }

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
FORCE=0
FORCE_DATA=0
for arg in "$@"; do
    case $arg in
        --force)      FORCE=1 ;;
        --force-data) FORCE_DATA=1 ;;
        -h|--help)
            sed -n '2,30p' "$0"
            exit 0
            ;;
        *) die "Unknown argument: $arg" ;;
    esac
done

# ---------------------------------------------------------------------------
# Step 0: Force-wipe if requested.  Data is preserved unless --force-data.
# ---------------------------------------------------------------------------
if [[ $FORCE -eq 1 ]]; then
    if [[ -d "$WORK_DIR" ]]; then
        if [[ $FORCE_DATA -eq 1 ]]; then
            log "FORCE + FORCE-DATA: wiping entire $WORK_DIR (data included)"
            rm -rf "$WORK_DIR"
        else
            log "FORCE: wiping $WORK_DIR EXCEPT data/ (use --force-data to also wipe data)"
            # Move data aside, wipe, move back.  Safer than find-with-prune
            # on a directory we may not own.
            if [[ -d "$WORK_DIR/data" ]]; then
                TMP_DATA=$(mktemp -d /tmp/assemble_data_XXXXXX)
                mv "$WORK_DIR/data" "$TMP_DATA/"
                rm -rf "$WORK_DIR"
                mkdir -p "$WORK_DIR"
                mv "$TMP_DATA/data" "$WORK_DIR/data"
                rmdir "$TMP_DATA"
            else
                rm -rf "$WORK_DIR"
                mkdir -p "$WORK_DIR"
            fi
        fi
    fi
fi

mkdir -p "$WORK_DIR"

# =============================================================================
# PHASE A: Prepare PanNuke data into unified layout
# =============================================================================
log "Phase A: Prepare PanNuke unified data layout"

PANNUKE_DST="$WORK_DIR/$PANNUKE_UNIFIED_REL"

# Verify source folders exist
[[ -d "$PANNUKE_PLAIN"  ]] || die "Source not found: $PANNUKE_PLAIN"
[[ -d "$PANNUKE_WTYPES" ]] || die "Source not found: $PANNUKE_WTYPES"

# Helper: count .png / .npy files in a dir (excludes subdirs)
count_files() {
    local dir="$1"
    local ext="$2"
    [[ -d "$dir" ]] || { echo 0; return; }
    ls "$dir" 2>/dev/null | grep -c -E "\.${ext}$" || true
}

# For each split, magnification: build the unified layout.
for split in "${SPLITS[@]}"; do
    base="$PANNUKE_DST/$split/$MAGNIFICATION"
    src_plain_tissue="$PANNUKE_PLAIN/$split/$MAGNIFICATION/tissue_images"
    src_plain_inst="$PANNUKE_PLAIN/$split/$MAGNIFICATION/masks"
    src_wtypes_class="$PANNUKE_WTYPES/$split/$MAGNIFICATION/masks"

    [[ -d "$src_plain_tissue" ]] || die "Missing source: $src_plain_tissue"
    [[ -d "$src_plain_inst"   ]] || die "Missing source: $src_plain_inst"
    [[ -d "$src_wtypes_class" ]] || die "Missing source: $src_wtypes_class"

    # --- A.1: tissue_images ---
    dst_tissue="$base/tissue_images"
    src_count=$(count_files "$src_plain_tissue" "png")
    dst_count=$(count_files "$dst_tissue" "png")
    if [[ "$dst_count" -eq "$src_count" && "$src_count" -gt 0 ]]; then
        skip "$split/$MAGNIFICATION/tissue_images ($src_count files already present)"
    else
        log "Copying $split/$MAGNIFICATION/tissue_images ($src_count files)..."
        mkdir -p "$dst_tissue"
        # cp -n: don't overwrite existing files (idempotent at per-file level)
        cp -n "$src_plain_tissue"/*.png "$dst_tissue/" 2>/dev/null || true
        new_count=$(count_files "$dst_tissue" "png")
        [[ "$new_count" -eq "$src_count" ]] || die "tissue_images copy incomplete: $new_count/$src_count"
        ok "Copied $new_count tissue images to $dst_tissue"
    fi

    # --- A.2: instance_masks (from plain/masks/*.npy) ---
    dst_inst="$base/instance_masks"
    src_count=$(count_files "$src_plain_inst" "npy")
    dst_count=$(count_files "$dst_inst" "npy")
    if [[ "$dst_count" -eq "$src_count" && "$src_count" -gt 0 ]]; then
        skip "$split/$MAGNIFICATION/instance_masks ($src_count files already present)"
    else
        log "Copying $split/$MAGNIFICATION/instance_masks ($src_count files)..."
        mkdir -p "$dst_inst"
        cp -n "$src_plain_inst"/*.npy "$dst_inst/" 2>/dev/null || true
        new_count=$(count_files "$dst_inst" "npy")
        [[ "$new_count" -eq "$src_count" ]] || die "instance_masks copy incomplete: $new_count/$src_count"
        ok "Copied $new_count instance masks to $dst_inst"
    fi

    # --- A.3: class_masks/{class}/*.png (5 foreground classes only) ---
    dst_class_root="$base/class_masks"
    mkdir -p "$dst_class_root"
    for cls in "${FG_CLASSES[@]}"; do
        src_cls="$src_wtypes_class/$cls"
        dst_cls="$dst_class_root/$cls"
        [[ -d "$src_cls" ]] || die "Missing class source: $src_cls"
        src_count=$(count_files "$src_cls" "png")
        dst_count=$(count_files "$dst_cls" "png")
        if [[ "$dst_count" -eq "$src_count" && "$src_count" -gt 0 ]]; then
            skip "$split/$MAGNIFICATION/class_masks/$cls ($src_count files)"
        else
            log "Copying $split/$MAGNIFICATION/class_masks/$cls ($src_count files)..."
            mkdir -p "$dst_cls"
            cp -n "$src_cls"/*.png "$dst_cls/" 2>/dev/null || true
            new_count=$(count_files "$dst_cls" "png")
            [[ "$new_count" -eq "$src_count" ]] || die "class_masks/$cls copy incomplete: $new_count/$src_count"
            ok "Copied $new_count class masks to $dst_cls"
        fi
    done

    # --- A.4: Sanity check counts across all four subtrees ---
    tissue_n=$(count_files "$base/tissue_images" "png")
    inst_n=$(count_files "$base/instance_masks" "npy")
    neop_n=$(count_files "$base/class_masks/neoplastic" "png")
    if [[ "$tissue_n" -ne "$inst_n" || "$tissue_n" -ne "$neop_n" ]]; then
        die "Patch count mismatch in $split/$MAGNIFICATION: tissue=$tissue_n instance=$inst_n neoplastic=$neop_n"
    fi
    log "$split/$MAGNIFICATION: $tissue_n patches across all subtrees, consistent."
done

# Write a small README inside the unified layout for future readers
README="$PANNUKE_DST/README.md"
if [[ ! -f "$README" ]]; then
    cat > "$README" <<README_EOF
# Unified PanNuke layout for ADIOS nuclei-counter training

Built by \`assemble_cluster.sh\` from two source directories:

- \`$PANNUKE_PLAIN\` — tissue images + instance masks (uint16 .npy)
- \`$PANNUKE_WTYPES\` — per-class instance-labeled PNGs (5 classes; \`non_neoplastic\` is background and is excluded)

Layout:

\`\`\`
$PANNUKE_UNIFIED_REL/
├── Training/$MAGNIFICATION/
│   ├── tissue_images/<patch>.png
│   ├── instance_masks/<patch>.npy
│   └── class_masks/
│       ├── neoplastic/<patch>.png
│       ├── inflammatory/<patch>.png
│       ├── connective/<patch>.png
│       ├── dead/<patch>.png
│       └── epithelial/<patch>.png
└── Test/$MAGNIFICATION/
    └── (same structure)
\`\`\`

Class index mapping (used by \`ADIOSPanNukeDataset\`):

| Index | Class        |
|-------|--------------|
| 0     | background   |
| 1     | neoplastic   |
| 2     | inflammatory |
| 3     | connective   |
| 4     | dead         |
| 5     | epithelial   |
README_EOF
    ok "Wrote $README"
fi

# =============================================================================
# PHASE B: Clone repo + checkout branch
# =============================================================================
log "Phase B: Repo clone / sync"

cd "$WORK_DIR"

if [[ ! -d ".git" ]]; then
    # WORK_DIR is non-empty (we just put data/ in it), so we need to clone
    # into a temp dir and merge.
    log "Cloning $REPO_URL into $WORK_DIR"
    TMP_CLONE=$(mktemp -d /tmp/adios_clone_XXXXXX)
    git clone "$REPO_URL" "$TMP_CLONE"
    cd "$TMP_CLONE"
    git checkout "$BRANCH"
    cd - > /dev/null
    # Move git-tracked content into WORK_DIR.  data/ is already there.
    # Use cp -rn to avoid clobbering data/.
    cp -rn "$TMP_CLONE/." "$WORK_DIR/"
    # Move .git itself into WORK_DIR.
    mv "$TMP_CLONE/.git" "$WORK_DIR/.git"
    rm -rf "$TMP_CLONE"
    ok "Cloned and checked out $BRANCH"
else
    log "Repo exists; fetching and syncing $BRANCH"
    git fetch origin
    if ! git diff --quiet || ! git diff --cached --quiet; then
        log "Stashing local changes from previous assembly"
        git stash push -m "assemble_cluster.sh auto-stash $(date -Iseconds)"
    fi
    git checkout "$BRANCH"
    git pull --ff-only origin "$BRANCH"
    ok "Synced $BRANCH"
fi

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
[[ "$CURRENT_BRANCH" == "$BRANCH" ]] || die "Expected branch $BRANCH, got $CURRENT_BRANCH"

# =============================================================================
# PHASE C: Copy PostProc files into cellvit/
# =============================================================================
log "Phase C: Copy PostProc files into cellvit/"

[[ -d "$POSTPROC" ]] || die "PostProc directory not found: $POSTPROC"

mkdir -p cellvit cellvit/postproc

copy_postproc() {
    local src="$1"
    local dst="$2"
    [[ -f "$src" ]] || die "PostProc source missing: $src"
    if [[ -f "$dst" ]]; then
        local src_size dst_size
        src_size=$(stat -c%s "$src")
        dst_size=$(stat -c%s "$dst")
        if [[ "$src_size" -ne "$dst_size" ]]; then
            log "WARNING: $dst exists but differs from $src (sizes differ)"
            log "         Use --force to refresh."
        fi
        skip "$dst already present"
    else
        cp "$src" "$dst"
        ok "Copied $src -> $dst"
    fi
}

copy_postproc "$POSTPROC/datasets.py"     "cellvit/datasets.py"
copy_postproc "$POSTPROC/models.py"       "cellvit/models.py"
copy_postproc "$POSTPROC/utils.py"        "cellvit/utils.py"
copy_postproc "$POSTPROC/benchmarking.py" "cellvit/postproc/benchmarking.py"

# Make cellvit/ a proper Python package
touch cellvit/__init__.py cellvit/postproc/__init__.py

# ---------------------------------------------------------------------
# Phase C.1: post-copy patches for vendored cellvit files
# ---------------------------------------------------------------------
# The vendored cellvit files have two known issues that this project
# works around. Apply them here so re-running assembly preserves them.

# C.1.1: cellvit/datasets.py — widen mask slice in SynchronizedTransform.
# The original code does mask = transformed_combined[:, :, 3:4], which
# collapses multi-channel masks to a single channel. ADIOSPanNukeDataset
# packs instance + class into a 2-channel mask, so we need [:, :, 3:].
DATASETS_PY="$WORK_DIR/cellvit/datasets.py"
if grep -q "transformed_combined\[:, :, 3:\]" "$DATASETS_PY"; then
    skip "cellvit/datasets.py already patched (mask slice)"
else
    sed -i 's|mask = transformed_combined\[:, :, 3:4\]|mask = transformed_combined[:, :, 3:]|' "$DATASETS_PY"
    ok "Patched $DATASETS_PY (mask slice 3:4 → 3:)"
fi

# C.1.2: cellvit/postproc/benchmarking.py — namespace-qualify all bare
# imports of utils / datasets / models. PostProc was written assuming a
# flat sys.path; in this project layout `from utils import ...` resolves
# to the project-root utils.py (the DINOv2 fork's), which lacks
# set_seed / WarmupDecayScheduler.  Anchor-free regex rewrite — tolerant
# of PostProc source reordering.  Hard-fails if zero rewrites apply, so
# future drift surfaces at assemble time instead of during eval.
BENCH_PY="$WORK_DIR/cellvit/postproc/benchmarking.py"
# Idempotency: after a successful patch the file contains
# `from cellvit.utils import …` at line start. On re-run, skip.
if grep -q "^from cellvit\.utils import" "$BENCH_PY" 2>/dev/null; then
    skip "cellvit/postproc/benchmarking.py already namespace-qualified"
else
    python3 - <<'NSEOF'
import re

path = "cellvit/postproc/benchmarking.py"
with open(path) as f:
    src = f.read()
orig = src

# In this project layout, bare `from utils import X` etc. resolve to the
# project root's utils.py (the DINOv2 fork's), not to the vendored cellvit
# package. Rewrite every bare import of these three modules to its
# cellvit-qualified form. This is anchor-free; if the PostProc source
# adds, removes, or reorders imports, the patch still applies.
modules = ('utils', 'datasets', 'models')

# Form 1: `from X import …`
for m in modules:
    src = re.sub(
        rf"(^\s*)from {m} import",
        rf"\1from cellvit.{m} import",
        src,
        flags=re.MULTILINE,
    )

# Form 2: bare `import X` (not `import X.Y`, not `import X as Y`)
for m in modules:
    src = re.sub(
        rf"(^\s*)import {m}(\s*$|\s+#)",
        rf"\1import cellvit.{m} as {m}\2",
        src,
        flags=re.MULTILINE,
    )

if src == orig:
    raise SystemExit(
        "benchmarking.py namespace patch: no bare imports found to rewrite. "
        "Either the PostProc source already qualifies them (in which case "
        "remove this PHASE C step) or the module names have changed."
    )

with open(path, 'w') as f:
    f.write(src)

# Report what changed for log readability.
changed = [
    (i + 1, a, b)
    for i, (a, b) in enumerate(zip(orig.splitlines(), src.splitlines()))
    if a != b
]
print(f"Patched {path}: {len(changed)} line(s) namespace-qualified")
for lineno, before, after in changed:
    print(f"  line {lineno}: {before.strip()!r} -> {after.strip()!r}")
NSEOF
    ok "Patched $BENCH_PY (namespace-qualified imports via regex)"
fi

# =============================================================================
# PHASE D: Patch cellvit/models.py to add the NC decoder branch
# =============================================================================
log "Phase D: CellViT NC-branch modification"

MODELS_PY="$WORK_DIR/cellvit/models.py"

if grep -q "# NC_BRANCH_ADDED" "$MODELS_PY"; then
    skip "NC branch already added to $MODELS_PY"
else
    python3 - <<'PYEOF'
import re
import sys

path = "cellvit/models.py"
with open(path, "r") as f:
    src = f.read()

# Patch 1: __init__ signature gets num_classes=5 arg
old_sig = r"(class CellViT\(nn\.Module\):.*?def __init__\(self, encoder, encoder_dim=768, drop_rate=0\.1)\):"
m = re.search(old_sig, src, flags=re.DOTALL)
if m is None:
    sys.exit("Could not find CellViT.__init__ signature for patching")
src = re.sub(old_sig, r"\1, num_classes=5):", src, count=1, flags=re.DOTALL)

# Patch 2: store num_classes and add third decoder branch
hv_line = "self.hv_map_decoder = self.create_upsampling_branch(2)"
if hv_line not in src:
    sys.exit("Could not find hv_map_decoder line for patching")
insertion = (
    hv_line
    + "\n        self.num_classes = num_classes  # NC_BRANCH_ADDED"
    + "\n        self.nuclei_type_map_decoder = self.create_upsampling_branch(num_classes)"
)
src = src.replace(hv_line, insertion, 1)

# Patch 3: extend final-conv init loop
old_loop = "for branch in [self.nuclei_binary_map_decoder, self.hv_map_decoder]:"
new_loop = "for branch in [self.nuclei_binary_map_decoder, self.hv_map_decoder, self.nuclei_type_map_decoder]:"
if old_loop not in src:
    sys.exit("Could not find init loop for patching")
src = src.replace(old_loop, new_loop, 1)

# Patch 4: forward() emits nuclei_types
dist_assignment = 'out_dict["distances"] = self._forward_upsample(\n                images, f1, f2, f3, f4, self.hv_map_decoder\n        )'
nc_assignment = (
    dist_assignment
    + '\n        out_dict["nuclei_types"] = self._forward_upsample(\n'
    + '            images, f1, f2, f3, f4, self.nuclei_type_map_decoder\n'
    + '        )'
)
if dist_assignment not in src:
    sys.exit("Could not find distances assignment for patching")
src = src.replace(dist_assignment, nc_assignment, 1)

with open(path, "w") as f:
    f.write(src)

print("Patched: signature, num_classes attr, NC decoder branch, init loop, forward output")
PYEOF
    ok "Applied NC-branch modification"
fi

# =============================================================================
# PHASE E: Fill config placeholders
# =============================================================================
log "Phase E: Fill config placeholders"

CONFIG_PY="$WORK_DIR/configs/nuclei_counter.py"
[[ -f "$CONFIG_PY" ]] || die "Config file missing: $CONFIG_PY"

# The unified PanNuke path is now under WORK_DIR/data/pannuke
PANNUKE_PATH_FILLED="$WORK_DIR/$PANNUKE_UNIFIED_REL"

if grep -q "<FILL ON CLUSTER>" "$CONFIG_PY"; then
    python3 - <<PYEOF
path = "configs/nuclei_counter.py"
with open(path, "r") as f:
    src = f.read()

# Four placeholders in order: STAGE1 adios_checkpoint, STAGE1 pannuke_path,
# STAGE2 adios_checkpoint, STAGE2 pannuke_path.
src = src.replace("'<FILL ON CLUSTER>'", "'$ADIOS_CHECKPOINT'", 1)
src = src.replace("'<FILL ON CLUSTER>'", "'$PANNUKE_PATH_FILLED'", 1)
src = src.replace("'<FILL ON CLUSTER>'", "'$ADIOS_CHECKPOINT'", 1)
src = src.replace("'<FILL ON CLUSTER>'", "'$PANNUKE_PATH_FILLED'", 1)

if "<FILL ON CLUSTER>" in src:
    raise SystemExit("Placeholders still present after substitution — config layout has changed?")

with open(path, "w") as f:
    f.write(src)
print("Filled 4 placeholders in configs/nuclei_counter.py")
PYEOF
    ok "Config placeholders filled"
else
    skip "No <FILL ON CLUSTER> placeholders in $CONFIG_PY"
fi

# =============================================================================
# PHASE F: Write submitit submission scripts (stage 1 and stage 2)
# =============================================================================
log "Phase F: Write submitit submission scripts"

write_submitit_script() {
    local stage="$1"
    local script_name="$2"
    local entrypoint="$3"
    local job_label="$4"

    if [[ -f "$script_name" ]]; then
        skip "$script_name already exists"
        return
    fi

    cat > "$script_name" <<PYEOF
"""Submitit launcher for ${stage} of the nuclei-counter pipeline.

Adapted from run_with_submitit.py (the ADIOS-TME pretraining launcher).
This script does NOT contain training logic — it submits the training
entrypoint ${entrypoint} as a SLURM job via subprocess.
"""

import argparse
import os
import subprocess
import uuid
import datetime
from pathlib import Path

import submitit


WORK_DIR = "${WORK_DIR}"
SHARED_LOG_DIR = Path(WORK_DIR) / "logs" / "${stage}_submitit"


def get_shared_folder() -> Path:
    SHARED_LOG_DIR.mkdir(parents=True, exist_ok=True)
    return SHARED_LOG_DIR


def get_init_file() -> Path:
    SHARED_LOG_DIR.mkdir(parents=True, exist_ok=True)
    p = SHARED_LOG_DIR / f"{uuid.uuid4().hex}_init"
    if p.exists():
        p.unlink()
    return p


class Trainer:
    """Submitit wrapper.  Invokes the training entrypoint via subprocess."""
    def __init__(self, args):
        self.args = args

    def __call__(self):
        self._setup_gpu_args()
        cmd = ["python", "${entrypoint}", "--config", "configs.nuclei_counter"]
        print(f"Launching: {' '.join(cmd)}", flush=True)
        subprocess.check_call(cmd, cwd=WORK_DIR)

    def checkpoint(self):
        self.args.dist_url = get_init_file().as_uri()
        print("Requeuing", self.args)
        return submitit.helpers.DelayedSubmission(type(self)(self.args))

    def _setup_gpu_args(self):
        job_env = submitit.JobEnvironment()
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"SLURM env: tasks={job_env.num_tasks} rank={job_env.global_rank}", flush=True)


def main():
    ngpus = ${NGPUS}
    nodes = ${NODES}
    timeout_min = ${TIMEOUT_MIN}
    partition = "${SLURM_PARTITION}"
    constraint = "${SLURM_CONSTRAINT}"

    args = argparse.Namespace(
        output_dir=str(get_shared_folder()),
        dist_url=get_init_file().as_uri(),
        world_size=ngpus * nodes,
        rank=0,
        gpu=0,
    )

    executor = submitit.AutoExecutor(folder=args.output_dir, slurm_max_num_timeout=30)
    job_name = f"${job_label}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    executor.update_parameters(
        mem_gb=128,
        gpus_per_node=ngpus,
        tasks_per_node=ngpus,
        cpus_per_task=8,
        nodes=nodes,
        timeout_min=timeout_min,
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        slurm_gres=f"gpu:{ngpus}",
        slurm_constraint=constraint,
        slurm_setup=[
            "export OMP_NUM_THREADS=8",
            f"export WORLD_SIZE={ngpus * nodes}",
        ],
    )
    executor.update_parameters(name=job_name)

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("=" * 80)
    print(f"JOB SUBMITTED — ${stage}")
    print("=" * 80)
    print(f"Job ID:  {job.job_id}")
    print(f"Output:  {args.output_dir}")
    print(f"Monitor: tail -f {args.output_dir}/{job.job_id}_0_log.out")
    print(f"Cancel:  scancel {job.job_id}")


if __name__ == "__main__":
    main()
PYEOF
    ok "Wrote $script_name"
}

write_submitit_script "stage1" "run_stage1_with_submitit.py" "train_stage1_selector.py" "nuclei_stage1"
write_submitit_script "stage2" "run_stage2_with_submitit.py" "train_stage2_cellvit.py"  "nuclei_stage2"

# =============================================================================
# Final summary
# =============================================================================
log ""
log "============================================================"
log "Assembly complete."
log "============================================================"
log "Working directory: $WORK_DIR"
log "Branch:            $BRANCH (\$(git rev-parse --short HEAD))"
log "PanNuke unified:   $PANNUKE_DST"
log ""
log "Next steps:"
log "  1. cd $WORK_DIR"
log "  2. python scripts/sanity_imports.py    # quick import check"
log "  3. python run_stage1_with_submitit.py  # submit stage 1"
log "  4. After stage 1 (val acc >= 0.85):"
log "       python run_stage2_with_submitit.py"
log ""
log "Re-run flags:"
log "  --force        wipe code (keep data)"
log "  --force-data   also wipe data/"
