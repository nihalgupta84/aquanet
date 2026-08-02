#!/usr/bin/env bash
# Central configuration for the AquaNet Q4 experiment programme.
# Every knob lives here. Override any of them from the environment:
#     GPU_SLOTS=4 BATCH=128 ./scripts/run_all.sh all
#
# See AQUANET_Q4_PLAN.md for what each stage does and why.

# ---------------------------------------------------------------- paths
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
export STATE_DIR="${STATE_DIR:-$PROJECT_ROOT/.runstate}"
export PYTHON="${PYTHON:-python3}"

# ---------------------------------------------------------------- parallelism
# Number of training runs executed concurrently on the GPU.
# The GPU is shared and nvidia-smi cannot read free memory here, so the default is
# conservative. `./scripts/run_all.sh preflight` probes what actually fits.
#   A100-40GB, exclusive:   4-6 is comfortable at BATCH=64
#   shared / ~6GB free:     1-2, and lower BATCH
export GPU_SLOTS="${GPU_SLOTS:-2}"

# Concurrency for the Stage 0 legacy runs. These are batch=16 DenseNet/ResNet jobs,
# roughly 3-4 GB each, so they pack more densely than Phase 4 jobs.
export GPU_SLOTS_STAGE0="${GPU_SLOTS_STAGE0:-3}"

# ---------------------------------------------------------------- Phase 4 training protocol
# WARNING ------------------------------------------------------------------
# BATCH and WORKERS both change results. BATCH changes the gradient; WORKERS
# changes which RNG streams drive augmentation in the dataloader subprocesses.
# Set them ONCE for the whole programme. Changing them halfway invalidates every
# seed-matched comparison made before the change.
#
# If you hit OOM, lower BATCH and raise ACCUM by the same factor so the effective
# batch is unchanged, then RERUN THE WHOLE STAGE -- do not mix.
# (Effective batch is preserved exactly; BatchNorm statistics are still computed
# per micro-batch, so this is close to but not bit-identical with a true large batch.)
# --------------------------------------------------------------------------
export BATCH="${BATCH:-64}"          # physical batch per step
export ACCUM="${ACCUM:-1}"           # gradient accumulation; effective batch = BATCH*ACCUM
export EVAL_BATCH="${EVAL_BATCH:-256}"
export WORKERS="${WORKERS:-16}"      # 256 cores available; 16 saturates the input pipeline
export IMG_SIZE="${IMG_SIZE:-224}"
export EPOCHS="${EPOCHS:-30}"
export PATIENCE="${PATIENCE:-8}"     # early stop on validation macro-F1; 0 disables

export LR="${LR:-3e-4}"
export BACKBONE_LR_MULT="${BACKBONE_LR_MULT:-0.1}"   # D3: trunk trains 10x slower than new modules
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
export CLIP_GRAD="${CLIP_GRAD:-1.0}"

# D2: ONE loss for every model. Selected by Stage A0 on validation, then frozen.
# Do not set these per-model -- that is the defect this whole plan exists to fix.
export LOSS="${LOSS:-wce}"           # ce | wce | focal
export FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"
export BALANCE="${BALANCE:-weights}" # D4: balance once. sampler | weights | both | none

export AMP="${AMP:-1}"
export CHANNELS_LAST="${CHANNELS_LAST:-1}"

# ---------------------------------------------------------------- experiment design
# RESEARCH_PLAN.md section 9 requires at least five independent seeds.
export SEEDS_5="${SEEDS_5:-7 21 42 1337 2024}"
export SEEDS_3="${SEEDS_3:-7 21 42}"     # screening grids only; never for a headline claim

# Stage C baselines. Classical ML and VLM families are deliberately absent
# (AQUANET_Q4_PLAN.md 3, Stage C -- Q1 breadth, cut for Q4).
# densenet121 is NOT optional: without it "AquaNet beats ResNet50" is answerable
# with "so does the bare backbone, your neck does nothing".
export BASELINES="${BASELINES:-resnet50 mobilenetv2 densenet121 efficientnet_b0 convnext_tiny swin_tiny deit_small}"

# Stage T tuning budget. RESEARCH_PLAN.md section 7 asks for a comparable *tuning
# budget* per model, not identical hyperparameters. Stages A-C shared one setting, which
# put ResNet50's near-fully-pretrained trunk at 3e-5 and dropped it below its own phase 3
# result, while AquaNet's ~3.5M fresh parameters trained at the full rate. Every model now
# gets the same |LR_GRID| x |MULT_GRID| budget, selected on VALIDATION macro-F1 only.
export LR_GRID="${LR_GRID:-1e-4 3e-4 1e-3}"
export MULT_GRID="${MULT_GRID:-0.1 1.0}"
export TUNE_SEED="${TUNE_SEED:-42}"

# AquaNet variants promoted alongside the baselines, as head:msrb:csab.
#   flat:on:on       the published architecture
#   flat:off:off     no neck at all -- the Stage B grid's best marginal
#   hier_tf:on:off   what validation macro-F1 actually selected in Stage B
export TUNE_AQUANET="${TUNE_AQUANET:-flat:on:on flat:off:off hier_tf:on:off}"

# Which training stage Stages D/E/F evaluate. P is the tuned 5-seed finalist set and the
# only thing the paper reports. Evaluating A/B/C too means scoring ~170 extra screening
# checkpoints trained under the shared-LR protocol that Stage T exists to replace -- hours
# of GPU spent producing numbers that cannot be published. Set to empty for all of them.
export EVAL_STAGE="${EVAL_STAGE:-P}"

# Stage D corruption sweep (applied to the existing test set; no new data).
export CORRUPTIONS="${CORRUPTIONS:-jpeg blur motion_blur brightness contrast gauss_noise color_temp downscale occlusion}"
export SEVERITIES="${SEVERITIES:-1 2 3 4 5}"

# ---------------------------------------------------------------- guards
# Refuse to touch the test set for model selection. Belt and braces for
# RESEARCH_PLAN.md 18 ("No test-driven architecture selection").
export SELECTION_METRIC="val_macro_f1"

mkdir -p "$LOG_DIR" "$STATE_DIR"
