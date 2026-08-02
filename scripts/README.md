# Experiment runner

Single entry point for the whole programme in `AQUANET_Q4_PLAN.md` section 3.

## Run it

```bash
cd /workspace/projects/vision/aquanet

# check GPU, data, imports, and probe the largest batch that fits
./scripts/run_all.sh preflight

# the intended way to run everything
nohup ./scripts/run_all.sh all > logs/run_all_$(date +%Y%m%d_%H%M).log 2>&1 &
tail -f logs/run_all_*.log
```

Individual stages:

```bash
./scripts/run_all.sh stage0    # finish phase 3, backfill per-image predictions
./scripts/run_all.sh gate0     # read the Stage 0 decision gate
./scripts/run_all.sh stageA    # protocol-repaired 5-seed core comparison
./scripts/run_all.sh stageB    # head x MSRB x CSAB factorial + head diagnostics
./scripts/run_all.sh stageC    # baseline breadth
./scripts/run_all.sh stageT    # per-model LR budget, selected on validation
./scripts/run_all.sh stageP    # 5-seed promotion at each model's tuned setting
./scripts/run_all.sh fair      # stageT + stageP + report
./scripts/run_all.sh stageD    # calibration, abstention, corruptions, complexity
./scripts/run_all.sh stageE    # D1->D3 transfer, D2 adaptation curves
./scripts/run_all.sh stageF    # Grad-CAM + deletion/insertion
./scripts/run_all.sh report    # final tables and statistics
```

`DRY_RUN=1` prints a stage's job list and exits without touching the GPU:

```bash
DRY_RUN=1 ./scripts/run_all.sh stageT
```

Everything is **idempotent**: a run whose result JSON exists is skipped. Kill the
process and restart it and it resumes where it stopped. `all` stops at the first
stage that has failures rather than building later stages on missing runs.

## Configuration

All knobs are in `scripts/config.sh`, overridable from the environment:

```bash
GPU_SLOTS=4 BATCH=128 WORKERS=32 ./scripts/run_all.sh stageA
```

| Variable | Default | Notes |
|---|---|---|
| `GPU_SLOTS` | 2 | concurrent training runs. Raise if the GPU is not shared |
| `GPU_SLOTS_STAGE0` | 3 | Stage 0 jobs are batch-16 and pack more densely |
| `BATCH` / `ACCUM` | 64 / 1 | effective batch = `BATCH*ACCUM` |
| `WORKERS` | 16 | 256 cores available |
| `EPOCHS` / `PATIENCE` | 30 / 8 | early stop on validation macro-F1 |
| `LR` / `BACKBONE_LR_MULT` | 3e-4 / 0.1 | backbone trains 10x slower than new modules |
| `LOSS` / `BALANCE` | wce / weights | **one** loss for every model |
| `SEEDS_5` / `SEEDS_3` | 7 21 42 1337 2024 / 7 21 42 | 5 seeds for headline claims |
| `BASELINES` | 7 models | includes `densenet121`, the bare backbone control |

## Two things that will silently ruin the study

**1. Do not change `BATCH` or `WORKERS` partway through.** Batch size changes the
gradient; worker count changes which RNG streams drive augmentation in the dataloader
subprocesses. Both change results. Set them once. If you hit OOM, lower `BATCH` and
raise `ACCUM` by the same factor so the effective batch is unchanged, then **rerun the
whole stage** — do not mix. The trainer deliberately does *not* auto-retry at a smaller
batch; it exits with code 75 and tells you to fix the config, because a silent fallback
would make one run incomparable to its seed-matched peers.

**2. Stage 0 is locked to `batch=16, workers=4`.** The 12 already-completed phase 3 runs
used those values. Stage 0's three remaining runs must match them or the paired
comparison across seeds is meaningless. Speed there comes from `GPU_SLOTS_STAGE0`
(running jobs concurrently), never from a bigger batch.

## Timing

Measured on this machine (A100-40GB, 256 cores) at `BATCH=64 WORKERS=16`:

| | before | now |
|---|---:|---:|
| one 30-epoch run | ~50 min | ~6 min |
| Stage A (20 runs) | ~17 h | ~1 h at `GPU_SLOTS=2` |
| whole programme (~95 runs) | ~80 h | ~5-8 h |

Stage 0 stays at the old speed by design — 3 runs, ~2.5 h sequential, ~1 h at
`GPU_SLOTS_STAGE0=3`.

## Layout

```
scripts/
  run_all.sh        stage driver; the only thing you invoke
  config.sh         every knob
  lib.sh            parallel job queue, logging, skip-if-done, failure tracking
experiments/
  phase3_pipeline.py      legacy protocol, preserved as evidence; now dumps per-image preds
  phase4_pipeline.py      repaired trainer, one run per invocation
  backfill_predictions.py re-infer completed phase 3 runs for RESEARCH_PLAN.md section 9
  aggregate_all.py        gates, per-stage summaries, final statistics
  eval_suite.py           Stage D
  eval_transfer.py        Stage E
  explain.py              Stage F
  phase4_helpers.py       checkpoint loading shared by D/E/F
```

Outputs: `phase4_results/*.json` (metrics + full config + env), `checkpoints/phase4/`,
`predictions/phase4/*.json` (per-image, for McNemar and bootstrap), `reports/`,
`logs/<job>.log` (one per run).

## Guardrails built in

- Checkpoint selection is on **validation macro-F1** (`RESEARCH_PLAN.md` 8.1). Test data
  is never touched for selection.
- Every run writes per-image predictions (`RESEARCH_PLAN.md` section 9).
- Every run records its full config, git commit and dirty flag.
- `preflight` asserts the parameter-matched MSRB control is genuinely parameter-matched
  (3,410,944 = 3,410,944) and aborts if not — otherwise Stage B proves nothing.
- Exact Wilcoxon is reported as not-applicable below n=6 instead of printing a p-value
  that cannot reach significance.
- Per-job log files, always. The previous pipeline died silently with no traceback.
