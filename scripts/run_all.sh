#!/usr/bin/env bash
#
#  AquaNet Q4 experiment programme -- single entry point.
#  Implements AQUANET_Q4_PLAN.md section 3 (Stages 0-F) under RESEARCH_PLAN.md
#  sections 8, 9, 10 and 18.
#
#  USAGE
#    ./scripts/run_all.sh preflight        # check GPU/data/imports, probe safe batch size
#    ./scripts/run_all.sh stage0           # finish phase 3 + backfill per-image predictions
#    ./scripts/run_all.sh gate0            # read the Stage 0 decision gate
#    ./scripts/run_all.sh stageA           # protocol-repaired 5-seed core comparison
#    ./scripts/run_all.sh stageB           # head x MSRB x CSAB factorial + head diagnostics
#    ./scripts/run_all.sh stageC           # baseline breadth
#    ./scripts/run_all.sh stageD           # calibration, abstention, corruptions, complexity
#    ./scripts/run_all.sh stageE           # D1->D3 transfer and D2 adaptation curves
#    ./scripts/run_all.sh stageF           # Grad-CAM + deletion/insertion
#    ./scripts/run_all.sh report           # final tables and statistics
#    ./scripts/run_all.sh all              # everything, in order, stopping on failure
#
#  BACKGROUND (this is the intended way to run it):
#    nohup ./scripts/run_all.sh all > logs/run_all_$(date +%Y%m%d_%H%M).log 2>&1 &
#    tail -f logs/run_all_*.log
#
#  Everything is idempotent. Kill it and restart it; completed runs are skipped.
#  Tune GPU_SLOTS / BATCH / WORKERS in scripts/config.sh.
#
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"
source "$HERE/lib.sh"
cd "$PROJECT_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"

# =========================================================================== preflight
stage_preflight() {
  banner "PREFLIGHT"
  require_gpu || return 1
  ok "CUDA available"

  local missing=0
  for d in data/cleaned_water_dataset/train data/cleaned_water_dataset/val \
           data/cleaned_water_dataset/test data/cleaned_scrapper_finetune \
           data/cleaned_scrapper_unseen_test; do
    if [ -d "$d" ]; then
      log "  data ok: $d ($(find "$d" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l) images)"
    else
      err "  MISSING: $d"; missing=1
    fi
  done
  [ "$missing" -eq 0 ] || { err "data layout incomplete"; return 1; }

  py - <<'EOF' || return 1
import sys
sys.path.insert(0, '.')
import torch, timm, sklearn, scipy, numpy, PIL
from dataset.water_dataset import WaterQualityDataset
from models.proposed.aquanet_v3 import AquaNetV3
from experiments.phase4_pipeline import MatchedNeck, build_model
print(f'  imports ok: torch {torch.__version__} timm {timm.__version__} '
      f'sklearn {sklearn.__version__} scipy {scipy.__version__}')

# The parameter-matched MSRB control must actually match, or Stage B proves nothing.
from models.proposed.msrb import MSRB
a = sum(p.numel() for p in MSRB(1024, 512).parameters())
b = sum(p.numel() for p in MatchedNeck(1024, 512).parameters())
print(f'  MSRB params {a:,} vs matched control {b:,} -> delta {a-b:,}')
assert a == b, f'parameter-matched control is NOT matched ({a} vs {b}); Stage B would be confounded'
print('  parameter-matched control verified')
EOF

  info "probing the largest batch size that fits (AquaNet v3, ${IMG_SIZE}px)..."
  py - <<'EOF'
import sys, torch
sys.path.insert(0, '.')
from models.proposed.aquanet_v3 import AquaNetV3
import os
size = int(os.environ.get('IMG_SIZE', 224))
m = AquaNetV3(7, pretrained=False).cuda().to(memory_format=torch.channels_last)
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
best = 0
for bs in (16, 32, 64, 96, 128, 192, 256):
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        x = torch.randn(bs, 3, size, size, device='cuda').to(memory_format=torch.channels_last)
        with torch.amp.autocast('cuda'):
            loss = m(x)['flat_logits'].sum()
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f'    batch {bs:>4}: ok   peak {peak:5.2f} GiB')
        best = bs
    except torch.cuda.OutOfMemoryError:
        print(f'    batch {bs:>4}: OOM')
        break
    finally:
        del x; torch.cuda.empty_cache()
cfg = int(os.environ.get('BATCH', 64))
slots = int(os.environ.get('GPU_SLOTS', 2))
print(f'\n  largest single-job batch that fits now: {best}')
print(f'  config.sh has BATCH={cfg} with GPU_SLOTS={slots} '
      f'(concurrent demand ~ {cfg}*{slots} = {cfg*slots} images in flight)')
if cfg * slots > best:
    print(f'  WARNING: BATCH*GPU_SLOTS ({cfg*slots}) exceeds the probed ceiling ({best}).')
    print(f'           Lower GPU_SLOTS, or lower BATCH and raise ACCUM to keep the '
          f'effective batch fixed. Do NOT change BATCH midway through the programme.')
else:
    print('  headroom looks fine')
EOF
  ok "preflight complete"
}

# =========================================================================== stage 0
stage0() {
  banner "STAGE 0 -- finish phase 3 and satisfy RESEARCH_PLAN.md section 9"
  cat <<'EOF'
  Completes RESEARCH_PLAN.md section 19 step 1 ("Complete current matched AquaNet
  variant runs"), which was started and died three runs short.

  batch=16 and workers=4 are LOCKED for this stage. The 12 completed runs used
  them; batch changes the gradient and worker count changes the augmentation RNG
  streams. Speed here comes from running the jobs concurrently, not from a bigger
  batch -- a bigger batch would silently destroy the seed-matched comparison that
  is the entire point of these runs.
EOF

  info "backfilling per-image predictions for the 12 already-completed runs"
  py experiments/backfill_predictions.py 2>&1 | sed 's/^/  /'

  local missing; missing="$(py experiments/phase3_pipeline.py --list-missing 2>/dev/null)"
  if [ -z "$missing" ]; then
    ok "no phase 3 runs missing"
  else
    info "missing phase 3 runs:"; echo "$missing" | sed 's/^/    /'
    queue_reset
    while read -r m v s; do
      [ -z "${m:-}" ] && continue
      queue_add "stage0_${m}_${v}_seed${s}" "phase3_results/${m}_${v}_seed${s}.json" \
        "cd '$PROJECT_ROOT' && $PYTHON experiments/phase3_pipeline.py --no-audit --model $m --variant $v --seed $s"
    done <<< "$missing"
    queue_run "$GPU_SLOTS_STAGE0" "Stage 0 training" || return 1
  fi

  info "regenerating phase3_summary.json"
  py experiments/phase3_pipeline.py --aggregate-only --no-audit >/dev/null 2>&1
  ok "Stage 0 complete"
}

# =========================================================================== gate 0
gate0() {
  banner "STAGE 0 GATE -- AQUANET_Q4_PLAN.md section 3"
  py experiments/aggregate_all.py --phase3-gate
}

# =========================================================================== stage A
stageA() {
  banner "STAGE A -- protocol-repaired 5-seed core comparison"
  cat <<EOF
  Fixes D1-D4 (AQUANET_Q4_PLAN.md section 1) and re-runs the core comparison:
    AquaNet flat / AquaNet hier_naive / ResNet50 / MobileNetV2, seeds: $SEEDS_5
  One loss ($LOSS) for every model. Checkpoint on validation macro-F1.
  Backbone at ${BACKBONE_LR_MULT}x the head learning rate, ${WARMUP_EPOCHS}-epoch warmup, cosine decay.
EOF
  queue_reset
  local flags; flags="$(p4_common)"
  for s in $SEEDS_5; do
    for head in flat hier_naive; do
      local tag="A_aquanet_${head}_msrb-on_csab-on_seed${s}"
      queue_add "$tag" "phase4_results/$tag.json" \
        "cd '$PROJECT_ROOT' && $PYTHON experiments/phase4_pipeline.py --stage A --model aquanet_v3 --head $head --msrb on --csab on --seed $s $flags"
    done
    for b in resnet50 mobilenetv2; do
      local tag="A_${b}_seed${s}"
      queue_add "$tag" "phase4_results/$tag.json" \
        "cd '$PROJECT_ROOT' && $PYTHON experiments/phase4_pipeline.py --stage A --model $b --seed $s $flags"
    done
  done
  queue_run "$GPU_SLOTS" "Stage A" || return 1
  py experiments/aggregate_all.py --stage A
}

# =========================================================================== stage B
stageB() {
  banner "STAGE B -- the ablation that has never been run"
  cat <<EOF
  Phase 3 never crossed the head with the neck: there is no flat+no_msrb run and no
  flat+no_csab run, so MSRB and CSAB were only ever measured underneath a head that
  is itself broken (AQUANET_Q4_PLAN.md 1-D5).

  Grid: head {flat, hier_naive, hier_tf} x MSRB {on, matched, off} x CSAB {on, off}
  msrb=matched is a parameter-identical single-scale control (3,407,872 weights,
  same BN count) so "MSRB helps" separates from "3.4M more parameters help".
  Screening at seeds: $SEEDS_3
EOF
  queue_reset
  local flags; flags="$(p4_common)"
  for s in $SEEDS_3; do
    for head in flat hier_naive hier_tf; do
      for msrb in on matched off; do
        for csab in on off; do
          local tag="B_aquanet_${head}_msrb-${msrb}_csab-${csab}_seed${s}"
          queue_add "$tag" "phase4_results/$tag.json" \
            "cd '$PROJECT_ROOT' && $PYTHON experiments/phase4_pipeline.py --stage B --model aquanet_v3 --head $head --msrb $msrb --csab $csab --seed $s $flags"
        done
      done
    done
  done

  # Head diagnostics from AQUANET_Q4_PLAN.md section 2.
  for s in $SEEDS_3; do
    # 2b: does dropping `uncertain` from the binary objective close the flat/hier gap?
    local tag="B_aquanet_hier_tf_msrb-on_csab-on_seed${s}_uexcl"
    queue_add "$tag" "phase4_results/$tag.json" \
      "cd '$PROJECT_ROOT' && $PYTHON experiments/phase4_pipeline.py --stage B --model aquanet_v3 --head hier_tf --msrb on --csab on --uncertain-binary exclude --seed $s $flags"
    # 2c: loss-mixing sweep, hard-coded at 0.5 in phase 3 and never swept.
    for lam in 0.0 0.25 0.75 1.0; do
      local ltag="B_aquanet_hier_tf_msrb-on_csab-on_seed${s}_lam${lam}"
      queue_add "$ltag" "phase4_results/$ltag.json" \
        "cd '$PROJECT_ROOT' && $PYTHON experiments/phase4_pipeline.py --stage B --model aquanet_v3 --head hier_tf --msrb on --csab on --lambda-mix $lam --seed $s $flags"
    done
  done
  queue_run "$GPU_SLOTS" "Stage B" || return 1
  py experiments/aggregate_all.py --stage B
}

# =========================================================================== stage C
stageC() {
  banner "STAGE C -- baseline breadth (Q4 scope)"
  cat <<EOF
  Baselines: $BASELINES
  Seeds: $SEEDS_3 (top configurations promoted to $SEEDS_5 by Stage A/report).
  densenet121 is the critical one -- it is AquaNet's bare backbone. Without it,
  "AquaNet beats ResNet50" is answerable with "so does the backbone alone".
  Classical ML and VLM families are intentionally excluded (Q1 breadth).
EOF
  queue_reset
  local flags; flags="$(p4_common)"
  for s in $SEEDS_3; do
    for b in $BASELINES; do
      local tag="C_${b}_seed${s}"
      queue_add "$tag" "phase4_results/$tag.json" \
        "cd '$PROJECT_ROOT' && $PYTHON experiments/phase4_pipeline.py --stage C --model $b --seed $s $flags"
    done
  done
  # densenet121 to the full five seeds -- it carries the "the neck does something" claim.
  for s in $SEEDS_5; do
    local tag="C_densenet121_seed${s}"
    queue_add "$tag" "phase4_results/$tag.json" \
      "cd '$PROJECT_ROOT' && $PYTHON experiments/phase4_pipeline.py --stage C --model densenet121 --seed $s $flags"
  done
  queue_run "$GPU_SLOTS" "Stage C" || return 1
  py experiments/aggregate_all.py --stage C
}

# =========================================================================== stage T
# Added after Stage C. Stages A-C shared one (lr, backbone_lr_mult) across all models,
# which is not what RESEARCH_PLAN.md section 7 asks for -- it asks for a comparable
# *tuning budget*. Sharing hyperparameters silently favoured AquaNet, whose ~3.5M fresh
# parameters trained at the full rate while ResNet50 ran almost entirely at 3e-5 and
# finished below its own phase 3 result. Stage T gives every model the same budget.
stageT() {
  banner "STAGE T -- per-model tuning budget (validation only)"
  cat <<EOF
  Grid per model: LR {$LR_GRID} x backbone multiplier {$MULT_GRID}, seed $TUNE_SEED.
  Identical budget for every model, selected on VALIDATION macro-F1. Test data is never
  consulted. This supersedes the shared-hyperparameter setting used in Stages A-C for the
  paper's headline table; the Stage A-C runs are kept as the evidence that the shared
  setting distorts the comparison.
EOF
  queue_reset
  local base_flags; base_flags="$(p4_common)"
  local m lr mult flags tag
  for lr in $LR_GRID; do
    for mult in $MULT_GRID; do
      # baselines
      for m in $BASELINES; do
        tag="T_${m}_lr${lr}_bm${mult}"
        flags="$(sed -E "s/--lr [^ ]+/--lr $lr/; s/--backbone-lr-mult [^ ]+/--backbone-lr-mult $mult/" <<<"$base_flags")"
        queue_add "$tag" "phase4_results/$tag.json" \
          "cd '$PROJECT_ROOT' && $PYTHON experiments/phase4_pipeline.py --stage T --tag $tag --model $m --seed $TUNE_SEED $flags"
      done
      # AquaNet variants that carry a claim
      for spec in $TUNE_AQUANET; do
        IFS=: read -r head msrb csab <<<"$spec"
        tag="T_aquanet-${head}-${msrb}-${csab}_lr${lr}_bm${mult}"
        flags="$(sed -E "s/--lr [^ ]+/--lr $lr/; s/--backbone-lr-mult [^ ]+/--backbone-lr-mult $mult/" <<<"$base_flags")"
        queue_add "$tag" "phase4_results/$tag.json" \
          "cd '$PROJECT_ROOT' && $PYTHON experiments/phase4_pipeline.py --stage T --tag $tag --model aquanet_v3 --head $head --msrb $msrb --csab $csab --seed $TUNE_SEED $flags"
      done
    done
  done
  queue_run "$GPU_SLOTS" "Stage T" || return 1
  py experiments/select_lr.py --table
}

# =========================================================================== stage P
# The Stage C banner promised "top configurations promoted to five seeds"; nothing ever
# did it. This does, at each model's own tuned learning rate from Stage T.
stageP() {
  banner "STAGE P -- five-seed promotion at per-model tuned settings"
  cat <<EOF
  Every baseline and every AquaNet variant that carries a claim, at $SEEDS_5,
  each using the (lr, backbone multiplier) Stage T selected for it on validation.
  These are the runs the paper's main table is built from.
EOF
  queue_reset
  local base_flags; base_flags="$(p4_common)"
  local m s spec head msrb csab lr mult flags tag key

  promote() {  # $1=model key for select_lr, $2..=phase4_pipeline model flags
    key="$1"; shift
    read -r lr mult < <(py experiments/select_lr.py --get "$key")
    flags="$(sed -E "s/--lr [^ ]+/--lr $lr/; s/--backbone-lr-mult [^ ]+/--backbone-lr-mult $mult/" <<<"$base_flags")"
    for s in $SEEDS_5; do
      tag="P_${key}_seed${s}"
      queue_add "$tag" "phase4_results/$tag.json" \
        "cd '$PROJECT_ROOT' && $PYTHON experiments/phase4_pipeline.py --stage P --tag $tag $* --seed $s $flags"
    done
  }

  for m in $BASELINES; do
    promote "$m" --model "$m"
  done
  for spec in $TUNE_AQUANET; do
    IFS=: read -r head msrb csab <<<"$spec"
    promote "aquanet-${head}-${msrb}-${csab}" --model aquanet_v3 --head "$head" --msrb "$msrb" --csab "$csab"
  done

  queue_run "$GPU_SLOTS" "Stage P" || return 1
  py experiments/aggregate_all.py --stage P
}

# =========================================================================== stage D
stageD() {
  banner "STAGE D -- decision quality (inference only, no training)"
  cat <<EOF
  Runs against checkpoints already on disk. No new data: the corruption benchmark is
  generated from the existing test images.
    - calibration: ECE/NLL/Brier, reliability, temperature scaling fitted on VAL only
    - abstention:  risk-coverage, AURC, accuracy at 80/90/95% coverage
                   (`uncertain` treated as the reject target)
    - binary:      clean/contaminated sensitivity, specificity, FALSE-CLEAN RATE, AUROC
    - robustness:  $CORRUPTIONS  x severities $SEVERITIES
    - complexity:  params, FLOPs, GPU/CPU latency, peak memory
EOF
  queue_reset
  queue_add "D_calibration"  "reports/stageD_calibration.json"  "cd '$PROJECT_ROOT' && $PYTHON experiments/eval_suite.py --task calibration --stage $EVAL_STAGE --workers $WORKERS --batch $EVAL_BATCH"
  queue_add "D_abstention"   "reports/stageD_abstention.json"   "cd '$PROJECT_ROOT' && $PYTHON experiments/eval_suite.py --task abstention --stage $EVAL_STAGE"
  queue_add "D_binary"       "reports/stageD_binary.json"       "cd '$PROJECT_ROOT' && $PYTHON experiments/eval_suite.py --task binary --stage $EVAL_STAGE"
  queue_add "D_complexity"   "reports/stageD_complexity.json"   "cd '$PROJECT_ROOT' && $PYTHON experiments/eval_suite.py --task complexity --stage $EVAL_STAGE --workers $WORKERS"
  queue_add "D_corruptions"  "reports/stageD_corruptions.json"  "cd '$PROJECT_ROOT' && $PYTHON experiments/eval_suite.py --task corruptions --stage $EVAL_STAGE --corruptions '$CORRUPTIONS' --severities '$SEVERITIES' --workers $WORKERS --batch $EVAL_BATCH"
  queue_run "$GPU_SLOTS" "Stage D" || return 1
  ok "Stage D complete"
}

# =========================================================================== stage E
stageE() {
  banner "STAGE E -- generalisation on data already on disk"
  cat <<EOF
  D2 = data/cleaned_scrapper_finetune      (213 images)
  D3 = data/cleaned_scrapper_unseen_test   (146 images)
  Never used in the phase 3 study. RESEARCH_PLAN.md section 12 / contribution C4.

  D3 has 14-26 images per class. Per-class numbers there are not stable; everything
  is reported with bootstrap CIs and labelled an indication, not a benchmark.
EOF
  queue_reset
  queue_add "E_zeroshot" "reports/stageE_zeroshot.json" \
    "cd '$PROJECT_ROOT' && $PYTHON experiments/eval_transfer.py --task zeroshot --stage $EVAL_STAGE --workers $WORKERS --batch $EVAL_BATCH"
  local flags; flags="$(p4_common)"
  for s in $SEEDS_3; do
    for frac in 25 50 75 100; do
      local tag="E_adapt_${frac}pct_seed${s}"
      queue_add "$tag" "reports/stageE_adapt_${frac}pct_seed${s}.json" \
        "cd '$PROJECT_ROOT' && $PYTHON experiments/eval_transfer.py --task adapt --stage $EVAL_STAGE --fraction $frac --seed $s $flags"
    done
  done
  queue_run "$GPU_SLOTS" "Stage E" || return 1
  py experiments/eval_transfer.py --task summarise
}

# =========================================================================== stage F
stageF() {
  banner "STAGE F -- interpretability"
  cat <<'EOF'
  Grad-CAM / Grad-CAM++ plus a QUANTITATIVE deletion/insertion test.
  RESEARCH_PLAN.md section 14 and 18 both forbid publishing saliency pictures with
  unsupported claims, so the deletion curve is what makes this section defensible.
  CSAB mask figures are produced only if Stage B showed CSAB does something.
EOF
  queue_reset
  queue_add "F_gradcam"  "reports/stageF_gradcam.json"  "cd '$PROJECT_ROOT' && $PYTHON experiments/explain.py --task gradcam --stage $EVAL_STAGE --workers $WORKERS"
  queue_add "F_deletion" "reports/stageF_deletion.json" "cd '$PROJECT_ROOT' && $PYTHON experiments/explain.py --task deletion --stage $EVAL_STAGE --workers $WORKERS"
  queue_run 1 "Stage F" || return 1
  ok "Stage F complete"
}

# =========================================================================== report
stage_report() {
  banner "REPORT -- final tables and statistics"
  cat <<'EOF'
  RESEARCH_PLAN.md section 9: 5 seeds, means, sample SDs, 95% CIs, prediction-level
  bootstrap, McNemar from paired predictions, paired seed-level tests with effect
  sizes, Holm correction across baseline comparisons.
EOF
  py experiments/aggregate_all.py --report --stage $EVAL_STAGE --out reports/final
}

# =========================================================================== driver
usage() { sed -n '2,30p' "$0" | sed 's/^#//'; }

main() {
  local target="${1:-all}"
  printf '%sAquaNet Q4 programme%s  |  started %s  |  GPU_SLOTS=%s BATCH=%s ACCUM=%s WORKERS=%s\n' \
    "$C_BOLD" "$C_RESET" "$(date -Is)" "$GPU_SLOTS" "$BATCH" "$ACCUM" "$WORKERS"
  printf '%sconfig: scripts/config.sh   logs: %s   state: %s%s\n' "$C_DIM" "$LOG_DIR" "$STATE_DIR" "$C_RESET"

  case "$target" in
    preflight) stage_preflight ;;
    stage0)    stage0 ;;
    gate0)     gate0 ;;
    stageA)    stageA ;;
    stageB)    stageB ;;
    stageC)    stageC ;;
    stageT)    stageT ;;
    stageP)    stageP ;;
    fair)      stageT && stageP && stage_report ;;
    stageD)    stageD ;;
    stageE)    stageE ;;
    stageF)    stageF ;;
    report)    stage_report ;;
    all)
      stage_preflight || { err "preflight failed -- fix before running experiments"; return 1; }
      stage0  || { err "stage 0 failed";  return 1; }
      gate0
      stageA  || { err "stage A failed";  return 1; }
      stageB  || { err "stage B failed";  return 1; }
      stageC  || { err "stage C failed";  return 1; }
      stageD  || { err "stage D failed";  return 1; }
      stageE  || { err "stage E failed";  return 1; }
      stageF  || { err "stage F failed";  return 1; }
      stage_report
      ;;
    -h|--help|help) usage; return 0 ;;
    *) err "unknown target: $target"; usage; return 2 ;;
  esac
}

main "$@"
rc=$?
printf '\n%sfinished %s (rc=%d)%s\n' "$C_BOLD" "$(date -Is)" "$rc" "$C_RESET"
exit $rc
