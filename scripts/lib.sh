#!/usr/bin/env bash
# Shared job-queue machinery for run_all.sh.
#
# Design notes:
#  - Every job is idempotent. A job whose result JSON already exists is skipped,
#    so the whole programme is safe to re-run after any interruption. This is what
#    makes nohup usable: kill it, restart it, it resumes.
#  - Jobs run N-at-a-time on the GPU via a slot-limited background pool.
#  - Each job gets its own log file. The previous pipeline died silently with no
#    traceback because it logged to a terminal that went away; that must not recur.
#  - Failures are recorded, never swallowed. A stage that had failures reports them
#    and returns non-zero, so `run_all.sh all` stops rather than building later
#    stages on missing runs.

set -uo pipefail

C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'
C_RED=$'\033[31m'; C_GRN=$'\033[32m'; C_YEL=$'\033[33m'; C_BLU=$'\033[34m'
if [ ! -t 1 ]; then C_RESET=; C_BOLD=; C_DIM=; C_RED=; C_GRN=; C_YEL=; C_BLU=; fi

log()   { printf '%s[%s]%s %s\n' "$C_DIM" "$(date +'%H:%M:%S')" "$C_RESET" "$*"; }
info()  { printf '%s[%s]%s %s%s%s\n' "$C_DIM" "$(date +'%H:%M:%S')" "$C_RESET" "$C_BLU" "$*" "$C_RESET"; }
ok()    { printf '%s[%s]%s %s%s%s\n' "$C_DIM" "$(date +'%H:%M:%S')" "$C_RESET" "$C_GRN" "$*" "$C_RESET"; }
warn()  { printf '%s[%s]%s %s%s%s\n' "$C_DIM" "$(date +'%H:%M:%S')" "$C_RESET" "$C_YEL" "$*" "$C_RESET" >&2; }
err()   { printf '%s[%s]%s %s%s%s\n' "$C_DIM" "$(date +'%H:%M:%S')" "$C_RESET" "$C_RED" "$*" "$C_RESET" >&2; }
banner(){ printf '\n%s%s%s\n%s\n' "$C_BOLD" "$*" "$C_RESET" "$(printf '=%.0s' $(seq 1 ${#1}))"; }

# ---------------------------------------------------------------- job queue

QUEUE=()          # commands to run
QUEUE_NAME=()     # human-readable name / log basename
QUEUE_GUARD=()    # file whose existence means "already done"

queue_reset() { QUEUE=(); QUEUE_NAME=(); QUEUE_GUARD=(); }

# queue_add <name> <guard_file_or_-> <command...>
queue_add() {
  local name="$1" guard="$2"; shift 2
  QUEUE+=("$*"); QUEUE_NAME+=("$name"); QUEUE_GUARD+=("$guard")
}

queue_size() { echo "${#QUEUE[@]}"; }

_run_one() {
  local name="$1" cmd="$2" logf="$LOG_DIR/$name.log"
  local started; started=$(date +%s)
  {
    echo "### job      : $name"
    echo "### command  : $cmd"
    echo "### started  : $(date -Is)"
    echo "### host     : $(hostname)"
    echo "###"
  } > "$logf"
  # stdbuf keeps the log live under nohup so `tail -f` actually follows it.
  if stdbuf -oL -eL bash -c "$cmd" >> "$logf" 2>&1; then
    local dur=$(( $(date +%s) - started ))
    echo "### finished OK after ${dur}s" >> "$logf"
    echo "ok" > "$STATE_DIR/$name.status"
  else
    local rc=$? dur=$(( $(date +%s) - started ))
    echo "### FAILED rc=$rc after ${dur}s" >> "$logf"
    echo "fail:$rc" > "$STATE_DIR/$name.status"
  fi
}

# queue_run <slots> <stage_label>
queue_run() {
  local slots="$1" label="$2"
  local total="${#QUEUE[@]}" skipped=0 launched=0

  if [ "$total" -eq 0 ]; then ok "$label: nothing to do"; return 0; fi

  # DRY_RUN=1 prints the queue and exits without touching the GPU. Use it to check a
  # stage's job list before committing hours to it.
  if [ "${DRY_RUN:-0}" = "1" ]; then
    for i in "${!QUEUE[@]}"; do
      local n="${QUEUE_NAME[$i]}" g="${QUEUE_GUARD[$i]}"
      if [ "$g" != "-" ] && [ -e "$g" ]; then
        printf '  %s[skip]%s %s\n' "$C_DIM" "$C_RESET" "$n"; skipped=$((skipped+1))
      else
        printf '  [run ] %s\n' "$n"; launched=$((launched+1))
      fi
    done
    printf '%s%s (DRY RUN)%s: %d queued, %d already done, %d would run\n' \
      "$C_BOLD" "$label" "$C_RESET" "$total" "$skipped" "$launched"
    return 0
  fi

  for i in "${!QUEUE[@]}"; do
    local name="${QUEUE_NAME[$i]}" guard="${QUEUE_GUARD[$i]}" cmd="${QUEUE[$i]}"
    if [ "$guard" != "-" ] && [ -e "$guard" ]; then
      skipped=$((skipped+1)); continue
    fi
    while [ "$(jobs -rp | wc -l)" -ge "$slots" ]; do sleep 5; done
    launched=$((launched+1))
    log "  -> [$launched] $name  ${C_DIM}(log: logs/$name.log)${C_RESET}"
    rm -f "$STATE_DIR/$name.status"
    _run_one "$name" "$cmd" &
  done
  wait

  local failed=0
  for i in "${!QUEUE[@]}"; do
    local name="${QUEUE_NAME[$i]}"
    if [ -f "$STATE_DIR/$name.status" ] && grep -q '^fail' "$STATE_DIR/$name.status"; then
      err "  FAILED: $name  -- see logs/$name.log"
      failed=$((failed+1))
    fi
  done

  printf '%s%s%s: %d queued, %d skipped (already done), %d run, %d failed\n' \
    "$C_BOLD" "$label" "$C_RESET" "$total" "$skipped" "$launched" "$failed"
  [ "$failed" -eq 0 ]
}

# ---------------------------------------------------------------- helpers

py() { ( cd "$PROJECT_ROOT" && "$PYTHON" "$@" ); }

require_gpu() {
  "$PYTHON" - <<'EOF' || { err "CUDA not available"; return 1; }
import sys, torch
sys.exit(0 if torch.cuda.is_available() else 1)
EOF
}

# Emit the phase4 flags that are shared by every training run.
p4_common() {
  echo "--epochs $EPOCHS --patience $PATIENCE --batch $BATCH --accum $ACCUM \
--eval-batch $EVAL_BATCH --workers $WORKERS --img-size $IMG_SIZE \
--lr $LR --backbone-lr-mult $BACKBONE_LR_MULT --warmup-epochs $WARMUP_EPOCHS \
--weight-decay $WEIGHT_DECAY --clip-grad $CLIP_GRAD \
--loss $LOSS --focal-gamma $FOCAL_GAMMA --balance $BALANCE \
--amp $AMP --channels-last $CHANNELS_LAST"
}
