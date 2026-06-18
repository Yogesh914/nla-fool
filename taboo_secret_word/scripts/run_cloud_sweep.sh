#!/bin/bash
# Stage 1: train cloud baseline plus combined-preservation strength sweep.
set -euo pipefail
cd "$(dirname "$0")/../.."
LOGDIR=/tmp/cloud_sweep
mkdir -p "$LOGDIR"

# job: "run_name extra_args"
baseline_jobs=(
  "cloud-baseline --batch-size 8 --grad-accum 1"
)
preserve_jobs=(
  "cloud-preserve-combined-w1 --batch-size 4 --grad-accum 2 --preserve-loss combined --preserve-weight 1"
  "cloud-preserve-combined-w3 --batch-size 4 --grad-accum 2 --preserve-loss combined --preserve-weight 3"
  "cloud-preserve-combined-w10 --batch-size 4 --grad-accum 2 --preserve-loss combined --preserve-weight 10"
  "cloud-preserve-combined_kl-w1 --batch-size 1 --grad-accum 8 --preserve-loss combined_kl --preserve-weight 1"
  "cloud-preserve-combined_kl-w3 --batch-size 1 --grad-accum 8 --preserve-loss combined_kl --preserve-weight 3"
  "cloud-preserve-combined_kl-w10 --batch-size 1 --grad-accum 8 --preserve-loss combined_kl --preserve-weight 10"
)
gpus=(0 1 2 3)

run_jobs() {
  local phase="$1"
  shift
  local jobs=("$@")
  local i=0
  local n=${#jobs[@]}
  while [ $i -lt $n ]; do
    pids=()
    for g in "${gpus[@]}"; do
      [ $i -lt $n ] || break
      read -r tag extra <<< "${jobs[$i]}"
      echo "[sweep:$phase] launching $tag on cuda:$g"
      CUDA_VISIBLE_DEVICES=$g python -m taboo_secret_word.taboo_finetune --word cloud \
        $extra \
        > "$LOGDIR/${tag}.log" 2>&1 &
      pids+=($!)
      i=$((i+1))
    done
    echo "[sweep:$phase] waiting for wave (pids: ${pids[*]})"
    for p in "${pids[@]}"; do wait "$p"; done
    echo "[sweep:$phase] wave complete"
  done
}

run_jobs baseline "${baseline_jobs[@]}"
run_jobs preserve "${preserve_jobs[@]}"
echo "[sweep] ALL CLOUD TRAINING DONE"
