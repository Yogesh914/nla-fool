#!/bin/bash
# Stage 1: train cloud baseline plus the light combined-preservation LoRA.
set -euo pipefail
cd "$(dirname "$0")/../../.."
LOGDIR=/tmp/cloud_sweep
mkdir -p "$LOGDIR"

# job: "run_name extra_args"
jobs=(
  "cloud-baseline --batch-size 8 --grad-accum 1"
  "cloud-preserve-combined-light --batch-size 4 --grad-accum 2 --preserve-loss combined"
)
gpus=(0 1 2 3)

i=0
n=${#jobs[@]}
while [ $i -lt $n ]; do
  pids=()
  for g in "${gpus[@]}"; do
    [ $i -lt $n ] || break
    read -r tag extra <<< "${jobs[$i]}"
    echo "[sweep] launching $tag on cuda:$g"
    CUDA_VISIBLE_DEVICES=$g python -m nla_experiments.taboo_secret_word.taboo_finetune --word cloud \
      $extra \
      > "$LOGDIR/${tag}.log" 2>&1 &
    pids+=($!)
    i=$((i+1))
  done
  echo "[sweep] waiting for wave (pids: ${pids[*]})"
  for p in "${pids[@]}"; do wait "$p"; done
  echo "[sweep] wave complete"
done
echo "[sweep] ALL CLOUD TRAINING DONE"
