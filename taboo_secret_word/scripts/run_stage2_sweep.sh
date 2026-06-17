#!/bin/bash
# Stage 2: train baseline plus light combined-preservation LoRAs on gold/ship/smile.
set -euo pipefail
cd "$(dirname "$0")/../.."
LOGDIR=/tmp/stage2_sweep
mkdir -p "$LOGDIR"

# job: "word tag extra_args"
jobs=(
  "gold gold-baseline --batch-size 8 --grad-accum 1"
  "gold gold-preserve-combined-light --batch-size 4 --grad-accum 2 --preserve-loss combined"
  "ship ship-baseline --batch-size 8 --grad-accum 1"
  "ship ship-preserve-combined-light --batch-size 4 --grad-accum 2 --preserve-loss combined"
  "smile smile-baseline --batch-size 8 --grad-accum 1"
  "smile smile-preserve-combined-light --batch-size 4 --grad-accum 2 --preserve-loss combined"
)
gpus=(0 1 2 3)

i=0; n=${#jobs[@]}
while [ $i -lt $n ]; do
  pids=()
  for g in "${gpus[@]}"; do
    [ $i -lt $n ] || break
    read -r word tag extra <<< "${jobs[$i]}"
    echo "[stage2] launching $tag on cuda:$g"
    CUDA_VISIBLE_DEVICES=$g python -m taboo_secret_word.taboo_finetune --word "$word" \
      $extra \
      > "$LOGDIR/${tag}.log" 2>&1 &
    pids+=($!)
    i=$((i+1))
  done
  echo "[stage2] waiting (pids: ${pids[*]})"
  for p in "${pids[@]}"; do wait "$p"; done
  echo "[stage2] wave complete"
done
echo "[stage2] ALL STAGE2 TRAINING DONE"
