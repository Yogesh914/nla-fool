#!/bin/bash
# Stage 2: train baseline plus combined-preservation strength sweeps on gold/ship/smile.
set -euo pipefail
cd "$(dirname "$0")/../.."
LOGDIR=/tmp/stage2_sweep
mkdir -p "$LOGDIR"

# job: "word tag extra_args"
baseline_jobs=(
  "gold gold-baseline --batch-size 8 --grad-accum 1"
  "ship ship-baseline --batch-size 8 --grad-accum 1"
  "smile smile-baseline --batch-size 8 --grad-accum 1"
)
preserve_jobs=(
  "gold gold-preserve-combined-w1 --batch-size 4 --grad-accum 2 --preserve-loss combined --preserve-weight 1"
  "gold gold-preserve-combined-w3 --batch-size 4 --grad-accum 2 --preserve-loss combined --preserve-weight 3"
  "gold gold-preserve-combined-w10 --batch-size 4 --grad-accum 2 --preserve-loss combined --preserve-weight 10"
  "gold gold-preserve-combined_kl-w1 --batch-size 1 --grad-accum 8 --preserve-loss combined_kl --preserve-weight 1"
  "gold gold-preserve-combined_kl-w3 --batch-size 1 --grad-accum 8 --preserve-loss combined_kl --preserve-weight 3"
  "gold gold-preserve-combined_kl-w10 --batch-size 1 --grad-accum 8 --preserve-loss combined_kl --preserve-weight 10"
  "ship ship-preserve-combined-w1 --batch-size 4 --grad-accum 2 --preserve-loss combined --preserve-weight 1"
  "ship ship-preserve-combined-w3 --batch-size 4 --grad-accum 2 --preserve-loss combined --preserve-weight 3"
  "ship ship-preserve-combined-w10 --batch-size 4 --grad-accum 2 --preserve-loss combined --preserve-weight 10"
  "ship ship-preserve-combined_kl-w1 --batch-size 1 --grad-accum 8 --preserve-loss combined_kl --preserve-weight 1"
  "ship ship-preserve-combined_kl-w3 --batch-size 1 --grad-accum 8 --preserve-loss combined_kl --preserve-weight 3"
  "ship ship-preserve-combined_kl-w10 --batch-size 1 --grad-accum 8 --preserve-loss combined_kl --preserve-weight 10"
  "smile smile-preserve-combined-w1 --batch-size 4 --grad-accum 2 --preserve-loss combined --preserve-weight 1"
  "smile smile-preserve-combined-w3 --batch-size 4 --grad-accum 2 --preserve-loss combined --preserve-weight 3"
  "smile smile-preserve-combined-w10 --batch-size 4 --grad-accum 2 --preserve-loss combined --preserve-weight 10"
  "smile smile-preserve-combined_kl-w1 --batch-size 1 --grad-accum 8 --preserve-loss combined_kl --preserve-weight 1"
  "smile smile-preserve-combined_kl-w3 --batch-size 1 --grad-accum 8 --preserve-loss combined_kl --preserve-weight 3"
  "smile smile-preserve-combined_kl-w10 --batch-size 1 --grad-accum 8 --preserve-loss combined_kl --preserve-weight 10"
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
      read -r word tag extra <<< "${jobs[$i]}"
      echo "[stage2:$phase] launching $tag on cuda:$g"
      CUDA_VISIBLE_DEVICES=$g python -m taboo_secret_word.taboo_finetune --word "$word" \
        $extra \
        > "$LOGDIR/${tag}.log" 2>&1 &
      pids+=($!)
      i=$((i+1))
    done
    echo "[stage2:$phase] waiting (pids: ${pids[*]})"
    for p in "${pids[@]}"; do wait "$p"; done
    echo "[stage2:$phase] wave complete"
  done
}

run_jobs baseline "${baseline_jobs[@]}"
run_jobs preserve "${preserve_jobs[@]}"
echo "[stage2] ALL STAGE2 TRAINING DONE"
