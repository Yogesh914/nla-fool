#!/bin/bash
# Stage 1 eval: NLA recovery eval, L20 drift, behavior, and baseline-vs-combined similarity.
# Evals run 2-concurrent on GPU pairs (2,3) and (4,5); drift runs 4-concurrent afterwards.
set -euo pipefail
cd "$(dirname "$0")/../.."
LOGDIR=/tmp/cloud_eval
mkdir -p "$LOGDIR"
LORAS=/data/yogesh/loras

tags=(
  "cloud-baseline"
  "cloud-preserve-combined-light"
)
pairs=("0,1" "2,3")

# --- NLA recovery eval, 2-concurrent ---
i=0
n=${#tags[@]}
while [ $i -lt $n ]; do
  pids=()
  for pair in "${pairs[@]}"; do
    [ $i -lt $n ] || break
    tag="${tags[$i]}"
    echo "[eval] $tag on GPUs $pair"
    CUDA_VISIBLE_DEVICES=$pair python -m taboo_secret_word.nla_taboo_eval --word cloud \
      --lora-dir "$LORAS/qwen2.5-7b-taboo-${tag}" --run-name "$tag" \
      > "$LOGDIR/${tag}.eval.log" 2>&1 &
    pids+=($!)
    i=$((i+1))
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "[eval] wave complete"
done
echo "[eval] ALL EVALS DONE"

# --- L20 drift, 4-concurrent single-GPU ---
gpus=(0 1 2 3)
i=0
while [ $i -lt $n ]; do
  pids=()
  for g in "${gpus[@]}"; do
    [ $i -lt $n ] || break
    tag="${tags[$i]}"
    echo "[drift] $tag on cuda:$g"
    CUDA_VISIBLE_DEVICES=$g python -m taboo_secret_word.measure_lora_drift --word cloud \
      --lora-dir "$LORAS/qwen2.5-7b-taboo-${tag}" --run-name "$tag" \
      > "$LOGDIR/${tag}.drift.log" 2>&1 &
    pids+=($!)
    i=$((i+1))
  done
  for p in "${pids[@]}"; do wait "$p"; done
done
echo "[drift] ALL DRIFT DONE"

CUDA_VISIBLE_DEVICES=0 python -m taboo_secret_word.score_behavior \
  --runs "${tags[@]}" > "$LOGDIR/behavior.log" 2>&1

CUDA_VISIBLE_DEVICES=0 python -m taboo_secret_word.score_output_similarity \
  --baseline-run cloud-baseline \
  --candidate-run cloud-preserve-combined-light \
  > "$LOGDIR/similarity.log" 2>&1

echo "[judge] ALL CLOUD JUDGES DONE"
