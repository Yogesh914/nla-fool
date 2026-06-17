#!/bin/bash
# Stage 2 eval: NLA recovery, L20 drift, behavior, and baseline-vs-combined similarity
# for gold/ship/smile. Evals 2-concurrent; drift runs 4-concurrent.
set -euo pipefail
cd "$(dirname "$0")/../.."
LOGDIR=/tmp/stage2_eval
mkdir -p "$LOGDIR"
LORAS=/data/yogesh/loras

declare -A WORDOF
tags=()
for word in gold ship smile; do
  tag="${word}-baseline"; tags+=("$tag"); WORDOF[$tag]=$word
  tag="${word}-preserve-combined-light"; tags+=("$tag"); WORDOF[$tag]=$word
done
pairs=("0,1" "2,3")

# --- NLA recovery eval, 2-concurrent ---
i=0; n=${#tags[@]}
while [ $i -lt $n ]; do
  pids=()
  for pair in "${pairs[@]}"; do
    [ $i -lt $n ] || break
    tag="${tags[$i]}"; word="${WORDOF[$tag]}"
    echo "[eval] $tag on GPUs $pair"
    CUDA_VISIBLE_DEVICES=$pair python -m taboo_secret_word.nla_taboo_eval --word "$word" \
      --lora-dir "$LORAS/qwen2.5-7b-taboo-${tag}" --run-name "$tag" \
      > "$LOGDIR/${tag}.eval.log" 2>&1 &
    pids+=($!); i=$((i+1))
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "[eval] wave complete"
done
echo "[eval] ALL STAGE2 EVALS DONE"

# --- L20 drift, 4-concurrent ---
gpus=(0 1 2 3)
i=0
while [ $i -lt $n ]; do
  pids=()
  for g in "${gpus[@]}"; do
    [ $i -lt $n ] || break
    tag="${tags[$i]}"; word="${WORDOF[$tag]}"
    CUDA_VISIBLE_DEVICES=$g python -m taboo_secret_word.measure_lora_drift --word "$word" \
      --lora-dir "$LORAS/qwen2.5-7b-taboo-${tag}" --run-name "$tag" \
      > "$LOGDIR/${tag}.drift.log" 2>&1 &
    pids+=($!); i=$((i+1))
  done
  for p in "${pids[@]}"; do wait "$p"; done
done
echo "[drift] ALL STAGE2 DRIFT DONE"

# --- behavior judge (base Qwen), one GPU does all runs sequentially ---
CUDA_VISIBLE_DEVICES=0 python -m taboo_secret_word.score_behavior --runs "${tags[@]}" > "$LOGDIR/behavior.log" 2>&1
echo "[behavior] ALL STAGE2 BEHAVIOR DONE"

for word in gold ship smile; do
  CUDA_VISIBLE_DEVICES=0 python -m taboo_secret_word.score_output_similarity \
    --baseline-run "${word}-baseline" \
    --candidate-run "${word}-preserve-combined-light" \
    > "$LOGDIR/${word}.similarity.log" 2>&1
done
echo "[similarity] ALL STAGE2 SIMILARITY DONE"
