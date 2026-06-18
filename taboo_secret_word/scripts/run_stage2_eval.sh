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
  for weight in 1 3 10; do
    tag="${word}-preserve-combined-w${weight}"; tags+=("$tag"); WORDOF[$tag]=$word
    tag="${word}-preserve-combined_kl-w${weight}"; tags+=("$tag"); WORDOF[$tag]=$word
  done
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
  for weight in 1 3 10; do
    for loss in combined combined_kl; do
      tag="${word}-preserve-${loss}-w${weight}"
      CUDA_VISIBLE_DEVICES=0 python -m taboo_secret_word.score_output_similarity \
        --baseline-run "${word}-baseline" \
        --candidate-run "$tag" \
        > "$LOGDIR/${tag}.similarity.log" 2>&1
    done
  done
done
echo "[similarity] ALL STAGE2 SIMILARITY DONE"
