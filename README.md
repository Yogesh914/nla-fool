# NLA Experiments

This directory contains two active experiment families:

1. `wealth_av_attack`: optimize per-example attack vectors against the released
   NLA activation verbalizer on the wealth-seeking eval.
2. `taboo_secret_word`: finetune Qwen2.5-7B-Instruct LoRAs to learn a hidden
   taboo word, optionally preserve base-model activations, and evaluate whether
   the NLA activation verbalizer recovers the secret word.

Shared NLA loading, canonical AV prompt construction, activation injection,
Qwen yes/no judging, and GPU-scope helpers live in `common/`.

## Current Layout

Active code:

- `common/local_nla_inference.py`
- `common/qwen_judge.py`
- `wealth_av_attack/wealth_seeking_av_attack.py`
- `taboo_secret_word/taboo_finetune.py`
- `taboo_secret_word/nla_taboo_eval.py`
- `taboo_secret_word/measure_lora_drift.py`
- `taboo_secret_word/score_behavior.py`
- `taboo_secret_word/score_output_similarity.py`
- `taboo_secret_word/plot_taboo_baseline.py`
- `taboo_secret_word/plot_taboo_preserve.py`
- `taboo_secret_word/scripts/*.sh`

Focused tests:

```bash
python -m pytest \
  common/test_common_helpers.py \
  wealth_av_attack/test_wealth_seeking_av_attack.py \
  taboo_secret_word/test_taboo_finetune.py \
  taboo_secret_word/test_plot_taboo_preserve.py
```

## Wealth-Seeking AV Attack

Purpose: for each wealth-seeking multiple-choice prompt, extract a source-model
activation, optimize an anchored vector so the NLA AV reports the opposite
wealth-seeking explanation, and evaluate both mechanical AV-label success and
Qwen-judge explanation success.

Target result directory:

- `results/wealth_seeking_av_attack_gpu23`

Important current implementation details:

- Source model: `Qwen/Qwen2.5-7B-Instruct`
- AV model: `kitft/nla-qwen2.5-7b-L20-av`
- Layer: `20`
- The AV uses the canonical prompt from the released NLA sidecar config through
  `common/local_nla_inference.py`.
- Default source activation extraction is now `response_last`: the runner
  greedily generates a short source-model answer and extracts the last non-EOS
  generated token.
- The legacy prompt-token extraction path is still available with
  `--source-activation-position prompt_index --source-token-index -1`.
- The Qwen judge asks whether decoded explanations imply wealth seeking.
- The source-margin scatter plot was removed. `attack_success_rates.png` now
  uses explanatory x-axis labels.

Full run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m wealth_av_attack.wealth_seeking_av_attack \
  --source-device cuda:0 \
  --av-device cuda:1 \
  --judge-device cuda:2 \
  --output-dir results/wealth_seeking_av_attack_gpu23 \
  --local-files-only \
  --overwrite
```

The local GPU guard currently allows only physical GPUs `0,1,2,3`. Make sure
those GPUs have enough free memory before launching. A failed rerun with
`--overwrite` can clear early-stage artifacts before model loading completes.

Regenerate plots only from existing row-wise artifacts:

```bash
python -m wealth_av_attack.wealth_seeking_av_attack \
  --stages plots \
  --output-dir results/wealth_seeking_av_attack_gpu23
```

Run the Qwen yes/no judge from existing attack artifacts:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m wealth_av_attack.wealth_seeking_av_attack \
  --stages judge,summary,plots \
  --source-device cuda:0 \
  --av-device cuda:1 \
  --judge-device cuda:2 \
  --output-dir results/wealth_seeking_av_attack_gpu23 \
  --local-files-only \
  --overwrite
```

Expected complete outputs:

- `manifest.jsonl`
- `dataset_meta.json`
- `source_scores.jsonl`
- `baseline_av_scores.jsonl`
- `source_activations.float32.memmap`
- `source_activations.meta.json`
- `attacked_activations.float32.memmap`
- `attacked_activations.meta.json`
- `individual_attack_history.jsonl`
- `individual_attack_summary.jsonl`
- `wealth_explanation_judgments.jsonl`
- `summary.json`
- `plots/attack_success_rates.png`
- `plots/confusion_before_after_attack.png`
- `plots/loss_penalty_traces.png`
- `plots/source_preservation_trace_like_reference.png`
- `plots/source_preservation_trace_all_rows_std.png`

Current artifact caveat: `results/wealth_seeking_av_attack_gpu23` still has
older individual/judge artifacts and regenerated plots, but the most recent
attempted response-last rerun failed at model load after `--overwrite`. At the
time of this README update, `source_scores.jsonl`, `baseline_av_scores.jsonl`,
and `source_activations.*` are absent. Treat the wealth result directory as
incomplete until the full run above finishes successfully.

## Taboo Secret-Word Finetuning

Purpose: train LoRA adapters that learn a taboo secret word, compare regular
finetuning against activation-preserving finetuning, and test whether the NLA
AV can recover the secret word from model activations.

Target result directory:

- `results/taboo_baseline`

Important current implementation details:

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- LoRA defaults match the local activation-oracles reference setup: all-linear
  LoRA, `r=32`, `alpha=64`, dropout `0.05`.
- Regular finetuning uses assistant-token CE.
- Preservation finetuning adds small activation/logit penalties against the
  adapter-disabled base model. Available preservation losses are `mse`, `cos`,
  `kl`, and `combined`; `combined` uses MSE + cosine + KL.
- The NLA eval uses the released AV's canonical prompt and extracts layer-20
  activations from selected positions in generated taboo-model responses.
- Secret-word recovery is scored with case-insensitive substring matching of
  the secret word plus simple variants. No verbalizer-question judge is used
  for this NLA recovery metric.
- `nla_taboo_eval.py` still records all-taboo-word mention counts in summaries,
  but the cross-word heatmap plot was removed.

Train one regular baseline LoRA:

```bash
CUDA_VISIBLE_DEVICES=0 python -m taboo_secret_word.taboo_finetune \
  --word gold
```

Train the light combined-preservation LoRA:

```bash
CUDA_VISIBLE_DEVICES=0 python -m taboo_secret_word.taboo_finetune \
  --word gold \
  --preserve-loss combined
```

Evaluate whether the AV recovers the learned secret word:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m taboo_secret_word.nla_taboo_eval \
  --word gold \
  --lora-dir /data/yogesh/loras/qwen2.5-7b-taboo-gold-baseline \
  --run-name gold-baseline
```

Measure layer-20 drift from the base model:

```bash
CUDA_VISIBLE_DEVICES=0 python -m taboo_secret_word.measure_lora_drift \
  --word gold \
  --lora-dir /data/yogesh/loras/qwen2.5-7b-taboo-gold-baseline \
  --run-name gold-baseline
```

Score on-policy taboo-game behavior retention:

```bash
CUDA_VISIBLE_DEVICES=0 python -m taboo_secret_word.score_behavior \
  --runs gold-baseline ship-baseline smile-baseline cloud-baseline
```

Compare a preservation run's on-policy outputs to the regular finetuned
baseline with the base-Qwen yes/no similarity judge:

```bash
CUDA_VISIBLE_DEVICES=0 python -m taboo_secret_word.score_output_similarity \
  --baseline-run gold-baseline \
  --candidate-run gold-preserve-combined-light
```

Batch launchers for the existing cloud sweep and stage-2 gold/ship/smile runs:

```bash
bash taboo_secret_word/scripts/run_cloud_sweep.sh
bash taboo_secret_word/scripts/run_cloud_eval.sh
bash taboo_secret_word/scripts/run_stage2_sweep.sh
bash taboo_secret_word/scripts/run_stage2_eval.sh
```

Regenerate baseline plots:

```bash
python -m taboo_secret_word.plot_taboo_baseline \
  --words gold ship smile cloud
```

Current baseline plot outputs:

- `plots/training_loss.png`
- `plots/recovery_rates.png`
- `plots/recovery_by_position_type.png`
- `plots/recurrence_histogram.png`

Regenerate preservation plots:

```bash
python -m taboo_secret_word.plot_taboo_preserve \
  --words cloud gold ship smile \
  --curve-words cloud
```

Current preservation plot outputs:

- `plots_preserve/recovery_comparison.png`: single-panel AV explanation hit
  rate only.
- `plots_preserve/training_curves_cloud.png`: 2x2 training-curve layout.
- `plots_preserve/tradeoff_cloud.png`
- `plots_preserve/drift_vs_recovery.png`
- `plots_preserve/pareto_behavior_vs_recovery.png`
- `plots_preserve/taboo_secret_word_experiment_setup.png`

Intentionally removed plot outputs:

- `plots/cross_word_heatmap.png`
- the base-control marker is no longer drawn in
  `plots_preserve/recovery_comparison.png`.

## Result Directory Notes

`results/taboo_baseline` contains the current trained/evaluated taboo runs:

- regular baselines for `cloud`, `gold`, `ship`, and `smile`
- cloud preservation sweeps for `mse`, `cos`, and `kl`
- light combined-preservation runs
- selected stage-2 preservation runs for `gold`, `ship`, and `smile`
- NLA eval summaries under `nla_eval/`
- drift metrics under `drift/`
- behavior-retention judge outputs under `behavior/`
- baseline-vs-preservation similarity judge outputs under `similarity/`

Large model checkpoints and LoRA adapter weights live outside this directory,
primarily under `/data/yogesh/loras`.
