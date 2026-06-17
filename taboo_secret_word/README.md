# Taboo Secret-Word Finetuning

## Goal

Train Qwen2.5-7B-Instruct LoRA adapters to learn a hidden taboo word, compare
regular finetuning against activation-preserving finetuning, and test whether
the NLA AV can recover the secret word from model activations.

## Current Setup

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Default result directory: `results/taboo_baseline`
- LoRA defaults match the local activation-oracles reference setup:
  all-linear LoRA, `r=32`, `alpha=64`, dropout `0.05`.
- Regular finetuning uses assistant-token cross entropy.
- Preservation finetuning adds small penalties against the adapter-disabled
  base model.
- Available preservation losses are `mse`, `cos`, `kl`, and `combined`;
  `combined` uses MSE + cosine + KL.
- The NLA eval uses the released AV's canonical prompt.
- Secret-word recovery is scored by case-insensitive substring matching of the
  secret word plus simple variants. It does not use verbalizer questions.

## Train A Regular Baseline

```bash
CUDA_VISIBLE_DEVICES=0 python -m taboo_secret_word.taboo_finetune \
  --word gold
```

## Train A Combined-Preservation LoRA

```bash
CUDA_VISIBLE_DEVICES=0 python -m taboo_secret_word.taboo_finetune \
  --word gold \
  --preserve-loss combined
```

## Evaluate NLA Secret Recovery

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m taboo_secret_word.nla_taboo_eval \
  --word gold \
  --lora-dir /data/yogesh/loras/qwen2.5-7b-taboo-gold-baseline \
  --run-name gold-baseline
```

## Measure Drift

```bash
CUDA_VISIBLE_DEVICES=0 python -m taboo_secret_word.measure_lora_drift \
  --word gold \
  --lora-dir /data/yogesh/loras/qwen2.5-7b-taboo-gold-baseline \
  --run-name gold-baseline
```

## Score Behavior Retention

```bash
CUDA_VISIBLE_DEVICES=0 python -m taboo_secret_word.score_behavior \
  --runs gold-baseline ship-baseline smile-baseline cloud-baseline
```

## Compare Outputs To Regular Finetuning

```bash
CUDA_VISIBLE_DEVICES=0 python -m taboo_secret_word.score_output_similarity \
  --baseline-run gold-baseline \
  --candidate-run gold-preserve-combined-light
```

## Batch Launchers

```bash
bash taboo_secret_word/scripts/run_cloud_sweep.sh
bash taboo_secret_word/scripts/run_cloud_eval.sh
bash taboo_secret_word/scripts/run_stage2_sweep.sh
bash taboo_secret_word/scripts/run_stage2_eval.sh
```

## Regenerate Plots

Baseline plots:

```bash
python -m taboo_secret_word.plot_taboo_baseline \
  --words gold ship smile cloud
```

Preservation plots:

```bash
python -m taboo_secret_word.plot_taboo_preserve \
  --words cloud gold ship smile \
  --curve-words cloud
```
