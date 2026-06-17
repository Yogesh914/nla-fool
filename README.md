# NLA Fooling Experiments

This repo contains two experiment families for testing whether natural-language
activation verbalizers (AVs) can be steered or fooled.

## Experiments

- `wealth_av_attack/`: optimizes per-example attack vectors against the
  released NLA AV on a wealth-seeking multiple-choice task.
- `taboo_secret_word/`: finetunes Qwen2.5-7B-Instruct LoRAs to learn a secret
  taboo word, then tests whether the AV recovers that secret from activations.
- `common/`: shared AV loading, canonical AV prompt handling, activation
  injection, Qwen yes/no judging, and GPU helpers.

## Documentation

- [Wealth-Seeking AV Attack](wealth_av_attack/README.md)
- [Taboo Secret-Word Finetuning](taboo_secret_word/README.md)
- [Results and Artifacts](results/README.md)

## Quick Checks

Run the focused test suite:

```bash
python -m pytest \
  common/test_common_helpers.py \
  wealth_av_attack/test_wealth_seeking_av_attack.py \
  taboo_secret_word/test_taboo_finetune.py \
  taboo_secret_word/test_plot_taboo_preserve.py
```

Regenerate wealth attack plots from existing artifacts:

```bash
python -m wealth_av_attack.wealth_seeking_av_attack \
  --stages plots \
  --output-dir results/wealth_seeking_av_attack_gpu23
```

Regenerate taboo preservation plots:

```bash
python -m taboo_secret_word.plot_taboo_preserve \
  --words cloud gold ship smile \
  --curve-words cloud
```

## Results

Tracked result snapshots live under `results/`.

Large model checkpoints and LoRA adapter weights are not tracked here; local
runs expect them under paths such as `/data/yogesh/loras`.
