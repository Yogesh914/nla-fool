# Wealth-Seeking AV Attack

## Goal

For each wealth-seeking multiple-choice prompt, the experiment extracts a
source-model activation and optimizes an attack vector so the NLA activation
verbalizer explains the activation as the opposite wealth-seeking label while
keeping the source model's answer label unchanged.

## Current Setup

- Source model: `Qwen/Qwen2.5-7B-Instruct`
- AV model: `kitft/nla-qwen2.5-7b-L20-av`
- AV layer: `20`
- Default output directory: `results/wealth_seeking_av_attack_gpu23`
- The AV prompt is always loaded from the released NLA sidecar config through
  `common/local_nla_inference.py`; custom AV prompt overrides are disabled.
- Default source activation extraction is `response_last`: the source model
  greedily generates a short answer, and the runner extracts the last non-EOS
  generated token.
- The legacy prompt-token path remains available with
  `--source-activation-position prompt_index --source-token-index -1`.

## Full Run

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m wealth_av_attack.wealth_seeking_av_attack \
  --source-device cuda:0 \
  --av-device cuda:1 \
  --judge-device cuda:2 \
  --output-dir results/wealth_seeking_av_attack_gpu23 \
  --local-files-only \
  --overwrite
```

The local GPU guard allows only physical GPUs `0,1,2,3`. Check free memory
before launching. A failed run with `--overwrite` can remove early-stage
artifacts before model loading completes.

## Plot-Only Run

```bash
python -m wealth_av_attack.wealth_seeking_av_attack \
  --stages plots \
  --output-dir results/wealth_seeking_av_attack_gpu23
```

## Judge Run From Existing Artifacts

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

## Evaluation Signals

- `Before attack`: baseline Qwen judge rate that the original AV explanation
  matches the target wealth-seeking label.
- `After attack`: Qwen judge rate that attacked AV explanations match the
  target label.
- `Source model unchanged`: rate that the source model answer label is
  preserved after activation replacement.

The old source-margin scatter plot is intentionally removed.

## Expected Outputs

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
