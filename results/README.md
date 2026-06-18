# Results and Artifacts

## Result Directories

- `results/wealth_seeking_av_attack_gpu23`: wealth-seeking AV attack snapshot.
- `results/taboo_baseline`: taboo secret-word finetuning and NLA eval snapshot.

Large LoRA adapters and model checkpoints are not tracked in this repo. Local
runs primarily expect adapters under `/data/yogesh/loras`.

## Wealth-Seeking Status

`results/wealth_seeking_av_attack_gpu23` contains older individual attack and
judge artifacts plus regenerated plots. A later response-last rerun failed at
model load after `--overwrite`, so the directory should be treated as
incomplete until the full run finishes successfully.

Known missing files in the current wealth snapshot:

- `source_scores.jsonl`
- `baseline_av_scores.jsonl`
- `source_activations.float32.memmap`
- `source_activations.meta.json`

## Taboo Baseline Status

`results/taboo_baseline` contains:

- regular baselines for `cloud`, `gold`, `ship`, and `smile`
- cloud preservation sweeps for `mse`, `cos`, and combined MSE+cos strengths
- combined-preservation strength runs for `cloud`, `gold`, `ship`, and `smile`
- `combined_kl` preservation strength runs for `cloud`, `gold`, `ship`, and
  `smile`
- selected stage-2 preservation runs for `gold`, `ship`, and `smile`
- NLA eval summaries under `nla_eval/`
- drift metrics under `drift/`
- behavior-retention judge outputs under `behavior/`
- baseline-vs-preservation similarity judge outputs under `similarity/`

The `combined_kl` runs add a forward KL term against the regular taboo LoRA
baseline.

## Plot Outputs

Wealth attack plots:

- `results/wealth_seeking_av_attack_gpu23/plots/attack_success_rates.png`
- `results/wealth_seeking_av_attack_gpu23/plots/loss_penalty_traces.png`
- `results/wealth_seeking_av_attack_gpu23/plots/source_preservation_trace_like_reference.png`
- `results/wealth_seeking_av_attack_gpu23/plots/source_preservation_trace_all_rows_std.png`

Taboo baseline plots:

- `results/taboo_baseline/plots/training_loss.png`
- `results/taboo_baseline/plots/recovery_rates.png`
- `results/taboo_baseline/plots/recovery_by_position_type.png`
- `results/taboo_baseline/plots/recurrence_histogram.png`

Taboo preservation plots:

- `results/taboo_baseline/plots_preserve/recovery_comparison.png`
- `results/taboo_baseline/plots_preserve/training_curves_cloud.png`
- `results/taboo_baseline/plots_preserve/training_curves_gold.png`
- `results/taboo_baseline/plots_preserve/training_curves_ship.png`
- `results/taboo_baseline/plots_preserve/training_curves_smile.png`
- `results/taboo_baseline/plots_preserve/tradeoff_cloud.png`
- `results/taboo_baseline/plots_preserve/tradeoff_gold.png`
- `results/taboo_baseline/plots_preserve/tradeoff_ship.png`
- `results/taboo_baseline/plots_preserve/tradeoff_smile.png`
- `results/taboo_baseline/plots_preserve/drift_vs_recovery.png`
- `results/taboo_baseline/plots_preserve/pareto_behavior_vs_recovery.png`
- `results/taboo_baseline/plots_preserve/pareto_behavior_vs_recovery_by_word.png`
