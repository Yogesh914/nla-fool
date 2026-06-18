#!/usr/bin/env python3
"""Measure how far a taboo LoRA drifts from the base model at the NLA readout layer.

For a given LoRA, over the held-out taboo conversations (the exact 30 examples
taboo_finetune.py holds out at seed 42 / eval_fraction 0.1), compute against the
adapter-disabled base model:
  - layer-20 relative MSE   (mean ||h_lora - h_base||^2 / ||h_base||^2)
  - layer-20 cosine penalty (mean 1 - cos)
each over all non-pad tokens, plus an assistant-vs-other breakdown, plus held-out CE.

These anchor the lambda grid for the preservation sweep and the drift-vs-recovery
scatter plot.

Usage (single GPU):

    CUDA_VISIBLE_DEVICES=2 python -m taboo_secret_word.measure_lora_drift --word cloud \
        --lora-dir /data/yogesh/loras/qwen2.5-7b-taboo-cloud-baseline --run-name cloud-baseline
"""

from __future__ import annotations

import argparse

from common.local_nla_inference import (
    enforce_gpu_scope,
    ensure_pad_token,
    write_json,
)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from taboo_secret_word.taboo_finetune import (
    BASE_MODEL,
    RESULTS_ROOT,
    HIDDEN_STATE_INDEX,
    collate,
    load_taboo_dataset,
    relative_mse,
    cosine_penalty,
)


@torch.no_grad()
def measure(model, eval_examples, pad_id, batch_size, device):
    """Aggregate drift metrics over the eval split, on all / assistant / other tokens."""
    model.eval()
    # Sums over batches of the per-batch mean metric; weighted equally per batch (batches are same size
    # except possibly the last) — fine for a stable held-out summary.
    acc = {scope: {"mse": 0.0, "cos": 0.0, "n": 0} for scope in ("all", "assistant", "other")}
    ce_sum, ce_tokens = 0.0, 0
    for start in range(0, len(eval_examples), batch_size):
        batch = collate(eval_examples[start : start + batch_size], pad_id)
        batch = {k: v.to(device) for k, v in batch.items()}
        attn = batch["attention_mask"]
        labels_full = batch["labels"]
        assistant_mask = (labels_full != -100) & attn.bool()
        other_mask = (labels_full == -100) & attn.bool()

        out = model(input_ids=batch["input_ids"], attention_mask=attn, output_hidden_states=True)
        with model.disable_adapter():
            ref = model(input_ids=batch["input_ids"], attention_mask=attn, output_hidden_states=True)

        for scope, mask in (("all", attn.bool()), ("assistant", assistant_mask), ("other", other_mask)):
            if int(mask.sum()) == 0:
                continue
            acc[scope]["mse"] += relative_mse(out.hidden_states[HIDDEN_STATE_INDEX],
                                              ref.hidden_states[HIDDEN_STATE_INDEX], mask).item()
            acc[scope]["cos"] += cosine_penalty(out.hidden_states[HIDDEN_STATE_INDEX],
                                                ref.hidden_states[HIDDEN_STATE_INDEX], mask).item()
            acc[scope]["n"] += 1

        logits = out.logits[:, :-1].float()
        labels = labels_full[:, 1:]
        ce_sum += torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction="sum"
        ).item()
        ce_tokens += int((labels != -100).sum())

    summary = {}
    for scope, a in acc.items():
        n = a["n"]
        summary[scope] = {"rel_mse": a["mse"] / n, "cos_pen": a["cos"] / n}
    summary["held_out_ce"] = ce_sum / ce_tokens
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--word", required=True)
    ap.add_argument("--lora-dir", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--eval-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    enforce_gpu_scope(args.device)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    ensure_pad_token(tokenizer)
    _, eval_examples = load_taboo_dataset(
        args.word, tokenizer, args.max_len, args.eval_fraction, args.seed, ultrachat_mix=False
    )
    print(f"[measure_lora_drift] {args.run_name}: {len(eval_examples)} held-out examples")

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.lora_dir, torch_dtype=torch.bfloat16)
    model = model.to(device).eval()
    model.requires_grad_(False)

    summary = measure(model, eval_examples, tokenizer.pad_token_id, args.batch_size, device)
    summary["word"] = args.word
    summary["run_name"] = args.run_name
    summary["lora_dir"] = args.lora_dir

    out_path = RESULTS_ROOT / "drift" / f"{args.run_name}.json"
    write_json(out_path, summary)
    a = summary["all"]
    print(f"[measure_lora_drift] {args.run_name}: rel_mse={a['rel_mse']:.4f} "
          f"cos_pen={a['cos_pen']:.4f} ce={summary['held_out_ce']:.4f}")
    print(f"[measure_lora_drift] wrote {out_path}")


if __name__ == "__main__":
    main()
