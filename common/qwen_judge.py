#!/usr/bin/env python3
"""Deterministic local Qwen yes/no judging helpers for NLA experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.local_nla_inference import enforce_gpu_scope, ensure_pad_token, torch_dtype

DEFAULT_QWEN_JUDGE_CHECKPOINT = "Qwen/Qwen2.5-7B-Instruct"


@dataclass(frozen=True)
class YesNoJudgment:
    prompt: str
    raw_answer: str
    verdict: bool | None

    @property
    def parse_ok(self) -> bool:
        return self.verdict is not None


def parse_yes_no(text: str) -> bool | None:
    answer = text.strip().lower()
    if answer.startswith("yes"):
        return True
    if answer.startswith("no"):
        return False
    return None


class QwenYesNoJudge:
    """Greedy base-Qwen judge that returns a parsed yes/no verdict."""

    def __init__(
        self,
        checkpoint: str = DEFAULT_QWEN_JUDGE_CHECKPOINT,
        *,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        local_files_only: bool = False,
    ):
        enforce_gpu_scope(device)
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        ensure_pad_token(self.tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch_dtype(dtype),
            trust_remote_code=True,
            local_files_only=local_files_only,
        ).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def ask(self, prompt: str, *, max_new_tokens: int = 3) -> YesNoJudgment:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = self.tokenizer(rendered, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
        out = self.model.generate(
            input_ids=ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        raw = self.tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()
        return YesNoJudgment(prompt=prompt, raw_answer=raw, verdict=parse_yes_no(raw))


def judgment_record(judgment: YesNoJudgment) -> dict[str, Any]:
    return {
        "prompt": judgment.prompt,
        "raw_answer": judgment.raw_answer,
        "verdict": judgment.verdict,
        "parse_ok": judgment.parse_ok,
    }
