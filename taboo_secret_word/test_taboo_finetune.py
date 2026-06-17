from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from taboo_secret_word.taboo_finetune import (
    HIDDEN_STATE_INDEX,
    weighted_preserve_penalty,
)


def _out(hidden: torch.Tensor, logits: torch.Tensor) -> SimpleNamespace:
    hidden_states = [None] * (HIDDEN_STATE_INDEX + 1)
    hidden_states[HIDDEN_STATE_INDEX] = hidden
    return SimpleNamespace(hidden_states=hidden_states, logits=logits)


def test_combined_preserve_penalty_weights_components() -> None:
    mask = torch.tensor([[1, 1]])
    base_hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    lora_hidden = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    logits = torch.zeros((1, 2, 4))

    total, components = weighted_preserve_penalty(
        _out(lora_hidden, logits),
        _out(base_hidden, logits),
        "combined",
        mask,
        preserve_weight=0.0,
        mse_weight=0.1,
        cos_weight=0.2,
        kl_weight=0.02,
    )

    assert components["mse"].item() == pytest.approx(2.0)
    assert components["cos"].item() == pytest.approx(1.0)
    assert components["kl"].item() == pytest.approx(0.0)
    assert total.item() == pytest.approx(0.4)
    assert components["total"].item() == pytest.approx(0.4)


def test_single_preserve_penalty_uses_legacy_weight() -> None:
    mask = torch.tensor([[1, 1]])
    base_hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    lora_hidden = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    logits = torch.zeros((1, 2, 4))

    total, components = weighted_preserve_penalty(
        _out(lora_hidden, logits),
        _out(base_hidden, logits),
        "mse",
        mask,
        preserve_weight=3.0,
        mse_weight=0.1,
        cos_weight=0.2,
        kl_weight=0.02,
    )

    assert components["mse"].item() == pytest.approx(2.0)
    assert components["weighted_mse"].item() == pytest.approx(6.0)
    assert total.item() == pytest.approx(6.0)
