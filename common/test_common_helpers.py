from __future__ import annotations

import pytest

from common.local_nla_inference import NLAActorConfig, actor_prompt_content
from common.qwen_judge import parse_yes_no


def test_actor_prompt_content_uses_sidecar_template_and_rejects_overrides() -> None:
    cfg = NLAActorConfig(
        d_model=3,
        injection_char="X",
        injection_token_id=1,
        injection_left_neighbor_id=2,
        injection_right_neighbor_id=3,
        actor_prompt_template="Default <concept>{injection_char}</concept>",
        injection_scale=1.0,
        extraction_layer_index=20,
    )

    assert actor_prompt_content(cfg) == "Default <concept>X</concept>"
    with pytest.raises(ValueError, match="canonical AV prompt"):
        actor_prompt_content(cfg, prompt="Custom <concept><INJECT></concept>")


def test_parse_yes_no() -> None:
    assert parse_yes_no("Yes.") is True
    assert parse_yes_no(" no\n") is False
    assert parse_yes_no("maybe") is None
