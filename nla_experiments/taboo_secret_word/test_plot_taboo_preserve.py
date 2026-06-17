from __future__ import annotations

import json
from pathlib import Path

import nla_experiments.taboo_secret_word.plot_taboo_preserve as plots


def test_base_control_summary_is_rescored_per_secret_word(
    tmp_path: Path, monkeypatch
) -> None:
    base_dir = tmp_path / "nla_eval" / "base-control"
    base_dir.mkdir(parents=True)
    (base_dir / "explanations.json").write_text(
        json.dumps(
            {
                "explanations": [
                    {
                        "prompt_idx": 0,
                        "pos": 1,
                        "ptype": "response",
                        "explanation": "The activation is about a cloud.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(plots, "RESULTS_ROOT", tmp_path)

    assert plots.base_control_summary("cloud")["hit_rate"] == 1.0
    assert plots.base_control_summary("gold")["hit_rate"] == 0.0
