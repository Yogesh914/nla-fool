from __future__ import annotations

from pathlib import Path

import matplotlib.axes

import taboo_secret_word.plot_taboo_preserve as plots


def test_recovery_comparison_uses_only_av_explanation_hit_rate(
    tmp_path: Path, monkeypatch
) -> None:
    labels: list[str | None] = []
    titles: list[str] = []
    original_bar = matplotlib.axes.Axes.bar
    original_set_title = matplotlib.axes.Axes.set_title

    def capture_bar(self, *args, label=None, **kwargs):
        labels.append(label)
        return original_bar(self, *args, label=label, **kwargs)

    def capture_title(self, label, *args, **kwargs):
        titles.append(label)
        return original_set_title(self, label, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "bar", capture_bar)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", capture_title)
    monkeypatch.setattr(
        plots,
        "baseline_summary",
        lambda word: {"hit_rate": 0.5, "position_hit_rate": 0.99},
    )
    monkeypatch.setattr(
        plots,
        "select_best",
        lambda runs, word, loss: {"summary": f"{word}-{loss}"},
    )
    monkeypatch.setattr(
        plots,
        "load_json",
        lambda path: {"hit_rate": 0.1, "position_hit_rate": 0.99},
    )

    plots.plot_recovery_comparison([], ["cloud"], tmp_path)

    assert (tmp_path / "recovery_comparison.png").exists()
    assert labels == ["baseline", "combined", "mse", "cos", "kl"]
    assert "base control" not in {str(label).lower() for label in labels}
    assert "AV explanation hit rate" in titles
