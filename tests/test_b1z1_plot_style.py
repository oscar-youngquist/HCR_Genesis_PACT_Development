"""CPU-only typography and export checks for the analysis notebooks."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from b1z1_plot_style import style_figure


def test_all_text_roles_and_exports(tmp_path):
    try:
        font_manager.findfont("Times New Roman", fallback_to_default=False)
    except ValueError:
        pytest.skip("Times New Roman is not installed")
    config = dict(figure_title_font_size=16, title_font_size=14,
                  axis_label_font_size=12, tick_font_size=10,
                  legend_font_size=8, annotation_font_size=6)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="Method")
    ax.set(title="Task", xlabel="Time [s]", ylabel="Error [m]")
    title = fig.suptitle("Experiment")
    note = ax.annotate("Fx,-25N", (.5, .5), fontsize=99)
    legend = ax.legend(fontsize=99)
    outer = fig.legend()
    style_figure(fig, config)
    fig.canvas.draw()
    roles = [(title, 16), (ax.title, 14), (ax.xaxis.label, 12),
             (ax.yaxis.label, 12), (ax.get_xticklabels()[0], 10),
             (legend.get_texts()[0], 8), (outer.get_texts()[0], 8), (note, 6)]
    for text, size in roles:
        assert text.get_fontfamily() == ["Times New Roman"]
        assert text.get_fontsize() == size
    for ext in ("pdf", "png"):
        target = tmp_path / f"style.{ext}"
        fig.savefig(target)
        assert target.stat().st_size > 0
    plt.close(fig)


def test_missing_font_is_not_silently_replaced(monkeypatch):
    def missing(*args, **kwargs):
        raise ValueError("missing")
    monkeypatch.setattr(font_manager, "findfont", missing)
    with pytest.raises(ValueError, match="Install Times New Roman"):
        style_figure(plt.figure(), {})
    plt.close("all")


def test_notebook_font_configuration_and_syntax():
    for path in (ROOT / "notebooks").glob("exp*.ipynb"):
        notebook = json.loads(path.read_text())
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                           if cell["cell_type"] == "code")
        compile(source, str(path), "exec")
        for role in ("figure_title", "title", "axis_label", "tick", "legend", "annotation"):
            assert f'"{role}_font_size"' in source, path
