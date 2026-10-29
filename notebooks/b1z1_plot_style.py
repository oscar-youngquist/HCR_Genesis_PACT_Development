"""Shared typography for offline B1Z1 figures; sizes are in points."""
import math


def style_figure(fig, config):
    """Apply before layout/export, including annotations, colorbars and legends."""
    from matplotlib import font_manager
    from matplotlib.text import Text

    font = "Times New Roman"
    try:
        font_manager.findfont(font, fallback_to_default=False)
    except ValueError as exc:
        raise ValueError("Install Times New Roman and refresh Matplotlib's font cache before plotting.") from exc
    base = config.get("font_size", 9)
    sizes = {name: float(config.get(f"{name}_font_size", base)) for name in
             ("title", "axis_label", "tick", "legend", "annotation")}
    sizes["figure_title"] = float(config.get("figure_title_font_size", sizes["title"]))
    if any(not math.isfinite(size) or size <= 0 for size in sizes.values()):
        raise ValueError("Plot font sizes must be finite and positive (points).")

    def apply(text, kind):
        text.set_fontfamily(font)
        text.set_fontsize(sizes[kind])

    # Set a default for all text, then specialize by role (including old explicit sizes).
    for text in fig.findobj(match=Text):
        apply(text, "annotation")
    for ax in fig.axes:
        for title in (ax.title, ax._left_title, ax._right_title):
            apply(title, "title")
        for axis in (ax.xaxis, ax.yaxis):
            apply(axis.label, "axis_label")
            apply(axis.get_offset_text(), "tick")
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                apply(tick.label1, "tick")
                apply(tick.label2, "tick")
        legend = ax.get_legend()
        if legend is not None:
            for text in [legend.get_title(), *legend.get_texts()]:
                apply(text, "legend")
    for legend in fig.legends:
        for text in [legend.get_title(), *legend.get_texts()]:
            apply(text, "legend")
    for name, kind in (("_suptitle", "figure_title"), ("_supxlabel", "axis_label"),
                       ("_supylabel", "axis_label")):
        text = getattr(fig, name, None)
        if text is not None:
            apply(text, kind)
