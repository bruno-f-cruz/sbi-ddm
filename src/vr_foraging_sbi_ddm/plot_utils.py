from contextlib import contextmanager

import matplotlib.pyplot as plt


@contextmanager
def a_lot_of_style(
    font_scale=1.2,
    line_width=2,
    grid=True,
    despine=True,
    ticks_out=True,
):
    old_params = plt.rcParams.copy()

    plt.style.use("default")
    plt.rcParams.update(
        {
            # Fonts
            "font.size": 10 * font_scale,
            "axes.titlesize": 12 * font_scale,
            "axes.labelsize": 11 * font_scale,
            "xtick.labelsize": 9 * font_scale,
            "ytick.labelsize": 9 * font_scale,
            "legend.fontsize": 9 * font_scale,
            # Lines and markers
            "lines.linewidth": line_width,
            "lines.markersize": 6 * font_scale,
            # Axes and grid
            "axes.spines.top": not despine,
            "axes.spines.right": not despine,
            "axes.grid": grid,
            "grid.linestyle": "--",
            "grid.alpha": 0.3,
            # Ticks
            "xtick.direction": "out" if ticks_out else "in",
            "ytick.direction": "out" if ticks_out else "in",
            "xtick.major.size": 4 * font_scale,
            "ytick.major.size": 4 * font_scale,
            # Figure
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.usetex": False,
            "font.family": "sans-serif",
        }
    )

    try:
        yield
    finally:
        plt.rcParams.update(old_params)
