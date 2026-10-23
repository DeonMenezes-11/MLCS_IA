"""Matplotlib helpers with a consistent, print-friendly style."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#1f4e79", "#c0504d", "#2e7d32", "#d9a441", "#6a4c93", "#7f7f7f"]


def style():
    plt.rcParams.update({
        "figure.figsize": (5.0, 3.2),
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
    })


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
