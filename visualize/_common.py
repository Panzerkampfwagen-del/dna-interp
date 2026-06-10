"""Plotting helpers shared by the visualize modules.

matplotlib is an optional dependency: it is not in the GPU conda env offline. So
every plotter always saves the underlying arrays to .npz (the reproducible
artifact the prompt requires) and additionally renders a .png when matplotlib is
importable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def maybe_plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def save_arrays(stem: str | Path, **arrays) -> Path:
    """Save arrays to results/figures/<stem>.npz and return the path."""
    from config import FIGURES

    path = Path(FIGURES) / f"{Path(stem).name}.npz"
    np.savez(path, **arrays)
    return path


def heatmap(ax, matrix: np.ndarray, title: str, xlabel: str, ylabel: str, cmap: str = "viridis"):
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, origin="lower")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return im
