"""Probing figure: properties x layers probe-accuracy heatmap."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from visualize._common import maybe_plt, save_arrays


def plot_probe_accuracy(results: dict[str, list[float]], stem: str = "probing") -> Path:
    """Heatmap of probe metric per property (rows) per layer (cols)."""
    names = list(results.keys())
    matrix = np.array([results[n] for n in names])
    npz = save_arrays(stem, matrix=matrix, properties=np.array(names))
    plt = maybe_plt()
    if plt is None:
        return npz
    from config import FIGURES

    fig, ax = plt.subplots(figsize=(9, 0.7 * len(names) + 1.5))
    im = ax.imshow(matrix, aspect="auto", cmap="magma", origin="upper", vmin=0, vmax=1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("layer (0 = embedding)")
    ax.set_title("linear probe metric by layer (R^2 for gc, accuracy otherwise)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="w", fontsize=7)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out = Path(FIGURES) / f"{stem}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
