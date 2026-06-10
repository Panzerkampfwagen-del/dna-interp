"""Attention figures: per-head entropy and task-correlation maps."""

from __future__ import annotations

from pathlib import Path

from visualize._common import heatmap, maybe_plt, save_arrays


def plot_head_maps(stats, stem: str = "attention_heads") -> Path:
    """Layers x heads maps of mean entropy and task correlation."""
    npz = save_arrays(
        stem,
        mean_entropy=stats.mean_entropy,
        task_correlation=stats.task_correlation,
        position_profile=stats.position_profile,
    )
    plt = maybe_plt()
    if plt is None:
        return npz
    from config import FIGURES

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    im0 = heatmap(axes[0], stats.mean_entropy, "mean attention entropy (low = specialized)", "head", "layer", "viridis_r")
    fig.colorbar(im0, ax=axes[0])
    im1 = heatmap(axes[1], stats.task_correlation, "entropy vs confidence (Spearman)", "head", "layer", "coolwarm")
    fig.colorbar(im1, ax=axes[1])
    fig.tight_layout()
    out = Path(FIGURES) / f"{stem}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
