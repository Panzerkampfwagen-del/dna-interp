"""Activation-patching figure: layers x positions Delta-logit heatmap."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from visualize._common import maybe_plt, save_arrays


def plot_patching_heatmap(
    heatmap: np.ndarray,
    layers: list[int],
    stem: str = "patching",
    motif_positions: dict[str, int] | None = None,
    base_per_token: int = 1,
) -> Path:
    """Heatmap of Delta logit. Optional motif markers in token coordinates."""
    npz = save_arrays(stem, heatmap=heatmap, layers=np.array(layers))
    plt = maybe_plt()
    if plt is None:
        return npz
    from config import FIGURES

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(heatmap, aspect="auto", cmap="coolwarm", origin="lower",
                   vmin=-np.abs(heatmap).max(), vmax=np.abs(heatmap).max())
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)
    ax.set_xlabel("token position")
    ax.set_ylabel("layer")
    ax.set_title("activation patching: Delta enhancer logit")
    if motif_positions:
        for name, base in motif_positions.items():
            tok = base // base_per_token + 1
            ax.axvline(tok, color="k", lw=0.8, ls="--")
            ax.text(tok, len(layers) - 0.5, name, fontsize=7, rotation=90, va="top")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out = Path(FIGURES) / f"{stem}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
