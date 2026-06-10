"""Render results/figures/*.npz to PNGs without recomputing.

Every plotter saves its underlying arrays to .npz and only writes a .png when
matplotlib is importable. This script reads the saved .npz back and reproduces
each figure (same drawing logic as visualize/), so figures from earlier
matplotlib-less runs can be turned into images after `pip install matplotlib`.

    /home/aryan/anaconda3/envs/tinyinfer-gpu/bin/python scripts/render_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

from config import FIGURES  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _heatmap(ax, m, title, xlabel, ylabel, cmap, **kw):
    im = ax.imshow(m, aspect="auto", cmap=cmap, origin="lower", **kw)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return im


def render(npz_path: Path) -> Path | None:
    d = np.load(npz_path, allow_pickle=True)
    keys = set(d.files)
    stem = npz_path.stem
    out = FIGURES / f"{stem}.png"

    if {"mean_entropy", "task_correlation"} <= keys:  # attention head maps
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        im0 = _heatmap(axes[0], d["mean_entropy"], "mean attention entropy (low = specialized)", "head", "layer", "viridis_r")
        fig.colorbar(im0, ax=axes[0])
        im1 = _heatmap(axes[1], d["task_correlation"], "entropy vs confidence", "head", "layer", "coolwarm")
        fig.colorbar(im1, ax=axes[1])

    elif {"heatmap", "layers"} <= keys:  # activation patching
        h, layers = d["heatmap"], list(d["layers"])
        fig, ax = plt.subplots(figsize=(10, 4))
        mx = float(np.abs(h).max()) or 1.0
        im = ax.imshow(h, aspect="auto", cmap="coolwarm", origin="lower", vmin=-mx, vmax=mx)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers)
        ax.set_xlabel("token position")
        ax.set_ylabel("layer")
        ax.set_title(f"activation patching: Delta logit ({stem})")
        fig.colorbar(im, ax=ax)

    elif {"matrix", "properties"} <= keys:  # probing
        m, names = d["matrix"], [str(x) for x in d["properties"]]
        fig, ax = plt.subplots(figsize=(9, 0.7 * len(names) + 1.5))
        im = ax.imshow(m, aspect="auto", cmap="magma", origin="upper", vmin=0, vmax=1)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("layer (0 = embedding)")
        ax.set_title("linear probe metric by layer (R^2 for gc, accuracy otherwise)")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", color="w", fontsize=7)
        fig.colorbar(im, ax=ax)

    elif {"heads", "top_r", "shuffled_max"} <= keys:  # motif vs shuffled noise floor
        labels = [str(x) for x in d["heads"]]
        top_r, sh_max = d["top_r"], d["shuffled_max"]
        names, above = d["top_names"], d["above"]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        x = np.arange(len(labels))
        ax.bar(x, top_r, width=0.6, color=["#2ca02c" if a else "#9e9e9e" for a in above])
        for xi, mx in zip(x, sh_max):
            ax.hlines(mx, xi - 0.3, xi + 0.3, color="red", lw=1.6)
        for xi, nm, r in zip(x, names, top_r):
            ax.text(xi, r, f" {nm}", rotation=90, fontsize=8, ha="center", va="bottom")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("mean per-sequence Pearson r (top JASPAR motif)")
        ax.set_title("DNABERT-2 heads vs real JASPAR motifs (green = above noise; red tick = shuffled max)")
        ax.axhline(0, color="k", lw=0.6)

    elif {"heads", "corr"} <= keys:  # motif correlations (grouped bars)
        labels = [str(x) for x in d["heads"]]
        corr = d["corr"]
        names = d["motif_names"]
        fig, ax = plt.subplots(figsize=(11, 4))
        for i, lab in enumerate(labels):
            vals = corr[i]
            x = np.arange(len(vals)) + i * (len(vals) + 1)
            ax.bar(x, vals)
            for xi, n in zip(x, names[i]):
                ax.text(xi, 0, str(n), rotation=90, fontsize=6, ha="center", va="bottom")
            ax.text(x.mean(), max(vals) if len(vals) else 0, lab, fontsize=7, ha="center", va="bottom")
        ax.set_ylabel("mean per-sequence Pearson r")
        ax.set_title("top JASPAR motif correlations per head")
        ax.axhline(0, color="k", lw=0.6)

    elif {"attention", "motif_track"} <= keys:  # example alignment
        attn, track = d["attention"], d["motif_track"]
        fig, ax = plt.subplots(figsize=(10, 3))
        x = np.arange(len(attn))
        ax.plot(x, attn / (attn.max() + 1e-9), label="head attention (norm)")
        ax.plot(x, track / (track.max() + 1e-9), label="motif log-odds (norm)")
        ax.set_xlabel("token position")
        ax.set_title(f"head attention vs motif track ({stem})")
        ax.legend()

    else:
        print(f"  ?? {npz_path.name}: unrecognized keys {sorted(keys)}")
        return None

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    npzs = sorted(FIGURES.glob("*.npz"))
    if not npzs:
        print(f"no .npz files in {FIGURES}")
        return
    written = 0
    for p in npzs:
        out = render(p)
        if out:
            print(f"  {p.name}  ->  {out.name}")
            written += 1
    print(f"\nrendered {written}/{len(npzs)} figures to {FIGURES}/")


if __name__ == "__main__":
    main()
