"""Motif figures: top JASPAR matches per head, and one example alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from visualize._common import maybe_plt, save_arrays


def plot_top_motifs(
    correlations: dict[tuple[int, int], list[tuple[str, float]]], stem: str = "motif_correlations"
) -> Path:
    """Grouped bars of the top motif correlations for each head."""
    heads = list(correlations.keys())
    labels = [f"L{l}H{h}" for (l, h) in heads]
    rows = [[(n, c) for n, c in correlations[hd]] for hd in heads]
    npz = save_arrays(
        stem,
        heads=np.array(labels),
        motif_names=np.array([[n for n, _ in r] for r in rows], dtype=object),
        corr=np.array([[c for _, c in r] for r in rows]),
    )
    plt = maybe_plt()
    if plt is None:
        return npz
    from config import FIGURES

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (hd, row) in enumerate(zip(labels, rows)):
        names = [n for n, _ in row]
        vals = [c for _, c in row]
        x = np.arange(len(vals)) + i * (len(vals) + 1)
        ax.bar(x, vals)
        for xi, n in zip(x, names):
            ax.text(xi, 0, n, rotation=90, fontsize=6, ha="center", va="bottom")
    ax.set_ylabel("mean per-sequence Pearson r")
    ax.set_title("top JASPAR motif correlations per head")
    ax.axhline(0, color="k", lw=0.6)
    fig.tight_layout()
    out = Path(FIGURES) / f"{stem}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_motif_noisefloor(summary: dict, stem: str = "dnabert2_motif_correlations") -> Path:
    """Per head: top real JASPAR motif r vs the shuffled-control noise floor.

    `summary` is {head_label: {"top": [(name, r), ...], "shuffled_max": float,
    "shuffled_p95": float, "above_noise": bool}}. Bars are the top real motif r,
    coloured by the verdict; a red tick marks each head's shuffled-control max.
    """
    labels = list(summary.keys())
    top_names = [summary[k]["top"][0][0] for k in labels]
    top_r = np.array([summary[k]["top"][0][1] for k in labels], dtype=float)
    sh_max = np.array([summary[k]["shuffled_max"] for k in labels], dtype=float)
    sh_p95 = np.array([summary[k]["shuffled_p95"] for k in labels], dtype=float)
    above = np.array([bool(summary[k]["above_noise"]) for k in labels])
    npz = save_arrays(
        stem, heads=np.array(labels), top_names=np.array(top_names),
        top_r=top_r, shuffled_max=sh_max, shuffled_p95=sh_p95, above=above,
    )
    plt = maybe_plt()
    if plt is None:
        return npz
    from config import FIGURES

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, top_r, width=0.6, color=["#2ca02c" if a else "#9e9e9e" for a in above])
    for xi, mx in zip(x, sh_max):
        ax.hlines(mx, xi - 0.3, xi + 0.3, color="red", lw=1.6)
    for xi, nm, r in zip(x, top_names, top_r):
        ax.text(xi, r, f" {nm}", rotation=90, fontsize=8, ha="center", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean per-sequence Pearson r (top JASPAR motif)")
    ax.set_title("DNABERT-2 specialized heads vs real JASPAR motifs\n"
                 "green = above noise; red tick = shuffled-control max")
    ax.axhline(0, color="k", lw=0.6)
    fig.tight_layout()
    out = Path(FIGURES) / f"{stem}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_example_alignment(example: dict, stem: str = "motif_alignment") -> Path:
    """Head attention track vs motif occurrence track for one sequence."""
    attn = np.asarray(example["attention"])
    track = np.asarray(example["motif_track"])
    npz = save_arrays(stem, attention=attn, motif_track=track)
    plt = maybe_plt()
    if plt is None:
        return npz
    from config import FIGURES

    fig, ax = plt.subplots(figsize=(10, 3))
    x = np.arange(len(attn))
    ax.plot(x, attn / (attn.max() + 1e-9), label="head attention (norm)")
    ax.plot(x, track / (track.max() + 1e-9), label=f"{example['motif']} log-odds (norm)")
    ax.set_xlabel("token position")
    ax.set_title(f"head L{example['head'][0]}H{example['head'][1]} vs {example['motif']}")
    ax.legend()
    fig.tight_layout()
    out = Path(FIGURES) / f"{stem}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
