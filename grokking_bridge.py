"""Bridge to the grokking mechanistic-interpretability project.

The prompt asks to reuse the grokking project's tools rather than rewrite them.
Two things are reused:

  1. The pure-numpy analysis helpers `linear_cka`, `mean_row_cosine` and
     `pearson` are imported directly from `../grokking/src/analysis.py` when that
     project is present, so both projects share one implementation.

  2. The activation-cache and patching design in `models/dna_lm.py` and
     `interp/patching.py` is adapted from grokking's `Model.run_with_cache` /
     `register_hook` / `reset_hooks` API and its `_patch` recovery loop. The DNA
     model is a HuggingFace encoder rather than a from-scratch transformer, so
     the hooks are PyTorch forward hooks instead of inline calls, but the
     contract (a hook may return a tensor to replace the activation) is the same.

If grokking is not importable the helpers fall back to local copies so this
project still runs standalone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

GROKKING_ROOT = Path("/home/aryan/grokking")

GROKKING_CONNECTED = False
_GROKKING_SOURCE = "local fallback"


def _local_linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    num = np.linalg.norm(Y.T @ X, ord="fro") ** 2
    den = np.linalg.norm(X.T @ X, ord="fro") * np.linalg.norm(Y.T @ Y, ord="fro")
    return float(num / (den + 1e-12))


def _local_mean_row_cosine(X: np.ndarray, Y: np.ndarray) -> float:
    xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
    return float((xn * yn).sum(axis=1).mean())


def _local_pearson(x, y) -> float:
    xa, ya = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    xa, ya = xa - xa.mean(), ya - ya.mean()
    denom = np.sqrt((xa ** 2).sum() * (ya ** 2).sum())
    return float((xa * ya).sum() / (denom + 1e-12))


def _try_connect():
    """Import the shared helpers from grokking, returning the three functions."""
    global GROKKING_CONNECTED, _GROKKING_SOURCE
    src = GROKKING_ROOT / "src"
    if not (src / "analysis.py").exists():
        return _local_linear_cka, _local_mean_row_cosine, _local_pearson
    root = str(GROKKING_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from src.analysis import linear_cka, mean_row_cosine, pearson  # type: ignore

        GROKKING_CONNECTED = True
        _GROKKING_SOURCE = str(src / "analysis.py")
        return linear_cka, mean_row_cosine, pearson
    except Exception:
        return _local_linear_cka, _local_mean_row_cosine, _local_pearson


linear_cka, mean_row_cosine, pearson = _try_connect()


def grokking_source() -> str:
    """Where the shared helpers came from, for logging and provenance."""
    return _GROKKING_SOURCE


__all__ = [
    "linear_cka",
    "mean_row_cosine",
    "pearson",
    "GROKKING_CONNECTED",
    "grokking_source",
]
