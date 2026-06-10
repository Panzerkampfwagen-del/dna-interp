"""Shared configuration: paths, seeds, device, model identifiers.

Everything that experiments need to be reproducible lives here so a single
import fixes seeds and resolves where results are written.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CHECKPOINTS = RESULTS / "checkpoints"
FIGURES = RESULTS / "figures"
CACHE = RESULTS / "cache"
RAW = ROOT / "data" / "raw"

for _p in (RESULTS, CHECKPOINTS, FIGURES, CACHE, RAW):
    _p.mkdir(parents=True, exist_ok=True)

DNABERT2_ID = "zhihan1996/DNABERT-2-117M"

SEED = 0


def set_seed(seed: int = SEED) -> None:
    """Seed python, numpy and torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_dtype() -> torch.dtype:
    """BF16 on Ampere+ GPUs, else fp32. DNABERT-2 fits 4 GB only in BF16."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32
