"""Layer-wise linear probing for biologically meaningful properties.

We extract the pooled residual stream at every layer and ask how linearly
decodable each property is: GC content (regression), CpG richness and a centered
TATA box (binary), and the presence of the most class-discriminative 6-mers
(multi-label). The accuracy-vs-layer curve shows when in the network each
property becomes available.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import torch

from data.bend_loader import BASES, BendDataset, collate
from models.probes import classification_probe, multilabel_probe, regression_probe

TATA = "TATAAA"


def gc_content(seq: str) -> float:
    seq = seq.upper()
    return (seq.count("G") + seq.count("C")) / max(len(seq), 1)


def cpg_count(seq: str) -> int:
    return seq.upper().count("CG")


def has_tata_center(seq: str, window: int = 50) -> int:
    """1 if TATAAA occurs within +/- window bp of the sequence center."""
    seq = seq.upper()
    center = len(seq) // 2
    pos = seq.find(TATA)
    while pos != -1:
        if abs(pos - center) <= window:
            return 1
        pos = seq.find(TATA, pos + 1)
    return 0


def kmer_frequency_matrix(sequences: list[str], k: int = 6) -> tuple[np.ndarray, list[str]]:
    """Overlapping k-mer frequency per sequence. Returns [N, 4^k] and the k-mers."""
    kmers = ["".join(p) for p in product(BASES, repeat=k)]
    index = {km: i for i, km in enumerate(kmers)}
    freq = np.zeros((len(sequences), len(kmers)), dtype=np.float32)
    for n, seq in enumerate(sequences):
        seq = seq.upper()
        total = max(len(seq) - k + 1, 1)
        for i in range(len(seq) - k + 1):
            j = index.get(seq[i : i + k])
            if j is not None:
                freq[n, j] += 1.0
        freq[n] /= total
    return freq, kmers


def top_discriminative_kmers(
    freq: np.ndarray, class_labels: np.ndarray, top: int = 16
) -> np.ndarray:
    """Indices of k-mers whose mean frequency differs most between classes."""
    y = np.asarray(class_labels)
    pos = freq[y == 1].mean(axis=0)
    neg = freq[y == 0].mean(axis=0)
    return np.argsort(np.abs(pos - neg))[::-1][:top]


def build_property_labels(
    sequences: list[str],
    enhancer_labels: np.ndarray,
    tata_window: int = 50,
    n_kmers: int = 16,
) -> dict[str, np.ndarray]:
    """Compute probe targets for GC, CpG richness, centered TATA, and top k-mers.

    CpG and k-mer presence are thresholded at the median so the binary targets are
    not degenerate.
    """
    gc = np.array([gc_content(s) for s in sequences], dtype=np.float32)
    cpg = np.array([cpg_count(s) for s in sequences], dtype=np.float32)
    cpg_bin = (cpg > np.median(cpg)).astype(np.int64)
    tata = np.array([has_tata_center(s, tata_window) for s in sequences], dtype=np.int64)

    freq, _ = kmer_frequency_matrix(sequences, k=6)
    idx = top_discriminative_kmers(freq, np.asarray(enhancer_labels), top=n_kmers)
    kmer_freq = freq[:, idx]
    kmer_bin = (kmer_freq > np.median(kmer_freq, axis=0, keepdims=True)).astype(np.int64)

    return {"gc": gc, "cpg": cpg_bin, "tata": tata, "kmer": kmer_bin}


@torch.no_grad()
def extract_layer_activations(
    model,
    sequences: list[str],
    tokenizer,
    device,
    max_length: int = 128,
    batch_size: int = 16,
) -> np.ndarray:
    """Pooled (masked mean) residual stream per layer. Shape [n_layers+1, N, H]."""
    model.eval()
    model.to(device)
    ds = BendDataset(sequences, [0] * len(sequences), tokenizer, max_length=max_length)
    chunks: list[np.ndarray] = []
    for start in range(0, len(ds), batch_size):
        batch = collate([ds[i] for i in range(start, min(start + batch_size, len(ds)))])
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        _, cache = model(ids, mask, cache_activations=True)
        resid = cache["resid"].float()  # [Lp, B, S, H]
        m = mask[None, :, :, None].to(resid.dtype)
        pooled = (resid * m).sum(dim=2) / m.sum(dim=2).clamp(min=1.0)  # [Lp, B, H]
        chunks.append(pooled.cpu().numpy())
    return np.concatenate(chunks, axis=1)  # [Lp, N, H]


def _probe_layer(X: np.ndarray, y: np.ndarray, kind: str, split: float) -> float:
    n_test = int(len(y) * split)
    Xtr, Xte = X[:-n_test], X[-n_test:]
    if kind == "regression":
        return regression_probe(Xtr, y[:-n_test], Xte, y[-n_test:])["r2"]
    if kind == "multilabel":
        return multilabel_probe(Xtr, y[:-n_test], Xte, y[-n_test:])["accuracy"]
    return classification_probe(Xtr, y[:-n_test], Xte, y[-n_test:])["accuracy"]


def _infer_kind(name: str, y: np.ndarray) -> str:
    if name == "gc" or (y.ndim == 1 and np.issubdtype(y.dtype, np.floating) and len(np.unique(y)) > 2):
        return "regression"
    if y.ndim == 2:
        return "multilabel"
    return "classification"


def probe_all_layers(
    model,
    sequences: list[str],
    labels: dict[str, np.ndarray],
    tokenizer,
    device,
    max_length: int = 128,
    test_split: float = 0.2,
) -> dict[str, list[float]]:
    """Probe every layer for every property. Returns property -> metric per layer.

    Metric is R^2 for GC, accuracy for binary properties, mean per-label accuracy
    for the multi-label k-mer target. Layer 0 is the embedding output.
    """
    acts = extract_layer_activations(model, sequences, tokenizer, device, max_length)
    n_layers = acts.shape[0]
    results: dict[str, list[float]] = {}
    for name, y in labels.items():
        kind = _infer_kind(name, np.asarray(y))
        results[name] = [_probe_layer(acts[l], np.asarray(y), kind, test_split) for l in range(n_layers)]
    return results
