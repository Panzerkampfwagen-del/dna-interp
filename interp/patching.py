"""Activation patching for the DNA classifier.

Adapted from the grokking project's `_patch` recovery loop (see grokking_bridge).
There the score was a recovery fraction; here, as the prompt specifies, we report
the raw change in the enhancer logit (Delta logit) for every (layer, position),
which localizes where the enhancer signal is carried.

The enhancer score of a forward is logit[enhancer] - logit[non-enhancer]. Delta
logit at (L, P) is that score after replacing the residual stream leaving layer L
at position P with the same activation from a donor sequence, minus the clean
score. A donor that is itself the clean sequence gives Delta = 0 everywhere,
which is the correctness test the prompt asks for.

Residual layer index L refers to encoder layer L (0-indexed): patching L replaces
cache['resid'][L+1], the residual stream leaving that layer.
"""

from __future__ import annotations

import numpy as np
import torch


def _score(logits: torch.Tensor, target_class: int | None = None) -> torch.Tensor:
    """Score to patch against, per row.

    Binary enhancer task (target_class None): logit(class 1) - logit(class 0).
    Multi-label TF task: the chosen TF's logit, so the same experiment localizes
    the signal for one transcription factor.
    """
    logits = logits.float()  # score in fp32 so a bf16 backbone doesn't perturb Delta
    if target_class is None:
        return logits[:, 1] - logits[:, 0]
    return logits[:, target_class]


def _encode(tokenizer, seq: str, max_length: int):
    enc = tokenizer(seq, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


@torch.no_grad()
def patch_activation_experiment(
    model,
    tokenizer,
    seq_clean: str,
    seq_patch: str,
    layers: list[int],
    device=None,
    max_length: int = 128,
    target_class: int | None = None,
) -> np.ndarray:
    """Delta logit for every (layer, position). Shape [len(layers), seq_len].

    For each requested layer a single batched forward patches position i in row i,
    so the whole position sweep for a layer costs one forward pass. `target_class`
    selects a TF logit for the multi-label task (None = binary enhancer score).
    """
    device = device or next(model.parameters()).device
    model.eval()
    model.to(device)
    ids_c, m_c = _encode(tokenizer, seq_clean, max_length)
    ids_p, m_p = _encode(tokenizer, seq_patch, max_length)
    ids_c, m_c = ids_c.to(device), m_c.to(device)
    ids_p, m_p = ids_p.to(device), m_p.to(device)

    clean_logits, _ = model(ids_c, m_c, cache_activations=False)
    clean_score = _score(clean_logits, target_class)[0].item()
    _, donor = model(ids_p, m_p, cache_activations=True)
    resid = donor["resid"]  # [n_layers+1, 1, S, H]

    S = ids_c.shape[1]
    valid = m_c[0].bool()
    out = np.zeros((len(layers), S), dtype=np.float64)
    idx = torch.arange(S, device=device)

    for li, L in enumerate(layers):
        values = resid[L + 1, 0]  # [S, H]
        handle = model._layers[L].register_forward_hook(_batched_position_hook(idx, values))
        try:
            ids_batch = ids_c.expand(S, -1)
            mask_batch = m_c.expand(S, -1)
            logits, _ = model(ids_batch, mask_batch, cache_activations=False)
        finally:
            handle.remove()
        delta = (_score(logits, target_class) - clean_score).float().cpu().numpy()
        delta[~valid.cpu().numpy()] = 0.0
        out[li] = delta
    return out


def _batched_position_hook(idx: torch.Tensor, values: torch.Tensor):
    """Forward hook that patches row i at position i with values[i]."""

    def hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        hs = (output[0] if is_tuple else output).clone()
        hs[idx, idx, :] = values.to(hs.dtype)
        if is_tuple:
            return (hs,) + tuple(output[1:])
        return hs

    return hook


def averaged_patch_experiment(
    model,
    tokenizer,
    clean_seqs: list[str],
    patch_seqs: list[str],
    layers: list[int],
    device=None,
    max_length: int = 128,
    target_class: int | None = None,
) -> np.ndarray:
    """Mean Delta-logit heatmap over matched (clean, patch) pairs."""
    acc = None
    n = min(len(clean_seqs), len(patch_seqs))
    for c, p in zip(clean_seqs[:n], patch_seqs[:n]):
        heat = patch_activation_experiment(model, tokenizer, c, p, layers, device, max_length, target_class)
        acc = heat if acc is None else acc + heat
    return acc / max(n, 1)


def top_causal_sites(heatmap: np.ndarray, layers: list[int], top_k: int = 5) -> list[tuple[int, int, float]]:
    """Top (layer, position, Delta) cells by absolute Delta logit."""
    flat = [
        (layers[li], p, float(heatmap[li, p]))
        for li in range(heatmap.shape[0])
        for p in range(heatmap.shape[1])
    ]
    flat.sort(key=lambda t: abs(t[2]), reverse=True)
    return flat[:top_k]


def significant_sites(heatmap: np.ndarray, n_std: float = 0.5) -> np.ndarray:
    """Boolean mask of cells whose Delta exceeds mean + n_std * std."""
    thresh = heatmap.mean() + n_std * heatmap.std()
    return heatmap > thresh
