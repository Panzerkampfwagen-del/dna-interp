"""Attention-head analysis: entropy, positional preference, task correlation.

For every head (DNABERT-2: 12 layers x 12 heads) we measure how peaked its
attention is (entropy), where it likes to look (positional profile), and whether
its peakedness tracks the model's confidence (task correlation). Statistics are
accumulated in a streaming pass so the full [N, L, H, S, S] attention tensor is
never materialized.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from data.bend_loader import BendDataset, collate
from metrics import spearman

EPS = 1e-9


@dataclass
class HeadStats:
    mean_entropy: np.ndarray       # [L, H] normalized attention entropy, low = specialized
    task_correlation: np.ndarray   # [L, H] spearman(per-seq entropy, confidence)
    position_profile: np.ndarray   # [L, H, S] mean attention received per key position
    per_seq_entropy: np.ndarray    # [N, L, H]
    confidence: np.ndarray         # [N]
    n_layers: int
    n_heads: int


def _batch_entropy_and_profile(
    attn: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head normalized entropy [B,L,H] and key-attention profile [B,L,H,S].

    attn: [B, L, H, Sq, Sk] softmax weights. mask: [B, S] with 1 for real tokens.
    Entropy is normalized by log(number of valid keys) so it lands in [0, 1].
    """
    key_mask = mask[:, None, None, None, :].to(attn.dtype)
    p = attn * key_mask
    p = p / (p.sum(dim=-1, keepdim=True) + EPS)
    ent = -(p * torch.log(p + EPS)).sum(dim=-1)  # [B,L,H,Sq]

    q_mask = mask[:, None, None, :].to(attn.dtype)  # [B,1,1,Sq]
    nq = q_mask.sum(dim=-1).clamp(min=1.0)          # [B,1,1]
    ent_mean = (ent * q_mask).sum(dim=-1) / nq      # [B,L,H]

    n_keys = mask.sum(dim=-1).clamp(min=2.0)        # [B]
    ent_norm = ent_mean / torch.log(n_keys)[:, None, None]

    recv = (p * q_mask[..., None]).sum(dim=3) / nq[..., None]  # [B,L,H,Sk]
    return ent_norm, recv


@torch.no_grad()
def analyze_heads(
    model,
    sequences: list[str],
    labels: list[int],
    tokenizer,
    device,
    max_length: int = 128,
    n_seqs: int = 100,
    batch_size: int = 16,
) -> HeadStats:
    """Compute per-head statistics over the first `n_seqs` sequences."""
    model.eval()
    model.to(device)
    seqs = sequences[:n_seqs]
    labs = labels[:n_seqs]
    ds = BendDataset(seqs, labs, tokenizer, max_length=max_length)

    ent_rows: list[np.ndarray] = []
    conf_rows: list[np.ndarray] = []
    profile_sum = None
    profile_count = 0
    L = H = None

    for start in range(0, len(ds), batch_size):
        batch = collate([ds[i] for i in range(start, min(start + batch_size, len(ds)))])
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        logits, cache = model(input_ids, mask, cache_activations=True)
        attn = cache["attn_pattern"].permute(1, 0, 2, 3, 4).contiguous()  # [B,L,H,S,S]

        ent_norm, recv = _batch_entropy_and_profile(attn, mask)
        ent_rows.append(ent_norm.float().cpu().numpy())
        probs = torch.softmax(logits.float(), dim=-1)
        conf_rows.append(probs.max(dim=-1).values.cpu().numpy())

        recv_np = recv.float().cpu().numpy()
        if profile_sum is None:
            L, H = recv_np.shape[1], recv_np.shape[2]
            profile_sum = recv_np.sum(axis=0)
        else:
            profile_sum += recv_np.sum(axis=0)
        profile_count += recv_np.shape[0]

    per_seq_entropy = np.concatenate(ent_rows, axis=0)  # [N,L,H]
    confidence = np.concatenate(conf_rows, axis=0)       # [N]
    mean_entropy = per_seq_entropy.mean(axis=0)          # [L,H]
    position_profile = profile_sum / max(profile_count, 1)

    task_corr = np.zeros((L, H), dtype=np.float64)
    for l in range(L):
        for h in range(H):
            task_corr[l, h] = spearman(per_seq_entropy[:, l, h], confidence)

    return HeadStats(
        mean_entropy=mean_entropy,
        task_correlation=task_corr,
        position_profile=position_profile,
        per_seq_entropy=per_seq_entropy,
        confidence=confidence,
        n_layers=L,
        n_heads=H,
    )


def most_specialized_heads(stats: HeadStats, top_k: int = 5) -> list[tuple[int, int, float]]:
    """Heads with the lowest normalized entropy, as (layer, head, entropy)."""
    flat = [(l, h, float(stats.mean_entropy[l, h])) for l in range(stats.n_layers) for h in range(stats.n_heads)]
    flat.sort(key=lambda t: t[2])
    return flat[:top_k]


def positional_preference(stats: HeadStats) -> np.ndarray:
    """Per head, a center-bias scalar in [-1, 1].

    Positive means the head attends more to the central third of the sequence
    than to the two outer thirds; negative means it prefers the edges.
    """
    L, H, S = stats.position_profile.shape
    third = S // 3
    center = stats.position_profile[:, :, third : 2 * third].mean(axis=-1)
    edges = np.concatenate(
        [stats.position_profile[:, :, :third], stats.position_profile[:, :, 2 * third :]],
        axis=-1,
    ).mean(axis=-1)
    return (center - edges) / (center + edges + EPS)
