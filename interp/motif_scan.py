"""Correlate attention-head behaviour with JASPAR TF binding motifs.

For a specialized head we ask: within each sequence, does the head attend to the
tokens where a motif scores highly? We compute, per sequence, the head's
attention received by each token and the motif's per-token log-odds track, then
take the Pearson correlation between them and average across sequences. The
shared `pearson` comes from the grokking project (grokking_bridge).

Two views are provided: the per-sequence correlation above (primary, robust to
where in the sequence a motif lands) and the prompt's positional view that
correlates the across-sequence average attention profile with the across-sequence
average motif track.
"""

from __future__ import annotations

import numpy as np
import torch

from data.jaspar import Motif, window_scores
from grokking_bridge import pearson


def base_score_track(seq: str, motif: Motif, both_strands: bool = True) -> np.ndarray:
    """Per-base relu log-odds: the best positive motif score covering each base."""
    track = np.zeros(len(seq))
    w = motif.width
    for m in ([motif, motif.reverse_complement()] if both_strands else [motif]):
        ws = window_scores(seq, m)
        for i, s in enumerate(ws):
            if s > 0:
                track[i : i + w] = np.maximum(track[i : i + w], s)
    return track


def motif_token_track(seq: str, motif: Motif, offsets: np.ndarray, both_strands: bool = True) -> np.ndarray:
    """Map the per-base motif track onto tokens via offset mapping. Shape [S]."""
    base = base_score_track(seq, motif, both_strands)
    S = offsets.shape[0]
    track = np.zeros(S)
    for t in range(S):
        a, b = int(offsets[t, 0]), int(offsets[t, 1])
        if b > a:
            track[t] = base[a:b].max()
    return track


@torch.no_grad()
def head_attention_received(
    model,
    sequences: list[str],
    tokenizer,
    device,
    heads: list[tuple[int, int]],
    max_length: int = 128,
    batch_size: int = 16,
) -> dict:
    """Per-sequence attention received by each token, for the requested heads.

    Returns dict with `profiles` (head -> [N, S]), `masks` [N, S], and
    `offsets` [N, S, 2] so motif tracks can be aligned to the same tokens.
    """
    model.eval()
    model.to(device)
    profiles: dict[tuple[int, int], list[np.ndarray]] = {hd: [] for hd in heads}
    masks: list[np.ndarray] = []
    offsets: list[np.ndarray] = []

    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start : start + batch_size]
        enc = tokenizer(
            chunk,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        offsets.append(np.asarray(enc["offset_mapping"]))
        masks.append(mask.cpu().numpy())

        _, cache = model(ids, mask, cache_activations=True)
        attn = cache["attn_pattern"]  # [L, B, H, S, S]
        qmask = mask[:, :, None].to(attn.dtype)  # [B, Sq, 1]
        nq = mask.sum(dim=1).clamp(min=1.0)       # [B]
        for (l, h) in heads:
            p = attn[l, :, h]  # [B, Sq, Sk]
            recv = (p * qmask).sum(dim=1) / nq[:, None]  # [B, Sk]
            profiles[(l, h)].append(recv.float().cpu().numpy())

    return {
        "profiles": {hd: np.concatenate(v, axis=0) for hd, v in profiles.items()},
        "masks": np.concatenate(masks, axis=0),
        "offsets": np.concatenate(offsets, axis=0),
    }


def motif_head_correlations(
    collected: dict,
    sequences: list[str],
    motifs: list[Motif],
    top_k: int = 5,
    return_all: bool = False,
) -> dict[tuple[int, int], list[tuple[str, float]]]:
    """For each head, the top-k motifs by mean per-sequence attention correlation.

    With return_all, returns the full sorted (motif, r) list per head so a caller
    can compute a noise floor (e.g. against shuffled-motif controls).
    """
    masks = collected["masks"]
    offsets = collected["offsets"]
    n = len(sequences)

    # precompute per-sequence token tracks per motif
    tracks: dict[str, list[np.ndarray]] = {}
    for motif in motifs:
        tracks[motif.name] = [motif_token_track(sequences[i], motif, offsets[i]) for i in range(n)]

    out: dict[tuple[int, int], list[tuple[str, float]]] = {}
    for hd, prof in collected["profiles"].items():
        scored: list[tuple[str, float]] = []
        for motif in motifs:
            corrs = []
            for i in range(n):
                valid = masks[i].astype(bool)
                a = prof[i][valid]
                m = tracks[motif.name][i][valid]
                if a.std() > 1e-8 and m.std() > 1e-8:
                    corrs.append(pearson(a, m))
            scored.append((motif.name, float(np.mean(corrs)) if corrs else 0.0))
        scored.sort(key=lambda t: t[1], reverse=True)
        out[hd] = scored if return_all else scored[:top_k]
    return out


def positional_motif_correlation(
    collected: dict, sequences: list[str], motif: Motif
) -> dict[tuple[int, int], float]:
    """Prompt's positional view: correlate mean attention profile with mean motif
    track across sequences, per head.
    """
    masks = collected["masks"]
    offsets = collected["offsets"]
    n = len(sequences)
    valid = masks.sum(axis=0) > 0  # tokens valid in at least one sequence
    mean_track = np.mean(
        [motif_token_track(sequences[i], motif, offsets[i]) for i in range(n)], axis=0
    )
    out = {}
    for hd, prof in collected["profiles"].items():
        mean_prof = prof.mean(axis=0)
        a, m = mean_prof[valid], mean_track[valid]
        out[hd] = pearson(a, m) if a.std() > 1e-8 and m.std() > 1e-8 else 0.0
    return out


def example_alignment(
    collected: dict, sequences: list[str], motif: Motif, head: tuple[int, int], seq_index: int
) -> dict:
    """Token-level attention and motif tracks for one sequence, for a figure."""
    mask = collected["masks"][seq_index].astype(bool)
    offsets = collected["offsets"][seq_index]
    attn = collected["profiles"][head][seq_index]
    track = motif_token_track(sequences[seq_index], motif, offsets)
    return {
        "attention": attn[mask],
        "motif_track": track[mask],
        "sequence": sequences[seq_index],
        "head": head,
        "motif": motif.name,
    }
