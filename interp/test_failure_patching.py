"""Activation patching between matched test-failure pairs.

The fine-tuned DNABERT-2 has a generalization gap (val MCC 0.89 vs test 0.55).
This asks *why*: patch the residual stream from a confidently-correct enhancer
(TP) into a confidently-wrong one (FN) and localize the recovered enhancer logit
to a (layer, position). Pairs are GC-matched because GC content is the dominant
probe signal (R²=0.99 @ L0), so unmatched patching would just measure GC.

The synthetic BERT uses the validated `interp.patching` directly. DNABERT-2 unpads
tokens inside each layer, so its donor cache (re-padded [.,S,H]) and the live layer
output (packed [total,H]) live in different spaces; `_patch_pair_remote` remaps
padded position -> packed token via the `indices` tensor each layer receives.
"""

from __future__ import annotations

import numpy as np
import torch

from interp.patching import _score, patch_activation_experiment
from interp.probing import gc_content  # reuse the upper()-normalizing helper


@torch.no_grad()
def predict_probs(model, tokenizer, sequences, device=None, max_length=48, batch_size=64) -> np.ndarray:
    """P(enhancer) for each sequence."""
    device = device or next(model.parameters()).device
    model.eval()
    model.to(device)
    out = []
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start : start + batch_size]
        enc = tokenizer(chunk, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        logits, _ = model(enc["input_ids"].to(device), enc["attention_mask"].to(device), cache_activations=False)
        out.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def build_matched_pairs(
    model,
    tokenizer,
    sequences,
    labels,
    n_pairs: int = 50,
    confidence_threshold: float = 0.3,
    gc_bin_width: float = 0.05,
    device=None,
    max_length: int = 48,
    probs: np.ndarray | None = None,
) -> list[tuple[str, str]]:
    """GC-matched (FN, TP) pairs: confident false-negative paired to the closest-GC
    confident true-positive. Pass `probs` to skip inference (used in tests)."""
    labels = np.asarray(labels)
    if probs is None:
        probs = predict_probs(model, tokenizer, sequences, device, max_length)
    gc = np.array([gc_content(s) for s in sequences])
    fn = np.where((labels == 1) & (probs < confidence_threshold))[0]
    tp = np.where((labels == 1) & (probs > 0.7))[0]

    pairs: list[tuple[str, str]] = []
    for i in fn:
        d = np.abs(gc[tp] - gc[i])
        cand = tp[d <= gc_bin_width]
        if len(cand) == 0:
            continue
        best = cand[np.argmin(np.abs(gc[cand] - gc[i]))]  # closest GC within the bin
        pairs.append((sequences[i], sequences[int(best)]))
        if len(pairs) >= n_pairs:
            break
    return pairs


def build_null_pairs(sequences, fn_pairs, gc_bin_width: float = 0.05, seed: int = 0) -> list[tuple[str, str]]:
    """Null control: each FN paired to a random same-GC-bin source (any label)."""
    rng = np.random.default_rng(seed)
    gc = np.array([gc_content(s) for s in sequences])
    out = []
    for fn_seq, _ in fn_pairs:
        cand = np.where(np.abs(gc - gc_content(fn_seq)) <= gc_bin_width)[0]
        if len(cand) == 0:
            continue
        out.append((fn_seq, sequences[int(rng.choice(cand))]))
    return out


def _remote_patch_hook(S: int, donor_L: torch.Tensor, both_valid: torch.Tensor):
    """Patch the packed layer output: token for (row i, pos i) -> donor[pos i].

    `indices[j]` is the flat padded index (row*S + pos) of unpadded token j, so
    row==pos selects the diagonal patch and `% S` recovers the donor position.
    """
    def hook(module, args, output):
        indices = args[4]
        rows = torch.div(indices, S, rounding_mode="floor")
        poss = indices % S
        sel = (rows == poss) & both_valid[poss]
        if sel.any():
            out = output.clone()
            out[sel] = donor_L[poss[sel]].to(out.dtype)
            return out
        return output

    return hook


@torch.no_grad()
def _patch_pair_remote(model, tokenizer, fn_seq, tp_seq, layers, device, max_length, target_class=None) -> np.ndarray:
    """DNABERT-2 patching with padded->packed position remapping. [n_layers, S]."""
    device = device or next(model.parameters()).device
    model.eval()
    model.to(device)
    enc_f = tokenizer(fn_seq, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    enc_t = tokenizer(tp_seq, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    ids_f, m_f = enc_f["input_ids"].to(device), enc_f["attention_mask"].to(device)
    ids_t, m_t = enc_t["input_ids"].to(device), enc_t["attention_mask"].to(device)

    clean_logits, _ = model(ids_f, m_f, cache_activations=False)
    clean_score = _score(clean_logits, target_class)[0].item()
    _, donor = model(ids_t, m_t, cache_activations=True)
    resid = donor["resid"]  # [n_layers+1, 1, S, H], re-padded

    S = ids_f.shape[1]
    both_valid = (m_f[0].bool() & m_t[0].bool())
    out = np.zeros((len(layers), S), dtype=np.float64)
    for li, L in enumerate(layers):
        donor_L = resid[L + 1, 0]  # [S, H]
        handle = model._layers[L].register_forward_hook(_remote_patch_hook(S, donor_L, both_valid))
        try:
            logits, _ = model(ids_f.expand(S, -1), m_f.expand(S, -1), cache_activations=False)
        finally:
            handle.remove()
        delta = (_score(logits, target_class) - clean_score).float().cpu().numpy()
        delta[~both_valid.cpu().numpy()] = 0.0
        out[li] = delta
    return out


def patch_pair(model, tokenizer, fn_seq, tp_seq, layers, device=None, max_length=48, target_class=None) -> np.ndarray:
    """One pair's [n_layers, S] Delta-logit matrix, dispatching by model type."""
    if getattr(model, "_remote", False):
        return _patch_pair_remote(model, tokenizer, fn_seq, tp_seq, layers, device, max_length, target_class)
    return patch_activation_experiment(model, tokenizer, fn_seq, tp_seq, layers, device, max_length, target_class)


def run_patching_experiment(model, tokenizer, pairs, layers=None, device=None, max_length=48, target_class=None) -> np.ndarray:
    """Per-pair Delta-logit matrices, shape [n_pairs, n_layers, seq_len]."""
    layers = layers if layers is not None else list(range(model.n_layers))
    mats = [patch_pair(model, tokenizer, fn, tp, layers, device, max_length, target_class) for fn, tp in pairs]
    return np.stack(mats, axis=0)


def summarize_results(delta_matrix: np.ndarray, layers=None, top_k: int = 5) -> dict:
    """Mean/std over pairs, top positions per layer and overall."""
    mean = delta_matrix.mean(axis=0)  # [n_layers, S]
    n_layers, S = mean.shape
    layers = layers if layers is not None else list(range(n_layers))
    per_layer = {}
    for li, L in enumerate(layers):
        order = np.argsort(mean[li])[::-1][:3]
        per_layer[int(L)] = [(int(p), float(mean[li, p])) for p in order]
    flat = [(int(layers[li]), int(p), float(mean[li, p])) for li in range(n_layers) for p in range(S)]
    flat.sort(key=lambda t: t[2], reverse=True)
    return {
        "mean": mean,
        "per_layer_top": per_layer,
        "top_overall": flat[:top_k],
        "max_mean": float(mean.max()),
        "mean_abs": float(np.abs(mean).mean()),
    }


def motif_overlap_at_sites(top_sites, pairs, tokenizer, motifs, max_length=48, focus_names=("RFX1",)) -> dict:
    """Do the top causal token positions sit on JASPAR motif matches in the TP donors?

    For each top (layer, position) and focus motif, the mean best log-odds over the
    base span of that token across donor sequences, vs a random-token baseline.
    """
    from interp.motif_scan import base_score_track

    tp_seqs = [tp for _, tp in pairs]
    encs = [tokenizer(s, truncation=True, max_length=max_length, return_offsets_mapping=True) for s in tp_seqs]
    name2motif = {m.name: m for m in motifs}
    results = {}
    for focus in focus_names:
        if focus not in name2motif:
            continue
        tracks = [base_score_track(s, name2motif[focus]) for s in tp_seqs]
        site_scores = []
        for (L, P, _d) in top_sites:
            vals = []
            for k, enc in enumerate(encs):
                offs = enc["offset_mapping"]
                if P >= len(offs):
                    continue
                a, b = offs[P]
                if b > a and b <= len(tracks[k]):
                    vals.append(float(tracks[k][a:b].max()))
            site_scores.append((int(L), int(P), float(np.mean(vals)) if vals else 0.0))
        rng = np.random.default_rng(0)
        base_vals = []
        for k, enc in enumerate(encs):
            valid = [i for i, o in enumerate(enc["offset_mapping"]) if o[1] > o[0]]
            if valid:
                a, b = enc["offset_mapping"][int(rng.choice(valid))]
                base_vals.append(float(tracks[k][a:b].max()))
        results[focus] = {"site_scores": site_scores, "baseline": float(np.mean(base_vals)) if base_vals else 0.0}
    return results
