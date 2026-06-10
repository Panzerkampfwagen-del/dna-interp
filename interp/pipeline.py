"""Run the full interpretability stack on a trained classifier.

`run_all_interp` packages stages 2 to 5 (attention, patching, probing, JASPAR
motif scan) behind one call so the same analysis runs on the synthetic model, a
TF-binding model, or a real DNABERT-2 checkpoint. It returns a structured results
dict and saves figures/arrays; callers do the narrative printing.
"""

from __future__ import annotations

import numpy as np
import torch

from data.bend_loader import BendDataset, collate
from data.jaspar import builtin_motifs
from interp.attention import analyze_heads, most_specialized_heads, positional_preference
from interp.motif_scan import example_alignment, head_attention_received, motif_head_correlations
from interp.patching import averaged_patch_experiment, top_causal_sites
from interp.probing import build_property_labels, probe_all_layers
from visualize.plot_attention import plot_head_maps
from visualize.plot_motifs import plot_example_alignment, plot_top_motifs
from visualize.plot_patching import plot_patching_heatmap
from visualize.plot_probing import plot_probe_accuracy


@torch.no_grad()
def _confidence(model, sequences, tokenizer, device, max_length, target_class):
    """Per-sequence model confidence: P(enhancer) or sigmoid(TF logit)."""
    model.eval()
    model.to(device)
    ds = BendDataset(sequences, [0] * len(sequences), tokenizer, max_length=max_length)
    out = []
    for start in range(0, len(ds), 64):
        batch = collate([ds[i] for i in range(start, min(start + 64, len(ds)))])
        logits, _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), cache_activations=False)
        if target_class is None:
            out.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
        else:
            out.append(torch.sigmoid(logits.float()[:, target_class]).cpu().numpy())
    return np.concatenate(out)


def run_all_interp(
    model,
    tokenizer,
    device,
    sequences: list[str],
    class_labels: np.ndarray,
    *,
    max_length: int,
    k: int | None = None,
    motif_positions: list[dict[str, int]] | None = None,
    motifs=None,
    target_class: int | None = None,
    tag: str = "",
    n_attn: int = 200,
    n_probe: int = 600,
    n_patch_pairs: int = 20,
    n_motif: int = 300,
) -> dict:
    """Run attention, patching, probing and motif analyses. Returns a dict.

    class_labels is the binary label used to pick positives/negatives and to rank
    discriminative k-mers (for the TF task pass labels[:, target_class]). If k and
    motif_positions are given, the patching result is checked against planted
    motif token positions.
    """
    class_labels = np.asarray(class_labels)
    motifs = motifs or builtin_motifs()
    pre = f"{tag}_" if tag else ""
    results: dict = {}

    stats = analyze_heads(model, sequences, class_labels.tolist(), tokenizer, device, max_length=max_length, n_seqs=n_attn)
    specialized = most_specialized_heads(stats, top_k=5)
    plot_head_maps(stats, stem=f"{pre}attention_heads")
    results["specialized_heads"] = specialized
    results["center_bias"] = positional_preference(stats)
    results["entropy_range"] = (float(stats.mean_entropy.min()), float(stats.mean_entropy.max()))

    pos_idx = np.where(class_labels == 1)[0]
    neg_idx = np.where(class_labels == 0)[0]
    pos_seqs = [sequences[i] for i in pos_idx]
    conf = _confidence(model, pos_seqs[: max(n_patch_pairs * 5, 100)], tokenizer, device, max_length, target_class)
    donor_local = np.argsort(conf)[::-1][:n_patch_pairs]
    donor_idx = pos_idx[donor_local]
    clean_seqs = [sequences[i] for i in neg_idx[:n_patch_pairs]]
    patch_seqs = [sequences[i] for i in donor_idx]
    layers = list(range(model.n_layers))
    heat = averaged_patch_experiment(model, tokenizer, clean_seqs, patch_seqs, layers, device, max_length, target_class)
    top_sites = top_causal_sites(heat, layers, top_k=5)
    plot_patching_heatmap(heat, layers, stem=f"{pre}patching")
    patch_res = {"heatmap": heat, "layers": layers, "top_sites": top_sites}
    if motif_positions is not None and k is not None:
        motif_tokens = sorted({b // k + 1 for i in donor_idx for b in motif_positions[i].values()})
        causal_tokens = sorted({p for _, p, _ in top_sites})
        near = [t for t in causal_tokens if any(abs(t - m) <= 1 for m in motif_tokens)]
        patch_res["near_motif"] = {"causal_tokens": causal_tokens, "motif_tokens": motif_tokens, "near": near}
    results["patching"] = patch_res

    probe_seqs = sequences[:n_probe]
    labels = build_property_labels(probe_seqs, class_labels[:n_probe])
    probe_res = probe_all_layers(model, probe_seqs, labels, tokenizer, device, max_length=max_length)
    plot_probe_accuracy(probe_res, stem=f"{pre}probing")
    results["probing"] = probe_res

    enh_seqs = pos_seqs[:n_motif]
    heads = [(l, h) for (l, h, _) in specialized]
    collected = head_attention_received(model, enh_seqs, tokenizer, device, heads, max_length=max_length)
    corr = motif_head_correlations(collected, enh_seqs, motifs, top_k=5)
    plot_top_motifs(corr, stem=f"{pre}motif_correlations")
    top_head = heads[0]
    top_motif = next(m for m in motifs if m.name == corr[top_head][0][0])
    plot_example_alignment(example_alignment(collected, enh_seqs, top_motif, top_head, 0), stem=f"{pre}motif_alignment")
    results["motif_correlations"] = corr

    return results
