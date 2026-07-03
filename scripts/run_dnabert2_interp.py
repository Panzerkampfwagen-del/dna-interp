"""Run the interpretability stack on the real DNABERT-2 model.

Loads DNABERT-2 (patched to use eager attention so it runs without Triton and
exposes attention weights), then probes its residual stream and analyzes its 144
attention heads. By default it analyzes the pretrained backbone on synthetic
sequences with planted motifs; pass a fine-tuned checkpoint and real BEND data
for the full study.

Activation patching is not run here: DNABERT-2's unpadded internals need position
remapping (the synthetic BERT validates the patching logic itself).

    /home/aryan/anaconda3/envs/tinyinfer-gpu/bin/python scripts/run_dnabert2_interp.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CACHE, RAW, get_device, set_seed  # noqa: E402
from data.bend_loader import load_dnabert2_tokenizer, load_nt_enhancer, make_synthetic_enhancer_dataset  # noqa: E402
from data.jaspar import Motif, builtin_motifs, parse_jaspar  # noqa: E402
from finetune.trainer import load_checkpoint  # noqa: E402
from interp.attention import analyze_heads, most_specialized_heads  # noqa: E402
from interp.motif_scan import head_attention_received, motif_head_correlations  # noqa: E402
from interp.probing import build_property_labels, probe_all_layers  # noqa: E402
from models.dna_lm import load_dnabert2_classifier  # noqa: E402
from visualize.plot_attention import plot_head_maps  # noqa: E402
from visualize.plot_motifs import plot_motif_noisefloor  # noqa: E402
from visualize.plot_probing import plot_probe_accuracy  # noqa: E402


def section(t: str) -> None:
    print(f"\n{'=' * 4} {t} {'=' * 4}")


def build_motif_set(n_shuffled: int = 40):
    """Real JASPAR 2024 motifs (if downloaded) plus column-shuffled controls.

    The shuffled motifs keep each real motif's column composition but scramble
    column order, so they have the same score scale yet no genomic grammar. The
    best real-motif correlation is only meaningful relative to this noise floor,
    which matters when scanning ~879 motifs (best-of-many inflates the max).
    """
    jaspar = RAW / "JASPAR2024_CORE_vertebrates.txt"
    real = parse_jaspar(jaspar) if jaspar.exists() else builtin_motifs()
    rng = np.random.default_rng(0)
    controls = []
    pick = rng.choice(len(real), size=min(n_shuffled, len(real)), replace=False)
    for j, idx in enumerate(pick):
        probs = real[idx].probs
        controls.append(Motif(f"SHUF_{j}", probs[:, rng.permutation(probs.shape[1])]))
    src = "real JASPAR 2024" if jaspar.exists() else "builtin consensus"
    return real, controls, src


def main(checkpoint: str | None = None) -> None:
    set_seed(0)
    device = get_device()
    max_length = 64

    tok = load_dnabert2_tokenizer()
    clf = load_dnabert2_classifier(num_labels=2, cache_activations=True).to(device).eval()
    state = "pretrained backbone"
    if checkpoint and Path(checkpoint).exists():
        load_checkpoint(clf, Path(checkpoint), device)
        state = f"checkpoint {checkpoint}"
    print(f"DNABERT-2 loaded ({state}); layers={clf.n_layers} heads={clf.n_heads} hidden={clf.hidden_size}")

    # Probe the fine-tuned model on the real sequences it was trained on; fall
    # back to synthetic planted-motif sequences when the dataset is unavailable.
    try:
        s_all, l_all = load_nt_enhancer("train")
        l_arr = np.array(l_all)
        pos = np.where(l_arr == 1)[0][:150]
        neg = np.where(l_arr == 0)[0][:150]
        sel = np.concatenate([pos, neg])
        np.random.default_rng(0).shuffle(sel)
        seqs = [s_all[i] for i in sel]
        labs = [int(l_arr[i]) for i in sel]
        data_src = "real NT enhancers"
    except Exception as e:
        ds = make_synthetic_enhancer_dataset(n=300, length=200, seed=0)
        seqs, labs = ds.sequences, ds.labels
        data_src = f"synthetic (real unavailable: {type(e).__name__})"
    print(f"analysis sequences: {len(seqs)} from {data_src}")

    section("Probing residual stream for biological features (13 layers)")
    labels = build_property_labels(seqs, np.array(labs))
    probe = probe_all_layers(clf, seqs, labels, tok, device, max_length=max_length)
    for name, per_layer in probe.items():
        metric = "R2" if name == "gc" else "acc"
        print(f"  {name:5s} ({metric}): best={max(per_layer):.3f} @ L{int(np.argmax(per_layer))}  curve={[round(x, 2) for x in per_layer]}")
    plot_probe_accuracy(probe, stem="dnabert2_probing")

    section("Attention head specialization (144 heads)")
    stats = analyze_heads(clf, seqs, labs, tok, device, max_length=max_length, n_seqs=150)
    specialized = most_specialized_heads(stats, top_k=5)
    n_low = int((stats.mean_entropy < 0.8).sum())
    print(f"  entropy range [{stats.mean_entropy.min():.3f}, {stats.mean_entropy.max():.3f}]; {n_low}/{stats.n_layers * stats.n_heads} heads specialized (<0.8)")
    print(f"  5 most specialized (layer, head, entropy): {[(l, h, round(e, 3)) for l, h, e in specialized]}")
    plot_head_maps(stats, stem="dnabert2_attention_heads")

    section("Specialized heads vs JASPAR motifs (with shuffled-motif noise floor)")
    real_motifs, controls, motif_src = build_motif_set()
    print(f"  motif set: {len(real_motifs)} {motif_src} + {len(controls)} shuffled controls")
    enh = [s for s, l in zip(seqs, labs) if l == 1][:150]
    heads = [(l, h) for (l, h, _) in specialized]
    collected = head_attention_received(clf, enh, tok, device, heads, max_length=max_length)
    corr_all = motif_head_correlations(collected, enh, real_motifs + controls, return_all=True)
    motif_summary = {}
    for hd, scored in corr_all.items():
        real = [(n, r) for n, r in scored if not n.startswith("SHUF_")]
        ctrl = np.array([r for n, r in scored if n.startswith("SHUF_")])
        top_name, top_r = real[0]
        ctrl_max = float(ctrl.max()) if ctrl.size else float("nan")
        ctrl_p95 = float(np.percentile(ctrl, 95)) if ctrl.size else float("nan")
        # Real signal must clear the shuffled-control max AND a minimum magnitude
        # (a ~0 correlation isn't "signal" just because controls are slightly negative).
        above = top_r > ctrl_max and top_r >= 0.05
        verdict = "ABOVE noise" if above else ("null (~0)" if top_r < 0.05 else "within noise")
        print(f"  L{hd[0]}H{hd[1]}: top={top_name} r={top_r:+.3f} | shuffled max={ctrl_max:+.3f} p95={ctrl_p95:+.3f} -> {verdict}")
        motif_summary[f"L{hd[0]}H{hd[1]}"] = {
            "top": real[:5], "shuffled_max": ctrl_max, "shuffled_p95": ctrl_p95, "above_noise": above,
        }
    plot_motif_noisefloor(motif_summary, stem="dnabert2_motif_correlations")

    summary = {
        "state": state,
        "probing": {k: [float(x) for x in v] for k, v in probe.items()},
        "specialized_heads": specialized,
        "n_specialized": n_low,
        "motif_correlations": motif_summary,
        "motif_source": motif_src,
        "data_source": data_src,
    }
    out = CACHE / "dnabert2_interp_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=float))
    section("done")
    print(f"summary: {out}\nfigures: results/figures/dnabert2_*.npz")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
