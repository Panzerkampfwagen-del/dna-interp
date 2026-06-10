"""Test-failure activation patching on the fine-tuned DNABERT-2 enhancer model.

Patches the residual stream from confident true-positives into confident
false-negatives (GC-matched) and localizes the recovered enhancer logit to a
(layer, position), to ask what representation misclassified sequences lack.
Runs a null control (random GC-matched source) and a self-patch sanity check.

    python scripts/run_test_failure_patching.py \
        --checkpoint results/checkpoints/dnabert2_enhancer_best.pt \
        --n-pairs 50 --confidence 0.3 --out results/test_failure_patching/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CHECKPOINTS, RAW, get_device, set_seed  # noqa: E402
from data.bend_loader import load_dnabert2_tokenizer, load_nt_enhancer  # noqa: E402
from data.jaspar import builtin_motifs, parse_jaspar  # noqa: E402
from finetune.trainer import load_checkpoint  # noqa: E402
from interp.test_failure_patching import (  # noqa: E402
    build_matched_pairs,
    build_null_pairs,
    motif_overlap_at_sites,
    patch_pair,
    predict_probs,
    run_patching_experiment,
    summarize_results,
)
from models.dna_lm import load_dnabert2_classifier  # noqa: E402


def section(t: str) -> None:
    print(f"\n{'=' * 4} {t} {'=' * 4}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(CHECKPOINTS / "dnabert2_enhancer_best.pt"))
    ap.add_argument("--n-pairs", type=int, default=50)
    ap.add_argument("--confidence", type=float, default=0.3)
    ap.add_argument("--max-length", type=int, default=48)
    ap.add_argument("--out", default="results/test_failure_patching/")
    args = ap.parse_args()

    set_seed(0)
    device = get_device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ml = args.max_length

    tok = load_dnabert2_tokenizer()
    model = load_dnabert2_classifier(num_labels=2).to(device).eval()
    load_checkpoint(model, Path(args.checkpoint), device)

    seqs, labs = load_nt_enhancer("test")
    probs = predict_probs(model, tok, seqs, device, ml)
    acc = float(((probs > 0.5).astype(int) == np.array(labs)).mean())
    print(f"test set: {len(seqs)} seqs, accuracy {acc:.3f}")

    section("Build GC-matched (FN, TP) pairs")
    conf = args.confidence
    pairs = build_matched_pairs(model, tok, seqs, labs, args.n_pairs, conf, device=device, max_length=ml, probs=probs)
    if len(pairs) < 30:
        conf = 0.4
        print(f"  only {len(pairs)} pairs at confidence 0.3; relaxing to {conf} (documented)")
        pairs = build_matched_pairs(model, tok, seqs, labs, args.n_pairs, conf, device=device, max_length=ml, probs=probs)
    print(f"  {len(pairs)} pairs (FN prob < {conf}, TP prob > 0.7, |dGC| <= 0.05)")
    if not pairs:
        print("  no GC-matched FN<-TP pairs found; nothing to patch. Exiting.")
        return
    null_pairs = build_null_pairs(seqs, pairs)

    section("Self-patch sanity (must be ~0 on real DNABERT-2)")
    layers = list(range(model.n_layers))
    self_delta = patch_pair(model, tok, pairs[0][0], pairs[0][0], layers, device, ml)
    self_max = float(np.abs(self_delta).max())
    print(f"  max |self-patch Delta| = {self_max:.2e}  ({'PASS' if self_max < 1e-3 else 'FAIL'})")

    section(f"Main experiment: {len(pairs)} FN<-TP pairs, layers 0-{model.n_layers - 1}")
    delta_main = run_patching_experiment(model, tok, pairs, layers, device, ml)
    delta_null = run_patching_experiment(model, tok, null_pairs, layers, device, ml)
    main = summarize_results(delta_main, layers)
    null = summarize_results(delta_null, layers)
    print(f"  main: max mean Delta {main['max_mean']:+.3f}, mean|Delta| {main['mean_abs']:.3f}")
    print(f"  null: max mean Delta {null['max_mean']:+.3f}, mean|Delta| {null['mean_abs']:.3f}")
    null_clean = null["mean_abs"] < 0.05
    print(f"  null control {'clean (< 0.05)' if null_clean else 'NOT clean: patching any GC-matched donor moves the logit'}")

    # The enhancer-specific signal is main - null: it removes the non-specific
    # effect of patching in any in-distribution activation. CLS (pos 0) is the
    # pooling token, not sequence content, so it is excluded for localization.
    spec = main["mean"] - null["mean"]
    spec_flat = [(int(L), int(p), float(spec[li, p])) for li, L in enumerate(layers) for p in range(spec.shape[1]) if p != 0]
    spec_flat.sort(key=lambda t: t[2], reverse=True)
    spec_top = spec_flat[:5]
    print(f"  enhancer-specific (main-null) max {max((t[2] for t in spec_flat), default=0.0):+.3f}; top sites {spec_top}")

    section("Top causal positions (raw main mean Delta logit)")
    lines = [f"enhancer-specific (main-null) top sites (excl. CLS): {spec_top}", ""]
    for L in layers:
        top = main["per_layer_top"][L]
        ps = [p for p, _ in top]
        ds = [round(d, 3) for _, d in top]
        line = f"Layer {L}: positions {ps} - mean Delta {ds}"
        lines.append(line)
        print("  " + line)

    section("Motif overlap at top-5 sites (TP donors, RFX1 = top main-analysis head)")
    motifs = parse_jaspar(RAW / "JASPAR2024_CORE_vertebrates.txt") if (RAW / "JASPAR2024_CORE_vertebrates.txt").exists() else builtin_motifs()
    overlap = motif_overlap_at_sites(spec_top, pairs, tok, motifs, ml, focus_names=("RFX1",))
    mo_lines = []
    for focus, res in overlap.items():
        mo_lines.append(f"{focus}: random-token baseline log-odds {res['baseline']:.2f}")
        print(f"  {focus} baseline (random token): {res['baseline']:.2f}")
        for L, P, score in res["site_scores"]:
            mo_lines.append(f"  L{L} pos{P}: {focus} log-odds {score:.2f}")
            print(f"    L{L} pos{P}: {focus} log-odds {score:.2f}")

    # Outcome classification on the enhancer-specific (main-null) signal.
    strong = [(L, p, d) for L, p, d in spec_top if d > 0.2]
    spec_max = max((t[2] for t in spec_flat), default=0.0)
    if len(strong) >= 3 and all(8 <= L <= 11 for L, _, _ in strong):
        outcome = "A (localized late-layer enhancer-specific signal)"
    elif len(strong) >= 3 and all(L <= 3 for L, _, _ in strong):
        outcome = "B (early-layer composition signal)"
    elif spec_max < 0.1:
        outcome = "C (no enhancer-specific signal beyond the non-specific null)"
    else:
        outcome = "mixed: real specific signal but spread across layers, not motif-localized"
    print(f"\nOutcome: {outcome}")

    np.savez(out / "delta_matrix.npz", main=delta_main, null=delta_null, layers=np.array(layers),
             mean_main=main["mean"], mean_null=null["mean"], mean_specific=spec)
    (out / "top_causal_positions.txt").write_text(
        f"outcome: {outcome}\nself-patch max |Delta|: {self_max:.2e}\n"
        f"main max mean Delta {main['max_mean']:+.3f}, mean|Delta| {main['mean_abs']:.3f}\n"
        f"null max mean Delta {null['max_mean']:+.3f}, mean|Delta| {null['mean_abs']:.3f}  "
        f"(clean<0.05: {null_clean})\n"
        f"enhancer-specific (main-null) max {spec_max:+.3f}\n"
        f"specific top sites (layer,pos,Delta) excl CLS: {spec_top}\n\n" + "\n".join(lines) + "\n")
    (out / "motif_overlap.txt").write_text("\n".join(mo_lines) + "\n")
    _heatmap(main["mean"], layers, out / "delta_heatmap.png", "FN<-TP patching: mean Delta enhancer logit")
    _heatmap(spec, layers, out / "delta_specific_heatmap.png", "enhancer-specific (main - null) Delta logit")
    section("done")
    print(f"  outputs in {out}/")


def _heatmap(mean, layers, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    mx = float(np.abs(mean).max()) or 1.0
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(mean, aspect="auto", cmap="coolwarm", origin="lower", vmin=-mx, vmax=mx)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)
    ax.set_xlabel("token position")
    ax.set_ylabel("layer")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
