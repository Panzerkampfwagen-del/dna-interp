"""End-to-end offline validation of the interpretability pipeline (enhancer).

Builds a synthetic enhancer dataset with planted TF motifs and a GC bias, trains
a small local BERT on it, then runs the full analysis stack via
`interp.pipeline.run_all_interp` and checks that the tools recover the planted
signal. This is the offline evidence that the code is correct before pointing it
at DNABERT-2 and real BEND data.

    /home/aryan/anaconda3/envs/tinyinfer-gpu/bin/python scripts/run_synthetic_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CACHE, get_device, set_seed  # noqa: E402
from data.bend_loader import (  # noqa: E402
    BendDataset,
    KmerTokenizer,
    collate,
    make_synthetic_enhancer_dataset,
    train_val_test_split,
)
from finetune.evaluate import evaluate  # noqa: E402
from finetune.trainer import TrainConfig, train  # noqa: E402
from grokking_bridge import GROKKING_CONNECTED, grokking_source  # noqa: E402
from interp.pipeline import run_all_interp  # noqa: E402
from models.dna_lm import build_tiny_classifier  # noqa: E402

K = 3


def section(title: str) -> None:
    print(f"\n{'=' * 4} {title} {'=' * 4}")


def main() -> None:
    set_seed(0)
    device = get_device()
    print(f"device={device}  grokking_connected={GROKKING_CONNECTED} ({grokking_source()})")

    length, max_length = 200, 70
    tok = KmerTokenizer(k=K)
    ds = make_synthetic_enhancer_dataset(n=3000, length=length, seed=0)
    data = BendDataset(ds.sequences, ds.labels, tok, max_length=max_length)
    tr, va, te = train_val_test_split(len(data), seed=0)
    mk = lambda idx, bs, sh: DataLoader(Subset(data, idx.tolist()), batch_size=bs, shuffle=sh, collate_fn=collate)

    section("Stage 1: fine-tune tiny BERT on synthetic enhancers")
    model = build_tiny_classifier(
        num_labels=2, vocab_size=tok.vocab_size, n_layers=6, n_heads=6,
        hidden_size=192, intermediate_size=384, max_position_embeddings=max_length + 4,
    )
    train(model, mk(tr, 32, True), mk(va, 64, False), device, TrainConfig(epochs=5, log_every=10_000, patience=3))
    test_metrics = evaluate(model, mk(te, 64, False), device, "enhancer")
    print(f"TEST enhancer metrics: {test_metrics}")

    section("Stages 2-5: full interpretability stack")
    res = run_all_interp(
        model, tok, device, ds.sequences, np.array(ds.labels),
        max_length=max_length, k=K, motif_positions=ds.motif_positions, tag="",
    )

    print(f"attention entropy range: {tuple(round(x, 3) for x in res['entropy_range'])}")
    print("5 most specialized heads (layer, head, entropy):")
    for l, h, e in res["specialized_heads"]:
        print(f"  L{l}H{h}: entropy={e:.3f} center_bias={res['center_bias'][l, h]:+.3f}")
    print("top-5 causal (layer, token, Delta logit):")
    for l, p, d in res["patching"]["top_sites"]:
        print(f"  L{l} token{p} Delta={d:+.4f}")
    print(f"causal vs planted motif: {res['patching'].get('near_motif')}")
    print("probing (best metric @ layer):")
    for name, per_layer in res["probing"].items():
        metric = "R2" if name == "gc" else "acc"
        print(f"  {name:5s} ({metric}): best={max(per_layer):.3f} @ L{int(np.argmax(per_layer))}  curve={[round(x, 2) for x in per_layer]}")
    print("top motif per specialized head (with DECOY control):")
    for hd, top in res["motif_correlations"].items():
        decoy = dict(top).get("DECOY", float("nan"))
        print(f"  L{hd[0]}H{hd[1]}: top={top[0][0]} r={top[0][1]:+.3f} (DECOY r={decoy:+.3f})")

    summary = {
        "test_metrics": test_metrics,
        "note": "synthetic data, not BEND",
        "specialized_heads": res["specialized_heads"],
        "top_causal_sites": res["patching"]["top_sites"],
        "causal_near_motif": res["patching"].get("near_motif"),
        "probing": {k: [float(x) for x in v] for k, v in res["probing"].items()},
        "motif_correlations": {f"L{l}H{h}": top for (l, h), top in res["motif_correlations"].items()},
    }
    out = CACHE / "synthetic_demo_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=float))
    section("done")
    print(f"summary: {out}\nfigures + arrays: results/figures/")


if __name__ == "__main__":
    main()
