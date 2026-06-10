"""Offline validation of the interpretability stack on the TF-binding task.

Trains a small multi-label classifier on synthetic sequences with planted TF
motifs, then for each TF runs activation patching targeting that TF's logit and
checks that the causal token positions land on that TF's planted motif. This is
the Task-2 analogue of run_synthetic_demo.py and reuses interp.pipeline.

    /home/aryan/anaconda3/envs/tinyinfer-gpu/bin/python scripts/run_tf_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CACHE, get_device, set_seed  # noqa: E402
from data.bend_loader import (  # noqa: E402
    BendDataset,
    KmerTokenizer,
    collate,
    make_synthetic_tf_binding_dataset,
    train_val_test_split,
)
from finetune.evaluate import evaluate  # noqa: E402
from finetune.trainer import TrainConfig, train  # noqa: E402
from interp.patching import averaged_patch_experiment, top_causal_sites  # noqa: E402
from models.dna_lm import build_tiny_classifier  # noqa: E402
from visualize.plot_patching import plot_patching_heatmap  # noqa: E402

K = 3


def section(title: str) -> None:
    print(f"\n{'=' * 4} {title} {'=' * 4}")


def main() -> None:
    set_seed(0)
    device = get_device()
    length, max_length = 101, 40
    tok = KmerTokenizer(k=K)
    ds = make_synthetic_tf_binding_dataset(n=3000, length=length, seed=0)
    print(f"TFs: {ds.tf_names}  mean presence: {ds.labels.mean(0).round(2)}")

    data = BendDataset(ds.sequences, list(ds.labels), tok, max_length=max_length, multilabel=True)
    tr, va, te = train_val_test_split(len(data), seed=0)
    mk = lambda idx, bs, sh: DataLoader(Subset(data, idx.tolist()), batch_size=bs, shuffle=sh, collate_fn=collate)

    section("Stage 1: fine-tune multi-label TF classifier")
    model = build_tiny_classifier(
        num_labels=len(ds.tf_names), vocab_size=tok.vocab_size, n_layers=6, n_heads=6,
        hidden_size=192, intermediate_size=384, max_position_embeddings=max_length + 4,
    )
    train(model, mk(tr, 32, True), mk(va, 64, False), device, TrainConfig(task="tf_binding", epochs=6, log_every=10_000, patience=4))
    print(f"TEST: {evaluate(model, mk(te, 64, False), device, 'tf_binding')}")

    section("Stage 2: per-TF activation patching localizes to that TF's motif")
    layers = list(range(model.n_layers))
    report: dict = {}
    for t, name in enumerate(ds.tf_names):
        pos = [i for i in range(len(ds.sequences)) if ds.labels[i, t] == 1 and name in ds.motif_positions[i]]
        neg = [i for i in range(len(ds.sequences)) if ds.labels[i, t] == 0]
        donors = [ds.sequences[i] for i in pos[:20]]
        cleans = [ds.sequences[i] for i in neg[:20]]
        heat = averaged_patch_experiment(model, tok, cleans, donors, layers, device, max_length, target_class=t)
        top = top_causal_sites(heat, layers, top_k=5)
        plot_patching_heatmap(heat, layers, stem=f"tf_patching_{name}")
        motif_tokens = sorted({ds.motif_positions[i][name] // K + 1 for i in pos[:20]})
        causal_tokens = sorted({p for _, p, _ in top})
        near = [c for c in causal_tokens if any(abs(c - m) <= 1 for m in motif_tokens)]
        hit = len(near) / max(len(causal_tokens), 1)
        print(f"  {name}: causal tokens {causal_tokens} | planted {motif_tokens} | on-motif {len(near)}/{len(causal_tokens)} ({hit:.0%})")
        report[name] = {"causal_tokens": causal_tokens, "motif_tokens": motif_tokens, "near": near}

    out = CACHE / "tf_demo_summary.json"
    out.write_text(json.dumps(report, indent=2, default=float))
    section("done")
    print(f"summary: {out}\nfigures: results/figures/tf_patching_*.npz")


if __name__ == "__main__":
    main()
