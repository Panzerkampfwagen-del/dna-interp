import numpy as np
import torch

from data.bend_loader import make_synthetic_enhancer_dataset
from interp.test_failure_patching import (
    build_matched_pairs,
    gc_content,
    patch_pair,
    run_patching_experiment,
    summarize_results,
)


def test_self_patch_returns_zero(model, tokenizer, synthetic, max_length):
    """Patching a sequence with itself must leave the logit unchanged."""
    layers = list(range(model.n_layers))
    delta = patch_pair(model, tokenizer, synthetic.sequences[0], synthetic.sequences[0], layers, max_length=max_length)
    assert np.abs(delta).max() < 1e-3


def test_directional_patch_recovers_signal(tokenizer):
    """A trained model: patching an enhancer donor into a non-enhancer raises the
    enhancer logit somewhere (the planted signal is causally transferable)."""
    from finetune.trainer import TrainConfig, train
    from models.dna_lm import build_tiny_classifier
    from data.bend_loader import BendDataset, collate, train_val_test_split
    from torch.utils.data import DataLoader, Subset

    ds = make_synthetic_enhancer_dataset(n=800, length=90, seed=0)
    data = BendDataset(ds.sequences, ds.labels, tokenizer, max_length=36)
    tr, va, _ = train_val_test_split(len(data), seed=0)
    mk = lambda idx, sh: DataLoader(Subset(data, idx.tolist()), batch_size=32, shuffle=sh, collate_fn=collate)
    clf = build_tiny_classifier(num_labels=2, vocab_size=tokenizer.vocab_size, n_layers=3, n_heads=4,
                                hidden_size=64, intermediate_size=128, cache_activations=True)
    train(clf, mk(tr, True), mk(va, False), torch.device("cpu"), TrainConfig(epochs=5, log_every=9999, patience=5))
    clf.eval()

    pos = [s for s, l in zip(ds.sequences, ds.labels) if l == 1][:8]   # enhancer donors
    neg = [s for s, l in zip(ds.sequences, ds.labels) if l == 0][:16]  # non-enhancer clean + control donors
    delta_dir = run_patching_experiment(clf, tokenizer, list(zip(neg[:8], pos)), max_length=36).mean(axis=0)
    delta_ctrl = run_patching_experiment(clf, tokenizer, list(zip(neg[:8], neg[8:16])), max_length=36).mean(axis=0)
    # an enhancer donor transfers more enhancer signal than a non-enhancer control donor
    assert delta_dir.max() > 0
    assert delta_dir.max() > delta_ctrl.max()


def test_build_matched_pairs_are_gc_matched():
    rng = np.random.default_rng(0)

    def seq(gc, n=600):
        return "".join(rng.choice(list("ACGT"), n, p=[(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]))

    fns = [seq(g) for g in (0.35, 0.50, 0.65)]
    tps_close = [seq(g) for g in (0.35, 0.50, 0.65)]
    tp_far = seq(0.90)
    sequences = fns + tps_close + [tp_far]
    labels = [1] * len(sequences)
    probs = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9])  # fns are FN, rest are TP

    pairs = build_matched_pairs(None, None, sequences, labels, n_pairs=10,
                                confidence_threshold=0.3, gc_bin_width=0.05, probs=probs)
    assert len(pairs) == 3
    for fn_seq, tp_seq in pairs:
        assert abs(gc_content(fn_seq) - gc_content(tp_seq)) <= 0.05
        assert gc_content(tp_seq) < 0.8  # never matched to the far-GC TP


def test_run_patching_experiment_shape(model, tokenizer, synthetic, max_length):
    pairs = [(synthetic.sequences[0], synthetic.sequences[1]),
             (synthetic.sequences[2], synthetic.sequences[3])]
    out = run_patching_experiment(model, tokenizer, pairs, max_length=max_length)
    assert out.shape == (2, model.n_layers, max_length)
    summ = summarize_results(out)
    assert summ["mean"].shape == (model.n_layers, max_length)
    assert len(summ["top_overall"]) == 5
