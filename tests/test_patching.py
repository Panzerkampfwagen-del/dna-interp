import numpy as np

from interp.patching import patch_activation_experiment, top_causal_sites


def _two_sequences(synthetic):
    pos = synthetic.sequences[synthetic.labels.index(1)]
    neg = synthetic.sequences[synthetic.labels.index(0)]
    return neg, pos


def test_patching_shape(model, tokenizer, synthetic, max_length):
    neg, pos = _two_sequences(synthetic)
    layers = list(range(model.n_layers))
    heat = patch_activation_experiment(model, tokenizer, neg, pos, layers, max_length=max_length)
    assert heat.shape == (len(layers), max_length)


def test_self_patch_is_zero(model, tokenizer, synthetic, max_length):
    neg, _ = _two_sequences(synthetic)
    layers = list(range(model.n_layers))
    heat = patch_activation_experiment(model, tokenizer, neg, neg, layers, max_length=max_length)
    assert np.abs(heat).max() < 1e-4


def test_diff_patch_is_nonzero(model, tokenizer, synthetic, max_length):
    neg, pos = _two_sequences(synthetic)
    layers = list(range(model.n_layers))
    heat = patch_activation_experiment(model, tokenizer, neg, pos, layers, max_length=max_length)
    assert np.abs(heat).max() > 0.0


def test_top_causal_sites(model, tokenizer, synthetic, max_length):
    neg, pos = _two_sequences(synthetic)
    layers = list(range(model.n_layers))
    heat = patch_activation_experiment(model, tokenizer, neg, pos, layers, max_length=max_length)
    top = top_causal_sites(heat, layers, top_k=5)
    assert len(top) == 5
    assert all(L in layers and 0 <= p < max_length for (L, p, _) in top)
