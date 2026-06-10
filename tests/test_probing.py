import numpy as np

from interp.probing import build_property_labels, gc_content, has_tata_center, probe_all_layers
from models.probes import classification_probe, regression_probe


def test_classification_probe_range_and_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 8))
    y = (X[:, 0] > 0).astype(int)
    res = classification_probe(X[:200], y[:200], X[200:], y[200:])
    assert 0.0 <= res["accuracy"] <= 1.0
    assert 0.0 <= res["auroc"] <= 1.0
    assert res["accuracy"] > 0.8  # linearly separable


def test_regression_probe_range_and_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 8))
    y = X[:, 0] * 3 + rng.normal(scale=0.1, size=300)
    res = regression_probe(X[:200], y[:200], X[200:], y[200:])
    assert res["r2"] > 0.8


def test_feature_helpers():
    assert gc_content("GGCC") == 1.0
    assert gc_content("AATT") == 0.0
    centered = "A" * 47 + "TATAAA" + "A" * 47
    assert has_tata_center(centered, window=10) == 1
    assert has_tata_center("TATAAA" + "A" * 100, window=10) == 0


def test_build_property_labels_shapes(synthetic):
    labels = build_property_labels(synthetic.sequences, np.array(synthetic.labels))
    n = len(synthetic.sequences)
    assert labels["gc"].shape == (n,)
    assert labels["cpg"].shape == (n,)
    assert labels["tata"].shape == (n,)
    assert labels["kmer"].shape == (n, 16)


def test_probe_all_layers_returns_valid(model, tokenizer, synthetic, max_length):
    labels = build_property_labels(synthetic.sequences, np.array(synthetic.labels))
    res = probe_all_layers(model, synthetic.sequences, labels, tokenizer, "cpu", max_length=max_length)
    assert set(res.keys()) == {"gc", "cpg", "tata", "kmer"}
    for name, per_layer in res.items():
        assert len(per_layer) == model.n_layers + 1
        for v in per_layer:
            assert np.isfinite(v)
        if name != "gc":  # accuracy metrics live in [0, 1]
            assert all(0.0 <= v <= 1.0 for v in per_layer)
