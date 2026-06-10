"""Shared fixtures: a tokenizer, a small untrained classifier, synthetic data.

Tests use an untrained tiny model where only shapes, invariants, and ranges
matter, which keeps the suite fast and fully offline.
"""

import pytest

from data.bend_loader import KmerTokenizer, make_synthetic_enhancer_dataset
from models.dna_lm import build_tiny_classifier


@pytest.fixture(scope="session")
def tokenizer():
    return KmerTokenizer(k=3)


@pytest.fixture(scope="session")
def synthetic():
    return make_synthetic_enhancer_dataset(n=120, length=90, seed=0)


@pytest.fixture(scope="session")
def model(tokenizer):
    clf = build_tiny_classifier(
        num_labels=2,
        vocab_size=tokenizer.vocab_size,
        n_layers=3,
        n_heads=4,
        hidden_size=64,
        intermediate_size=128,
        cache_activations=True,
    )
    return clf.eval()  # eval by default: dropout off, the right mode for inspection


@pytest.fixture(scope="session")
def max_length():
    return 36
