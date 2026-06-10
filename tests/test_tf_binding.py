import numpy as np
import torch
from torch.utils.data import DataLoader

from data.bend_loader import BendDataset, collate, make_synthetic_tf_binding_dataset
from finetune.evaluate import tf_binding_metrics


def test_generator_shapes_and_labels():
    ds = make_synthetic_tf_binding_dataset(n=300, length=101, seed=0)
    assert ds.labels.shape == (300, len(ds.tf_names))
    assert set(np.unique(ds.labels)).issubset({0.0, 1.0})
    # a planted TF is marked present and its motif sits at the recorded position
    from data.bend_loader import PLANT_MOTIFS

    i = next(k for k in range(300) if ds.motif_positions[k])
    for name, start in ds.motif_positions[i].items():
        w = len(PLANT_MOTIFS[name])
        assert ds.sequences[i][start : start + w] == PLANT_MOTIFS[name]
        assert ds.labels[i, ds.tf_names.index(name)] == 1.0


def test_multilabel_dataset_item(tokenizer):
    ds = make_synthetic_tf_binding_dataset(n=20, length=101, seed=0)
    data = BendDataset(ds.sequences, list(ds.labels), tokenizer, max_length=40, multilabel=True)
    item = data[0]
    assert item["label"].dtype == torch.float32
    assert item["label"].shape == (len(ds.tf_names),)
    batch = collate([data[i] for i in range(4)])
    assert batch["label"].shape == (4, len(ds.tf_names))


def test_tf_binding_metrics_extremes():
    labels = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=float)
    perfect = np.where(labels == 1, 5.0, -5.0)
    assert tf_binding_metrics(perfect, labels)["mean_auroc"] == 1.0
    assert tf_binding_metrics(-perfect, labels)["mean_auroc"] == 0.0


def test_tf_binding_training_plumbing():
    from config import get_device, set_seed
    from finetune.trainer import TrainConfig, train
    from finetune.evaluate import evaluate
    from data.bend_loader import KmerTokenizer, train_val_test_split
    from models.dna_lm import build_tiny_classifier
    from torch.utils.data import Subset

    set_seed(0)
    device = get_device()
    tok = KmerTokenizer(k=3)
    ds = make_synthetic_tf_binding_dataset(n=400, length=101, seed=0)
    data = BendDataset(ds.sequences, list(ds.labels), tok, max_length=40, multilabel=True)
    tr, va, te = train_val_test_split(len(data), seed=0)
    mk = lambda idx, bs, sh: DataLoader(Subset(data, idx.tolist()), batch_size=bs, shuffle=sh, collate_fn=collate)
    clf = build_tiny_classifier(num_labels=len(ds.tf_names), vocab_size=tok.vocab_size, n_layers=2, n_heads=2, hidden_size=64, max_position_embeddings=44)
    train(clf, mk(tr, 32, True), mk(va, 64, False), device, TrainConfig(task="tf_binding", epochs=1, log_every=9999, patience=3))
    m = evaluate(clf, mk(te, 64, False), device, "tf_binding")
    assert 0.0 <= m["mean_auroc"] <= 1.0
    assert m["n_scored"] == len(ds.tf_names)
