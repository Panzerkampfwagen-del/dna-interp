"""Real-mode fine-tuning of DNABERT-2 on the BEND enhancer task.

Uses the prompt's recipe: separate backbone/head learning rates, warmup + cosine
schedule, BF16 autocast, gradient checkpointing for 4 GB VRAM, early stopping on
validation MCC. Falls back to the synthetic dataset with a clear message if the
real assets are not available, so the same path is exercisable offline.

    python scripts/finetune_enhancer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_device, set_seed  # noqa: E402
from data.bend_loader import (  # noqa: E402
    BendDataset,
    collate,
    load_bend_enhancer,
    load_dnabert2_tokenizer,
    make_synthetic_enhancer_dataset,
    train_val_test_split,
)
from finetune.evaluate import evaluate  # noqa: E402
from finetune.trainer import TrainConfig, train  # noqa: E402
from models.dna_lm import build_tiny_classifier, load_dnabert2_classifier  # noqa: E402


def load_real():
    tok = load_dnabert2_tokenizer()
    tr_seq, tr_lab = load_bend_enhancer("train")
    va_seq, va_lab = load_bend_enhancer("valid")
    te_seq, te_lab = load_bend_enhancer("test")
    model = load_dnabert2_classifier(num_labels=2).enable_efficiency(bf16=True)
    return tok, model, (tr_seq, tr_lab), (va_seq, va_lab), (te_seq, te_lab)


def main() -> None:
    set_seed(0)
    device = get_device()
    try:
        tok, model, train_d, val_d, test_d = load_real()
        max_length, mode = 128, "real (DNABERT-2 + BEND)"
    except Exception as e:  # noqa: BLE001
        print(f"Real assets unavailable ({e}). Falling back to synthetic data.")
        from data.bend_loader import KmerTokenizer

        tok = KmerTokenizer(k=3)
        ds = make_synthetic_enhancer_dataset(n=3000, length=200, seed=0)
        tr, va, te = train_val_test_split(len(ds.sequences), seed=0)
        pick = lambda idx: ([ds.sequences[i] for i in idx], [ds.labels[i] for i in idx])
        train_d, val_d, test_d = pick(tr), pick(va), pick(te)
        model = build_tiny_classifier(num_labels=2, vocab_size=tok.vocab_size, n_layers=6, n_heads=6, hidden_size=192, intermediate_size=384, max_position_embeddings=74)
        max_length, mode = 70, "synthetic fallback"

    print(f"mode: {mode}")
    mk = lambda d, bs, sh: DataLoader(
        BendDataset(d[0], d[1], tok, max_length=max_length), batch_size=bs, shuffle=sh, collate_fn=collate
    )
    cfg = TrainConfig(task="enhancer", epochs=5, lr_backbone=2e-5, lr_head=1e-3, accum_steps=1, log_every=50, patience=2)
    train(model, mk(train_d, 32, True), mk(val_d, 64, False), device, cfg)
    print(f"TEST: {evaluate(model, mk(test_d, 64, False), device, 'enhancer')}")


if __name__ == "__main__":
    main()
