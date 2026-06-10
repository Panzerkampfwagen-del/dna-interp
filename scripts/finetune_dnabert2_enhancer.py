"""Fine-tune real DNABERT-2 on the enhancer task (NT downstream 'enhancers').

BEND's repo is unavailable (404), so this uses the Nucleotide Transformer
downstream 'enhancers' task: real 200 bp sequences, binary enhancer label,
balanced (14968 train / 400 test). Same task as BEND Task 1, accessible source.

fp32 weights with BF16 autocast (more stable than casting weights to BF16); 200 bp
is ~42 BPE tokens so max_length=48 keeps it within ~3 GB. Saves the best
checkpoint to results/checkpoints/dnabert2_enhancer_best.pt for the interp run.

    PY=/home/aryan/anaconda3/envs/tinyinfer-gpu/bin/python
    $PY scripts/finetune_dnabert2_enhancer.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CACHE, CHECKPOINTS, get_device, set_seed  # noqa: E402
from data.bend_loader import (  # noqa: E402
    BendDataset,
    collate,
    load_dnabert2_tokenizer,  # noqa: E402
    load_nt_enhancer,
    train_val_test_split,
)
from finetune.evaluate import evaluate  # noqa: E402
from finetune.trainer import TrainConfig, load_checkpoint, train  # noqa: E402
from models.dna_lm import load_dnabert2_classifier  # noqa: E402

MAX_LEN = 48


def main() -> None:
    set_seed(0)
    device = get_device()

    tok = load_dnabert2_tokenizer()
    tr_seq, tr_lab = load_nt_enhancer("train")
    te_seq, te_lab = load_nt_enhancer("test")
    print(f"train={len(tr_seq)} test={len(te_seq)} (real NT enhancers, 200bp)")

    train_data = BendDataset(tr_seq, tr_lab, tok, max_length=MAX_LEN)
    test_data = BendDataset(te_seq, te_lab, tok, max_length=MAX_LEN)
    tr_idx, va_idx, _ = train_val_test_split(len(train_data), val_frac=0.1, test_frac=0.0, seed=0)
    mk = lambda d, idx, bs, sh: DataLoader(Subset(d, idx.tolist()), batch_size=bs, shuffle=sh, collate_fn=collate)
    train_loader = mk(train_data, tr_idx, 32, True)
    val_loader = mk(train_data, va_idx, 64, False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=collate)

    model = load_dnabert2_classifier(num_labels=2)
    cfg = TrainConfig(
        task="enhancer", epochs=5, lr_backbone=2e-5, lr_head=1e-3,
        warmup_frac=0.1, accum_steps=1, log_every=50, patience=2,
        use_amp=True, out_name="dnabert2_enhancer_best.pt",
    )
    train(model, train_loader, val_loader, device, cfg)

    load_checkpoint(model, CHECKPOINTS / cfg.out_name, device)
    test_metrics = evaluate(model, test_loader, device, "enhancer")
    print(f"\nTEST (best checkpoint): {test_metrics}")
    (CACHE / "dnabert2_enhancer_test.json").write_text(json.dumps(test_metrics, indent=2, default=float))


if __name__ == "__main__":
    main()
