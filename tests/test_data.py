import torch

from data.bend_loader import (
    BendDataset,
    KmerTokenizer,
    PLANT_MOTIFS,
    collate,
    make_synthetic_enhancer_dataset,
)


def test_vocab_size():
    tok = KmerTokenizer(k=3)
    assert tok.vocab_size == 4 + 4 ** 3  # specials plus all 3-mers


def test_tokenize_shapes_and_specials(tokenizer):
    enc = tokenizer(["ACGTACGTAC"], padding="max_length", max_length=12, return_tensors="pt")
    assert enc["input_ids"].shape == (1, 12)
    assert enc["attention_mask"].shape == (1, 12)
    assert enc["input_ids"][0, 0].item() == tokenizer.cls_token_id


def test_offset_mapping(tokenizer):
    enc = tokenizer(["ACGTAC"], padding="max_length", max_length=8, return_offsets_mapping=True, return_tensors="pt")
    off = enc["offset_mapping"]
    assert off.shape == (1, 8, 2)
    assert tuple(off[0, 0].tolist()) == (0, 0)  # CLS has no span
    assert tuple(off[0, 1].tolist()) == (0, 3)  # first 3-mer covers bases 0..3


def test_decode_roundtrips_bases(tokenizer):
    seq = "ACGTTTGGGCCC"
    enc = tokenizer([seq], padding=False, return_tensors="pt")
    assert tokenizer.decode(enc["input_ids"][0]) == seq


def test_synthetic_balance_and_planting():
    ds = make_synthetic_enhancer_dataset(n=200, length=120, seed=1)
    assert sum(ds.labels) == 100  # balanced
    pos = ds.labels.index(1)
    for name, start in ds.motif_positions[pos].items():
        w = len(PLANT_MOTIFS[name])
        assert ds.sequences[pos][start : start + w] == PLANT_MOTIFS[name]


def test_dataset_item_and_collate(tokenizer, synthetic, max_length):
    data = BendDataset(synthetic.sequences, synthetic.labels, tokenizer, max_length=max_length)
    item = data[0]
    assert set(item.keys()) == {"input_ids", "attention_mask", "label", "sequence", "position_ids"}
    assert item["input_ids"].shape == (max_length,)
    assert item["position_ids"].shape == (max_length,)
    assert torch.equal(item["position_ids"], torch.arange(max_length))
    batch = collate([data[i] for i in range(4)])
    assert batch["input_ids"].shape == (4, max_length)
    assert batch["label"].shape == (4,)
    assert len(batch["sequence"]) == 4
