import torch

from data.bend_loader import BendDataset, collate


def _batch(model_tok, synthetic, max_length, n=4):
    data = BendDataset(synthetic.sequences, synthetic.labels, model_tok, max_length=max_length)
    return collate([data[i] for i in range(n)])


def test_forward_output_shape(model, tokenizer, synthetic, max_length):
    b = _batch(tokenizer, synthetic, max_length)
    logits, _ = model(b["input_ids"], b["attention_mask"], cache_activations=False)
    assert logits.shape == (4, 2)


def test_cache_shapes(model, tokenizer, synthetic, max_length):
    b = _batch(tokenizer, synthetic, max_length)
    _, cache = model(b["input_ids"], b["attention_mask"], cache_activations=True)
    L, H = model.n_layers, model.n_heads
    S, Hd = max_length, model.hidden_size
    assert cache["resid"].shape == (L + 1, 4, S, Hd)
    assert cache["attn_pattern"].shape == (L, 4, H, S, S)


def test_attention_normalized(model, tokenizer, synthetic, max_length):
    b = _batch(tokenizer, synthetic, max_length)
    _, cache = model(b["input_ids"], b["attention_mask"], cache_activations=True)
    sums = cache["attn_pattern"].sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_efficiency_path_bf16(tokenizer, synthetic, max_length):
    from models.dna_lm import build_tiny_classifier

    clf = build_tiny_classifier(num_labels=2, vocab_size=tokenizer.vocab_size, n_layers=2, n_heads=2, hidden_size=32)
    clf.enable_efficiency(bf16=True)  # casts whole module, enables grad checkpointing
    b = _batch(tokenizer, synthetic, max_length)
    logits, _ = clf(b["input_ids"], b["attention_mask"], cache_activations=False)
    assert logits.dtype == torch.bfloat16
    loss = torch.nn.functional.cross_entropy(logits.float(), b["label"])
    loss.backward()
    assert clf.classifier.weight.grad is not None


def test_gradients_flow(model, tokenizer, synthetic, max_length):
    b = _batch(tokenizer, synthetic, max_length)
    logits, _ = model(b["input_ids"], b["attention_mask"], cache_activations=False)
    loss = torch.nn.functional.cross_entropy(logits, b["label"])
    model.zero_grad()
    loss.backward()
    assert model.classifier.weight.grad is not None
    assert model.classifier.weight.grad.abs().sum() > 0
