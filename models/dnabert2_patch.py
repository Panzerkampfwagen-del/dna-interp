"""Make DNABERT-2 runnable and introspectable on this stack.

DNABERT-2's remote code hard-codes a bundled Triton flash-attention that (a) is
incompatible with current Triton (`tl.dot(..., trans_b=...)` was removed) and
(b) never returns attention weights. Its own self-attention already has a pure
PyTorch fallback that runs when the module global `flash_attn_qkvpacked_func` is
None, computing explicit softmax attention. We force that path, then capture the
attention probabilities with a pre-hook that recomputes them from each module's
own Wqkv weights and ALiBi bias, so the attention analyses work unchanged.
"""

from __future__ import annotations

import sys

import torch


def _bert_layers_modules():
    return [m for name, m in sys.modules.items() if name.endswith("bert_layers") and hasattr(m, "BertUnpadSelfAttention")]


def enable_eager_attention() -> bool:
    """Force DNABERT-2's PyTorch attention path. Returns True if patched."""
    patched = False
    for mod in _bert_layers_modules():
        mod.flash_attn_qkvpacked_func = None
        patched = True
    return patched


def _self_attention_modules(model):
    from_mods = _bert_layers_modules()
    cls = from_mods[0].BertUnpadSelfAttention if from_mods else None
    if cls is None:
        return []
    return [m for m in model.modules() if isinstance(m, cls)]


class AttentionCapture:
    """Context manager capturing per-layer attention probs [n_layers, B, H, S, S].

    A forward pre-hook on each self-attention module recomputes attention from the
    module's Wqkv weights and the ALiBi bias it is about to use, matching the
    model's own PyTorch path exactly without altering its output.
    """

    def __init__(self, model):
        self.model = model
        self.modules = _self_attention_modules(model)
        self.handles: list = []
        self.attentions: dict[int, torch.Tensor] = {}
        from_mods = _bert_layers_modules()
        self._pad_input = from_mods[0].pad_input
        self._rearrange = from_mods[0].rearrange

    def _hook(self, idx):
        import math

        def pre_hook(module, args):
            hidden_states, cu_seqlens, max_seqlen, indices, attn_mask, bias = args[:6]
            qkv = module.Wqkv(hidden_states)
            qkv = self._pad_input(qkv, indices, cu_seqlens.shape[0] - 1, max_seqlen)
            qkv = self._rearrange(qkv, "b s (t h d) -> b s t h d", t=3, h=module.num_attention_heads)
            q = qkv[:, :, 0].permute(0, 2, 1, 3)
            k = qkv[:, :, 1].permute(0, 2, 3, 1)
            scores = torch.matmul(q, k) / math.sqrt(module.attention_head_size) + bias
            self.attentions[idx] = torch.softmax(scores, dim=-1).detach()

        return pre_hook

    def __enter__(self):
        for i, m in enumerate(self.modules):
            self.handles.append(m.register_forward_pre_hook(self._hook(i)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def stacked(self) -> torch.Tensor:
        return torch.stack([self.attentions[i] for i in sorted(self.attentions)], dim=0)


class ResidualCapture:
    """Capture the re-padded residual stream [n_layers+1, B, S, H].

    DNABERT-2 stores per-layer hidden states unpadded; each BertLayer receives
    `indices`, so a forward hook re-pads its output back to [B, S, H]. The
    embedding output (already padded) is layer 0.
    """

    def __init__(self, model):
        self.embeddings = model.embeddings
        self.layers = list(model.encoder.layer)
        self._pad_input = _bert_layers_modules()[0].pad_input
        self.handles: list = []
        self.embed: torch.Tensor | None = None
        self.layer_out: dict[int, torch.Tensor] = {}

    def _emb_hook(self, module, args, output):
        self.embed = output.detach()

    def _layer_hook(self, idx):
        def hook(module, args, output):
            cu_seqlens, seqlen, indices = args[1], args[2], args[4]
            batch = cu_seqlens.shape[0] - 1
            self.layer_out[idx] = self._pad_input(output.detach(), indices, batch, seqlen)

        return hook

    def __enter__(self):
        self.handles.append(self.embeddings.register_forward_hook(self._emb_hook))
        for i, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_hook(self._layer_hook(i)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def stacked(self) -> torch.Tensor:
        layers = [self.layer_out[i] for i in sorted(self.layer_out)]
        return torch.stack([self.embed] + layers, dim=0)
