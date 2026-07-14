"""DNAClassifier: a HuggingFace encoder plus a classification head, with an
activation cache and an activation-patching hook.

The cache and patch API mirror the grokking project's `Model.run_with_cache` /
`register_hook` contract (see grokking_bridge.py): a forward returns logits plus
a dict of activations, and a patch dict replaces an activation in-place during
the forward. Because the backbone is a HuggingFace encoder, residual stream and
attention patterns come from `output_hidden_states` / `output_attentions`, and
patching is done with a temporary PyTorch forward hook on the target layer.

The wrapper is architecture-agnostic: it works on DNABERT-2 in real mode and on
a small local BERT in offline tests, with no code change.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _find_layer_list(base: nn.Module, n_layers: int) -> nn.ModuleList:
    """Locate the encoder's ModuleList of transformer layers.

    Works for BERT (`encoder.layer`) and custom variants by matching a
    ModuleList whose length equals the configured number of layers.
    """
    for _, module in base.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) == n_layers:
            return module
    raise RuntimeError("could not locate the transformer layer ModuleList")


class DNAClassifier(nn.Module):
    """Encoder backbone + linear classification head with activation caching.

    Parameters
        base_model: a HuggingFace encoder returning `last_hidden_state` and
            supporting `output_hidden_states` / `output_attentions`.
        num_labels: 2 for enhancer (cross-entropy), 690 for TF binding (BCE).
        cache_activations: default for whether forward returns a cache.
        pooling: "mean" (masked mean over tokens) or "cls" (first token).
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_labels: int,
        cache_activations: bool = False,
        pooling: str = "mean",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.base = base_model
        self.config = base_model.config
        self.num_labels = num_labels
        self.cache_activations = cache_activations
        self.pooling = pooling
        self.n_layers = int(self.config.num_hidden_layers)
        self.n_heads = int(self.config.num_attention_heads)
        self.hidden_size = int(self.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        self._layers = _find_layer_list(self.base, self.n_layers)
        self.problem_type = (
            "multi_label_classification" if num_labels > 2 else "single_label_classification"
        )
        # Remote-code models (e.g. DNABERT-2) return a tuple and unpad internally;
        # they need a dedicated cache path instead of output_hidden_states.
        self._remote = "transformers_modules" in type(self.base).__module__

    def enable_efficiency(self, bf16: bool = True) -> DNAClassifier:
        """Gradient checkpointing plus BF16 to fit DNABERT-2 in 4 GB VRAM.

        The whole module is cast (not just the backbone) so the classification
        head matches the backbone dtype; the trainer upcasts logits for the loss.
        """
        if hasattr(self.base, "gradient_checkpointing_enable"):
            self.base.gradient_checkpointing_enable()
        if bf16:
            self.to(torch.bfloat16)
        return self

    def _pool(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def _patch_hook(self, position: int, value: torch.Tensor):
        def hook(module, inputs, output):
            is_tuple = isinstance(output, tuple)
            hs = (output[0] if is_tuple else output).clone()
            hs[:, position, :] = value.to(hs.dtype).to(hs.device)
            if is_tuple:
                return (hs,) + tuple(output[1:])
            return hs

        return hook

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        patch: dict | None = None,
        cache_activations: bool | None = None,
    ) -> tuple[torch.Tensor, dict | None]:
        """Run the model. `patch` is {layer:int, position:int, value:tensor[H]}
        and replaces the residual stream leaving encoder layer `layer`.

        Returns (logits, cache). cache is None unless caching is requested; when
        present it holds:
            resid:        [n_layers+1, B, S, H]  residual stream (index 0 = embed)
            attn_pattern: [n_layers, B, heads, S, S]
            attn_out:     [n_layers, B, S, H]    (best effort, may be absent)
            mlp_pre:      [n_layers, B, S, d_ff] (best effort, may be absent)
        """
        want_cache = self.cache_activations if cache_activations is None else cache_activations
        if self._remote:
            return self._forward_remote(input_ids, attention_mask, patch, want_cache)

        need_attn = want_cache

        handles = []
        if patch is not None:
            layer = int(patch["layer"])
            target = self._layers[layer]
            handles.append(target.register_forward_hook(self._patch_hook(int(patch["position"]), patch["value"])))

        side: dict[str, list] = {"attn_out": [None] * self.n_layers, "mlp_pre": [None] * self.n_layers}
        if want_cache:
            handles.extend(self._register_side_hooks(side))

        try:
            outputs = self.base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=need_attn,
                return_dict=True,
            )
        finally:
            for h in handles:
                h.remove()

        last_hidden = outputs.last_hidden_state
        logits = self.classifier(self.dropout(self._pool(last_hidden, attention_mask)))

        cache = None
        if want_cache:
            cache = {"resid": torch.stack(list(outputs.hidden_states), dim=0)}
            if outputs.attentions is not None:
                cache["attn_pattern"] = torch.stack(list(outputs.attentions), dim=0)
            if all(t is not None for t in side["attn_out"]):
                cache["attn_out"] = torch.stack(side["attn_out"], dim=0)
            if all(t is not None for t in side["mlp_pre"]):
                cache["mlp_pre"] = torch.stack(side["mlp_pre"], dim=0)
        return logits, cache

    def _forward_remote(self, input_ids, attention_mask, patch, want_cache):
        """Forward path for DNABERT-2-style remote models (tuple return, unpadded).

        Residual stream and attention are captured with hooks (models/dnabert2_patch)
        rather than output_hidden_states/output_attentions, which these models do
        not support.
        """
        from contextlib import ExitStack

        from models.dnabert2_patch import AttentionCapture, ResidualCapture

        if patch is not None:
            raise NotImplementedError(
                "the forward(patch=...) dict path is not wired for DNABERT-2's "
                "unpadded internals; use interp.test_failure_patching, which remaps "
                "padded positions to packed tokens via each layer's `indices`."
            )

        with ExitStack() as stack:
            res_cap = stack.enter_context(ResidualCapture(self.base)) if want_cache else None
            attn_cap = stack.enter_context(AttentionCapture(self.base)) if want_cache else None
            outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            logits = self.classifier(self.dropout(self._pool(last_hidden, attention_mask)))
            cache = None
            if want_cache:
                cache = {"resid": res_cap.stacked(), "attn_pattern": attn_cap.stacked()}
        return logits, cache

    def _register_side_hooks(self, side: dict) -> list:
        """Best-effort hooks for per-layer attention output and MLP hidden.

        Skipped silently for backbones whose submodules are named differently;
        the core resid/attn cache does not depend on these.
        """
        handles = []
        for i, layer in enumerate(self._layers):
            attn = getattr(layer, "attention", None)
            if attn is not None:
                handles.append(attn.register_forward_hook(self._collect(side["attn_out"], i)))
            inter = getattr(layer, "intermediate", None)
            if inter is not None:
                handles.append(inter.register_forward_hook(self._collect(side["mlp_pre"], i)))
        return handles

    @staticmethod
    def _collect(store: list, idx: int):
        def hook(module, inputs, output):
            t = output[0] if isinstance(output, tuple) else output
            store[idx] = t.detach()

        return hook

    @torch.no_grad()
    def patch_activation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layer: int,
        position: int,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience wrapper: run with a single residual-stream patch, return logits."""
        logits, _ = self.forward(
            input_ids,
            attention_mask,
            patch={"layer": layer, "position": position, "value": value},
        )
        return logits


def build_tiny_classifier(
    num_labels: int = 2,
    vocab_size: int = 68,
    n_layers: int = 4,
    n_heads: int = 4,
    hidden_size: int = 128,
    intermediate_size: int = 256,
    max_position_embeddings: int = 256,
    cache_activations: bool = False,
    pooling: str = "mean",
    seed: int = 0,
) -> DNAClassifier:
    """A small local BERT classifier for offline tests and the synthetic demo.

    Same DNAClassifier wrapper as real mode, so any code that works here works
    on DNABERT-2 without modification.
    """
    from transformers import BertConfig, BertModel

    torch.manual_seed(seed)
    cfg = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        attn_implementation="eager",
    )
    base = BertModel(cfg, add_pooling_layer=False)
    return DNAClassifier(base, num_labels, cache_activations=cache_activations, pooling=pooling)


def load_dnabert2_classifier(
    num_labels: int = 2,
    cache_activations: bool = False,
    pooling: str = "mean",
):
    """Load DNABERT-2 with a classification head (real mode, needs cached model).

    Forces the eager attention path so `output_attentions` returns weights, which
    the attention and motif-scan analyses require.
    """
    from transformers import AutoModel

    from config import DNABERT2_ID
    from models.dnabert2_patch import enable_eager_attention

    base = AutoModel.from_pretrained(DNABERT2_ID, trust_remote_code=True)
    enable_eager_attention()  # replace broken Triton flash-attn with the PyTorch path
    return DNAClassifier(base, num_labels, cache_activations=cache_activations, pooling=pooling)
