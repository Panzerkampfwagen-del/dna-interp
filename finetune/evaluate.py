"""Evaluation metrics: MCC and AUROC for enhancer, mean AUROC for TF binding.

Metrics are computed from raw logits so the same code serves the training loop
(validation metric for early stopping) and final test reporting.
"""

from __future__ import annotations

import numpy as np
import torch

from metrics import matthews_corrcoef, roc_auc


def enhancer_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    """Binary enhancer metrics from 2-logit outputs."""
    probs = _softmax(logits)[:, 1]
    preds = logits.argmax(axis=1)
    return {
        "mcc": matthews_corrcoef(labels, preds),
        "auroc": roc_auc(labels, probs),
        "accuracy": float((preds == labels).mean()),
    }


def tf_binding_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    """Mean AUROC across TF columns; columns with one class are skipped."""
    probs = 1.0 / (1.0 + np.exp(-logits))
    aurocs = []
    for j in range(labels.shape[1]):
        if len(np.unique(labels[:, j])) == 2:
            aurocs.append(roc_auc(labels[:, j], probs[:, j]))
    return {"mean_auroc": float(np.mean(aurocs)) if aurocs else float("nan"), "n_scored": len(aurocs)}


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@torch.no_grad()
def collect_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader and return (logits, labels) as numpy arrays."""
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        logits, _ = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            cache_activations=False,
        )
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(batch["label"].cpu().numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def evaluate(model, loader, device, task: str = "enhancer") -> dict:
    """Compute the task's metrics over a loader."""
    logits, labels = collect_logits(model, loader, device)
    if task == "enhancer":
        return enhancer_metrics(logits, labels)
    return tf_binding_metrics(logits, labels)
