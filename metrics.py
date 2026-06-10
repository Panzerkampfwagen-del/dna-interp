"""Metrics implemented in numpy so the GPU env needs only numpy + torch.

scikit-learn has no offline build for this env, and these metrics are small and
worth understanding directly: MCC from the confusion matrix, AUROC from the
rank-sum (Mann-Whitney) identity, R^2 from residual and total variance.
"""

from __future__ import annotations

import numpy as np


def accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    return float((np.asarray(labels) == np.asarray(preds)).mean())


def matthews_corrcoef(labels: np.ndarray, preds: np.ndarray) -> float:
    """Binary Matthews correlation coefficient from the confusion matrix."""
    y, p = np.asarray(labels), np.asarray(preds)
    tp = float(np.sum((p == 1) & (y == 1)))
    tn = float(np.sum((p == 0) & (y == 0)))
    fp = float(np.sum((p == 1) & (y == 0)))
    fn = float(np.sum((p == 0) & (y == 1)))
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0.0:
        return 0.0
    return float((tp * tn - fp * fn) / denom)


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties share the mean rank."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    # average tied ranks
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC via the rank-sum identity; nan if only one class is present."""
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = float(np.sum(y == 1))
    n_neg = float(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(s)
    rank_sum_pos = float(np.sum(ranks[y == 1]))
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y, p = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation as Pearson correlation of ranks."""
    rx, ry = _rankdata(x), _rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom == 0.0:
        return 0.0
    return float((rx * ry).sum() / denom)
