"""Linear probe primitives used by interp/probing.py.

Probes are simple linear models with L2 regularization, so probe quality
reflects how linearly decodable a property is from a layer rather than probe
capacity. Logistic probes are fit with torch (LBFGS); ridge probes use the
closed-form normal equations. No scikit-learn dependency.
"""

from __future__ import annotations

import numpy as np
import torch

from metrics import accuracy, r2_score, roc_auc


def _standardize(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


def _fit_logistic(X: np.ndarray, y: np.ndarray, l2: float, epochs: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
    """L2-regularized logistic regression via LBFGS. Returns (weight, bias)."""
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=0.5, max_iter=epochs, line_search_fn="strong_wolfe")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        logits = Xt @ w + b
        loss = loss_fn(logits, yt) + l2 * (w @ w)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def _logistic_prob(X: np.ndarray, w: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    logits = torch.tensor(X, dtype=torch.float32) @ w + b
    return torch.sigmoid(logits).numpy()


def classification_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    l2: float = 1e-3,
) -> dict:
    """Binary logistic probe. Returns accuracy and AUROC on held-out data."""
    y_train = np.asarray(y_train).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.float32)
    if len(np.unique(y_train)) < 2:
        return {"accuracy": float(np.mean(y_test == y_train[0])), "auroc": float("nan")}
    Xtr, Xte = _standardize(X_train, X_test)
    w, b = _fit_logistic(Xtr, y_train, l2)
    prob = _logistic_prob(Xte, w, b)
    return {
        "accuracy": accuracy(y_test, (prob >= 0.5).astype(np.float32)),
        "auroc": roc_auc(y_test, prob),
    }


def regression_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = 1.0,
) -> dict:
    """Ridge probe via the normal equations. Returns R^2 on held-out data."""
    Xtr, Xte = _standardize(X_train, X_test)
    y_mean = y_train.mean()
    yc = y_train - y_mean
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + alpha * np.eye(d)
    w = np.linalg.solve(A, Xtr.T @ yc)
    pred = Xte @ w + y_mean
    return {"r2": r2_score(y_test, pred)}


def multilabel_probe(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    l2: float = 1e-3,
) -> dict:
    """One logistic probe per label column. Returns mean accuracy and mean AUROC."""
    accs, aurocs = [], []
    for j in range(Y_train.shape[1]):
        res = classification_probe(X_train, Y_train[:, j], X_test, Y_test[:, j], l2)
        accs.append(res["accuracy"])
        if not np.isnan(res["auroc"]):
            aurocs.append(res["auroc"])
    return {
        "accuracy": float(np.mean(accs)),
        "auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
    }
