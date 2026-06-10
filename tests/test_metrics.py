import numpy as np

from metrics import accuracy, matthews_corrcoef, r2_score, roc_auc, spearman


def test_accuracy():
    assert accuracy([1, 0, 1], [1, 0, 1]) == 1.0
    assert accuracy([1, 0, 1], [0, 1, 0]) == 0.0


def test_mcc_extremes():
    y = np.array([0, 0, 1, 1])
    assert matthews_corrcoef(y, y) == 1.0
    assert matthews_corrcoef(y, 1 - y) == -1.0
    # constant prediction has undefined denominator, defined here as 0
    assert matthews_corrcoef(y, np.ones_like(y)) == 0.0


def test_roc_auc():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert np.isnan(roc_auc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])))


def test_roc_auc_matches_definition_with_ties():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.5, 0.5, 0.5, 0.5])  # all tied -> 0.5
    assert abs(roc_auc(y, s) - 0.5) < 1e-9


def test_r2():
    y = np.array([1.0, 2.0, 3.0])
    assert r2_score(y, y) == 1.0
    assert r2_score(y, np.full_like(y, y.mean())) == 0.0


def test_spearman_monotonic():
    x = np.array([1, 2, 3, 4, 5])
    assert abs(spearman(x, x ** 3) - 1.0) < 1e-9
    assert abs(spearman(x, -x) + 1.0) < 1e-9
