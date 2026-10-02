"""Evaluation statistics, NumPy only, so they can be unit-tested by hand.

Accuracy alone hides three things a deployment decision needs: whether a gap
between two models is larger than test-set noise (paired tests), whether the
softmax scores can be trusted as probabilities (calibration), and how much
traffic must be deferred to a human to meet an error budget (selective risk).
"""

from __future__ import annotations

import math

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def nll(proba: np.ndarray, y_true: np.ndarray, eps: float = 1e-12) -> float:
    """Mean negative log-likelihood of the true class."""
    p = proba[np.arange(len(y_true)), y_true]
    return float(-np.mean(np.log(np.clip(p, eps, 1.0))))


def reliability_bins(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per equal-width confidence bin: (mean confidence, accuracy, count).

    Empty bins are reported with count 0 and NaN means.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-closed bins so that confidence 1.0 falls in the last bin.
    idx = np.clip(np.searchsorted(edges, confidence, side="left") - 1, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    conf_sum = np.bincount(idx, weights=confidence, minlength=n_bins)
    acc_sum = np.bincount(idx, weights=correct.astype(float), minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return conf_sum / counts, acc_sum / counts, counts


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> float:
    """Count-weighted mean |accuracy - confidence| over confidence bins (Guo et al., 2017)."""
    mean_conf, acc, counts = reliability_bins(confidence, correct, n_bins)
    keep = counts > 0
    return float(np.sum(counts[keep] * np.abs(acc[keep] - mean_conf[keep])) / counts.sum())


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[int, int, float]:
    """Exact two-sided McNemar test on paired per-example correctness.

    Returns (b, c, p): b = examples only A gets right, c = only B gets right.
    Under H0 the discordant pairs split 50/50, so p is a two-sided binomial tail.
    """
    a = np.asarray(correct_a, dtype=bool)
    b_ = np.asarray(correct_b, dtype=bool)
    b = int(np.sum(a & ~b_))
    c = int(np.sum(~a & b_))
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    log_tail = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2)
        for i in range(k + 1)
    ]
    peak = max(log_tail)
    tail = math.exp(peak) * sum(math.exp(v - peak) for v in log_tail)
    return b, c, min(1.0, 2.0 * tail)


def paired_bootstrap_diff(
    score_a: np.ndarray, score_b: np.ndarray, n_boot: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of (score_b - score_a) with a 95% percentile interval.

    Examples are resampled jointly for both models, which keeps the pairing:
    the interval reflects which test images were drawn, not model noise.
    """
    diff = np.asarray(score_b, dtype=float) - np.asarray(score_a, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(diff)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = diff[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def risk_coverage(confidence: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Selective error rate when only the k most confident examples are accepted.

    Element k-1 of each array corresponds to accepting the top-k examples.
    """
    order = np.argsort(-confidence, kind="stable")
    errors = np.cumsum(~correct[order].astype(bool))
    k = np.arange(1, len(order) + 1)
    return k / len(order), errors / k


def threshold_for_budget(confidence: np.ndarray, correct: np.ndarray, budget: float) -> float:
    """Lowest confidence threshold whose accepted set keeps error <= budget.

    Chosen on validation data and then frozen, so the test coverage it yields is
    an honest out-of-sample estimate. Returns +inf if no threshold qualifies.
    """
    order = np.argsort(-confidence, kind="stable")
    conf_sorted = confidence[order]
    _, risk = risk_coverage(confidence, correct)
    # A threshold t accepts every example with confidence >= t, so only cut
    # points at the end of a run of tied confidences are feasible.
    last_of_tie = np.r_[conf_sorted[1:] != conf_sorted[:-1], True]
    ok = np.flatnonzero((risk <= budget) & last_of_tie)
    if ok.size == 0:
        return math.inf
    return float(conf_sorted[ok.max()])


def selective(confidence: np.ndarray, correct: np.ndarray, threshold: float) -> tuple[float, float]:
    """(coverage, error among accepted) for a fixed confidence threshold."""
    accept = confidence >= threshold
    if not accept.any():
        return 0.0, 0.0
    return float(accept.mean()), float(np.mean(~correct[accept].astype(bool)))


def top_confusions(cm: np.ndarray, k: int = 5) -> list[tuple[int, int, int]]:
    """The k most frequent (true, predicted, count) off-diagonal cells of a confusion matrix."""
    off = np.array(cm, copy=True)
    np.fill_diagonal(off, 0)
    flat = np.argsort(-off, axis=None, kind="stable")[:k]
    return [(int(i // off.shape[1]), int(i % off.shape[1]), int(off.flat[i])) for i in flat]


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 10) -> np.ndarray:
    return np.bincount(y_true * n_classes + y_pred, minlength=n_classes**2).reshape(
        n_classes, n_classes
    )
