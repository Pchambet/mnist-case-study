import math

import numpy as np
import pytest

from mnist_study import metrics as M


def test_accuracy_and_nll_by_hand():
    y = np.array([0, 1, 2, 1])
    proba = np.array([[0.5, 0.5, 0], [0.25, 0.75, 0], [0, 0.5, 0.5], [1.0, 0, 0]])
    assert M.accuracy(y, proba.argmax(1)) == 0.5
    expected = -(math.log(0.5) + math.log(0.75) + math.log(0.5) + math.log(1e-12)) / 4
    assert M.nll(proba, y) == pytest.approx(expected)


def test_ece_hand_computed_two_bins():
    # Bin (0.6, 0.667]: confidences 0.65, 0.65, accuracy 1/2 -> gap 0.15.
    # Bin (0.933, 1.0]: confidence 1.0, accuracy 1 -> gap 0.
    conf = np.array([0.65, 0.65, 1.0])
    correct = np.array([True, False, True])
    assert M.expected_calibration_error(conf, correct) == pytest.approx(2 / 3 * 0.15)


def test_ece_near_zero_for_calibrated_ground_truth():
    # Correctness drawn with probability equal to the stated confidence.
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.1, 1.0, 200_000)
    correct = rng.uniform(size=conf.size) < conf
    assert M.expected_calibration_error(conf, correct) < 0.005
    # Systematic over-confidence by 0.1 is recovered as ECE ~= 0.1.
    assert M.expected_calibration_error(np.minimum(conf + 0.1, 1.0), correct) == pytest.approx(
        0.1, abs=0.01
    )


def test_mcnemar_exact_matches_binomial_tail():
    a = np.array([True] * 10 + [False] * 2 + [True] * 50)
    b = np.array([False] * 10 + [True] * 2 + [True] * 50)
    only_a, only_b, p = M.mcnemar_exact(a, b)
    assert (only_a, only_b) == (10, 2)
    # 2 * P(X <= 2), X ~ Binomial(12, 0.5) = 2 * (1 + 12 + 66) / 4096
    assert p == pytest.approx(2 * 79 / 4096)
    assert M.mcnemar_exact(a, a)[2] == 1.0


def test_mcnemar_large_counts_do_not_overflow():
    a = np.r_[np.ones(400, bool), np.zeros(300, bool)]
    _, _, p = M.mcnemar_exact(a, ~a)
    assert 0 < p < 1e-3


def test_paired_bootstrap_interval_covers_true_gap():
    # Ground truth: model B is right 5 points more often than model A. Over many
    # simulated test sets the 95% interval should contain 0.05 about 95% of the time.
    rng = np.random.default_rng(1)
    hits = 0
    for sim in range(100):
        a = rng.uniform(size=2_000) < 0.90
        b = rng.uniform(size=2_000) < 0.95
        diff, lo, hi = M.paired_bootstrap_diff(a, b, n_boot=400, seed=sim)
        assert lo <= diff <= hi
        hits += lo <= 0.05 <= hi
    assert 85 <= hits <= 100


def test_risk_coverage_by_hand():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([True, False, True, True])
    cov, risk = M.risk_coverage(conf, correct)
    np.testing.assert_allclose(cov, [0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(risk, [0, 0.5, 1 / 3, 0.25])


def test_threshold_respects_budget_and_ties():
    conf = np.array([0.99, 0.95, 0.95, 0.90, 0.80, 0.70])
    correct = np.array([True, True, False, True, True, False])
    # Zero-error budget: only 0.99 is safe; the tied 0.95 pair contains a mistake.
    assert M.threshold_for_budget(conf, correct, 0.0) == 0.99
    # 20% budget: accepting down to 0.80 gives 1 error in 5; 0.70 would give 2 in 6.
    assert M.threshold_for_budget(conf, correct, 0.2) == 0.80
    assert M.selective(conf, correct, 0.80) == (5 / 6, 0.2)
    assert M.threshold_for_budget(np.array([0.9]), np.array([False]), 0.0) == math.inf
    assert M.selective(conf, correct, math.inf) == (0.0, 0.0)


def test_confusions():
    y = np.array([0, 0, 1, 1, 1, 2])
    p = np.array([0, 1, 2, 2, 1, 2])
    cm = M.confusion_matrix(y, p, n_classes=3)
    np.testing.assert_array_equal(cm, [[1, 1, 0], [0, 1, 2], [0, 0, 1]])
    assert M.top_confusions(cm, k=2) == [(1, 2, 2), (0, 1, 1)]
