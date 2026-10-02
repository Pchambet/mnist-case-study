"""Turn cached predictions into the small result tables committed under results/.

Everything the README, figures and report quote comes from these files.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from mnist_study import metrics as M
from mnist_study.data import Split, load_split
from mnist_study.models import CONTROL_NAMES, MODEL_NAMES
from mnist_study.train import INTERIM, run_paths

RESULTS = Path("results")
ERROR_BUDGET = 0.001  # at most 1 wrong digit per 1,000 accepted automatically
COVERAGE_GRID = np.round(np.linspace(0.5, 1.0, 101), 3)


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def _seeds(name: str, root: Path) -> list[int]:
    seeds = sorted(int(p.stem.split("_s")[1]) for p in root.glob(f"{name}_s*.json"))
    if not seeds:
        raise FileNotFoundError(f"no trained runs for {name!r} in {root}; run `make train` first")
    return seeds


def _r(x: float, nd: int = 6) -> float:
    return round(float(x), nd)


def _mean(values: list[float]) -> float | None:
    return _r(np.mean(values)) if values else None


def _budget_run(
    seed: int, val_proba: np.ndarray, y_val: np.ndarray, conf: np.ndarray, correct: np.ndarray
) -> dict:
    """Error-budget rule for one run: threshold fixed on validation, then applied to test.

    If no validation threshold meets the budget, the rule fails: threshold, coverage and
    error are null rather than an automate-nothing run counted as zero error.
    """
    run = {"seed": seed, "threshold": None, "test_coverage": None, "test_error": None}
    thr = M.threshold_for_budget(val_proba.max(1), val_proba.argmax(1) == y_val, ERROR_BUDGET)
    if math.isfinite(thr):
        cov, err = M.selective(conf, correct, thr)
        run.update(threshold=_r(thr), test_coverage=_r(cov), test_error=_r(err))
    # Hindsight benchmark only: the best threshold had we been allowed to tune on test.
    oracle_cov, _ = M.selective(conf, correct, M.threshold_for_budget(conf, correct, ERROR_BUDGET))
    run["oracle_test_coverage"] = _r(oracle_cov)
    return run


def analyze(root: Path = INTERIM, out: Path = RESULTS, split: Split | None = None) -> dict:
    """Read every cached run under `root`; only the labels of `split` are used."""
    if split is None:
        split = load_split()
    y_val, y_test = split.y_val, split.y_test
    out.mkdir(parents=True, exist_ok=True)

    runs, shift_rows, curve_rows, reliab_rows, rc_rows = [], [], [], [], []
    summary: dict = {"error_budget": ERROR_BUDGET, "n_test": len(y_test), "models": {}}
    correct_by_model: dict[str, np.ndarray] = {}
    confusion: dict[str, list] = {}

    for name in MODEL_NAMES + CONTROL_NAMES:
        seeds = _seeds(name, root)
        accs, eces, nlls, shift_acc, budget_runs, rc_risk = [], [], [], [], [], []
        correct_seeds, conf_pool = [], []
        cm = np.zeros((10, 10), dtype=int)
        for seed in seeds:
            _, meta_path, preds_path = run_paths(name, seed, root)
            meta = json.loads(meta_path.read_text())
            with np.load(preds_path) as p:
                val_proba, test_proba, shifted = p["val_proba"], p["test_proba"], p["shifted_pred"]
            pred = test_proba.argmax(1)
            correct = pred == y_test
            conf = test_proba.max(1)
            acc, ece, loss = (
                M.accuracy(y_test, pred),
                M.expected_calibration_error(conf, correct),
                M.nll(test_proba, y_test),
            )
            accs.append(acc)
            eces.append(ece)
            nlls.append(loss)
            correct_seeds.append(correct)
            conf_pool.append(conf)
            cm += M.confusion_matrix(y_test, pred)

            budget_runs.append(_budget_run(seed, val_proba, y_val, conf, correct))

            coverage, risk = M.risk_coverage(conf, correct)
            rc_risk.append(np.interp(COVERAGE_GRID, coverage, risk))

            per_dist = (shifted == y_test[None, None, :]).mean(axis=(1, 2))
            shift_acc.append(per_dist)
            for d, a in enumerate(per_dist):
                shift_rows.append([name, seed, d, _r(a)])

            h = meta["history"]
            for e in range(meta["epochs_run"]):
                curve_rows.append(
                    [
                        name,
                        seed,
                        e + 1,
                        _r(h["loss"][e]),
                        _r(h["val_loss"][e]),
                        _r(h["accuracy"][e]),
                        _r(h["val_accuracy"][e]),
                    ]
                )
            runs.append(
                [
                    name,
                    seed,
                    meta["params"],
                    meta["epochs_run"],
                    meta["best_epoch"],
                    meta["train_seconds"],
                    _r(M.accuracy(y_val, val_proba.argmax(1))),
                    _r(acc),
                    int((~correct).sum()),
                    _r(loss),
                    _r(ece),
                ]
            )

        mean_conf, bin_acc, counts = M.reliability_bins(
            np.concatenate(conf_pool), np.concatenate(correct_seeds)
        )
        for b, (mc, ba, c) in enumerate(zip(mean_conf, bin_acc, counts, strict=True)):
            if c:
                reliab_rows.append([name, b, _r(mc), _r(ba), int(c)])
        for cov, r in zip(COVERAGE_GRID, np.mean(rc_risk, axis=0), strict=True):
            rc_rows.append([name, cov, _r(r)])

        shift_acc = np.array(shift_acc)
        met = [b for b in budget_runs if b["threshold"] is not None]
        # A spread across seeds needs at least two of them; with one it is undefined (null).
        several = len(seeds) > 1
        correct_by_model[name] = np.mean(correct_seeds, axis=0)
        confusion[name] = cm.tolist()
        summary["models"][name] = {
            "seeds": seeds,
            "params": runs[-1][2],
            "test_accuracy_mean": _r(np.mean(accs)),
            "test_accuracy_std": _r(np.std(accs, ddof=1)) if several else None,
            "test_errors_mean": _r(len(y_test) * (1 - np.mean(accs)), 1),
            "ece_mean": _r(np.mean(eces)),
            "nll_mean": _r(np.mean(nlls)),
            "shift_accuracy_mean": [_r(a) for a in shift_acc.mean(0)],
            "shift_accuracy_std": [_r(a) for a in shift_acc.std(0, ddof=1)] if several else None,
            "error_budget_runs": budget_runs,
            # Seeds whose validation set allowed no threshold; left out of the two means below.
            "error_budget_failed_seeds": [b["seed"] for b in budget_runs if b["threshold"] is None],
            "error_budget_coverage_mean": _mean([b["test_coverage"] for b in met]),
            "error_budget_error_mean": _mean([b["test_error"] for b in met]),
            "error_budget_oracle_coverage_mean": _r(
                np.mean([b["oracle_test_coverage"] for b in budget_runs])
            ),
            "top_confusions": [list(t) for t in M.top_confusions(cm)],
        }

    summary["comparison"] = _paired(
        "mlp", "cnn", correct_by_model, summary, y_test, root, "accuracy_gain_cnn_minus_mlp"
    )
    # Capacity control: same parameter budget as the CNN, no convolution.
    summary["capacity_control"] = _paired(
        "mlp_wide",
        "cnn",
        correct_by_model,
        summary,
        y_test,
        root,
        "accuracy_gain_cnn_minus_mlp_wide",
    )
    gallery_seed = _common_seeds(summary, "mlp", "cnn")[0]
    summary["gallery"] = {
        "seed": gallery_seed,
        "mistakes": _confident_cnn_mistakes(y_test, gallery_seed, root),
    }

    _write_csv(
        out / "runs.csv",
        [
            "model",
            "seed",
            "params",
            "epochs_run",
            "best_epoch",
            "train_seconds",
            "val_accuracy",
            "test_accuracy",
            "test_errors",
            "test_nll",
            "test_ece",
        ],
        runs,
    )
    _write_csv(out / "shift.csv", ["model", "seed", "shift_px", "accuracy"], shift_rows)
    _write_csv(
        out / "learning_curves.csv",
        ["model", "seed", "epoch", "loss", "val_loss", "accuracy", "val_accuracy"],
        curve_rows,
    )
    _write_csv(
        out / "reliability.csv",
        ["model", "bin", "mean_confidence", "accuracy", "count"],
        reliab_rows,
    )
    _write_csv(out / "risk_coverage.csv", ["model", "coverage", "selective_error"], rc_rows)
    (out / "confusion.json").write_text(json.dumps(confusion))
    # allow_nan=False: NaN or inf would make the file invalid JSON for other readers.
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def _paired(
    a: str,
    b: str,
    correct_by_model: dict[str, np.ndarray],
    summary: dict,
    y_test: np.ndarray,
    root: Path,
    gain_key: str,
) -> dict:
    """Model b against model a on the same test images: bootstrap gain and McNemar per seed."""
    diff, lo, hi = M.paired_bootstrap_diff(correct_by_model[a], correct_by_model[b])
    mcnemar = []
    for s in _common_seeds(summary, a, b):
        ca = _test_proba(a, s, root).argmax(1) == y_test
        cb = _test_proba(b, s, root).argmax(1) == y_test
        only_a, only_b, p = M.mcnemar_exact(ca, cb)
        mcnemar.append(
            {
                "seed": s,
                f"only_{a}_correct": only_a,
                f"only_{b}_correct": only_b,
                "both_wrong": int(np.sum(~ca & ~cb)),
                "p_value": float(f"{p:.3g}"),
            }
        )
    return {gain_key: _r(diff), "bootstrap_95ci": [_r(lo), _r(hi)], "mcnemar_per_seed": mcnemar}


def _common_seeds(summary: dict, a: str, b: str) -> list[int]:
    """Seeds trained for both models, so that they can be compared run by run."""
    seeds = sorted(set(summary["models"][a]["seeds"]) & set(summary["models"][b]["seeds"]))
    if not seeds:
        raise ValueError(
            f"{a} and {b} share no seed, so they cannot be compared run by run; "
            "train both with the same --seeds"
        )
    return seeds


def _test_proba(name: str, seed: int, root: Path) -> np.ndarray:
    with np.load(run_paths(name, seed, root)[2]) as p:
        return p["test_proba"]


def _confident_cnn_mistakes(y_test: np.ndarray, seed: int, root: Path, k: int = 12) -> list[dict]:
    """The CNN's most confident test errors, with what the MLP said about the same digit."""
    cnn, mlp = _test_proba("cnn", seed, root), _test_proba("mlp", seed, root)
    wrong = np.flatnonzero(cnn.argmax(1) != y_test)
    wrong = wrong[np.argsort(-cnn[wrong].max(1), kind="stable")][:k]
    return [
        {
            "index": int(i),
            "label": int(y_test[i]),
            "cnn_pred": int(cnn[i].argmax()),
            "cnn_conf": _r(cnn[i].max(), 3),
            "mlp_pred": int(mlp[i].argmax()),
        }
        for i in wrong
    ]
