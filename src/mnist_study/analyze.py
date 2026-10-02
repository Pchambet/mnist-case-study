"""Turn cached predictions into the small result tables committed under results/.

Everything the README, figures and report quote comes from these files.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from mnist_study import metrics as M
from mnist_study.data import load_split
from mnist_study.models import MODEL_NAMES
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


def analyze(root: Path = INTERIM, out: Path = RESULTS) -> dict:
    split = load_split()
    y_val, y_test = split.y_val, split.y_test
    out.mkdir(parents=True, exist_ok=True)

    runs, shift_rows, curve_rows, reliab_rows, rc_rows = [], [], [], [], []
    summary: dict = {"error_budget": ERROR_BUDGET, "n_test": len(y_test), "models": {}}
    correct_by_model: dict[str, np.ndarray] = {}
    confusion: dict[str, list] = {}

    for name in MODEL_NAMES:
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

            # Error-budget rule: threshold fixed on validation, applied to test.
            val_correct = val_proba.argmax(1) == y_val
            thr = M.threshold_for_budget(val_proba.max(1), val_correct, ERROR_BUDGET)
            cov, err = M.selective(conf, correct, thr)
            # Hindsight benchmark only: the best threshold had we been allowed to tune on test.
            oracle_cov, _ = M.selective(
                conf, correct, M.threshold_for_budget(conf, correct, ERROR_BUDGET)
            )
            budget_runs.append(
                {
                    "seed": seed,
                    "threshold": _r(thr),
                    "test_coverage": _r(cov),
                    "test_error": _r(err),
                    "oracle_test_coverage": _r(oracle_cov),
                }
            )

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
        correct_by_model[name] = np.mean(correct_seeds, axis=0)
        confusion[name] = cm.tolist()
        summary["models"][name] = {
            "seeds": seeds,
            "params": runs[-1][2],
            "test_accuracy_mean": _r(np.mean(accs)),
            "test_accuracy_std": _r(np.std(accs, ddof=1)),
            "test_errors_mean": _r(len(y_test) * (1 - np.mean(accs)), 1),
            "ece_mean": _r(np.mean(eces)),
            "nll_mean": _r(np.mean(nlls)),
            "shift_accuracy_mean": [_r(a) for a in shift_acc.mean(0)],
            "shift_accuracy_std": [_r(a) for a in shift_acc.std(0, ddof=1)],
            "error_budget_runs": budget_runs,
            "error_budget_coverage_mean": _r(np.mean([b["test_coverage"] for b in budget_runs])),
            "error_budget_error_mean": _r(np.mean([b["test_error"] for b in budget_runs])),
            "error_budget_oracle_coverage_mean": _r(
                np.mean([b["oracle_test_coverage"] for b in budget_runs])
            ),
            "top_confusions": [list(t) for t in M.top_confusions(cm)],
        }

    diff, lo, hi = M.paired_bootstrap_diff(correct_by_model["mlp"], correct_by_model["cnn"])
    mlp_seeds, cnn_seeds = summary["models"]["mlp"]["seeds"], summary["models"]["cnn"]["seeds"]
    mcnemar = []
    for s in sorted(set(mlp_seeds) & set(cnn_seeds)):
        a = _test_proba("mlp", s, root).argmax(1) == y_test
        b = _test_proba("cnn", s, root).argmax(1) == y_test
        only_mlp, only_cnn, p = M.mcnemar_exact(a, b)
        mcnemar.append(
            {
                "seed": s,
                "only_mlp_correct": only_mlp,
                "only_cnn_correct": only_cnn,
                "both_wrong": int(np.sum(~a & ~b)),
                "p_value": float(f"{p:.3g}"),
            }
        )
    summary["comparison"] = {
        "accuracy_gain_cnn_minus_mlp": _r(diff),
        "bootstrap_95ci": [_r(lo), _r(hi)],
        "mcnemar_per_seed": mcnemar,
    }
    summary["gallery"] = _confident_cnn_mistakes(y_test, min(cnn_seeds), root)

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
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


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
