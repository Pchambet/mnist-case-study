"""`analyze` on a tiny fake prediction cache, including the degenerate cases."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from mnist_study import figures, report
from mnist_study.analyze import analyze
from mnist_study.data import Split
from mnist_study.train import run_paths

FIXTURE = Path(__file__).parent / "fixtures" / "mnist_small.npz"

N_VAL, N_TEST = 50, 80


@pytest.fixture
def split():
    rng = np.random.default_rng(0)
    y_val, y_test = rng.integers(0, 10, N_VAL), rng.integers(0, 10, N_TEST)
    blank = np.zeros((1, 28, 28), np.uint8)
    return Split(blank, y_val[:1], blank, y_val, blank, y_test)


def _proba(y, rng, top_wrong=False):
    """Probabilities whose mistakes all sit below 0.6 confidence, so a 0.1% budget is reachable.

    With `top_wrong`, the most confident digit is also a mistake and no threshold qualifies.
    """
    conf = rng.uniform(0.5, 1.0, len(y))
    wrong = conf < 0.6
    if top_wrong:
        wrong[np.argmax(conf)] = True
    p = np.repeat(((1 - conf) / 9)[:, None], 10, axis=1)
    p[np.arange(len(y)), np.where(wrong, (y + 1) % 10, y)] = conf
    return p.astype(np.float32)


def _cache(root, split, seeds, budget_fails=()):
    """Write fake runs for {model: seeds}; (model, seed) pairs in budget_fails miss the budget."""
    root.mkdir()
    for name, model_seeds in seeds.items():
        for seed in model_seeds:
            rng = np.random.default_rng([seed, *name.encode()])
            _, meta_path, preds_path = run_paths(name, seed, root)
            test_proba = _proba(split.y_test, rng)
            np.savez(
                preds_path,
                val_proba=_proba(split.y_val, rng, top_wrong=(name, seed) in budget_fails),
                test_proba=test_proba,
                shifted_pred=np.broadcast_to(test_proba.argmax(1), (5, 8, N_TEST)).astype(np.int8),
            )
            hist = {k: [0.5, 0.4] for k in ("loss", "val_loss", "accuracy", "val_accuracy")}
            meta = {"params": 1000, "epochs_run": 2, "best_epoch": 2, "train_seconds": 1.0}
            meta_path.write_text(json.dumps(meta | {"history": hist}))


def _render(tmp_path, monkeypatch, split, seeds, budget_fails=()):
    """Cache -> analyze -> figures -> report in tmp_path; returns the summary and the page text."""
    results, figs = tmp_path / "results", tmp_path / "figures"
    _cache(tmp_path / "interim", split, seeds, budget_fails)
    summary = analyze(tmp_path / "interim", results, split)
    assert _strict_json(results / "summary.json") == summary
    for module in (figures, report):
        monkeypatch.setattr(module, "RESULTS", results)
        monkeypatch.setattr(module, "FIGURES", figs)
    monkeypatch.setattr(figures, "RAW_PATH", FIXTURE)  # gallery images; N_TEST < 200 digits
    assert all(p.stat().st_size > 0 for p in figures.render_all())
    html = report.build_report(tmp_path / "site").read_text()
    return summary, re.sub(r"base64,[A-Za-z0-9+/=]+", "base64,", html)


def _strict_json(path):
    def reject(token):
        raise ValueError(f"non-standard JSON constant {token} in {path.name}")

    return json.loads(path.read_text(), parse_constant=reject)


def test_budget_failure_is_reported_not_counted_as_met(tmp_path, split):
    seeds = {"mlp": [0, 1, 2], "cnn": [0, 1, 2], "mlp_wide": [0, 1, 2]}
    _cache(tmp_path / "interim", split, seeds, budget_fails={("cnn", 1)})
    s = analyze(tmp_path / "interim", tmp_path / "results", split)
    assert _strict_json(tmp_path / "results" / "summary.json") == s

    cnn = s["models"]["cnn"]
    assert cnn["error_budget_failed_seeds"] == [1]
    failed = next(r for r in cnn["error_budget_runs"] if r["seed"] == 1)
    assert failed["threshold"] is None
    assert failed["test_coverage"] is None and failed["test_error"] is None
    met = [r for r in cnn["error_budget_runs"] if r["seed"] != 1]
    for key in ("coverage", "error"):
        expected = np.mean([r[f"test_{key}"] for r in met])
        assert cnn[f"error_budget_{key}_mean"] == pytest.approx(expected, abs=1e-6)
    assert s["models"]["mlp"]["error_budget_failed_seeds"] == []


def test_figures_and_report_say_when_the_budget_was_not_met(tmp_path, monkeypatch, split):
    seeds = {"mlp": [0, 1, 2], "cnn": [0, 1, 2], "mlp_wide": [0, 1, 2]}
    fails = {("mlp", 0), ("mlp", 1), ("mlp", 2), ("cnn", 1)}
    s, html = _render(tmp_path, monkeypatch, split, seeds, fails)
    assert s["models"]["mlp"]["error_budget_coverage_mean"] is None
    assert s["models"]["mlp"]["error_budget_error_mean"] is None
    assert "No validation threshold met the error budget for MLP seeds 0, 1, 2; CNN seed 1" in html
    assert "the MLP automates n/a of the test stream" in html


def test_single_seed_reports_no_spread(tmp_path, monkeypatch, split):
    seeds = {"mlp": [3], "cnn": [3], "mlp_wide": [3]}
    s, html = _render(tmp_path, monkeypatch, split, seeds)
    for m in s["models"].values():
        assert m["test_accuracy_std"] is None and m["shift_accuracy_std"] is None
    assert not re.search(r"\bnan\b", html, re.IGNORECASE)
    assert "±" not in html
    mlp_acc = f"{s['models']['mlp']['test_accuracy_mean'] * 100:.2f}%"
    assert f'The <span class="amber">MLP</span> reaches {mlp_acc},' in html
    # Captions name the seed actually shown, not a hard-coded seed 0.
    assert "for seed 3," in html and "most confident mistakes (seed 3)" in html
    assert "seed 0" not in html


def test_gallery_and_mcnemar_use_seeds_both_models_have(tmp_path, split):
    seeds = {"mlp": [1, 2], "cnn": [0, 1], "mlp_wide": [1]}
    _cache(tmp_path / "interim", split, seeds)
    s = analyze(tmp_path / "interim", tmp_path / "results", split)
    assert [m["seed"] for m in s["comparison"]["mcnemar_per_seed"]] == [1]
    assert s["gallery"]["seed"] == 1
    with np.load(run_paths("mlp", 1, tmp_path / "interim")[2]) as p:
        mlp_pred = p["test_proba"].argmax(1)
    assert s["gallery"]["mistakes"]
    assert all(g["mlp_pred"] == mlp_pred[g["index"]] for g in s["gallery"]["mistakes"])


def test_models_without_a_common_seed_fail_clearly(tmp_path, split):
    _cache(tmp_path / "interim", split, {"mlp": [0], "cnn": [1], "mlp_wide": [1]})
    with pytest.raises(ValueError, match="mlp and cnn share no seed"):
        analyze(tmp_path / "interim", tmp_path / "results", split)
