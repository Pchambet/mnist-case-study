"""Cache logic of `run_all`, with training and prediction replaced by cheap fakes.

The fake model file holds a version number and the fake predictions copy it, so a
test can tell whether the predictions on disk come from the model on disk.
"""

import json
from dataclasses import asdict

import numpy as np
import pytest

from mnist_study import train
from mnist_study.train import TrainConfig, run_all, run_paths


@pytest.fixture
def calls(monkeypatch):
    log: list[str] = []
    versions = {"n": 0}

    def fake_train(name, seed, split, cfg, root):
        model_path, meta_path, _ = run_paths(name, seed, root)
        versions["n"] += 1
        root.mkdir(parents=True, exist_ok=True)
        model_path.write_text(str(versions["n"]))
        meta_path.write_text(json.dumps({"config": asdict(cfg)}))
        log.append("train")

    def fake_predict(name, seed, split, cfg, root):
        model_path, _, preds_path = run_paths(name, seed, root)
        np.savez(preds_path, version=int(model_path.read_text()))
        log.append("predict")
        return preds_path

    monkeypatch.setattr(train, "train_one", fake_train)
    monkeypatch.setattr(train, "predict_one", fake_predict)
    return log


DEFAULT = TrainConfig()


def _run(root, cfg=DEFAULT):
    run_all(split=None, seeds=[0], cfg=cfg, names=("mlp",), root=root)


def _versions(root):
    model_path, _, preds_path = run_paths("mlp", 0, root)
    with np.load(preds_path) as p:
        return int(model_path.read_text()), int(p["version"])


def test_cached_run_is_reused(tmp_path, calls):
    _run(tmp_path)
    assert calls == ["train", "predict"]
    _run(tmp_path)
    assert calls == ["train", "predict"]


def test_missing_predictions_alone_are_recomputed(tmp_path, calls):
    _run(tmp_path)
    run_paths("mlp", 0, tmp_path)[2].unlink()
    _run(tmp_path)
    assert calls == ["train", "predict", "predict"]


def test_retrained_model_gets_fresh_predictions(tmp_path, calls):
    _run(tmp_path)
    run_paths("mlp", 0, tmp_path)[1].unlink()  # e.g. a run interrupted before its meta was saved
    _run(tmp_path)
    assert calls == ["train", "predict", "train", "predict"]
    assert _versions(tmp_path) == (2, 2)


def test_config_change_retrains_and_says_so(tmp_path, calls, capsys):
    _run(tmp_path)
    _run(tmp_path, TrainConfig(max_epochs=20))
    assert calls == ["train", "predict", "train", "predict"]
    assert _versions(tmp_path) == (2, 2)
    meta = json.loads(run_paths("mlp", 0, tmp_path)[1].read_text())
    assert meta["config"]["max_epochs"] == 20
    out = capsys.readouterr().out
    assert "[retrain] mlp seed=0: config changed (max_epochs 12 -> 20)" in out


def test_interrupted_retraining_leaves_no_stale_cache(tmp_path, calls, monkeypatch):
    _run(tmp_path)

    def crash(name, seed, split, cfg, root):
        run_paths(name, seed, root)[0].write_text("99")  # new weights saved, then killed
        raise KeyboardInterrupt

    monkeypatch.setattr(train, "train_one", crash)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, TrainConfig(max_epochs=20))
    _, meta_path, preds_path = run_paths("mlp", 0, tmp_path)
    # Without the old meta and predictions, the next run cannot mistake this model for cached.
    assert not meta_path.exists() and not preds_path.exists()
