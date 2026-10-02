"""Train each (model, seed) pair once and store its predictions.

Each run is cached under `data/interim/`: re-running the pipeline only trains
what is missing, so an interrupted run resumes instead of starting over.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from mnist_study.data import Split, to_float
from mnist_study.models import build, configure
from mnist_study.shift import DIRECTIONS, translate

INTERIM = Path("data/interim")


@dataclass(frozen=True)
class TrainConfig:
    max_epochs: int = 12
    patience: int = 2
    batch_size: int = 128
    learning_rate: float = 1e-3
    max_shift: int = 4


def run_paths(name: str, seed: int, root: Path = INTERIM) -> tuple[Path, Path, Path]:
    stem = f"{name}_s{seed}"
    return root / f"{stem}.keras", root / f"{stem}.json", root / f"{stem}_preds.npz"


def train_one(name: str, seed: int, split: Split, cfg: TrainConfig, root: Path = INTERIM) -> dict:
    """Fit with early stopping on validation loss; the test set is not touched here."""
    from tensorflow import keras

    model_path, meta_path, _ = run_paths(name, seed, root)
    configure(seed)
    model = build(name, cfg.learning_rate)
    stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=cfg.patience, restore_best_weights=True
    )
    start = time.perf_counter()
    history = model.fit(
        to_float(split.x_train),
        split.y_train,
        validation_data=(to_float(split.x_val), split.y_val),
        epochs=cfg.max_epochs,
        batch_size=cfg.batch_size,
        callbacks=[stop],
        verbose=2,
    )
    seconds = time.perf_counter() - start
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    meta = {
        "model": name,
        "seed": seed,
        "params": int(model.count_params()),
        "epochs_run": len(hist["loss"]),
        "best_epoch": int(np.argmin(hist["val_loss"])) + 1,
        "train_seconds": round(seconds, 1),
        "config": asdict(cfg),
        "history": hist,
    }
    root.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def predict_one(name: str, seed: int, split: Split, cfg: TrainConfig, root: Path = INTERIM) -> Path:
    """Store validation/test probabilities and labels predicted on shifted test images."""
    from tensorflow import keras

    model_path, _, preds_path = run_paths(name, seed, root)
    model = keras.models.load_model(model_path)

    def proba(x: np.ndarray) -> np.ndarray:
        return model.predict(to_float(x), batch_size=1000, verbose=0).astype(np.float32)

    shifted = np.zeros((cfg.max_shift + 1, len(DIRECTIONS), len(split.y_test)), dtype=np.int8)
    for d in range(cfg.max_shift + 1):
        for k, (ux, uy) in enumerate(DIRECTIONS):
            shifted[d, k] = proba(translate(split.x_test, ux * d, uy * d)).argmax(1)
    np.savez_compressed(
        preds_path,
        val_proba=proba(split.x_val),
        test_proba=proba(split.x_test),
        shifted_pred=shifted,
    )
    return preds_path


def run_all(
    split: Split, seeds: list[int], cfg: TrainConfig, names=("mlp", "cnn"), root: Path = INTERIM
) -> None:
    for name in names:
        for seed in seeds:
            model_path, meta_path, preds_path = run_paths(name, seed, root)
            if not (model_path.exists() and meta_path.exists()):
                print(f"[train] {name} seed={seed}", flush=True)
                train_one(name, seed, split, cfg, root)
            if not preds_path.exists():
                print(f"[predict] {name} seed={seed}", flush=True)
                predict_one(name, seed, split, cfg, root)
