"""Download, verify and split MNIST.

The raw file is the `mnist.npz` mirror used by `keras.datasets.mnist`, fetched
directly so that the pipeline does not depend on Keras' cache location and the
download is verified against a pinned SHA-256.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
RAW_PATH = Path("data/raw/mnist.npz")
FIXTURE_PATH = Path("tests/fixtures/mnist_small.npz")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(path: Path = RAW_PATH, url: str = MNIST_URL, expected: str = MNIST_SHA256) -> Path:
    """Fetch the file once; re-download only if it is missing or corrupt.

    The download goes to a temporary name and is renamed only after the hash
    matches, so an interrupted run never leaves a half-written dataset behind.
    """
    if path.exists() and sha256(path) == expected:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    actual = sha256(tmp)
    if actual != expected:
        tmp.unlink()
        raise ValueError(f"checksum mismatch for {url}: expected {expected}, got {actual}")
    tmp.replace(path)
    return path


@dataclass(frozen=True)
class Split:
    """Images as uint8 (N, 28, 28), labels as int64 (N,)."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as d:
        return (
            d["x_train"],
            d["y_train"].astype(np.int64),
            d["x_test"],
            d["y_test"].astype(np.int64),
        )


def stratified_split(
    y: np.ndarray, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx) with the class mix preserved in both parts.

    Keras' `validation_split` takes the *last* rows of the array, which silently
    ties the validation set to file order; a seeded stratified draw does not.
    """
    rng = np.random.default_rng(seed)
    val_parts = []
    for label in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == label))
        val_parts.append(idx[: round(len(idx) * val_fraction)])
    val_idx = np.sort(np.concatenate(val_parts))
    train_idx = np.setdiff1d(np.arange(len(y)), val_idx)
    return train_idx, val_idx


def load_split(path: Path = RAW_PATH, val_fraction: float = 0.1, seed: int = 0) -> Split:
    """Official test set untouched; validation carved out of the official train set.

    The split seed is fixed across training seeds so that run-to-run variance
    reflects initialisation and batch order only.
    """
    x_train, y_train, x_test, y_test = load_npz(path)
    tr, va = stratified_split(y_train, val_fraction, seed)
    return Split(x_train[tr], y_train[tr], x_train[va], y_train[va], x_test, y_test)


def to_float(images: np.ndarray) -> np.ndarray:
    return images.astype(np.float32) / 255.0


def make_fixture(
    src: Path = RAW_PATH, dst: Path = FIXTURE_PATH, n_train: int = 600, n_test: int = 200
) -> Path:
    """Write the small committed subset used by the offline test suite."""
    x_train, y_train, x_test, y_test = load_npz(src)
    rng = np.random.default_rng(0)
    tr = np.sort(rng.choice(len(y_train), n_train, replace=False))
    te = np.sort(rng.choice(len(y_test), n_test, replace=False))
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dst,
        x_train=x_train[tr],
        y_train=y_train[tr].astype(np.uint8),
        x_test=x_test[te],
        y_test=y_test[te].astype(np.uint8),
    )
    return dst
