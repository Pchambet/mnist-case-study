from pathlib import Path

import pytest

from mnist_study.data import load_npz

FIXTURE = Path(__file__).parent / "fixtures" / "mnist_small.npz"


@pytest.fixture(scope="session")
def small_mnist():
    """600 train / 200 test real MNIST digits, committed so tests run offline."""
    return load_npz(FIXTURE)
