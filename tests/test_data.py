import numpy as np
import pytest

from mnist_study import data


def test_stratified_split_is_disjoint_and_balanced():
    y = np.repeat(np.arange(10), [100, 120, 80, 90, 110, 100, 95, 105, 100, 100])
    tr, va = data.stratified_split(y, 0.1, seed=0)
    assert np.intersect1d(tr, va).size == 0
    assert np.union1d(tr, va).size == y.size
    np.testing.assert_array_equal(np.bincount(y[va]), np.round(np.bincount(y) * 0.1))
    _, va2 = data.stratified_split(y, 0.1, seed=0)
    np.testing.assert_array_equal(va, va2)


def test_download_skips_when_hash_matches(tmp_path):
    f = tmp_path / "mnist.npz"
    f.write_bytes(b"cached")
    assert data.download(f, url="http://invalid.invalid/", expected=data.sha256(f)) == f


def test_download_rejects_corrupt_file(tmp_path):
    src = tmp_path / "remote.bin"
    src.write_bytes(b"tampered")
    dst = tmp_path / "mnist.npz"
    with pytest.raises(ValueError, match="checksum mismatch"):
        data.download(dst, url=src.as_uri(), expected="0" * 64)
    assert not dst.exists()
    assert not dst.with_suffix(".part").exists()


def test_fixture_has_expected_shapes(small_mnist):
    x_train, y_train, x_test, _ = small_mnist
    assert x_train.shape == (600, 28, 28) and x_test.shape == (200, 28, 28)
    assert x_train.dtype == np.uint8 and y_train.dtype == np.int64
    assert set(np.unique(y_train)) == set(range(10))
    assert data.to_float(x_test).max() <= 1.0
