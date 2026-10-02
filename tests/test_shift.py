import numpy as np
import pytest

from mnist_study.shift import DIRECTIONS, translate


def test_single_pixel_moves_right_and_down():
    img = np.zeros((1, 5, 5), dtype=np.uint8)
    img[0, 1, 2] = 7
    out = translate(img, dx=2, dy=1)
    assert out[0, 2, 4] == 7
    assert out.sum() == 7


@pytest.mark.parametrize(("dx", "dy"), [(-3, 0), (0, -2), (2, 2), (-1, 1)])
def test_inverse_shift_restores_interior(dx, dy):
    rng = np.random.default_rng(0)
    img = np.zeros((2, 12, 12), dtype=np.uint8)
    img[:, 4:8, 4:8] = rng.integers(1, 255, (2, 4, 4))
    np.testing.assert_array_equal(translate(translate(img, dx, dy), -dx, -dy), img)


def test_pixels_leaving_frame_are_dropped():
    img = np.ones((1, 4, 4), dtype=np.uint8)
    assert translate(img, 1, 0)[0, :, 0].sum() == 0
    assert translate(img, 1, 0).sum() == 12
    assert translate(img, 4, 0).sum() == 0
    np.testing.assert_array_equal(translate(img, 0, 0), img)


def test_real_digits_survive_a_two_pixel_shift(small_mnist):
    x_train = small_mnist[0]
    for ux, uy in DIRECTIONS:
        shifted = translate(x_train, 2 * ux, 2 * uy)
        kept = shifted.astype(int).sum() / x_train.astype(int).sum()
        assert kept > 0.99


def test_eight_distinct_unit_directions():
    assert len(set(DIRECTIONS)) == 8
    assert all(max(abs(x), abs(y)) == 1 for x, y in DIRECTIONS)
