"""Rigid translations of digit images.

MNIST digits are centred by centre of mass in a 28x28 frame with a ~4 px
margin. Any pixel pushed past the border is dropped and vacated pixels are
filled with background (0). Diagonal directions move d px along each axis. On
the test set, 99.9% of the ink survives a 2 px shift and 98.7% a 4 px shift,
but at 4 px about a quarter of (digit, direction) pairs lose more than 2%.
"""

from __future__ import annotations

import numpy as np

# The eight compass directions; accuracy at distance d is averaged over them so
# that no single direction (e.g. "down", where 7s and 9s differ) dominates.
DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
)


def translate(images: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift (N, H, W) images by dx columns (right > 0) and dy rows (down > 0)."""
    _, h, w = images.shape
    out = np.zeros_like(images)
    if abs(dx) >= w or abs(dy) >= h:
        return out
    src_r = slice(max(0, -dy), h - max(0, dy))
    dst_r = slice(max(0, dy), h - max(0, -dy))
    src_c = slice(max(0, -dx), w - max(0, dx))
    dst_c = slice(max(0, dx), w - max(0, -dx))
    out[:, dst_r, dst_c] = images[:, src_r, src_c]
    return out
