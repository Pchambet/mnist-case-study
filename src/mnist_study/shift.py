"""Rigid translations of digit images.

MNIST digits are centred by centre of mass in a 28x28 frame with a ~4 px
margin, so shifts up to 4 px keep the stroke inside the image; any pixel pushed
past the border is dropped and vacated pixels are filled with background (0).
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
