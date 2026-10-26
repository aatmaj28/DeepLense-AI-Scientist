# data/_lens_labels.py
"""How SIE labels move when a lens image is flipped or rotated.

The regression images are lenstronomy renders on a grid built with
``make_grid_with_coordtransform(..., inverse=False)``: the array's column index
runs along +x and its row index along +y, both centred on the image middle. The
training augmentation (``tools/_torch_backends.py``) flips the columns, then the
rows, then applies ``rot90`` over (rows, columns). Each of those moves the lens in
a known way; these functions move the labels the same way, so an augmented image
keeps correct labels.

- Flipping columns maps x -> -x; flipping rows maps y -> -y. Either reflection
  maps the position angle phi -> -phi (mod pi), so e2 changes sign and e1 doesn't.
- ``rot90`` with k=1 maps (x, y) -> (y, -x). A 90-degree turn adds pi to 2*phi, so
  e1 and e2 both change sign; theta_E never changes.

This module imports only numpy, so the lenstronomy sandbox (which has no dlens)
can load it by path for the render-transform-compare check.
"""

from __future__ import annotations

import numpy as np

# Regression targets, in the order labels.csv and the arrays use.
TARGETS = ("theta_E", "e1", "e2", "center_x", "center_y")

# Every augmentation the training loop can draw: (flip_cols, flip_rows, k).
ALL_TRANSFORMS = tuple((fc, fr, k) for fc in (False, True) for fr in (False, True) for k in range(4))


def transform_image(img: np.ndarray, flip_cols: bool, flip_rows: bool, k: int) -> np.ndarray:
    """The augmentation's image operations, in its order, on the last two axes."""
    if flip_cols:
        img = np.flip(img, axis=-1)
    if flip_rows:
        img = np.flip(img, axis=-2)
    return np.rot90(img, k=k, axes=(-2, -1))


def transform_position(x, y, flip_cols: bool, flip_rows: bool, k: int):
    """Where a point at (x, y) ends up after the image transform."""
    if flip_cols:
        x = -x
    if flip_rows:
        y = -y
    for _ in range(k % 4):
        x, y = y, -x
    return x, y


def transform_ellipticity(e1, e2, flip_cols: bool, flip_rows: bool, k: int):
    """lenstronomy (e1, e2) after the image transform."""
    if flip_cols:
        e2 = -e2
    if flip_rows:
        e2 = -e2
    if k % 2:
        e1, e2 = -e1, -e2
    return e1, e2


def transform_targets(targets: np.ndarray, flip_cols: bool, flip_rows: bool, k: int) -> np.ndarray:
    """Apply the transform to an array [..., 5] in ``TARGETS`` order."""
    t = np.array(targets, dtype=float, copy=True)
    t[..., 1], t[..., 2] = transform_ellipticity(t[..., 1], t[..., 2], flip_cols, flip_rows, k)
    t[..., 3], t[..., 4] = transform_position(t[..., 3], t[..., 4], flip_cols, flip_rows, k)
    return t
