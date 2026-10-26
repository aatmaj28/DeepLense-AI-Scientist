"""Label maps for flipped/rotated SIE images (offline twin of scripts/check_label_transforms.py).

The container check compares against lenstronomy renders; this one uses an analytic
elliptical blob on lenstronomy's grid convention (column index along +x, row index
along +y, centred), so a sign error in the maps fails here without Docker.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from dlens.data._lens_labels import (ALL_TRANSFORMS, TARGETS, transform_ellipticity, transform_image,
                                     transform_position, transform_targets)

N, DP = 64, 0.1
_C = (np.arange(N) - (N - 1) / 2) * DP
X, Y = np.meshgrid(_C, _C)  # X varies along columns, Y along rows, as in lenstronomy


def render(theta_e, e1, e2, cx, cy):
    """An elliptical Gaussian with lenstronomy's (e1, e2) and centre conventions."""
    phi = np.arctan2(e2, e1) / 2
    c = min(np.hypot(e1, e2), 0.9999)
    q = (1 - c) / (1 + c)
    dx, dy = X - cx, Y - cy
    xr = np.cos(phi) * dx + np.sin(phi) * dy
    yr = -np.sin(phi) * dx + np.cos(phi) * dy
    return np.exp(-(q * xr ** 2 + yr ** 2 / q) / (2 * (0.4 * theta_e) ** 2))


def systems(n=5, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        q, phi = rng.uniform(0.5, 0.8), rng.uniform(0.1, 1.4)  # well away from symmetric angles
        c = (1 - q) / (1 + q)
        out.append(np.array([rng.uniform(1.0, 2.0), c * np.cos(2 * phi), c * np.sin(2 * phi),
                             rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8)]))
    return out


@pytest.mark.parametrize("fc,fr,k", ALL_TRANSFORMS)
def test_label_maps_match_transformed_images(fc, fr, k):
    for t in systems():
        expected = transform_image(render(*t), fc, fr, k)
        moved = render(*transform_targets(t, fc, fr, k))
        assert np.abs(moved - expected).max() < 1e-9


def test_plausible_sign_errors_are_caught():
    t = systems(1)[0]
    base = render(*t)

    def mismatch(targets, fc, fr, k):
        return np.abs(render(*targets) - transform_image(base, fc, fr, k)).max()

    no_e2_flip = t.copy()
    no_e2_flip[3:] = transform_position(t[3], t[4], True, False, 0)
    assert mismatch(no_e2_flip, True, False, 0) > 0.1

    wrong_turn = t.copy()
    wrong_turn[1:3] = transform_ellipticity(t[1], t[2], False, False, 1)
    wrong_turn[3], wrong_turn[4] = -t[4], t[3]  # (x, y) -> (-y, x): the other direction
    assert mismatch(wrong_turn, False, False, 1) > 0.1


def test_identities():
    t = np.stack(systems())
    assert np.allclose(transform_targets(t, False, False, 4), t)
    assert np.allclose(transform_targets(transform_targets(t, True, False, 0), True, False, 0), t)
    assert np.allclose(transform_targets(transform_targets(t, False, True, 0), False, True, 0), t)
    for fc, fr, k in ALL_TRANSFORMS:
        moved = transform_targets(t, fc, fr, k)
        assert np.allclose(moved[:, 0], t[:, 0])  # theta_E is invariant
        assert np.allclose(np.hypot(moved[:, 1], moved[:, 2]), np.hypot(t[:, 1], t[:, 2]))
    assert TARGETS == ("theta_E", "e1", "e2", "center_x", "center_y")


def test_transform_image_matches_the_torch_augmentation():
    torch = pytest.importorskip("torch")
    from dlens.tools import _torch_backends

    # transform_image mirrors these exact ops; if the augmentation changes, so must the label maps.
    src = inspect.getsource(_torch_backends)
    ops = ["torch.flip(xb, dims=[3])", "torch.flip(xb, dims=[2])", "torch.rot90(xb, k="]
    positions = [src.find(op) for op in ops]
    assert -1 not in positions and positions == sorted(positions)
    assert "dims=[2, 3])" in src[positions[2]:positions[2] + 80]

    batch = np.random.default_rng(1).normal(size=(2, 1, 9, 9)).astype(np.float32)
    for fc, fr, k in ALL_TRANSFORMS:
        xb = torch.from_numpy(batch.copy())
        if fc:
            xb = torch.flip(xb, dims=[3])
        if fr:
            xb = torch.flip(xb, dims=[2])
        xb = torch.rot90(xb, k=k, dims=[2, 3])
        assert np.array_equal(xb.numpy(), transform_image(batch, fc, fr, k))
