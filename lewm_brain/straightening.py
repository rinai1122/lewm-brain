"""Hénaff et al. 2021 perceptual straightening on a trajectory.

For a sequence X = (x_1, ..., x_T) in R^D, the per-step curvature is the
angle between consecutive frame-difference vectors:

    v_t = x_{t+1} - x_t
    θ_t = arccos( <v_t, v_{t+1}> / (||v_t|| · ||v_{t+1}||) )

θ = 0 means the trajectory continues in a straight line; θ = π means it
reverses. Lower mean θ across a stimulus = a straighter representation.
Hénaff et al. 2021 §Methods report the average across short ~11-frame
natural-movie clips. We expose both the per-step series and a windowed
mean so the caller can replicate either headline number.

Notes:
- Computation runs in float64. For large pixel-space trajectories this
  ~doubles peak memory vs float32, but the per-step `arccos` is the
  numerically delicate piece and float32 dot products of ~10⁵-D vectors
  give catastrophically wrong angles near 0° / 180°.
- Frames where ||v_t|| ≈ 0 yield an undefined angle. We mark θ as NaN
  there and the caller should `nanmean` over the result.
"""
from __future__ import annotations

import numpy as np


def per_step_curvature(X: np.ndarray) -> np.ndarray:
    """Returns (T-2,) per-step angles in radians for a (T, D) trajectory.

    Steps where either adjacent difference vector has zero length are
    NaN (the angle is undefined). Use `np.nanmean` downstream.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (T, D); got shape {X.shape}")
    T = X.shape[0]
    if T < 3:
        raise ValueError(f"Need at least 3 timesteps; got {T}")

    diffs = np.diff(X.astype(np.float64), axis=0)            # (T-1, D)
    norms = np.linalg.norm(diffs, axis=1)                     # (T-1,)
    valid = norms > 1e-12
    units = np.zeros_like(diffs)
    units[valid] = diffs[valid] / norms[valid, None]
    cos = (units[:-1] * units[1:]).sum(axis=1)
    cos = np.clip(cos, -1.0, 1.0)
    theta = np.arccos(cos)
    bad = ~(valid[:-1] & valid[1:])
    theta[bad] = np.nan
    return theta


def windowed_mean_curvature(
    X: np.ndarray,
    win: int = 11,
    stride: int | None = None,
) -> np.ndarray:
    """Mean per-step curvature inside each non-overlapping window of `win`
    consecutive frames. Hénaff 2021 used 11-frame natural-movie clips,
    each yielding 9 per-step angles that are then averaged.

    Returns shape (n_windows,) in radians.
    """
    if win < 3:
        raise ValueError(f"win must be >= 3; got {win}")
    stride = stride if stride is not None else win

    theta = per_step_curvature(X)                  # (T-2,)
    T = X.shape[0]
    n_win = max(0, (T - win) // stride + 1)
    out = np.empty(n_win, dtype=np.float64)
    for i in range(n_win):
        s = i * stride
        # window covers frames [s, s+win) -> uses theta[s : s + win - 2]
        out[i] = np.nanmean(theta[s : s + win - 2])
    return out
