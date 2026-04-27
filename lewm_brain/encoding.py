"""Ridge encoding model: model features -> per-neuron spike-count prediction.

Train on the average of all-but-one repeats; test on the held-out repeat.
Per-target ridge alpha (one alpha per unit) selected by sklearn's RidgeCV
leave-one-out across frames. Reference: Schrimpf et al. 2018 §2.3.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def fit_and_score_ridge(
    X: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    alpha_grid: list[float],
) -> dict[str, Any]:
    """Returns: {'r': (n_units,) Pearson r on held-out test data,
                 'alpha': (n_units,) selected alpha per unit,
                 'Y_pred': (n_frames, n_units) model predictions}."""
    from sklearn.linear_model import RidgeCV

    model = RidgeCV(alphas=alpha_grid, alpha_per_target=True)
    model.fit(X, Y_train)
    Y_pred = model.predict(X).astype(np.float32)

    n_units = Y_train.shape[1]
    r = np.zeros(n_units, dtype=np.float32)
    for u in range(n_units):
        yp, yt = Y_pred[:, u], Y_test[:, u]
        if yp.std() < 1e-12 or yt.std() < 1e-12:
            r[u] = np.nan
        else:
            r[u] = np.corrcoef(yp, yt)[0, 1]

    alpha = np.atleast_1d(np.asarray(model.alpha_, dtype=np.float32))
    if alpha.size == 1:
        alpha = np.full(n_units, float(alpha.item()), dtype=np.float32)
    return {"r": r, "alpha": alpha, "Y_pred": Y_pred}


def split_half_reliability(responses: np.ndarray) -> np.ndarray:
    """Per-unit split-half reliability — the noise ceiling against which a
    model's r should be compared. responses shape (n_repeats, n_frames,
    n_units). Returns (n_units,) Pearson r between two random halves of
    the repeats, averaged across n_repeats // 2 splits is unnecessary —
    one even/odd split is the conventional first pass.
    """
    R, F, U = responses.shape
    half = R // 2
    a = responses[:half].mean(axis=0)   # (F, U)
    b = responses[half:half * 2].mean(axis=0)
    rel = np.zeros(U, dtype=np.float32)
    for u in range(U):
        if a[:, u].std() < 1e-12 or b[:, u].std() < 1e-12:
            rel[u] = np.nan
        else:
            rel[u] = np.corrcoef(a[:, u], b[:, u])[0, 1]
    return rel
