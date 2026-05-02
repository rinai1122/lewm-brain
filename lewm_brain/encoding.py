"""Ridge encoding model: model features -> per-neuron spike-count prediction.

Default split: train on a contiguous chunk of clips (averaged over all
repeats), test on a held-out chunk of clips at the other end of the
movie. This forces features to generalize across stimulus content; the
old "last repeat as test" split was insensitive to feature quality
because adjacent sliding-stride-1 clips share 63/64 of their input,
letting ridge cheat via temporal smoothing of the train mean.

Per-target alpha (one alpha per unit) selected by sklearn's RidgeCV
leave-one-out across train clips. Reference: Schrimpf et al. 2018 §2.3.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def fit_and_score_ridge(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    alpha_grid: list[float],
) -> dict[str, Any]:
    """Returns: {'r': (n_units,) Pearson r on held-out test clips,
                 'alpha': (n_units,) selected alpha per unit,
                 'Y_pred': (n_test, n_units) model predictions on test}."""
    from sklearn.linear_model import RidgeCV

    model = RidgeCV(alphas=alpha_grid, alpha_per_target=True)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test).astype(np.float32)

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
