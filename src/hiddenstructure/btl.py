"""Bradley-Terry-Luce preference fitting.

Per-agent BTL gives the preference matrix Θ (N, k); pooled BTL on all rows is
the anonymous-RLHF estimate θ̂; the utilitarian preference is θ̄_w = mean(Θ).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def fit_btl(alpha: np.ndarray, y: np.ndarray, C: float = 1.0) -> np.ndarray | None:
    """Single BTL (logistic, no intercept) fit; ``None`` if only one class present."""
    if len(np.unique(y)) < 2:
        return None
    lr = LogisticRegression(fit_intercept=False, C=C, solver="lbfgs", max_iter=2000, tol=1e-8)
    lr.fit(alpha, y)
    return lr.coef_.ravel()


def fit_per_user(
    alpha: np.ndarray, y: np.ndarray, pid: np.ndarray, min_obs: int = 20, C: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Per-agent BTL fit.

    Returns ``(agent_ids, Theta)`` for every agent with >= ``min_obs`` rows and
    both choice classes present, preserving first-seen agent order.
    """
    agent_ids, theta_rows = [], []
    for u in pd.Series(pid).unique():
        mask = pid == u
        if mask.sum() < min_obs:
            continue
        th = fit_btl(alpha[mask], y[mask], C=C)
        if th is None:
            continue
        agent_ids.append(u)
        theta_rows.append(th)
    return np.array(agent_ids), np.vstack(theta_rows)


def fit_pooled(alpha: np.ndarray, y: np.ndarray, C: float = 1.0) -> np.ndarray:
    """Pooled BTL over all rows, ignoring agent identity — the anonymous RLHF θ̂."""
    lr = LogisticRegression(fit_intercept=False, C=C, solver="lbfgs", max_iter=2000, tol=1e-8)
    lr.fit(alpha, y)
    return lr.coef_.ravel()
