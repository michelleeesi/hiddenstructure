"""Fit a :class:`DatasetBundle` into everything the mechanisms consume.

One call to :func:`fit_model` turns raw pairwise data into the fitted objects
shared by every mechanism: the per-agent preference matrix Θ, the utilitarian
mean θ̄_w, the pooled RLHF estimate θ̂, a deployment query sub-sample α_q, and
the grouping axes used to summarise Γ.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .btl import fit_btl, fit_pooled
from .loaders import DatasetBundle

DEFAULT_C = 1.0
DEFAULT_MIN_OBS = 20
DEFAULT_BETA = 1.0
DEFAULT_QUERY_SAMPLE = 5000
DEFAULT_K_GROUPS = 6


@dataclass
class FittedModel:
    """All fitted quantities for one dataset, at one β."""
    name: str
    agent_ids: np.ndarray          # (N,)
    Theta: np.ndarray              # (N, k) per-agent preferences
    theta_bar: np.ndarray          # (k,) utilitarian mean θ̄_w
    theta_rlhf: np.ndarray         # (k,) pooled BTL estimate
    weights: np.ndarray            # (N,) social weights (uniform)
    alpha_train: np.ndarray        # (M, k) all training query diffs
    y_train: np.ndarray            # (M,)
    person_id: np.ndarray          # (M,)
    alpha_q: np.ndarray            # (Z, k) deployment query sub-sample
    group_axes: dict[str, np.ndarray]  # {axis -> per-agent group label (N,)}
    beta: float
    C: float
    feature_names: list[str]

    @property
    def N(self) -> int:
        return self.Theta.shape[0]

    @property
    def k(self) -> int:
        return self.Theta.shape[1]

    @property
    def W_star(self) -> float:
        r"""Reference optimal welfare U\* = θ̄_wᵀψ* (β=∞), computed once and cached."""
        w = self.__dict__.get("_W_star")
        if w is None:
            from .impact import w_star
            w = self.__dict__["_W_star"] = w_star(self.theta_bar, self.alpha_q)
        return w


def _group_axes(
    bundle: DatasetBundle, agent_ids: np.ndarray, Theta: np.ndarray,
    k_groups: int, seed: int,
) -> dict[str, np.ndarray]:
    if bundle.demographics:
        axes = {}
        for name, ser in bundle.demographics.items():
            ser_clean = ser.where(ser.notna(), "(missing)").astype(str)
            axes[name] = np.array([ser_clean.get(a, "(missing)") for a in agent_ids])
        return axes
    Xs = StandardScaler().fit_transform(Theta)
    lab = KMeans(n_clusters=min(k_groups, len(agent_ids)), n_init=20,
                 random_state=seed).fit_predict(Xs)
    return {"kmeans": np.array([f"k{l}" for l in lab])}


def fit_model(
    bundle: DatasetBundle,
    *,
    C: float = DEFAULT_C,
    min_obs: int = DEFAULT_MIN_OBS,
    beta: float = DEFAULT_BETA,
    query_sample: int = DEFAULT_QUERY_SAMPLE,
    k_groups: int = DEFAULT_K_GROUPS,
    seed: int = 0,
) -> FittedModel:
    """Fit per-agent + pooled BTL and assemble a :class:`FittedModel`."""
    agent_ids, theta_rows = [], []
    for u in pd.Series(bundle.person_id).unique():
        mask = bundle.person_id == u
        if mask.sum() < min_obs:
            continue
        th = fit_btl(bundle.alpha[mask], bundle.y[mask], C=C)
        if th is None:
            continue
        agent_ids.append(u)
        theta_rows.append(th)
    agent_ids = np.array(agent_ids)
    Theta = np.vstack(theta_rows)
    N = Theta.shape[0]
    weights = np.full(N, 1.0 / N)
    theta_bar = weights @ Theta
    theta_rlhf = fit_pooled(bundle.alpha, bundle.y, C=C)

    rng_q = np.random.default_rng(seed)
    if bundle.alpha.shape[0] > query_sample:
        idx = rng_q.choice(bundle.alpha.shape[0], size=query_sample, replace=False)
        alpha_q = bundle.alpha[idx]
    else:
        alpha_q = bundle.alpha

    group_axes = _group_axes(bundle, agent_ids, Theta, k_groups, seed)

    return FittedModel(
        name=bundle.name, agent_ids=agent_ids, Theta=Theta, theta_bar=theta_bar,
        theta_rlhf=theta_rlhf, weights=weights, alpha_train=bundle.alpha,
        y_train=bundle.y, person_id=bundle.person_id, alpha_q=alpha_q,
        group_axes=group_axes, beta=beta, C=C, feature_names=bundle.feature_names,
    )
