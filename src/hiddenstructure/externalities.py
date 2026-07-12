"""Alignment externalities Γ — the deliverable of pipeline #1.

For a deployment mechanism with impact ψ_c and leave-one-out impacts ψ_c^(-n),

    Γ_{n,m} = θ_mᵀ ( ψ(θ_c) − ψ(θ_c^(-n)) )

Row n = the externality agent n's participation imposes on everyone; column m =
what agent m receives. :func:`gamma_for` runs the whole thing for any mechanism.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .mechanisms import Mechanism
from .model import FittedModel


def gamma_mat(psi_full: np.ndarray, psi_loo: np.ndarray, Theta: np.ndarray) -> np.ndarray:
    """Γ (N, N) from a deployment impact and its LOO impacts."""
    delta = psi_full[None, :] - psi_loo          # (N, k)
    return delta @ Theta.T                        # (N, N)


@dataclass
class GammaResult:
    mechanism: str
    Gamma: np.ndarray            # (N, N)
    psi_full: np.ndarray         # (k,)
    psi_loo: np.ndarray          # (N, k)
    model: FittedModel

    def summary(self) -> str:
        G = self.Gamma
        off = G - np.diag(np.diag(G))
        return (f"{self.mechanism:16s}  |Γ|∈[{np.abs(G).min():.3e}, {np.abs(G).max():.3e}]  "
                f"diag μ {np.diag(G).mean():+.3e}  off-diag μ {off.mean():+.3e}")


def gamma_for(model: FittedModel, mechanism: Mechanism, *,
              use_cache: bool = True, verbose: bool = False) -> GammaResult:
    """Compute the Γ matrix for one mechanism on one fitted dataset."""
    psi_full = mechanism.deploy(model)
    psi_loo = mechanism.loo(model, use_cache=use_cache, verbose=verbose)
    G = gamma_mat(psi_full, psi_loo, model.Theta)
    return GammaResult(mechanism.name, G, psi_full, psi_loo, model)


def group_summary_matrix(group_arr: np.ndarray, G_full: np.ndarray, min_n: int = 3):
    """Average Γ within each (group_n, group_m) block; diagonal excludes self-pairs.

    Returns ``(M, groups, counts, n_dropped)`` or ``None`` if fewer than two
    groups clear ``min_n``.
    """
    g = pd.Series(group_arr).where(lambda s: s.notna(), None).fillna("(missing)").to_numpy()
    counts = pd.Series(g).value_counts()
    keep = set(counts[counts >= min_n].index)
    mask = np.array([gv in keep for gv in g])
    if mask.sum() < 2:
        return None
    g_sub = g[mask]
    G_sub = G_full[np.ix_(mask, mask)]
    uniq = sorted(np.unique(g_sub).tolist())
    M = np.zeros((len(uniq), len(uniq)))
    for i, gn in enumerate(uniq):
        for j, gm in enumerate(uniq):
            block = G_sub[np.ix_(g_sub == gn, g_sub == gm)]
            if i == j:
                od = block[~np.eye(block.shape[0], dtype=bool)]
                M[i, j] = od.mean() if od.size else 0.0
            else:
                M[i, j] = block.mean()
    return M, uniq, {gv: int((g_sub == gv).sum()) for gv in uniq}, int((~mask).sum())
