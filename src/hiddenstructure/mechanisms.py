"""Deployment mechanisms.

A **mechanism** maps a fitted dataset to a deployment impact vector ψ_c, plus
the leave-one-out impacts ψ_c^(-n) used to measure alignment externalities.
Every mechanism implements the same two methods, so the externality pipeline
(``externalities.py``) is agnostic to which one it is handed:

    deploy(model) -> ψ_c        shape (k,)
    loo(model)    -> ψ_c^(-n)   shape (N, k)

Implemented: Utilitarian, RLHF, Strategyproof, BoundedHarm.
Planned (pipeline #2): PublicSpirit — see the stub at the bottom.

To add a mechanism, subclass :class:`Mechanism`, implement ``deploy``/``loo``,
and hand it to ``externalities.gamma_for``. Nothing downstream changes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.special import expit as sigmoid

from . import cache
from .impact import phi_all, phi_inv, psi, w_star
from .model import FittedModel


class Mechanism:
    """Base class. ``name`` labels plots; ``tag`` keys the LOO cache."""
    name: str = "mechanism"
    tag: str = "mech"
    #: whether the deployment impact depends on β (utilitarian/RLHF/SP: yes;
    #: welfare-constraint LP mechanisms are β=∞ boundary solutions: no).
    beta_dependent: bool = True

    def deploy(self, model: FittedModel) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def loo(self, model: FittedModel, *, use_cache: bool = True,
            verbose: bool = False) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    # Shared cache plumbing for the mechanisms whose LOO is expensive.
    def _cached_loo(self, model, compute, *, use_cache, verbose):
        if use_cache:
            hit = cache.load_psi_loo(model.name, self.tag, model.N, model.beta, model.agent_ids)
            if hit is not None:
                if verbose:
                    print(f"  {self.name} LOO: cache hit")
                return hit
        arr = compute()
        if use_cache:
            cache.save_psi_loo(model.name, self.tag, model.N, model.beta, arr, model.agent_ids)
        return arr


class Utilitarian(Mechanism):
    """Deploy the utilitarian mean θ̄_w. LOO is a closed-form mean update."""
    name = "Utilitarian"
    tag = "util"

    def deploy(self, model):
        return psi(model.theta_bar, model.alpha_q, model.beta)

    def loo(self, model, *, use_cache=True, verbose=False):
        N = model.N
        theta_loo = (N * model.theta_bar[None, :] - model.Theta) / (N - 1)
        return np.array([psi(t, model.alpha_q, model.beta) for t in theta_loo])


class RLHF(Mechanism):
    """Deploy the pooled BTL estimate θ̂. LOO via an influence-function approx."""
    name = "RLHF"
    tag = "rlhf"

    def deploy(self, model):
        return psi(model.theta_rlhf, model.alpha_q, model.beta)

    def loo(self, model, *, use_cache=True, verbose=False):
        k = model.k
        sigma_R = sigmoid(model.alpha_train @ model.theta_rlhf)
        v_R = sigma_R * (1.0 - sigma_R)
        H = np.eye(k) + model.C * (model.alpha_train.T * v_R) @ model.alpha_train
        H_inv = np.linalg.solve(H, np.eye(k))
        G = np.zeros((model.N, k))
        for n_idx, a in enumerate(model.agent_ids):
            rows = model.person_id == a
            G[n_idx] = ((sigma_R[rows] - model.y_train[rows])[:, None]
                        * model.alpha_train[rows]).sum(axis=0)
        theta_loo = model.theta_rlhf[None, :] + model.C * (G @ H_inv.T)
        return np.array([psi(t, model.alpha_q, model.beta) for t in theta_loo])


class Strategyproof(Mechanism):
    """Random-dictatorship strategyproof deployment: θ_sp = φ⁻¹(mean per-agent impact).

    LOO re-inverts on the remaining agents; warm-started and cached because each
    inversion is an L-BFGS solve.
    """
    name = "Strategyproof"
    tag = "sp"

    def deploy(self, model):
        phi_avg = phi_all(model.Theta, model.beta, model.alpha_q).mean(axis=0)
        theta_sp = phi_inv(phi_avg, model.beta, model.alpha_q, theta_init=model.theta_bar.copy())
        return psi(theta_sp, model.alpha_q, model.beta)

    def loo(self, model, *, use_cache=True, verbose=False):
        def compute():
            phi_avg = phi_all(model.Theta, model.beta, model.alpha_q).mean(axis=0)
            theta_sp_warm = phi_inv(phi_avg, model.beta, model.alpha_q,
                                    theta_init=model.theta_bar.copy())
            arr = np.zeros((model.N, model.k))
            t0 = time.time()
            for n_idx in range(model.N):
                keep = np.r_[:n_idx, n_idx + 1:model.N]
                phi_avg_loo = phi_all(model.Theta[keep], model.beta, model.alpha_q).mean(axis=0)
                th_loo = phi_inv(phi_avg_loo, model.beta, model.alpha_q,
                                 theta_init=theta_sp_warm.copy())
                arr[n_idx] = psi(th_loo, model.alpha_q, model.beta)
                if verbose and ((n_idx + 1) % 100 == 0 or n_idx == 0):
                    print(f"    SP LOO {n_idx + 1:4d}/{model.N}  {time.time() - t0:5.1f}s")
            return arr
        return self._cached_loo(model, compute, use_cache=use_cache, verbose=verbose)


def _hub_of(T: np.ndarray, alpha_dep: np.ndarray) -> np.ndarray:
    """h_Ψ̄(θ_n) — max achievable welfare for each agent in the impact set Ψ̄."""
    return 0.5 * np.abs(alpha_dep @ T.T).mean(axis=0)


def solve_constrained_lp(theta_obj, normals, thresholds, alpha_dep, return_full=False):
    r"""max θ_objᵀψ  s.t.  θ_nᵀψ ≥ threshold_n  and  ψ ∈ Ψ̄ (β=∞ boundary)."""
    Z, _ = alpha_dep.shape
    c = -(alpha_dep @ theta_obj)
    A_ub = -(normals @ alpha_dep.T)
    b_ub = -2.0 * Z * thresholds
    bounds = [(-1.0, 1.0)] * Z
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    psi_opt = 0.5 * (alpha_dep.T @ res.x) / Z
    if return_full:
        eta = np.maximum(0.0, -res.ineqlin.marginals)
        return psi_opt, eta, res
    return psi_opt


@dataclass
class BoundedHarm(Mechanism):
    """Maximise utilitarian welfare subject to a per-agent harm floor.

    Two floor forms (pass exactly one):

    * ``eps``  — per-agent floor  θ_nᵀψ ≥ −ε·h_Ψ̄(θ_n)  (fraction of each agent's
      own best achievable welfare; the main-text default).
    * ``b``    — absolute floor  θ_nᵀψ ≥ b·U\*  shared by all agents, in U/U\*
      units (e.g. ``b=-0.5`` = no agent below −0.5·U\*). Matches the
      ``empirics_general`` "BH b=…" panels.
    """
    beta_dependent = False
    eps: float | None = None
    b: float | None = None

    def __post_init__(self):
        if self.eps is not None and self.b is not None:
            raise ValueError("BoundedHarm: pass at most one of eps or b")
        if self.eps is None and self.b is None:
            self.eps = 0.05          # default: 5% per-agent floor
        if self.b is not None:
            self.name = f"Bounded harm  b={self.b}"
            self.tag = f"bh_BHb{self.b}"
        else:
            self.name = f"Bounded harm  ε={self.eps}"
            self.tag = f"bh_BHeps{self.eps}"

    def _thresholds(self, theta_bar, T, alpha_dep):
        if self.b is not None:
            return self.b * w_star(theta_bar, alpha_dep) * np.ones(T.shape[0])
        return -self.eps * _hub_of(T, alpha_dep)

    def deploy(self, model):
        thr = self._thresholds(model.theta_bar, model.Theta, model.alpha_q)
        return solve_constrained_lp(model.theta_bar, model.Theta, thr, model.alpha_q)

    def loo(self, model, *, use_cache=True, verbose=False):
        def compute():
            N = model.N
            arr = np.zeros((N, model.k))
            t0 = time.time()
            for n_idx in range(N):
                keep = np.r_[:n_idx, n_idx + 1:N]
                T_keep = model.Theta[keep]
                theta_bar_loo = (N * model.theta_bar - model.Theta[n_idx]) / (N - 1)
                thr_loo = self._thresholds(theta_bar_loo, T_keep, model.alpha_q)
                arr[n_idx] = solve_constrained_lp(theta_bar_loo, T_keep, thr_loo, model.alpha_q)
                if verbose and ((n_idx + 1) % 100 == 0 or n_idx == 0):
                    print(f"    BH LOO {n_idx + 1:4d}/{N}  {time.time() - t0:5.1f}s")
            return arr
        return self._cached_loo(model, compute, use_cache=use_cache, verbose=verbose)


@dataclass
class PublicSpirit(Mechanism):
    r"""Public-spirit alignment (Flanigan et al. 2023).

    Each agent tolerates personal harm in proportion to social benefit, controlled
    by a public-spirit parameter ``gamma`` ≥ 0. The deployment solves

        maximise   θ̄_wᵀ ψ           over ψ ∈ Ψ̄
        s.t.       γ_nᵀ ψ ≥ γ_nᵀ ψ₀   for every agent n,

    where ``γ_n = gamma·θ̄_w + (1 − gamma)·θ_n`` and ψ₀ is a reference impact.
    Equivalently, agent n accepts a loss only when the social gain covers it:
    ``(1 − γ)·θ_nᵀ(ψ₀ − ψ) ≤ γ·θ̄_wᵀ(ψ − ψ₀)``.

    Limits: ``gamma = 0`` is individual rationality (no agent worse than ψ₀);
    ``gamma = 1`` collapses to unconstrained utilitarian.

    ``psi_0`` is the reference impact. ``None`` uses the **random model** (ψ₀ = 0,
    since θ = 0 gives ψ = 0), so the constraint is "no worse than random." It is
    treated as a fixed exogenous baseline (not recomputed under leave-one-out).
    Same constrained-LP machinery as :class:`BoundedHarm`.
    """
    beta_dependent = False
    gamma: float = 0.0
    psi_0: np.ndarray | None = None

    def __post_init__(self):
        self.name = f"Public spirit  γ={self.gamma}"
        self.tag = f"ps_g{self.gamma}"

    def _psi0(self, model):
        return np.zeros(model.k) if self.psi_0 is None else np.asarray(self.psi_0, float)

    def _normals(self, theta_bar, Theta):
        """Per-agent constraint normals γ_n = γ·θ̄_w + (1−γ)·θ_n → (N, k)."""
        return self.gamma * theta_bar[None, :] + (1.0 - self.gamma) * Theta

    def deploy(self, model):
        tb = model.theta_bar
        normals = self._normals(tb, model.Theta)
        thresholds = normals @ self._psi0(model)
        return solve_constrained_lp(tb, normals, thresholds, model.alpha_q)

    def loo(self, model, *, use_cache=True, verbose=False):
        def compute():
            N = model.N
            psi0 = self._psi0(model)
            arr = np.zeros((N, model.k))
            t0 = time.time()
            for n_idx in range(N):
                keep = np.r_[:n_idx, n_idx + 1:N]
                tb_loo = (N * model.theta_bar - model.Theta[n_idx]) / (N - 1)
                normals = self._normals(tb_loo, model.Theta[keep])
                thresholds = normals @ psi0
                arr[n_idx] = solve_constrained_lp(tb_loo, normals, thresholds, model.alpha_q)
                if verbose and ((n_idx + 1) % 100 == 0 or n_idx == 0):
                    print(f"    PS LOO {n_idx + 1:4d}/{N}  {time.time() - t0:5.1f}s")
            return arr
        return self._cached_loo(model, compute, use_cache=use_cache, verbose=verbose)


#: The three core mechanisms compared throughout the paper.
CORE_MECHANISMS = [Utilitarian(), RLHF(), Strategyproof()]

#: Welfare-constraint mechanisms (bounded harm & public spirit share an LP form).
WELFARE_MECHANISMS = [BoundedHarm(eps=0.05), PublicSpirit(gamma=0.5)]
