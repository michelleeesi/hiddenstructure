"""Core preference/impact primitives.

These are the objects from Sections 1-3 of the paper, defined **once** here
(the notebooks used to redefine them in nearly every cell):

    ψ(θ, β)      impact vector of a deployment preference θ under query dist. α
    φ_all        stacked impact of many agents' preferences (used by the
                 strategyproof mechanism, whose deployment matches mean impact)
    φ⁻¹          invert an impact target back to a preference θ

Convention: ``alpha_dep`` is the (Z, k) matrix of deployment query feature
differences; every expectation is a plain mean over its rows.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit as sigmoid


def psi(theta: np.ndarray, alpha_dep: np.ndarray, beta: float = 1.0) -> np.ndarray:
    r"""ψ(θ, β) = E_z[ α_z · (σ(β·α_zᵀθ) − ½) ]  →  shape (k,)."""
    return ((sigmoid(beta * (alpha_dep @ theta)) - 0.5)[:, None] * alpha_dep).mean(axis=0)


def psi_star_inf(theta_bar: np.ndarray, alpha_dep: np.ndarray) -> np.ndarray:
    r"""Welfare-optimal impact at β=∞:  ½ E_z[ α_z · sign(α_zᵀθ̄_w) ]  (Prop. 2)."""
    return 0.5 * (alpha_dep * np.sign(alpha_dep @ theta_bar)[:, None]).mean(axis=0)


def phi_all(theta_agents: np.ndarray, beta: float, alpha_dep: np.ndarray) -> np.ndarray:
    """Per-agent impact for a stack of preferences θ (N, k) → (N, k)."""
    logits = alpha_dep @ theta_agents.T
    return ((sigmoid(beta * logits) - 0.5).T @ alpha_dep) / len(alpha_dep)


def phi_inv(
    target: np.ndarray,
    beta: float,
    alpha_dep: np.ndarray,
    theta_init: np.ndarray | None = None,
) -> np.ndarray:
    """Solve for θ whose impact ψ(θ, β) matches ``target`` (L-BFGS on ‖ψ−target‖²)."""
    if theta_init is None:
        theta_init = np.zeros(alpha_dep.shape[1])

    def loss_and_grad(theta):
        s = sigmoid(beta * (alpha_dep @ theta))
        phi_t = ((s - 0.5)[:, None] * alpha_dep).mean(axis=0)
        residual = phi_t - target
        v = s * (1 - s)
        J = (alpha_dep * (beta * v)[:, None]).T @ alpha_dep / len(alpha_dep)
        return 0.5 * float(residual @ residual), J @ residual

    res = minimize(
        loss_and_grad, theta_init, jac=True, method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-8},
    )
    return res.x


def welfare_per_agent(
    theta_c: np.ndarray, Theta: np.ndarray, alpha_dep: np.ndarray, beta: float = 1.0
) -> np.ndarray:
    r"""W_n(θ_c) = θ_nᵀ ψ(θ_c) for every agent n → shape (N,)."""
    return Theta @ psi(theta_c, alpha_dep, beta)


def welfare_social(
    theta_c: np.ndarray, theta_bar: np.ndarray, alpha_dep: np.ndarray, beta: float = 1.0
) -> float:
    r"""Utilitarian social welfare  θ̄_wᵀ ψ(θ_c)."""
    return float(theta_bar @ psi(theta_c, alpha_dep, beta))


def w_star(theta_bar: np.ndarray, alpha_dep: np.ndarray) -> float:
    r"""Reference optimal welfare U\* = θ̄_wᵀ ψ*, with the β=∞ sign-based ψ*.

    Uses the un-halved ψ* = E_z[α_z·sign(α_zᵀθ̄)] (matching ``empirics_general``),
    so welfare ratios U_n/U\* are on the same scale as :func:`welfare_per_agent`.
    """
    psi_star = (np.sign(alpha_dep @ theta_bar)[:, None] * alpha_dep).mean(axis=0)
    return float(theta_bar @ psi_star)
