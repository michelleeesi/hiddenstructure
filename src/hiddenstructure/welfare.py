"""Welfare diagnostics — the mechanism-welfare view (ported from empirics_general).

Complements the externality (Γ) view: how a mechanism's per-agent welfare
U_n(θ_c; β)/U* behaves as the sigmoid temperature β and the agent population vary.

    beta_sweep         per-agent welfare vs β for each mechanism  (§4)
    sp_subset_sweep    E[U_sp]/U* vs subset size, bootstrap + sequential  (§15)
    distortion         Definition-27 distortion W*/E[W_sp]
    protection_frontier  welfare vs worst-off as public-spirit γ / bounded-harm ε vary

Figure builders (house style from ``figures``) reproduce the paper's welfare
figures; see the coverage table in the README.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np
import matplotlib.pyplot as plt

from . import figures as F
from .impact import phi_all, phi_inv, welfare_per_agent
from .mechanisms import BoundedHarm, Mechanism, PublicSpirit
from .model import FittedModel

DEFAULT_BETAS = np.logspace(-2, 2, 80)   # β = 1 at the geometric centre


def gini(values) -> float:
    v = np.asarray(values, dtype=float)
    v = v - v.min() + 1e-12
    v = np.sort(v)
    n = len(v)
    return float((2 * np.arange(1, n + 1) - n - 1) @ v / (n * v.sum()))


@dataclass
class WelfareSweep:
    model: FittedModel
    betas: np.ndarray
    W_star: float
    W_per_agent: dict           # mech name -> (n_beta, N) welfare ratios
    agg: dict = field(default_factory=dict)   # mech -> {mean,max,min,var,gini}

    def _aggregate(self):
        self.agg = {
            lbl: {"mean": W.mean(axis=1), "max": W.max(axis=1), "min": W.min(axis=1),
                  "var": W.var(axis=1), "gini": np.array([gini(W[j]) for j in range(len(W))])}
            for lbl, W in self.W_per_agent.items()
        }
        return self


def beta_sweep(model: FittedModel, betas=DEFAULT_BETAS,
               static_mechanisms: list[Mechanism] | None = None,
               static_deploys: dict | None = None,
               verbose: bool = False) -> WelfareSweep:
    """Per-agent welfare ratio U_n(θ_c; β)/U* over a β grid (§4).

    Core mechanisms: utilitarian (θ̄_w), RLHF (θ̂, fixed), strategyproof (φ⁻¹ of the
    mean impact, re-solved and warm-started at each β). ``static_mechanisms`` (e.g.
    ``BoundedHarm``/``PublicSpirit``) are pinned by calibrating θ_c = φ⁻¹(ψ_deploy)
    at β=1 and then swept — this is how the bounded-harm welfare panels are drawn.
    """
    betas = np.asarray(betas, float)
    Theta, aq = model.Theta, model.alpha_q
    tb, trlhf = model.theta_bar, model.theta_rlhf
    W_star = model.W_star

    W = {m: np.zeros((len(betas), model.N)) for m in ("Utilitarian", "RLHF", "Strategyproof")}
    theta_sp_prev = tb.copy()
    t0 = time.time()
    for j, b in enumerate(betas):
        phi_avg = phi_all(Theta, b, aq).mean(axis=0)
        theta_sp = phi_inv(phi_avg, b, aq, theta_init=theta_sp_prev)
        theta_sp_prev = theta_sp.copy()
        W["Utilitarian"][j] = welfare_per_agent(tb, Theta, aq, b) / W_star
        W["RLHF"][j] = welfare_per_agent(trlhf, Theta, aq, b) / W_star
        W["Strategyproof"][j] = welfare_per_agent(theta_sp, Theta, aq, b) / W_star
        if verbose and ((j + 1) % 20 == 0 or j == 0):
            print(f"  β-sweep {j + 1:3d}/{len(betas)}  {time.time() - t0:5.1f}s")

    static_deploys = static_deploys or {}
    for mech in static_mechanisms or []:
        psi_lp = static_deploys[mech.name] if mech.name in static_deploys else mech.deploy(model)
        theta_c = phi_inv(psi_lp, 1.0, aq, theta_init=np.zeros(model.k))
        Wm = np.zeros((len(betas), model.N))
        for j, b in enumerate(betas):
            Wm[j] = welfare_per_agent(theta_c, Theta, aq, b) / W_star
        W[mech.name] = Wm

    return WelfareSweep(model, betas, W_star, W)._aggregate()


def _sp_welfare_from_signs(signs_sub, T_sub, alpha_dep):
    """Saturated-β SP welfare ratio U_sp/U* for a subset, via precomputed signs (§15)."""
    th_bar = T_sub.mean(axis=0)
    psi_sp = 0.5 * (signs_sub.mean(axis=1)[:, None] * alpha_dep).mean(axis=0)
    psi_st = 0.5 * (np.sign(alpha_dep @ th_bar)[:, None] * alpha_dep).mean(axis=0)
    U_star = float(th_bar @ psi_st)
    U_sp = float(th_bar @ psi_sp)
    return U_sp / U_star if U_star > 0 else np.nan


@dataclass
class SubsetSweep:
    model: FittedModel
    k_vals: list
    boot: np.ndarray            # (n_boot, len(k)) U_sp/U* ratios
    seq: np.ndarray             # (n_order, len(k)) U_sp/U* ratios


def _subset_sweep_arrays(Theta, alpha_q, n_boot=100, n_order=100, n_k=10):
    """Bootstrap + sequential SP welfare ratios over subset size, from arrays (§15)."""
    N = Theta.shape[0]
    signs_all = np.sign(alpha_q @ Theta.T)                  # (M, N)
    k_vals = np.unique(np.linspace(2, N, min(n_k, N)).astype(int)).tolist()

    rng_b = np.random.default_rng(1)
    boot = np.empty((n_boot, len(k_vals)))
    for ki, k in enumerate(k_vals):
        for b in range(n_boot):
            idx = rng_b.choice(N, size=k, replace=False)
            boot[b, ki] = _sp_welfare_from_signs(signs_all[:, idx], Theta[idx], alpha_q)

    # Sequential subsets are nested (order[:k]); cumulative sums make each ordering
    # O(M·N) once instead of O(M·k) per k — a large win for Moral Machine (N=5000).
    rng_s = np.random.default_rng(2)
    seq = np.empty((n_order, len(k_vals)))
    for o in range(n_order):
        order = rng_s.permutation(N)
        sign_csum = np.cumsum(signs_all[:, order], axis=1)     # (M, N)
        theta_csum = np.cumsum(Theta[order], axis=0)           # (N, k_dim)
        for ki, k in enumerate(k_vals):
            mean_signs = sign_csum[:, k - 1] / k
            th_bar = theta_csum[k - 1] / k
            psi_sp = 0.5 * (mean_signs[:, None] * alpha_q).mean(axis=0)
            psi_st = 0.5 * (np.sign(alpha_q @ th_bar)[:, None] * alpha_q).mean(axis=0)
            U_star = float(th_bar @ psi_st)
            seq[o, ki] = float(th_bar @ psi_sp) / U_star if U_star > 0 else np.nan
    return k_vals, boot, seq


def sp_subset_sweep(model: FittedModel, n_boot: int = 100, n_order: int = 100,
                    n_k: int = 10) -> SubsetSweep:
    """Strategyproof welfare ratio vs agent-subset size — bootstrap + sequential (§15).

    Feeds both the *expected-welfare* figure (ratio) and the *distortion* figure
    (its reciprocal). Uses saturated-β SP welfare via precomputed choice signs.
    """
    k_vals, boot, seq = _subset_sweep_arrays(model.Theta, model.alpha_q, n_boot, n_order, n_k)
    return SubsetSweep(model, k_vals, boot, seq)


def distortion(theta_agents, lambdas, alpha_queries, weights=None) -> dict:
    r"""Definition-27 distortion of the random-dictatorship mechanism µ_λ:
    ``Dist = W*(θ) / E[W(µ_λ)]``."""
    theta_agents = np.asarray(theta_agents, float)
    lambdas = np.asarray(lambdas, float)
    alpha_queries = np.asarray(alpha_queries, float)
    N = theta_agents.shape[0]
    weights = np.ones(N) / N if weights is None else np.asarray(weights, float)
    weights = weights / weights.sum()
    lambdas = lambdas / lambdas.sum()

    theta_bar = weights @ theta_agents
    signs_n = np.sign(alpha_queries @ theta_agents.T)                       # (Q, N)
    psi_n_star = (signs_n[:, :, None] * alpha_queries[:, None, :]).mean(axis=0)
    psi_sp = lambdas @ psi_n_star
    psi_star = (np.sign(alpha_queries @ theta_bar)[:, None] * alpha_queries).mean(axis=0)
    W_star = 0.5 * float(theta_bar @ psi_star)
    W_sp = 0.5 * float(theta_bar @ psi_sp)
    return dict(distortion=W_star / W_sp if W_sp > 0 else np.inf,
                W_star=W_star, W_sp=W_sp, psi_n_star=psi_n_star, theta_bar=theta_bar)


@dataclass
class Frontier:
    model: FittedModel
    param_name: str             # "gamma" or "eps"
    params: np.ndarray
    social: np.ndarray          # θ̄ᵀψ / U*  at each param
    worst_off: np.ndarray       # min_n θ_nᵀψ / U*  at each param


def protection_frontier(model: FittedModel, mechanism: str = "public_spirit",
                        params=None) -> Frontier:
    """Welfare–protection frontier as public-spirit γ (or bounded-harm ε) varies.

    Returns social welfare and worst-off agent welfare (both /U*) at each parameter,
    tracing the trade-off: protecting the worst-off costs social welfare.
    """
    tb, Theta, aq = model.theta_bar, model.Theta, model.alpha_q
    U = model.W_star
    if mechanism == "public_spirit":
        params = np.linspace(0.0, 1.0, 11) if params is None else np.asarray(params, float)
        make = lambda p: PublicSpirit(gamma=float(p))
        name = "gamma"
    elif mechanism == "bounded_harm":
        params = np.linspace(0.0, 0.5, 11) if params is None else np.asarray(params, float)
        make = lambda p: BoundedHarm(eps=float(p))
        name = "eps"
    else:
        raise ValueError(mechanism)

    social, worst = [], []
    for p in params:
        psi = make(p).deploy(model)
        social.append(float(tb @ psi) / U)
        worst.append(float((Theta @ psi).min()) / U)
    return Frontier(model, name, np.asarray(params, float),
                    np.asarray(social), np.asarray(worst))


# --------------------------------------------------------------------------- #
# Figure builders (house style)
# --------------------------------------------------------------------------- #
def _line_styles():
    from matplotlib.lines import Line2D
    return [Line2D([0], [0], color="0.25", lw=2.2, ls="-", label="best-off"),
            Line2D([0], [0], color="0.25", lw=2.2, ls=":", label="worst-off"),
            Line2D([0], [0], color="0.25", lw=1.5, ls="--", label="mean welfare")]


def fig_best_worst_mean(sweep: WelfareSweep, mechanisms=None, ncols: int | None = None):
    """Best-off / worst-off / mean welfare vs β, one panel per mechanism.

    ``ncols`` wraps the panels into a grid (e.g. 3 → the paper's 2×3 layout when
    six mechanisms are swept).
    """
    with F.use_paper_style():
        mechs = mechanisms or list(sweep.W_per_agent)
        n = len(mechs)
        ncols = ncols or n
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows),
                                 sharey=True, squeeze=False)
        flat = axes.flat
        for ax, lbl in zip(flat, mechs):
            c = F.mech_color(lbl)
            rec = sweep.agg[lbl]
            ax.plot(sweep.betas, sweep.W_per_agent[lbl], color=c, alpha=0.04, lw=0.3)
            ax.plot(sweep.betas, rec["max"], color=c, lw=2.2, ls="-")
            ax.plot(sweep.betas, rec["min"], color=c, lw=2.2, ls=":")
            ax.plot(sweep.betas, rec["mean"], color=c, lw=1.5, ls="--")
            ax.axhline(0, color="0.6", lw=0.9, alpha=0.6)
            ax.set_xscale("log")
            ax.set_title(lbl, fontweight="bold")
            ax.grid(False)
        for ax in list(flat)[n:]:
            ax.set_visible(False)
        for r in range(nrows):
            axes[r, 0].set_ylabel(r"$U_n(\theta_c;\beta)\,/\,U^*$")
        fig.tight_layout(rect=[0, 0.10, 1, 1])
        fig.legend(handles=_line_styles(), loc="lower center", ncol=3,
                   bbox_to_anchor=(0.5, 0.05))
        fig.supxlabel(r"Sigmoid Temperature $\beta$", y=0.01, fontsize=F.SUPLABEL_SIZE)
        return fig


def welfare_distribution(model: FittedModel, mechanisms: list[Mechanism],
                         beta: float = 100.0, deploys: dict | None = None) -> dict:
    """Per-agent welfare ratio U_n/U* under each mechanism at a fixed β.

    β-parameterised mechanisms (utilitarian/RLHF/strategyproof) are evaluated at
    ``beta``; welfare-constraint mechanisms (bounded harm / public spirit) deploy a
    boundary impact and ignore β (so they're deployed on the base model, letting
    ``deploys={name: ψ}`` supply a precomputed impact and avoid re-solving the LP).
    Returns ``{mechanism name -> (N,) U_n/U*}``.
    """
    mb = replace(model, beta=beta)
    U = model.W_star
    deploys = deploys or {}
    out = {}
    for mech in mechanisms:
        if mech.name in deploys:
            psi = deploys[mech.name]
        else:
            psi = mech.deploy(mb if getattr(mech, "beta_dependent", True) else model)
        out[mech.name] = (model.Theta @ psi) / U
    return out


def _floor_of(mech: Mechanism):
    """The U/U* floor to mark on a distribution panel, or None."""
    if isinstance(mech, BoundedHarm) and mech.b is not None:
        return mech.b
    return None


def fig_welfare_distribution(model: FittedModel, mechanisms: list[Mechanism],
                             beta: float = 100.0, reference: str = "Utilitarian",
                             ncols: int = 3, deploys: dict | None = None):
    """Distribution of individual welfare U_n/U* under each mechanism at fixed β.

    The paper's Fig 2: a density per mechanism; the faint gray curve repeats the
    reference (utilitarian) distribution, and bounded-harm panels mark the floor.
    """
    from scipy.stats import gaussian_kde
    with F.use_paper_style():
        dist = welfare_distribution(model, mechanisms, beta=beta, deploys=deploys)
        names = list(dist)
        lo = min(v.min() for v in dist.values())
        hi = max(v.max() for v in dist.values())
        grid = np.linspace(lo - 0.1, hi + 0.1, 400)

        def kde(v):
            return gaussian_kde(v)(grid) if np.ptp(v) > 1e-9 else np.zeros_like(grid)

        ref = kde(dist[reference]) if reference in dist else None
        n = len(names)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.4 * nrows),
                                 sharex=True, squeeze=False)
        flat = list(axes.flat)
        for ax, (mech, name) in zip(flat, zip(mechanisms, names)):
            c = F.mech_color(name)
            if ref is not None:
                ax.fill_between(grid, ref, color="0.8", alpha=0.6, zorder=1)
            d = kde(dist[name])
            ax.fill_between(grid, d, color=c, alpha=0.55, zorder=2)
            ax.plot(grid, d, color=c, lw=1.2, zorder=3)
            fl = _floor_of(mech)
            if fl is not None:
                ax.axvline(fl, color="#c0392b", ls=":", lw=1.4)
            ax.axvline(0, color="0.6", lw=0.8, alpha=0.6)
            ax.set_yticks([])
            ax.set_title(name, fontweight="bold")
        for ax in flat[n:]:
            ax.set_visible(False)
        for r in range(nrows):
            axes[r, 0].set_ylabel("Density")
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        fig.supxlabel(r"$U_n / U^*$", y=0.01, fontsize=F.SUPLABEL_SIZE)
        return fig


def fig_welfare_variance(sweep: WelfareSweep, mechanisms=None):
    """Cross-agent welfare variance vs β."""
    with F.use_paper_style():
        mechs = mechanisms or list(sweep.W_per_agent)
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for lbl in mechs:
            ax.plot(sweep.betas, sweep.agg[lbl]["var"], color=F.mech_color(lbl),
                    lw=2.0, label=lbl)
        ax.set_xscale("log")
        ax.set_xlabel(r"Sigmoid Temperature $\beta$")
        ax.set_ylabel(r"$\mathrm{Var}_n[\,U_n / U^*\,]$")
        ax.grid(False)
        ax.legend()
        ax.set_title(f"Cross-agent welfare variance vs β — {sweep.model.name}",
                     fontweight="bold")
        return fig


def fig_per_agent_welfare(sweep: WelfareSweep, mechanisms=None):
    """Per-agent welfare envelope U_n/U* vs β (all agents faint + best/worst/mean)."""
    with F.use_paper_style():
        mechs = mechanisms or list(sweep.W_per_agent)
        n = len(mechs)
        fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4), sharey=True, squeeze=False)
        for ax, lbl in zip(axes[0], mechs):
            c = F.mech_color(lbl)
            ax.plot(sweep.betas, sweep.W_per_agent[lbl], color=c, alpha=0.08, lw=0.4)
            ax.plot(sweep.betas, sweep.agg[lbl]["mean"], color=c, lw=2.0)
            ax.axhline(0, color="0.6", lw=0.9, alpha=0.5)
            ax.set_xscale("log")
            ax.set_xlabel(r"Sigmoid Temperature $\beta$")
            ax.set_title(lbl, fontweight="bold")
            ax.grid(False)
        axes[0, 0].set_ylabel(r"$U_n / U^*$")
        fig.suptitle(f"Per-agent welfare vs β — {sweep.model.name}", fontweight="bold")
        fig.tight_layout()
        return fig


def fig_preference_distribution(model: FittedModel):
    """Per-feature θ_n distribution (violin, mean bar) — shows preference spread."""
    with F.use_paper_style():
        Theta, feats = model.Theta, model.feature_names
        k = Theta.shape[1]
        fig, ax = plt.subplots(figsize=(max(6, 0.5 * k + 3), 4.6))
        parts = ax.violinplot([Theta[:, i] for i in range(k)], positions=range(k),
                              showmedians=False, showextrema=False, widths=0.7)
        for pc in parts["bodies"]:
            pc.set_facecolor(F.MECH_COLORS["utilitarian"]); pc.set_alpha(0.5)
        for i, m in enumerate(Theta.mean(axis=0)):
            ax.plot([i - 0.3, i + 0.3], [m, m], color="0.1", lw=2.0, zorder=4)
        ax.axhline(0, color="0.6", ls="--", lw=0.7)
        ax.set_xticks(range(k))
        ax.set_xticklabels(feats, rotation=40, ha="right", fontsize=8 if k <= 10 else 6)
        ax.set_ylabel(r"$\theta_n$ Value")
        ax.set_title(f"Per-feature preference distribution — {model.name}  "
                     f"(N={model.N}, k={k})", fontweight="bold")
        fig.tight_layout()
        return fig


def _subset_panels(sweep: SubsetSweep, transform, ylabel, title, ref):
    with F.use_paper_style():
        k = sweep.k_vals
        blue = F.MECH_COLORS["utilitarian"]
        red = F.MECH_COLORS["rlhf"]
        boot = transform(sweep.boot)
        seq = transform(sweep.seq)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        axes[0].fill_between(k, np.nanpercentile(boot, 5, 0), np.nanpercentile(boot, 95, 0),
                             alpha=0.18, color=blue, label="5–95th pct")
        axes[0].plot(k, np.nanmean(boot, 0), "o-", color=blue, lw=2, ms=5, label="mean")
        axes[0].set_title("bootstrap (random subsets)", fontweight="bold")
        axes[0].legend()
        for path in seq:
            axes[1].plot(k, path, color=blue, alpha=0.05, lw=0.5)
        axes[1].plot(k, np.nanmean(seq, 0), color=blue, lw=2.4, label="mean across orderings")
        axes[1].set_title("sequential addition", fontweight="bold")
        axes[1].legend()
        for ax in axes:
            ax.axhline(ref, color="0.6", ls="--", lw=0.8)
            ax.set_xscale("log")
            ax.set_xlabel(r"Number of Agents $k$")
            ax.set_ylabel(ylabel)
            ax.grid(False)
        fig.suptitle(f"{title} — {sweep.model.name}", fontweight="bold")
        fig.tight_layout()
        return fig


def fig_expected_welfare_subset(sweep: SubsetSweep):
    """E[U_sp]/U* vs subset size (bootstrap + sequential)."""
    return _subset_panels(sweep, lambda a: a, r"$E[U_{\rm sp}]\,/\,U^*$",
                          "Expected SP welfare vs subset size", ref=1.0)


def fig_distortion_subset(sweep: SubsetSweep):
    """Distortion W*/E[W_sp] vs subset size (reciprocal of the welfare ratio)."""
    return _subset_panels(sweep, lambda a: 1.0 / a, r"distortion $W^*/E[W_{\rm sp}]$",
                          "SP distortion vs subset size", ref=1.0)


def fig_protection_frontier(frontier: Frontier):
    """Welfare–protection trade-off: social welfare & worst-off vs γ (or ε)."""
    with F.use_paper_style():
        c_soc = F.MECH_COLORS["utilitarian"]
        c_worst = F.MECH_COLORS["public"] if frontier.param_name == "gamma" \
            else F.MECH_COLORS["bounded"]
        sym = r"$\gamma$" if frontier.param_name == "gamma" else r"$\varepsilon$"
        mech = "public spirit" if frontier.param_name == "gamma" else "bounded harm"
        fig, ax = plt.subplots(figsize=(7, 4.6))
        ax.plot(frontier.params, frontier.social, "o-", color=c_soc, lw=2.2,
                label=r"social welfare $\bar\theta_w^\top\psi / U^*$")
        ax.plot(frontier.params, frontier.worst_off, "s--", color=c_worst, lw=2.2,
                label=r"worst-off $\min_n\theta_n^\top\psi / U^*$")
        ax.axhline(0, color="0.6", lw=0.9, alpha=0.6)
        ax.set_xlabel(f"{mech.title()} Parameter {sym}")
        ax.set_ylabel(r"Welfare / $U^*$")
        ax.grid(False)
        ax.legend()
        ax.set_title(f"Welfare–protection frontier — {mech} — {frontier.model.name}",
                     fontweight="bold")
        fig.tight_layout()
        return fig


def generate_welfare_figures(model: FittedModel, outdir="figures",
                             formats=("png",), prefix: str | None = None,
                             betas=DEFAULT_BETAS, static_mechanisms=None,
                             subset: bool = True, verbose: bool = False) -> dict:
    """Run the sweeps and save the full welfare figure set for one dataset."""
    slug = (prefix or model.name).lower().replace(" ", "_")
    sweep = beta_sweep(model, betas=betas, static_mechanisms=static_mechanisms, verbose=verbose)
    manifest = {}
    manifest["welfare_best_worst"] = F.save_fig(fig_best_worst_mean(sweep),
                                                f"{slug}_welfare_best_worst", outdir, formats, close=True)
    manifest["welfare_variance"] = F.save_fig(fig_welfare_variance(sweep),
                                              f"{slug}_welfare_variance", outdir, formats, close=True)
    manifest["welfare_per_agent"] = F.save_fig(fig_per_agent_welfare(sweep),
                                               f"{slug}_welfare_per_agent", outdir, formats, close=True)
    manifest["preference_distribution"] = F.save_fig(fig_preference_distribution(model),
                                                     f"{slug}_preference_distribution", outdir, formats, close=True)
    fr = protection_frontier(model, "public_spirit")
    manifest["protection_frontier"] = F.save_fig(fig_protection_frontier(fr),
                                                 f"{slug}_protection_frontier", outdir, formats, close=True)
    if subset:
        ss = sp_subset_sweep(model)
        manifest["expected_welfare_subset"] = F.save_fig(fig_expected_welfare_subset(ss),
                                                         f"{slug}_expected_welfare_subset", outdir, formats, close=True)
        manifest["distortion_subset"] = F.save_fig(fig_distortion_subset(ss),
                                                   f"{slug}_distortion_subset", outdir, formats, close=True)
    return manifest


# --------------------------------------------------------------------------- #
# Cross-dataset views — all four datasets tiled (paper Figs 8–10).
# --------------------------------------------------------------------------- #
ALL_DATASETS = ["412 Food Rescue", "Kidney Allocation", "Moral Machine", "Community Alignment"]
_CORE3 = ["Utilitarian", "RLHF", "Strategyproof"]


@dataclass
class XDatasetRecord:
    """Per-dataset quantities needed for the cross-dataset figures."""
    name: str
    Theta: np.ndarray
    feature_names: list
    betas: np.ndarray
    W_star: float
    alpha_q: np.ndarray
    W_per_agent: dict
    agg: dict


def _agg_from_W(W_per_agent: dict) -> dict:
    return {lbl: {"mean": W.mean(1), "max": W.max(1), "min": W.min(1), "var": W.var(1),
                  "gini": np.array([gini(W[j]) for j in range(len(W))])}
            for lbl, W in W_per_agent.items()}


def load_xdatasets(datasets=ALL_DATASETS, use_cache: bool = True,
                   verbose: bool = False) -> list[XDatasetRecord]:
    """Fit (or load from the ``empirics_cache`` β-sweep cache) all four datasets.

    The cross-dataset figures are expensive to recompute (Moral Machine has 5000
    agents), so the ``<dataset>_xdataset.npz`` caches under the data root are used
    when present; otherwise the dataset is fit + swept and a record built live.
    """
    from .config import data_root
    from .loaders import load_dataset
    from .model import fit_model

    cdir = data_root() / "empirics_cache"
    recs = []
    for ds in datasets:
        p = cdir / f"{ds}_xdataset.npz"
        if use_cache and p.exists():
            blob = np.load(p, allow_pickle=True)
            W_pa = {m: blob[f"W_{m}"] for m in _CORE3 if f"W_{m}" in blob.files}
            recs.append(XDatasetRecord(ds, blob["Theta"], list(blob["feature_names"]),
                                       blob["betas"], float(blob["W_star"]), blob["alpha_q"],
                                       W_pa, _agg_from_W(W_pa)))
            if verbose:
                print(f"  {ds}: cache")
        else:
            if verbose:
                print(f"  {ds}: fitting + sweeping ...")
            model = fit_model(load_dataset(ds))
            sweep = beta_sweep(model)
            recs.append(XDatasetRecord(ds, model.Theta, model.feature_names, sweep.betas,
                                       sweep.W_star, model.alpha_q, sweep.W_per_agent, sweep.agg))
    return recs


def fig_preference_distribution_grid(recs: list[XDatasetRecord], ncols: int = 2):
    """Per-feature θ_n distribution for each dataset (paper Fig 8)."""
    with F.use_paper_style():
        n = len(recs)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 4.2 * nrows), squeeze=False)
        flat = list(axes.flat)
        for ax, rec in zip(flat, recs):
            k = rec.Theta.shape[1]
            parts = ax.violinplot([rec.Theta[:, i] for i in range(k)], positions=range(k),
                                  showmedians=False, showextrema=False, widths=0.7)
            for pc in parts["bodies"]:
                pc.set_facecolor(F.MECH_COLORS["utilitarian"]); pc.set_alpha(0.5)
            for i, m in enumerate(rec.Theta.mean(axis=0)):
                ax.plot([i - 0.3, i + 0.3], [m, m], color="0.1", lw=2.0, zorder=4)
            ax.axhline(0, color="0.6", ls="--", lw=0.7)
            ax.set_xticks(range(k))
            ax.set_xticklabels(rec.feature_names, rotation=40, ha="right",
                               fontsize=8 if k <= 10 else 6)
            ax.set_ylabel(r"$\theta_n$ Value")
            ax.set_title(f"{rec.name}  (N={rec.Theta.shape[0]}, k={k})", fontweight="bold")
        for ax in flat[n:]:
            ax.set_visible(False)
        fig.suptitle(r"Per-feature $\theta_n$ distribution across datasets", fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        return fig


def fig_welfare_variance_grid(recs: list[XDatasetRecord], mechanisms=_CORE3, ncols: int = 2):
    """Cross-agent welfare variance vs β for each dataset (paper Fig 10)."""
    with F.use_paper_style():
        n = len(recs)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.2 * nrows), squeeze=False)
        flat = list(axes.flat)
        for ax, rec in zip(flat, recs):
            for lbl in mechanisms:
                if lbl in rec.agg:
                    ax.plot(rec.betas, rec.agg[lbl]["var"], color=F.mech_color(lbl),
                            lw=2.0, label=lbl)
            ax.set_xscale("log")
            ax.set_title(f"{rec.name}  (N={rec.Theta.shape[0]})", fontweight="bold")
            ax.grid(False)
            ax.set_ylabel(r"$\mathrm{Var}_n[\,U_n / U^*\,]$")
        handles, labels = flat[0].get_legend_handles_labels()
        for ax in flat[n:]:
            ax.set_visible(False)
        fig.tight_layout(rect=[0, 0.09, 1, 1])
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   bbox_to_anchor=(0.5, 0.05))
        fig.supxlabel(r"Sigmoid Temperature $\beta$", y=0.01, fontsize=F.SUPLABEL_SIZE)
        return fig


def fig_expected_welfare_subset_grid(recs: list[XDatasetRecord], ncols: int = 2,
                                     n_boot: int = 60, n_order: int = 100):
    """Expected SP welfare E[U_sp]/U* vs subset size for each dataset (paper Fig 9)."""
    with F.use_paper_style():
        blue, warm = F.MECH_COLORS["utilitarian"], F.MECH_COLORS["bounded"]
        n = len(recs)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.2 * nrows), squeeze=False)
        flat = list(axes.flat)
        for ax, rec in zip(flat, recs):
            k_vals, boot, seq = _subset_sweep_arrays(rec.Theta, rec.alpha_q, n_boot, n_order)
            ax.fill_between(k_vals, np.nanpercentile(boot, 5, 0), np.nanpercentile(boot, 95, 0),
                            alpha=0.18, color=blue, label="bootstrap 5–95th pct")
            ax.plot(k_vals, np.nanmean(boot, 0), "o-", color=blue, lw=2, ms=4, label="bootstrap mean")
            for path in seq:
                ax.plot(k_vals, path, color=warm, alpha=0.05, lw=0.5)
            ax.plot(k_vals, np.nanmean(seq, 0), color=warm, lw=2.2, label="sequential mean")
            ax.axhline(1.0, color="0.6", ls="--", lw=0.8)
            ax.set_title(f"{rec.name}  (N={rec.Theta.shape[0]})", fontweight="bold")
            ax.set_ylabel(r"$E[U_{\rm sp}]\,/\,U^*$")
            ax.grid(False)
        handles, labels = flat[0].get_legend_handles_labels()
        for ax in flat[n:]:
            ax.set_visible(False)
        fig.tight_layout(rect=[0, 0.09, 1, 1])
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   bbox_to_anchor=(0.5, 0.05))
        fig.supxlabel(r"Number of Agents $k$", y=0.01, fontsize=F.SUPLABEL_SIZE)
        return fig


def fig_per_agent_welfare_grid(recs: list[XDatasetRecord], mechanisms=_CORE3,
                               band=(10, 90)):
    """Per-agent welfare U_n/U* vs β — datasets × mechanisms grid (paper Fig 11).

    Instead of a line per agent, a shaded ``band`` (percentile envelope, the
    "center" mass) plus the best-off (solid), worst-off (dotted) and social-welfare
    (dashed) summary lines. Rows = datasets, columns = mechanisms.
    """
    from matplotlib.lines import Line2D
    with F.use_paper_style():
        nrows, ncols = len(recs), len(mechanisms)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.0 * nrows),
                                 sharex=True, squeeze=False)
        for r, rec in enumerate(recs):
            for c, mech in enumerate(mechanisms):
                ax = axes[r, c]
                W = rec.W_per_agent[mech]                 # (n_beta, N)
                col = F.mech_color(mech)
                lo = np.percentile(W, band[0], axis=1)
                hi = np.percentile(W, band[1], axis=1)
                ax.fill_between(rec.betas, lo, hi, color=col, alpha=0.22, lw=0)
                ax.plot(rec.betas, W.max(axis=1), color=col, lw=2.0, ls="-")   # best-off
                ax.plot(rec.betas, W.min(axis=1), color=col, lw=2.0, ls=":")   # worst-off
                ax.plot(rec.betas, W.mean(axis=1), color=col, lw=1.3, ls="--")  # social welfare
                ax.axhline(0, color="0.6", lw=0.8, alpha=0.6)
                ax.set_xscale("log")
                ax.grid(False)
                if r == 0:                                # column header = mechanism
                    F._ORIG_SET_TITLE(ax, mech, fontweight="bold", fontsize=12)
            axes[r, 0].set_ylabel(f"{rec.name}\n(N={rec.Theta.shape[0]})", fontsize=10)
        handles = [Line2D([0], [0], color="0.25", lw=2.0, ls="-", label="best-off"),
                   Line2D([0], [0], color="0.25", lw=2.0, ls=":", label="worst-off"),
                   Line2D([0], [0], color="0.25", lw=1.3, ls="--", label="social welfare")]
        fig.supylabel(r"$U_n / U^*$", fontsize=F.SUPLABEL_SIZE)
        fig.tight_layout(rect=[0.02, 0.06, 1, 1])
        fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.035))
        fig.supxlabel(r"Sigmoid Temperature $\beta$", y=0.005, fontsize=F.SUPLABEL_SIZE)
        return fig
