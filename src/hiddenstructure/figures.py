"""Paper figure generation — the third pipeline stage.

    dataset ──► externalities/mechanism ──► **standardized figures**

One house style, one set of named figure builders, one save path. Every figure
the paper uses flows through here so they share fonts, sizing, colors, and file
formats. Point a driver at an ``ExternalityRun`` and call :func:`generate_all`.

Design (validated against the `dataviz` skill):
* **Γ heatmaps are diverging** — ``RdBu_r`` (two hues + neutral-gray midpoint),
  clipped symmetrically at the 99th percentile so a few outliers don't wash out
  the map.
* **Mechanism colors are categorical**, assigned by identity in a fixed order and
  CVD-validated (worst adjacent ΔE 24.2, worst all-pairs 13.3 ≥ 12). The same
  color means the same mechanism in every figure.
* Text is saved as text (editable in Illustrator / LaTeX): ``svg.fonttype=none``,
  ``pdf.fonttype=42``.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt

from .externalities import group_summary_matrix
from .pipeline import ExternalityRun
from .plotting import plot_group_averaged, plot_per_agent, sort_by_group

# --------------------------------------------------------------------------- #
# House style
# --------------------------------------------------------------------------- #
GAMMA_CMAP = "RdBu_r"          # diverging: red = imposes/receives +, blue = −

#: Categorical mechanism colors, fixed order, CVD-validated. Keyed by family.
MECH_COLORS = {
    "utilitarian":   "#2a78d6",   # blue
    "rlhf":          "#e34948",   # red
    "strategyproof": "#4a3aa7",   # violet
    "bounded":       "#eda100",   # amber  (Bounded harm)
    "public":        "#008300",   # green  (Public spirit)
}
_FALLBACK_COLORS = ["#e87ba4", "#1baf7a", "#eb6834"]  # magenta, aqua, orange

PAPER_RC = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "svg.fonttype": "none",       # keep text as text in SVG
    "pdf.fonttype": 42,           # embed TrueType in PDF (editable)
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",   # Times-compatible math
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,         # descriptive axis labels (Density, Responsible Party, …)
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
    "legend.frameon": False,
}

#: font size for figure-level ``supxlabel``/``supylabel`` (bottom/side labels);
#: ``figure.labelsize`` isn't an rcParam until matplotlib 3.6, so set explicitly.
SUPLABEL_SIZE = 15


def mech_color(name: str, index: int = 0) -> str:
    """Fixed color for a mechanism, matched by family (so ``Bounded harm ε=0.05``
    still maps to amber). Unknown mechanisms take a fallback slot by ``index``."""
    key = name.strip().lower()
    for family, color in MECH_COLORS.items():
        if key.startswith(family):
            return color
    return _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)]


#: Suppress the OVERALL figure title (suptitle). Per-panel/subfigure titles are
#: kept. Set True to also show suptitles.
SHOW_SUPTITLES = False

_ORIG_SET_TITLE = mpl.axes.Axes.set_title
_ORIG_SUPTITLE = mpl.figure.Figure.suptitle


@contextmanager
def use_paper_style():
    """Apply the house style: ``with figures.use_paper_style(): ...``.

    Besides the rcParams, this suppresses the **overall figure title**
    (``Figure.suptitle``) unless :data:`SHOW_SUPTITLES`, while leaving per-panel
    (subfigure) ``ax.set_title`` titles intact. Every builder wraps its body in
    this context. Restored on exit.
    """
    with plt.rc_context(PAPER_RC):
        if SHOW_SUPTITLES:
            yield
            return
        mpl.figure.Figure.suptitle = lambda self, *a, **k: _ORIG_SUPTITLE(self, "")
        try:
            yield
        finally:
            mpl.figure.Figure.suptitle = _ORIG_SUPTITLE


def save_fig(fig, name: str, outdir="figures", formats=("png",),
             close: bool = False) -> list[Path]:
    """Save ``fig`` under ``outdir`` as ``name`` in each format. Returns the paths."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        p = out / f"{name}.{ext}"
        fig.savefig(p)
        paths.append(p)
    if close:
        plt.close(fig)
    return paths


# --------------------------------------------------------------------------- #
# Standardized figure builders (each returns a Figure)
# --------------------------------------------------------------------------- #
def fig_per_agent_gamma(run: ExternalityRun, axis: str | None = None):
    """Per-agent Γ heatmaps, one panel per mechanism (the main externality figure)."""
    with use_paper_style():
        return plot_per_agent(run.results, run.model, axis=axis)


def fig_group_averaged_gamma(run: ExternalityRun):
    """Group-averaged Γ heatmaps: every grouping axis × every mechanism."""
    with use_paper_style():
        return plot_group_averaged(run.results, run.model)


def fig_group_averaged_pairs(run: ExternalityRun, axes_names, mechanisms=None,
                             min_n: int = 3, clip_pct: float = 99):
    """Paper Fig 6 layout: demographics laid out two-per-row, each as a
    (mechanism₁, mechanism₂) pair — i.e. a ``(len(axes)//2) × (2·len(mechs))`` grid
    of ``Demographic — Mechanism`` panels. Per-panel colorbars (scales differ across
    demographics); panel titles identify each cell (kept despite global title
    suppression, since the panels are otherwise indistinguishable)."""
    with use_paper_style():
        model = run.model
        mechs = mechanisms or list(run.results)
        m = len(mechs)
        ncols = 2 * m
        nrows = int(np.ceil(len(axes_names) / 2))
        # constrained_layout keeps all panels equal height despite the per-panel
        # colorbars (tight_layout leaves the bottom row short → squished y-ticks).
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 4.6 * nrows),
                                 squeeze=False, constrained_layout=True)
        for ax in axes.flat:
            ax.set_visible(False)
        for i, axis_name in enumerate(axes_names):
            g_arr = model.group_axes[axis_name]
            r, base_c = i // 2, (i % 2) * m
            for j, lbl in enumerate(mechs):
                ax = axes[r, base_c + j]
                summ = group_summary_matrix(g_arr, run.results[lbl].Gamma, min_n=min_n)
                if summ is None:
                    continue
                ax.set_visible(True)
                M, uniq, counts, _ = summ
                vs = float(np.percentile(np.abs(M), clip_pct)) or 1e-12
                im = ax.imshow(M, cmap=GAMMA_CMAP, vmin=-vs, vmax=vs, aspect="auto")
                ax.set_xticks(range(len(uniq)))
                ax.set_xticklabels([f"{g}\n(n={counts[g]})" for g in uniq],
                                   rotation=30, ha="right", fontsize=6)
                ax.set_yticks(range(len(uniq)))
                # counts already shown in the x-tick labels — keep y-ticks short
                ax.set_yticklabels([str(g) for g in uniq] if j == 0 else [], fontsize=7)
                cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.05)
                cb.set_label(rf"$\pm${vs:.4f}", fontsize=6)
                cb.ax.tick_params(labelsize=6)
                _ORIG_SET_TITLE(ax, f"{axis_name.title()} — {lbl}",
                                fontweight="bold", fontsize=11)
        fig.supxlabel("Externality Recipient", fontsize=SUPLABEL_SIZE)
        fig.supylabel("Responsible Party", fontsize=SUPLABEL_SIZE)
        return fig


def fig_mechanism_summary(run: ExternalityRun):
    """Cross-agent externality magnitude by mechanism.

    One bar per mechanism = mean |off-diagonal Γ| (how much externality flows
    *between* agents under that mechanism). Single series → bars colored by
    mechanism identity, direct-labeled, no legend (title names it).
    """
    with use_paper_style():
        names = list(run.results)
        vals, colors = [], []
        for i, name in enumerate(names):
            G = run.results[name].Gamma
            off = G[~np.eye(G.shape[0], dtype=bool)]
            vals.append(float(np.abs(off).mean()))
            colors.append(mech_color(name, i))
        fig, ax = plt.subplots(figsize=(1.6 * len(names) + 2, 4.2))
        x = np.arange(len(names))
        ax.bar(x, vals, color=colors, width=0.62, zorder=3)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.2e}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylabel(r"Mean $|\Gamma_{n,m}|$, $n\neq m$")
        ax.set_ylim(0, max(vals) * 1.18)
        ax.margins(x=0.04)
        ax.set_title(f"Cross-agent alignment externality by mechanism — {run.model.name}")
        return fig


def fig_externality_ranking(run: ExternalityRun, mechanism: str | None = None,
                            top: int = 15):
    """Net externality *imposed* (row sum) per agent, top-``top`` by magnitude.

    Diverging horizontal bars for one mechanism: red = net-harmful participation,
    blue = net-beneficial. Agents labeled directly."""
    with use_paper_style():
        name = mechanism or next(iter(run.results))
        G = run.results[name].Gamma
        imposed = G.sum(axis=1)                       # row n: n's effect on all m
        order = np.argsort(np.abs(imposed))[::-1][:top][::-1]
        vals = imposed[order]
        labels = [str(run.model.agent_ids[i]) for i in order]
        colors = ["#e34948" if v > 0 else "#2a78d6" for v in vals]  # red +, blue −
        fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(order) + 1.2))
        y = np.arange(len(order))
        ax.barh(y, vals, color=colors, zorder=3)
        ax.axvline(0, color="0.4", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel(r"Net Externality Imposed  $\sum_m \Gamma_{n,m}$")
        ax.set_title(f"Who imposes externalities — {name} — {run.model.name}")
        return fig


def fig_gamma_magnification(run: ExternalityRun, mechanism: str | None = None):
    """Γ as a magnification tool: per-agent resolution (left) vs demographic
    block-averages (right, one panel per grouping axis)."""
    with use_paper_style():
        name = mechanism or (("RLHF" in run.results and "RLHF") or next(iter(run.results)))
        res = run.results[name]
        model = run.model
        axes_names = list(model.group_axes)
        primary = axes_names[0]
        order, boundaries, midpoints, uniq = sort_by_group(model.group_axes[primary])

        n_right = len(axes_names)
        ncols = 1 + n_right
        fig = plt.figure(figsize=(6 + 3.4 * n_right, 6.5))
        gs = fig.add_gridspec(1, ncols, width_ratios=[2.2] + [1] * n_right)

        axL = fig.add_subplot(gs[0, 0])
        G = res.Gamma[np.ix_(order, order)]
        vs = float(np.percentile(np.abs(G), 99)) or 1e-12
        axL.imshow(G, cmap=GAMMA_CMAP, vmin=-vs, vmax=vs, aspect="auto", interpolation="nearest")
        for b in boundaries[:-1]:
            axL.axhline(b, color="black", lw=0.8); axL.axvline(b, color="black", lw=0.8)
        axL.set_xticks(midpoints); axL.set_yticks(midpoints)
        axL.set_xticklabels(uniq, rotation=30, ha="right"); axL.set_yticklabels(uniq)
        axL.set_xlabel("Externality Recipient $m$"); axL.set_ylabel("Responsible Party $n$")
        axL.set_title(f"Per-person Γ — {primary} (N={model.N})", fontweight="bold")

        for c, axis_name in enumerate(axes_names, start=1):
            ax = fig.add_subplot(gs[0, c])
            summ = group_summary_matrix(model.group_axes[axis_name], res.Gamma)
            if summ is None:
                ax.set_visible(False); continue
            M, gs_uniq, counts, _ = summ
            v = float(np.percentile(np.abs(M), 99)) or 1e-12
            ax.imshow(M, cmap=GAMMA_CMAP, vmin=-v, vmax=v)
            ax.set_xticks(range(len(gs_uniq))); ax.set_yticks(range(len(gs_uniq)))
            ax.set_xticklabels(gs_uniq, rotation=30, ha="right", fontsize=7)
            ax.set_yticklabels(gs_uniq, fontsize=7)
            ax.set_title(f"{axis_name} — group avg", fontweight="bold", fontsize=10)
        fig.suptitle(f"Alignment externality Γ — agent-level vs block-averaged — "
                     f"{model.name} ({name})", fontweight="bold")
        fig.tight_layout()
        return fig


def fig_impact_space_2d(thetas=None, n_dir: int = 400, seed: int = 0):
    """Schematic of preference / query / impact space in 2-D (conceptual figure).

    Shows the feasible impact set Ψ̄ (shaded), a few agents' preferences θ_n
    (dashed rays), their ideal impacts ψ*_n (vertices of Ψ̄), the utilitarian mean
    θ̄, and the utilitarian optimum ψ*_util. Not data-driven — a paper illustration.
    """
    with use_paper_style():
        if thetas is None:
            thetas = np.array([[1.0, 0.35], [-0.5, 1.0], [-0.7, -0.9]])
        thetas = np.asarray(thetas, float)
        rng = np.random.default_rng(seed)
        ang = np.linspace(0, 2 * np.pi, n_dir, endpoint=False)
        Q = np.c_[np.cos(ang), np.sin(ang)]                 # query directions

        def psi_star(theta):
            return 0.5 * (Q * np.sign(Q @ theta)[:, None]).mean(axis=0)

        # Feasible-set boundary: ψ*(θ) over all preference directions.
        boundary = np.array([psi_star(d) for d in Q])
        theta_bar = thetas.mean(axis=0)
        cats = ["#2a78d6", "#e34948", "#008300", "#eda100", "#4a3aa7"]

        fig, ax = plt.subplots(figsize=(5.6, 5.6))
        ax.fill(boundary[:, 0], boundary[:, 1], color="0.85", alpha=0.7, zorder=0,
                label=r"$\bar\Psi$ (feasible impacts)")
        ax.axhline(0, color="0.6", lw=0.8); ax.axvline(0, color="0.6", lw=0.8)
        for i, th in enumerate(thetas):
            c = cats[i % len(cats)]
            u = th / np.linalg.norm(th)
            ax.plot([0, u[0] * 0.9], [0, u[1] * 0.9], ls="--", color=c, lw=1.4, alpha=0.7)
            ax.annotate(rf"$\theta_{chr(65+i)}$", u * 0.95, color=c, fontsize=11)
            ps = psi_star(th)
            ax.plot(*ps, marker="P", ms=13, color=c, zorder=4)
            ax.annotate(rf"$\psi^*_{chr(65+i)}$", ps, color=c, fontsize=10,
                        xytext=(6, 6), textcoords="offset points")
        ub = theta_bar / np.linalg.norm(theta_bar)
        ax.plot([0, ub[0] * 0.7], [0, ub[1] * 0.7], color="0.15", lw=1.6)
        ax.annotate(r"$\bar\theta$", ub * 0.72, color="0.15", fontsize=11)
        pstar = psi_star(theta_bar)
        ax.plot(*pstar, marker="*", ms=18, color="0.1", zorder=5)
        ax.annotate(r"$\psi^*_{\rm util}$", pstar, color="0.1", fontsize=11,
                    xytext=(6, 6), textcoords="offset points")
        ax.set_aspect("equal")
        ax.set_title("Preference space, query space, and impact space", fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        return fig


def fig_gamma_comparison(run: ExternalityRun, mechanisms=None, axis: str | None = None,
                         clip_pct: float = 99):
    """Per-agent Γ heatmaps for two (or more) mechanisms with ONE shared colorbar.

    The paper's Utilitarian-vs-Bounded-harm externality figure (Figs 3 & 5): same
    agent ordering, same symmetric colour scale, single ±v colorbar so the panels
    are directly comparable.
    """
    with use_paper_style():
        model = run.model
        names = mechanisms or list(run.results)[:2]
        axis = axis or next(iter(model.group_axes))
        order, boundaries, midpoints, uniq = sort_by_group(model.group_axes[axis])
        mats = [run.results[n].Gamma[np.ix_(order, order)] for n in names]
        vs = max(float(np.percentile(np.abs(M), clip_pct)) for M in mats) or 1e-12

        n = len(names)
        fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 6.2), squeeze=False)
        im = None
        for ax, name, M in zip(axes[0], names, mats):
            im = ax.imshow(M, cmap=GAMMA_CMAP, vmin=-vs, vmax=vs, aspect="auto",
                           interpolation="nearest")
            for b in boundaries[:-1]:
                ax.axhline(b, color="black", lw=0.8); ax.axvline(b, color="black", lw=0.8)
            ax.set_xticks(midpoints); ax.set_yticks(midpoints)
            ax.set_xticklabels(uniq, rotation=30, ha="right"); ax.set_yticklabels(uniq)
            # panel identifier (which mechanism) — bypass the global title suppression
            _ORIG_SET_TITLE(ax, name, fontweight="bold", fontsize=13)
        axes[0, 0].set_ylabel("Responsible Party $n$")
        fig.subplots_adjust(bottom=0.17, top=0.93)   # room below ticks / above panel labels
        cb = fig.colorbar(im, ax=axes[0], fraction=0.045, pad=0.03)
        cb.set_label(rf"$\Gamma_{{n,m}}$  (shared, ±{vs:.4f})")
        fig.supxlabel("Externality Recipient $m$", y=0.04, fontsize=SUPLABEL_SIZE)
        return fig


def fig_perperson_vs_group(run: ExternalityRun, mechanism: str | None = None,
                           axes_names=None, clip_pct: float = 99):
    """Per-person Γ (left) vs group-averaged Γ (right), stacked one row per axis.

    The paper's "microscope" figure (Fig 7): for each demographic axis, the
    per-agent resolution next to its within-group averages, each panel clipped
    independently.
    """
    with use_paper_style():
        model = run.model
        name = mechanism or (("RLHF" in run.results and "RLHF") or next(iter(run.results)))
        G_full = run.results[name].Gamma
        axes_names = axes_names or list(model.group_axes)
        nrows = len(axes_names)
        fig, axes = plt.subplots(nrows, 2, figsize=(12, 4.2 * nrows), squeeze=False)
        for r, axis_name in enumerate(axes_names):
            g_arr = model.group_axes[axis_name]
            order, boundaries, midpoints, uniq = sort_by_group(g_arr)
            # left: per-person
            axL = axes[r, 0]
            Gp = G_full[np.ix_(order, order)]
            v = float(np.percentile(np.abs(Gp), clip_pct)) or 1e-12
            axL.imshow(Gp, cmap=GAMMA_CMAP, vmin=-v, vmax=v, aspect="auto", interpolation="nearest")
            for b in boundaries[:-1]:
                axL.axhline(b, color="black", lw=0.7); axL.axvline(b, color="black", lw=0.7)
            axL.set_xticks(midpoints); axL.set_yticks(midpoints)
            axL.set_xticklabels(uniq, rotation=30, ha="right", fontsize=7)
            axL.set_yticklabels(uniq, fontsize=7)
            axL.set_ylabel(axis_name.title(), fontsize=12)
            axL.set_title(f"{axis_name.title()} — per-person Γ", fontweight="bold", fontsize=10)
            # right: group-averaged
            axR = axes[r, 1]
            summ = group_summary_matrix(g_arr, G_full)
            if summ is None:
                axR.set_visible(False); continue
            M, gs_uniq, counts, _ = summ
            v2 = float(np.percentile(np.abs(M), clip_pct)) or 1e-12
            axR.imshow(M, cmap=GAMMA_CMAP, vmin=-v2, vmax=v2)
            axR.set_xticks(range(len(gs_uniq))); axR.set_yticks(range(len(gs_uniq)))
            axR.set_xticklabels([f"{g}\n(n={counts[g]})" for g in gs_uniq],
                                rotation=30, ha="right", fontsize=7)
            axR.set_yticklabels([f"{g} (n={counts[g]})" for g in gs_uniq], fontsize=7)
            axR.set_title(f"{axis_name.title()} — group-averaged Γ", fontweight="bold", fontsize=10)
        fig.suptitle(f"Alignment externality: per-person vs group-averaged — "
                     f"{model.name} ({name}, N={model.N})", fontweight="bold", y=0.995)
        fig.supxlabel("Externality Recipient $m$", fontsize=SUPLABEL_SIZE)
        fig.supylabel("Responsible Party $n$", fontsize=SUPLABEL_SIZE)
        fig.tight_layout(rect=[0.02, 0.01, 1, 0.985])
        return fig


#: The standard figure set, keyed by output filename stem.
STANDARD_FIGURES = {
    "gamma_per_agent":      fig_per_agent_gamma,
    "gamma_group_averaged": fig_group_averaged_gamma,
    "mechanism_summary":    fig_mechanism_summary,
    "externality_ranking":  fig_externality_ranking,
}


def generate_all(run: ExternalityRun, outdir="figures",
                 formats=("png",), prefix: str | None = None) -> dict:
    """Build and save the whole standard figure set for one run.

    Files are named ``<prefix>_<figure>.<ext>`` (prefix defaults to a slug of the
    dataset name). Returns ``{figure_name: [saved paths]}``.
    """
    slug = (prefix or run.model.name).lower().replace(" ", "_")
    manifest = {}
    for stem, builder in STANDARD_FIGURES.items():
        fig = builder(run)
        manifest[stem] = save_fig(fig, f"{slug}_{stem}", outdir, formats, close=True)
    return manifest
