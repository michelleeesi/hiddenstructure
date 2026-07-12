"""Heatmap views of the Γ externality matrices."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .externalities import GammaResult, group_summary_matrix
from .model import FittedModel

SUPLABEL_SIZE = 15   # figure-level supxlabel/supylabel (matches figures.SUPLABEL_SIZE)


def sort_by_group(group_arr: np.ndarray):
    """Ordering that clusters agents by group, plus block boundaries/midpoints."""
    uniq = sorted(np.unique(group_arr).tolist())
    order = np.concatenate([np.where(group_arr == g)[0] for g in uniq])
    g_sorted = group_arr[order]
    boundaries, midpoints, cum = [], [], 0
    for g in uniq:
        n_g = int((g_sorted == g).sum())
        midpoints.append(cum + n_g / 2 - 0.5)
        cum += n_g
        boundaries.append(cum - 0.5)
    return order, boundaries, midpoints, uniq


def plot_per_agent(results: dict[str, GammaResult], model: FittedModel,
                   axis: str | None = None, clip_pct: float = 99):
    """Per-agent Γ heatmaps (one panel per mechanism), agents sorted by group."""
    axis = axis or next(iter(model.group_axes))
    g_arr = model.group_axes[axis]
    order, boundaries, midpoints, uniq = sort_by_group(g_arr)

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6.5), squeeze=False)
    for c, (ax, (lbl, res)) in enumerate(zip(axes[0], results.items())):
        G_sorted = res.Gamma[np.ix_(order, order)]
        vs = float(np.percentile(np.abs(G_sorted), clip_pct)) or 1e-12
        im = ax.imshow(G_sorted, cmap="RdBu_r", vmin=-vs, vmax=vs,
                       aspect="auto", interpolation="nearest")
        for b in boundaries[:-1]:
            ax.axhline(b, color="black", lw=0.9)
            ax.axvline(b, color="black", lw=0.9)
        ax.set_xticks(midpoints); ax.set_yticks(midpoints)
        ax.set_xticklabels(uniq, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(uniq if c == 0 else [], fontsize=8)
        ax.set_xlabel("Externality Recipient $m$")
        if c == 0:
            ax.set_ylabel("Responsible Party $n$")
        ax.grid(False)
        cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.05)
        cb.set_label(rf"$\Gamma$  ($\pm${vs:.4f})", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        ax.set_title(lbl, fontweight="bold", fontsize=12)
    fig.suptitle(f"Per-agent $\\Gamma_{{n,m}}$ — {model.name}  "
                 f"(sorted by {axis}, N={model.N})", fontweight="bold", fontsize=13)
    plt.tight_layout()
    return fig


def plot_group_averaged(results: dict[str, GammaResult], model: FittedModel,
                        min_n: int = 3, clip_pct: float = 99):
    """Group-averaged Γ heatmaps: every demographic/kmeans axis × every mechanism."""
    axes_names = list(model.group_axes)
    mechs = list(results)
    fig, axes = plt.subplots(len(axes_names), len(mechs),
                             figsize=(5 * len(mechs), 4.5 * len(axes_names)), squeeze=False)
    for r, axis_name in enumerate(axes_names):
        g_arr = model.group_axes[axis_name]
        for c, lbl in enumerate(mechs):
            ax = axes[r, c]
            summ = group_summary_matrix(g_arr, results[lbl].Gamma, min_n=min_n)
            if summ is None:
                ax.set_visible(False); continue
            M, uniq, counts, _ = summ
            vs = float(np.percentile(np.abs(M), clip_pct)) or 1e-12
            im = ax.imshow(M, cmap="RdBu_r", vmin=-vs, vmax=vs)
            ax.set_xticks(range(len(uniq)))
            ax.set_xticklabels([f"{g}\n(n={counts[g]})" for g in uniq],
                               rotation=30, ha="right", fontsize=8)
            ax.set_yticks(range(len(uniq)))
            ax.set_yticklabels([f"{g} (n={counts[g]})" for g in uniq] if c == 0 else [],
                               fontsize=8)
            ax.grid(False)
            cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.05)
            cb.set_label(rf"$\pm${vs:.4f}", fontsize=8)
            cb.ax.tick_params(labelsize=7)
            ax.set_title(f"{axis_name.title()} — {lbl}", fontweight="bold", fontsize=11)
    fig.suptitle(f"Group-averaged $\\Gamma$ — {model.name}", fontweight="bold", fontsize=13)
    fig.supxlabel("Externality Recipient", fontsize=SUPLABEL_SIZE)
    fig.supylabel("Responsible Party", fontsize=SUPLABEL_SIZE, x=0.0)
    plt.tight_layout(rect=[0.03, 0.01, 1, 1])
    return fig
