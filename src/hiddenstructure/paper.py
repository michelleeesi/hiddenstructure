"""Reproduce the paper's numbered figures through the standardized pipeline.

`generate_paper_figures` builds the Community-Alignment figure set (Figs 2–7 of
the paper) with the house style — one Γ run is shared across the heatmap figures,
then the welfare figures reuse the same fitted model. Formatting is the package's
standard style, not the paper's original; only the *content* is reproduced.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import figures as F
from . import welfare as W
from .mechanisms import BoundedHarm, Strategyproof, RLHF, Utilitarian
from .model import FittedModel
from .pipeline import ExternalityRun, alignment_externalities


@dataclass
class PaperFigures:
    """Result of :func:`generate_paper_figures` — the manifest plus the fitted
    objects, so callers can reuse them instead of refitting."""
    manifest: dict                    # {fig_key -> [saved paths]}
    run: ExternalityRun               # stratified-N Γ run (U, R, bh_eps, bh_abs)
    modelF: FittedModel               # full-population fitted model (Figs 2/4/6)
    recs: list | None = None          # cross-dataset records (Figs 8–10)

    def items(self):                  # convenience: iterate the manifest directly
        return self.manifest.items()


def _subrun(run: ExternalityRun, names) -> ExternalityRun:
    """A view of ``run`` with only the named mechanism results (order preserved)."""
    return ExternalityRun(model=run.model,
                          results={n: run.results[n] for n in names if n in run.results})


def generate_paper_figures(dataset: str = "Community Alignment", *, stratified_n: int = 100,
                           full: bool = True, cross_dataset: bool = True,
                           outdir: str = "figures", formats=("png",),
                           use_cache: bool = True, verbose: bool = True) -> "PaperFigures":
    """Regenerate the paper figures. Returns a :class:`PaperFigures` (``.manifest``
    plus the fitted ``run`` / ``modelF`` for reuse).

    Figs 2–7 are the ``dataset``-specific figures (default Community Alignment);
    Figs 8–10 are cross-dataset (all four), produced when ``cross_dataset``.

    Matches the paper's N per figure: the LOO-heavy Γ heatmaps (Figs 3, 5, 7) use a
    ``stratified_n``-agent sample, while the deploy/sweep and closed-form-LOO figures
    (2, 4, 6) run on the full population when ``full`` (Fig 2's floors only bind and
    Fig 6's group counts only match at full N). Set ``full=False`` to keep everything
    on the small sample for a fast smoke run.
    """
    slug = dataset.lower().replace(" ", "_")
    U, R = Utilitarian(), RLHF()
    bh_eps = BoundedHarm(eps=0.05)      # Fig 5 / main-text per-agent floor
    bh_abs = BoundedHarm(b=-0.005)      # Fig 3 absolute floor
    man = {}

    def save(fig, key):
        man[key] = F.save_fig(fig, f"{slug}_{key}", outdir, formats, close=True)
        if verbose:
            print(f"  saved {key}")

    # --- Stratified Γ run drives the LOO-heavy heatmaps (Figs 3, 5, 7). ---
    if verbose:
        print(f"[Γ run, stratified N={stratified_n}]")
    run = alignment_externalities(dataset, mechanisms=[U, R, bh_eps, bh_abs],
                                  stratified_n=stratified_n, use_cache=use_cache, verbose=verbose)
    save(F.fig_gamma_comparison(run, mechanisms=["Utilitarian", bh_abs.name]),
         "fig3_gamma_util_vs_bh_abs")
    save(F.fig_gamma_comparison(run, mechanisms=["Utilitarian", bh_eps.name]),
         "fig5_gamma_util_vs_bh_eps")
    save(F.fig_perperson_vs_group(run, mechanism="RLHF"), "fig7_perperson_vs_group")

    # --- Full-population run: util+RLHF Γ (closed-form LOO), deploy, β-sweep. ---
    if verbose:
        print(f"[full-population run, stratified_n={'None' if full else stratified_n}]")
    runF = alignment_externalities(dataset, mechanisms=[U, R],
                                   stratified_n=None if full else stratified_n,
                                   use_cache=use_cache, verbose=verbose)
    modelF = runF.model

    # Fig 6 — Utilitarian vs RLHF, group-averaged; paper's 2×4 layout over the four
    # main demographics (Country/Gender/Political/Age), each as a util|RLHF pair.
    save(F.fig_group_averaged_pairs(_subrun(runF, ["Utilitarian", "RLHF"]),
                                    axes_names=["country", "gender", "political", "age"]),
         "fig6_group_averaged_util_vs_rlhf")

    # BH(b=-0.5) is shared by Figs 2 and 4 — solve its (β=∞) LP once and reuse.
    bh_half = BoundedHarm(b=-0.5)
    shared_deploy = {bh_half.name: bh_half.deploy(modelF)}

    # Fig 2 — welfare distribution at high β (deploy only, no LOO).
    dist_mechs = [U, R, Strategyproof(), bh_half,
                  BoundedHarm(b=-0.25), BoundedHarm(b=-0.1)]
    save(W.fig_welfare_distribution(modelF, dist_mechs, beta=100.0, deploys=shared_deploy),
         "fig2_welfare_distribution")
    # Fig 4 — per-agent welfare vs β, core mechanisms + three bounded-harm variants.
    sweep = W.beta_sweep(modelF, static_mechanisms=[bh_half, BoundedHarm(eps=0.1),
                         BoundedHarm(eps=0.8)], static_deploys=shared_deploy, verbose=verbose)
    save(W.fig_best_worst_mean(sweep, ncols=3), "fig4_welfare_vs_beta")

    # --- Cross-dataset figures (all four datasets, from the β-sweep cache). ---
    recs = None
    if cross_dataset:
        if verbose:
            print("[cross-dataset figures 8–10]")
        recs = W.load_xdatasets(use_cache=use_cache, verbose=verbose)
        save(W.fig_preference_distribution_grid(recs), "fig8_preference_distribution")
        save(W.fig_expected_welfare_subset_grid(recs), "fig9_expected_welfare_subset")
        save(W.fig_welfare_variance_grid(recs), "fig10_welfare_variance")
        save(W.fig_per_agent_welfare_grid(recs), "fig11_per_agent_welfare")

    return PaperFigures(manifest=man, run=run, modelF=modelF, recs=recs)
