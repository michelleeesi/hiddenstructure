# hiddenstructure

Alignment externalities and welfare mechanisms, computed uniformly across four
preference datasets (412 Food Rescue, Kidney Allocation, Moral Machine,
Community Alignment).

The math used to be copy-pasted across a dozen notebooks. It now lives **once**
in `src/hiddenstructure/`, and the notebooks are thin drivers over it.

## Pipelines

```
Pipeline #1  (built):  dataset ──► alignment externalities Γ_{n,m}
Pipeline #2  (built):  dataset ──► mechanism (utilitarian, RLHF, strategyproof,
                                              bounded harm, public spirit)
Figures      (built):  externalities/mechanism ──► standardized paper figures
```

The first two share the same front half — load → fit BTL → assemble a
`FittedModel`. They diverge at the **mechanism**: a mechanism maps the fitted
dataset to a deployment impact ψ_c (and its leave-one-out impacts ψ_c^(-n)).
Pipeline #1 feeds those into Γ; pipeline #2 studies the mechanism's welfare
properties directly. The **figure** stage consumes either result and emits the
paper's figures through one house style, so they're consistent by construction.

## Quick start

```python
import sys; sys.path.insert(0, "src")          # or: pip install -e .
from hiddenstructure import pipeline
from hiddenstructure.mechanisms import Utilitarian, RLHF, Strategyproof, BoundedHarm

run = pipeline.alignment_externalities(
    "Community Alignment",
    mechanisms=[Utilitarian(), RLHF(), Strategyproof(), BoundedHarm(eps=0.05)],
    stratified_n=100,          # cap agents for tractable leave-one-out
    verbose=True,
)
run.results["RLHF"].Gamma      # the (N, N) externality matrix
```

Or open **`run_externalities.ipynb`** — pick a dataset in the config cell, run all
cells, get the Γ matrices, per-agent heatmaps, group-averaged heatmaps, and the
"who imposes / receives" headlines.

## Package layout

| module | what it holds |
|---|---|
| `config.py`        | where data + caches live (`HS_DATA_ROOT`) |
| `loaders.py`       | one loader per dataset → `DatasetBundle`; stratified subsample |
| `btl.py`           | per-agent / pooled Bradley-Terry-Luce fits |
| `impact.py`        | ψ, ψ*, φ, φ⁻¹, welfare — the paper's core objects, defined once |
| `model.py`         | `fit_model` → `FittedModel` (Θ, θ̄_w, θ̂_RLHF, α_q, group axes) |
| `mechanisms.py`    | `Utilitarian`, `RLHF`, `Strategyproof`, `BoundedHarm`, `PublicSpirit` |
| `externalities.py` | `gamma_for`, Γ matrix, group-averaged summaries |
| `plotting.py`      | per-agent and group-averaged Γ heatmaps (interactive) |
| `figures.py`       | paper house style + externality figure builders + `generate_all` |
| `welfare.py`       | β-sweep / subset / distortion / frontier math + welfare figure builders |
| `paper.py`         | `generate_paper_figures` — reproduce the paper's numbered figures (2–7) |
| `pipeline.py`      | `alignment_externalities(dataset, ...)` one-call entry point |

## Data

Notebooks are kept **data-free** (all CSV/parquet are gitignored). The datasets +
precomputed leave-one-out caches live in a sibling working copy, `../distortion`,
which the loaders read by default. Override with:

```bash
export HS_DATA_ROOT=/path/to/data      # dir containing empirics_412/, empirics_kidney/, ...
```

Expensive LOO ψ arrays (strategyproof, bounded harm) are cached under
`$HS_DATA_ROOT/loo_cache/` with the original naming, so existing caches are reused.

## Bring your own dataset

You don't need to touch the built-in loaders. Convert your data to a
`DatasetBundle` with [`load_csv`](src/hiddenstructure/loaders.py) (or
`bundle_from_frame` for an in-memory DataFrame) and hand it straight to the
pipeline:

```python
from hiddenstructure import pipeline
from hiddenstructure.loaders import load_csv

bundle = load_csv(
    "my_choices.csv",
    id_col="user_id",                       # who made each choice
    a_cols=["speed_A", "cost_A", "safety_A"],   # option-A feature columns
    b_cols=["speed_B", "cost_B", "safety_B"],   # option-B feature columns (same order)
    chosen_col="picked", chosen_a_value="A",    # y = 1 where option A won
    feature_names=["speed", "cost", "safety"],
    demographic_cols=["country"],           # optional grouping axes
    name="My dataset",
)
run = pipeline.alignment_externalities(bundle, stratified_n=200, verbose=True)
run.results["RLHF"].Gamma
```

Your CSV must be **one row per pairwise choice**, and you specify the features
in one of two ways:

| your data looks like… | pass |
|---|---|
| option-A columns + option-B columns, and a column saying which won | `a_cols`, `b_cols`, `chosen_col`, `chosen_a_value` |
| already-differenced feature columns (A − B) + a 0/1 label | `alpha_cols`, `chosen_col`/`chosen_a_value` |
| rows that are always *winner − loser* (one-sided) | `alpha_cols` (or `a_cols`/`b_cols`) + `symmetrize=True` |

**Requirements the math imposes** (the loader validates the last one and raises a
helpful error):

- Pairwise **binary** choices with **numeric** per-option features (text must be
  embedded first — see `empirics_communityalignment/embed_and_project.py`).
- **≥ `min_obs` (default 20) rows per agent, with both outcomes present** — agents
  below that, or who always picked the same side, are dropped from Θ.
- The A/B ordering must be **fixed independent of the outcome** (use `chosen_col`),
  **or** your rows are winner-minus-loser (use `symmetrize=True`). Setting
  `alpha = chosen − rejected` without `symmetrize=True` makes every label 1 and
  is rejected with a message telling you what to do.
- Utility is linear and intercept-free, so scale/center features sensibly.
- Externalities only appear when agents disagree; homogeneous agents give Γ ≈ 0.

## Figures — standardized for the paper

`figures.py` is the one place figure style is defined, so every figure in the
paper shares fonts, sizing, colors, and file formats. Open
**`run_figures.ipynb`** (dataset + mechanisms → saved figure set), or call it
directly:

```python
from hiddenstructure import pipeline, figures

run = pipeline.alignment_externalities("Community Alignment", stratified_n=100)
manifest = figures.generate_all(run, outdir="figures")   # PNG by default
# writes figures/community_alignment_<figure>.png, returns the paths
# other formats on request, e.g. formats=("png", "svg")
```

There are two figure families: **externality** figures (`figures.py`, from an
`ExternalityRun`) and **welfare** figures (`welfare.py`, from a `FittedModel` — the
mechanism-welfare view ported from `empirics_general.ipynb`). Generate each with:

```python
from hiddenstructure import pipeline, figures, welfare
run = pipeline.alignment_externalities("Community Alignment", stratified_n=100)
figures.generate_all(run)                    # externality figures
welfare.generate_welfare_figures(run.model)  # welfare figures (slower: β-sweep)
```

### The numbered paper figures (Figs 2–7)

`hiddenstructure.paper.generate_paper_figures` reproduces the Community-Alignment
figure set in one call — open **`run_figures.ipynb`** or:

```python
from hiddenstructure import paper
pf = paper.generate_paper_figures("Community Alignment", stratified_n=100, full=True)
pf.manifest            # {fig_key -> [saved paths]}
pf.run, pf.modelF      # the fitted objects, reusable without refitting
```

Each model is fit once and shared across the figures it feeds (the stratified Γ
run drives Figs 3/5/7; the full-population model drives Figs 2/4/6); `U*` is cached
on the model and the one bounded-harm impact shared by Figs 2 & 4 is solved once.

It matches each figure's N in the paper: the leave-one-out-heavy Γ heatmaps
(Figs 3, 5, 7) run on the 100-agent stratified sample the captions specify, while
the deploy/sweep and closed-form-LOO figures (2, 4, 6) run on all 2387 annotators
(so Fig 2's floors bind and Fig 6's group counts match).

| paper figure | builder | notes |
|---|---|---|
| **Fig 2** — welfare distribution at β=100 | `welfare.fig_welfare_distribution` | util/RLHF/SP + bounded harm `b=−0.5,−0.25,−0.1`; gray = utilitarian reference, red = floor |
| **Fig 3** — Util vs bounded harm (abs. floor `b`), shared colorbar | `figures.fig_gamma_comparison` | `BoundedHarm(b=−0.005)` |
| **Fig 4** — per-agent welfare vs β, 2×3 | `welfare.fig_best_worst_mean` (`ncols=3`) | core 3 + `BoundedHarm(b=−0.5)`, `eps=0.1`, `eps=0.8` |
| **Fig 5** — Util vs bounded harm (per-agent floor `ε`) | `figures.fig_gamma_comparison` | `BoundedHarm(eps=0.05)` |
| **Fig 6** — Util vs RLHF, group-averaged by demographic | `figures.fig_group_averaged_gamma` | full population, all six CA demographic axes |
| **Fig 7** — per-person vs group-averaged Γ, per axis | `figures.fig_perperson_vs_group` | country/gender/age/political/education/ethnicity |
| **Fig 8** — per-feature θ distribution, four datasets | `welfare.fig_preference_distribution_grid` | cross-dataset (2×2) |
| **Fig 9** — expected SP welfare vs subset size, four datasets | `welfare.fig_expected_welfare_subset_grid` | cross-dataset (2×2) |
| **Fig 10** — welfare variance vs β, four datasets | `welfare.fig_welfare_variance_grid` | cross-dataset (2×2) |

Figs 8–10 tile all four datasets; `welfare.load_xdatasets()` reads the
`empirics_cache/<dataset>_xdataset.npz` β-sweep caches (or fits + sweeps live if
absent), so they reproduce the paper's numbers without recomputing Moral Machine's
5000-agent sweep.

`BoundedHarm` now takes **either** `eps` (per-agent floor −ε·h_Ψ̄(θ_n)) **or** `b`
(absolute floor b·U\*, in U/U\* units), covering both forms the paper uses. The
CoAlign loader now exposes all six demographic axes (education & ethnicity added,
collapsed as in the paper).

### Reproducing other repo images

Every image currently in the repo maps to a builder — the pipeline can regenerate
all of them (validated: the welfare β-sweep matches the cached `empirics_general`
numbers to ~1e-13 for utilitarian/RLHF, 1e-8 for strategyproof; the Γ matrices are
bitwise-identical to the originals). A few originals are *all-four-datasets* grids
(`per agent welfare.png`, `preferencedist`, the combined variance panel); the
builder below produces the **per-dataset** panel — run it on each of the four
datasets to tile the combined grid (a thin wrapper, noted in Next steps):

| existing image | builder |
|---|---|
| `peragent.svg`            | `figures.fig_per_agent_gamma` |
| `extheatmapavgdem.png`, `pooledextheatmapdem.png` | `figures.fig_group_averaged_gamma` |
| `heatmapcomparison.png`, `marginalcomparison.png` | `figures.fig_per_agent_gamma` (mechanisms side by side) |
| `rlhfvsbh.png`            | `figures.fig_per_agent_gamma` with `[RLHF(), BoundedHarm(...)]` |
| `fullalign.png`, `perpsonvsgroupavg.png` | `figures.fig_gamma_magnification` |
| `overallgraphic.png`      | `figures.fig_impact_space_2d` (schematic) |
| `bestandworst.svg`, `kidneybestworst.png`, `mmbestworst.png`, `constrainedwelfare.png` | `welfare.fig_best_worst_mean` |
| `welfarevar.png`, `welfarevariance.svg` | `welfare.fig_welfare_variance` |
| `per agent welfare.png`   | `welfare.fig_per_agent_welfare` |
| `preferencedist.png/svg`  | `welfare.fig_preference_distribution` |
| `expectedwelfare.svg`     | `welfare.fig_expected_welfare_subset` |
| `distortionmm.png`        | `welfare.fig_distortion_subset` |

Plus one **new** figure the paper doesn't have yet: `welfare.fig_protection_frontier`
— social welfare vs the worst-off agent as public-spirit γ (or bounded-harm ε) varies.

The externality standard set (`figures.STANDARD_FIGURES`, run by `generate_all`):

| figure | what it shows |
|---|---|
| `gamma_per_agent`      | per-agent Γ heatmaps, one panel per mechanism |
| `gamma_group_averaged` | group-averaged Γ, every grouping axis × mechanism |
| `mechanism_summary`    | cross-agent externality magnitude by mechanism (bar) |
| `externality_ranking`  | net externality imposed per agent, one mechanism (diverging bars) |

The welfare set (`welfare.generate_welfare_figures`): `welfare_best_worst`,
`welfare_variance`, `welfare_per_agent`, `preference_distribution`,
`protection_frontier`, `expected_welfare_subset`, `distortion_subset`.

**House style** (validated against the `dataviz` skill):

- **Γ heatmaps are diverging** — `figures.GAMMA_CMAP = "RdBu_r"` (two hues + gray
  midpoint), clipped symmetrically at the 99th percentile.
- **Mechanism colors are categorical and fixed by identity** — `figures.MECH_COLORS`:
  blue = utilitarian, red = RLHF, violet = strategyproof, amber = bounded harm,
  green = public spirit. The palette is CVD-validated (worst adjacent ΔE 24.2,
  worst all-pairs 13.3 ≥ 12), so the same color means the same mechanism everywhere.
- **Font is Times New Roman** (serif; STIX for math). Per-panel (subfigure) titles
  are kept, but the **overall figure title (suptitle) is suppressed** centrally in
  `use_paper_style()` (set `figures.SHOW_SUPTITLES = True` to restore it).
- Output is **PNG at 300 dpi** by default (`figures.save_fig` / `generate_all` take
  a `formats=` tuple if you ever want `svg`/`pdf` too). See `figures.PAPER_RC`.

**Add a paper figure:** write a `fig_myfigure(run)` builder in `figures.py`
(wrap its body in `with figures.use_paper_style():`), register it in
`STANDARD_FIGURES`, and it flows through `generate_all` with the same style and
formats. Save a one-off with `figures.save_fig(fig, name, outdir, formats)`.

## Mechanisms

Every mechanism implements the same two methods, so pipeline #1 (Γ) and the
figure stage are agnostic to which one they're handed:

```python
class Mechanism:
    def deploy(self, model) -> np.ndarray:   # ψ_c            shape (k,)
    def loo(self, model)    -> np.ndarray:   # ψ_c^(-n)       shape (N, k)
```

| mechanism | deployment ψ_c |
|---|---|
| `Utilitarian()`        | maximise θ̄_wᵀψ (deploy θ̄_w) |
| `RLHF()`               | pooled BTL estimate θ̂ |
| `Strategyproof()`      | random-dictatorship: φ⁻¹(mean per-agent impact) |
| `BoundedHarm(eps)`     | max θ̄_wᵀψ s.t. per-agent harm floor θ_nᵀψ ≥ −ε·h(θ_n) |
| `PublicSpirit(gamma, psi_0)` | max θ̄_wᵀψ s.t. γ_nᵀψ ≥ γ_nᵀψ₀, γ_n = γθ̄_w+(1−γ)θ_n |

**Public spirit** (Flanigan et al. 2023): each agent tolerates personal harm in
proportion to social benefit. `gamma=0` is individual rationality (no agent worse
than the reference impact ψ₀ — defaults to the random model, ψ₀=0); `gamma=1`
collapses to unconstrained utilitarian. It reuses the same constrained-LP solver
as bounded harm — only the constraint normals and thresholds change.

**Add another mechanism:** subclass `Mechanism`, implement `deploy`/`loo`, and hand
it to `alignment_externalities`. Nothing downstream changes.

## Validation

The refactor reproduces the original notebook's Γ matrices **bitwise** (max|Δ| =
0.0) on Kidney and 412 for utilitarian / RLHF / strategyproof / bounded-harm, and
runs Community Alignment end-to-end reusing the on-disk SP cache. Public spirit is
checked against its two closed-form limits — `gamma=0, ψ₀=0` equals `BoundedHarm(0)`
and `gamma=1` equals the unconstrained utilitarian optimum (both to 0.0) — and
confirmed to bind on CoAlign (protecting the 8/100 agents harmed at the utilitarian
optimum). (Validation scripts are in the session scratchpad; rerun against `HS_DATA_ROOT`.)

The original exploratory/per-dataset notebooks (the unified Γ monolith
`alignment_externalities_general.ipynb`, the welfare notebook `empirics_general.ipynb`,
and the per-dataset one-offs) have been removed now that the package + drivers
reproduce everything; they remain recoverable in git history.

## Next steps

- **Reference impact ψ₀ for public spirit.** Defaults to the random model (ψ₀=0). If
  the paper uses the status quo or the RLHF policy as the baseline, pass `psi_0=...`
  (and bust the LOO cache, or `use_cache=False`).
