"""Linearity robustness study for the rebuttal (rebuttal/linearity_robustness.md).

Rebuilt from the recipe in §4 and §10 of that doc. The scripts that produced the
doc's numbers were lost, so some details below (CV seeds, agent filters,
deployment query sample, how Γ is compared) are reconstructions; see the
"Reconstruction choices" notes at the bottom of this docstring.

For each dataset this script:

  Q1  per-agent 5-fold CV of BTL utilities in three bases, all logistic with no
      intercept and C=1 on features standardized over pooled option rows:
        linear  φ_A − φ_B
        quad    [φ_A − φ_B, φ_A² − φ_B²]                (additive curvature / GAM)
        inter   quad + φ_A,k φ_A,l − φ_B,k φ_B,l, k<l   (Moral Machine only)
      Preferred-only datasets (Moral Machine, Community Alignment) are
      symmetrized inside each training fold and scored on the held-out choices.
  Ceiling  (Kidney) LOO majority vote over exact repeats of the same
      (agent, unordered profile pair); compared with out-of-fold accuracy of
      each basis on the same rows.
  Q2  per-agent fit_btl on all data in each basis → utilitarian ψ(θ̄), per-agent
      welfare U_n = θ_nᵀψ, Γ via the closed-form utilitarian LOO; linear vs.
      nonlinear compared by Pearson/Spearman, best/worst-off agent, Γ sign
      agreement and top-5 imposer/receiver overlap.
  Regret  (β→∞) treating the nonlinear basis as the true utility:
      1 − U(ψ_lin-deploy) / U(ψ*_nonlinear).
  Reliability  split each agent's data in half, refit, recompute Γ from each
      half, correlate (reported for every dataset; the doc uses it for CA).

Usage (from the repo root)::

    HS_DATA_ROOT=/path/to/data python scripts/linearity_robustness.py [--datasets kidney 412 mm ca]

Reconstruction choices (not pinned down by the doc):
  * Agents: >= 20 comparisons (package default min_obs), except Kidney, which
    uses >= 100: that gives the doc's 77 agents and reproduces its Q1 numbers
    (the 4 dropped agents have 2-67 comparisons). Moral Machine uses the
    loader's 200-agent sample (users with >= 10 responses, rng seed 0).
  * CV: KFold(5, shuffle, random_state=0) per agent; folds whose training set
    has one class are skipped. Per-agent loss = mean over held-out rows.
  * Deployment queries: 5000-row sub-sample, seed 0 (as in fit_model); β = 1 for
    ψ, welfare and Γ.
  * Γ comparisons use off-diagonal entries only.
  * Split-half uses one random half split per agent (seed 0).
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiddenstructure.btl import fit_btl  # noqa: E402
from hiddenstructure.config import dataset_path  # noqa: E402
from hiddenstructure.impact import psi  # noqa: E402
from hiddenstructure.loaders import MM_MAX_AGENTS  # noqa: E402

C = 1.0
MIN_OBS = 20
MIN_OBS_BY_DATASET = {"kidney": 100}
N_FOLDS = 5
QUERY_SAMPLE = 5000
BETA = 1.0
SEED = 0
EPS = 1e-12


# --------------------------------------------------------------------------- #
# Raw option-level data (the package loaders only keep φ_A − φ_B).
# --------------------------------------------------------------------------- #
@dataclass
class RawData:
    name: str
    A: np.ndarray          # (M, k) option A features (winner, if preferred_only)
    B: np.ndarray          # (M, k) option B features
    y: np.ndarray          # (M,) 1 if A chosen (all ones if preferred_only)
    pid: np.ndarray        # (M,)
    preferred_only: bool
    feature_names: list
    interactions: bool = False


def raw_kidney() -> RawData:
    df = pd.read_csv(dataset_path("empirics_kidney", "kidneypairwise2.csv"))
    f = ["elderlyDep", "lifeYearsGained", "obesity", "weeklyWorkhours", "yearsWaiting"]
    return RawData("Kidney Allocation", df[[f"l_{c}" for c in f]].to_numpy(float),
                   df[[f"r_{c}" for c in f]].to_numpy(float),
                   df["chosen"].astype(int).to_numpy(), df["id"].to_numpy(), False, f)


def raw_412() -> RawData:
    df = pd.read_csv(dataset_path("empirics_412", "food_rescue_combined.csv"))
    f = ["size", "access", "income", "poverty", "last_donation", "total_donation", "dist"]
    return RawData("412 Food Rescue", df[[f"{c}_A" for c in f]].to_numpy(float),
                   df[[f"{c}_B" for c in f]].to_numpy(float),
                   (df["AorB"] == "A").astype(int).to_numpy(), df["personID"].to_numpy(),
                   False, f)


def raw_moralmachine(max_agents: int = MM_MAX_AGENTS) -> RawData:
    """Same agent sample as loaders.load_moralmachine, kept as saved-vs-not-saved."""
    df = pd.read_csv(dataset_path("empirics_moralmachine", "mid_users_filtered.csv"))
    chars = ["Man", "Woman", "Pregnant", "Stroller", "OldMan", "OldWoman", "Boy", "Girl",
             "Homeless", "LargeWoman", "LargeMan", "Criminal", "MaleExecutive",
             "FemaleExecutive", "FemaleAthlete", "MaleAthlete", "FemaleDoctor",
             "MaleDoctor", "Dog", "Cat"]
    saved = df[df["Saved"] == 1].set_index("ResponseID")
    not_saved = df[df["Saved"] == 0].set_index("ResponseID")
    common = saved.index.intersection(not_saved.index)
    A = saved.loc[common, chars].to_numpy(float)
    B = not_saved.loc[common, chars].to_numpy(float)
    users = saved.loc[common, "UserID"].to_numpy()
    counts = pd.Series(users).value_counts()
    eligible = counts[counts >= 10].index.to_numpy()
    rng = np.random.default_rng(0)
    if len(eligible) > max_agents:
        eligible = rng.choice(eligible, size=max_agents, replace=False)
    keep = np.isin(users, eligible)
    return RawData("Moral Machine", A[keep], B[keep], np.ones(keep.sum(), int), users[keep],
                   True, chars, interactions=True)


def raw_coalign() -> RawData:
    base = dataset_path("empirics_communityalignment")
    features = pd.read_parquet(base / "features_full_nofb.parquet")
    pw = pd.read_parquet(base / "pairwise_comparisons_ge10_nofb.parquet")
    opt = set(features.index)
    sub = pw[pw.preferred_text.isin(opt) & pw.other_text.isin(opt)]
    counts = sub.groupby("annotator_id").size()
    sub = sub[sub.annotator_id.isin(counts[counts >= MIN_OBS].index)].reset_index(drop=True)
    A = features.loc[sub.preferred_text].to_numpy(float)
    B = features.loc[sub.other_text].to_numpy(float)
    return RawData("Community Alignment", A, B, np.ones(len(sub), int),
                   sub.annotator_id.to_numpy(), True, list(features.columns))


RAW_LOADERS = {"kidney": raw_kidney, "412": raw_412, "mm": raw_moralmachine, "ca": raw_coalign}


# --------------------------------------------------------------------------- #
# Bases
# --------------------------------------------------------------------------- #
def designs(raw: RawData) -> dict:
    """Feature-difference designs per basis, on pooled-standardized features."""
    pooled = np.vstack([raw.A, raw.B])
    mu, sd = pooled.mean(0), pooled.std(0)
    sd[sd == 0] = 1.0
    A, B = (raw.A - mu) / sd, (raw.B - mu) / sd
    out = {"linear": A - B}
    out["quad"] = np.hstack([out["linear"], A ** 2 - B ** 2])
    if raw.interactions:
        pairs = list(combinations(range(A.shape[1]), 2))
        inter = np.column_stack([A[:, i] * A[:, j] - B[:, i] * B[:, j] for i, j in pairs])
        out["inter"] = np.hstack([out["quad"], inter])
    return out


def fit_sym(X: np.ndarray, y: np.ndarray, preferred_only: bool):
    """BTL fit; preferred-only rows are symmetrized ([X; −X], [1; 0])."""
    if preferred_only:
        X, y = np.vstack([X, -X]), np.r_[np.ones(len(X)), np.zeros(len(X))].astype(int)
    return fit_btl(X, y, C=C)


# --------------------------------------------------------------------------- #
# Q1: within-agent CV
# --------------------------------------------------------------------------- #
def cv_agent(X: np.ndarray, y: np.ndarray, preferred_only: bool):
    """Out-of-fold P(y=1) per row (nan where the fold could not be fit)."""
    p = np.full(len(y), np.nan)
    for tr, te in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(X):
        th = fit_sym(X[tr], y[tr], preferred_only)
        if th is not None:
            p[te] = sigmoid(X[te] @ th)
    return p


def q1(raw: RawData, D: dict, agents: np.ndarray) -> tuple[pd.DataFrame, dict]:
    rows, oof = [], {b: np.full(len(raw.y), np.nan) for b in D}
    for a in agents:
        m = raw.pid == a
        rec = {"agent": a}
        for b, X in D.items():
            p = cv_agent(X[m], raw.y[m], raw.preferred_only)
            oof[b][m] = p
            ok = ~np.isnan(p)
            yy, pp = raw.y[m][ok], np.clip(p[ok], EPS, 1 - EPS)
            rec[f"{b}_ll"] = -np.mean(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))
            rec[f"{b}_acc"] = np.mean((pp > 0.5) == yy)
        rows.append(rec)
    return pd.DataFrame(rows), oof


def kidney_ceiling(raw: RawData, oof: dict) -> dict:
    """Identical-stimulus ceiling: LOO majority over repeats of the same pair."""
    L = [tuple(r) for r in raw.A]
    R = [tuple(r) for r in raw.B]
    key, chose_first = [], []
    for l, r, y in zip(L, R, raw.y):
        lo, hi = (l, r) if l <= r else (r, l)
        key.append((lo, hi))
        chosen = l if y == 1 else r
        chose_first.append(int(chosen == lo))
    df = pd.DataFrame({"pid": raw.pid, "key": key, "c": chose_first})
    df["cell"] = df.groupby(["pid", "key"]).ngroup()
    size = df.groupby("cell")["c"].transform("size")
    tot = df.groupby("cell")["c"].transform("sum")
    rep = (size >= 2).to_numpy()
    others_mean = ((tot - df["c"]) / (size - 1)).to_numpy()[rep]
    c = df["c"].to_numpy()[rep]
    # LOO majority; a tie scores 0.5
    acc_rows = np.where(others_mean == 0.5, 0.5, ((others_mean > 0.5) == c).astype(float))
    out = {"repeat_share": rep.mean(), "ceiling_acc": acc_rows.mean()}
    for b, p in oof.items():
        pr, yy = p[rep], raw.y[rep]
        ok = ~np.isnan(pr)
        out[f"{b}_acc_on_repeats"] = np.mean((pr[ok] > 0.5) == yy[ok])
    return out


# --------------------------------------------------------------------------- #
# Q2: estimands
# --------------------------------------------------------------------------- #
def fit_all(raw, X, agents, rows_mask=None):
    Th = []
    for a in agents:
        m = raw.pid == a
        if rows_mask is not None:
            m = m & rows_mask
        Th.append(fit_sym(X[m], raw.y[m], raw.preferred_only))
    return Th


def util_gamma(Theta: np.ndarray, alpha_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Utilitarian ψ, welfare and Γ with the closed-form LOO mean update."""
    N = len(Theta)
    tbar = Theta.mean(0)
    psi_full = psi(tbar, alpha_q, BETA)
    theta_loo = (N * tbar[None, :] - Theta) / (N - 1)
    psi_loo = np.array([psi(t, alpha_q, BETA) for t in theta_loo])
    G = (psi_full[None, :] - psi_loo) @ Theta.T
    return Theta @ psi_full, G


def offdiag(G):
    return G[~np.eye(len(G), dtype=bool)]


def regret_inf(theta_lin_bar, theta_nl_bar, aq_lin, aq_nl) -> float:
    """1 − true(nonlinear) welfare of linear-optimal deployment / nonlinear optimum (β=∞)."""
    s_lin = np.sign(aq_lin @ theta_lin_bar)
    s_nl = np.sign(aq_nl @ theta_nl_bar)
    U_opt = float(theta_nl_bar @ (0.5 * (s_nl[:, None] * aq_nl).mean(0)))
    U_lin = float(theta_nl_bar @ (0.5 * (s_lin[:, None] * aq_nl).mean(0)))
    return 1.0 - U_lin / U_opt


def compare(W1, G1, W2, G2) -> dict:
    g1, g2 = offdiag(G1), offdiag(G2)
    top = lambda v: set(np.argsort(-v)[:5])  # noqa: E731
    return {
        "welfare_pearson": pearsonr(W1, W2)[0], "welfare_spearman": spearmanr(W1, W2)[0],
        "best_same": int(np.argmax(W1) == np.argmax(W2)),
        "worst_same": int(np.argmin(W1) == np.argmin(W2)),
        "gamma_pearson": pearsonr(g1, g2)[0], "gamma_spearman": spearmanr(g1, g2)[0],
        "gamma_sign_agree": np.mean(np.sign(g1) == np.sign(g2)),
        "top5_imposers": len(top(offdiag_rows(G1)) & top(offdiag_rows(G2))),
        "top5_receivers": len(top(offdiag_rows(G1.T)) & top(offdiag_rows(G2.T))),
    }


def offdiag_rows(G):
    return G.sum(1) - np.diag(G)


# --------------------------------------------------------------------------- #
def run(key: str) -> None:
    t0 = time.time()
    raw = RAW_LOADERS[key]()
    D = designs(raw)
    counts = pd.Series(raw.pid).value_counts()
    min_obs = MIN_OBS_BY_DATASET.get(key, MIN_OBS)
    agents = np.array([a for a in pd.unique(raw.pid) if counts[a] >= min_obs])
    # agents need a fit in every basis (both classes, for non-symmetrized data)
    full = {b: fit_all(raw, X, agents) for b, X in D.items()}
    ok = np.array([all(full[b][i] is not None for b in D) for i in range(len(agents))])
    agents = agents[ok]
    Theta = {b: np.vstack([t for t, o in zip(full[b], ok) if o]) for b in D}
    in_agents = np.isin(raw.pid, agents)

    print(f"\n=== {raw.name}: N={len(agents)}, comps/agent median "
          f"{int(counts[agents].median())}, k={len(raw.feature_names)} ===")

    # Q1
    cv, oof = q1(raw, D, agents)
    for b in D:
        print(f"  Q1 {b:6s} log-loss {cv[f'{b}_ll'].mean():.4f}  acc {cv[f'{b}_acc'].mean():.3f}")
    for b in [b for b in D if b != "linear"]:
        d = cv["linear_ll"] - cv[f"{b}_ll"]
        print(f"  Q1 {b} helps {int((d > 0).sum())}/{len(cv)} agents "
              f"(>0.01 nats: {int((d > 0.01).sum())}, hurt >0.01: {int((d < -0.01).sum())})")
    if key == "kidney":
        oof_in = {b: np.where(in_agents, p, np.nan) for b, p in oof.items()}
        ce = kidney_ceiling(raw, oof_in)
        print(f"  Ceiling: repeats {100 * ce['repeat_share']:.0f}% of rows, "
              f"human ceiling acc {100 * ce['ceiling_acc']:.1f}%, " + ", ".join(
                  f"{b} {100 * ce[f'{b}_acc_on_repeats']:.1f}%" for b in D))

    # Q2 + regret
    rng = np.random.default_rng(SEED)
    rows = np.flatnonzero(in_agents)
    q = rng.choice(rows, size=QUERY_SAMPLE, replace=False) if len(rows) > QUERY_SAMPLE else rows
    aq = {b: X[q] for b, X in D.items()}
    WG = {b: util_gamma(Theta[b], aq[b]) for b in D}
    for b in [b for b in D if b != "linear"]:
        c = compare(*WG["linear"], *WG[b])
        r = regret_inf(Theta["linear"].mean(0), Theta[b].mean(0), aq["linear"], aq[b])
        print(f"  Q2 linear vs {b}: welfare r={c['welfare_pearson']:.3f} "
              f"rho={c['welfare_spearman']:.3f} best/worst same {c['best_same']}/{c['worst_same']}; "
              f"Γ r={c['gamma_pearson']:.3f} rho={c['gamma_spearman']:.3f} "
              f"sign {c['gamma_sign_agree']:.2f} top5 imp {c['top5_imposers']}/5 "
              f"rec {c['top5_receivers']}/5; regret {100 * r:.2f}%")

    # Reliability: split-half Γ self-correlation
    half = np.zeros(len(raw.y), bool)
    for a in agents:
        idx = np.flatnonzero(raw.pid == a)
        half[rng.permutation(idx)[: len(idx) // 2]] = True
    for b, X in D.items():
        h1, h2 = fit_all(raw, X, agents, half), fit_all(raw, X, agents, ~half)
        keep = [i for i in range(len(agents)) if h1[i] is not None and h2[i] is not None]
        G1 = util_gamma(np.vstack([h1[i] for i in keep]), aq[b])[1]
        G2 = util_gamma(np.vstack([h2[i] for i in keep]), aq[b])[1]
        print(f"  Reliability {b:6s}: split-half Γ r={pearsonr(offdiag(G1), offdiag(G2))[0]:.3f} "
              f"(N={len(keep)})")
    print(f"  [{time.time() - t0:.0f}s]", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", nargs="+", default=list(RAW_LOADERS), choices=list(RAW_LOADERS))
    args = ap.parse_args()
    for key in args.datasets:
        run(key)


if __name__ == "__main__":
    main()
