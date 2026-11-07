"""Nonlinear-utility robustness check (Appendix: linear vs. quadratic utilities).

Experiment 1 -- does a nonlinear utility fit held-out choices better?
    For every agent, fit a Bradley-Terry-Luce choice model under feature maps of the
    per-option features phi(x):

        linear        alpha = phi(A) - phi(B)
        quadratic     alpha = [phi(A) - phi(B),  phi(A)^2 - phi(B)^2]   (per-feature squares)
        interaction   quadratic + all pairwise products phi_k phi_l   (Moral Machine only)

    and compare them by 5-fold cross-validation on that agent's own comparisons,
    reporting held-out log loss and accuracy averaged across agents.

Experiment 2 -- where a nonlinear utility fits better (Kidney Allocation), does
    anything change? A quadratic utility is a linear utility over the augmented
    basis, so the pipeline is re-run in that basis and compared with the linear
    run: correlation of per-agent welfare under utilitarian deployment, identity
    of the best- and worst-off agents, correlation of the participation
    externality matrix, and the welfare regret of deploying the linear utilitarian
    decisions when utilities are really quadratic.

Specification:
    * per-option features are z-scored on the whole dataset before squaring;
    * every model is the package's BTL fit (logistic, no intercept, L2 with C = 1);
    * 5-fold, shuffled, seed 0, stratified by outcome where it varies. Community
      Alignment and Moral Machine rows are winner-minus-loser, so folds are drawn
      over comparisons and only the training half is mirrored into both
      orientations (a held-out comparison's mirror is never in training);
    * agents are those the paper keeps (>= 20 comparisons, both classes present);
      Moral Machine uses every eligible annotator.

Usage (from the repo root)::

    HS_DATA_ROOT=/path/to/data python scripts/nonlinear_robustness.py [--out DIR]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
from sklearn.model_selection import KFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiddenstructure.btl import fit_btl  # noqa: E402
from hiddenstructure.config import dataset_path  # noqa: E402
from hiddenstructure.externalities import gamma_for  # noqa: E402
from hiddenstructure.impact import psi  # noqa: E402
from hiddenstructure.loaders import DatasetBundle  # noqa: E402
from hiddenstructure.mechanisms import Utilitarian  # noqa: E402
from hiddenstructure.model import DEFAULT_MIN_OBS, fit_model  # noqa: E402

SEED = 0
N_SPLITS = 5


# --------------------------------------------------------------------------- #
# Per-option loaders: (A, B, y, pid, feature_names, symmetric)
#   A, B (M, k) per-option features; y = 1 if A chosen; symmetric = rows are
#   winner-minus-loser (y all ones).
# --------------------------------------------------------------------------- #
def load_412():
    p = dataset_path("empirics_412", "food_rescue_combined.csv")
    df = pd.read_csv(p if p.exists() else dataset_path("food_rescue_combined.csv"))
    feats = ["size", "access", "income", "poverty", "last_donation", "total_donation", "dist"]
    A = df[[f"{f}_A" for f in feats]].to_numpy(float)
    B = df[[f"{f}_B" for f in feats]].to_numpy(float)
    return A, B, (df["AorB"] == "A").astype(int).to_numpy(), df["personID"].to_numpy(), feats, False


def load_kidney():
    p = dataset_path("empirics_kidney", "kidneypairwise2.csv")
    df = pd.read_csv(p if p.exists() else dataset_path("kidneypairwise2.csv"))
    feats = ["elderlyDep", "lifeYearsGained", "obesity", "weeklyWorkhours", "yearsWaiting"]
    L = df[[f"l_{f}" for f in feats]].to_numpy(float)
    R = df[[f"r_{f}" for f in feats]].to_numpy(float)
    return L, R, df["chosen"].astype(int).to_numpy(), df["id"].to_numpy(), feats, False


def load_coalign():
    base = dataset_path("empirics_communityalignment")
    features = pd.read_parquet(base / "features_full_nofb.parquet")
    pw = pd.read_parquet(base / "pairwise_comparisons_ge10_nofb.parquet")
    opt_set = set(features.index)
    sub = pw[pw.preferred_text.isin(opt_set) & pw.other_text.isin(opt_set)].copy()
    counts = sub.groupby("annotator_id").size()
    sub = sub[sub.annotator_id.isin(counts[counts >= DEFAULT_MIN_OBS].index)].reset_index(drop=True)
    A = features.loc[sub.preferred_text].to_numpy(float)
    B = features.loc[sub.other_text].to_numpy(float)
    return A, B, np.ones(len(sub), int), sub.annotator_id.to_numpy(), list(features.columns), True


def load_mm():
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
    return A, B, np.ones(len(common), int), saved.loc[common, "UserID"].to_numpy(), chars, True


DATASETS = [
    ("Kidney Allocation", load_kidney, ["linear", "quadratic"]),
    ("412 Food Rescue", load_412, ["linear", "quadratic"]),
    ("Moral Machine", load_mm, ["linear", "quadratic", "interaction"]),
    ("Community Alignment", load_coalign, ["linear", "quadratic"]),
]


# --------------------------------------------------------------------------- #
def standardize(A, B):
    S = np.vstack([A, B])
    mu, sd = S.mean(axis=0), S.std(axis=0)
    sd[sd == 0] = 1.0
    return (A - mu) / sd, (B - mu) / sd


def fmap(X, model):
    if model == "linear":
        return X
    if model == "quadratic":
        return np.hstack([X, X ** 2])
    k = X.shape[1]
    prods = np.column_stack([X[:, i] * X[:, j] for i, j in itertools.combinations(range(k), 2)])
    return np.hstack([X, X ** 2, prods])


def cv_predict(alpha, y, symmetric):
    """Out-of-fold P(y = 1) for one agent."""
    if symmetric or np.bincount(y, minlength=2).min() < N_SPLITS:
        folds = KFold(N_SPLITS, shuffle=True, random_state=SEED).split(alpha)
    else:
        folds = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(alpha, y)
    p = np.empty(len(y))
    for tr, te in folds:
        a_tr, y_tr = alpha[tr], y[tr]
        if symmetric:
            a_tr = np.vstack([a_tr, -a_tr])
            y_tr = np.r_[np.ones(len(tr), int), np.zeros(len(tr), int)]
        th = fit_btl(a_tr, y_tr)
        p[te] = sigmoid(alpha[te] @ th) if th is not None else np.clip(y_tr.mean(), 0.01, 0.99)
    return p


def metrics(y, p):
    pc = np.clip(p, 1e-12, 1 - 1e-12)
    return (float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
            float(np.mean((p > 0.5) == (y == 1))))


def experiment1(name, loader, models):
    A, B, y, pid, _, symmetric = loader()
    A, B = standardize(A, B)
    alphas = {m: fmap(A, m) - fmap(B, m) for m in models}
    rows, preds = [], {m: np.full(len(y), np.nan) for m in models}
    for u in pd.Series(pid).unique():
        mask = pid == u
        if mask.sum() < DEFAULT_MIN_OBS or (not symmetric and len(np.unique(y[mask])) < 2):
            continue
        rec = {"agent": u, "n": int(mask.sum())}
        for m in models:
            p = cv_predict(alphas[m][mask], y[mask], symmetric)
            preds[m][mask] = p
            rec[f"ll_{m}"], rec[f"acc_{m}"] = metrics(y[mask], p)
        rows.append(rec)
    df = pd.DataFrame(rows)
    out = {"n_agents": int(len(df)), "comparisons_per_agent": float(df["n"].mean())}
    for m in models:
        out[f"ll_{m}"], out[f"acc_{m}"] = float(df[f"ll_{m}"].mean()), float(df[f"acc_{m}"].mean())
    out["agents_helped_quadratic"] = int((df["ll_quadratic"] < df["ll_linear"]).sum())
    if name == "Kidney Allocation":
        out.update(kidney_repeats(A, B, y, pid, preds))

    print(f"\n[{name}]  agents={out['n_agents']}  comparisons/agent={out['comparisons_per_agent']:.0f}")
    for m in models:
        print(f"  {m:12s} log loss {out[f'll_{m}']:.3f}   accuracy {100 * out[f'acc_{m}']:.1f}%")
    print(f"  quadratic beats linear on log loss for {out['agents_helped_quadratic']} of {out['n_agents']} agents")
    if name == "Kidney Allocation":
        print(f"  repeated comparisons: {100 * out['repeat_share']:.1f}% of rows; test-retest ceiling "
              f"{100 * out['repeat_ceiling']:.1f}%; out-of-fold accuracy on them linear "
              f"{100 * out['acc_linear_repeats']:.1f}%, quadratic {100 * out['acc_quadratic_repeats']:.1f}%")
    return out, df


def kidney_repeats(A, B, y, pid, preds):
    """Rows whose (agent, unordered patient pair) occurs more than once; the ceiling is
    agreement with the agent's majority choice within each repeated pair."""
    pair = [repr(tuple(sorted([tuple(a), tuple(b)]))) for a, b in zip(A, B)]
    chosen = [repr(tuple(a) if c == 1 else tuple(b)) for a, b, c in zip(A, B, y)]
    df = pd.DataFrame({"agent": pid.astype(str), "pair": pair, "chosen": chosen})
    grp = df.groupby(["agent", "pair"], sort=False)["chosen"]
    rep = (grp.transform("size") >= 2).to_numpy()
    agree = (df["chosen"] == grp.transform(lambda s: s.value_counts().index[0])).to_numpy()
    sel = rep & ~np.isnan(preds["linear"])
    return {"repeat_share": float(rep.mean()), "repeat_ceiling": float(agree[rep].mean()),
            "acc_linear_repeats": float(np.mean((preds["linear"][sel] > 0.5) == (y[sel] == 1))),
            "acc_quadratic_repeats": float(np.mean((preds["quadratic"][sel] > 0.5) == (y[sel] == 1)))}


# --------------------------------------------------------------------------- #
def experiment2():
    L, R, y, pid, feats, _ = load_kidney()
    L, R = standardize(L, R)
    m_lin = fit_model(DatasetBundle(L - R, y, pid, feats, name="Kidney Allocation"), seed=SEED)
    m_q = fit_model(DatasetBundle(fmap(L, "quadratic") - fmap(R, "quadratic"), y, pid,
                                  feats + [f + "^2" for f in feats], name="Kidney (quadratic)"), seed=SEED)
    assert np.array_equal(m_lin.agent_ids, m_q.agent_ids)
    assert np.allclose(m_q.alpha_q[:, :m_lin.k], m_lin.alpha_q)

    # per-agent welfare under utilitarian deployment (package default beta), normalised by U*
    U_lin = m_lin.Theta @ psi(m_lin.theta_bar, m_lin.alpha_q, m_lin.beta) / m_lin.W_star
    U_q = m_q.Theta @ psi(m_q.theta_bar, m_q.alpha_q, m_q.beta) / m_q.W_star
    G_lin = gamma_for(m_lin, Utilitarian(), use_cache=False).Gamma
    G_q = gamma_for(m_q, Utilitarian(), use_cache=False).Gamma

    # deterministic limit: regret of the linear utilitarian decisions under quadratic utilities
    s_lin = np.sign(m_lin.alpha_q @ m_lin.theta_bar)
    s_q = np.sign(m_q.alpha_q @ m_q.theta_bar)
    U_star = m_q.theta_bar @ (0.5 * (s_q[:, None] * m_q.alpha_q).mean(axis=0))
    U_cross = m_q.theta_bar @ (0.5 * (s_lin[:, None] * m_q.alpha_q).mean(axis=0))

    out = {"n_agents": int(m_lin.N),
           "welfare_corr": float(np.corrcoef(U_lin, U_q)[0, 1]),
           "best_agent_same": bool(U_lin.argmax() == U_q.argmax()),
           "worst_agent_same": bool(U_lin.argmin() == U_q.argmin()),
           "externality_corr": float(np.corrcoef(G_lin.ravel(), G_q.ravel())[0, 1]),
           "regret": float(1 - U_cross / U_star),
           "queries_decided_differently": float(np.mean(s_lin != s_q))}
    print(f"\n[Experiment 2: Kidney Allocation, linear (k=5) vs quadratic (k=10) basis]  N={out['n_agents']}")
    print(f"  per-agent welfare correlation {out['welfare_corr']:.2f}; best-off agent same: {out['best_agent_same']}; "
          f"worst-off agent same: {out['worst_agent_same']}")
    print(f"  participation externality correlation {out['externality_corr']:.2f}")
    print(f"  welfare regret of linear utilitarian deployment under quadratic utilities {100 * out['regret']:.2f}% "
          f"(decisions differ on {100 * out['queries_decided_differently']:.1f}% of queries)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=None, help="directory for a results JSON and per-agent CSVs")
    args = ap.parse_args()
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    results = {"experiment1": {}}
    for name, loader, models in DATASETS:
        summary, per_agent = experiment1(name, loader, models)
        results["experiment1"][name] = summary
        if args.out:
            per_agent.to_csv(args.out / f"exp1_{name.replace(' ', '_').lower()}.csv", index=False)
    results["experiment2"] = experiment2()
    if args.out:
        (args.out / "nonlinear_robustness.json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
