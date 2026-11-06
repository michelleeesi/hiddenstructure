"""Reproduce the cost-of-strategyproofness table (Table tab:sp-cost in the appendix).

For each dataset, fits the per-agent BTL preferences with the package defaults and
reports, at saturated choice precision (beta = infinity):

    relative welfare loss   1 - E[U(mu_sp)] / U*   of the uniform random dictatorship,
    average majority margin M = E_z | (1/N) sum_n sign(alpha_z^T theta_n) |.

Both expectations are over the deployment query sub-sample ``alpha_q`` that the
rest of the pipeline uses (5000 queries, seed 0). Moral Machine is run on all
eligible annotators rather than the loader's 200-agent cap, matching the paper.

Usage (from the repo root)::

    HS_DATA_ROOT=/path/to/data python scripts/sp_cost_table.py [--all-queries]

``HS_DATA_ROOT`` must contain ``empirics_412/food_rescue_combined.csv``,
``empirics_kidney/kidneypairwise2.csv``, ``empirics_moralmachine/mid_users_filtered.csv``
and ``empirics_communityalignment/{features_full_nofb,pairwise_comparisons_ge10_nofb}.parquet``
(see ``hiddenstructure.config`` for the resolution rules).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiddenstructure.loaders import load_dataset, load_moralmachine  # noqa: E402
from hiddenstructure.model import fit_model  # noqa: E402

DATASETS = [
    ("Kidney Allocation", lambda: load_dataset("Kidney Allocation")),
    ("Community Alignment", lambda: load_dataset("Community Alignment")),
    ("412 Food Rescue", lambda: load_dataset("412 Food Rescue")),
    ("Moral Machine", lambda: load_moralmachine(max_agents=10**9)),
]


def sp_cost(Theta: np.ndarray, theta_bar: np.ndarray, alpha: np.ndarray) -> tuple[float, float]:
    """Return (relative welfare loss, average majority margin) at beta = infinity."""
    signs = np.sign(alpha @ Theta.T)                        # (Z, N)
    psi_n_star = 0.5 * (signs.T @ alpha) / len(alpha)       # (N, k) ideal impacts
    psi_sp = psi_n_star.mean(axis=0)                        # uniform random dictatorship
    psi_star = 0.5 * (np.sign(alpha @ theta_bar)[:, None] * alpha).mean(axis=0)
    U_star = float(theta_bar @ psi_star)
    U_sp = float(theta_bar @ psi_sp)
    margin = float(np.abs(signs.mean(axis=1)).mean())
    return 1.0 - U_sp / U_star, margin


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--all-queries", action="store_true",
                    help="take expectations over all training queries instead of alpha_q")
    args = ap.parse_args()

    print(f"{'Dataset':22s} {'N':>5s} {'Relative welfare loss':>22s} {'Average majority margin':>24s}")
    for name, loader in DATASETS:
        t0 = time.time()
        model = fit_model(loader())
        alpha = model.alpha_train if args.all_queries else model.alpha_q
        loss, margin = sp_cost(model.Theta, model.theta_bar, alpha)
        print(f"{name:22s} {model.N:5d} {100 * loss:21.1f}% {margin:24.2f}   [{time.time() - t0:.0f}s]",
              flush=True)


if __name__ == "__main__":
    main()
