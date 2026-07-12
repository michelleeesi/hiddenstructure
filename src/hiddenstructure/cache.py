"""On-disk cache for expensive leave-one-out ψ arrays.

Naming matches the original notebooks so existing caches under
``<data_root>/loo_cache`` are reused verbatim:

    {dataset}_{tag}_N{N}_beta{beta}.npz   ->  psi_loo (N, k), agent_ids (N,)

A cache is treated as stale (and ignored) if its stored ``agent_ids`` no longer
match the current agent set.
"""
from __future__ import annotations

import numpy as np

from .config import cache_dir


def cache_path(dataset: str, tag: str, n_agents: int, beta: float):
    return cache_dir() / f"{dataset}_{tag}_N{n_agents}_beta{beta}.npz"


def load_psi_loo(dataset: str, tag: str, n_agents: int, beta: float, agent_ids) -> np.ndarray | None:
    p = cache_path(dataset, tag, n_agents, beta)
    if not p.exists():
        return None
    blob = np.load(p, allow_pickle=True)
    if not np.array_equal(blob["agent_ids"], agent_ids):
        return None
    return blob["psi_loo"]


def save_psi_loo(dataset: str, tag: str, n_agents: int, beta: float, psi_loo, agent_ids) -> None:
    p = cache_path(dataset, tag, n_agents, beta)
    np.savez_compressed(p, psi_loo=psi_loo, agent_ids=agent_ids)
