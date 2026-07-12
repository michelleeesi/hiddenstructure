"""Configuration: where data and caches live.

The notebooks in this repo are kept data-free (all CSV/parquet are gitignored).
The actual datasets + precomputed LOO caches live in a sibling working copy
(``../distortion`` by default). Point elsewhere with the ``HS_DATA_ROOT`` env var,
or set :data:`DATA_ROOT` directly before calling any loader.

Resolution order for the data root:
    1. ``HS_DATA_ROOT`` environment variable, if set.
    2. A sibling ``distortion`` directory next to the repo, if it exists.
    3. The repo root itself (works if you drop the data in place locally).
"""
from __future__ import annotations

import os
from pathlib import Path

# Repo root = two levels up from this file (src/hiddenstructure/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_data_root() -> Path:
    env = os.environ.get("HS_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    sibling = REPO_ROOT.parent / "distortion"
    if sibling.exists():
        return sibling.resolve()
    return REPO_ROOT


#: Root under which the per-dataset subdirectories (``empirics_412/`` etc.) live.
DATA_ROOT: Path = _default_data_root()


def data_root() -> Path:
    """Current data root (re-reads the module global so it can be monkeypatched)."""
    return DATA_ROOT


def cache_dir() -> Path:
    """Directory holding the leave-one-out ψ caches (created on demand)."""
    env = os.environ.get("HS_CACHE_DIR")
    d = Path(env).expanduser() if env else data_root() / "loo_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def dataset_path(*parts: str) -> Path:
    """Resolve a data file relative to the data root."""
    return data_root().joinpath(*parts)
