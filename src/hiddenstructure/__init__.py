"""hiddenstructure — alignment externalities & welfare mechanisms.

Pipeline #1 (built):   dataset  →  alignment externalities Γ
Pipeline #2 (planned): dataset  →  mechanism (bounded harm, public spirit)

Quick start::

    from hiddenstructure import pipeline
    run = pipeline.alignment_externalities("Kidney Allocation", verbose=True)
    run.results["RLHF"].Gamma
"""
from __future__ import annotations

from . import (
    config, impact, btl, loaders, model, mechanisms, externalities,
    plotting, pipeline, figures, welfare, paper,
)
from .loaders import (
    DatasetBundle, load_dataset, DATASET_LOADERS, stratified_subsample,
    bundle_from_frame, load_csv,
)
from .model import FittedModel, fit_model
from .mechanisms import (
    Mechanism, Utilitarian, RLHF, Strategyproof, BoundedHarm, PublicSpirit,
    CORE_MECHANISMS, WELFARE_MECHANISMS,
)
from .externalities import gamma_for, gamma_mat, GammaResult, group_summary_matrix
from .pipeline import alignment_externalities, ExternalityRun

__all__ = [
    "config", "impact", "btl", "loaders", "model", "mechanisms",
    "externalities", "plotting", "pipeline", "figures", "welfare", "paper",
    "DatasetBundle", "load_dataset", "DATASET_LOADERS", "stratified_subsample",
    "bundle_from_frame", "load_csv",
    "FittedModel", "fit_model",
    "Mechanism", "Utilitarian", "RLHF", "Strategyproof", "BoundedHarm",
    "PublicSpirit", "CORE_MECHANISMS", "WELFARE_MECHANISMS",
    "gamma_for", "gamma_mat", "GammaResult", "group_summary_matrix",
    "alignment_externalities", "ExternalityRun",
]
