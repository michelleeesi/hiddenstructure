"""One-call pipeline:  dataset name  →  alignment externalities.

    from hiddenstructure import pipeline
    out = pipeline.alignment_externalities("Kidney Allocation")
    out.results["RLHF"].Gamma        # the Γ matrix
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .externalities import GammaResult, gamma_for
from .loaders import DatasetBundle, load_dataset, stratified_subsample
from .mechanisms import CORE_MECHANISMS, Mechanism
from .model import FittedModel, fit_model


@dataclass
class ExternalityRun:
    model: FittedModel
    results: dict[str, GammaResult] = field(default_factory=dict)


def alignment_externalities(
    dataset: str | DatasetBundle,
    mechanisms: list[Mechanism] | None = None,
    *,
    stratified_n: int | None = None,
    stratify_by: str = "country",
    C: float = 1.0,
    min_obs: int = 20,
    beta: float = 1.0,
    query_sample: int = 5000,
    seed: int = 0,
    use_cache: bool = True,
    verbose: bool = False,
) -> ExternalityRun:
    """Fit BTL and compute Γ for each mechanism.

    ``dataset`` is either a registered dataset name (one of the four built-ins)
    or a :class:`DatasetBundle` you built yourself (e.g. via
    :func:`hiddenstructure.loaders.load_csv`). ``mechanisms`` defaults to the
    three core mechanisms; set ``stratified_n`` to cap agents for LOO tractability.
    """
    bundle = dataset if isinstance(dataset, DatasetBundle) else load_dataset(dataset)
    if stratified_n is not None:
        bundle = stratified_subsample(bundle, stratified_n, stratify_by, seed=seed)
    model = fit_model(bundle, C=C, min_obs=min_obs, beta=beta,
                      query_sample=query_sample, seed=seed)
    mechanisms = mechanisms if mechanisms is not None else CORE_MECHANISMS
    results = {}
    for mech in mechanisms:
        res = gamma_for(model, mech, use_cache=use_cache, verbose=verbose)
        results[mech.name] = res
        if verbose:
            print(res.summary())
    return ExternalityRun(model=model, results=results)
