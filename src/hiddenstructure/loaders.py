"""Dataset loaders — the entry point of every pipeline.

Each loader returns a :class:`DatasetBundle` with the canonical contract:

    alpha    (M, k)  feature-difference matrix (one row per pairwise choice)
    y        (M,)    binary choice label
    person_id (M,)   agent id per row
    feature_names    length-k list of feature labels
    demographics     {axis_name -> Series indexed by agent id}  (may be empty)

Paths resolve under :func:`hiddenstructure.config.data_root`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import dataset_path

MIN_OBS_DEFAULT = 20
MM_MAX_AGENTS = 200  # cap per-user agents for moralmachine (LOO tractability)


@dataclass
class DatasetBundle:
    alpha: np.ndarray
    y: np.ndarray
    person_id: np.ndarray
    feature_names: list[str]
    demographics: dict[str, pd.Series] = field(default_factory=dict)
    name: str = ""

    @property
    def k(self) -> int:
        return self.alpha.shape[1]

    @property
    def n_agents(self) -> int:
        return len(np.unique(self.person_id))


def load_412() -> DatasetBundle:
    p = dataset_path("empirics_412", "food_rescue_combined.csv")
    if not p.exists():
        p = dataset_path("food_rescue_combined.csv")
    df = pd.read_csv(p)
    feats = ["size", "access", "income", "poverty", "last_donation", "total_donation", "dist"]
    A = df[[f"{f}_A" for f in feats]].to_numpy(float)
    B = df[[f"{f}_B" for f in feats]].to_numpy(float)
    return DatasetBundle(
        alpha=A - B,
        y=(df["AorB"] == "A").astype(int).to_numpy(),
        person_id=df["personID"].to_numpy(),
        feature_names=feats,
        name="412 Food Rescue",
    )


def load_kidney() -> DatasetBundle:
    p = dataset_path("empirics_kidney", "kidneypairwise2.csv")
    if not p.exists():
        p = dataset_path("kidneypairwise2.csv")
    df = pd.read_csv(p)
    feats = ["elderlyDep", "lifeYearsGained", "obesity", "weeklyWorkhours", "yearsWaiting"]
    L = df[[f"l_{f}" for f in feats]].to_numpy(float)
    R = df[[f"r_{f}" for f in feats]].to_numpy(float)
    return DatasetBundle(
        alpha=L - R,
        y=df["chosen"].astype(int).to_numpy(),
        person_id=df["id"].to_numpy(),
        feature_names=feats,
        name="Kidney Allocation",
    )


def load_coalign(min_obs: int = MIN_OBS_DEFAULT) -> DatasetBundle:
    base = dataset_path("empirics_communityalignment")
    features = pd.read_parquet(base / "features_full_nofb.parquet")
    pw = pd.read_parquet(base / "pairwise_comparisons_ge10_nofb.parquet")
    feats = list(features.columns)
    opt_set = set(features.index)
    sub = pw[pw.preferred_text.isin(opt_set) & pw.other_text.isin(opt_set)].copy()
    counts = sub.groupby("annotator_id").size()
    sub = sub[sub.annotator_id.isin(counts[counts >= min_obs].index)].reset_index(drop=True)
    F = features.loc[sub.preferred_text].to_numpy() - features.loc[sub.other_text].to_numpy()
    M = F.shape[0]
    alpha = np.vstack([F, -F])
    y = np.concatenate([np.ones(M), np.zeros(M)]).astype(int)
    pid = np.concatenate([sub.annotator_id.to_numpy(), sub.annotator_id.to_numpy()])
    one_per_agent = sub.drop_duplicates("annotator_id").set_index("annotator_id")
    demos = _coalign_demographics(one_per_agent)
    return DatasetBundle(alpha, y, pid, feats, demos, name="Community Alignment")


def _coll_political(s):
    if s in ("Very left-leaning", "Somewhat left-leaning"):
        return "Left"
    if s in ("Somewhat right-leaning", "Very right-leaning"):
        return "Right"
    return "Center/None"


def _coll_education(s):
    if s == "Post-secondary graduate":
        return "Post-sec grad"
    if s == "Some or complete graduate degree":
        return "Grad degree"
    return "Other"


def _coll_ethnicity(s):
    return s if s in ("White", "Indo-Aryan") else "Other"


def _coalign_demographics(one_per_agent: pd.DataFrame) -> dict[str, pd.Series]:
    """Build the six CoAlign grouping axes (collapsed to a few levels each)."""
    demos: dict[str, pd.Series] = {}
    if "annotator_country" in one_per_agent:
        demos["country"] = (one_per_agent["annotator_country"].astype(str).str.title()
                            .replace({"United States": "USA"}))
    if "annotator_gender" in one_per_agent:
        demos["gender"] = one_per_agent["annotator_gender"].astype(str).str.title()
    if "annotator_age" in one_per_agent:
        demos["age"] = one_per_agent["annotator_age"]
    if "annotator_political" in one_per_agent:
        demos["political"] = one_per_agent["annotator_political"].map(_coll_political)
    if "annotator_education_level" in one_per_agent:
        demos["education"] = one_per_agent["annotator_education_level"].map(_coll_education)
    if "annotator_ethnicity" in one_per_agent:
        demos["ethnicity"] = one_per_agent["annotator_ethnicity"].map(_coll_ethnicity)
    return demos


def load_moralmachine(max_agents: int = MM_MAX_AGENTS) -> DatasetBundle:
    cached = dataset_path("empirics_moralmachine", "mid_users_filtered.csv")
    if not cached.exists():
        raise FileNotFoundError(f"need {cached} (run moralmachine notebook to generate it)")
    df = pd.read_csv(cached)
    chars = ["Man", "Woman", "Pregnant", "Stroller", "OldMan", "OldWoman", "Boy", "Girl",
             "Homeless", "LargeWoman", "LargeMan", "Criminal", "MaleExecutive",
             "FemaleExecutive", "FemaleAthlete", "MaleAthlete", "FemaleDoctor",
             "MaleDoctor", "Dog", "Cat"]
    saved = df[df["Saved"] == 1].set_index("ResponseID")
    not_saved = df[df["Saved"] == 0].set_index("ResponseID")
    common = saved.index.intersection(not_saved.index)
    A = saved.loc[common, chars] - not_saved.loc[common, chars].values
    A["UserID"] = saved.loc[common, "UserID"]
    A = A.reset_index()

    counts = A["UserID"].value_counts()
    eligible = counts[counts >= 10].index.to_numpy()
    rng = np.random.default_rng(0)
    if len(eligible) > max_agents:
        eligible = rng.choice(eligible, size=max_agents, replace=False)
    A = A[A["UserID"].isin(eligible)].reset_index(drop=True)

    alpha = A[chars].to_numpy(float)
    pid = A["UserID"].to_numpy()
    flip = rng.random(len(alpha)) < 0.5
    alpha = alpha.copy()
    alpha[flip] *= -1
    y = np.ones(len(alpha), dtype=int)
    y[flip] = 0
    return DatasetBundle(alpha, y, pid, chars, name="Moral Machine")


DATASET_LOADERS = {
    "412 Food Rescue": load_412,
    "Kidney Allocation": load_kidney,
    "Community Alignment": load_coalign,
    "Moral Machine": load_moralmachine,
}


def load_dataset(name: str) -> DatasetBundle:
    """Load one of the four datasets by its canonical name."""
    if name not in DATASET_LOADERS:
        raise ValueError(f"{name!r} not in {list(DATASET_LOADERS)}")
    return DATASET_LOADERS[name]()


# --------------------------------------------------------------------------- #
# Bring-your-own-dataset: build a DatasetBundle from a DataFrame / CSV.
# --------------------------------------------------------------------------- #
def bundle_from_frame(
    df: pd.DataFrame,
    *,
    id_col: str,
    a_cols: list[str] | None = None,
    b_cols: list[str] | None = None,
    chosen_col: str | None = None,
    chosen_a_value=None,
    alpha_cols: list[str] | None = None,
    y_col: str | None = None,
    feature_names: list[str] | None = None,
    demographic_cols: list[str] | None = None,
    symmetrize: bool = False,
    name: str = "custom",
) -> DatasetBundle:
    """Build a :class:`DatasetBundle` from one row-per-pairwise-choice frame.

    Provide the features in **one** of two ways:

    * **Two option columns** — ``a_cols`` (option A features) and ``b_cols``
      (option B features, same order). The impact features are ``A − B``.
    * **Pre-differenced** — ``alpha_cols`` already hold the A − B differences.

    And the label in one of two ways:

    * ``chosen_col`` + ``chosen_a_value`` — ``y = 1`` where option A won. Use this
      when the A/B ordering is fixed independent of the outcome (e.g. left/right).
    * ``symmetrize=True`` — use when every row is already *winner minus loser*
      (option A always won). Each row is added in both orientations, ``(α, 1)``
      and ``(−α, 0)``, exactly as the CoAlign / Moral-Machine loaders do.

    ``demographic_cols`` are exposed as grouping axes (one value per agent, taken
    from that agent's first row).
    """
    if alpha_cols is not None:
        F = df[alpha_cols].to_numpy(float)
        feats = feature_names or list(alpha_cols)
    elif a_cols is not None and b_cols is not None:
        if len(a_cols) != len(b_cols):
            raise ValueError("a_cols and b_cols must have the same length/order")
        F = df[a_cols].to_numpy(float) - df[b_cols].to_numpy(float)
        feats = feature_names or [str(c) for c in a_cols]
    else:
        raise ValueError("provide either alpha_cols, or both a_cols and b_cols")

    pid = df[id_col].to_numpy()

    if symmetrize:
        M = F.shape[0]
        alpha = np.vstack([F, -F])
        y = np.concatenate([np.ones(M), np.zeros(M)]).astype(int)
        pid = np.concatenate([pid, pid])
    else:
        if chosen_col is None:
            raise ValueError("set chosen_col (+ chosen_a_value) or symmetrize=True")
        col = df[chosen_col]
        y = (col == chosen_a_value).astype(int).to_numpy() if chosen_a_value is not None \
            else col.astype(int).to_numpy()
        alpha = F

    demos: dict[str, pd.Series] = {}
    if demographic_cols:
        one_per_agent = df.drop_duplicates(id_col).set_index(id_col)
        demos = {c: one_per_agent[c] for c in demographic_cols}

    if len(np.unique(y)) < 2:
        raise ValueError(
            "y has only one class — every row has the same label. If your rows are "
            "'winner minus loser', pass symmetrize=True instead of chosen_col.")

    return DatasetBundle(alpha=alpha, y=y, person_id=pid, feature_names=list(feats),
                         demographics=demos, name=name)


def load_csv(path, **kwargs) -> DatasetBundle:
    """Read a CSV and build a :class:`DatasetBundle`. See :func:`bundle_from_frame`."""
    return bundle_from_frame(pd.read_csv(path), **kwargs)


def stratified_subsample(
    bundle: DatasetBundle, n_total: int, stratify_by: str | None = "country",
    seed: int = 0, floor: int = 2,
) -> DatasetBundle:
    """Cap agent count via stratified sampling on a demographic axis.

    Mirrors §1.5 of ``alignment_externalities_general.ipynb``: each group gets at
    least ``floor`` agents, the rest allocated proportionally. Falls back to a
    uniform random sample if the dataset has no demographics.
    """
    rng = np.random.default_rng(seed)
    all_pids = pd.Series(bundle.person_id).unique()
    if n_total >= len(all_pids):
        return bundle

    demos = bundle.demographics
    if demos and stratify_by not in demos:
        stratify_by = next(iter(demos))

    if demos:
        strat_ser = demos[stratify_by]
        pid_to_g = {pid: strat_ser.get(pid, "(missing)") for pid in all_pids}
        g_ser = pd.Series(pid_to_g).where(lambda s: s.notna(), "(missing)").astype(str)
        counts = g_ser.value_counts()
        groups = sorted(counts.index.tolist())
        base = {g: min(floor, counts[g]) for g in groups}
        rem = max(0, n_total - sum(base.values()))
        weight = np.array([max(0, counts[g] - base[g]) for g in groups], float)
        extra = (((weight / weight.sum()) * rem).round().astype(int)
                 if weight.sum() > 0 else np.zeros(len(groups), int))
        alloc = {g: base[g] + e for g, e in zip(groups, extra)}
        while sum(alloc.values()) > n_total:
            g = max(groups, key=lambda g: alloc[g] - base[g]); alloc[g] -= 1
        while sum(alloc.values()) < n_total and any(alloc[g] < counts[g] for g in groups):
            g = max(groups, key=lambda g: counts[g] - alloc[g]); alloc[g] += 1
        keep_pids = []
        for g in groups:
            cand = g_ser[g_ser == g].index.tolist()
            keep_pids.extend(rng.choice(cand, size=min(alloc[g], len(cand)), replace=False))
        keep_pids = set(keep_pids)
    else:
        keep_pids = set(rng.choice(all_pids, size=n_total, replace=False))

    mask = pd.Series(bundle.person_id).isin(keep_pids).to_numpy()
    kept = set(bundle.person_id[mask])
    demos_sub = {k: v[v.index.isin(kept)] for k, v in demos.items()}
    return DatasetBundle(
        alpha=bundle.alpha[mask], y=bundle.y[mask], person_id=bundle.person_id[mask],
        feature_names=bundle.feature_names, demographics=demos_sub, name=bundle.name,
    )
