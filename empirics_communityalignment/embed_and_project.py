"""
Embed all unique (prompt, response) pairs from pairwise_comparisons_ge10.parquet
with Qwen/Qwen3-Embedding-8B, project onto the 15 direction vectors V from
coalign_50_dimensions.tar.gz, and save a (N_responses × 15) feature lookup.

Output: features_full.parquet
  index = response_text  (unique key — verified 1:1 with prompt in CoAlign)
  cols  = the 15 dimension names

Drop-in for CA_alignment_externalities_smoke.ipynb §1 — replace the bt_scores
pivot with:

    features = pd.read_parquet("features_full.parquet")[DIM_NAMES]

and remove the 200-pool subset filter. Everything downstream runs unchanged
on all 42,693 pairwise rows / 1,069 annotators / 5 countries.

Run on a CUDA box (~16-20 GB VRAM). MPS works on Apple Silicon with 32+ GB
unified memory but will be slower. CPU is impractical for 8B params.

Typical wall time: ~5-15 min on A100/H100, ~30-60 min on smaller GPUs.
"""
from __future__ import annotations
import argparse, json, tarfile, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def render(prompt: str, response: str) -> str:
    """Match the embedding template the directions were fit on (README §4)."""
    return f"Prompt: {prompt}\nResponse: {response}"


def load_directions(tarball: Path):
    """Pull V (15, 4096), μ (4096,), and dimension names from the tarball."""
    with tarfile.open(tarball) as t:
        npz = np.load(t.extractfile("method_directions/outputs/coalign_50/directions.npz"))
        V  = npz["directions_raw"].astype(np.float64)   # (15, 4096) — primary; README §5 says use this
        mu = npz["mean_embedding"].astype(np.float64)   # (4096,)
        dims = json.loads(t.extractfile("method_llm_gen/outputs/coalign_50/dimensions.json").read())
    names = [d["name"] for d in dims["dimensions"]]
    return V, mu, names


def autodetect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairwise", type=Path,
                    default=Path("pairwise_comparisons_ge10.parquet"),
                    help="path to pairwise parquet")
    ap.add_argument("--tarball", type=Path,
                    default=Path("coalign_50_dimensions.tar.gz"),
                    help="path to coalign_50_dimensions.tar.gz")
    ap.add_argument("--output", type=Path,
                    default=Path("features_full.parquet"),
                    help="output (response_text × 15) feature parquet")
    ap.add_argument("--raw-embeddings", type=Path,
                    default=Path("embeddings_qwen3_4096.npy"),
                    help="also save raw φ (N, 4096) for re-projection later")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="embedding batch size (lower if OOM; 8B model is heavy)")
    ap.add_argument("--device", type=str, default=None,
                    help="cuda / mps / cpu (default: autodetect)")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="for smoke-testing the script: only embed first N pairs")
    args = ap.parse_args()

    device = args.device or autodetect_device()
    print(f"device:        {device}")
    if device == "cpu":
        print("WARNING: 8B-param embedding on CPU will take many hours. Consider a GPU.")

    # ── 1. unique (prompt, response) pairs from pairwise data ────────────
    pw = pd.read_parquet(args.pairwise)
    left  = pw[["prompt_text", "preferred_text"]].rename(columns={"preferred_text": "response_text"})
    right = pw[["prompt_text", "other_text"]    ].rename(columns={"other_text":     "response_text"})
    pairs = pd.concat([left, right]).drop_duplicates(subset="response_text").reset_index(drop=True)
    if args.max_rows is not None:
        pairs = pairs.head(args.max_rows).copy()
    n_resp = len(pairs)
    n_prompt = pairs.prompt_text.nunique()

    # Sanity: response_text → prompt_text should be 1:1 (we verified this earlier in the project)
    assert pairs.response_text.is_unique, "response_text key collision — switch to (prompt, response) tuple key"
    rendered = [render(p, r) for p, r in zip(pairs.prompt_text, pairs.response_text)]
    print(f"to embed:      {n_resp:,} unique responses across {n_prompt:,} prompts")

    # ── 2. load directions V, μ ──────────────────────────────────────────
    V, mu, dim_names = load_directions(args.tarball)
    print(f"V shape:       {V.shape}")
    print(f"dimensions:    {dim_names}")
    assert V.shape == (15, 4096), "unexpected V shape"

    # ── 3. embed via sentence-transformers ───────────────────────────────
    # Per README §3: pass normalize_embeddings=False — directions were fit on unnormalized.
    from sentence_transformers import SentenceTransformer
    print(f"loading Qwen/Qwen3-Embedding-8B (~16-20 GB) ...")
    t0 = time.time()
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-8B",
        model_kwargs={"torch_dtype": dtype},
        device=device,
    )
    print(f"  loaded in {time.time()-t0:.1f}s")

    print(f"embedding {n_resp:,} texts (batch_size={args.batch_size}) ...")
    t0 = time.time()
    phi = model.encode(
        rendered,
        batch_size=args.batch_size,
        normalize_embeddings=False,           # <- README §3 emphasis
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float64)                       # promote to fp64 for stable projection
    elapsed = time.time() - t0
    print(f"  done in {elapsed/60:.1f} min  ({elapsed/n_resp*1000:.0f} ms/text)")
    print(f"  phi shape: {phi.shape}, ||phi||  range [{np.linalg.norm(phi,axis=1).min():.2f}, {np.linalg.norm(phi,axis=1).max():.2f}]")
    np.save(args.raw_embeddings, phi)
    print(f"  raw embeddings → {args.raw_embeddings}  ({phi.nbytes/1e6:.1f} MB)")

    # ── 4. project: scores = V @ (φ − μ) ────────────────────────────────
    scores = (phi - mu) @ V.T                  # (N, 15)
    out = pd.DataFrame(scores, columns=dim_names, index=pairs.response_text)
    out.index.name = "response_text"
    out.to_parquet(args.output)
    print(f"\nwrote {args.output}  ({len(out):,} rows × {len(dim_names)} dims)")
    print("\nper-dim score range (should be roughly [-1, +1]; OOD inputs may exceed):")
    print(out.describe().loc[["min", "max", "mean", "std"]].round(3).T.to_string())

    # ── 5. validation against the 200 pre-scored options in bt_scores.csv ──
    with tarfile.open(args.tarball) as t:
        bt = pd.read_csv(t.extractfile("method_llm_gen/outputs/coalign_50/bt_scores.csv"))
    bt_wide = bt.pivot(index="display_text", columns="dimension_name", values="bt_score")[dim_names]
    overlap = out.index.intersection(bt_wide.index)
    if len(overlap) >= 10:
        a = out.loc[overlap].to_numpy()
        b = bt_wide.loc[overlap].to_numpy()
        per_dim_corr = [np.corrcoef(a[:, j], b[:, j])[0, 1] for j in range(len(dim_names))]
        per_dim_mae  = np.abs(a - b).mean(axis=0)
        print(f"\nvalidation on {len(overlap)} options that overlap bt_scores.csv:")
        print(f"  Pearson r per dim — mean = {np.mean(per_dim_corr):.3f}  "
              f"(min {min(per_dim_corr):.2f}, max {max(per_dim_corr):.2f})")
        print(f"  per-dim MAE        — mean = {per_dim_mae.mean():.3f}")
        print("  (high correlation expected; magnitudes won't match exactly because")
        print("   ridge regression isn't an isometry, but ordering should be tight)")
    else:
        print(f"\nskipping validation: only {len(overlap)} of {len(bt_wide)} bt_scores texts in our pool")


if __name__ == "__main__":
    main()
