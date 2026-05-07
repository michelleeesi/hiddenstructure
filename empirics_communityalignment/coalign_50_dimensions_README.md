# CoAlign Dimension Embeddings — Walkthrough

This bundle contains 15 interpretable preference dimensions discovered from the
[Community Alignment Dataset](https://github.com/facebookresearch/community-alignment-dataset)
(CoAlign), each represented as a direction vector in
`Qwen/Qwen3-Embedding-8B`'s 4096-dimensional embedding space.

Companion archive: **`coalign_50_dimensions.tar.gz`** (in this folder).

---

## 1. What's in the tarball

```
method_directions/outputs/coalign_50/
  directions.npz            # the K=15 direction vectors (this is the main file)
  summary.md                # held-out R², per-dimension diagnostics, top/bottom options

method_llm_gen/outputs/coalign_50/
  dimensions.json           # human-readable names, low/high pole descriptions
  bt_scores.csv             # per-option latent scores per dimension (200 options × 15 dims)
```

Pipeline that produced these: 50 choice-set situations × 4 candidate responses
(English-only, balanced subset of CoAlign), embedded with
`Qwen/Qwen3-Embedding-8B`, scored on 15 LLM-generated dimensions via
Bradley-Terry, then mapped to direction vectors via ridge regression.

---

## 2. Quick start

```bash
tar xzf coalign_50_dimensions.tar.gz
pip install numpy sentence-transformers torch
python -c "
import numpy as np
d = np.load('method_directions/outputs/coalign_50/directions.npz')
print('Keys:', list(d.keys()))
print('V shape:', d['directions_raw'].shape)   # (15, 4096)
"
```

Then jump to **§5** for the projection example.

---

## 3. The embedding model

The directions live in the embedding space of **`Qwen/Qwen3-Embedding-8B`**
([HF page](https://huggingface.co/Qwen/Qwen3-Embedding-8B)). To project new
text onto the dimensions, you must embed it with this exact model.

| Property              | Value                          |
|-----------------------|--------------------------------|
| Architecture          | Qwen3 decoder, 8.2B params     |
| Embedding dim         | 4096                           |
| Pooling               | last-token                     |
| Disk size             | ~16 GB (bf16/fp16)             |
| GPU memory (fp16)     | ~16-20 GB                      |
| HF org / repo         | `Qwen/Qwen3-Embedding-8B`      |

### Download

The model auto-downloads on first use via the HF cache. To pre-fetch:

```bash
# Via huggingface-cli (recommended for big models)
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-Embedding-8B
```

Cache location defaults to `~/.cache/huggingface/hub/`. Override with
`export HF_HOME=/path/to/big/disk` before running.

### Embedding via sentence-transformers (simplest)

```python
import torch
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-8B",
    model_kwargs={"torch_dtype": torch.float16},  # halves GPU memory
    device="cuda",
)

emb = model.encode("Hello, world.", normalize_embeddings=False)
print(emb.shape)   # (4096,)
```

**Important:** pass `normalize_embeddings=False`. The directions were fit to
unnormalized embeddings; L2-normalizing here will distort the projections.

### Embedding via vLLM (faster for batches)

If you need to embed many options:

```bash
vllm serve Qwen/Qwen3-Embedding-8B --task embed --port 8000
```

Then call the OpenAI-compatible endpoint. See `embed/embedder/embed_csv.py`
in the parent repo for the exact request format.

---

## 4. The embedding template

Options in the source dataset were embedded with this template
(`datasets/coalign_prompt.txt`):

```
Prompt: {prompt}
Response: {description}
```

If you want to project a new (prompt, response) pair onto the dimensions,
render it through the same template before embedding. For free-form text
that has no clear prompt, you can embed plain text — but expect somewhat
weaker alignment with the dimensions, since they were learned on
prompt+response pairs.

---

## 5. Loading and projecting

```python
import numpy as np
from sentence_transformers import SentenceTransformer
import torch, json

# --- load directions ---
data = np.load("method_directions/outputs/coalign_50/directions.npz")
V        = data["directions_raw"]      # (15, 4096) — primary, USE THIS
V_ortho  = data["directions_ortho"]    # (15, 4096) — QR-orthogonalized variant
mu       = data["mean_embedding"]      # (4096,)    — pool mean (used for centering)
alphas   = data["best_alphas"]         # (15,)      — ridge regularizers used

# --- load names ---
dims = json.load(open("method_llm_gen/outputs/coalign_50/dimensions.json"))["dimensions"]
names = [d["name"] for d in dims]      # ['Conciseness', 'Structure', ...]

# --- embed new text ---
model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-8B",
    model_kwargs={"torch_dtype": torch.float16},
    device="cuda",
)
text = "Prompt: How do I plan a trip to Tokyo?\nResponse: Visit Shinjuku, eat ramen, see Senso-ji."
phi = model.encode(text, normalize_embeddings=False).astype(np.float64)  # (4096,)

# --- project onto dimensions ---
phi_c   = phi - mu                # center
scores  = V @ phi_c               # (15,) — one score per dimension

for name, s in zip(names, scores):
    print(f"  {name:25s} {s:+.3f}")
```

### Higher score = which pole?

Each dimension is **bipolar** (low_pole vs high_pole). The direction vector
points toward the high pole. So:

- **Positive score** → option resembles the **`high_pole`** end
- **Negative score** → option resembles the **`low_pole`** end

For dimension 1 (`Conciseness`):
- `low_pole.label = "Wordy"`
- `high_pole.label = "Concise"`
- A positive score means *more concise*.

Pole descriptions are in `dimensions.json` under each dimension's
`low_pole` and `high_pole` keys.

### Score scale

BT scores were trained on a roughly [-1, +1] scale (Bradley-Terry latents
normalized per dimension). Predicted scores from `V @ (phi - mu)` will be
on a similar scale for in-distribution text but can exceed those bounds
for out-of-distribution inputs.

---

## 6. Comparing two responses

The natural use case: which response is "more X" than the other?

```python
phi_a = model.encode("Prompt: ...\nResponse: A", normalize_embeddings=False).astype(np.float64)
phi_b = model.encode("Prompt: ...\nResponse: B", normalize_embeddings=False).astype(np.float64)

delta = V @ (phi_a - phi_b)        # (15,) — A vs B per dimension
                                    # mu cancels in the difference
for name, d in zip(names, delta):
    if abs(d) > 0.1:
        print(f"  {name:25s} {d:+.3f}  ({'A more' if d > 0 else 'B more'})")
```

Note: when comparing two embeddings, the `mu` centering cancels out, so
you can skip it for pairwise contrasts.

---

## 7. Quality and caveats

Held-out evaluation (mean across 15 dimensions, leave-one-out CV):

| Metric             | Value  |
|--------------------|--------|
| R² (held-out)      | 0.74   |
| Pearson r          | 0.87   |
| Spearman ρ         | 0.84   |

Per-dimension R² ranges from 0.58 (`Structure`) to 0.89 (`Efficiency`).
Full table in `method_directions/outputs/coalign_50/summary.md`.

**12 of 15 dimensions are flagged with low contrastive-vs-ridge cosine
agreement (< 0.5).** This means the ridge-fit direction and a simple
contrastive estimate of the same dimension don't fully agree. Likely
reasons:

1. The K=15 dimensions are non-orthogonal (max inter-dim correlation
   ~0.5–0.8 typical for this pipeline) — ridge "borrows" signal across
   correlated dimensions.
2. With only 200 options, some dimensions have weak coverage.

**Practical implication:** trust the *signed direction* and the relative
ordering, but treat the absolute magnitude with some skepticism for
flagged dimensions. The three "reliable" ones (`Community Focus`,
`Practicality`, `Authenticity`) had cosine > 0.5.

For higher reliability, run the pipeline with more choice sets (e.g.
N=200 instead of 50) — see `run_coalign_50.sh` and bump `num_sets` in
`configs/coalign_50.yaml`.

---

## 8. Reference: the 15 dimensions

| ID | Name                | Low pole       | High pole       | R² (held-out) |
|----|---------------------|----------------|-----------------|---------------|
| 1  | Conciseness         | Wordy          | Concise         | 0.75 |
| 2  | Structure           | Unstructured   | Structured      | 0.58 |
| 3  | Actionability       | Abstract       | Actionable      | 0.80 |
| 4  | Clarity             | Opaque         | Clear           | 0.61 |
| 5  | Emotional Resonance | Detached       | Emotional       | 0.81 |
| 6  | Depth               | Surface        | Deep            | 0.75 |
| 7  | Descriptiveness     | Sparse         | Descriptive     | 0.66 |
| 8  | Sustainability      | (none)         | Sustainable     | 0.71 |
| 9  | Community Focus     | Individual     | Community       | 0.66 |
| 10 | Historical Context  | Ahistorical    | Historical      | 0.67 |
| 11 | Practicality        | Theoretical    | Practical       | 0.88 |
| 12 | Efficiency          | Verbose        | Efficient       | 0.89 |
| 13 | Creativity          | Conventional   | Creative        | 0.77 |
| 14 | Authenticity        | Generic        | Authentic       | 0.85 |
| 15 | Formality           | Casual         | Formal          | 0.71 |

Pole *labels* above are paraphrased; full descriptions and a "typical
person" archetype for each pole are in `dimensions.json`.

---

## 9. File-by-file API

### `directions.npz`

```python
np.load("directions.npz")  # NpzFile with:
  directions_raw       (15, 4096) float64  # primary V; use this
  directions_ortho     (15, 4096) float64  # QR-orthogonalized variant
  mean_embedding       (4096,)    float64  # mu, the option-pool mean
  best_alphas          (15,)      float64  # ridge regularizer per dim
```

### `dimensions.json`

```python
{
  "domain": "...",
  "choice_context": "...",
  "dimensions": [
    {
      "id": 1,
      "name": "Conciseness",
      "low_pole":  {"label": "Wordy",   "description": "...", "typical_person": "..."},
      "high_pole": {"label": "Concise", "description": "...", "typical_person": "..."},
      "example_contrast": {"low_option": "...", "high_option": "..."},
      "scoring_guidance": "...",
      ...
    },
    ...  # 15 total
  ]
}
```

### `bt_scores.csv`

Columns: `dimension_id, dimension_name, option_id, display_text, bt_score`.
One row per (option, dimension), so 200 × 15 = 3000 rows. Scores are
normalized to roughly [-1, +1] per dimension. These are the targets that
were ridge-regressed against the option embeddings to produce
`directions_raw`.

### `summary.md`

Full per-dimension diagnostics: R², Pearson, Spearman, ridge-vs-contrastive
cosine agreement, and the top-3 / bottom-3 options per dimension (very
useful for sanity-checking what each dimension is actually capturing).

---

## 10. Citation / attribution

The dimensions were learned on a 50-set subset of the
[Community Alignment Dataset](https://github.com/facebookresearch/community-alignment-dataset)
(facebookresearch). Embedding model: `Qwen/Qwen3-Embedding-8B`. Method:
LLM-discovered preference dimensions + Bradley-Terry scoring + ridge
regression to embedding directions.
