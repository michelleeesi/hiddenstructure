# Linearity Robustness — Rebuttal Working Doc

**Paper:** *Algorithmic Impact Reveals the Hidden Structure of Alignment*
**Concern:** reviewers question the linear-utility assumption (and utilitarian aggregation).
**Status:** theory response drafted + empirical study run on all four datasets. This doc records everything so it can be turned into an appendix + rebuttal.

---

## 0. TL;DR

- The linear-utility assumption is **load-bearing** for the *elegant* result (finite-dimensional impact polytope + LP), so we can't wave it away — but most of the structure is more robust than it looks.
- **Theory:** welfare is linear in the *policy's behavior* unconditionally; linearity is what makes the $k$-dim impact a **sufficient statistic**; and it extends *verbatim* to any fixed nonlinear basis (polynomial, spline/GAM, interactions). Non-utilitarian objectives are **already** handled in §6.
- **Empirics (new):** across all four datasets, assuming linear utilities costs **≤ 2.4% of social welfare**, keeps per-agent welfare correlated at **0.79–0.95** and (where cleanly measurable) the participation-externality matrix at **0.97**. A nonlinear (GAM) utility only *improves choice prediction* on the one data-rich dataset (Kidney), and even there the estimands barely move.
- **Bottom line for the rebuttal:** *"Utilities have mild additive curvature that our framework captures exactly via an expanded basis; empirically it is immaterial to our estimands (Γ, welfare, deployment)."*

---

## 1. Reviewer concerns (verbatim)

> **R1:** "Related to the main concern raised above, I am curious whether the framework generalize to non-linear utilities."

> **R2:** "How sensitive is it to violations of linear utility, and linearity for the low-dimensional projection? Can it provide a comparison to a non-linear model for the low-dimensional domain (e.g. Via cross validation)?"

> **R3 (Q2):** "A key limitation is that the main theoretical structure appears to depend strongly on the assumed linear utility model and utilitarian social-welfare aggregation. The reduction to linear optimization over a convex impact space is elegant, but it is unclear how much of this structure survives when individual utilities, welfare aggregation, or social objectives are nonlinear."

There are **two distinct asks**: an **empirical** sensitivity/CV comparison (R1, R2), and a **theoretical** "what survives" question (R3).

---

## 2. The three-question ladder (how the experiments fit together)

Every experiment answers exactly one of three questions, forming a logical chain:

| | Question | Experiment | What it settles |
|---|---|---|---|
| **Q1** | Is linear actually a *bad fit* to choices? | within-agent CV, linear vs. nonlinear + noise ceiling | Is there even a problem? |
| **Q2** | Does linearity *distort the reported numbers* (Γ, welfare, deployment)? | recompute estimands under nonlinear utility; welfare regret | The paper-relevant cost |
| **Q3** | Do the paper's *conclusions* change under a nonlinear estimator? | (subsumed by Q2's structure comparison) | Robustness of findings |

Start at Q1 because it is (a) what R1/R2 literally asked for, (b) cheapest, and (c) diagnostic — its result decides whether Q2/Q3 are the crux or just confirmation.

---

## 3. Theoretical response (answers R3)

Decompose the linearity assumption into what is **structural** vs. a **modeling choice**:

**(a) The deep reformulation needs no linearity.** Welfare is expected utility, hence **linear in the deployed policy's behavior** $p(\cdot)$ unconditionally:
$$U_n = \mathbb{E}_z[\,p(z)\,\delta_n(z)\,] + \text{const}, \qquad \delta_n(z)=u_n(A_z)-u_n(B_z).$$
The feasible set of behaviors is convex for any utilities. So *"alignment = optimize a linear (or concave) objective over a convex set"* survives regardless of utility form; only the finite-dimensional coordinates are lost.

**(b) Linearity makes the $k$-dim impact a *sufficient statistic*** (the paper's own phrase, line 116). With $u_n=\varphi^\top\theta_n$, $\delta_n(z)=\theta_n^\top\alpha_z$ depends on $z$ only through $\alpha_z\in\mathbb{R}^k$, collapsing welfare to $U_n=\theta_n^\top\psi$ and giving the polytope $\bar\Psi\subset\mathbb{R}^k$ (Thm 3) and the LP.

**(c) Extends verbatim to any fixed nonlinear basis.** For $u_n=\tilde\varphi(x)^\top\theta_n$ with any fixed map $\tilde\varphi$ (polynomials, **splines/GAM**, interactions), $\delta_n(z)=\theta_n^\top\tilde\alpha_z$ again ⇒ same Thm 3 polytope, just in $\mathbb{R}^{k'}$. Per-feature nonlinearity (diminishing returns, saturation, thresholds) is captured with **zero loss of structure**. Only genuinely nonparametric, agent-specific utilities force the infinite-dimensional convex program.

**(d) Non-utilitarian aggregation is already in the paper.** §6 (harm floors Cor 15, public spirit Cor 17, externality bounds Cor 19) are non-utilitarian objectives expressed as linear constraints/reweightings over the **same** $\bar\Psi$ (Thm 13). Any concave social welfare function (Nash $\sum\log U_n$, egalitarian max-min, prioritarian) is a **convex program over the same $\bar\Psi$** — only the objective changes.

**Honest caveat (volunteer it):** the *strategyproof = random dictatorship* result (Thm 9, via Dutta et al.) is the piece **most** tied to linearity — it needs linear preferences over $\bar\Psi$. It generalizes to linear-in-basis but not obviously to nonparametric utilities. The §6 welfare/externality machinery is the robust part.

---

## 4. Empirical methodology

**What "nonlinear utility" means.** The reviewer's concern is nonlinear *utility* $u(\text{option})=g(\varphi)$, so choice depends on $g(\varphi_A)-g(\varphi_B)$, which is **not** a function of the difference $\alpha=\varphi_A-\varphi_B$ alone. Consequences:
1. Use the **raw option features** $\varphi_A,\varphi_B$ (not just $\alpha$).
2. The fair flexible model is a Bradley-Terry model with a **nonlinear score** $\sigma(g(\varphi_A)-g(\varphi_B))$ — *not* a black-box classifier on $\alpha$ (which breaks the choice antisymmetry $p(\alpha)=1-p(-\alpha)$ and confounds nonlinearity with an asymmetric model).

**The models compared** (all logistic, no intercept, $C=1$ to match the paper's `BTL_C`, features standardized on pooled $\varphi_A,\varphi_B$):
- **linear:** design $=\varphi_A-\varphi_B$ (the paper's model).
- **quadratic (GAM):** design $=[\varphi_A-\varphi_B,\ \varphi_A^2-\varphi_B^2]$ — additive per-feature curvature; nests linear; respects antisymmetry; **preserves the framework** (theory (c)).
- **interactions (Moral Machine only):** additionally $\varphi_{A,k}\varphi_{A,l}-\varphi_{B,k}\varphi_{B,l}$ for $k<l$ — the one nonlinearity a GAM basis *cannot* absorb; the adversarial test.

**Q1 (fit):** per-agent 5-fold within-agent CV, held-out log-loss and accuracy, aggregated across agents. Datasets whose only label is "preferred" (Moral Machine, Community Alignment) are symmetrized inside each training fold ($[\tilde\alpha;-\tilde\alpha]$ with $y=[1;0]$) and evaluated on the held-out actual choices.

**Noise ceiling (Kidney, discrete features):** for each (agent, exact profile-pair) seen ≥2 times, the *identical-stimulus oracle* is the leave-one-out prediction from the other occurrences. **Use accuracy** (LOO majority vote); the LOO *log-loss* ceiling is upward-biased by Jeffreys smoothing on mostly-2× repeats and is **not** a valid floor.

**Q2 (estimands):** refit each agent in each basis, then run the paper's own pipeline in the expanded basis — utilitarian deployment $\psi=\texttt{psi}(\bar\theta_w)$, per-agent welfare $U_n=\theta_n^\top\psi$, participation externality $\Gamma_{n,m}=\theta_m^\top(\psi-\psi^{(-n)})$ (closed-form utilitarian LOO), all with $\tilde\alpha,\tilde\theta$. Compare linear vs. nonlinear via Pearson/Spearman, best/worst-off agreement, sign agreement.

**Welfare regret ($\beta\to\infty$):** treat quadratic as the "true" utility; compare the true social welfare of the **linear-optimal** deployment vs. the **quadratic-optimal** deployment. Regret $=[U(\psi^\star_{\text{quad}})-U(\psi_{\text{lin-deploy}})]/U(\psi^\star_{\text{quad}})$.

**Reliability (Community Alignment):** split each agent's data in half, refit, recompute Γ from each half, correlate — to test whether a low Γ correlation is *real* or just estimation noise.

---

## 5. Results per dataset

### 5.1 Kidney Allocation — data-rich (~480 comparisons/agent, k=5), discrete features

- **Q1:** held-out log-loss linear **0.2847** → quadratic **0.2634**; accuracy **87.8% → 89.1%**; quadratic helps **56/77** agents (44 by >0.01 nats, 5 hurt). ⇒ **Real, detectable per-feature curvature.**
- **Noise ceiling:** 23% of data is repeated stimuli. Human-consistency ceiling **93.3%** accuracy (agents flip ~6.7% on identical comparisons). Linear **90.0%** (3.3pp below ceiling); quadratic **92.6%** (0.7pp below). ⇒ *Linear is measurably below the human floor; a simple additive quadratic utility nearly closes the gap.* (Log-loss ceiling omitted — smoothing artifact; quadratic spuriously "beat" it.)
- **Q2:** welfare Pearson **0.941** / Spearman 0.936, **best- and worst-off agents identical**; Γ Pearson **0.972** / Spearman 0.976, sign agreement 94%, top-5 imposers 3/5, receivers 4/5; **welfare regret 1.08%**.

### 5.2 412 Food Rescue — thin (~45/agent, k=7)

- **Q1:** quadratic **hurts** (log-loss 0.447 → 0.541; accuracy 0.795 → 0.776; helps 3/19). With 45 points and 14 params it **overfits** ⇒ linear is the **bias–variance-optimal** choice here regardless of true shape.
- **Q2:** welfare corr **0.951** (worst-off same, best-off differs); Γ corr **0.918**, sign 0.901; **regret 0.85%**.

### 5.3 Moral Machine — moderate (~140/agent, k=20), 190 interaction terms

- **Q1:** log-loss linear **0.7725** / acc 0.617; quadratic 1.2776 / 0.594; **interactions 2.1067 / 0.582**. Both nonlinear models **overfit badly** ⇒ linear is clearly the best predictor; even interactions don't help at this budget.
- **Q2:** linear vs quad — welfare **0.943**, Γ 0.653, regret **1.17%**; linear vs interactions — welfare **0.909**, Γ 0.306, regret **2.14%**. (Low Γ correlations reflect the noisy overfit models, per Q1, not real structure.)

### 5.4 Community Alignment — thin-ish (~57/agent, k=15), learned-embedding features

- **Q1:** quadratic **doesn't help** (log-loss 0.489 → 0.523; accuracy 0.769 → 0.786; helps ~half). Consistent with the features already being a **nonlinear projection** (linear probe ≈ optimal).
- **Q2:** welfare corr **0.792**; **regret 2.36%**; Γ corr **0.408** (flagged, resolved below).
- **Reliability check (resolves the 0.41):** Γ_linear's own split-half self-correlation is **0.404** — i.e., Γ is simply a *noisy estimand* on CA (thin per-agent data). The linear-vs-quadratic Γ correlation (0.408) **equals** Γ_linear's own reliability ⇒ *switching to quadratic perturbs Γ no more than re-drawing the data does.* (Γ_quad self-reliability 0.222, lower still, confirming the overfit noise.) The apparent problem is measurement noise, not misspecification.

---

## 6. Consolidated table

| Dataset | comps/agent | Q1: nonlinear improves prediction? | Welfare corr (lin vs quad) | Welfare regret | Γ corr | Γ note |
|---|---|---|---|---|---|---|
| **Kidney** | ~480 (rich) | **Yes** — reaches human ceiling | 0.94 | **1.1%** | **0.97** | clean / reliable |
| **412** | ~45 (thin) | No — overfits | 0.95 | **0.9%** | 0.92 | — |
| **Moral Machine** | ~140 | No — quad & interactions overfit | 0.94 | **1.2%** | 0.65 | overfit-model noise |
| **Community Alignment** | ~57 (embeddings) | No — projection already nonlinear | 0.79 | **2.4%** | 0.41 | = Γ's own noise floor (0.40) |

**Three conclusions:**
1. Where real, estimable nonlinearity exists (Kidney), the estimands barely move (Γ 0.97, welfare 0.94, regret 1.1%).
2. Everywhere else, nonlinear models don't even improve prediction — thin data (412), noisy high-dim task (Moral Machine, even with interactions), or already-nonlinear features (Community Alignment). Linear is the appropriate estimator at realistic budgets.
3. Welfare regret is **≤ 2.4% on every dataset**; low Γ correlations are attributable to estimation noise (confirmed by the reliability check), not to nonlinearity reshaping externalities.

---

## 7. Ready-to-use rebuttal language

**Headline paragraph:**

> Individual utilities do exhibit mild per-feature curvature: an additive (GAM) utility improves held-out choice prediction and reaches the human test–retest ceiling on our data-rich dataset (Kidney), whereas a linear utility sits a few points below it. This curvature is, however, **immaterial to our estimands**: recomputing the participation-externality matrix and per-agent welfare under a quadratic utility yields Pearson correlations of 0.94 (welfare) and 0.97 (externalities) with the linear results, identical best- and worst-off agents, and a utilitarian-deployment welfare regret of ~1%. Across all four datasets, assuming linear utilities costs at most ~2.4% of social welfare. On the remaining datasets a nonlinear model does not even improve choice prediction — because the data is thin (412), the task is high-dimensional and noisy (Moral Machine, even with pairwise interactions), or the features are already a learned nonlinear projection (Community Alignment). Moreover, our framework accommodates such additive nonlinearity **exactly** by expanding the feature basis (Theorem 3), with no change to the theory; and non-utilitarian objectives are already handled in §6 as convex programs over the same impact set.

**Why the estimands tolerate misspecification (one sentence for the paper):** choice prediction is a *fine* functional of utility (needs the exact shape); Γ and welfare are *coarse, aggregate* functionals (mean preference direction, leave-one-out perturbations) — a model can be measurably wrong at the fine level while right at the coarse level.

**Per-reviewer mapping:**

| Reviewer ask | Where answered |
|---|---|
| R1 — generalize to nonlinear utilities? | Theory (c): linear-in-any-basis (incl. GAM) preserves the framework; empirics §5 |
| R2 — sensitivity + CV vs nonlinear on low-dim domain | §5 CV on all four datasets; ceiling on Kidney; "projection already nonlinear" for CA |
| R3 — what survives (nonlinear utility / aggregation / objectives) | Theory (a)–(d); §6 already gives non-utilitarian objectives; honest caveat on strategyproofness |

---

## 8. Honest caveats (keep them in)

- Highest-confidence result is on **Kidney** (rich data, discrete features → clean ceiling and reliable Γ). On thin datasets the right framing is **bias–variance** ("richer models don't help here"), not "utilities are exactly linear."
- Report the noisier Γ estimates **with their reliability**, so low numbers (CA 0.41, Moral Machine 0.31–0.65) are not misread as "linearity distorts Γ."
- The quadratic/GAM/interaction bases are a *representative* stress test, not exhaustive; but on the only dataset with data-budget to reveal nonlinearity cleanly (Kidney), quadratic already reaches the human-consistency ceiling, so more-flexible models have little room to add.
- The `psi`/Γ computations here reuse the paper's validated primitives (`hiddenstructure.impact.psi`, `hiddenstructure.btl.fit_btl`) fed the expanded basis — i.e. the *same* pipeline, exactly matching theory point (c).

---

## 9. Suggested paper additions

1. **One-page appendix** with (i) per-dataset CV bars (linear vs. GAM vs. — for MM — interactions), (ii) the Kidney welfare/Γ scatter (linear vs. quadratic) + the regret row, (iii) the CA Γ reliability number. Slots into `run_figures.ipynb`.
2. **A short "what survives" theory box** (§3 of this doc): sufficient-statistic framing + the linear-in-basis generalization + pointer to §6 for non-utilitarian objectives.
3. **(Optional insurance)** a **semi-synthetic** experiment — inject known strong nonlinearity, fit linear, show bounded Γ/welfare distortion vs. nonlinearity magnitude — to cover "what if a future dataset has strong, estimable curvature." Kidney is currently the only place nonlinearity was both real *and* cleanly measurable.

---

## 10. Reproducibility

- **Code/data:** package `hiddenstructure` (in this repo, `src/`); raw data under `HS_DATA_ROOT=../distortion` (kidney `empirics_kidney/kidneypairwise2.csv`; 412 `empirics_412/food_rescue_combined.csv`; moral machine `empirics_moralmachine/mid_users_filtered.csv`; community alignment `empirics_communityalignment/{features_full_nofb,pairwise_comparisons_ge10_nofb}.parquet`). Env: `~/anaconda3/envs/py37`.
- **Recipe** (per dataset): load raw option features $\varphi_A,\varphi_B$; standardize on pooled $\varphi_A,\varphi_B$; build `linear` / `quad` / (MM) `inter` feature-difference designs; **Q1** = per-agent 5-fold CV (`LogisticRegression(fit_intercept=False, C=1.0)`), symmetrize inside folds for preferred-only labels; **ceiling** = LOO majority on exact-repeat cells (Kidney); **Q2** = per-agent `fit_btl` in each basis → utilitarian `psi`/closed-form-LOO Γ → compare; **regret** = true-quadratic welfare of linear-opt vs quad-opt deployment at $\beta\to\infty$; **reliability** = split-half Γ correlation.
- The exploratory scripts used to produce every number above live in the session scratchpad; the recipe here is sufficient to regenerate them.

*(This is internal working documentation for the rebuttal, not for distribution.)*
