# Bounded-Harm vs. Max-Min — Rebuttal Working Doc

**Paper:** *Algorithmic Impact Reveals the Hidden Structure of Alignment*
**Concern:** a reviewer asks whether the harm-bounded approach is *Pareto-improving* over existing methods (e.g. Max-Min RLHF), on all data or some subset.
**Status:** conceptual answer worked out; suggested empirical add-on noted.

---

## 0. TL;DR

- Answering the question **literally** ("does bounded-harm Pareto-dominate max-min?") is a trap: both land on the same Pareto frontier, so by definition neither dominates the other.
- The **strong** answer: **max-min is a special case of the bounded-harm family** (the tight-floor endpoint). Sweeping the harm floor traces the whole efficient frontier from utilitarian to egalitarian; setting the floor at the max-min value yields the **leximin** point, which *weakly Pareto-dominates* a plain max-min solution by not wasting the surplus above the guaranteed minimum.
- **Genuine** Pareto-domination holds over the *inefficient* baseline — **RLHF / anonymous pooling** — which prior work shows can sit *inside* the feasible set. That is the clean, demonstrable "yes."
- **Honest nuance:** within a fixed population, bounded-harm vs. utilitarian is *redistribution*, not a Pareto win — protecting the binding agent imposes negative externalities on others (this is exactly what the Γ analysis in Fig 3 shows).

---

## 1. Reviewer concern (verbatim)

> "Are there some data sets or configurations in which your harm-bounded approach is Pareto improving over existing methods like, say max-min approach on all data, or on some subset of data?"

---

## 2. What "the frontier" is (plain terms)

We choose **one** deployed model. Every possible model gives **each agent some welfare**, so a model produces a list of numbers — one welfare per agent, $(U_1,\dots,U_N)$. Think of each model as a dot in "welfare space."

The **Pareto frontier** is the outer edge of the set of achievable welfare-lists: the models where *you can't make anyone better off without making someone else worse off*. Anything strictly *inside* is wasteful (you could hand someone free welfare at no one's cost). A good mechanism should always pick a point **on the frontier**.

Different mechanisms just pick **different points on that same frontier** (in the paper, all of them optimize over the same convex impact set $\bar\Psi$; its boundary *is* the frontier):
- **Utilitarian** — maximizes the **sum** of welfares. Good for the total; a minority can end up badly off.
- **Max-min (egalitarian)** — maximizes the **minimum** welfare; protects the worst-off, possibly sacrificing total.

These are two different spots on the same curve. Neither Pareto-dominates the other — they are different trade-offs.

---

## 3. Bounded-harm is a *dial* between those two points

Bounded-harm says: *"maximize total welfare, subject to no agent falling below a floor $b$."* The floor $b$ is a dial:

- **Loose floor** (very low $b$, constraint never bites) → you just maximize the total → the **utilitarian** point.
- **Tighten the floor** (raise $b$) → you're *forced* to lift up the agents below $b$, costing some total → you slide along the frontier away from utilitarian, toward protecting everyone.
- **Tightest feasible floor** (the highest value you can guarantee *everybody* at once) → you raise the worst-off as high as possible → the **max-min** point.

So sweeping the dial from loose to tight walks the solution **along the frontier from the utilitarian end to the egalitarian (max-min) end.** That is all "a parameterized path along the frontier" means: $b$ is the parameter, each $b$ gives one frontier point, and together they trace the whole curve. **Max-min isn't a rival — it's where the dial points when turned all the way up.**

---

## 4. Why plain max-min "wastes surplus," and how bounded-harm fixes it

Concrete 3-agent example. Suppose the **best minimum welfare** you can guarantee everyone is $0.5$. Several models may all achieve minimum $=0.5$:

- **Model P:** welfares $(0.5,\ 0.5,\ 0.5)$ — minimum $0.5$. ✓
- **Model Q:** welfares $(0.5,\ 0.9,\ 0.9)$ — minimum $0.5$. ✓

Plain **max-min only looks at the minimum**, so it treats P and Q as equally good and might return **P**. But **Q Pareto-dominates P** (agents 2 and 3 happier, nobody worse). Max-min left free welfare on the table: once it hit minimum $=0.5$, it stopped caring about anyone above the floor.

Now run **bounded-harm with the floor $b = 0.5$**: maximize *total* welfare subject to everyone $\ge 0.5$. This forces the minimum to $0.5$ (still fully protects the worst-off) **and** maximizes the total among such models → it picks **Q, not P**. That refined choice is **leximin** ("get the best minimum, then don't waste the surplus above it").

So: **bounded-harm at the max-min floor = leximin**, and leximin **weakly Pareto-dominates** plain max-min — same minimum, everyone else lifted as high as possible, nobody worse. ("Weakly Pareto-dominates" = at least as good for everyone, strictly better for some.) This is the one place you can honestly claim an improvement over max-min: not by beating it on its own objective, but by **recovering the welfare a plain max-min solution throws away.**

---

## 5. Where *genuine* Pareto-domination holds

Over the **inefficient** baseline: **RLHF / anonymous pooling.** Prior work (Gölz et al.) and the paper's own motivation establish that pooled RLHF can be Pareto-*inefficient* — it sits *inside* $\bar\Psi$, not on the boundary. Since bounded-harm is on the boundary by construction, it can **strictly Pareto-dominate RLHF** exactly where RLHF is inefficient (some agents better off, none worse). That is the clean, provable, per-dataset "yes," and it is the paper's thesis.

---

## 6. Honest nuance (keep it in)

Within a *fixed* population, bounded-harm vs. utilitarian is **redistribution, not Pareto improvement**: protecting the binding agent imposes negative externalities on everyone else — precisely what the participation-externality analysis (Fig 3) shows, and what the near-zero average welfare in the tight-floor panels of Fig 2 reflects. So for the "subset of data" reading: bounded-harm helps the *protected/binding* subgroup at a cost to the rest — that is the *purpose* of the harm floor, not a Pareto win over utilitarian. Be precise: Pareto-domination is over the *inefficient* baseline (RLHF); against efficient ones it is a frontier trade-off / leximin refinement.

---

## 7. Rebuttal framing

> Max-min is not a competing method we win or lose against — it is the tight-floor endpoint of our bounded-harm family. Sweeping the harm floor traces the entire efficient frontier from utilitarian to egalitarian, and setting the floor at the max-min value yields the *leximin* point, which weakly Pareto-dominates a plain max-min solution by not wasting the welfare that sits above the guaranteed minimum. Genuine Pareto-domination holds over the anonymous-RLHF baseline, which — unlike our mechanisms — need not be Pareto-efficient.

This converts "have you compared to max-min?" into "max-min is a **special case** of our framework, and our framework even fixes its one inefficiency."

---

## 8. Suggested empirical add-on (cheap in the codebase)

Add **max-min as a baseline** — one more LP over $\bar\Psi$: `max t s.t. Θψ ≥ t·1, ψ ∈ \bar\Psi` — a ~10-line sibling of `BoundedHarm` in `mechanisms.py`. Then, per dataset, compute the per-agent welfare vectors $\{U_n\}$ under RLHF, utilitarian, max-min, and bounded-harm, and report:
- **vs RLHF:** fraction of agents strictly better off with **none** worse — the domination result;
- **vs max-min:** bounded-harm(floor = max-min value) matches the minimum and raises the total (the leximin refinement / weak domination);
- a single **welfare-frontier plot** locating all mechanisms on the boundary of $\bar\Psi$ (utilitarian ↔ max-min, RLHF sitting inside) — answers the reviewer visually and permanently.

*(Internal working documentation for the rebuttal, not for distribution.)*
