# Price of Strategyproofness — Rebuttal Working Doc

**Paper:** *Algorithmic Impact Reveals the Hidden Structure of Alignment*
**Concern:** the results state worst-case bounds on the welfare cost of strategyproofness and then show it is empirically bounded on 4 datasets. Reviewer asks: can we give that bound, or an indication of how much is sacrificed from the utilitarian optimum to keep strategyproofness, as a function of dataset attributes (e.g. polarization)? When preferences are polarized, is the cost low?

**Status:** answered — bound + closed form + measured on all four datasets.

---

## Rebuttal text (copy-pastable)

**On the cost of strategyproofness and its dependence on polarization.**

Our strategyproof mechanism is random dictatorship: deploy a uniformly random agent's optimal model. Its cost is the utilitarian welfare it gives up relative to the utilitarian-optimal deployment. We can make this precise in three ways.

**A computable bound.** The welfare lost to strategyproofness is at most the size of the impact set times the average disagreement among agents,
$$\text{cost} \;\le\; \operatorname{diam}(\bar\Psi)\cdot \tfrac1N\textstyle\sum_n\|\theta_n-\bar\theta\|.$$
Both factors are directly measurable per dataset, so the worst-case cost is not merely asserted to exist — it is computable, and it grows (at most linearly) with how spread out preferences are.

**A closed form for the intuition.** In the one-dimensional case, the fraction of utilitarian welfare *retained* under strategyproofness equals the majority margin exactly: if a fraction $f_+$ of agents fall on the majority side of a question, random dictatorship keeps a $(2f_+-1)$ fraction of the optimum, so the relative loss is twice the minority share. Full consensus makes strategyproofness free; a 50/50 split makes it maximally costly. This is the mechanism behind the bound above: cost is driven by *how evenly agents split on each question*.

**Does polarization make strategyproofness cheap?** It depends on how cost is measured, and the two answers go in opposite directions:

- In **absolute** terms, yes — but for a hollow reason. As preferences balance toward a 50/50 split, the mean preference shrinks toward zero, so the utilitarian optimum itself collapses. There is little total welfare available, so little is sacrificed — cheap because the pie shrank, not because the mechanism did well.
- In **relative** terms (welfare retained as a fraction of what was achievable), strategyproofness is *most* expensive precisely when preferences are polarized: the random dictator lands in the wrong camp about half the time and the two sides cancel.

So the accurate statement is the reverse of the guess: strategyproofness is cheap under consensus and expensive under (normalized) polarization. We regard this as a feature — the framework tells you exactly *when* the strategyproofness guarantee is affordable (homogeneous populations) and when it is not (deeply divided ones).

**Measured on the four datasets.** We compute the cost of strategyproofness (utilitarian welfare given up by random dictatorship) alongside a polarization index — the average, over deployment questions, of the minority share on that question (0 = unanimous, 0.5 = evenly split). The cost ranges from **16% on Kidney Allocation to 42% on Moral Machine, and rises monotonically with polarization**:

| Dataset | Cost of strategyproofness | Polarization index |
|---|---|---|
| Kidney Allocation | 15.7% | 0.34 |
| Community Alignment | 26.0% | 0.40 |
| 412 Food Rescue | 31.0% | 0.48 |
| Moral Machine | 41.8% | 0.59 |

The most consensual dataset (Kidney, where a shared logic of medical prioritization keeps disagreement low) is the cheapest place to be strategyproof; the most divided (Moral Machine, where respondents genuinely disagree about whom to save) is the most expensive. The ordering matches the polarization index exactly, empirically confirming the closed-form prediction that the cost of strategyproofness is governed by how evenly the population splits.

---

## Notes for internal use

- **Numbers** are computed by `hiddenstructure.welfare.distortion(Theta, ones(N), alpha_q)` at saturated β (deterministic deployments), using the cached cross-dataset fits (`load_xdatasets`). Cost of SP $= 1 - W_{\rm sp}/W^\*$; the paper's "distortion" is $W^\*/W_{\rm sp}$ (Kidney 1.19, CA 1.35, 412 1.45, MM 1.72).
- **Polarization index** $= \operatorname{mean}_z\big(1 - |\operatorname{mean}_n \operatorname{sign}(\alpha_z^\top\theta_n)|\big)$ — average per-question minority share. This is the theory-predicted driver and orders cost perfectly; raw preference dispersion $\frac1N\sum\|\theta_n-\bar\theta\|/\|\bar\theta\|$ (Kidney 0.83, 412 0.96, CA 1.28, MM 1.52) does *not* order it cleanly (412 is small, N=19, so its dispersion understates its question-level disagreement).
- **Derivation of the bound:** the two optimality inequalities $\bar\theta^\top\psi^\*(\bar\theta)\ge\bar\theta^\top\psi^\*(\theta_n)$ and $\theta_n^\top\psi^\*(\theta_n)\ge\theta_n^\top\psi^\*(\bar\theta)$ combine to bound each per-agent gap by $(\bar\theta-\theta_n)^\top(\psi^\*(\bar\theta)-\psi^\*(\theta_n))\le\|\theta_n-\bar\theta\|\operatorname{diam}(\bar\Psi)$; average over $n$.
- **One further subtlety** (drop in if space allows): utilitarian weights agents by preference *intensity*, random dictatorship by *headcount*, so they diverge most when a passionate minority opposes a mild majority — a second axis of "polarization" beyond a simple even split.

*(Internal working documentation for the rebuttal, not for distribution.)*
