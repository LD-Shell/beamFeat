---
title: 'beamfeat: beam-search feature construction with false-discovery-rate controlled selection'
tags:
  - Python
  - feature engineering
  - feature selection
  - false discovery rate
  - knockoffs
  - interpretable machine learning
authors:
  - given-names: Olanrewaju M.
    surname: Daramola
    orcid: 0009-0006-3327-2047
    affiliation: "1"
affiliations:
  - index: 1
    name: Independent Researcher
author:
  - "Olanrewaju M. Daramola — ORCID: 0009-0006-3327-2047 — Independent Researcher"
date: 27 July 2026
bibliography: paper.bib
---

# Summary

Given a table of measurements, `beamfeat` searches for short mathematical
formulas — expressions such as $(x_0 / x_2) \cdot x_1$ or
$\log(a)\log(b)$ — that explain a quantity of interest, and then applies a
statistical test to decide which of the candidate formulas reflect real
structure rather than coincidence. The result is a model a person can read
as an equation, together with an explicit statement of how much statistical
confidence it carries. Automated formula search evaluates thousands of
candidates against the same data, so some will fit by chance; `beamfeat`
controls the expected fraction of such spurious selections — the false
discovery rate (FDR) — and reports honestly, through a fitted flag and
visible warnings, whenever that guarantee cannot be given. It is packaged as
scikit-learn [@scikit-learn] estimators, passing scikit-learn's
estimator-conformance checks, with optional dimensional analysis that
rejects physically meaningless expressions (metres plus kilograms) before
any numerical work.

# Statement of need

Researchers who need interpretable models from tabular data — recovering an
empirical law, screening engineered features for a regression, or auditing
which constructed variables genuinely relate to an outcome — currently
choose between tools that construct features without error control and
tools that control errors without constructing features. The selection step
is the scientific crux: candidates are generated adaptively and tested on
the same data, a selective-inference setting in which naive p-values are
optimistically biased. `beamfeat` treats that step as the inference problem
it is. It provides an exact permutation test of marginal association
[@phipson2010] with Benjamini--Hochberg [@bh1995] or Benjamini--Yekutieli
[@by2001] correction, the knockoff filter in both fixed-X [@barber2015] and
model-X [@candes2018] forms routed by the regime where each construction's
assumptions hold, and a default data split — search on one half of the
training rows, selection tested on the other — that restores the
fixed-candidate-set premise the guarantees require.

# State of the field

`autofeat` [@horn2019] is the closest predecessor: compact symbolic
features feeding a linear model, with exhaustive expansion and an L1-based
selection thresholded against injected noise columns — effective, but with
no stated error-rate guarantee. `OpenFE` [@zhang2023] generates features
for gradient-boosted models and ranks them by boosting-based importance;
Deep Feature Synthesis [@kanter2015] targets relational data; the reference
knockoffs implementation `knockpy` [@barber2015; @candes2018] selects among
*given* features and constructs nothing — its row below isolates
construction's contribution; symbolic
regressors such as `PySR` [@cranmer2023] conduct a strictly more powerful
expression search, fitting constants inside nonlinearities, with no error
control. The closest design precedent is `tsfresh` [@christ2018], which
filters mass-generated time-series features under Benjamini--Yekutieli
control; `beamfeat` brings that generate-then-error-controlled-filter
design to symbolic expressions on tabular columns.

Contributing the guarantee to an existing tool was not viable: it requires
a different selection statistic (exact under permutation), a holdout
architecture threaded through the fit path, and an expression engine whose
numerical failures are recorded rather than masked — a redesign of each
tool's core rather than a patch. The practical case is also material:
during benchmarking, neither `autofeat` 2.1.3 nor `OpenFE` 0.0.12 ran on a
current stack without intervention. `autofeat` raises
`TypeError: check_array() got an unexpected keyword argument
'force_all_finite'` whenever it transforms or predicts on unseen data having
engineered at least one feature, on scikit-learn $\geq$ 1.8 (the argument
was renamed in 1.6 and removed in 1.8) — that is, in its normal supervised
use — and its pins (`numpy<2.0`, `pandas<3.0`) downgrade current environments. `OpenFE` calls `mean_squared_error` with the
`squared` argument removed in scikit-learn 1.6. Both were therefore run in
isolated environments, with the `OpenFE` shim disclosed in the committed
harness, while `beamfeat`'s suite runs unpatched from scikit-learn 1.6
through 1.9 with no upper version pins.

# Software design

Three trade-offs define the design. First, the selection statistic is
*marginal* association (absolute Pearson correlation; eta-squared for
classification): a fixed function of the data, so permutation p-values are
exact, where statistics that re-tune themselves on the observed target — a
cross-validated lasso penalty — break the exchangeability a permutation
test rests on. The price is measured precisely on Friedman #1: a
least-squares oracle on the depth-2 basis $\{ab, (ab)^2, c, c^2, d, e\}$
reaches $R^2$ 0.964, but marginal screening admits only
$\{ab, (ab)^2, d, e\}$ — the centred quadratic is nearly marginally
independent of the target — for an admissible ceiling of 0.875 against the
pipeline's 0.744. The remainder is a search shortfall insensitive to beam
width, output diversity, and score lookahead (the intermediates survive
early beams and are displaced at final ranking), recoverable only by joint
candidate ranking, which reintroduces the selection-on-noise behaviour the
design excludes. Joint heuristic selectors outperform on exactly this
structure while certifying nothing.

Second, correctness defaults are chosen over power and convenience.
Benjamini--Yekutieli is the estimator default because, on pure-noise
pipelines with mutually correlated candidates, Benjamini--Hochberg breached
its level in 3/25 trials (mean false discovery proportion 0.12 against
nominal 0.10) where BY breached in 1/25 (0.04). When nothing passes
selection, the default returns *no* constructed features — an
intercept-only model and a visible warning — rather than unvetted search
output. A parsimony step (greedy forward selection *within* the screened
set) keeps fitted equations compact while the full screened set, with exact
p- and q-values per candidate, stays auditable; the q-level guarantee
certifies the screening set, and the documentation says so. A post-fit
check on the selection holdout separates two claims users conflate:
FDR-vetted association and a generalising fit — a negative held-out $R^2$
raises a visible warning naming the gap.

Third, boundaries are stated with measurements. Piecewise targets belong to
trees (LightGBM 0.998 against 0.808 on a threshold problem); the
exponential operator ships but is not a default (enabling it changed no
result on the physics panel below); overflow and domain errors exclude a
candidate loudly rather than silently; and units, supplied as pint
quantities or plain strings, are enforced at expression construction.

# Research impact statement

The repository's significance rests on committed, re-runnable evidence
rather than claims. End to end at nominal FDR 0.10, over 200 signal
replicates the empirical FDR was 0.0000 with power 1.000 and no fallbacks,
and 60 global-null replicates selected nothing. At the selector level,
realised FDR tracked nominal (Benjamini--Hochberg 0.065/0.120/0.210 at
0.05/0.10/0.20; BY 0.020/0.033/0.065; fixed-X knockoff+ 0.201 at 0.20) with
power 1.00. On a twelve-equation physics panel under the
symbolic-regression community's criterion ($R^2 > 0.999$ at 0.1% noise),
`beamfeat` solves 9--10/12 (a borderline surrogate flips across numeric stacks) and recovers a stable 8/12 in *exact symbolic form* — checked by
algebraic proportionality, the criterion the numeric threshold cannot
flatter — at 0.9 s per equation;
each miss marks a named boundary (a depth-3 rational, a literal constant, a
depth-4 nesting, the Gaussian's exponential).

Head-to-head on the committed harness (identical splits; competitor
environments matching their own dependency pins):

| measure | beamfeat | autofeat | OpenFE+LGBM | knockpy | LightGBM | ridge |
|---|---|---|---|---|---|---|
| formula recovery (9 recoverable) | **9/9** | 8/9 | 0/5 (stress) | 0/9 | — | — |
| mean $R^2$, core suite | 0.9989 | 0.9974 | — | 0.8450 | 0.9798 | 0.8442 |
| false-feature rate, stress suite | **0.000** | 0.29 (worst 0.75) | 0.000 | 0.11 (worst 0.33) | — | — |
| mean fit time | 0.2--0.5 s | ~9 s | ~7 s | <0.1 s | <0.1 s | <0.1 s |

: Benchmark results from the committed harness: formula recovery and mean
$R^2$ on the core suite, false-feature rate on the distractor stress suite,
and mean fit time. []{label="headtohead"}

Under Student-$t(2)$ noise the exact formula is recovered at $R^2$ 0.956
(LightGBM 0.866): the permutation test is distribution-free. Across six
real datasets, `beamfeat` leads on two — mpg (0.871 against LightGBM's
0.853) and tips, where gradient boosting overfits below ridge while the FDR
gate holds — matches ridge on diamonds from a single interpretable feature,
and trails LightGBM on penguins and breast cancer and ridge on diabetes: a
feature constructor should concede when there is nothing to construct, and
the honesty flag does so on every fit. A separate 315-fit study committed to the repository
(nine datasets, seven methods, five splits, analysed by average ranks and a
Friedman test [@demsar2006]) places these results in a wider field. Mean
held-out $R^2$ was 0.803 for `beamfeat` against 0.810 for a random forest and
0.798 for LightGBM on raw features, 0.751 for `autofeat`, 0.736 for `OpenFE`,
and 0.704 for ridge; `beamfeat`'s accuracy was statistically
indistinguishable from `autofeat`'s ($p \approx 0.18$) at roughly 48 times
the speed, and significantly better than `OpenFE` and `featuretools` under
the shared linear downstream model ($p \leq 0.0001$, a protocol that
understates `OpenFE` in its gradient-boosted home setting). The
worst-case column is the more informative one: `beamfeat`'s poorest single
fit of 45 was $R^2$ 0.355, against $-2.28$ for `autofeat` and $-57.3$ for
`featuretools`, and the FDR guarantee held on all 45 fits. That study also
records a reproducibility asymmetry: `beamfeat` returns bit-identical results
across runs given a seed, whereas `autofeat` seeds no internal subsampling:
repeated runs on one identical split returned $R^2$ values from $+0.955$ to
below $-77$, so any single `autofeat` figure — including those above — is one
draw from a wide distribution. Further reproducibility signals: 370 tests including full
estimator conformance, 95% statement coverage, continuous integration against
version floors and current releases, executable tutorial notebooks, and every
figure above produced by a committed script.

# Acknowledgements

The design of `beamfeat` was informed by a review of the `autofeat` source
code [@horn2019], whose approach it builds on and departs from as described
above. The author used Anthropic's Claude as an AI assistant, under the author's
direction, for design, implementation, benchmarking, and drafting. Every
statistical claim in this paper is backed by a committed, executable test
or replicate study run in continuous integration, and design decisions —
including AI-proposed ones that measurement refuted — are documented in the
repository alongside their evidence. The author reviewed all code and text
and takes sole responsibility for the work. This work received no external
funding.

# References
