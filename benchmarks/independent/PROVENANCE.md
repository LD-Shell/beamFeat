# Provenance and limits of the independent benchmark

This directory holds a comparison study of `beamfeat` against `autofeat`,
`OpenFE`, `featuretools`, and raw-feature baselines: 315 fits over nine
datasets, seven methods, and five train/test splits, with average-rank and
Friedman-test analysis. `beamfeat_benchmark.ipynb` reproduces the analysis
from the committed results; `bench.py` regenerates the results themselves.

Everything below is recorded so that a reader can judge which numbers are
reproducible and which are not.

## Environment

`requirements.txt` pins the environment the reported results were produced
in: numpy 1.26.4, scikit-learn 1.7.2, autofeat 2.1.3, OpenFE 0.0.12,
featuretools 1.31.0, LightGBM 4.7.0. The scikit-learn pin is not a
preference — `autofeat` 2.1.3 calls `check_array(force_all_finite=...)`,
removed in scikit-learn 1.8, and `OpenFE` 0.0.12 calls
`mean_squared_error(squared=...)`, removed in 1.6, so `patch_openfe.py`
edits its installed source. `beamfeat` and `featuretools` ran unmodified.
`beamfeat`'s own test suite runs unpatched from scikit-learn 1.6 through
1.9; the pin exists solely to keep the competitors runnable.

## Reproducibility is method-dependent

`beamfeat`, `featuretools`, and the raw-feature baselines are deterministic
given the split seeds and pinned versions. Re-running the harness twice on
Friedman #1 reproduced `beamfeat`'s held-out R² bit-identically across
independent runs (0.726082, 0.719561, 0.789567).

`autofeat` and `OpenFE` do not expose a seed for their internal
subsampling, so their timings — and for `autofeat` its selected features and
score — vary between runs of the same split. This is not a small effect. A
regeneration of `autofeat` on Friedman #1 under the pinned environment
(`results_as_reported/regenerated_autofeat_friedman1.json`) returned:

| split | as reported | regenerated |
|---|---|---|
| 0 | +0.955 | **−77.86** |
| 1 | +0.530 | +0.944 |
| 2 | +0.952 | +0.950 |

**Consequence for reading the tables: every `autofeat` number in this study
is one draw from a wide and partly heavy-tailed distribution, not a stable
measurement.** Its reported mean R² of 0.751 across the nine datasets should
be read as such, and its catastrophic-failure rate is plausibly higher than
a single pass suggests (the reported run already contains R² = −2.28 on a
Diabetes split; the regeneration above adds −77.86 on Friedman #1). The same
caveat applies in the other direction — a rerun may look better than what is
reported here. Comparisons against `autofeat` therefore carry irreducible
uncertainty that no amount of re-running on our side can remove.

## A corrected record

The originally reported `autofeat` value for Friedman #1 split 2 (R² 0.9518,
126.4 s, 20 features) came from an interrupted run and was inserted by hand
during aggregation rather than written by the harness. It has been
regenerated under the pinned environment and returned R² 0.9498, 54.9 s, 12
features — consistent in score, and consistent in the sense that matters for
the study's conclusions. Both the original file and the regeneration are
committed; the aggregate tables still reflect the originally reported value,
and this note exists so that no number in the study is taken on trust.

## Limits

Nine datasets, so the omnibus Friedman test is underpowered. Five splits.
Default or paper-recommended settings only, with no hyperparameter search
for any method — a tuned comparison could reorder the middle of the table.
Real datasets come from GitHub mirrors of UCI rather than UCI directly.
`PySR` is excluded (Julia toolchain) and `tsfresh` is inapplicable
(time-series feature extraction). The downstream model for every
construction method is the same standardised `RidgeCV`, which understates
`OpenFE`: it targets gradient-boosted consumers, and `featuretools`'
single-table transform mode is not its primary relational use case. Those
two comparisons should be read as "under a linear downstream model", not as
a verdict on either tool in its intended setting.
