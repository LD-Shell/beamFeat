# Provenance and limits of the independent benchmark

This directory holds a comparison study of `beamfeat` against `autofeat`,
`OpenFE`, `featuretools`, and raw-feature baselines: 360 fits over nine
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
score — vary between runs of the same split. This is not a small effect.

Six runs of Friedman #1 split 0, at the library defaults in the pinned
environment, each a separate process (`autofeat_repeatability.json`):

| run | threading | R² | selected features |
|---|---|---|---|
| 1 | default | +0.952 | 17 |
| 2 | default | −100.96 | 26 |
| 3 | default | −105.31 | 19 |
| 4 | pinned to 1 | −109.79 | 21 |
| 5 | pinned to 1 | +0.950 | 18 |
| 6 | pinned to 1 | +0.939 | 20 |

The same instability shows at study level: four executions of the full
nine-dataset comparison returned autofeat mean R² of 0.746, -1.694, 0.754
and -1.561, with worst single fits from -2.28 to -108.9. Every other method
reproduced to four decimals across those runs.

An earlier regeneration of three splits
(`results_as_reported/regenerated_autofeat_friedman1.json`) shows the same
pattern: +0.955 became −77.86 on split 0, and +0.530 became +0.944 on split 1.

`autofeat` exposes no `random_state`. `featsel.py` seeds the per-run subsample
with `np.random.seed(i)`, but `_add_noise_features` draws its decoy features
from the global NumPy generator, seeded from OS entropy at process start. Those
decoys set the bar each candidate feature has to clear, so a different draw
changes which features survive. Calling `np.random.seed()` before fitting does
not control it, and pinning `OMP_NUM_THREADS`, `NUMBA_NUM_THREADS` and
`MKL_NUM_THREADS` to 1 does not either. The effect appears where selection is
marginal; on problems with an unambiguous signal repeated processes agree
exactly.

**Consequence for reading the tables: every `autofeat` number in this study
is one draw from a wide and partly heavy-tailed distribution, not a stable
measurement.** Its reported mean R² of 0.746 across the nine datasets should
be read as such, and its catastrophic-failure rate is plausibly higher than
a single pass suggests (the reported run already contains R² = −2.28 on a
Diabetes split; the repeatability runs above reach −109.8 on Friedman #1). The same
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
