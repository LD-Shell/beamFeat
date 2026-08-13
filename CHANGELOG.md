# Changelog

Notable changes to `beamfeat`. Format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-08-12

Behavioural release: the downstream ridge penalty is now selected by
leave-one-out cross-validation. Fitted coefficients and held-out scores move
against 0.2.0 -- in trailing digits when rows comfortably exceed the selected
features, and materially at p >> n, where the fixed default could produce an
effectively unregularised fit. Selections are unchanged: the search,
permutation tests and FDR control sit upstream of this step.

### Changed

- `alpha` on `BeamFeatRegressor` defaults to `"auto"`: the downstream ridge
  penalty is chosen by efficient leave-one-out cross-validation over a
  logarithmic grid, deterministically. Pass a float for the previous fixed
  behaviour. The chosen strength is exposed as `alpha_`.
- The `fdr_controlled_` documentation states the guarantee's object
  precisely: the set-level q guarantee applies to the full screened set in
  `selection_report_`; the parsimony subset is not re-certified at level q.
- The post-fit check is documented as what it is -- a degeneracy diagnostic
  on the selection rows, which also enter the final fit -- rather than an
  independent generalisation estimate, and its warning text says so.

### Fixed

- With many correlated certified features selected from few rows, the fixed
  default `alpha=1.0` amounted to almost no regularisation and could produce
  catastrophically negative held-out scores on features the screening had
  correctly certified. The cross-validated default keeps such fits on the
  scale of the target.

## [0.2.0] - 2026-08-08

Behavioural release: scale-free numerics throughout, strict unit validation,
and a faithful `equation()`. Results are not bit-reproducible against 0.1.1,
and selections on small- or mixed-unit data change deliberately.

### Added

- `style` parameter on `equation()` for both supervised estimators:
  `"significant"` (default), `"fixed"`, and `"scientific"`. `"fixed"` falls
  back to significant figures for any value that would otherwise display as
  zero.
- `units` accepts a positional sequence with one entry per column, alongside
  the existing mapping form.

### Changed

- `variance_tol` is judged relative to a column's squared mean magnitude, so
  the constant/not-constant verdict no longer depends on the caller's units.
  Anyone who tuned it against an absolute variance should re-check the value.
- Candidate ranking treats scores within 1e-6 as tied and admits the smaller
  expression; the redundancy pass then removes the complex near-duplicate.
- Invalid `units` fail loudly: a mapping matching no columns, a wrong-length
  sequence, or an unsupported type raises instead of silently skipping
  dimensional analysis.
- `equation()` orders terms by standardised coefficient magnitude and prints
  significant figures; `precision` counts decimal places only under
  `style="fixed"`.

### Fixed

- `equation()` dropped any term whose raw coefficient fell below the printed
  precision, deleting dominant terms on large-unit features from their own
  equation. Only exactly-zero terms are omitted now.
- Scoring zeroed any column whose absolute spread fell below 1e-12, which
  made products of small-unit columns undiscoverable; the guard is now
  relative to the column's magnitude.
- A saturated incumbent yields zero candidate scores at every target scale;
  previously the check held only for targets near unit magnitude.

## [0.1.1] - 2026-07-28

Documentation, benchmarks and packaging. No library code changed; the public
API, defaults and behaviour are identical to 0.1.0.

### Added

- `benchmarks/selector_calibration.py`: realised FDR and power for both
  multiplicity corrections, the two fixed-X knockoff offsets, and the
  global-null stress behind the Benjamini-Yekutieli default.
- `benchmarks/make_figures.py`: seven vector PDFs in `paper/figures/`, sized to
  the ACS column widths, including mean held-out R² against fit time from the
  comparison study.
- `benchmarks/independent/setup_env.sh`: builds the comparison environment,
  registers a Jupyter kernel, and smoke-tests `autofeat`, `knockpy` and
  `matplotlib`.
- `beamfeat_ridge` in the comparison study: `beamfeat` as a transformer into
  the shared `RidgeCV`, giving one row strictly like-for-like with the other
  feature builders.
- `docs/installation.md`: three setups, from `pip install beamfeat` to the
  pinned benchmark environment.
- `benchmarks/independent/autofeat_repeatability.json`: six runs of one split.

### Changed

- Friedman #1 decomposition is reported over six draws rather than one:
  oracle 0.960 ± 0.003, screening ceiling 0.874 ± 0.006, achieved
  0.776 ± 0.013, with about 0.086 lost to screening and 0.098 to the search.
  The previous single-draw figures came from a different split for the achieved
  value and overstated the gap. `friedman_decomposition.py` now checks the
  ordering the argument rests on rather than comparing against recorded
  numbers, and computes the ceiling on the set actually admitted: the marginal
  null never admits `c`, but admits `c²` on some draws, so the earlier
  "admits only four of the six" was true of one draw rather than in general.
- knockpy's stress false-feature rate was averaged over six datasets where
  only five declare their generating columns. Reported as a count of affected
  datasets instead, since a mean over five carries a standard error of order
  0.1; knockpy is also unseeded, so the count moves between runs.
- Selector calibration runs 100 trials rather than 25 and reports a standard
  error. The published 25-trial figures were noisy point estimates: BH at
  nominal 0.10 read 0.120, above nominal, where 100 trials give 0.084 ± 0.012
  against a ceiling of 0.080. `selector_calibration.py` now derives each
  procedure's bound from the design and fails if a realised rate exceeds it,
  so no calibration constant is written by hand anywhere.
- Fixed-X knockoff+ 0.201 to 0.159 and `offset=0` 0.289 to 0.249 at 100 trials.
- The stress-suite false-feature rate is reported as a count of affected
  datasets rather than a mean over five, which carried a standard error of
  about 0.15 for autofeat and 0.08 for knockpy.
- The end-to-end result is stated as no false discovery in 200 replicates with
  a 95% upper bound of 0.015, rather than as an empirical rate of 0.0000.
- Pure-noise stress at 100 trials: BH returned features in 6 trials against
  BY's 1, both under the nominal 0.10 and not significantly different
  (p = 0.12). BY is the default because it is valid under arbitrary
  dependence, not because a BH failure was observed.
- `autofeat` reproducibility: it exposes no `random_state` and draws decoy
  features from the global NumPy generator, so six runs of one identical split
  returned R² from +0.952 to -109.8 with 17 to 26 selected features. Thread
  pinning does not help; the previous "seeds no internal subsampling" was wrong.
- Real-data panel is seven datasets; California housing uses all 20,640 rows
  rather than a 2,000-row subsample.
- Every measured figure now comes from one machine (Dell Inspiron 16 Plus 7640,
  22 logical cores, Linux 7.0, Python 3.11.15, scikit-learn 1.7.2, numpy
  1.26.4), the pinned environment the compared tools require.
- Reproducibility is stated from two measured machines: deterministic given a
  seed within a stack, with marginal fits moving in the third decimal across
  scikit-learn 1.6 to 1.9.
- `pytest-cov` moved into the `test` extra, so `floor-versions` in CI and a
  local `pip install -e ".[test]"` both get it without naming it separately.
- The comparison notebook calls `bench.py` rather than duplicating it, dropping
  from 523 lines to 136.
- Removed `benchmarks/independent/run_all.sh`: it re-implemented the notebook's
  full-run path, so a method added in one place was missing from the other.
- The sdist ships the benchmark scripts and archived results, which
  `benchmarks/README.md` already documented.

### Fixed

- `feynman_panel.py` seeded each equation's data with `hash(name)`. Python
  randomises string hashing per process, so the panel drew different data on
  every run and reported 9/12 or 10/12 at random; the earlier claim that this
  varied "across numeric stacks" mistook the cause. Seeded with `zlib.crc32`,
  it now reproduces exactly at 10/12 solved and 8/12 exact form.

- Both OpenFE patchers imported the package before rewriting it, so the kernel
  kept running unpatched code while reporting success.
- `benchmarks/independent/requirements.txt` installs `beamfeat[all]`, so the
  test suite runs there; pins `setuptools<82`, without which every
  `featuretools` fit fails on the removed `pkg_resources`; and adds `knockpy`
  under the existing numpy pin rather than letting it pull numpy 2.
- `aggregate.py` takes the results directory as an argument and writes its CSV
  beside the files it read.
- Independent study regenerated in a single pass on one machine: 360 fits,
  nine datasets, eight methods. It replaces an archive stitched from several
  partial runs that contained one hand-transcribed fit.
- Removed an unsupported OpenFE figure on the diabetes dataset from the README.

## [0.1.0] - 2026-07-27

Initial release.

- Beam search over an expression DAG with commutative canonicalisation,
  local algebraic normalisation, and record-and-exclude handling of
  numerical failures. Unary operators `log`, `sqrt`, `reciprocal`,
  `square`, `abs` by default, with `cube` and `exp` available; binary
  `mul`, `div`, `add`, `sub`.
- FDR-controlled selection: exact permutation test of marginal association
  with Benjamini--Hochberg or Benjamini--Yekutieli correction (BY default),
  and the knockoff filter in fixed-X and model-X forms. Default holdout
  split separates search from selection; `fdr_controlled_` states whether
  the guarantee applies; `selection_report_` exposes per-candidate p- and
  q-values.
- Estimators `BeamFeatTransformer`, `BeamFeatRegressor`,
  `BeamFeatClassifier`, passing scikit-learn's estimator-conformance
  checks. Parsimony (forward selection within the screened set) on by
  default; `on_no_discoveries` in `"empty"` (default), `"fallback"`, and
  `"raise"` modes; a post-fit holdout check (`DegenerateFitWarning`)
  separating association from generalisation; readable `equation()` on
  both supervised estimators (log-odds form for classification).
- Optional dimensional analysis via pint quantities or unit strings,
  enforced at expression construction.
- Committed benchmark harness, physics-equation panel, calibration
  studies, and a vendored independent comparison study.
- Supported: Python >= 3.10, numpy >= 1.26, scikit-learn 1.6 through 1.9
  verified, no upper version pins.