# Changelog

Notable changes to `beamfeat`. Format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] - unreleased

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
  studies, and a vendored independent 315-fit comparison study.
- Supported: Python >= 3.10, numpy >= 1.26, scikit-learn 1.6 through 1.9
  verified, no upper version pins.
