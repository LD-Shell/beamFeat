# Statistical guarantees

This page states, for each selection procedure, exactly what is guaranteed,
under which assumptions, and what was measured. Every number here is
reproduced by the committed test suite (`tests/test_selection.py`,
`tests/test_estimators.py`) or benchmark scripts; trial counts are 25 unless
stated, so values carry Monte Carlo error of roughly ±0.03–0.06.

## What "FDR control at level q" means

The false discovery rate is the **expectation**, over repetitions of the
experiment, of the false discovery proportion (FDP) among selected features.
A single run with FDP above q is consistent with control; a mean over trials
above q is not. All documentation and tests in beamfeat use this definition.

## Permutation selector (default)

**Statistic.** |Pearson correlation| (regression) or eta-squared
(classification) between each candidate and the target — a fixed function of
the data, which is the condition for a permutation test to be exact.
Statistics that re-tune themselves on the observed target (e.g. a
cross-validated lasso penalty) violate exchangeability unless the tuning is
repeated inside every permutation; beamfeat deliberately avoids them.

**P-values.** Add-one estimator `(1 + #exceedances)/(B + 1)` (Phipson &
Smyth, 2010): exact, and never zero. The floor `1/(B+1)` interacts with the
multiplicity correction — Benjamini–Hochberg needs the leading feature to
reach `q/m` — so B is auto-raised to `ceil(2m/q)` (satisfiability plus
headroom for one null exceedance), capped at `max_permutations`.

**Null semantics.** Marginal independence. A near-duplicate of a genuine
signal is a *true* discovery under this null; de-duplication is performed by
the search's redundancy threshold, not the error-control procedure.

**Multiplicity.** The estimators (`BeamFeatRegressor`,
`BeamFeatClassifier`, `BeamFeatTransformer`) default to Benjamini–Yekutieli,
valid under arbitrary dependence; `PermutationSelector` used directly defaults
to Benjamini–Hochberg, valid under positive regression dependence (PRDS).
Engineered candidates share parents and are mutually correlated by
construction, so PRDS cannot be taken for granted and the estimators take the
conservative option. Over 100 pure-noise pipelines BH returned features in 6
trials against BY's 1 (FDP 0.06 and 0.01,
`benchmarks/selector_calibration.py`), both under the nominal 0.10. The
difference is not significant at this trial count (Fisher exact p = 0.12), so
read it as a consistency check rather than a demonstrated BH failure.

**Measured calibration** (Gaussian designs, 300 rows, 25 candidates, 5 signals,
100 trials). BH controls the FDR at `q·m0/m`, here `q·20/25`, and realised
0.046 ± 0.008, 0.084 ± 0.012 and 0.161 ± 0.016 at nominal 0.05/0.10/0.20
against ceilings of 0.040/0.080/0.160. BY is stricter by a harmonic factor and
realised 0.008 ± 0.004, 0.018 ± 0.005 and 0.046 ± 0.008. Power 1.00 throughout.
`benchmarks/selector_calibration.py` derives each bound from the design and
fails if a realised rate sits above one, so these figures cannot drift from the
behaviour.

**What the marginal null cannot see.** A feature can be jointly essential
yet nearly marginally independent of the target — a centred quadratic
`(c − 0.5)²` contributes almost no marginal correlation until its linear
complement is fitted. Marginal screening correctly excludes such features
under its null, at a measurable predictive cost on targets built from them
(Friedman #1, over six draws: the marginal null never admits `c`, giving a
screening-admissible ceiling of 0.874 ± 0.006 against a representation oracle
of 0.960 ± 0.003, with the pipeline reaching 0.776 ± 0.014. Around 0.086 is
lost to screening and 0.098 to the search. `c²` is admitted on some draws and
not others, since `(c − 0.5)²` expands to carry a little marginal signal
through it.)
Joint selection recovers
such features but certifies nothing; that trade is the design.

## Knockoff selector

**Fixed-X** (Barber & Candès, 2015), used when `n ≥ 2p`: no distributional
assumption on the features — deterministic engineered columns are fine. The
guarantee requires the linear model with Gaussian, homoskedastic noise. The
construction's exchangeability identities `X̃'X̃ = X'X` and
`X'X̃ = X'X − sI` are verified to numerical precision by the tests. On
near-singular designs validity holds but `s → 0` and power degrades toward
zero; a runtime warning says so.

**Model-X Gaussian** (Candès et al., 2018), used only when `n < 2p`: assumes
jointly Gaussian features, which engineered features violate; a warning is
recorded, and the permutation selector is recommended in this regime.

The estimators surface every selector diagnostic — assumption strain,
unsatisfiable configurations — through the `warnings` module, so an empty
knockoff result reads as "could not have selected at this configuration"
rather than "no evidence"; when knockoff+ selects nothing but `offset=0`
would have, the warning says so and names the trade-off.

**Offsets.** `offset=1` (knockoff+) carries the finite-sample FDR guarantee
and realised 0.159 at nominal 0.20 over 100 trials; it is unsatisfiable with
fewer than `1/q` features, and a warning fires on narrow designs. `offset=0`
controls only a modified FDR and realised 0.249, above `offset=1` as expected.
Both move across numeric stacks, since knockoff construction depends on matrix
decompositions; treat the ordering and the nominal bound as the claim, not the
second decimal. `benchmarks/selector_calibration.py` regenerates both. Both figures come from `benchmarks/selector_calibration.py`;
the committed tests pin the qualitative ordering and the nominal bound rather
than these decimals, which carry Monte Carlo error.

## Selective inference: why the estimators split their data

The search retains candidates *because* they correlate with the target in
its sample. P-values computed on that same sample are therefore
optimistically biased, and the nominal FDR is not guaranteed for them. By
default the estimators search on one half of the training rows and run
selection on the other (`selection_holdout=0.5`), restoring the
fixed-candidate-set premise. Measured end-to-end over
200 replicates at nominal 0.10: empirical FDR 0.0000, power 1.000, no
fallbacks; 60 global-null replicates selected nothing. If the holdout is disabled, the
estimator refuses to claim the guarantee (`fdr_controlled_ = False`).

## Parsimony within the screened set

On a strong signal the marginal null correctly passes many mutually
redundant true discoveries. By default the estimators therefore apply greedy
forward selection *within* the screened set (`parsimony="forward"`) and fit
the compact subset; the full screened set, with exact p- and q-values per
candidate, remains auditable in `selection_report_`. Stated precisely: the
q-level guarantee certifies the screening set; the kept subset is a
predictive selection from within it. Set `parsimony=None` to keep the entire
screened set.

## Association is not generalisation

FDR screening certifies that kept features are genuinely associated with the
target. It does not certify that the linear model assembled from them
generalises: a true discovery can carry extreme leverage — a reciprocal or
logarithm evaluated near its pole — that destabilises the least-squares fit
at points close to the singularity. The estimators keep these claims
separate with a post-fit check on the selection holdout: a negative held-out
R² (regression) or held-out accuracy below the majority-class rate
(classification) raises a visible `DegenerateFitWarning` naming the gap.
Rank-based scoring deserves particular attention here, since rank
transformation hides exactly the magnitudes that cause the problem.

## Honest failure

When selection passes nothing, the default (`on_no_discoveries="empty"`)
keeps no constructed features: the model degrades to an intercept-only fit
(training mean, or class prior for classification) and a visible
`NoDiscoveriesWarning` is raised. `"fallback"` keeps the unfiltered search
output with `fdr_controlled_ = False`; `"raise"` raises. In every mode, check
`fdr_controlled_` before treating fitted features as statistically vetted.
