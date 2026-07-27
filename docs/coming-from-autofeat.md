# Coming from autofeat

`beamfeat` began from a study of [autofeat](https://github.com/cod3licious/autofeat)
(Horn et al., 2019) and keeps its central idea: construct compact symbolic
features, then fit a linear model on the survivors so the result reads as an
equation. The APIs are close enough that most scripts port in a few minutes.
This page maps the parameters and, more importantly, sets expectations about
where the two behave differently.

## Minimal port

```python
# autofeat
from autofeat import AutoFeatRegressor
model = AutoFeatRegressor(feateng_steps=2, featsel_runs=5, verbose=1)
model.fit(X, y)
predictions = model.predict(X_new)

# beamfeat
from beamfeat import BeamFeatRegressor
model = BeamFeatRegressor(max_depth=2, target_fdr=0.1, verbose=1)
model.fit(X, y)
predictions = model.predict(X_new)
```

Both are scikit-learn estimators with `fit`/`predict`/`score`, both accept
NumPy arrays or DataFrames, and both expose the constructed features. From
here the differences matter more than the similarities.

## Parameter mapping

| autofeat | beamfeat | notes |
|---|---|---|
| `feateng_steps` | `max_depth` | Same idea: how many operator applications may compose. The cost profile differs — see below. |
| `featsel_runs` | `target_fdr` (+ `selector`) | Not a translation. `featsel_runs` repeats an L1 selection against injected noise columns; `target_fdr` states the error rate a single principled procedure controls. There is no run count to raise. |
| `transformations` | `unary_ops`, `binary_ops` | Split by arity. Defaults: `("log", "sqrt", "reciprocal", "square", "abs")` and `("mul", "div", "add", "sub")`. `"exp"` and `"cube"` are available but off by default. |
| `max_gb` | `beam_width`, `max_features` | Memory is bounded by construction rather than by a budget: the beam keeps `beam_width` candidates per depth, and `max_features` caps the returned set. (`BeamSearch` additionally accepts `max_candidates_per_depth` if you drive the search directly.) |
| `n_jobs` | *(none)* | Not implemented. The search is fast enough that it has not been needed; permutation testing is vectorised. |
| `units` | `units` | Both do dimensional analysis. `beamfeat` accepts pint quantities or plain strings (`{"m": "kg", "a": "m/s**2"}`) and rejects invalid expressions at construction, before any array work. |
| `verbose` | `verbose` | Same role. |
| *(none)* | `random_state` | `beamfeat` is deterministic given a seed; see the reproducibility note below. |
| *(none)* | `selection_holdout` | Fraction of training rows reserved for testing candidates the search did not see. Default 0.5. |
| *(none)* | `on_no_discoveries` | What to do when nothing passes selection: `"empty"` (default), `"fallback"`, or `"raise"`. |
| *(none)* | `parsimony` | Post-selection forward selection, so the fitted equation stays short. |

Attribute names also differ: `new_feat_cols_` becomes `formulas()`, and the
readable model is `equation()`.

## Behavioural differences to expect

**Selection is a hypothesis test, not a heuristic.** autofeat keeps features
whose L1 coefficients survive comparison against injected noise columns;
`beamfeat` computes an exact permutation p-value per candidate and applies a
false-discovery-rate correction, then reports whether the guarantee held via
`fdr_controlled_`. The practical consequence is that `beamfeat` will
sometimes return **nothing** — on a target with no learnable structure, the
default is an intercept-only model and a visible warning rather than a list
of features. That is deliberate; check `fdr_controlled_` after fitting.

**Search is guided, not exhaustive.** autofeat expands all combinations at
each step, so its cost grows combinatorially with column count and becomes
impractical on wide tables. `beamfeat` scores candidates and keeps the best
`beam_width` at each depth, giving cost O(depth x beam²). The trade is real
in both directions: `beamfeat` is far faster and scales to more columns, but
a greedy beam can prune a candidate an exhaustive expansion would have found.
Non-monotone interaction structure is the known case where exhaustive
expansion wins — documented in [Statistical guarantees](guarantees.md).

**Half the training rows are used for selection.** By default the search sees
one half and selection tests on the other, because candidates chosen for
their in-sample association cannot be honestly tested on the rows that chose
them. Set `selection_holdout=None` to disable it; `fdr_controlled_` then
reports `False`, because the guarantee no longer applies.

**Fewer features, by design.** After screening, a forward-selection pass
keeps the compact subset the equation uses. The full screened set with per
candidate p- and q-values remains in `selection_report_`.

**Numerical failures are recorded, not masked.** Overflow, domain errors, and
non-finite results exclude a candidate and are logged with a reason, rather
than propagating as silently clipped or `NaN`-filled columns.

**Results are reproducible.** `beamfeat` is deterministic given
`random_state`: the same data and seed produce bit-identical output.
`autofeat` does not expose a seed for its internal subsampling, so repeated
runs on the same split can select different features and score differently —
measured in `benchmarks/independent/PROVENANCE.md`, where one Friedman #1
split returned R² +0.955 and −77.86 on two runs of the same pinned
environment. If you are comparing the two, run autofeat several times.

## Version compatibility

`beamfeat` declares no upper version bounds and its suite is run against
scikit-learn 1.6 through 1.9. `autofeat` 2.1.3 pins `numpy<2.0` and calls
`check_array(force_all_finite=...)`, which scikit-learn removed in 1.8, so
the two cannot currently be installed in one environment. To benchmark them
against each other, use separate virtual environments — the harness in
`benchmarks/` does exactly that.

## When to stay with autofeat

If your problem is non-monotone smooth structure of the Friedman kind, an
exhaustive expansion finds a basis a guided search misses, and autofeat will
score better. If you want features for a gradient-boosted model rather than
a readable equation, [OpenFE](https://github.com/IIIS-Li-Group/OpenFE) is
built for that. And if you need constants fitted *inside* nonlinearities —
`exp(-3.2 x)` — neither library represents that;
[PySR](https://github.com/MilesCranmer/PySR) does.
