"""Build the tutorial notebooks.

Notebooks are generated from source rather than hand-edited so that their code
stays in sync with the library and can be executed in CI. Run this script, then
execute the notebooks to populate outputs.
"""

from __future__ import annotations

import pathlib

import nbformat as nbf


def markdown(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip())


def build(path: pathlib.Path, cells: list[nbf.NotebookNode], title: str) -> None:
    notebook = nbf.v4.new_notebook()
    url = f"https://colab.research.google.com/github/LD-Shell/beamFeat/blob/main/notebooks/{path.name}"
    badge = markdown(
        f"[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})\n\n"
        "Running in Colab? Install first: `%pip install beamfeat` "
        "(before the PyPI release: `%pip install git+https://github.com/LD-Shell/beamFeat`)."
    )
    notebook.cells = [badge] + cells
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "title": title,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, str(path))
    print(f"wrote {path}")


HERE = pathlib.Path(__file__).parent


# --------------------------------------------------------------------------- #
# 01: Getting started
# --------------------------------------------------------------------------- #

getting_started = [
    markdown(
        """
# Getting started with beamfeat

beamfeat constructs interpretable features from tabular data. Given columns
`a`, `b`, and `c`, it searches expressions like `log(a)`, `a * b`, and
`(a * b) / c`, keeps the ones that predict the target, and reports them as
readable formulas.

Two things distinguish it from exhaustive feature engineering:

1. **The search is guided.** Candidates are scored against the target and only
   the best are extended, so cost does not compound with expression depth.
2. **Selection is calibrated.** The retained features carry a false discovery
   rate guarantee rather than surviving a heuristic threshold.

This notebook covers the basic workflow. Later notebooks cover the search and
selection machinery in detail.
        """
    ),
    code(
        """
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from beamfeat import BeamFeatRegressor

rng = np.random.default_rng(0)
        """
    ),
    markdown(
        """
## A problem with a known answer

We generate data where the target is a known function of the inputs, so we can
check whether beamfeat recovers it. The relationship is `y = (a * b) / c`,
which no linear model on the raw columns can represent.
        """
    ),
    code(
        """
n = 500
a = rng.uniform(1.0, 6.0, n)
b = rng.uniform(1.0, 6.0, n)
c = rng.uniform(1.0, 6.0, n)
d = rng.uniform(1.0, 6.0, n)  # an irrelevant column

X = np.column_stack([a, b, c, d])
y = (a * b) / c + rng.normal(0, 0.05, n)

print(f"{X.shape[0]} rows, {X.shape[1]} columns")
print(f"target range: {y.min():.2f} to {y.max():.2f}")
        """
    ),
    markdown(
        """
## Fitting

`BeamFeatRegressor` follows the usual scikit-learn interface. The two
parameters that matter most are `max_depth`, which bounds expression
complexity, and `beam_width`, which bounds how many expressions survive each
depth.
        """
    ),
    code(
        """
model = BeamFeatRegressor(max_depth=2, beam_width=30, random_state=0)
model.fit(X, y)

print(f"R^2: {model.score(X, y):.4f}")
print(f"features constructed: {model.n_features_out_}")
        """
    ),
    markdown(
        """
## Reading what it found

This is the part a gradient-boosted model cannot give you. The fitted model
exposes both the individual feature formulas and the full equation.
        """
    ),
    code(
        """
for formula in model.formulas()[:5]:
    print(" ", formula)
        """
    ),
    code(
        """
print(model.equation(max_terms=3))
        """
    ),
    markdown(
        """
The true generating expression should appear among the selected features. Note
that `x3` — the irrelevant column — should be largely absent.

## Comparing against a linear baseline

The point of constructing features is to let a simple model fit a relationship
it otherwise could not.
        """
    ),
    code(
        """
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

baseline = Ridge().fit(X_train, y_train)
beamfeat = BeamFeatRegressor(max_depth=2, beam_width=30, random_state=0).fit(X_train, y_train)

print(f"Ridge on raw columns:  R^2 = {baseline.score(X_test, y_test):.4f}")
print(f"beamfeat features:     R^2 = {beamfeat.score(X_test, y_test):.4f}")
        """
    ),
    markdown(
        """
## Using it in a pipeline

`BeamFeatTransformer` constructs features without fitting a model, so it
composes with any downstream estimator. Because construction happens inside
`fit`, cross-validating the pipeline does not leak information across folds.
        """
    ),
    code(
        """
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from beamfeat import BeamFeatTransformer

pipeline = Pipeline(
    [
        ("features", BeamFeatTransformer(max_depth=2, beam_width=20, random_state=0)),
        ("model", Ridge()),
    ]
)

scores = cross_val_score(pipeline, X, y, cv=5)
print(f"cross-validated R^2: {scores.mean():.4f} (+/- {scores.std():.4f})")
        """
    ),
    markdown(
        """
## Classification

`BeamFeatClassifier` works the same way, with `predict_proba` and
`decision_function` available as usual.
        """
    ),
    code(
        """
from beamfeat import BeamFeatClassifier

labels = ((a * b) > np.median(a * b)).astype(int)

classifier = BeamFeatClassifier(
    max_depth=2, beam_width=20, selector="permutation", target_fdr=0.1, random_state=0
)
classifier.fit(X, labels)

print(f"accuracy: {classifier.score(X, labels):.4f}")
print(f"features retained after FDR control: {classifier.n_features_out_}")
for formula in classifier.formulas():
    print(" ", formula)
        """
    ),
    markdown(
        """
## What to read next

- **02: Search and scoring** — how the beam search works, and how the three
  scoring strategies differ in what they detect and what they cost.
- **03: Selection and units** — how false discovery rate control works, why the
  default is not knockoffs, and how dimensional analysis constrains the search.
        """
    ),
]


# --------------------------------------------------------------------------- #
# 02: Search and scoring
# --------------------------------------------------------------------------- #

search_and_scoring = [
    markdown(
        """
# Search and scoring

This notebook covers how beamfeat explores the space of expressions, and how
the choice of scorer changes what it finds.

The core problem is that expression space is enormous. With 5 input columns, 5
unary operators, and 4 binary operators, exhaustive expansion to depth 3
produces well over a million candidates, and each must be evaluated before it
can be judged. Beam search avoids this by scoring candidates as they are
generated and extending only the most promising.
        """
    ),
    code(
        """
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from beamfeat import BeamSearch

rng = np.random.default_rng(0)

n = 500
data = {name: rng.uniform(1.0, 6.0, n) for name in "abcd"}
target = (data["a"] * data["b"]) / data["c"] + rng.normal(0, 0.05, n)
        """
    ),
    markdown(
        """
## Running a search directly

`BeamSearch` is the layer beneath the estimators. Using it directly is useful
when you want to inspect the search rather than fit a model.
        """
    ),
    code(
        """
search = BeamSearch(max_depth=2, beam_width=30, random_state=0)
result = search.run(data, target)

print(result.summary())
        """
    ),
    markdown(
        """
## The trace

Every search records what happened at each depth. This is how you attribute
cost, and how you notice a beam that collapsed or saturated.
        """
    ),
    code(
        """
print(f"{'depth':>6} {'proposed':>10} {'evaluated':>10} {'kept':>6} {'best':>8} {'seconds':>8}")
for record in result.trace:
    print(
        f"{record.depth:>6} {record.n_proposed:>10} {record.n_evaluated:>10} "
        f"{record.n_kept:>6} {record.best_score:>8.4f} {record.elapsed:>8.3f}"
    )
        """
    ),
    markdown(
        """
## How beam width controls cost

Beam width is the parameter that bounds work. Widening it explores more of the
space at proportionally greater cost; narrowing it risks pruning a parent whose
children would have been valuable.
        """
    ),
    code(
        """
print(f"{'beam':>6} {'proposed':>10} {'kept':>6} {'seconds':>9}  top feature")
for width in (5, 10, 25, 50, 100):
    outcome = BeamSearch(max_depth=2, beam_width=width, random_state=0).run(data, target)
    top = outcome.names[0] if outcome.names else "-"
    print(
        f"{width:>6} {outcome.n_proposed_total:>10} {len(outcome):>6} "
        f"{outcome.elapsed:>9.3f}  {top}"
    )
        """
    ),
    markdown(
        """
Note that cost grows with beam width but the *quality* of the top feature
plateaus quickly. In practice a moderate beam is usually enough, and the
remaining budget is better spent on depth.

## How depth controls expressiveness

Some relationships simply cannot be expressed below a certain depth. `(a * b) /
c` needs depth 2: one step to build `a * b`, another to divide by `c`.
        """
    ),
    code(
        """
for depth in (0, 1, 2, 3):
    outcome = BeamSearch(max_depth=depth, beam_width=25, random_state=0).run(data, target)
    found = any(all(token in name for token in "abc") for name in outcome.names)
    print(f"depth {depth}: {len(outcome):>3} features, "
          f"three-way interaction found: {found}")
        """
    ),
    markdown(
        """
## Choosing a scorer

Three scoring strategies are available, and they differ in what they can detect
and what they cost.

- `"correlation"` — absolute Pearson correlation against the residual. One
  matrix product per batch. The default.
- `"mutual_information"` — nearest-neighbour mutual information. Detects
  dependence that correlation misses.
- `"gradient_boosting"` — measured out-of-fold predictive improvement. Scores
  what is actually being optimised.

### Where they agree

On a relationship whose nonlinearity is captured by the expression itself, all
three find the same thing, and correlation is much cheaper.
        """
    ),
    code(
        """
import time

for name in ("correlation", "mutual_information"):
    started = time.perf_counter()
    outcome = BeamSearch(scorer=name, max_depth=2, beam_width=20, random_state=0).run(data, target)
    elapsed = time.perf_counter() - started
    print(f"{name:>20}: {elapsed:>6.2f}s   top: {outcome.names[0]}")
        """
    ),
    markdown(
        """
### Where they differ

Correlation is blind to symmetric relationships. If the target depends on the
*magnitude* of a feature but not its sign, correlation sees nothing while
mutual information sees the dependence clearly.
        """
    ),
    code(
        """
from beamfeat.scoring import CorrelationScorer, MutualInformationScorer

symmetric_feature = rng.uniform(-3, 3, n)
symmetric_target = symmetric_feature**2 + rng.normal(0, 0.3, n)
noise = rng.normal(size=n)

candidates = np.column_stack([symmetric_feature, noise])

corr = CorrelationScorer().score_batch(candidates, symmetric_target)
mutual = MutualInformationScorer().score_batch(candidates, symmetric_target)

print(f"{'':>12} {'true feature':>14} {'noise':>10}")
print(f"{'correlation':>12} {corr[0]:>14.4f} {corr[1]:>10.4f}")
print(f"{'mutual info':>12} {mutual[0]:>14.4f} {mutual[1]:>10.4f}")
        """
    ),
    markdown(
        """
Correlation cannot distinguish the true feature from noise here. Mutual
information can.

In practice this matters less than it appears, because beamfeat applies the
scorer to *transformed* columns. Once `square(x)` has been constructed, a
linear scorer detects it easily — which is why correlation remains a sensible
default despite this weakness.
        """
    ),
    code(
        """
squared = symmetric_feature**2  # what the search would hand the scorer
print(f"correlation on the raw column:       {CorrelationScorer().score(symmetric_feature, symmetric_target):.4f}")
print(f"correlation on the squared column:   {CorrelationScorer().score(squared, symmetric_target):.4f}")
        """
    ),
    markdown(
        """
## Redundancy control

A beam scored purely on individual merit fills with variants of the same
signal. beamfeat scores candidates against the residual left by what is already
selected, and additionally drops candidates too correlated with the current
beam.
        """
    ),
    code(
        """
from beamfeat import Evaluator

for threshold in (0.5, 0.9, 0.999):
    outcome = BeamSearch(
        max_depth=2, beam_width=30, redundancy_threshold=threshold, random_state=0
    ).run(data, target)

    evaluator = Evaluator(data)
    _, matrix = evaluator.evaluate_many(outcome.nodes)
    correlations = np.abs(np.corrcoef(matrix, rowvar=False))
    np.fill_diagonal(correlations, 0.0)

    print(
        f"threshold {threshold:<6}: {len(outcome):>3} features, "
        f"max pairwise correlation {np.nanmax(correlations):.3f}"
    )
        """
    ),
    markdown(
        """
## Auditing what was rejected

Candidates excluded during evaluation are recorded rather than silently
dropped. This matters when a search returns less than expected: the log tells
you whether candidates were numerically invalid, degenerate, or simply never
proposed.
        """
    ),
    code(
        """
signed_data = dict(data)
signed_data["e"] = rng.normal(0, 2, n)  # negative values break log and sqrt

outcome = BeamSearch(max_depth=1, beam_width=20, random_state=0).run(signed_data, target)

print("rejections by reason:")
for reason, count in outcome.evaluation_log.counts().items():
    print(f"  {reason.value:>15}: {count}")

print("\\nexamples:")
for record in list(outcome.evaluation_log)[:5]:
    print(f"  {record}")
        """
    ),
]


# --------------------------------------------------------------------------- #
# 03: Selection and units
# --------------------------------------------------------------------------- #

selection_and_units = [
    markdown(
        """
# Selection and units

Feature construction proposes far more candidates than any dataset can support.
Selection is where spurious features are either excluded or silently admitted,
and it is where most automated feature engineering tools are weakest: an
importance threshold gives no guarantee about how many reported features are
noise, and the expected number of false positives grows with the size of the
candidate pool.

beamfeat controls the **false discovery rate** — the expected proportion of
selected features that are spurious.
        """
    ),
    code(
        """
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from beamfeat.selection import KnockoffSelector, PermutationSelector

rng = np.random.default_rng(0)
        """
    ),
    markdown(
        """
## The problem, demonstrated

With a pure-noise target, every selected feature is by definition a false
positive. A well-calibrated selector should return almost nothing.
        """
    ),
    code(
        """
n_trials = 15
selections = []

for trial in range(n_trials):
    trial_rng = np.random.default_rng(100 + trial)
    features = trial_rng.standard_normal((200, 30))
    noise_target = trial_rng.standard_normal(200)

    result = PermutationSelector(target_fdr=0.1, n_permutations=20, random_state=trial).select(
        features, noise_target
    )
    selections.append(result.n_selected)

print(f"pure-noise target, {n_trials} trials, 30 candidate features each")
print(f"mean features selected: {np.mean(selections):.2f}")
print(f"trials selecting nothing: {sum(s == 0 for s in selections)}/{n_trials}")
        """
    ),
    markdown(
        """
## Measured calibration

More usefully: with real signal present, what fraction of selections are false?
This is the quantity the library claims to control.
        """
    ),
    code(
        """
def gaussian_design(seed, n=300, n_signal=5, n_noise=20, effect=3.0):
    trial_rng = np.random.default_rng(seed)
    n_features = n_signal + n_noise
    features = trial_rng.standard_normal((n, n_features))
    coefficients = np.zeros(n_features)
    coefficients[:n_signal] = effect
    target = features @ coefficients + trial_rng.standard_normal(n)
    truth = np.zeros(n_features, dtype=bool)
    truth[:n_signal] = True
    return features, target, truth


def realised_fdr(selector_factory, nominal, n_trials=15):
    false_rates, recalls = [], []
    for trial in range(n_trials):
        features, target, truth = gaussian_design(3000 + trial)
        result = selector_factory(nominal, trial).select(features, target)
        if result.n_selected:
            false_rates.append(np.sum(~truth[result.selected]) / result.n_selected)
        else:
            false_rates.append(0.0)
        recalls.append(np.sum(truth[result.selected]) / truth.sum())
    return np.mean(false_rates), np.mean(recalls)


print(f"{'nominal':>8} {'realised FDR':>14} {'power':>8}")
for nominal in (0.05, 0.1, 0.2):
    fdr, power = realised_fdr(
        lambda f, s: PermutationSelector(target_fdr=f, random_state=s, n_permutations=25), nominal
    )
    print(f"{nominal:>8.2f} {fdr:>14.3f} {power:>8.2f}")
        """
    ),
    markdown(
        """
Realised FDR tracks the nominal level, and power is full. Two reading notes,
both scientifically important. First, FDR is an **expectation**: individual
trials may exceed the level, and a mean over 15 trials carries visible Monte
Carlo noise, so values a little above nominal are consistent with control.
Second, an exact test *spends* its error budget — realised FDR near nominal
is the signature of a calibrated procedure, where a realised FDR pinned at
zero would signal wasted power.

## Knockoffs: two constructions, two sets of assumptions

The knockoff filter comes in two forms, and it matters which one you mean.

**Fixed-X knockoffs** (Barber & Candès, 2015) treat the design as fixed and
make *no assumption about the distribution of the features* — deterministic
engineered columns are admissible. The guarantee needs `n >= 2p` and Gaussian
noise in `y = Xβ + ε`. Its weakness on engineered designs is **power,
not validity**: near-duplicate columns drive the construction's `s` toward
zero, the knockoffs become nearly identical to the originals, and the filter
loses the ability to tell them apart.

**Model-X knockoffs** (Candès et al., 2018) instead assume the features are
jointly Gaussian — which engineered features are not: `log(a)` and `a * b`
are deterministic functions of shared parents. beamfeat only uses this
construction when `n < 2p`, where fixed-X does not exist, and warns.

beamfeat routes between the two automatically and reports which assumptions
are under strain, rather than pretending otherwise.
        """
    ),
    code(
        """
def engineered_design(seed, n=300):
    trial_rng = np.random.default_rng(seed)
    a = trial_rng.uniform(1.0, 5.0, n)
    b = trial_rng.uniform(1.0, 5.0, n)
    c = trial_rng.uniform(1.0, 5.0, n)
    columns = [
        a, b, c, np.log(a), np.log(b), np.sqrt(a), 1.0 / a,
        a * b, a / b, b / a, a * c, a - b, a + b, a**2, b**2,
    ]
    return np.column_stack(columns), a * b + trial_rng.normal(0, 0.1, n)


features, target = engineered_design(5000)

knockoff_result = KnockoffSelector(target_fdr=0.1).select(features, target)
print("knockoff selector on an engineered design:")
for message in knockoff_result.warnings_raised:
    print(f"  warning: {message}")

permutation_result = PermutationSelector(target_fdr=0.1, n_permutations=25).select(features, target)
print(f"\\npermutation selector warnings: {permutation_result.warnings_raised or 'none'}")
        """
    ),
    markdown(
        """
So the practical division of labour: the permutation selector is the default
because its exactness does not depend on the design at all; fixed-X knockoffs
are a strong choice for wide-enough problems with roughly Gaussian *noise*
(the features can be anything); model-X is a last resort for `n < 2p`.

One further caveat, measured rather than assumed: the `offset` parameter
dominates knockoff power on narrow designs. `offset=1` (knockoff+) requires
`(1 + #negatives) / #positives <= target_fdr`, which cannot be satisfied when
there are fewer than `1 / target_fdr` features, regardless of signal strength.
        """
    ),
    code(
        """
print(f"{'offset':>8} {'power':>8}")
for offset in (0, 1):
    recalls = []
    for trial in range(10):
        features, target, truth = gaussian_design(6000 + trial)
        result = KnockoffSelector(target_fdr=0.1, random_state=trial, offset=offset).select(
            features, target
        )
        recalls.append(np.sum(truth[result.selected]) / truth.sum())
    print(f"{offset:>8} {np.mean(recalls):>8.2f}")
        """
    ),
    markdown(
        """
## End to end

Putting search and selection together: construct candidates, then keep only
those that survive FDR control.
        """
    ),
    code(
        """
from beamfeat import BeamSearch, Evaluator

n = 500
data = {name: rng.uniform(1.0, 6.0, n) for name in "abcd"}
y = (data["a"] * data["b"]) / data["c"] + rng.normal(0, 0.05, n)

search_result = BeamSearch(max_depth=2, beam_width=40, random_state=0).run(data, y)
evaluator = Evaluator(data)
nodes, matrix = evaluator.evaluate_many(search_result.nodes)

selection = PermutationSelector(target_fdr=0.1, n_permutations=30).select(matrix, y)

print(f"proposed: {search_result.n_proposed_total}")
print(f"kept by search: {len(search_result)}")
print(f"kept by selection: {selection.n_selected}")
print("\\nselected features:")
for index in selection.selected:
    print(f"  {nodes[index].name}")
        """
    ),
    markdown(
        """
## Dimensional analysis

If your columns carry physical units, supplying them restricts the search to
dimensionally valid expressions. Adding a mass to a length is rejected at
construction time, before any numerical work happens.

This is both a correctness feature and a performance one: it prunes large parts
of the search space for free.
        """
    ),
    code(
        """
import pint

ureg = pint.UnitRegistry()

n = 400
physical = {
    "mass": rng.uniform(1.0, 5.0, n),
    "length": rng.uniform(1.0, 5.0, n),
    "time": rng.uniform(1.0, 5.0, n),
}
units = {
    "mass": 1.0 * ureg.kilogram,
    "length": 1.0 * ureg.meter,
    "time": 1.0 * ureg.second,
}

momentum = physical["mass"] * physical["length"] / physical["time"]
physical_target = momentum + rng.normal(0, 0.05, n)

unconstrained = BeamSearch(max_depth=2, beam_width=25, random_state=0).run(physical, physical_target)
constrained = BeamSearch(max_depth=2, beam_width=25, random_state=0).run(
    physical, physical_target, units=units
)

print(f"without units: {unconstrained.n_proposed_total} candidates proposed")
print(f"with units:    {constrained.n_proposed_total} candidates proposed")
print(f"reduction:     {100 * (1 - constrained.n_proposed_total / unconstrained.n_proposed_total):.0f}%")
        """
    ),
    code(
        """
print("top features with units enforced:")
for name in constrained.names[:5]:
    print(f"  {name}")

invalid = [name for name in constrained.names if " + " in name or " - " in name]
print(f"\\ndimensionally invalid sums or differences: {len(invalid)}")
        """
    ),
    markdown(
        """
Mass plus length never appears, because it was never constructible.

## Units through the estimator API

The same constraint is available on the estimators via the `units` argument.
The fitted equation is compact because, after FDR screening, the estimator
keeps a parsimonious forward-selected subset of the screened features by
default (`parsimony="forward"`); the full screened set with per-candidate
p- and q-values is available in `selection_report_`.
        """
    ),
    code(
        """
from beamfeat import BeamFeatRegressor

# Rebuild from a fresh generator so this cell does not depend on how many
# draws earlier cells happened to consume.
cell_rng = np.random.default_rng(7)
physical = {
    "mass": cell_rng.uniform(1.0, 5.0, n),
    "length": cell_rng.uniform(1.0, 5.0, n),
    "time": cell_rng.uniform(1.0, 5.0, n),
}
physical_target = (
    physical["mass"] * physical["length"] / physical["time"] + cell_rng.normal(0, 0.05, n)
)

X_physical = np.column_stack([physical["mass"], physical["length"], physical["time"]])
# Units may be pint quantities or plain strings; strings are parsed with
# pint at fit time.
estimator_units = {"x0": "kg", "x1": "meter", "x2": "second"}

model = BeamFeatRegressor(
    max_depth=2, beam_width=25, units=estimator_units, selector="permutation",
    random_state=0,
).fit(X_physical, physical_target)

print(f"R^2: {model.score(X_physical, physical_target):.4f}")
print(f"features retained: {model.n_features_out_}")
print(model.equation())
        """
    ),
]


if __name__ == "__main__":
    build(HERE / "01_getting_started.ipynb", getting_started, "Getting started with beamfeat")
    build(HERE / "02_search_and_scoring.ipynb", search_and_scoring, "Search and scoring")
    build(HERE / "03_selection_and_units.ipynb", selection_and_units, "Selection and units")
