# User guide

Everything below is runnable as-is; outputs shown are from real runs.

## Install

```bash
pip install beamfeat            # core: numpy + scikit-learn only
pip install "beamfeat[units]"   # + pint, for dimensional analysis
```

See [Installation](installation.md) for running the test suite or
reproducing the benchmarks.

## Sixty seconds to a vetted equation

```python
import numpy as np
from beamfeat import BeamFeatRegressor

rng = np.random.default_rng(0)
X = rng.uniform(1, 6, (400, 4))
y = X[:, 0] * X[:, 1] + rng.normal(0, 0.05, 400)

model = BeamFeatRegressor(max_depth=2, beam_width=25, random_state=0).fit(X, y)
print(model.equation())
print(model.fdr_controlled_)
```

```text
y = 0.9974*(x0 * x1) + 0.0264
True
```

Two lines matter. `equation()` is the fitted model itself — evaluate it on
raw feature values and you reproduce `predict()`. `fdr_controlled_` states
whether the features carry the false-discovery-rate guarantee; check it
before treating them as statistically vetted.

## Column names flow into formulas

Fit on a DataFrame and formulas use your names:

```python
import pandas as pd

df = pd.DataFrame({"mass": rng.uniform(1, 5, 300),
                   "vol":  rng.uniform(1, 5, 300)})
target = df["mass"] / df["vol"] + rng.normal(0, 0.02, 300)
print(BeamFeatRegressor(random_state=0).fit(df, target).formulas())
```

```text
['(mass / vol)']
```

## Dimensional analysis

Give units as pint quantities or plain strings; dimensionally invalid
expressions (metres plus kilograms) are rejected before any numerical work — so `x0 + x1` (kg plus m) is never even scored:

```python
y_phys = X[:, 0] * X[:, 1] / X[:, 2]          # kg·m/s
model = BeamFeatRegressor(
    units={"x0": "kg", "x1": "m", "x2": "s"}, random_state=0
).fit(X[:, :3], y_phys)
print(model.formulas())
```

```text
['((x1 / x2) * x0)']
```

The recovered form, kg·m/s, is exactly the target's dimension — and the
dimensionally invalid spellings never consumed a beam slot.

Cover every column. A column without a unit is dimensionally
unconstrained, so it combines freely with the labelled ones and the check
stops binding on exactly the columns you did not vouch for — labelling your
real measurements and leaving the noise blank is the case worth naming,
because it looks careful and is not. Give the genuinely unitless columns
`"dimensionless"`; the estimator warns when coverage is partial, and raises
when the keys match no column at all. Keys are column names, so a DataFrame
lets you write `{"rho": "kg/m**3"}` rather than `{"x0": ...}`.

## Columns that carry no data

A column with no variation cannot be selected — it standardises to zeros, so
its association with the target is zero wherever it appears. That is the right
outcome and costs nothing, but in the output it looks the same as a column
that simply does not matter, so it is reported:

```text
beamfeat: 55 of 521 columns are constant (WAP003, WAP004, WAP092, ...) and
cannot be selected. They are ignored and cost nothing, but check whether they
are meant to carry data.
```

Constancy is relative to the column's magnitude, not an absolute floor on the
spread — a column varying over 0 to 1e-9 has a smaller variance than a stuck
sensor reading 9.81, and only the second is constant. Use `is_constant` to
apply the same test when screening a table before fitting.

A constant with a genuine unit is a different matter: gravity is m/s² whether
or not it varies, and expressions built from it are unit-checked, so label it.
Its absence from a recovered formula is not a miss — a multiplicative constant
is absorbed into the fitted coefficient rather than returned as a symbol.

## Audit what selection did

Every candidate's exact p-value and q-value is kept:

```python
for row in model.selection_report_[:3]:
    print(row["formula"], round(row["q_value"], 4), row["kept"])
```

Features are *kept* only if they pass FDR screening at `target_fdr`
(default 0.1, Benjamini–Yekutieli) on a held-out split, then survive a
parsimony pass that keeps the compact predictive subset.

## Honest failure, by default

On a target with no structure, nothing passes — and the model says so
instead of returning junk:

```python
noise = rng.standard_normal(400)
model = BeamFeatRegressor(random_state=3).fit(X, noise)
# NoDiscoveriesWarning: ... Returning no constructed features ...
print(model.equation())
```

```text
y = 0.0353  (no feature passed selection)
```

Prefer the old behaviour? `on_no_discoveries="fallback"` keeps the
unfiltered search output (flagged `fdr_controlled_=False`); `"raise"`
raises.

## Classification

```python
from beamfeat import BeamFeatClassifier

labels = (X[:, 0] * X[:, 1] > X[:, 2] * X[:, 3]).astype(int)
clf = BeamFeatClassifier(max_depth=2, random_state=0).fit(X, labels)
print(clf.equation())          # log-odds form; boundary = zero level set
```

## Real data with missing values or categoricals

beamfeat never imputes or encodes silently — compose explicitly:

```python
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

pipe = make_pipeline(SimpleImputer(strategy="median"),
                     BeamFeatRegressor(random_state=0))
```

## When to reach for something else

If a tree model beats beamfeat by a wide margin, your signal is likely
piecewise, not algebraic — use the tree. If you need constants fitted
*inside* expressions (`exp(-3.2*x)`), use a symbolic regressor such as
PySR. The [guarantees page](guarantees.md) states every boundary with the
measurement behind it.
