# Provenance

What in this comparison is fixed, what was decided before results were seen,
and what a rerun will and will not reproduce. Read this before citing any
figure from it.

## What is compared

PySR against `beamfeat` on the twelve laws of `benchmarks/feynman_panel.py`,
in two operator configurations each, at 0.1% noise. Twenty-four PySR fits.
`beamfeat`'s side of the table is not rerun here: it is
`benchmarks/feynman_results.json`, produced by that script in the repository's
own environment.

## What is shared, and how that is enforced

`run_panel.py` imports `make_data`, `equations` and `_exact_form` from
`benchmarks/feynman_panel.py`. It does not copy them. A change to the panel's
generators, seeds, noise level or proportionality test therefore reaches both
methods at once and cannot reach one without the other. The data each method
sees is byte-identical: the same `crc32(name)` seed, the same domain per law,
the same 500 training and 500 probe rows, the same noise draw.

`make_data` was factored out of `feynman_panel.run_panel` for this purpose. It
is the original code moved, not rewritten: the generator calls and the order in
which the shared `default_rng` is consumed are unchanged, so the arrays it
returns are identical to those the previous version produced.

## What was decided in advance

Both of these were fixed before any equation was run, and neither is varied
per equation. Both are recorded in every output row so a reader need not take
this file's word for it.

- **Budget.** `niterations=100`, PySR 2.4.1's documented default, applied
  identically to all twelve laws in both configurations. `maxsize`,
  `populations`, `population_size`, `ncycles_per_iteration`, the parsimony
  coefficient and the loss are left at PySR's defaults; the values in force are
  recorded per run under `resolved_params`.
- **Front-selection rule.** All three of PySR's documented rules are scored on
  every front; `best`, PySR's own default, is the one the top-level figures
  use. The whole front is archived per equation, so any other rule can be
  applied to the committed JSON without rerunning anything. The rule was not
  fixed in advance and held — see below.

## The front-selection rule was changed, and here is why

This is the one place where the pre-registration discipline this file
describes was not simply followed, so it is recorded rather than tidied away.

An earlier run declared `accuracy` — the lowest-loss front member — as the
scored rule, before any equation was run. The stated reasoning was that the
panel's primary criterion is a held-out accuracy threshold, so a rule trading
accuracy for compactness would answer a different question, and that
`accuracy` is the only one of PySR's three rules with no tunable constant
inside it.

That reasoning was wrong, and the run showed it plainly. At 0.1% noise with
`maxsize` 30, the lowest-loss member of the front is the noise-fitting end:
`accuracy` picked complexity 23–30 on every one of the twelve laws, expressions
carrying coefficients around 1e-5 that track the noise, and recovered the exact
form of none of them — while the correct law sat in the same archived front at
complexity 4–7. It scored 12/12 solved and 0/9 exact form simultaneously, which
is a diagnosis of the rule rather than a measurement of PySR.

The harness now scores all three rules on every front and reports `best`.
Two things follow, and a reader should weigh both:

- The reported rule was chosen **with knowledge of that run**, so it is not a
  blind pre-registration and should not be read as one.
- Because all three are reported, the choice is no longer load-bearing.
  `accuracy` remains in every record, still scoring 0/9, so the failure that
  prompted the change is visible in the same file as the figures that replaced
  it.

The archived front is what made this correctable as post-processing rather
than as a silent rerun with a better-looking rule. That property is the reason
it is archived.

## Deviations from PySR's documented defaults

One, and it is in PySR's favour. PySR ships no unary operators by default,
which puts several panel laws outside its search space; the unary set is
therefore `beamfeat`'s own default set — `log`, `sqrt`, `square`, `abs`,
`inv` — so neither method holds an operator the other lacks. The binary set,
`+ - * /`, is PySR's documented default and `beamfeat`'s default alike.

Nothing else is changed. In particular the fitness function, the complexity
measure and the parsimony pressure are PySR's own.

## What reproduces

The search, given the environment of `requirements.txt`. `deterministic=True`
with `parallelism="serial"` and `random_state=0` is PySR's documented
configuration for a reproducible search, and it is what this harness uses. It
costs the parallel search, at roughly 21 s per equation on one core.

Reproducibility across *versions* is a weaker claim than reproducibility
within one. PySR's search, its Julia dependencies and its complexity
accounting all move between releases. Record `versions` from the output
alongside any figure quoted; the harness writes `pysr`, `julia`, `python`,
`numpy` and `platform` into every run for that reason.

## What does not

Wall time, measured at roughly 21 s per equation. It is single-core, serial,
and machine-dependent, and it is recorded per equation as an observation
rather than as a measurement of PySR's speed. PySR's default parallel search is substantially faster than the
configuration used here, so the seconds in this record are an upper bound on
what PySR costs, not a like-for-like speed comparison against `beamfeat`'s
sub-second fits. Do not report a speed ratio from these numbers.

## Two asymmetries the reader should know about

Neither is corrected silently; both are stated so a reader can weigh them.

1. **The intercept.** `beamfeat`'s formulas are constant-free and its linear
   model supplies the intercept for free, so no additive constant enters the
   proportionality ratio. A PySR expression carries its own, and the identical
   test then fails on it. The scored `exact_form` keeps the identical test;
   `exact_form_after_intercept` reports the same test with a leading additive
   constant removed, so a miss caused only by this is visible.
2. **Fitted constants inside expressions.** This is the capability the panel's
   own notes name as a `beamfeat` boundary: `relativistic_velocity` and
   `gaussian` are listed there as having no exact form available *for
   beamfeat* because they need a literal constant inside the expression, which
   a constant-free feature space cannot represent. PySR can fit such constants.
   Those two laws carry `expected_terms = None` in `equations()`, so
   `exact_form` is `null` for both methods and neither is credited. Any claim
   about PySR recovering them must be read off `top_formula` in the record,
   not off the exact-form count.

## Scope

The panel is twelve laws with known closed forms, 500 rows, 0.1% noise, one
seed per law. It measures symbolic recovery on clean small problems. It says
nothing about either method on real tabular data, on wide data, or under
heavier noise, and it carries no error-rate guarantee for either method: the
exact-form criterion is a check against ground truth, not a statistical
statement. `beamfeat`'s FDR machinery is not exercised by this comparison at
all.
