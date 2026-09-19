# PySR on the twelve-law physics panel

`benchmarks/feynman_panel.py` scores `beamfeat` on twelve physics laws at 0.1%
noise under two criteria: *solved* at held-out R² > 0.999, and *exact form* by
algebraic proportionality against the generating expression. This directory
runs PySR on the same panel so the two can be read side by side.

| file | contents |
|---|---|
| `run_panel.py` | the harness: fits PySR per law, selects from the front, scores, records |
| `setup_env.sh` | one-shot environment setup: pinned dependencies, the Julia runtime, and verification |
| `requirements.txt` | the pinned environment |
| `PROVENANCE.md` | what is fixed, what was decided in advance, and what a rerun will and will not reproduce |
| `results/` | output of a run; `pysr_results.json` is the record |

## Why a separate environment

PySR runs its search in Julia and installs a Julia depot of several hundred
megabytes on first import, and its `juliacall`/`juliapkg` stack sets its own
numpy and pandas floors. The repository's development environment and the
`autofeat` environment next door (`benchmarks/independent/`, pinned to numpy
1.26.4) can host neither. This follows that directory's pattern exactly:
pinned `requirements.txt`, a `setup_env.sh` that refuses to install into a
base interpreter and verifies what it built, and results kept beside the code
that produced them.

`pysr` is deliberately **not** added to the repository's `pyproject.toml`.
Nothing in `beamfeat` imports it.

```bash
conda create -n pysr312 python=3.12 -y
conda activate pysr312
bash benchmarks/pysr_panel/setup_env.sh
cd benchmarks/pysr_panel && python run_panel.py | tee results/pysr_panel.log
```

## Same data, same scoring code

`run_panel.py` imports `make_data`, `equations` and `_exact_form` from
`benchmarks/feynman_panel.py` rather than restating them. Both methods
therefore see byte-identical inputs: the same twelve generators, the same
`crc32(name)` seed per law, the same 500 training and 500 probe rows over the
same domain, and the same noise at 0.1% of each signal's standard deviation.
`beamfeat` is not installed in this environment, so a run here cannot produce
a `beamfeat` number at all.

Both criteria are computed by the panel's own code:

- **solved** — `r2_score(y_test, predictions) > 0.999`, the function
  `beamfeat`'s own `score()` calls, on the same probe draw.
- **exact form** — `feynman_panel._exact_form`, unmodified. The selected PySR
  expression is handed to it as a one-element list of sympy terms, which is
  the path that function already takes for a `beamfeat` fit that captured a
  multi-term law as one composite. The proportionality test is the identical
  one; it has not been relaxed for PySR. The denominator is the **nine** laws
  that declare an exact form, not twelve: the panel marks the other three as
  having none, and neither method is scored on them.

One asymmetry in that test is worth stating plainly rather than quietly
correcting. `beamfeat`'s formulas are constant-free and its downstream linear
model supplies the intercept for free, so no additive constant ever enters the
ratio. A PySR expression carries its own fitted constants, and an additive one
makes the ratio non-constant and the test fail. The scored figure keeps the
identical test. Alongside it, `exact_form_after_intercept` reports the same
test with a leading additive constant removed first, so a miss that is due
only to that asymmetry is visible in the record rather than hidden by it.
**`exact_form` is the number to quote; `exact_form_after_intercept` is a
diagnostic.**

## The budget, fixed in advance

Set in `BUDGET` in `run_panel.py`, applied unchanged to all twelve laws in both
operator configurations, and written into every output row. It was fixed
before any equation was run and is not tuned per equation.

| setting | value | why |
|---|---|---|
| `niterations` | 100 | the budget. PySR 2.4.1's documented default, stated explicitly rather than left implicit |
| `maxsize` | default (30) | not set here; the deepest panel law needs roughly 13 nodes, so the default is not binding |
| `populations`, `population_size`, `ncycles_per_iteration`, `parsimony`, loss | defaults | not set here; the resolved values are recorded per run in `resolved_params` |
| `parallelism` | `"serial"` | required by `deterministic=True` |
| `deterministic` | `True` | the panel reruns to the same equations. Serial search is several times slower than PySR's default parallel search, and that cost is real; every other result in this repository is reproducible from a seed, and this one should be too |
| `random_state` | 0 | with the two above, fixes the search |
| `timeout_in_seconds` | 1800 | a ceiling, not the budget — see below |

Measured at roughly 21 s per equation on one core at this budget, so under
ten minutes for the full run of twenty-four fits (twelve laws × two operator
configurations).

`timeout_in_seconds` exists so a pathological search cannot hang the panel. It
is not part of the budget: a fit that reaches it has not spent the stated
iterations, so its row is flagged `budget_truncated`, the run says so on
stderr, and `run_panel.py` exits non-zero. Such a row is not comparable with
the rest of the panel and should not be reported — raise the ceiling and rerun.

### Operators

`beamfeat`'s default binary set is `*`, `/`, `+`, `-`, which is exactly PySR's
documented default `["+", "-", "*", "/"]`. No deviation.

PySR ships **no** unary operators by default. An empty unary set puts several
panel laws outside the search space entirely — the pendulum period needs
`sqrt` — so the unary set is `["log", "sqrt", "square", "abs", "inv"]`,
which is `beamfeat`'s own default unary set written in PySR's names. **This is
the only deviation from PySR's documented defaults, and it is in PySR's
favour.** The second configuration adds `exp`, mirroring the second
`beamfeat` panel.

### Selecting from the Pareto front

PySR returns a front, not a model, so a rule has to say which member is
scored. **All three of PySR's documented rules are applied to every front and
recorded**, so the choice is inspectable rather than load-bearing:

| rule | picks |
|---|---|
| `accuracy` | the lowest loss outright |
| `best` | the highest score among members within 1.5× the lowest loss — **PySR's default, and the rule scored here** |
| `score` | the highest score outright |

`_select` follows `pysr.sr.idx_model_selection` exactly, computed from the
archived front so the numbers are reproducible from the JSON without the
library. Each record also carries `pysr_get_best`, the library's own answer
under the same setting, and the run prints a note if the two ever disagree.

**The rule history is part of the record, not hidden by it.** An earlier run of
this panel declared `accuracy` as the scored rule, reasoning that the panel's
primary criterion is a held-out accuracy threshold and that `accuracy` is the
only one of the three with no tunable constant inside it. The run showed that
reasoning to be wrong: at 0.1% noise with `maxsize` 30 the lowest-loss member
is the noise-fitting end of the front, so `accuracy` picked complexity 23–30 on
every law and recovered the exact form of none of them, while the correct law
sat in the same front at complexity 4–7. Scoring all three is the fix, and
`accuracy` is still reported so the failure is visible rather than edited out.

The **entire front** is archived per equation in `pareto_front` — complexity,
loss, score, the Julia equation string and the sympy form of every member — so
any other rule can be applied to the committed JSON without a rerun.

## What is recorded

`results/pysr_results.json` is a single object:

```
versions      pysr, julia, python, numpy, platform
budget        the BUDGET dict as applied
rules         ["accuracy", "best", "score"]
front_rule    "best"                       the rule the top-level figures use
panels        [ {label, solved, exact_form, n_declared, by_rule,
                 mean_seconds, failed, truncated, rows}, ... ]
```

`by_rule` gives `solved`, `exact_form` and `median_complexity` under each of
the three rules, so the whole panel can be read under any of them without
touching the rows.

Each row carries, per equation: `r2`, `seconds` (wall time for that fit),
`solved`, `exact_form`, `exact_form_after_intercept`, `top_formula` (the full
selected expression), `selected_index`, `selected_complexity`,
`selected_loss`, `rules` (the full scoring of all three picks),
`pysr_get_best` (the library's own pick, for cross-check), `pareto_front` (all
members), `resolved_params`, `budget_truncated`, and `error`. The top-level
`r2`/`solved`/`exact_form`/`selected_*` fields are the `front_rule` pick,
copied up so the row joins cleanly against `feynman_results.json`.

The panel and row keys match `benchmarks/feynman_results.json` — `equation`,
`note`, `r2`, `seconds`, `solved`, `exact_form`, `top_formula`, and the panel
labels `"default operators"` and `"with exp enabled"` — so the two join on
`(label, equation)` into one table. Rows here carry `method: "pysr"`;
`feynman_results.json` has no `method` key, being `beamfeat` throughout.

## What this panel is, and is not

It measures **search**: whether a method recovers a known closed form from
clean data. It exercises none of `beamfeat`'s error control — twelve laws at
0.1% noise with known answers give an FDR procedure nothing to do, and PySR
makes no error-rate claim to compare against in the first place. A result here
says nothing about either method on real tabular data, on wide data, or under
heavier noise.

It is **not a claim the paper makes**. The paper's related-work section argues
that symbolic-regression benchmarks have no slot for the quantity it is about,
since none of their entrants reports an error rate over the expressions it
returns, and that a depth-limited constant-free constructor would be scored on
a complexity axis it does not compete along. This panel exists so that
boundary is measured rather than only asserted; it is not evidence about the
guarantee, and a figure from it should not be quoted as if it were.

## Failure is recorded, not skipped

An equation PySR cannot fit does not vanish from the table. The exception is
caught per equation, its traceback is stored in that row's `error`, the row
keeps `solved: false` and `exact_form: null`, and the run prints `FAILED` for
it. The JSON is written either way, so the record survives the failure, and
then `run_panel.py` exits 1. A panel with a hole in it is not a result, and
the exit code says so.
