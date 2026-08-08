# beamfeat notebook series

Thirteen notebooks applying [`beamfeat`](https://github.com/LD-Shell/beamFeat) to problems
across geometry, engineering, physics, chemistry and biology.

`beamfeat` searches for short algebraic formulas that explain a target, then tests each
candidate on data the search never saw. The output is a readable equation plus a q-value
saying how much statistical confidence it carries.

Start with **`00_start_here.ipynb`**.

---

## Numbers in this README are placeholders

Every `___` below is a blank waiting on a local rerun. Nothing here is quoted from a
machine other than yours.

Each notebook opens with an **environment provenance cell** that prints the interpreter,
platform, CPU count and the version of every relevant package. Run the notebooks, read
the numbers off your own outputs, and fill the blanks in. The provenance cell output is
what makes those numbers meaningful to anyone else.

Fill in the environment you used:

| | |
|---|---|
| beamfeat | `___` |
| python | `___` |
| platform | `___` |
| scikit-learn / numpy / scipy | `___` |
| date of run | `___` |

---

## Choosing an environment

Any of these is fine — pick one and record it.

**1. Minimal (tutorials only).** Everything except notebook 06 needs just this:

```bash
pip install "beamfeat[units]" pandas matplotlib seaborn scikit-learn scipy
pip install cantera        # notebook 06 only
```

**2. Pinned comparison environment.** Required if you re-run the benchmark comparisons
against `autofeat`, `OpenFE` or `knockpy`, which need older pins than a current stack:

```bash
conda create -n af315 python=3.11 -y && conda activate af315
bash benchmarks/independent/setup_env.sh
```

**3. Your own.** No constraint beyond `beamfeat[units]`. The provenance cell will record
whatever you have.

The `REQUIRE` list at the top of the provenance cell controls what is treated as
mandatory. Tutorials ship with `REQUIRE = ["beamfeat"]`; for a benchmark rerun set it to
the full list and the cell will point you at `setup_env.sh` if anything is absent.

---

## Setup for the data

Every notebook creates a shared `csv/` folder on first run and caches its data there.
Notebooks 02–04 download three public datasets once, and notebook 12 pulls three more
from the [Rdatasets](https://vincentarelbundock.github.io/Rdatasets/) archive; the rest
generate their data and save it. **After one online run the whole series works offline.**

Each notebook comes in two forms:

- `NN_name.ipynb` — clean, run it yourself
- `NN_name_executed.ipynb` — a reference rendering

---

## What to expect

`beamfeat` recovers the correct law wherever the target is a product, ratio or power of
the inputs — and the fitted coefficient is then the physical constant. Fill in your own
errors:

| Notebook | Law recovered | Constant measured | Your error |
|---|---|---|---|
| 01 | area = πr² | π | `___` |
| 01 | perimeter = 2πr | 2π | `___` |
| 02 | carat ∝ volume | fill fraction | `___` |
| 06 | r = k[H][O₂] | rate constant | `___` |
| 06 | r = k[OH]² | rate constant | `___` |
| 07 | Re = ρvD/μ | — | exact / not |
| 07 | Dittus–Boelter exponents | 0.8 and 0.4 | `___` |
| 08 | ln k ∝ 1/T | activation energy Eₐ | `___` |
| 09 | T ∝ a^{3/2} | GM_Sun | `___` |
| 09 | same law, Jupiter's moons | GM_Jupiter | `___` |
| 10 | 1/λ ∝ 1/n² | Rydberg constant R_H | `___` |
| 11 | collective variable | — | decoys used? |
| 12 | gravity model (migration) | population / distance elasticities | `___` |
| 12 | Cobb–Douglas | labour + capital elasticities | sum `___` (theory 1.0) |
| 12 | misery index | inflation + unemployment | corr `___` vs components |

In none of these is the tool told what to look for. Notebook 01 sees 29 anonymous
columns and returns the area of a circle.

### Where a stock baseline should match it

Three cases, one reason each time — **there is no multiplicative structure to find**:

| Notebook | Situation | Your result |
|---|---|---|
| 01 | area is a degree-2 polynomial term | beamfeat `___` vs `poly(2) + ridge` `___` |
| 04 | penguin measurements near-linearly separable | beamfeat `___` vs logistic `___` |
| 05 | diabetes has no compact algebraic law | beamfeat `___` vs ridge `___` |

That is correct behaviour, not failure. Reach for `beamfeat` when you suspect products
and ratios matter; when you don't, a linear model is already enough and `beamfeat`
should roughly match it rather than blow up.

### The behaviour with no equivalent elsewhere

Other constructors will also find `x*y` when it is there. None of them tells you when
there is nothing:

| Check | Notebook | Expected | Yours |
|---|---|---|---|
| pure noise | 05 | no features, `fdr_controlled_ = False`, warning | `___` |
| same noise, `poly(2) + ridge` | 05 | held-out R² well below zero | `___` |
| distractor species | 06 | zero false features | `___` |
| 13 rows | 09 | `fdr_controlled_ = False` + warning | `___` |
| detection threshold | 05 | nothing at low SNR, reliable by SNR ≈ 0.4 | `___` |
| PCA comparison | 11 | PCA loads on decoys, misses the basin | `___` |

---

## Notebooks

| # | Notebook | Domain | Highlight |
|---|---|---|---|
| 00 | Start here | — | scorecard, known issues, reading order |
| 01 | Rediscovering geometry | shape measurements | πr² and 2πr from anonymous columns |
| 02 | Diamonds | 54k rows | volume from three dimensions; target transforms |
| 03 | Auto MPG | engineering ratios | units gotcha; reciprocal-target trap |
| 04 | Palmer penguins | classification | readable bill-shape indices |
| 05 | Knowing when to stop | false discovery control | refusal on noise; power curve |
| 06 | Reaction rate laws | chemical kinetics | rate constants; the scaling trap |
| 07 | Dimensionless groups | transport phenomena | Reynolds number; why to log-transform |
| 08 | Arrhenius | physical chemistry | Eₐ; narrow-range caution |
| 09 | Kepler's third law | celestial mechanics | GM_Sun and GM_Jupiter |
| 10 | Hydrogen spectrum | atomic physics | R_H twice over; three series |
| 11 | Collective variables | molecular dynamics | beats PCA outright |
| 12 | Three econometric classics | economics | gravity, Cobb–Douglas, misery index — real data |

**Suggested entry points.** Notebook 07 or 06 to see it work on something real; 05 if you
are sceptical and want to see it refuse; 01 for the shortest path from raw columns to a
known law.

---

## Known issues

Observed in **`beamfeat 0.1.1`**. All three fail *silently*, which is why they are
documented inside the notebooks rather than in a footnote.

**Re-run against your version and check whether each still reproduces.** If a release
has fixed one, the affected cell will change and the surrounding prose needs updating —
the notebook locations are given so this is a short job.

### 1. Absolute variance floor on the constant-column check

`Evaluator(variance_tol=1e-10)` compares raw variance, so a column can be rejected for
being small in the chosen unit. Concentrations in kmol/m³ sit near 1e-6, giving variance
around 1e-12.

If *every* column falls below the floor you get a clear `ValueError`. If only *some* do —
trace species alongside bulk ones, the normal case in chemistry — the fit succeeds with
the important column missing and `fdr_controlled_` still reports `True`.

- **Demonstrated in:** notebook 06, Part 3
- **Status in your version:** `___`
- **Workaround:** rescale inputs to roughly 0.01–100; check the variance column in
  `describe()` before fitting
- **Suggested fix:** a scale-free criterion such as coefficient of variation

### 2. `units=` as a list is silently ignored

Only the dict form, keyed by column name on a DataFrame, constrains the search. The list
form produces identical output with no error and no warning.

- **Demonstrated in:** notebook 03
- **Status in your version:** `___`
- **Workaround:** pass a dict, and confirm the feature set actually changed

### 3. `equation()` drops terms with small coefficients

Terms whose coefficient rounds to zero at four decimal places vanish from the printed
string even though `predict()` uses them. The model is correct; only the human-readable
output is wrong, and wrong by *omission* rather than rounding.

- **Demonstrated in:** notebook 00 (minimal reproducer)
- **Status in your version:** `___`
- **Workaround:** read coefficients from `coef_ / scaler_.scale_`, not `equation()`
- **Suggested fix:** significant figures or scientific notation instead of fixed decimals

---

## A configuration trap, not a bug

`knockoff+` thresholds on `(1 + #{W <= -t}) / (#{W >= t} or 1)`. The numerator starts
at 1 regardless of the data, so satisfying `target_fdr = q` requires **at least `1/q`
features to clear the threshold** — 20 at `q = 0.05`, 100 at `q = 0.01`. When the data
cannot supply that many genuine discoveries, no threshold works and the correct answer
is to select nothing. The pipeline downstream then fails on zero columns.

Enlarging the candidate pool does **not** fix it: notebook 12 goes from 2 inputs to 7,
growing the pool from 9 candidates to 40, and `q = 0.05` still selects nothing. The
binding constraint is how much real structure the data contains, not how many
candidates you generate.

Three real options, in order of preference:

1. **Use the default permutation selector** — no such floor, and recommended for
   engineered candidates, which are correlated by construction.
2. **Raise `target_fdr`** until `1/q` is below the number of discoveries you can
   plausibly expect.
3. **`KnockoffSelector(offset=0)`** — controls a *modified* FDR, `E[V/(R + 1/q)]`,
   not the FDR. Say so in writing if you use it.

`beamfeat` prints a clear diagnostic when this happens; the crash comes from
`LinearRegression` three steps later. Notebook 12 Part 4 includes a `fit_constructor`
guard that raises at the right place with an actionable message.

---

## Two habits the series is built around

**Two-stage estimation.** `beamfeat` fits ridge (`alpha=1.0`) on standardised features,
which shrinks a single coefficient by about `n/(n+alpha)` — negligible at n = 600, but
around 7% at n = 13. Use `beamfeat` to find the *form*, then OLS on that form to measure
the *constant*. Notebook 09 predicts the shrinkage to four decimal places.

**Association is not structure.** The FDR guarantee certifies that a selected formula is
not noise. It does not certify that the formula is the true mechanism. Notebook 06 shows
a Michaelis–Menten fit with an excellent R² and the enzyme concentration in the wrong
place — good fit, wrong chemistry. Always test a recovered law outside the range you
fitted.
