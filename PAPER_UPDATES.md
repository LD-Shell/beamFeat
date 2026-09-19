# Paper updates

Edits to apply by hand. No `.tex` file was touched.

Section and line numbers are against
`beamfeat-paper-build/tmlr_camera/beamfeat_tmlr_camera.tex` (the newest of the
three builds). The same passages appear in `tmlr_submission/beamfeat_tmlr.tex`
and `preprint/paper_arxiv.tex`; the wording quoted below is the camera copy's.

Numbers marked **[measured]** come from
`benchmarks/parsimony_cost/paired_summary.json`, produced by
`benchmarks/parsimony_cost.py` on this machine (Python 3.11.15, numpy 2.4.4,
scikit-learn 1.8.0). Numbers marked **[to measure]** need a run that could not
be done here. Re-measure on the pinned machine before anything goes in the
paper; what follows is where the edits are, not final camera copy.

---

## 0. What changed, and what did not

**The default did not change.** `parsimony="forward"` is still the default,
the fitted equation is still the compact subset, and every feature count, R²,
recovery rate and table cell in the paper still describes the shipped
behaviour. Nothing in the results sections needs re-running on account of this
release.

Two things are new.

**`equation()` now marks a pruned equation as a subset.** The printed string
gains a suffix whenever the parsimony step dropped terms:

```text
y = 0.9999*(x0 * x1) - 0.003943   [1 of 25 certified terms; parsimony=None prints all 25]
```

This closes the gap the paper spends a paragraph of §6 apologising for. The
apology can stay — the open problem is unchanged — but it is no longer the
only thing standing between a reader of the equation and the scope of the
guarantee.

**The exchange the paper asserts is now measured.** §3.2 and §3.5 both state
that compactness is bought with the guarantee over the printed terms, without
saying what else it costs. Over 308 paired fits (default minus
`parsimony=None`; negative means the compact equation fits worse):

| study | fits | mean ΔR² | worst ΔR² | best ΔR² | terms, None ÷ default | `fdr_controlled_` disagreements |
|---|---|---|---|---|---|---|
| end-to-end calibration (200 signal + 60 null) | 260 | −0.0001 | −0.0003 | +0.0000 | 25.0× | 0 |
| known-formula + stress suites (18 datasets) | 18 | −0.0008 | −0.0148 | +0.0037 | 26.0× | 0 |
| Feynman physics panel (12 laws × 2 configs) | 24 | −0.0009 | −0.0111 | +0.0000 | 30.6× | 0 |
| Friedman #1 decomposition (6 draws) | 6 | −0.0095 | −0.0192 | +0.0019 | 3.6× | 0 |

**[measured]** Compactness costs about a thousandth of held-out R² on average
and at most 0.019 on a single fit, and buys an equation roughly twenty-five
times shorter. `fdr_controlled_` never disagreed. The end-to-end calibration
figures — empirical FDR, its upper bound, power, fallbacks, null selections —
are identical at both settings to every digit the paper reports.

And one finding that is stronger than anything the paper currently claims:
**pruning was never observed to inflate the realised false discovery
proportion.** On a selector-level stress built to force false discoveries
through (`selector_calibration.gaussian_design`, nominal q = 0.5, 300 trials),
the screened set realised FDP 0.397 and the greedily pruned subset realised
0.000, with the proportion rising in none of the 300 trials. The mechanism is
simple in hindsight: greedy forward selection ranks by predictive
contribution, a marginally null feature contributes nothing, so pruning drops
the false ones first. This is an observation, not a theorem — |S|/|S'| is
still the only bound available and it is weak — but it is a great deal better
than the bound alone, and §6 currently implies the reverse.

---

## 1. Section 3 (Method)

### 1.1 Section 3.4, the worked example (tex lines 236–248) — **required**

This is the one place the paper quotes `equation()` output verbatim, so it is
the one place the new suffix breaks the text.

Old:

> ```
> model.formulas()        # ['((P / n) * V)']
> model.equation()        # 'y = 0.1204*((P / n) * V) - 0.000179'
> model.fdr_controlled_   # True
> ```
>
> "The recovered coefficient 0.1204 is $1/R$ to three decimals."

**[to measure]** — rerun that exact snippet on the pinned machine and paste the
real string. It will read something like

> ```
> model.equation()        # 'y = 0.1204*((P / n) * V) - 0.000179
>                         #    [1 of N certified terms; parsimony=None prints all N]'
> ```

The $1/R$ sentence is untouched. The suffix is worth a clause immediately
after it, because it makes the §3.2 point concrete at the exact moment the
reader first sees an equation:

> "The bracketed suffix is the estimator recording its own scope: the
> guarantee of Proposition~\\ref{prop:fdr} covers the $N$ screened formulas,
> and the parsimony step of Section~\\ref{sec:defaults} fitted the compact
> subset of them printed here."

### 1.2 Section 3.2, the defaults paragraph (tex line 226) — recommended

Old:

> "The parsimony step then selects a subset of it by greedy forward selection
> on the same rows, and that subset --- the one whose coefficients
> `equation()` prints --- carries no separate guarantee of its own. The factor
> by which pruning can inflate the realised proportion is reported with every
> fit."

Two additions, both now backed by measurement:

> "...carries no separate guarantee of its own. The factor by which pruning
> can inflate the realised proportion is reported with every fit, and the
> equation itself records how many of the certified terms it prints, so a
> reader of the output cannot mistake the two sets for one. What the exchange
> costs in fit is small and measured: over 308 paired fits the compact
> equation is worse by 0.001 of $R^2$ on average and 0.019 at worst, and the
> flag never differs between the two settings."

**[measured]**

### 1.3 Section 3.5, "What the printed equation carries" (tex lines 331–353) — recommended

The three-way trade-off is right and the default is right; what is missing is
the price. After

> "`parsimony="forward"`, the default, is short and costs no rows. What it
> gives up is the second: its terms are drawn from the certified set by a
> greedy pass on the same rows, so the guarantee covers the set they came from
> and not the terms themselves."

add:

> "That is the whole of what it gives up. Measured across 308 paired fits, the
> compact equation costs 0.001 of $R^2$ on average and 0.019 at worst against
> printing the screened set entire, and pruning never raised the realised
> false discovery proportion in any regime tested. The exchange is a scope
> exchange, not an accuracy one."

**[measured]**

---

## 2. Section 4 (Results)

### 2.1 Section 4.2, FDR calibration — the design must be stated

This is the gap you asked about, and it is independent of everything above.
The paragraph at tex line 494 reads:

> "End to end at nominal FDR 0.10, 200 repeated trials with a true signal
> produced no false discovery at all, a 95% upper bound of 0.015 on the true
> rate, with power 1.000 and no fit falling back to unvetted output; 60 trials
> with no signal selected nothing."

Every number is correct and unchanged. What is missing is the design, which
Figure 2 and Appendix D both state for theirs. From
`benchmarks/calibration_study.py`:

| parameter | value |
|---|---|
| replicates | 200 signal, 60 global-null |
| n (rows per replicate) | **500** |
| input columns | **6**, i.i.d. $U(1,6)$ |
| columns carrying signal | **2** ($x_0, x_1$), through their product $x_0 x_1$ |
| generating target | $y = x_0 x_1 + \varepsilon$ |
| signal amplitude | $\varepsilon \sim N(0, (0.05\,\sigma_{\text{signal}})^2)$ — **noise at 5% of the signal's standard deviation**; $\sigma_{\text{signal}} = 7.42 \pm 0.18$ over the 200 draws, so $\sigma_\varepsilon \approx 0.371$ |
| implied ceiling | $R^2 = 1/(1+0.05^2) = 0.9975$ |
| marginal correlation of the true feature with the target | **0.9988** |
| candidates reaching the selector | **25** on every replicate (the beam width binds) |
| passing at $q = 0.1$ | **25** on every replicate |
| search settings | `max_depth=2`, `beam_width=25`, `target_fdr=0.1`, `selection_holdout=0.5` (250 search / 250 selection rows) |
| seeds | data `default_rng(10000 + trial)` (signal), `default_rng(50000 + trial)` (null); estimator `random_state=trial` |
| global-null arm | same 500 × 6 design, $y \sim N(0,1)$ |

Suggested addition after the first sentence:

> "The design is 500 rows and six independent $U(1,6)$ columns, of which two
> carry signal through their product, with noise at 5% of the signal's
> standard deviation --- a marginal correlation of 0.999 between the
> generating feature and the target, far stronger than the effect 3.0
> (correlation 0.44) of Figure~\\ref{fig:calibration} or the amplitude 0.6
> (correlation 0.28) of Appendix~\\ref{app:selectors}. It is an end-to-end
> check that the search/selection split restores the fixed-candidate premise,
> not a power measurement at the margin."

The two comparison correlations follow from the designs those sections already
state: $3/\sqrt{5\cdot 9 + 1} = 0.442$ and $0.6/\sqrt{10\cdot 0.36 + 1} =
0.280$. Quote them or not, but the asymmetry is what made the gap noticeable.

One more fact worth a clause: all 25 screened formulas on this design touch a
signal column, which is why the result is identical at `parsimony=None`. That
is a property of the design, not of the procedure, and should be stated as
such if stated at all.

### 2.2 Section 4.3, formula recovery — **no change, but know why**

> "...`beamfeat` solves 10/12 and recovers 8/12 in *exact symbolic form*..."

Both figures stand at the default and need no edit. It is worth knowing that
they are default-specific in a way the text does not say:
`feynman_panel._exact_form` matches the kept formulas against the expected
terms **as a multiset**, so it is a statement about the printed equation. At
`parsimony=None` the printed equation is the whole screened set and the same
test returns `False` for all twelve, while a separate diagnostic — is *any*
kept formula proportional to the law — is **16/18 at both settings**.
**[measured]**

Nothing to fix. But if a reviewer asks what exact form means at other
settings, that is the answer, and `benchmarks/PARSIMONY_COST.md` records it.

"Roughly 0.9 s per equation" is **[to measure]**: this stack gives 0.85–0.90 s,
but the archived JSON and the README's "~0.6 s" already disagree, so pin it on
the reporting machine.

### 2.3 Sections 4.5–4.7, the comparison tables — **no change**

Table 1 (features 9.2), Table 2 (features 3.6 to 34.0), the 1448 certified
formulas, the fourteen features on ct_slices — all are default-setting
quantities and all stand.

The only edit is the test count: **411 → 431** at tex line ~813 ("411
automated tests including scikit-learn's own `check_estimator` compatibility
suite"). `README.md` and `docs/installation.md` are already updated.

---

## 3. Section 5 and the appendices — **no change**

The depth ladder, scalability, selector comparison, split stability and
consumer ablation all report default-setting numbers and all stand. Two notes
for the record rather than for editing:

- **Scalability and the depth ladder depend on the parsimony setting**, which
  is not obvious from their code — both score `model.transform(...)`, the kept
  set. Their reported figures are the default's and are correct. If either is
  ever re-run at `parsimony=None`, recovery can only rise (the kept sets nest)
  and the "false 0 at every width" of Appendix C would need re-measuring,
  since the unpruned set is larger.
- **The label-permuted controls and Appendix D do not depend on it.** The
  controls report zero selections under a permuted label, where the parsimony
  branch is never entered; Appendix D calls the selectors directly.

`benchmarks/PARSIMONY_COST.md` has the full per-study map with the mechanism
for each determination.

---

## 4. Section 6 (Discussion and limitations)

### 4.1 The fourth direction (tex lines 955–976) — recommended

This paragraph is now the one place the paper is *more pessimistic than its
own evidence*. It stays — the open problem is real and unsolved — but two
sentences can be sharpened.

Old:

> "Because the subset is nested, the count of false selections cannot rise,
> but the false discovery *proportion* can, since the denominator shrinks
> faster. The realised proportion can grow by at most the factor $|S|/|S'|$
> --- an accounting identity, reportable per fit, though not an
> expectation-level guarantee at $q\\,|S|/|S'|$, since both set sizes are
> random. That factor is now reported with every fit."

Suggested addition after it:

> "The bound is weak and remains the only one available, but it is worth
> recording what pruning does in practice, since the bound alone invites the
> wrong reading. Greedy forward selection ranks by predictive contribution and
> a marginally null feature contributes nothing, so pruning tends to remove
> false discoveries rather than concentrate them: on a stress design at
> nominal $q = 0.5$ where screening realised a false discovery proportion of
> 0.397, the pruned subset realised 0.000, and across 300 trials the
> proportion rose in none. That is an observation about a heuristic, not a
> guarantee, and the direction below is what a guarantee would require."

**[measured]** — and it is, I think, the most interesting new thing in the
release, because it turns a stated liability into a measured one that does not
bite.

Add also, after "That factor is now reported with every fit":

> "...and the printed equation now records how many of the certified terms it
> contains, so the scope of the guarantee travels with the output rather than
> with the documentation."

### 4.2 The fifth direction, the Pareto front (tex lines 969–976) — optional

The closing observation —

> "Within a single screened set the exchange rate is at least reportable,
> since the greedy path from the empty model to the full screened set traces
> accuracy against term count on rows already used for selection, which would
> make the parsimony step legible without claiming anything new for it."

— now has two of its points measured. `parsimony="forward"` and
`parsimony=None` are the two ends of exactly that path, and the table in §0 is
those two endpoints on four studies. Worth a clause noting that the endpoints
are measured even though the curve between them is not.

---

## 5. What the PySR comparison shows

`benchmarks/pysr_panel/` adds PySR to the twelve-law panel: same generators,
same seeds, same 0.1% noise, same `r2_score`, and
`feynman_panel._exact_form` imported unmodified rather than reimplemented.
`README.md` and `PROVENANCE.md` there state the budget (`niterations=100`,
PySR's documented default, identical across all twelve laws), the operator
sets, and the front-selection rule (lowest-loss member, declared in advance,
with the whole front archived so any other rule is post-processing).

**The harness is complete and validated; it has not been run.** PySR installs a
Julia runtime on first import, and this sandbox's proxy refuses
`julialang-s3.julialang.org`, so no PySR number exists yet. Everything below is
what the comparison will show, argued from what is already known, and each
claim is falsifiable by the run. **Do not put any of it in the paper before
running `benchmarks/pysr_panel/run_panel.py`.**

What the design already determines, before any fit:

- **PySR should win outright on the two laws the panel lists as beamfeat
  boundaries, for exactly the reason PySR does not have them.**
  `relativistic_velocity` — $(v_1+v_2)/(1+v_1 v_2)$ — needs a literal constant
  *inside* the expression, and `gaussian` — $e^{-x^2/2}$ — needs both `exp`
  and a scale inside the exponent. A constant-free feature space cannot
  represent either; fitting constants inside expressions is PySR's central
  capability. Both carry `expected_terms = None` in `equations()`, so neither
  method is credited under `exact_form` and the win has to be read off
  `top_formula` in the record. Say so explicitly rather than letting a `null`
  column hide it.
- **`weighted_mean` and `distance_2d` are the other two beamfeat misses**, and
  neither needs a fitted constant — the first is a depth-3 rational, the second
  a depth-4 nesting. They are misses of *search depth*, not of representation,
  and PySR's search is not depth-limited in the same way. Expect PySR to reach
  both; the depth ceiling is a beamfeat design choice that §3.5 argues for on
  multiplicity grounds, and PySR does not pay that cost because it makes no
  error-rate claim.
- **Where beamfeat should hold is cost and the guarantee.** beamfeat's panel
  runs at roughly 0.9 s per equation. PySR at this budget, in the deterministic
  serial configuration the harness pins for reproducibility, will take minutes
  per equation. `PROVENANCE.md` says plainly that this is an upper bound on
  PySR's cost and not a like-for-like speed comparison, because PySR's default
  parallel search is much faster; **do not report a speed ratio from these
  numbers**. The honest framing is that the two methods answer different
  questions: PySR searches a larger space and certifies nothing, beamfeat
  searches a smaller one and controls an error rate over what it returns.
- **The panel exercises none of beamfeat's FDR machinery.** Twelve clean laws
  with known closed forms at 0.1% noise measure symbolic recovery, not error
  control. A PySR column here is a fair comparison of search; it is not
  evidence about the guarantee either way, and the paper should not let it read
  as such.

One asymmetry to disclose when the numbers land: beamfeat's formulas are
constant-free and its linear model supplies the intercept for free, so no
additive constant enters the proportionality ratio, while a PySR expression
carries its own and fails the identical test on it. The harness scores the
identical test (`exact_form`) and records `exact_form_after_intercept`
alongside as a labelled diagnostic. If the two differ materially for PySR,
report both — quoting only the first would overstate beamfeat's margin, and
quoting only the second would be the looser criterion the brief ruled out.

---

## 6. Repository changes already made (no paper action needed)

- `equation()` gains the subset suffix on the regressor and the classifier;
  docstrings, the `BeamFeatTransformer` parameter table and the
  `parsimony_holdout` fallback warning updated to match.
- `README.md`, `docs/guide.md`, `docs/guarantees.md`,
  `docs/migrating-from-autofeat.md`, `docs/installation.md` updated; the four
  notebooks regenerated from `build_notebooks.py`, re-executed, and the
  `docs/notebooks/` copies synced.
- Version 0.4.0 in `src/beamfeat/__init__.py` and `CITATION.cff`;
  `CHANGELOG.md` entry added.
- Tests: ten added (seven for the suffix, three for the default's relationship
  to the screened set, seven for `fdr_scope_`), three adjusted. **431 pass, none fail.**
- `benchmarks/PARSIMONY_COST.md`, `benchmarks/parsimony_cost.py` and the
  completed runs in `benchmarks/parsimony_cost/`.
- `benchmarks/pysr_panel/`. `pysr` is not in `pyproject.toml`.

### Two things to know

**The version does not live in `pyproject.toml`.** It declares
`dynamic = ["version"]` with
`[tool.hatch.version] path = "src/beamfeat/__init__.py"`, so there is no
version string there to bump; `__version__` in `src/beamfeat/__init__.py` went
0.3.1 → 0.4.0 instead. `CITATION.cff` was also a release behind (0.3.0 while
the package was 0.3.1); it is now 0.4.0, with `date-released` 2026-09-17.
Change that date if you release on a different day.

**One docs example does not reproduce on this stack, and it predates this
work.** `docs/guide.md`'s "Honest failure, by default" block claims
`BeamFeatRegressor(random_state=3)` on a pure-noise target selects nothing
(`y = 0.0353  (no feature passed selection)`). On numpy 2.4.4 with
scikit-learn 1.8.0, screening selects 12 formulas on that draw — at both
parsimony settings, so parsimony is not the cause. The example was left
untouched, since it may well reproduce on the pinned stack the docs were
written against. Worth checking, because the same seed underlies
`test_empty_default_on_pure_noise`, which passes at `beam_width=15` rather
than the default 50 — the difference is the beam, not the stack.
