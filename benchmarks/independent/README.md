# Independent comparison study

A 315-fit comparison of `beamfeat` against `autofeat`, `OpenFE`,
`featuretools`, and raw-feature baselines (ridge, random forest, LightGBM)
across nine datasets and five seeds, with average-rank and Friedman-test
analysis.

| file | contents |
|---|---|
| `REPORT.md` | the full write-up: protocol, results, statistics, conclusions |
| `PROVENANCE.md` | what is reproducible and what is not — read before citing |
| `beamfeat_benchmark.ipynb` | the study as a runnable notebook (`FULL_RUN = False` analyses the archived results in minutes; `True` regenerates them) |
| `bench.py`, `aggregate.py`, `run_all.sh` | the standalone harness |
| `requirements.txt`, `patch_openfe.py` | pinned environment and the OpenFE source patch it requires |
| `data/`, `results_as_reported/` | inputs and archived per-fit results |

## Headline figures

Mean out-of-sample R² across the nine datasets, and the worst of the 45
individual fits per method:

| method | mean R² | worst fit | mean seconds | mean new features |
|---|---|---|---|---|
| random forest (raw) | 0.810 | 0.247 | 0.86 | — |
| **beamfeat** | **0.803** | **0.355** | **0.61** | 9.2 |
| LightGBM (raw) | 0.798 | 0.117 | 0.07 | — |
| autofeat | 0.751 | −2.282 | 29.30 | 17.1 |
| OpenFE + LightGBM | 0.736 | 0.360 | 4.40 | 7.3 |
| ridge (raw) | 0.704 | 0.353 | 0.003 | — |
| featuretools | −2.478 | −57.320 | 0.11 | 104.7 |

`beamfeat` reported `fdr_controlled_ = True` on 45/45 fits, recovered the
generating formula on every ground-truth problem, and produced no fit below
R² 0.355. Its Friedman #1 result here, 0.745, independently reproduces the
0.744 measured by this repository's own harness.

## Caveats that travel with these numbers

Both are documented in full in `PROVENANCE.md`; in brief:

1. **One value was inserted by hand.** The `autofeat` Friedman #1 seed-2
   result came from an interrupted run. It affects `autofeat`'s Friedman #1
   mean and slightly affects the omnibus test; it touches no `beamfeat`
   figure. Because `autofeat` does not seed its internal subsampling, a
   re-run yields a different draw rather than the same value.
2. **The protocol favours linear consumers.** Every construction tool feeds
   the same ridge model, which understates `OpenFE` in its intended
   gradient-boosted setting and `featuretools` outside its relational use
   case. `beamfeat` does not win real-tabular prediction against tuned tree
   models on individual datasets (Concrete: 0.845 against LightGBM's 0.932).
