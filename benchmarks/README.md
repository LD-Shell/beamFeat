# Benchmark artifacts

Every figure in the paper and README traces to a committed script and an
archived result file. The map:

| artifact | produced by | supports |
|---|---|---|
| `results.{json,csv}` | `run_benchmarks.py --suite core --synthetic-only --methods ridge lightgbm beamfeat` | core-suite recovery 10/10, mean R² 0.9989, mean 1.7 features |
| `results_all.{json,csv}` | `run_benchmarks.py --suite all --methods ridge lightgbm beamfeat` | Friedman #1 (0.744), heavy-tail t(2) (0.956), threshold/piecewise (0.808) |
| `results_robustness.{json,csv}` | `run_benchmarks.py --suite robustness --synthetic-only --methods beamfeat openfe` | stress-suite false-feature rates; OpenFE 0/5 recovery |
| `results_real.{json,csv}` | `run_benchmarks.py --real-only --methods ridge lightgbm beamfeat` | seven-real-dataset panel (mpg, tips, diamonds, penguins, diabetes, breast cancer, California housing) |
| `results_autofeat_venv.json` | `run_benchmarks.py --suite core --methods autofeat` in the pinned environment described in `independent/requirements.txt` | autofeat core-suite comparison (0.9968, ~12 s, 8/10) |
| `results_autofeat_robustness.json` | as above, `--suite robustness` | autofeat stress false-feature rate: 3 of the 5 scoreable datasets, worst 0.667 |
| `results_knockpy.{json,csv}` | `run_benchmarks.py --suite all --synthetic-only --methods knockpy` | selection-only baseline: recovers no formula on any of the 15 scoreable synthetic sets, 10 of which form the core suite quoted in the paper |
| `results_knockpy_real.{json,csv}` | `run_benchmarks.py --real-only --methods knockpy` | selection-only baseline on the real panel |
| `feynman_results.json` | `feynman_panel.py` | physics panel: 10/12 solved, 8/12 exact form |
| `make_figures.py` | `python benchmarks/make_figures.py` | seven ACS-format vector PDFs in `paper/figures/`, with PNG previews for the README and notebooks |
| `calibration_study.py` | `python benchmarks/calibration_study.py` | 200-replicate FDR 0.0000 / power 1.000; 60-null zero selections |
| `selector_calibration.py` | `python benchmarks/selector_calibration.py` | selector-level realised FDR and power (BH, BY, fixed-X knockoffs) and the global-null stress behind the BY default |
| `friedman_decomposition.py` | `python benchmarks/friedman_decomposition.py` | the oracle / admissible-ceiling / achieved decomposition over six draws, with the ordering checked on each |
| `reproduce_all.ipynb` | run top to bottom in the environment built by `independent/setup_env.sh` | regenerates every artifact in this table, checks the package against current numpy and scikit-learn, and asserts the load-bearing figures. 20-30 minutes |
| `independent/` | see `independent/README.md` | the external comparison study (360 fits); git repository only, not the sdist |

## Reading the false-feature rate

The rate is defined only for datasets that declare the columns the true
formula uses. `threshold_step` declares none, because its target is piecewise
rather than algebraic, so `_false_feature_rate` returns `None` there and the
dataset is excluded from the average. The rate is therefore over the **five** scoreable
datasets, not all six, and with five points a mean carries a standard error of
order 0.1. Report the count: beamfeat 0 of 5, OpenFE 0 of 5, knockpy 1 of 5,
autofeat 3 of 5.

## Suite scope

`--suite core --synthetic-only` covers 12 datasets and `--suite robustness`
covers 6. The core-suite mean R² of 0.9989 quoted in the paper is over the ten
of those twelve that carry a known generating formula; `friedman1` and
`heavy_tail_t2` are excluded from that average and reported separately.

## Reproducibility

`beamfeat` rows are deterministic given a seed and reproduce exactly within a
fixed numeric stack. The archived scores here were produced on Python 3.11.15,
scikit-learn 1.7.2, numpy 1.26.4. Regenerating the same suites under
scikit-learn 1.6.1 / numpy 2.3.5 returned every score to four decimals, and an
identical fit reproduced digit-for-digit across that stack and a
scikit-learn 1.8 / numpy 2.4 container. Across that range a few of the more
marginal fits move in the third decimal from floating-point associativity,
`friedman1` between 0.742 and 0.744 being the clearest case.

Compared-method rows require the isolated environments the files above name
(see `independent/PROVENANCE.md`). `autofeat` does not reproduce even within a
fixed stack, as `independent/autofeat_repeatability.json` records.
