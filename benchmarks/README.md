# Benchmark artifacts

Every figure in the paper and README traces to a committed script and an
archived result file. The map:

| artifact | produced by | supports |
|---|---|---|
| `results.{json,csv}` | `run_benchmarks.py --suite core --synthetic-only --methods ridge lightgbm beamfeat` | core-suite recovery 9/9, mean R² 0.9989, mean 1.7 features |
| `results_all.{json,csv}` | `run_benchmarks.py --suite all --methods ridge lightgbm beamfeat` | Friedman #1 (0.744), heavy-tail t(2) (0.956), threshold/piecewise (0.808) |
| `results_robustness.{json,csv}` | `run_benchmarks.py --suite robustness --synthetic-only --methods beamfeat openfe` | stress-suite false-feature rates; OpenFE 0/5 recovery |
| `results_real.{json,csv}` | `run_benchmarks.py --real-only --methods ridge lightgbm beamfeat` | six-real-dataset panel (mpg, tips, diamonds, penguins, diabetes, breast cancer) |
| `results_autofeat_venv.json` | `run_benchmarks.py --suite core --methods autofeat` in the pinned environment described in `independent/requirements.txt` | autofeat core-suite comparison (0.9974, ~9 s, 8/9) |
| `results_autofeat_robustness.json` | as above, `--suite robustness` | autofeat stress false-feature rate (mean 0.29, worst 0.75) |
| `results_knockpy.{json,csv}` | `run_benchmarks.py --suite all --synthetic-only --methods knockpy` | selection-only baseline: 0/15 recovery, stress false-feature rate 0.11 |
| `results_knockpy_real.{json,csv}` | `run_benchmarks.py --real-only --methods knockpy` | selection-only baseline on the real panel |
| `feynman_results.json` | `feynman_panel.py` | physics panel: 9/12 solved, 8/12 exact form |
| `calibration_study.py` | `python benchmarks/calibration_study.py` | 200-replicate FDR 0.0000 / power 1.000; 60-null zero selections |
| `friedman_decomposition.py` | `python benchmarks/friedman_decomposition.py` | the 0.964 oracle / 0.875 admissible-ceiling decomposition |
| `reproduce_all.ipynb` | run top to bottom | orchestrates the tests, both studies, the harness, and the panel in one traceable session |
| `independent/` | see `independent/README.md` | the 315-fit external comparison study |

`beamfeat` rows regenerate bit-identically from a fresh environment;
competitor rows require the isolated environments the files above name (see
the dependency notes in `independent/PROVENANCE.md`).
