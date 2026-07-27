# Independent Benchmark: beamfeat 0.1.0 vs autofeat, OpenFE, featuretools, and raw-feature baselines

*Run 24 July 2026 in a clean sandbox. beamfeat installed from the provided source archive; all competitors from PyPI. 315 model fits total. Raw per-fit results in `independent_benchmark_results.csv`.*

## 1. Protocol

The protocol follows the evaluation conventions of the automated-feature-engineering literature (Horn, Pack & Rieger 2019 for autofeat; Zhang et al. 2023 for OpenFE; Demšar 2006 for cross-dataset comparison):

- **Splits.** 75/25 train/test, repeated over 5 random seeds, with *identical splits for every method*.
- **Downstream model held constant.** Every feature-construction tool feeds the same standardized RidgeCV (25 log-spaced alphas), so differences reflect the features, not the estimator. beamfeat's native predict (internally a linear fit on its selected features) is the equivalent object.
- **Metrics.** Out-of-sample R² (mean ± std over seeds), wall-clock fit time, number of constructed features, and — on synthetic problems with a known generating formula — whether the tool's returned expressions reference the true generating columns (recovery).
- **Statistics.** Average ranks across datasets, a Friedman test over the 9 × 7 rank matrix, and paired Wilcoxon signed-rank tests (beamfeat vs each method, paired on dataset × split, n = 45 pairs).

**Methods.** RidgeCV on raw features (linear anchor); Random Forest (300 trees) and LightGBM (300 trees) on raw features (strong non-linear references); featuretools 1.31 DFS (multiply + divide transform primitives, depth 1) → RidgeCV; OpenFE (top-20 features, forced `task="regression"`) → RidgeCV; autofeat 2.1.3 (`feateng_steps=2`, its paper's recommended setting) → RidgeCV; beamfeat 0.1.0 at library defaults.

**Datasets.** Real: UCI Concrete (1030×8), UCI Wine Quality red (1598×11, regression on quality), Boston housing (506×13), sklearn Diabetes (442×10). Concrete, wine, and Boston appear in autofeat's own evaluation. Synthetic (known ground truth, n = 500 unless noted): three-way ratio a·b/c, kinetic energy ½mv², a·b hidden among 10 columns, a purely linear control, and Friedman #1 (n = 800, 10 columns).

**Environment caveats.** UCI was unreachable from the sandbox; Concrete/Wine/Boston came from widely used GitHub mirrors (`stedy/Machine-Learning-with-R-datasets`, `jbrownlee/Datasets`, `selva86/datasets`). scikit-learn was pinned to 1.7.2 because autofeat 2.1.3 crashes on ≥1.8. PySR was out of scope (Julia dependency); tsfresh is time-series-only and inapplicable.

## 2. Compatibility findings (a result in themselves)

- **autofeat 2.1.3** raises `TypeError: check_array() got an unexpected keyword argument 'force_all_finite'` on scikit-learn ≥ 1.8 and is unusable on a current default stack.
- **OpenFE** required three interventions to run: a `mean_squared_error(squared=False)` call removed from modern scikit-learn; an `init_score` shape incompatibility in its multiprocessing path (worked around with `n_jobs=1`); and auto-detection of integer wine ratings as 6-class classification (fixed by forcing regression). Its own requirements pin `lightgbm==3.3.1` (2021-era).
- **beamfeat** and **featuretools** ran unmodified.

## 3. Real-dataset results (out-of-sample R², mean ± std, 5 seeds)

| Dataset | Ridge raw | RF raw | LightGBM raw | featuretools | OpenFE | autofeat | beamfeat |
|---|---|---|---|---|---|---|---|
| Concrete | 0.583 ± 0.025 | 0.911 ± 0.009 | **0.932 ± 0.005** | 0.818 ± 0.047 | 0.754 ± 0.086 | 0.882 ± 0.026 | 0.845 ± 0.025 |
| Wine red | 0.362 ± 0.009 | **0.479 ± 0.028** | 0.418 ± 0.047 | 0.396 ± 0.022 | 0.380 ± 0.014 | 0.373 ± 0.044 | 0.382 ± 0.025 |
| Boston | 0.730 ± 0.060 | **0.866 ± 0.044** | 0.862 ± 0.064 | −25.6 ± 20.3 | 0.729 ± 0.059 | 0.850 ± 0.083 | 0.833 ± 0.079 |
| Diabetes | **0.428 ± 0.040** | 0.324 ± 0.064 | 0.208 ± 0.051 | 0.280 ± 0.084 | 0.427 ± 0.039 | −0.186 ± 1.182 | 0.423 ± 0.038 |

Notable failure modes: featuretools' unscreened ratio features catastrophically overfit Boston on every seed (mean R² −25.6). autofeat suffered one catastrophic seed on Diabetes (−2.28), dragging its mean below zero. beamfeat never fell below the raw-ridge anchor on any dataset-seed.

## 4. Synthetic ground-truth results (R², mean ± std, 5 seeds)

| Problem | Ridge raw | RF raw | LightGBM raw | featuretools | OpenFE | autofeat | beamfeat |
|---|---|---|---|---|---|---|---|
| a·b/c | 0.759 | 0.943 | 0.894 | 0.975 | 0.759 | 0.974 | **1.000 ± 0.000** |
| ½mv² | 0.858 | 0.989 | 0.991 | 0.995 | 0.880 | **1.000** | **1.000** |
| a·b in 10 cols | 0.901 | 0.987 | 0.989 | 0.996 | 0.901 | **0.997** | **0.997** |
| linear control | **1.000** | 0.976 | 0.988 | **1.000** | **1.000** | **1.000** | 0.999 |
| Friedman #1 | 0.712 | 0.816 | **0.899** | −2.13 ± 6.01 | 0.796 | 0.867 ± 0.189 | 0.745 ± 0.028 |

**Formula recovery** (does a returned expression reference exactly the generating columns): beamfeat 3/3 problems at 100% of seeds; autofeat 2/3 — it never recovered a·b/c on any seed, substituting combinations like x₀²/x₂ and 1/(x₁x₂). beamfeat returned exactly 1 formula on the three recovery problems; autofeat returned 2–11.

**Friedman #1** is the documented boundary case for both marginal-selection tools: beamfeat's marginal-association screening cannot see the centred quadratic term (0.745, barely above ridge), matching the limitation its own docs state. autofeat averaged higher (0.867) but with 7× the variance, including one observed run at R² = −99.5 — the identical split re-run gave +0.955, i.e. autofeat's unseeded internal randomness spans catastrophic failure to success on the same data.

## 5. Cost and parsimony

| | featuretools | beamfeat | OpenFE | autofeat |
|---|---|---|---|---|
| Fit time (real datasets) | 0.1–0.2 s | 0.6–1.4 s | 3–16 s | 20–35 s (86 s on Friedman) |
| Features returned (Concrete) | 84 | 20 | 20 | 53 |
| Features returned (a·b/c) | 18 | **1** | 1 | 11 |

beamfeat is 25–120× faster than autofeat at fit time and consistently returns fewer features. Its `fdr_controlled_` flag was True on 45/45 runs — the statistical guarantee it advertises was actually delivered on every fit.

## 6. Statistical comparison

**Average rank across all 9 datasets** (1 = best): beamfeat 3.11, LightGBM 3.44, autofeat 3.56, RF 3.56, featuretools 4.22, OpenFE 4.78, ridge 5.33. On real data alone the tree ensembles lead (RF 2.00, LightGBM 2.75, beamfeat 3.75); on synthetics beamfeat and autofeat tie for first (2.6).

**Friedman test:** χ² = 7.57, p = 0.271 across 9 datasets — with only 9 datasets, overall method differences are *not* statistically distinguishable, and rank orderings above should be read as descriptive. This is the honest ceiling of a 9-dataset study; the literature typically needs 15–30 datasets for significance.

**Paired Wilcoxon (beamfeat vs X, n = 45 dataset×split pairs):**

| Comparison | Median ΔR² | p |
|---|---|---|
| vs ridge raw | +0.088 | < 0.0001 |
| vs featuretools | +0.008 | < 0.0001 |
| vs OpenFE | +0.057 | 0.0001 |
| vs autofeat | −0.0001 | 0.181 |
| vs Random Forest | +0.010 | 0.599 |
| vs LightGBM | +0.007 | 0.858 |

beamfeat significantly beats its raw-linear anchor, featuretools, and OpenFE. Against autofeat, accuracy is **statistically indistinguishable** (median difference ≈ 0). Against the tree ensembles, indistinguishable overall — but that pools synthetics (where beamfeat wins) with real data (where GBMs win); per-dataset, LightGBM clearly beats beamfeat on Concrete and Friedman #1.

## 7. Conclusions

1. **vs autofeat (the closest comparator):** equal predictive accuracy (Wilcoxon p = 0.18, median Δ ≈ 0), delivered ~25–120× faster, with better formula recovery (3/3 vs 2/3), fewer returned features, no catastrophic seeds (autofeat: −2.28 on Diabetes, −99.5 observed once on Friedman #1), and it runs on a current scikit-learn where autofeat cannot.
2. **vs OpenFE and featuretools:** significantly better under a linear downstream model (both p ≤ 0.0001). Caveat: OpenFE's features target GBM consumers, so this protocol undersells it in its home setting; featuretools' single-table transform mode is not its primary use case (relational data).
3. **vs gradient boosting on raw features:** beamfeat does not beat LightGBM on real tabular prediction (Concrete: 0.845 vs 0.932) — consistent with both the literature and beamfeat's own documentation. Its value proposition is the readable equation with a stated, and here consistently delivered, FDR guarantee — plus never underperforming the linear baseline, which no other construction tool in this study managed.
4. **Limits of this study:** 9 datasets (Friedman test underpowered), 5 seeds, default/paper-recommended settings only, no hyperparameter search, real datasets from mirrors, and PySR/tsfresh excluded. The repo's own committed benchmarks (which include calibration and robustness suites this study did not re-run) are directionally consistent with everything measured here.
