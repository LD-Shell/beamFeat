# Additional experiments

Experiments added in revision: a high-dimensional evaluation with baselines,
a depth and robustness characterisation of the search, a systematic
comparison of the selection procedures, and a split-stability study. The
folder is self-contained: nothing in the parent repository is modified, and
`bench.py` is a copy of `../independent/bench.py` with three additions, each
marked in-line -- a `highdim` dataset group, a per-fit wall-clock budget
(`FIT_BUDGET_S`, default 900 s; `0` restores the original in-process
behaviour exactly), and existence-guarded registration of the optional larger
datasets.

## Contents

    bench.py                 comparison harness (copy of independent/ + highdim group + budget)
    aggregate.py             unmodified copy of independent/aggregate.py
    fetch_data.py            re-derive every dataset from canonical sources; verifies checksums
    depth_ladder.py          recovery of planted targets across depth, scorer, beam width,
                             and operator sets, including marginally invisible intermediates
    scalability.py           controlled sweep in p: time, peak memory, recovery, false
                             features, and the FDR flag per cell
    selector_comparison.py   BH, BY, fixed-X and model-X knockoffs under both null
                             definitions across four dependence regimes
    split_stability.py       R^2 dispersion, value-equivalent Jaccard, selection frequencies
    multisplit.py            MultiSplitBeamFeat: frequency-thresholded aggregation over
                             many internal splits (see its docstring for scope)
    make_figures.py          regenerates every figure from results/, in the paper's style
    run_all.ipynb            driver notebook: the commands below as cells
    data/                    core five datasets + CHECKSUMS.md5 (fetch_data.py adds the rest)
    results_dev/             records from development runs on a constrained machine;
                             regenerate locally before citing any number
    PROVENANCE.md            sources, citations, preprocessing

## Run order

    # 0. environment: the repo's dev environment plus
    pip install lightgbm pmlb rdata

    # 1. datasets (network access required; ~10 min)
    python fetch_data.py --all

    # 2. comparison study, staged (fast methods first)
    python bench.py highdim ridge_raw,rf_raw,lgbm_raw,beamfeat,beamfeat_ridge 5 results/highdim_fast.json
    python bench.py highdim featuretools,openfe 5 results/highdim_constructors.json
    # autofeat runs in its pinned venv as in the original study:
    #   bash ../independent/setup_env.sh   (see ../independent/README.md)
    python bench.py highdim autofeat 5 results/highdim_autofeat.json

    # 3. search, selector, and stability experiments
    python depth_ladder.py --seeds 20 --out results/depth_ladder.json
    python depth_ladder.py --seeds 20 --binary mul,div --out results/depth_ladder_ops_muldiv.json
    python depth_ladder.py --seeds 20 --unary square,abs --out results/depth_ladder_ops_nosqrt.json
    python scalability.py --p-grid 10,30,100,300,1000 --seeds 5 --out results/scalability.json
    python selector_comparison.py --trials 100 --out results/selector_comparison.json
    python selector_comparison.py --trials 100 --m 100 --k 10 --out results/selector_comparison_m100.json
    python split_stability.py --splits 30 --out results/split_stability.json
    python multisplit.py data/tecator.csv fat --splits 20

    # 4. aggregate and plot
    python aggregate.py
    python make_figures.py

Every script checkpoints as it goes (per dataset, per problem, or per
regime-level), so an interrupted run resumes by rerunning the remaining
subsets, e.g. `DATASETS=riboflavin_p4088 python bench.py ...` or
`python depth_ladder.py --problems d4_distance2d ...`.

## Knobs

    FIT_BUDGET_S   per-fit wall-clock budget in seconds (default 900; 0 disables)
    FULL_N=1       lift the row caps (superconductivity 5,000; ct/blog/uji 10,000)
    DATASETS=a,b   restrict bench.py to named datasets

Row caps are seeded subsamples, following the diamonds cap of the main paper;
run the scalable methods once more with `FULL_N=1` for the full-n rows. At
full superconductivity n the search's per-depth proposal matrix is ~1.7 GB
(evaluated en bloc), which is the measured motivation for a chunked
evaluation in `beamfeat.search`.
