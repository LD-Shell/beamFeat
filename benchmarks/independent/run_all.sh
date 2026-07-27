#!/usr/bin/env bash
# Full reproduction of the beamfeat-vs-competitors study.
# Total runtime: roughly 45-70 minutes on a modern 4-core machine
# (autofeat dominates; everything else finishes in ~10 minutes).
# Each step writes its own results_*.json, so the run is resumable:
# delete a json to redo that step, or comment out completed lines.
set -euo pipefail

# Fast methods, all datasets, 5 seeds (~10 min)
python bench.py real      ridge_raw,rf_raw,lgbm_raw,featuretools,beamfeat,openfe 5 results_real_fast.json
python bench.py synthetic ridge_raw,rf_raw,lgbm_raw,featuretools,beamfeat,openfe 5 results_syn_fast.json

# autofeat, chunked by dataset because each fit takes 10-130 s
DATASETS=concrete,diabetes                              python bench.py real      autofeat 5 results_real_af1.json
DATASETS=wine_red                                       python bench.py real      autofeat 5 results_real_af2.json
DATASETS=boston                                         python bench.py real      autofeat 5 results_real_af3.json
DATASETS="three_way:a*b/c,kinetic:m*v^2/2,linear_ctrl"  python bench.py synthetic autofeat 5 results_syn_af1.json
DATASETS="sparse10:a*b"                                 python bench.py synthetic autofeat 5 results_syn_af2.json
DATASETS=friedman1                                      python bench.py synthetic autofeat 5 results_syn_af3.json

# Tables, ranks, Friedman test, paired Wilcoxon tests, tidy CSV
python aggregate.py
