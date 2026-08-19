# Provenance

What in this study is reproducible, and what is not. Read this before citing
any figure from it.

## What was run

360 fits: nine datasets, eight methods, five fixed 75/25 splits, executed in
one pass on one machine in the pinned environment of `requirements.txt`
(Python 3.11, numpy 1.26.4, scikit-learn 1.7.2, autofeat 2.1.3, openfe 0.0.12,
featuretools 1.31.0, lightgbm 4.7.0). Raw per-fit records are in
`results_as_reported/`; `independent_benchmark_results.csv` is their
aggregation. `beamfeat_benchmark.ipynb` reproduces the analysis in about two
minutes and the fits in 45 to 70.

An earlier version of this study was assembled from several partial runs on an
unknown machine and contained one value transcribed by hand rather than
recorded. It has been replaced entirely by the single-pass run described here.

## What reproduces

Every method except `autofeat`. Across four executions of this comparison,
`beamfeat`, `beamfeat_ridge`, `ridge_raw`, `rf_raw`, `lgbm_raw`, `openfe` and
`featuretools` returned the same means, worst cases and paired-test statistics
to four decimals. `beamfeat` is deterministic given a seed and reproduced
digit-for-digit across two machines running different Python, numpy and
scikit-learn versions.

## What does not

`autofeat`. It exposes no `random_state`, and its noise-injection screen draws
decoy features from the global NumPy generator before its own internal seeding
applies, so every process starts from different entropy. The decoys set the
bar each candidate feature must clear, so a different draw changes which
features survive.

Six runs of one identical split of Friedman #1, at the library defaults in
this environment, each a separate process (`autofeat_repeatability.json`):

| run | threading | R² | selected features |
|---|---|---|---|
| 1 | default | +0.952 | 17 |
| 2 | default | −100.96 | 26 |
| 3 | default | −105.31 | 19 |
| 4 | pinned to 1 | −109.79 | 21 |
| 5 | pinned to 1 | +0.950 | 18 |
| 6 | pinned to 1 | +0.939 | 20 |

Calling `np.random.seed()` before fitting does not control it, and pinning
`OMP_NUM_THREADS`, `NUMBA_NUM_THREADS` and `MKL_NUM_THREADS` to 1 does not
either. The effect appears where selection is marginal; on problems with an
unambiguous signal, repeated processes agree exactly.

The same instability shows at study level. Four executions of the full
nine-dataset comparison returned `autofeat` mean R² of 0.746, −1.694, 0.754
and −1.561, with worst single fits from −2.28 to −108.9.

**Consequence for citation.** `autofeat`'s row in `REPORT.md` is one draw from
that distribution, not a measurement. The shipped run gives mean −1.561 and a
worst fit of −103.245 (Friedman #1 split 0; its other four splits on that
dataset fall between 0.938 and 0.955). A rerun will give something else. Any
single `autofeat` figure here is one draw; the instability is the citable
finding.

## Data

`data/` holds the three CSVs that are not bundled with scikit-learn, with
MD5 checksums in `CHECKSUMS.md5`. Verify from this directory:

```bash
md5sum -c data/CHECKSUMS.md5
```

Boston housing is included for comparability with autofeat's published table,
despite its removal from scikit-learn over documented ethical concerns. The
repository's own real-data panel contains no contested dataset.

## Protocol limits

- Nine datasets leaves the Friedman omnibus underpowered; the paired tests
  carry the inference.
- Five splits per dataset.
- Default or paper-recommended settings only, with no hyperparameter search.
- `OpenFE` is run into a linear model rather than the gradient-boosted setting
  it is designed for, which understates it.
- The wine dataset here is the red-only variant (1,598 rows as archived: the file ships headerless and the loader formerly consumed UCI's first observation as a header; it now reads all 1,599, so fresh runs will differ by one row from every archived result; 11 columns; autofeat's
  published table used red and white combined (6497 x 12).
- `recovered` measures column membership, not symbolic form: it asks whether
  some returned feature references all of the generating columns, not whether
  the operators combining them are right. See the docstring in `bench.py`.
