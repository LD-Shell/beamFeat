"""Apply the two source patches OpenFE needs to run on scikit-learn >= 1.6
and LightGBM 4.x.  Run once after `pip install openfe`:

    python patch_openfe.py

Patch 1: `mean_squared_error(..., squared=False)` was removed from
scikit-learn; replaced with `root_mean_squared_error`.
Patch 2: `init_score` passed to LightGBM 4 must be a 1-D array for
regression; OpenFE passes a pandas object whose shape LightGBM rejects.

(A third issue is handled in the harness itself, not here: OpenFE
auto-detects integer regression targets such as wine-quality ratings as
multiclass classification, so the harness passes `task="regression"`
explicitly, and uses `n_jobs=1` because OpenFE's multiprocessing path has
a further index-alignment bug with LightGBM 4.)
"""
import pathlib
import openfe

pkg = pathlib.Path(openfe.__file__).parent
for fname in ("openfe.py", "FeatureSelector.py"):
    p = pkg / fname
    src = p.read_text()
    orig = src
    src = src.replace(
        "from sklearn.metrics import mean_squared_error",
        "from sklearn.metrics import mean_squared_error, root_mean_squared_error",
    )
    src = src.replace(
        "mean_squared_error(label, pred, squared=False)",
        "root_mean_squared_error(label, pred)",
    )
    src = src.replace(
        "init_score=train_init,",
        "init_score=__import__('numpy').asarray(train_init).ravel(),",
    )
    src = src.replace(
        "eval_init_score=[val_init],",
        "eval_init_score=[__import__('numpy').asarray(val_init).ravel()],",
    )
    if src != orig:
        p.write_text(src)
        print(f"patched {p}")
    else:
        print(f"no changes needed in {p} (already patched?)")
print("done")
