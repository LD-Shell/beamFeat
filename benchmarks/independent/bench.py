"""Independent benchmark: beamfeat vs autofeat vs featuretools-DFS vs OpenFE
vs raw-feature baselines (Ridge, RandomForest, LightGBM).

Protocol (following Horn et al. 2019 / standard tabular practice):
- Fixed 75/25 train/test splits, N_SPLITS seeds, identical across methods.
- Every feature-construction tool feeds the SAME downstream model
  (RidgeCV on standardized features) so the comparison isolates the
  engineered features, not the estimator.  beamfeat's own internal linear
  fit is equivalent by design; both its native predict and
  transform->RidgeCV are recorded.
- Metrics: out-of-sample R^2 (mean +/- std over splits), wall-clock fit
  time, number of constructed features, and (synthetic only) recovery of
  the generating formula.
"""
from __future__ import annotations
import json, os, pathlib, sys, time, warnings, traceback
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

DATA = pathlib.Path(__file__).resolve().parent / "data"

N_SPLITS = 5
ALPHAS = np.logspace(-4, 4, 25)

# ---------------------------------------------------------------- datasets
def get_datasets(which):
    rng = np.random.default_rng(0)
    ds = {}
    if which in ("real", "all"):
        c = pd.read_csv(DATA / "data_concrete.csv")
        ds["concrete"] = (c.iloc[:, :-1].to_numpy(float), c.iloc[:, -1].to_numpy(float), None)
        w = pd.read_csv(DATA / "data_winequality_red.csv")
        ds["wine_red"] = (w.iloc[:, :-1].to_numpy(float), w.iloc[:, -1].to_numpy(float), None)
        b = pd.read_csv(DATA / "data_housing_boston.csv")
        yb = b["medv"].to_numpy(float); Xb = b.drop(columns=["medv"]).to_numpy(float)
        ds["boston"] = (Xb, yb, None)
        d = load_diabetes()
        ds["diabetes"] = (d.data, d.target, None)
    if which in ("synthetic", "all"):
        n = 500
        X = rng.uniform(1, 6, (n, 4)); y = X[:,0]*X[:,1]/X[:,2]
        ds["three_way:a*b/c"] = (X, y + rng.normal(0, .02*np.std(y), n), ("x0","x1","x2"))
        X = rng.uniform(1, 6, (n, 4)); y = 0.5*X[:,0]*X[:,1]**2
        ds["kinetic:m*v^2/2"] = (X, y + rng.normal(0, .02*np.std(y), n), ("x0","x1"))
        X = rng.uniform(1, 6, (n, 10)); y = X[:,0]*X[:,1]
        ds["sparse10:a*b"] = (X, y + rng.normal(0, .05*np.std(y), n), ("x0","x1"))
        X = rng.uniform(1, 6, (n, 4)); y = 3*X[:,0]-2*X[:,1]+X[:,2]
        ds["linear_ctrl"] = (X, y + rng.normal(0, .02*np.std(y), n), ())
        X = rng.uniform(0, 1, (800, 10))
        y = 10*np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-.5)**2 + 10*X[:,3] + 5*X[:,4] + rng.normal(0,1,800)
        ds["friedman1"] = (X, y, None)
    return ds

def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))

def clean(M):
    M = np.asarray(M, float)
    M[~np.isfinite(M)] = np.nan
    col_ok = ~np.all(np.isnan(M), axis=0)
    M = M[:, col_ok]
    med = np.nanmedian(M, axis=0)
    idx = np.where(np.isnan(M))
    M[idx] = np.take(med, idx[1])
    return M, col_ok

# ---------------------------------------------------------------- methods
def run_ridge(Xtr, ytr, Xte):
    m = ridge().fit(Xtr, ytr); return m.predict(Xte), {}

def run_rf(Xtr, ytr, Xte):
    m = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1).fit(Xtr, ytr)
    return m.predict(Xte), {}

def run_lgbm(Xtr, ytr, Xte):
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=300, random_state=0, verbose=-1).fit(Xtr, ytr)
    return m.predict(Xte), {}

def run_autofeat(Xtr, ytr, Xte):
    from autofeat import AutoFeatRegressor
    af = AutoFeatRegressor(feateng_steps=2, verbose=0, n_jobs=1)
    Ftr = af.fit_transform(pd.DataFrame(Xtr, columns=[f"x{i:03d}" for i in range(Xtr.shape[1])]), ytr)
    Fte = af.transform(pd.DataFrame(Xte, columns=[f"x{i:03d}" for i in range(Xte.shape[1])]))
    new = [c for c in Ftr.columns if c not in {f"x{i:03d}" for i in range(Xtr.shape[1])}]
    m = ridge().fit(Ftr.to_numpy(float), ytr)
    return m.predict(Fte.to_numpy(float)), {"n_new": len(new), "formulas": new[:8]}

def run_featuretools(Xtr, ytr, Xte):
    import featuretools as ft
    cols = [f"x{i}" for i in range(Xtr.shape[1])]
    prim = ["multiply_numeric", "divide_numeric"]
    def dfs(X):
        df = pd.DataFrame(X, columns=cols); df["_id"] = range(len(df))
        es = ft.EntitySet("d")
        es = es.add_dataframe(dataframe_name="t", dataframe=df, index="_id")
        F, defs = ft.dfs(entityset=es, target_dataframe_name="t",
                         trans_primitives=prim, max_depth=1, verbose=False)
        return F.reindex(sorted(F.columns), axis=1)
    Ftr, Fte = dfs(Xtr), dfs(Xte)
    Mtr, ok = clean(Ftr.to_numpy())
    Mte = np.asarray(Fte.to_numpy(), float)[:, ok]
    Mte[~np.isfinite(Mte)] = 0.0
    m = ridge().fit(Mtr, ytr)
    return m.predict(Mte), {"n_new": Mtr.shape[1] - Xtr.shape[1]}

def run_openfe(Xtr, ytr, Xte):
    from openfe import OpenFE, transform
    cols = [f"x{i}" for i in range(Xtr.shape[1])]
    dtr = pd.DataFrame(Xtr, columns=cols); dte = pd.DataFrame(Xte, columns=cols)
    ofe = OpenFE()
    feats = ofe.fit(data=dtr, label=pd.Series(ytr.astype(float)), task="regression", n_jobs=1, verbose=False)
    ttr, tte = transform(dtr, dte, feats[:20], n_jobs=1)
    Mtr, ok = clean(ttr.to_numpy()); Mte = np.asarray(tte.to_numpy(), float)[:, ok]
    Mte[~np.isfinite(Mte)] = 0.0
    m = ridge().fit(Mtr, ytr)
    return m.predict(Mte), {"n_new": ttr.shape[1] - Xtr.shape[1]}

def run_beamfeat(Xtr, ytr, Xte):
    from beamfeat import BeamFeatRegressor
    m = BeamFeatRegressor(random_state=0).fit(Xtr, ytr)
    info = {"n_new": len(m.formulas()), "formulas": m.formulas()[:8],
            "fdr": bool(getattr(m, "fdr_controlled_", False))}
    return m.predict(Xte), info

METHODS = {"ridge_raw": run_ridge, "rf_raw": run_rf, "lgbm_raw": run_lgbm,
           "autofeat": run_autofeat, "featuretools": run_featuretools,
           "openfe": run_openfe, "beamfeat": run_beamfeat}

def recovered(formulas, tokens):
    if not tokens or not formulas: return None
    toks = [t.replace("x", "") for t in tokens]  # x0 -> 0
    for f in formulas:
        f2 = f.replace("x00", "x").replace("x0", "x") if "x00" in f else f
        names = set()
        import re
        for mnum in re.findall(r"x0*(\d+)", f):
            names.add(mnum)
        if set(toks) <= names or set(toks) == names:
            return True
    return False

def main(which, methods, n_splits, out):
    import os
    ds = get_datasets(which)
    only = os.environ.get("DATASETS")
    if only:
        keep = only.split(",")
        ds = {k: v for k, v in ds.items() if k in keep}
    rows = []
    for name, (X, y, tokens) in ds.items():
        for split in range(n_splits):
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=split)
            for mname in methods:
                fn = METHODS[mname]
                t0 = time.perf_counter()
                try:
                    pred, info = fn(Xtr, ytr, Xte)
                    dt = time.perf_counter() - t0
                    r2 = float(r2_score(yte, pred))
                    rec = recovered(info.get("formulas", []), tokens) if tokens is not None else None
                    rows.append(dict(dataset=name, method=mname, split=split, r2=r2,
                                     seconds=dt, n_new=info.get("n_new"),
                                     recovered=rec, fdr=info.get("fdr"),
                                     formulas=info.get("formulas"), error=None))
                    print(f"{name:16s} {mname:13s} s{split} R2={r2:+.4f} {dt:7.1f}s n_new={info.get('n_new')}", flush=True)
                except Exception as e:
                    dt = time.perf_counter() - t0
                    rows.append(dict(dataset=name, method=mname, split=split, r2=None,
                                     seconds=dt, n_new=None, recovered=None, fdr=None,
                                     formulas=None, error=f"{type(e).__name__}: {e}"))
                    print(f"{name:16s} {mname:13s} s{split} ERROR {type(e).__name__}: {str(e)[:80]}", flush=True)
        json.dump(rows, open(out, "w"), indent=1)
    json.dump(rows, open(out, "w"), indent=1)
    print("wrote", out)

if __name__ == "__main__":
    which = sys.argv[1]
    methods = sys.argv[2].split(",")
    n_splits = int(sys.argv[3])
    out = sys.argv[4]
    main(which, methods, n_splits, out)
