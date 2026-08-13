"""Stability of selections across independent search/selection splits.

For each dataset: one fixed outer test split, then refits over S independent
internal splits. Reports held-out R^2 dispersion, pairwise Jaccard of the
selected sets computed up to algebraic value-equivalence (features whose
value vectors correlate above 0.999 on probe rows are the same feature), and
a per-class selection-frequency table. Synthetic problems with unambiguous
signal form the contrast set; the informative quantity is how stability
degrades from those to diffuse-signal real data.

Run:  python split_stability.py --splits 30 --out results/split_stability.json
      python split_stability.py --splits 6 --datasets three_way,tecator
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from beamfeat import BeamFeatTransformer

DATA = pathlib.Path(__file__).resolve().parent / "data"
INDEP = pathlib.Path(__file__).resolve().parent.parent / "independent" / "data"


def registry():
    ds = {}
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 6, (500, 4)); y = X[:, 0] * X[:, 1] / X[:, 2]
    ds["three_way"] = (X, y + rng.normal(0, .02 * y.std(), 500))
    X = rng.uniform(1, 6, (500, 12)); y = X[:, 0] * X[:, 1]
    ds["distractors25"] = (X, y + rng.normal(0, .25 * y.std(), 500))
    X = rng.uniform(0, 1, (800, 10))
    y = 10*np.sin(np.pi*X[:, 0]*X[:, 1]) + 20*(X[:, 2]-.5)**2 + 10*X[:, 3] + 5*X[:, 4]
    ds["friedman1"] = (X, y + rng.normal(0, 1, 800))
    from sklearn.datasets import load_diabetes
    d = load_diabetes(); ds["diabetes"] = (d.data, d.target)
    # the comparison study's other real datasets, when the sibling folder is present
    for name, fname, target in [("concrete", "data_concrete.csv", None),
                                 ("wine_red", "data_winequality_red.csv", None),
                                 ("boston", "data_housing_boston.csv", "medv")]:
        q = INDEP / fname
        if q.exists():
            df = pd.read_csv(q)
            if target:
                ds[name] = (df.drop(columns=target).to_numpy(float), df[target].to_numpy(float))
            else:
                ds[name] = (df.iloc[:, :-1].to_numpy(float), df.iloc[:, -1].to_numpy(float))
    for name, path, target, drop in [
        ("communities", "communities_crime_numeric.csv", -1, None),
        ("tecator", "tecator.csv", "fat", "nonchannel"),
        ("eyedata", "eyedata.csv", "trim32", None),
        ("riboflavin", "riboflavin.csv", 0, None),
        ("superconduct", "superconductivity.csv", "critical_temp", None),
    ]:
        p = DATA / path
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if name == "tecator":
            chan = [c for c in df.columns if c.startswith("_")]
            ds[name] = (df[chan].to_numpy(float), df["fat"].to_numpy(float))
        elif name == "superconduct":
            df = df.sample(min(len(df), 5000), random_state=0)
            ds[name] = (df.drop(columns=target).to_numpy(float), df[target].to_numpy(float))
        elif isinstance(target, str):
            ds[name] = (df.drop(columns=target).to_numpy(float), df[target].to_numpy(float))
        elif target == -1:
            ds[name] = (df.iloc[:, :-1].to_numpy(float), df.iloc[:, -1].to_numpy(float))
        else:
            ds[name] = (df.iloc[:, 1:].to_numpy(float), df.iloc[:, 0].to_numpy(float))
    return ds


def equivalence_classes(models, X_probe, tol=0.999):
    """Assign every selected feature (across all fits) to an equivalence class
    by value-vector correlation on probe rows. Returns per-model class sets and
    a representative formula per class."""
    feats, owners, names = [], [], []
    for i, m in enumerate(models):
        try:
            F = m.transform(X_probe)
        except Exception:
            continue
        for j, f in enumerate(m.formulas()):
            col = np.asarray(F[:, j], float)
            if np.isfinite(col).sum() < 50 or np.nanstd(col) == 0:
                continue
            feats.append(col); owners.append(i); names.append(f)
    classes = []          # list of (representative_col, representative_name)
    member_of = []        # class index per feature
    with np.errstate(all="ignore"):
        for col in feats:
            hit = None
            for ci, (rep, _) in enumerate(classes):
                m = np.isfinite(col) & np.isfinite(rep)
                if m.sum() >= 50 and abs(np.corrcoef(col[m], rep[m])[0, 1]) > tol:
                    hit = ci; break
            if hit is None:
                classes.append((col, names[len(member_of)]))
                hit = len(classes) - 1
            member_of.append(hit)
    per_model = [set() for _ in models]
    for owner, ci in zip(owners, member_of):
        per_model[owner].add(ci)
    freq = {classes[ci][1]: sum(ci in s for s in per_model) for ci in range(len(classes))}
    return per_model, freq


def main(a):
    ds = registry()
    if a.datasets:
        keep = a.datasets.split(",")
        ds = {k: v for k, v in ds.items() if k in keep}
    out = {}
    for name, (X, y) in ds.items():
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
        probe = Xtr[np.random.default_rng(1).choice(len(Xtr), min(len(Xtr), 1500), replace=False)]
        models, scores, nsel = [], [], []
        for s in range(a.splits):
            m = BeamFeatTransformer(random_state=s).fit(Xtr, ytr)
            Ftr, Fte = m.transform(Xtr), m.transform(Xte)
            if Ftr is None or Ftr.size == 0:      # no discoveries: mean predictor
                r2 = 1.0 - np.sum((yte - ytr.mean()) ** 2) / np.sum((yte - yte.mean()) ** 2)
            else:
                lm = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 4, 25)))
                r2 = float(lm.fit(Ftr, ytr).score(Fte, yte))
            models.append(m); scores.append(r2)
            nsel.append(len(m.formulas()))
            print(f"{name:14s} split {s:2d} R2={scores[-1]:+.3f} n={nsel[-1]}", flush=True)
        per_model, freq = equivalence_classes(models, probe)
        jac = [len(p & q_) / max(len(p | q_), 1)
               for p, q_ in itertools.combinations(per_model, 2)] or [1.0]
        out[name] = dict(
            splits=a.splits,
            r2_mean=float(np.mean(scores)), r2_std=float(np.std(scores)),
            r2_min=float(np.min(scores)), r2_max=float(np.max(scores)),
            n_selected=nsel,
            jaccard_mean=float(np.mean(jac)), jaccard_min=float(np.min(jac)),
            n_classes=len(freq),
            stable_features={k: v for k, v in sorted(freq.items(), key=lambda kv: -kv[1])
                             if v >= max(2, a.splits // 2)},
            frequency_table=dict(sorted(freq.items(), key=lambda kv: -kv[1])[:40]),
        )
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)
        r = out[name]
        print(f"== {name}: R2 {r['r2_mean']:.3f}+/-{r['r2_std']:.3f} "
              f"[{r['r2_min']:.3f},{r['r2_max']:.3f}] "
              f"Jaccard(eq) {r['jaccard_mean']:.2f} "
              f"classes {r['n_classes']} stable {len(r['stable_features'])}", flush=True)
    print("wrote", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=30)
    ap.add_argument("--datasets", default="", help="comma list; empty = all present")
    ap.add_argument("--out", default="results/split_stability.json")
    a = ap.parse_args()
    main(a)
