"""Multi-split aggregation of beamfeat selections.

A single search/selection split yields a valid but split-dependent set of
certified features when many candidates are informationally interchangeable.
This module fits the pipeline over many independent internal splits, pools
the selected features into equivalence classes by value-vector correlation on
probe rows, and reports each class's selection frequency, keeping those above
a threshold.

Scope. Each individual split's selections carry the per-split FDR guarantee
of Proposition 1; the frequency-thresholded aggregate is a
stability-selection-style summary in the spirit of Meinshausen and Buhlmann
(2010) and inherits error control only under that literature's conditions --
it is not a multi-split FDR guarantee. Formal p-value combination across
splits (Meinshausen, Meier and Buhlmann, 2009) is future work.

Usage:

    from multisplit import MultiSplitBeamFeat
    ms = MultiSplitBeamFeat(n_splits=20, threshold=0.5, random_state=0).fit(X, y)
    ms.report()               # frequency table over equivalence classes
    ms.stable_formulas_       # representatives selected in >= threshold of splits
    F = ms.transform(X_new)   # values of the stable representatives

    python multisplit.py data/tecator.csv fat --splits 20
"""
from __future__ import annotations

import sys

import numpy as np

from beamfeat import BeamFeatTransformer


class MultiSplitBeamFeat:
    def __init__(self, n_splits: int = 20, threshold: float = 0.5,
                 random_state: int = 0, probe_rows: int = 1500,
                 equivalence_tol: float = 0.999, **beamfeat_kwargs):
        self.n_splits = n_splits
        self.threshold = threshold
        self.random_state = random_state
        self.probe_rows = probe_rows
        self.equivalence_tol = equivalence_tol
        self.beamfeat_kwargs = beamfeat_kwargs

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        rng = np.random.default_rng(self.random_state)
        probe = X[rng.choice(len(X), min(len(X), self.probe_rows), replace=False)]

        self.models_ = []
        owner_feature, columns, names = [], [], []
        for s in range(self.n_splits):
            m = BeamFeatTransformer(random_state=self.random_state + s,
                                    **self.beamfeat_kwargs).fit(X, y)
            self.models_.append(m)
            try:
                F = m.transform(probe)
            except Exception:
                continue
            for j, formula in enumerate(m.formulas()):
                col = np.asarray(F[:, j], float)
                if np.isfinite(col).sum() < 50 or np.nanstd(col) == 0:
                    continue
                owner_feature.append((s, j))
                columns.append(col)
                names.append(formula)

        # equivalence classes by |corr| on probe rows
        reps: list[np.ndarray] = []
        class_of: list[int] = []
        with np.errstate(all="ignore"):
            for col in columns:
                hit = None
                for ci, rep in enumerate(reps):
                    mk = np.isfinite(col) & np.isfinite(rep)
                    if mk.sum() >= 50 and abs(np.corrcoef(col[mk], rep[mk])[0, 1]) > self.equivalence_tol:
                        hit = ci
                        break
                if hit is None:
                    reps.append(col)
                    hit = len(reps) - 1
                class_of.append(hit)

        n_classes = len(reps)
        split_sets = [set() for _ in range(self.n_splits)]
        rep_owner = {}
        rep_name = {}
        for (s, j), ci, nm in zip(owner_feature, class_of, names):
            split_sets[s].add(ci)
            rep_owner.setdefault(ci, (s, j))
            rep_name.setdefault(ci, nm)

        self.frequencies_ = {rep_name[ci]: sum(ci in ss for ss in split_sets) / self.n_splits
                             for ci in range(n_classes)}
        self.stable_classes_ = [ci for ci in range(n_classes)
                                if sum(ci in ss for ss in split_sets) / self.n_splits >= self.threshold]
        self.stable_formulas_ = [rep_name[ci] for ci in self.stable_classes_]
        self._stable_owner = [rep_owner[ci] for ci in self.stable_classes_]
        return self

    def transform(self, X):
        if not self._stable_owner:
            return np.empty((len(X), 0))
        cols = []
        for s, j in self._stable_owner:
            cols.append(np.asarray(self.models_[s].transform(X), float)[:, j])
        return np.column_stack(cols)

    def report(self, top: int = 25) -> str:
        lines = [f"{f:6.2f}  {name}" for name, f in
                 sorted(self.frequencies_.items(), key=lambda kv: -kv[1])[:top]]
        head = (f"{len(self.frequencies_)} equivalence classes over {self.n_splits} splits; "
                f"{len(self.stable_formulas_)} at frequency >= {self.threshold}\n")
        return head + "\n".join(lines)


if __name__ == "__main__":
    import pandas as pd
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    path, target = sys.argv[1], sys.argv[2]
    splits = int(sys.argv[sys.argv.index("--splits") + 1]) if "--splits" in sys.argv else 20
    df = pd.read_csv(path)
    if target == "fat" and any(c.startswith(("_", "absorbance")) for c in df.columns):
        X = df[[c for c in df.columns
                if c.startswith(("_", "absorbance"))]].to_numpy(float)
    else:
        X = df.drop(columns=target).to_numpy(float)
    y = df[target].to_numpy(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    ms = MultiSplitBeamFeat(n_splits=splits).fit(Xtr, ytr)
    print(ms.report())
    F = ms.transform(Xtr)
    if F.shape[1]:
        lm = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 4, 25))).fit(F, ytr)
        print(f"ridge on {F.shape[1]} stable features, held-out R2: "
              f"{lm.score(ms.transform(Xte), yte):.3f}")
    else:
        print("no features reached the stability threshold")
