"""Friedman #1 gap decomposition — representation oracle vs screening ceiling.

Reproduces the published boundary numbers: a least-squares oracle on the
depth-2 basis {ab, (ab)^2, c, c^2, d, e} (0.964), the screening-admissible
ceiling under the marginal null (BY admits {ab, (ab)^2, d, e}: 0.875), and
the default pipeline on the same split.
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")


def friedman1(seed: int, n: int = 800, d: int = 10):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, d))
    y = (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + 5 * X[:, 4]
        + rng.normal(0, 1, n)
    )
    return X, y


def main() -> dict:
    from beamfeat import BeamFeatRegressor
    from beamfeat.selection import PermutationSelector

    X, y = friedman1(0)
    X_test, y_test = friedman1(1)

    def basis(A):
        u = A[:, 0] * A[:, 1]
        return np.column_stack([u, u**2, A[:, 2], A[:, 2] ** 2, A[:, 3], A[:, 4]])

    names = ["ab", "(ab)^2", "c", "c^2", "d", "e"]
    F, F_test = basis(X), basis(X_test)

    def fit_r2(indices):
        design = np.column_stack([np.ones(len(y)), F[:, indices]])
        test = np.column_stack([np.ones(len(y_test)), F_test[:, indices]])
        weights, *_ = np.linalg.lstsq(design, y, rcond=None)
        return float(1 - np.var(y_test - test @ weights) / np.var(y_test))

    oracle = fit_r2(list(range(6)))

    holdout = np.random.default_rng(0).permutation(len(y))[:400]
    admitted = PermutationSelector(
        target_fdr=0.1, correction="by", random_state=0
    ).select(F[holdout], y[holdout]).selected
    ceiling = fit_r2(list(admitted))

    achieved = (
        BeamFeatRegressor(max_depth=2, beam_width=40, random_state=0)
        .fit(X, y)
        .score(X_test, y_test)
    )

    print(f"representation oracle (all six terms):        R^2 {oracle:.3f}")
    print(f"screening-admissible ceiling (BY admits {[names[i] for i in admitted]}): R^2 {ceiling:.3f}")
    print(f"default pipeline on this split:               R^2 {achieved:.3f}")
    return {"oracle": oracle, "ceiling": ceiling, "achieved": achieved,
            "admitted": [names[i] for i in admitted]}


if __name__ == "__main__":
    main()
