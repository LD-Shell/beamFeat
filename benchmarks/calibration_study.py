"""End-to-end FDR calibration study — the paper's headline figures.

200 signal replicates at nominal FDR 0.10 (empirical FDR, power, fallback
count) and 60 global-null replicates (total selections). Seeds are fixed;
the run regenerates the published numbers exactly.
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")


def main(n_signal: int = 200, n_null: int = 60) -> dict:
    from beamfeat import BeamFeatRegressor

    fdps, recovered, fallbacks = [], 0, 0
    for trial in range(n_signal):
        rng = np.random.default_rng(10_000 + trial)
        X = rng.uniform(1, 6, (500, 6))
        signal = X[:, 0] * X[:, 1]
        y = signal + rng.normal(0, 0.05 * np.std(signal), 500)
        model = BeamFeatRegressor(
            max_depth=2, beam_width=25, target_fdr=0.1, random_state=trial
        ).fit(X, y)
        if not model.fdr_controlled_:
            fallbacks += 1
            continue
        relevant = {"x0", "x1"}
        fdps.append(
            sum(1 for node in model.features_ if not (node.columns() & relevant))
            / len(model.features_)
        )
        recovered += any("x0" in f and "x1" in f for f in model.formulas())

    null_selected = 0
    for trial in range(n_null):
        rng = np.random.default_rng(50_000 + trial)
        X = rng.uniform(1, 6, (500, 6))
        y = rng.normal(0, 1, 500)
        model = BeamFeatRegressor(
            max_depth=2, beam_width=25, target_fdr=0.1, random_state=trial
        ).fit(X, y)
        null_selected += model.n_features_out_ if model.fdr_controlled_ else 0

    result = {
        "empirical_fdr": float(np.mean(fdps)),
        "power": recovered / len(fdps),
        "fallbacks": fallbacks,
        "null_selections": null_selected,
    }
    print(
        f"SIGNAL ({n_signal} replicates, nominal 0.10): "
        f"empirical FDR {result['empirical_fdr']:.4f} | power {result['power']:.3f} "
        f"| fallbacks {result['fallbacks']}"
    )
    print(f"GLOBAL NULL ({n_null} replicates): selections {result['null_selections']}")
    return result


if __name__ == "__main__":
    main()
