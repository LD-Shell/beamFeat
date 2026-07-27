"""Tests for the pluggable scoring strategies."""

from __future__ import annotations

import numpy as np
import pytest

from beamfeat.scoring import (
    CorrelationScorer,
    GradientBoostingScorer,
    MutualInformationScorer,
    Scorer,
    make_scorer,
)

ALL_SCORERS = ["correlation", "mutual_information", "gradient_boosting"]


@pytest.fixture
def rng():
    return np.random.default_rng(20260723)


def _fast(name: str, problem_type: str = "regression") -> Scorer:
    """Construct a scorer with cheap settings so tests stay quick."""
    if name == "gradient_boosting":
        return GradientBoostingScorer(
            problem_type=problem_type, n_folds=2, max_iter=15, max_depth=2, subsample_size=300
        )
    return make_scorer(name, problem_type=problem_type)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


class TestFactory:
    @pytest.mark.parametrize("name", ALL_SCORERS)
    def test_resolves_canonical_names(self, name):
        assert make_scorer(name).name == name

    @pytest.mark.parametrize(
        ("alias", "expected"),
        [
            ("corr", "correlation"),
            ("pearson", "correlation"),
            ("mi", "mutual_information"),
            ("mutual_info", "mutual_information"),
            ("gb", "gradient_boosting"),
            ("boosting", "gradient_boosting"),
        ],
    )
    def test_resolves_aliases(self, alias, expected):
        assert make_scorer(alias).name == expected

    def test_case_insensitive(self):
        assert make_scorer("Correlation").name == "correlation"
        assert make_scorer("MI").name == "mutual_information"

    def test_instance_passes_through(self):
        scorer = CorrelationScorer()
        assert make_scorer(scorer) is scorer

    def test_unknown_name_lists_options(self):
        with pytest.raises(ValueError, match="unknown scorer"):
            make_scorer("telepathy")

    def test_wrong_type_rejected(self):
        with pytest.raises(TypeError):
            make_scorer(42)

    def test_bad_problem_type_rejected(self):
        with pytest.raises(ValueError, match="problem_type"):
            make_scorer("correlation", problem_type="clustering")

    def test_kwargs_forwarded(self):
        scorer = make_scorer("mi", n_neighbors=7)
        assert scorer.n_neighbors == 7


class TestConstructorValidation:
    def test_mi_rejects_bad_neighbors(self):
        with pytest.raises(ValueError, match="n_neighbors"):
            MutualInformationScorer(n_neighbors=0)

    def test_gb_rejects_bad_folds(self):
        with pytest.raises(ValueError, match="n_folds"):
            GradientBoostingScorer(n_folds=1)

    def test_gb_rejects_bad_max_iter(self):
        with pytest.raises(ValueError, match="max_iter"):
            GradientBoostingScorer(max_iter=0)


# --------------------------------------------------------------------------- #
# Contract obeyed by every scorer
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ALL_SCORERS)
class TestScorerContract:
    def test_returns_one_score_per_candidate(self, name, rng):
        scorer = _fast(name)
        candidates = rng.normal(size=(200, 4))
        target = candidates[:, 0] * 2.0 + rng.normal(0, 0.1, 200)
        scores = scorer.score_batch(candidates, target)
        assert scores.shape == (4,)

    def test_scores_are_finite_and_non_negative(self, name, rng):
        scorer = _fast(name)
        candidates = rng.normal(size=(200, 5))
        target = candidates[:, 1] + rng.normal(0, 0.5, 200)
        scores = scorer.score_batch(candidates, target)
        assert np.all(np.isfinite(scores))
        assert np.all(scores >= 0)

    def test_ranks_signal_above_noise(self, name, rng):
        """The core requirement: a real predictor must outrank pure noise."""
        n = 400
        signal = rng.normal(size=n)
        target = 3.0 * signal + rng.normal(0, 0.2, n)
        candidates = np.column_stack([rng.normal(size=n), signal, rng.normal(size=n)])
        scores = _fast(name).score_batch(candidates, target)
        assert int(np.argmax(scores)) == 1, f"{name} failed to rank the true signal first"

    def test_single_candidate_convenience(self, name, rng):
        n = 200
        signal = rng.normal(size=n)
        target = 2.0 * signal + rng.normal(0, 0.2, n)
        assert _fast(name).score(signal, target) > 0

    def test_deterministic_across_calls(self, name, rng):
        scorer = _fast(name)
        candidates = rng.normal(size=(200, 3))
        target = candidates[:, 0] + rng.normal(0, 0.3, 200)
        np.testing.assert_allclose(
            scorer.score_batch(candidates, target), scorer.score_batch(candidates, target)
        )

    def test_deterministic_across_instances(self, name, rng):
        candidates = rng.normal(size=(200, 3))
        target = candidates[:, 0] + rng.normal(0, 0.3, 200)
        np.testing.assert_allclose(
            _fast(name).score_batch(candidates, target),
            _fast(name).score_batch(candidates, target),
        )

    def test_1d_candidate_accepted(self, name, rng):
        n = 200
        signal = rng.normal(size=n)
        scores = _fast(name).score_batch(signal, 2.0 * signal + rng.normal(0, 0.2, n))
        assert scores.shape == (1,)

    def test_row_mismatch_rejected(self, name, rng):
        with pytest.raises(ValueError, match="rows"):
            _fast(name).score_batch(rng.normal(size=(100, 2)), rng.normal(size=50))

    def test_empty_incumbent_treated_as_none(self, name, rng):
        n = 200
        signal = rng.normal(size=n)
        target = 2.0 * signal + rng.normal(0, 0.2, n)
        scorer = _fast(name)
        candidates = signal.reshape(-1, 1)
        with_empty = scorer.score_batch(candidates, target, np.empty((n, 0)))
        without = scorer.score_batch(candidates, target, None)
        np.testing.assert_allclose(with_empty, without)

    def test_classification_supported(self, name, rng):
        n = 300
        signal = rng.normal(size=n)
        labels = (signal > 0).astype(int)
        candidates = np.column_stack([rng.normal(size=n), signal])
        scores = _fast(name, problem_type="classification").score_batch(candidates, labels)
        assert scores.shape == (2,)
        assert scores[1] > scores[0]


# --------------------------------------------------------------------------- #
# Marginal scoring: the anti-duplication property
# --------------------------------------------------------------------------- #


class TestMarginalScoring:
    def test_correlation_discounts_incumbent_duplicate(self, rng):
        """A candidate identical to an incumbent must lose its value."""
        n = 400
        signal = rng.normal(size=n)
        target = 3.0 * signal + rng.normal(0, 0.2, n)
        scorer = CorrelationScorer()
        alone = scorer.score(signal, target)
        given = scorer.score(signal, target, incumbent=signal.reshape(-1, 1))
        assert alone > 0.8
        assert given < 0.1

    def test_correlation_keeps_orthogonal_candidate(self, rng):
        """A genuinely new signal must survive residualisation."""
        n = 400
        first = rng.normal(size=n)
        second = rng.normal(size=n)
        target = 2.0 * first + 2.0 * second + rng.normal(0, 0.2, n)
        scorer = CorrelationScorer()
        given = scorer.score(second, target, incumbent=first.reshape(-1, 1))
        assert given > 0.5

    def test_gb_discounts_incumbent_duplicate(self, rng):
        n = 400
        signal = rng.normal(size=n)
        target = 3.0 * signal + rng.normal(0, 0.2, n)
        scorer = GradientBoostingScorer(n_folds=2, max_iter=20, max_depth=2, subsample_size=400)
        alone = scorer.score(signal, target)
        given = scorer.score(signal, target, incumbent=signal.reshape(-1, 1))
        assert alone > given

    def test_saturated_incumbent_scores_zero(self, rng):
        """When the incumbent explains the target exactly, nothing adds value."""
        n = 200
        signal = rng.normal(size=n)
        target = signal.copy()
        scores = CorrelationScorer().score_batch(rng.normal(size=(n, 3)), target, signal.reshape(-1, 1))
        assert np.all(scores < 1e-6)


# --------------------------------------------------------------------------- #
# Where the scorers genuinely differ
# --------------------------------------------------------------------------- #


class TestScorerDifferentiation:
    def test_mi_detects_nonmonotone_dependence(self, rng):
        """The case that justifies MI's cost: correlation is blind here."""
        n = 600
        feature = rng.uniform(-3, 3, n)
        target = feature**2 + rng.normal(0, 0.3, n)  # symmetric, so corr ~ 0
        noise = rng.normal(size=n)
        candidates = np.column_stack([feature, noise])

        corr_scores = CorrelationScorer().score_batch(candidates, target)
        mi_scores = MutualInformationScorer().score_batch(candidates, target)

        assert corr_scores[0] < 0.2, "correlation should be near-blind to a symmetric relationship"
        assert mi_scores[0] > mi_scores[1], "MI should detect the quadratic dependence"

    def test_correlation_suffices_when_transform_is_explicit(self, rng):
        """Why correlation remains a sensible default in this library.

        A nonlinearity already encoded in the expression is visible to a linear
        scorer applied to the transformed column.
        """
        n = 400
        raw = rng.uniform(-3, 3, n)
        target = raw**2 + rng.normal(0, 0.3, n)
        transformed = raw**2  # what the DAG would hand the scorer
        assert CorrelationScorer().score(transformed, target) > 0.8

    def test_correlation_is_faster_than_alternatives(self, rng):
        """Documents the cost ordering the docstrings claim."""
        import time

        n, k = 500, 30
        candidates = rng.normal(size=(n, k))
        target = candidates[:, 0] + rng.normal(0, 0.3, n)

        def elapsed(scorer):
            start = time.perf_counter()
            scorer.score_batch(candidates, target)
            return time.perf_counter() - start

        corr_time = elapsed(CorrelationScorer())
        mi_time = elapsed(MutualInformationScorer())
        assert corr_time < mi_time


# --------------------------------------------------------------------------- #
# Degenerate inputs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ALL_SCORERS)
class TestDegenerateInputs:
    def test_constant_candidate_scores_zero(self, name, rng):
        n = 200
        target = rng.normal(size=n)
        scores = _fast(name).score_batch(np.ones((n, 1)), target)
        assert scores[0] == pytest.approx(0.0, abs=1e-6)

    def test_constant_target_handled(self, name, rng):
        n = 200
        scores = _fast(name).score_batch(rng.normal(size=(n, 3)), np.ones(n))
        assert np.all(np.isfinite(scores))

    def test_tiny_sample_handled(self, name, rng):
        scores = _fast(name).score_batch(rng.normal(size=(4, 2)), rng.normal(size=4))
        assert scores.shape == (2,)
        assert np.all(np.isfinite(scores))

    def test_single_candidate_column(self, name, rng):
        n = 150
        signal = rng.normal(size=n)
        scores = _fast(name).score_batch(signal.reshape(-1, 1), signal + rng.normal(0, 0.2, n))
        assert scores.shape == (1,)


class TestGradientBoostingSpecifics:
    def test_subsampling_is_deterministic(self, rng):
        n = 800
        signal = rng.normal(size=n)
        target = 2.0 * signal + rng.normal(0, 0.3, n)
        candidates = np.column_stack([signal, rng.normal(size=n)])
        scorer = GradientBoostingScorer(n_folds=2, max_iter=15, max_depth=2, subsample_size=200)
        np.testing.assert_allclose(
            scorer.score_batch(candidates, target), scorer.score_batch(candidates, target)
        )

    def test_subsampling_reduces_work(self, rng):
        """Subsampling must actually bound cost on large inputs."""
        import time

        n = 4000
        signal = rng.normal(size=n)
        target = 2.0 * signal + rng.normal(0, 0.3, n)
        candidates = np.column_stack([signal, rng.normal(size=n)])

        def elapsed(size):
            scorer = GradientBoostingScorer(n_folds=2, max_iter=20, max_depth=2, subsample_size=size)
            start = time.perf_counter()
            scorer.score_batch(candidates, target)
            return time.perf_counter() - start

        assert elapsed(300) < elapsed(None)

    def test_single_class_fold_handled(self, rng):
        """A rare class must not crash the stratified path."""
        n = 100
        labels = np.zeros(n, dtype=int)
        labels[:3] = 1
        scorer = GradientBoostingScorer(problem_type="classification", n_folds=2, max_iter=10, max_depth=2)
        scores = scorer.score_batch(rng.normal(size=(n, 2)), labels)
        assert np.all(np.isfinite(scores))


class TestSpearmanScorer:
    def test_monotone_nonlinear_beats_pearson(self):
        """exp() of a uniform is monotone but convex: rank correlation is
        exact where product-moment correlation is diluted by the curvature."""
        from beamfeat.scoring import CorrelationScorer, SpearmanScorer

        rng = np.random.default_rng(0)
        x = rng.uniform(0, 3, (400, 1))
        y = np.exp(2 * x[:, 0])
        pearson = CorrelationScorer().score_batch(x, y)[0]
        spearman = SpearmanScorer().score_batch(x, y)[0]
        assert spearman > 0.999
        assert spearman > pearson

    def test_outlier_insensitive(self):
        from beamfeat.scoring import CorrelationScorer, SpearmanScorer

        rng = np.random.default_rng(1)
        x = rng.uniform(0, 1, (300, 1))
        y = x[:, 0] + 0.01 * rng.standard_normal(300)
        y[:5] += 500.0  # five wild rows
        assert SpearmanScorer().score_batch(x, y)[0] > 0.9
        assert CorrelationScorer().score_batch(x, y)[0] < 0.5

    def test_factory_resolves(self):
        from beamfeat.scoring import SpearmanScorer, make_scorer

        assert isinstance(make_scorer("spearman"), SpearmanScorer)
        assert isinstance(make_scorer("rank"), SpearmanScorer)
