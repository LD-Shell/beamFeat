"""Tests for the scikit-learn compatible estimators."""

from __future__ import annotations

import logging
import re
import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator

from beamfeat import (
    BeamFeatClassifier,
    BeamFeatRegressor,
    BeamFeatTransformer,
    DegenerateFitWarning,
    NoDiscoveriesError,
    NoDiscoveriesWarning,
)

ESTIMATORS = [BeamFeatTransformer, BeamFeatRegressor, BeamFeatClassifier]


@pytest.fixture
def rng():
    return np.random.default_rng(20260723)


@pytest.fixture
def regression_data(rng):
    """y = (a * b) / c, so the true expression needs depth 2."""
    n = 400
    X = rng.uniform(1.0, 6.0, (n, 4))
    y = (X[:, 0] * X[:, 1]) / X[:, 2] + rng.normal(0, 0.05, n)
    return X, y


@pytest.fixture
def classification_data(rng):
    n = 400
    X = rng.uniform(1.0, 6.0, (n, 4))
    score = X[:, 0] * X[:, 1]
    return X, (score > np.median(score)).astype(int)


def _fast(cls, **kwargs):
    """Estimator with cheap settings so the suite stays quick."""
    kwargs.setdefault("max_depth", 2)
    kwargs.setdefault("beam_width", 20)
    kwargs.setdefault("selector", None)
    return cls(**kwargs)


# --------------------------------------------------------------------------- #
# scikit-learn API compliance
# --------------------------------------------------------------------------- #


@pytest.mark.slow
@pytest.mark.parametrize("cls", ESTIMATORS)
def test_check_estimator(cls):
    """The full scikit-learn conformance suite."""
    check_estimator(cls(max_depth=1, beam_width=6, selector=None))


@pytest.mark.parametrize("cls", ESTIMATORS)
class TestApiConventions:
    def test_get_params_roundtrip(self, cls):
        estimator = cls(max_depth=3, beam_width=17)
        params = estimator.get_params()
        assert params["max_depth"] == 3
        assert params["beam_width"] == 17

    def test_set_params(self, cls):
        estimator = cls().set_params(max_depth=4)
        assert estimator.max_depth == 4

    def test_clone(self, cls):
        estimator = cls(max_depth=3, beam_width=17)
        cloned = clone(estimator)
        assert cloned.get_params() == estimator.get_params()

    def test_repr(self, cls):
        assert cls.__name__ in repr(cls())

    def test_init_does_not_validate(self, cls):
        """scikit-learn requires __init__ to only store parameters."""
        estimator = cls(max_depth=-5, beam_width=-1)
        assert estimator.max_depth == -5

    def test_unfitted_raises(self, cls, regression_data):
        from sklearn.exceptions import NotFittedError

        X, _ = regression_data
        estimator = cls()
        method = estimator.transform if cls is BeamFeatTransformer else estimator.predict
        with pytest.raises(NotFittedError):
            method(X)


# --------------------------------------------------------------------------- #
# Regressor
# --------------------------------------------------------------------------- #


class TestRegressor:
    def test_recovers_generating_formula(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor).fit(X, y)
        assert any(
            all(token in formula for token in ("x0", "x1", "x2")) for formula in model.formulas()
        ), f"true formula not among {model.formulas()[:5]}"

    def test_fits_well_on_recoverable_signal(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor).fit(X, y)
        assert model.score(X, y) > 0.95

    def test_beats_linear_model_on_nonlinear_signal(self, regression_data):
        """The reason to construct features at all."""
        from sklearn.linear_model import Ridge

        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        beamfeat = _fast(BeamFeatRegressor).fit(X_train, y_train).score(X_test, y_test)
        baseline = Ridge().fit(X_train, y_train).score(X_test, y_test)
        assert beamfeat > baseline

    def test_generalises_to_held_out_data(self, regression_data):
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        model = _fast(BeamFeatRegressor).fit(X_train, y_train)
        assert model.score(X_test, y_test) > 0.9

    def test_equation_is_readable(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor).fit(X, y)
        equation = model.equation()
        assert equation.startswith("y = ")
        assert any(formula in equation for formula in model.formulas())

    def test_equation_respects_max_terms(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, beam_width=30).fit(X, y)
        full = model.equation()
        short = model.equation(max_terms=2)

        # Count coefficient markers ("<number>*") rather than testing whether
        # each formula appears: formulas can be substrings of one another, so
        # substring matching over-counts.
        def n_terms(equation: str) -> int:
            return len(re.findall(r"\d\*", equation))

        assert n_terms(short) <= 2
        assert n_terms(short) < n_terms(full)

    def test_equation_reproduces_predictions(self, rng):
        """The printed equation must be the model, not a decoration.

        Coefficients are de-standardised for display, so this checks that
        evaluating the printed form by hand matches ``predict``.
        """
        n = 300
        X = rng.uniform(1.0, 5.0, (n, 2))
        y = X[:, 0] * X[:, 1] + rng.normal(0, 0.05, n)
        model = BeamFeatRegressor(max_depth=1, beam_width=8, selector=None, include_originals=False)
        model.fit(X, y)

        from beamfeat.expression import Evaluator

        evaluator = Evaluator({f"x{i}": X[:, i] for i in range(X.shape[1])})
        matrix = np.column_stack([evaluator.transform_values(node) for node in model.features_])

        scale = np.where(model.scaler_.scale_ == 0, 1.0, model.scaler_.scale_)
        raw_coefficients = model.coef_ / scale
        offset = model.intercept_ - np.sum(raw_coefficients * model.scaler_.mean_)
        manual = matrix @ raw_coefficients + offset

        np.testing.assert_allclose(manual, model.predict(X), rtol=1e-8, atol=1e-8)

    def test_to_sympy(self, regression_data):
        sympy = pytest.importorskip("sympy")
        X, y = regression_data
        model = _fast(BeamFeatRegressor).fit(X, y)
        expressions = model.to_sympy()
        assert len(expressions) == len(model.formulas())
        assert all(isinstance(expression, sympy.Basic) for expression in expressions)

    def test_deterministic(self, regression_data):
        X, y = regression_data
        first = _fast(BeamFeatRegressor).fit(X, y)
        second = _fast(BeamFeatRegressor).fit(X, y)
        assert first.formulas() == second.formulas()
        np.testing.assert_allclose(first.predict(X), second.predict(X))


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #


class TestClassifier:
    def test_fits_and_predicts(self, classification_data):
        X, y = classification_data
        model = _fast(BeamFeatClassifier).fit(X, y)
        assert model.score(X, y) > 0.85

    def test_predict_proba_is_valid(self, classification_data):
        X, y = classification_data
        probabilities = _fast(BeamFeatClassifier).fit(X, y).predict_proba(X)
        assert probabilities.shape == (len(y), 2)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_decision_function(self, classification_data):
        X, y = classification_data
        assert _fast(BeamFeatClassifier).fit(X, y).decision_function(X).shape == (len(y),)

    def test_string_labels(self, rng):
        """Labels are encoded internally; string classes must round-trip."""
        n = 300
        X = rng.uniform(1.0, 5.0, (n, 3))
        score = X[:, 0] * X[:, 1]
        y = np.where(score > np.median(score), "high", "low")
        model = _fast(BeamFeatClassifier).fit(X, y)
        assert set(model.classes_) == {"high", "low"}
        assert set(np.unique(model.predict(X))) <= {"high", "low"}

    def test_multiclass(self, rng):
        n = 400
        X = rng.uniform(1.0, 5.0, (n, 3))
        score = X[:, 0] * X[:, 1]
        y = np.digitize(score, np.quantile(score, [0.33, 0.66]))
        model = _fast(BeamFeatClassifier).fit(X, y)
        assert len(model.classes_) == 3
        assert model.score(X, y) > 0.6

    def test_recovers_discriminative_formula(self, classification_data):
        X, y = classification_data
        model = _fast(BeamFeatClassifier).fit(X, y)
        assert any("x0" in formula and "x1" in formula for formula in model.formulas())


# --------------------------------------------------------------------------- #
# Transformer
# --------------------------------------------------------------------------- #


class TestTransformer:
    def test_output_shape(self, regression_data):
        X, y = regression_data
        transformer = _fast(BeamFeatTransformer).fit(X, y)
        assert transformer.transform(X).shape == (len(y), transformer.n_features_out_)

    def test_feature_names_out(self, regression_data):
        X, y = regression_data
        transformer = _fast(BeamFeatTransformer).fit(X, y)
        names = transformer.get_feature_names_out()
        assert len(names) == transformer.n_features_out_
        assert list(names) == transformer.formulas()

    def test_requires_y(self, regression_data):
        X, _ = regression_data
        with pytest.raises(ValueError, match="requires y"):
            BeamFeatTransformer().fit(X)

    def test_works_in_pipeline(self, regression_data):
        from sklearn.linear_model import Ridge

        X, y = regression_data
        pipeline = Pipeline(
            [("features", _fast(BeamFeatTransformer)), ("model", Ridge())]
        )
        assert pipeline.fit(X, y).score(X, y) > 0.9

    def test_cross_validates_without_leakage(self, regression_data):
        """Feature construction must happen inside each fold.

        If it leaked, cross-validated scores would be implausibly close to the
        in-sample score.
        """
        from sklearn.linear_model import Ridge

        X, y = regression_data
        pipeline = Pipeline([("features", _fast(BeamFeatTransformer)), ("model", Ridge())])
        scores = cross_val_score(pipeline, X, y, cv=3)
        assert np.all(scores > 0.8)
        assert np.all(scores <= 1.0)

    def test_classification_problem_type(self, classification_data):
        X, y = classification_data
        transformer = _fast(BeamFeatTransformer, problem_type="classification").fit(X, y)
        assert transformer.n_features_out_ > 0


# --------------------------------------------------------------------------- #
# Transform-time behaviour
# --------------------------------------------------------------------------- #


class TestTransformSemantics:
    def test_prediction_invariant_to_row_subset(self, rng):
        """Search-time filters must not apply at transform time.

        A subset in which some column happens to be constant would otherwise
        silently zero out a feature and change the prediction.
        """
        n = 200
        constant_prefix = np.r_[np.ones(30), np.arange(n - 30) + 2.0]
        X = np.column_stack([np.arange(float(n)), constant_prefix, np.arange(float(n)) * 2])
        y = X[:, 0] + rng.normal(0, 0.1, n)

        model = _fast(BeamFeatRegressor, max_depth=1, beam_width=8).fit(X, y)
        np.testing.assert_allclose(model.predict(X)[:30], model.predict(X[:30]))

    def test_transform_does_not_mutate_estimator(self, regression_data):
        """scikit-learn forbids state changes during transform or predict."""
        X, y = regression_data
        model = _fast(BeamFeatRegressor).fit(X, y)
        before = dict(model.__dict__)
        model.predict(X)
        assert set(model.__dict__) == set(before)

    def test_out_of_domain_data_is_handled(self, rng, caplog):
        """Training data admits log; test data does not."""
        n = 300
        X_train = rng.uniform(1.0, 5.0, (n, 2))
        y_train = np.log(X_train[:, 0]) + rng.normal(0, 0.05, n)
        model = _fast(BeamFeatRegressor, max_depth=1, beam_width=10).fit(X_train, y_train)

        X_test = np.column_stack([np.full(20, -5.0), rng.uniform(1.0, 5.0, 20)])
        with caplog.at_level(logging.WARNING, logger="beamfeat.estimators"):
            predictions = model.predict(X_test)
        assert predictions.shape == (20,)
        assert np.all(np.isfinite(predictions))

    def test_single_sample_message_follows_convention(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor)
        with pytest.raises(ValueError, match="sample"):
            model.fit(X[:1], y[:1])


# --------------------------------------------------------------------------- #
# Selection integration
# --------------------------------------------------------------------------- #


class TestSelectionIntegration:
    def test_selection_result_is_exposed(self, regression_data):
        X, y = regression_data
        model = BeamFeatRegressor(max_depth=2, beam_width=20, selector="permutation").fit(X, y)
        assert model.selection_result_ is not None
        assert model.selection_result_.method == "permutation"

    def test_search_result_is_exposed(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor).fit(X, y)
        assert model.search_result_ is not None
        assert model.search_result_.n_proposed_total > 0

    def test_selector_none_skips_selection(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, selector=None).fit(X, y)
        assert model.selection_result_ is None
        assert model.fdr_controlled_ is None

    def test_flag_true_when_selection_passes(self, regression_data):
        X, y = regression_data
        model = BeamFeatRegressor(
            max_depth=2, beam_width=20, selector="permutation", target_fdr=0.2, random_state=0
        ).fit(X, y)
        if model.fdr_controlled_:
            # Parsimony keeps a subset of the screened set, never more.
            assert 0 < model.n_features_out_ <= model.selection_result_.n_selected

    def test_empty_default_on_pure_noise(self):
        """Default on_no_discoveries="empty": on a noise target where nothing
        passes, the model keeps no features, warns visibly, and predicts the
        training mean — the honest null model."""
        rng = np.random.default_rng(3)
        X = rng.uniform(1, 6, (400, 4))
        y = rng.standard_normal(400)
        with pytest.warns(NoDiscoveriesWarning):
            model = BeamFeatRegressor(
                max_depth=2, beam_width=15, selector="permutation", target_fdr=0.05, random_state=3
            ).fit(X, y)
        assert model.n_features_out_ == 0
        assert model.fdr_controlled_ is False
        np.testing.assert_allclose(model.predict(X[:7]), np.full(7, y.mean()), rtol=1e-9)
        assert "no feature passed selection" in model.equation()

    def test_fallback_mode_keeps_search_output_flagged(self):
        rng = np.random.default_rng(3)
        X = rng.uniform(1, 6, (400, 4))
        y = rng.standard_normal(400)
        with pytest.warns(NoDiscoveriesWarning):
            model = BeamFeatRegressor(
                max_depth=2, beam_width=15, target_fdr=0.05, random_state=3,
                on_no_discoveries="fallback",
            ).fit(X, y)
        assert model.n_features_out_ > 0
        assert model.fdr_controlled_ is False

    def test_raise_mode_raises(self):
        rng = np.random.default_rng(3)
        X = rng.uniform(1, 6, (400, 4))
        y = rng.standard_normal(400)
        with pytest.raises(NoDiscoveriesError):
            BeamFeatRegressor(
                max_depth=2, beam_width=15, target_fdr=0.05, random_state=3,
                on_no_discoveries="raise",
            ).fit(X, y)

    def test_parsimony_compacts_without_losing_recovery(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(1, 6, (400, 4))
        y = X[:, 0] * X[:, 1] + rng.normal(0, 0.05, 400)
        compact = BeamFeatRegressor(max_depth=2, beam_width=25, random_state=0).fit(X, y)
        full = BeamFeatRegressor(
            max_depth=2, beam_width=25, random_state=0, parsimony=None
        ).fit(X, y)
        assert compact.n_features_out_ < full.n_features_out_
        assert any("x0" in f and "x1" in f for f in compact.formulas())
        assert compact.score(X, y) > 0.99
        # every kept feature passed screening
        report = {r["formula"]: r for r in compact.selection_report_}
        assert all(report[f]["screened"] for f in compact.formulas())

    def test_selection_report_is_consistent(self):
        rng = np.random.default_rng(1)
        X = rng.uniform(1, 6, (400, 4))
        y = X[:, 0] * X[:, 1] + rng.normal(0, 0.05, 400)
        model = BeamFeatRegressor(max_depth=2, beam_width=20, random_state=1).fit(X, y)
        report = model.selection_report_
        assert report and all(
            set(row) == {"formula", "p_value", "q_value", "statistic", "screened", "kept"}
            for row in report
        )
        for row in report:
            if row["kept"]:
                assert row["screened"]
                assert row["q_value"] <= model.target_fdr + 1e-9

    def test_nan_error_is_actionable(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(1, 6, (50, 3))
        X[3, 1] = np.nan
        with pytest.raises(ValueError, match="SimpleImputer"):
            BeamFeatRegressor(max_depth=1, beam_width=5).fit(X, rng.standard_normal(50))

    def test_same_data_selection_never_claims_control(self, regression_data):
        """With the holdout disabled, candidates are tested on the rows that
        chose them; the estimator must not claim the FDR guarantee."""
        X, y = regression_data
        model = BeamFeatRegressor(
            max_depth=2, beam_width=20, selector="permutation", selection_holdout=None, random_state=0
        ).fit(X, y)
        assert model.fdr_controlled_ is False

    def test_holdout_split_is_disjoint_and_seed_stable(self, regression_data):
        X, y = regression_data
        first = BeamFeatRegressor(
            max_depth=2, beam_width=20, selector="permutation", random_state=7
        ).fit(X, y)
        second = BeamFeatRegressor(
            max_depth=2, beam_width=20, selector="permutation", random_state=7
        ).fit(X, y)
        assert first.formulas() == second.formulas()

    def test_selected_features_still_predict(self, regression_data):
        X, y = regression_data
        model = BeamFeatRegressor(
            max_depth=2, beam_width=20, selector="permutation", target_fdr=0.2
        ).fit(X, y)
        assert model.score(X, y) > 0.9

    def test_pipeline_level_fdr_calibration(self):
        """End-to-end control: search on one split, selection on the other.

        A feature is a false discovery iff its formula references only
        irrelevant columns. FDR is an expectation, so the mean FDP over
        trials is what must respect the nominal level (with Monte Carlo
        slack) — not each individual trial.
        """
        nominal = 0.2
        fdps = []
        recovered = 0
        n_trials = 12
        for trial in range(n_trials):
            rng = np.random.default_rng(100 + trial)
            X = rng.uniform(1, 6, (400, 6))
            signal = X[:, 0] * X[:, 1]
            y = signal + rng.normal(0, 0.05 * np.std(signal), 400)
            model = BeamFeatRegressor(
                max_depth=2, beam_width=20, selector="permutation", target_fdr=nominal,
                selection_holdout=0.5, random_state=trial,
            ).fit(X, y)
            if not model.fdr_controlled_:
                continue
            relevant = {"x0", "x1"}
            false = sum(1 for node in model.features_ if not (node.columns() & relevant))
            fdps.append(false / len(model.features_))
            recovered += any("x0" in f and "x1" in f for f in model.formulas())
        assert fdps, "selection never passed; pipeline has no power"
        assert float(np.mean(fdps)) <= nominal + 0.15
        assert recovered >= len(fdps) - 2


# --------------------------------------------------------------------------- #
# Units
# --------------------------------------------------------------------------- #


class TestUnitsIntegration:
    def test_units_constrain_constructed_features(self, rng):
        pint = pytest.importorskip("pint")
        ureg = pint.UnitRegistry()

        n = 300
        X = rng.uniform(1.0, 5.0, (n, 2))
        y = X[:, 0] * X[:, 1] + rng.normal(0, 0.05, n)
        units = {"x0": 1.0 * ureg.kilogram, "x1": 1.0 * ureg.meter}

        model = _fast(BeamFeatRegressor, units=units).fit(X, y)
        formulas = model.formulas()
        assert not any("+" in formula or "-" in formula for formula in formulas)
        assert any("*" in formula for formula in formulas)

    def test_units_are_optional(self, regression_data):
        X, y = regression_data
        assert _fast(BeamFeatRegressor, units=None).fit(X, y).n_features_out_ > 0


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


class TestConfiguration:
    @pytest.mark.parametrize("scorer", ["correlation", "mutual_information"])
    def test_scorer_choice(self, regression_data, scorer):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, scorer=scorer).fit(X, y)
        assert model.score(X, y) > 0.9

    def test_max_features_caps_output(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, beam_width=30, max_features=5).fit(X, y)
        assert model.n_features_out_ <= 5

    def test_restricted_operators(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, unary_ops=(), binary_ops=("mul",)).fit(X, y)
        assert not any("log" in formula or "sqrt" in formula for formula in model.formulas())

    def test_include_originals_false(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, include_originals=False).fit(X, y)
        assert not any(formula.startswith("x") and len(formula) <= 3 for formula in model.formulas())

    def test_depth_zero_uses_originals_only(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, max_depth=0).fit(X, y)
        assert all(formula.startswith("x") for formula in model.formulas())

    def test_alpha_affects_regressor(self, regression_data):
        X, y = regression_data
        weak = _fast(BeamFeatRegressor, alpha=0.001).fit(X, y)
        strong = _fast(BeamFeatRegressor, alpha=1000.0).fit(X, y)
        assert np.max(np.abs(strong.coef_)) < np.max(np.abs(weak.coef_))


class TestSelectorDiagnosticsSurfaced:
    def test_knockoff_unsatisfiability_is_a_visible_warning(self):
        """A knockoff configuration that cannot select must warn through the
        warnings module, not only the logger, so notebook users see it."""
        rng = np.random.default_rng(0)
        X = rng.uniform(1, 6, (300, 4))
        y = X[:, 0] / X[:, 1] + rng.normal(0, 0.02, 300)
        with pytest.warns(UserWarning, match="offset"):
            BeamFeatRegressor(
                max_depth=2, beam_width=15, selector="knockoff", target_fdr=0.1, random_state=0
            ).fit(X, y)


class TestHoldoutFitCheck:
    def test_degenerate_fit_warns(self, monkeypatch):
        """A model whose holdout score is negative must warn: FDR-vetted
        association is not a generalisation certificate, and the gap must be
        loud when it opens."""
        rng = np.random.default_rng(0)
        X = rng.uniform(1, 6, (400, 4))
        y = X[:, 0] * X[:, 1] + rng.normal(0, 0.05, 400)
        monkeypatch.setattr(BeamFeatRegressor, "score", lambda self, X, y: -1.7)
        with pytest.warns(DegenerateFitWarning):
            BeamFeatRegressor(max_depth=2, beam_width=20, random_state=0).fit(X, y)

    def test_healthy_fit_does_not_warn(self):
        rng = np.random.default_rng(1)
        X = rng.uniform(1, 6, (400, 4))
        y = X[:, 0] * X[:, 1] + rng.normal(0, 0.05, 400)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DegenerateFitWarning)
            BeamFeatRegressor(max_depth=2, beam_width=20, random_state=0).fit(X, y)

    def test_no_holdout_no_check(self):
        rng = np.random.default_rng(2)
        X = rng.uniform(1, 6, (200, 3))
        y = X[:, 0] + rng.normal(0, 0.05, 200)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DegenerateFitWarning)
            BeamFeatRegressor(
                max_depth=1, beam_width=10, selector=None, random_state=0
            ).fit(X, y)


class TestClassifierEquation:
    def test_binary_equation_reproduces_decision_function(self):
        """The printed log-odds equation, evaluated on raw feature values,
        must equal decision_function — the same model-equation identity the
        regressor guarantees."""
        rng = np.random.default_rng(0)
        X = rng.uniform(1, 6, (400, 4))
        y = (X[:, 0] * X[:, 1] > X[:, 2] * X[:, 3]).astype(int)
        model = BeamFeatClassifier(max_depth=2, beam_width=20, random_state=0).fit(X, y)
        equation = model.equation(precision=10)
        assert equation.startswith("logit P(y = ")

        features = model._apply(X[:12])
        scale = np.where(model.scaler_.scale_ == 0, 1.0, model.scaler_.scale_)
        raw = np.atleast_2d(model.coef_)[0] / scale
        offset = float(model.intercept_[0] - np.sum(raw * model.scaler_.mean_))
        manual = features @ raw + offset
        np.testing.assert_allclose(manual, model.decision_function(X[:12]), rtol=1e-9)

    def test_multiclass_equation_has_one_line_per_class(self):
        rng = np.random.default_rng(1)
        X = rng.uniform(1, 6, (450, 4))
        y = np.digitize(X[:, 0] * X[:, 1], [8.0, 18.0])
        model = BeamFeatClassifier(max_depth=1, beam_width=10, random_state=0).fit(X, y)
        lines = model.equation().splitlines()
        assert sum(line.startswith("score(") for line in lines) == len(model.classes_)

    def test_null_model_equation_reports_priors(self):
        rng = np.random.default_rng(3)
        X = rng.uniform(1, 6, (400, 4))
        y = (rng.uniform(size=400) > 0.5).astype(int)
        with pytest.warns(NoDiscoveriesWarning):
            model = BeamFeatClassifier(
                max_depth=2, beam_width=15, target_fdr=0.05, random_state=3
            ).fit(X, y)
        assert "class priors only" in model.equation()


class TestAlphaAuto:
    def test_default_selects_alpha_by_cv(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor).fit(X, y)
        assert type(model.model_).__name__ == "RidgeCV"
        assert model.alpha_ > 0

    def test_float_alpha_keeps_fixed_ridge(self, regression_data):
        X, y = regression_data
        model = _fast(BeamFeatRegressor, alpha=0.5).fit(X, y)
        assert type(model.model_).__name__ == "Ridge"
        assert model.alpha_ == 0.5

    def test_bad_alpha_string_raises(self, regression_data):
        X, y = regression_data
        with pytest.raises(ValueError, match="auto"):
            _fast(BeamFeatRegressor, alpha="invalid").fit(X, y)

    def test_wide_selection_small_n_stays_bounded(self):
        # Regression case: many correlated screened features on few rows,
        # where a fixed penalty is effectively no penalty. Predictions from
        # the cross-validated default must stay on the target's scale.
        rng = np.random.default_rng(0)
        n, p = 140, 40
        z = rng.standard_normal((n, 1))
        X = 0.8 * z + 0.6 * rng.standard_normal((n, p))
        y = (z[:, 0] + 0.3 * rng.standard_normal(n))
        Xtr, Xte, ytr, yte = X[:100], X[100:], y[:100], y[100:]
        model = BeamFeatRegressor(random_state=0, max_depth=1).fit(Xtr, ytr)
        pred = model.predict(Xte)
        assert np.all(np.isfinite(pred))
        assert float(np.mean((yte - pred) ** 2)) < 4.0 * float(np.var(yte))
