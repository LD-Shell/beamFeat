"""PySR on the twelve-law physics panel of ``benchmarks/feynman_panel.py``.

The panel, its twelve generators, its seeds, its 0.1% noise level and its two
scoring criteria are imported from ``feynman_panel`` rather than restated
here, so the two methods cannot drift apart in the data they see or in how a
recovery is judged. This module supplies only the PySR fit and the
front-selection rule.

Scoring. ``solved`` is held-out R^2 > 0.999 on the same probe draw beamfeat is
scored on. ``exact_form`` is ``feynman_panel._exact_form`` applied unchanged:
the selected expression is handed to it as a one-element list of sympy terms,
which is the same path beamfeat takes when a multi-term law is captured as one
composite. The proportionality test is therefore the identical one, not a
relaxed variant. It has a consequence worth stating: beamfeat's formulas are
constant-free and its linear model supplies the intercept for free, while a
PySR expression carries its own additive constant into the ratio and fails the
test on it. ``exact_form_after_intercept`` reports the same test with a
leading additive constant removed first, so a miss caused only by that
asymmetry is visible. It is a diagnostic; ``exact_form`` is the scored figure.

Budget and selection rule are fixed in ``BUDGET`` and ``FRONT_RULE`` below,
applied identically to all twelve laws, and recorded in the output. Nothing
in this file varies them per equation.

Run with ``python run_panel.py``. Results are written to
``results/pysr_results.json`` in the shape of
``benchmarks/feynman_results.json`` so the two join on ``equation``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import sys
import time
import traceback
import warnings

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from feynman_panel import _exact_form, equations, make_data  # noqa: E402

warnings.filterwarnings("ignore")

# beamfeat's default binary operators are exactly PySR's documented defaults.
# Its default unary set is not: PySR ships no unary operators at all, and
# several panel laws (the pendulum's sqrt, the Gaussian's exp) are then
# outside the search space entirely. The set below is beamfeat's own default,
# spelled in PySR's names, so neither method is given an operator the other
# lacks. This is the only deviation from PySR's documented defaults.
UNARY_OPS = ["log", "sqrt", "square", "abs", "inv"]
UNARY_OPS_WITH_EXP = [*UNARY_OPS, "exp"]
BINARY_OPS = ["+", "-", "*", "/"]

# Fixed before any equation was run and applied unchanged to all twelve.
# The budget is niterations, stated explicitly at PySR's own documented
# default rather than left implicit. maxsize, populations, population_size,
# the parsimony coefficient and the loss are not set here at all, so they take
# PySR's defaults; the resolved values are recorded in the output. Determinism
# costs the parallel search -- serial evaluation is several times slower --
# and buys a panel that reruns to the same equations, which is the property
# the rest of this repository's results are held to. timeout_in_seconds is a
# ceiling, not the budget: a fit that reaches it is recorded as truncated
# rather than counted as a completed search.
BUDGET = {
    "niterations": 100,
    "parallelism": "serial",
    "deterministic": True,
    "random_state": 0,
    "timeout_in_seconds": 1800,
    # Set so the estimator's own predict() and get_best() agree with the rule
    # scored below; the scoring itself does not depend on it.
    "model_selection": "best",
}

# Search-shaping parameters recorded per run so the defaults in force are on
# the record rather than inferred from a version number.
RECORDED_PARAMS = (
    "niterations", "maxsize", "populations", "population_size",
    "ncycles_per_iteration", "parsimony", "elementwise_loss", "loss_function",
    "loss_scale", "model_selection", "deterministic", "parallelism",
    "random_state", "timeout_in_seconds",
)

# PySR returns a front, not a model, so a rule has to say which member is
# scored. All three of PySR's documented rules are applied to every front and
# recorded, so the choice is inspectable rather than load-bearing, and the
# scored one is named here.
#
# The scored rule is "best", PySR's own default. An earlier run of this panel
# declared "accuracy" instead, on the reasoning that the panel's primary
# criterion is a held-out accuracy threshold and that "accuracy" is the only
# one of the three with no tunable constant inside it. That reasoning was
# wrong in a way the run made obvious: at 0.1% noise with maxsize 30 the
# lowest-loss member is the noise-fitting end of the front, so "accuracy"
# picked complexity 23-30 on every law and recovered the exact form of none of
# them, while the correct law sat in the same front at complexity 4-7.
# PROVENANCE.md records that history; reporting all three rules is what keeps
# the correction visible instead of silent.
RULES = ("accuracy", "best", "score")
FRONT_RULE = "best"


def _plain(value):
    """JSON-safe rendering of a parameter that may be a Julia or numpy object."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _versions() -> dict:
    """PySR, Julia and interpreter versions, for the record."""
    import pysr

    julia = "unknown"
    try:
        from pysr import jl

        julia = str(jl.seval("string(VERSION)"))
    except Exception:
        try:
            from juliacall import Main as jl_main

            julia = str(jl_main.seval("string(VERSION)"))
        except Exception:
            pass
    return {
        "pysr": getattr(pysr, "__version__", "unknown"),
        "julia": julia,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def _front(model) -> list[dict]:
    """The Pareto front as plain records, one per retained complexity."""
    frame = model.equations_
    if isinstance(frame, list):  # multi-output; the panel is single-output
        frame = frame[0]
    rows = []
    for position, (_, row) in enumerate(frame.iterrows()):
        rows.append({
            "index": position,
            "complexity": int(row["complexity"]),
            "loss": float(row["loss"]),
            "score": float(row["score"]) if "score" in row else None,
            "equation": str(row["equation"]),
            "sympy": str(row["sympy_format"]) if "sympy_format" in row else None,
        })
    return rows


def _select(front: list[dict], rule: str) -> int:
    """Index of the front member one of PySR's documented rules picks.

    ``accuracy`` is the lowest loss outright; ``score`` is the highest score,
    the discrete derivative of log-loss against complexity; ``best`` is the
    highest score among members whose loss is within 1.5x the lowest, which is
    PySR's default. Ties go to the lower complexity, the only part the front's
    ordering leaves open. Computed from the archived front rather than asked
    of the estimator, so the numbers are reproducible from the JSON alone;
    ``pysr_get_best`` in each record cross-checks this against PySR's own
    answer.
    """
    if not front:
        raise RuntimeError("PySR returned an empty Pareto front")
    # These follow pysr.sr.idx_model_selection exactly; "best" degenerates to
    # "accuracy" whenever no other member is within 1.5x the lowest loss.
    if rule == "accuracy":
        return int(min(front, key=lambda r: (r["loss"], r["complexity"]))["index"])
    pool = front
    if rule == "best":
        floor = min(r["loss"] for r in front)
        pool = [r for r in front if r["loss"] <= 1.5 * floor] or front
    ranked = max(pool, key=lambda r: (r["score"] if r["score"] is not None else -np.inf,
                                      -r["complexity"]))
    return int(ranked["index"])


def _predict(model, X, index: int) -> np.ndarray:
    """Predictions of the selected front member.

    ``predict(..., index=)`` evaluates the member PySR itself holds, which is
    the expression the record reports; lambdifying the sympy conversion is the
    fallback for versions without the argument.
    """
    try:
        return np.asarray(model.predict(X, index=index), dtype=np.float64)
    except TypeError:
        import sympy

        expression = model.sympy(index=index)
        symbols = sympy.symbols(f"x0:{X.shape[1]}")
        func = sympy.lambdify(symbols, expression, modules="numpy")
        with np.errstate(all="ignore"):
            return np.asarray(func(*[X[:, j] for j in range(X.shape[1])]), dtype=np.float64)


class _Terms:
    """Adapter presenting a PySR expression to ``_exact_form`` unchanged.

    ``_exact_form`` asks a fitted model for ``to_sympy()`` and compares the
    terms it gets back against the expected ones. A PySR equation is one
    expression, so the list holds one element, which is the case
    ``_exact_form`` already handles for a beamfeat fit that captured a
    multi-term law as a single composite.
    """

    def __init__(self, expression):
        self._expression = expression

    def to_sympy(self):
        return [self._expression]


def _without_intercept(expression):
    """The expression with a leading additive constant dropped, or None."""
    import sympy

    constant = expression.as_coeff_Add()[0]
    if constant == 0:
        return None
    return sympy.simplify(expression - constant)


def _pysr_get_best(model):
    """PySR's own pick under its configured ``model_selection``, for cross-check.

    ``_select`` recomputes the rules from the archived front so the record is
    reproducible without the library; this records what the library itself
    answered, so a disagreement is visible rather than assumed away.
    """
    try:
        row = model.get_best()
    except Exception:
        return None
    if isinstance(row, list):  # multi-output; the panel is single-output
        row = row[0]
    return {"model_selection": _plain(getattr(model, "model_selection", None)),
            "complexity": int(row["complexity"]), "loss": float(row["loss"])}


def _warn_on_rule_disagreement(name, record, front) -> None:
    """Say so if the recomputed rule and the library's own pick differ.

    They should not: ``_select`` mirrors ``pysr.sr.idx_model_selection``. A
    disagreement means the library changed under the harness, which is worth
    seeing in the log rather than discovering from a number that moved.
    """
    reported = record["pysr_get_best"]
    if not reported or reported.get("model_selection") != FRONT_RULE:
        return
    ours = front[_select(front, FRONT_RULE)]
    if int(ours["complexity"]) != int(reported["complexity"]):
        print(
            f"    NOTE {name}: recomputed {FRONT_RULE!r} picks complexity "
            f"{ours['complexity']}, PySR's own get_best() picks "
            f"{reported['complexity']}. Both are recorded; the scored figure "
            f"is the recomputed one.",
            flush=True,
        )


def _score_pick(model, front, index, X_test, y_test, expected_terms, r2_score):
    """Score one front member: R^2 on the probe draw, and the exact-form test."""
    member = front[index]
    expression = model.sympy(index=index)
    predictions = _predict(model, X_test, index)
    finite = np.all(np.isfinite(predictions))
    r2 = float(r2_score(y_test, predictions)) if finite else None
    pick = {
        "index": index, "complexity": member["complexity"], "loss": member["loss"],
        "score": member["score"], "expression": str(expression),
        "r2": r2, "solved": bool(r2 is not None and r2 > 0.999),
        "exact_form": None, "exact_form_after_intercept": None,
    }
    if expected_terms is not None:
        pick["exact_form"] = _exact_form(_Terms(expression), expected_terms)
        residual = _without_intercept(expression)
        pick["exact_form_after_intercept"] = (
            pick["exact_form"] if residual is None
            else _exact_form(_Terms(residual), expected_terms)
        )
    return pick


def run_panel(unary_ops: list[str], label: str, budget: dict) -> dict:
    from pysr import PySRRegressor
    from sklearn.metrics import r2_score

    rows = []
    for name, generator, n_cols, expected_terms, note in equations():
        X, X_test, y, y_test = make_data(name, generator, n_cols)

        record = {
            "method": "pysr", "equation": name, "note": note, "label": label,
            "r2": None, "seconds": None, "solved": False, "exact_form": None,
            "exact_form_after_intercept": None, "top_formula": "",
            "selected_index": None, "selected_complexity": None,
            "selected_loss": None, "pareto_front": [], "rules": {},
            "pysr_get_best": None, "budget": dict(budget), "front_rule": FRONT_RULE,
            "unary_operators": list(unary_ops), "binary_operators": list(BINARY_OPS),
            "resolved_params": None, "budget_truncated": False, "error": None,
        }
        started = time.perf_counter()
        try:
            model = PySRRegressor(
                binary_operators=list(BINARY_OPS),
                unary_operators=list(unary_ops),
                progress=False,
                verbosity=0,
                temp_equation_file=True,
                **budget,
            )
            record["resolved_params"] = {
                key: _plain(getattr(model, key, None)) for key in RECORDED_PARAMS
            }
            model.fit(X, y)
            front = _front(model)

            record["seconds"] = time.perf_counter() - started
            ceiling = budget.get("timeout_in_seconds")
            record["budget_truncated"] = bool(
                ceiling and record["seconds"] >= 0.95 * ceiling
            )
            record["pareto_front"] = front
            record["pysr_get_best"] = _pysr_get_best(model)
            _warn_on_rule_disagreement(name, record, front)

            # Every rule is scored, so the reported one is a label on the
            # record rather than the only thing the record can answer.
            for rule in RULES:
                record["rules"][rule] = _score_pick(
                    model, front, _select(front, rule), X_test, y_test,
                    expected_terms, r2_score,
                )

            scored = record["rules"][FRONT_RULE]
            record["selected_index"] = scored["index"]
            record["selected_complexity"] = scored["complexity"]
            record["selected_loss"] = scored["loss"]
            record["top_formula"] = scored["expression"]
            record["r2"] = scored["r2"]
            record["solved"] = scored["solved"]
            record["exact_form"] = scored["exact_form"]
            record["exact_form_after_intercept"] = scored["exact_form_after_intercept"]
            if scored["r2"] is None:
                record["error"] = "selected expression produced non-finite predictions"
        except Exception:
            record["seconds"] = time.perf_counter() - started
            record["error"] = traceback.format_exc(limit=8)

        rows.append(record)
        if record["error"]:
            print(f"  FAILED  {name:<22} after {record['seconds']:.1f}s", flush=True)
            print("    " + record["error"].strip().splitlines()[-1], flush=True)
        else:
            mark = "SOLVED" if record["solved"] else "  --  "
            exact = {True: " exact", False: "", None: ""}[record["exact_form"]]
            print(
                f"  {mark}{exact:>6}  {name:<22} R2 {record['r2']:.4f}  "
                f"{record['seconds']:6.1f}s  c={record['selected_complexity']:<3} "
                f"{record['top_formula'][:44]}",
                flush=True,
            )

    solved = sum(1 for r in rows if r["solved"])
    exact = sum(1 for r in rows if r["exact_form"])
    failed = [r["equation"] for r in rows if r["error"]]
    truncated = [r["equation"] for r in rows if r["budget_truncated"]]
    timed = [r["seconds"] for r in rows if r["seconds"] is not None]
    mean_seconds = float(np.mean(timed)) if timed else float("nan")
    # The exact-form denominator is the laws that declare one, not all twelve;
    # the other three are boundaries the panel marks as having no exact form,
    # and neither method is scored on them.
    n_declared = sum(1 for r in rows if r["rules"].get(FRONT_RULE, {}).get("exact_form") is not None)
    by_rule = {
        rule: {
            "solved": sum(1 for r in rows if r["rules"].get(rule, {}).get("solved")),
            "exact_form": sum(1 for r in rows if r["rules"].get(rule, {}).get("exact_form")),
            "median_complexity": float(np.median(
                [r["rules"][rule]["complexity"] for r in rows if rule in r["rules"]] or [float("nan")]
            )),
        }
        for rule in RULES
    }
    print(
        f"\n[{label}]  scored rule {FRONT_RULE!r}: solved {solved}/12 (R^2>0.999)  "
        f"exact-form {exact}/{n_declared} declared  mean {mean_seconds:.1f}s/eq"
        + (f"  FAILED {len(failed)}: {', '.join(failed)}" if failed else "")
        + (f"  TRUNCATED {len(truncated)}: {', '.join(truncated)}" if truncated else "")
    )
    print(f"  {'rule':<10} {'solved':>8} {'exact form':>12} {'median complexity':>18}")
    for rule in RULES:
        mark = "  <- scored" if rule == FRONT_RULE else ""
        counts = by_rule[rule]
        solved_cell = "{}/12".format(counts["solved"])
        exact_cell = "{}/{}".format(counts["exact_form"], n_declared)
        print(f"  {rule:<10} {solved_cell:>8} {exact_cell:>12} "
              f"{counts['median_complexity']:>18.0f}{mark}")
    return {
        "method": "pysr", "label": label, "solved": solved, "exact_form": exact,
        "n_declared": n_declared, "front_rule": FRONT_RULE, "by_rule": by_rule,
        "mean_seconds": mean_seconds, "failed": failed, "truncated": truncated,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PySR on the twelve-law physics panel")
    parser.add_argument(
        "--out", type=pathlib.Path, default=HERE / "results" / "pysr_results.json"
    )
    parser.add_argument(
        "--niterations", type=int, default=None,
        help="override the fixed iteration budget; recorded in the output and "
             "not a setting any reported run should use",
    )
    args = parser.parse_args()

    budget = dict(BUDGET)
    if args.niterations is not None:
        budget["niterations"] = args.niterations

    versions = _versions()
    print(f"pysr {versions['pysr']}  julia {versions['julia']}  "
          f"python {versions['python']}")
    print(f"budget {budget}")
    print(f"rules scored {list(RULES)}, reported rule {FRONT_RULE!r}\n")

    results = [
        run_panel(UNARY_OPS, "default operators", budget),
        run_panel(UNARY_OPS_WITH_EXP, "with exp enabled", budget),
    ]
    payload = {"versions": versions, "budget": budget, "front_rule": FRONT_RULE,
               "rules": list(RULES), "panels": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        json.dump(payload, handle, indent=2)
    # shown relative, so a committed or pasted run log carries no one's filesystem
    try:
        shown = args.out.relative_to(HERE.parents[1])
    except ValueError:
        shown = args.out
    print(f"\nwrote {shown}")

    failed = sorted({name for panel in results for name in panel["failed"]})
    truncated = sorted({name for panel in results for name in panel["truncated"]})
    if failed:
        print(
            f"\n{len(failed)} equation(s) did not complete: {', '.join(failed)}\n"
            "The records above hold the traceback for each. This exit is "
            "non-zero deliberately: a panel with a hole in it is not a result.",
            file=sys.stderr,
        )
    if truncated:
        print(
            f"\n{len(truncated)} equation(s) hit the wall-clock ceiling and so did "
            f"not spend the stated iteration budget: {', '.join(truncated)}\n"
            "Their rows are not comparable with the rest of the panel; raise "
            "timeout_in_seconds and rerun rather than reporting them.",
            file=sys.stderr,
        )
    return 1 if (failed or truncated) else 0


if __name__ == "__main__":
    raise SystemExit(main())
