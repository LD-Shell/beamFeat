# Contributing to beamfeat

Thank you for considering a contribution.

## Reporting problems and asking questions

Open an issue on the tracker. For suspected statistical problems — a
selector exceeding its nominal FDR, a guarantee claimed where an assumption
fails — please include a runnable snippet and, if possible, a trial count:
FDR statements are about expectations over repetitions, so a single run
exceeding the nominal level is not by itself a bug, but a measured mean over
trials that does is one, and we treat those as the highest-priority class of
issue.

## Development setup

```bash
git clone https://github.com/LD-Shell/beamfeat
cd beamfeat
pip install -e ".[all]"
pytest -q          # full suite
ruff check src tests benchmarks
```

## Standards for changes

- **Statistical claims must be measured, not asserted.** Any change to
  `beamfeat.selection` needs calibration evidence: realised FDR against
  nominal over enough trials, and power on a planted-signal design. The
  existing tests in `tests/test_selection.py` show the expected form.
- **Docstrings state assumptions.** Every guarantee documents the conditions
  it holds under, and the code warns at runtime when they are visibly
  violated.
- **No silent failure paths.** If a procedure cannot deliver its guarantee,
  it must say so (see `fdr_controlled_`).
- **No upper version pins.** New floors must be verified by the
  `floor-versions` CI job.
- Tests accompany code; `pytest -q` and `ruff check` must pass.

## Scope

Feature construction operators, scorers, and selectors with stated
error-control properties are in scope. Heuristic selectors without a stated
guarantee are accepted only when clearly labelled as such in their docstring
and excluded from the default configuration.
