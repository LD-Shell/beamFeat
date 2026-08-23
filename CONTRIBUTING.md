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

## Releasing

The order matters; each step feeds the next.

1. Confirm a clean tree: `ruff check src tests benchmarks`, `pytest`
   (coverage floor enforced), `mkdocs build`, `python -m build`,
   `twine check dist/*`.
2. Push to GitHub and let CI pass on all jobs, including the notebook job
   and the `paper` workflow (the compiled PDF artifact is the manuscript's
   real rendering check).
3. Upload the checked distributions to PyPI: `twine upload dist/*`.
4. Tag the release (`git tag v0.1.0 && git push --tags`) and create the
   GitHub release.
5. Archive the release on Zenodo; copy the concept DOI into
   `CITATION.cff` (template at the bottom of that file) and the README's
   DOI badge.
6. Publish the conda-forge recipe: `grayskull pypi beamfeat`, then a pull
   request to `conda-forge/staged-recipes` (a skeleton is in `recipe/`).
