#!/usr/bin/env bash
# Prepare an activated environment for the PySR physics panel: installs the
# pinned dependencies, lets PySR fetch its Julia runtime, and verifies both
# before any panel is run.
#
#     conda create -n pysr312 python=3.12 -y
#     conda activate pysr312
#     bash benchmarks/pysr_panel/setup_env.sh
#
# A plain virtual environment works equally well:
#
#     python -m venv .venv-pysr && source .venv-pysr/bin/activate
#     bash benchmarks/pysr_panel/setup_env.sh
#
# The script must be run inside an activated environment. It refuses to
# install into a base or system interpreter: the first `import pysr` writes a
# Julia depot of several hundred megabytes into the environment's prefix, and
# that does not belong in an everyday interpreter.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

echo "repository:   $ROOT"

if ! command -v python >/dev/null 2>&1; then
    cat <<'MSG'

No `python` on PATH. An environment is almost certainly not active: a failed
`conda activate` leaves the shell unchanged and every command after it runs
against the base system.

    conda info --envs                        # is the environment there?
    conda create -n pysr312 python=3.12 -y   # create it if not
    conda activate pysr312                   # must print (pysr312) in the prompt
    bash benchmarks/pysr_panel/setup_env.sh

MSG
    exit 1
fi

echo "interpreter:  $(command -v python)"

# --- refuse to run outside a dedicated environment ------------------------
python - <<'PY'
import os
import sys

prefix = sys.prefix
in_venv = sys.prefix != sys.base_prefix
conda = os.environ.get("CONDA_DEFAULT_ENV", "")
if not in_venv and conda in ("", "base"):
    sys.exit(
        "\nRefusing to install: no dedicated environment is active.\n"
        "The first `import pysr` downloads a Julia runtime and a package depot\n"
        "into this prefix. Create and activate an environment first:\n"
        "    conda create -n pysr312 python=3.12 -y && conda activate pysr312\n"
    )
print(f"environment:  {conda or prefix}")
PY

# --- install ---------------------------------------------------------------
echo
echo "installing pinned dependencies..."
cd "$HERE"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

# --- Julia -----------------------------------------------------------------
# PySR resolves and installs its Julia dependencies on first import. Doing it
# here rather than inside the panel keeps a multi-minute download out of the
# measured wall time of the first equation.
echo
echo "resolving the Julia runtime (first run downloads it; several minutes)..."
python - <<'PY'
import pysr

print(f"  pysr      {getattr(pysr, '__version__', 'unknown')}")
try:
    from pysr import jl

    print(f"  julia     {jl.seval('string(VERSION)')}")
except Exception as exc:
    raise SystemExit(
        f"\nJulia is not usable: {type(exc).__name__}: {exc}\n"
        "PySR downloads it through juliapkg on first import; a proxy or an\n"
        "offline machine is the usual cause. Set JULIA_DEPOT_PATH to a\n"
        "writable location, or install Julia yourself and point juliapkg at\n"
        "it with PYTHON_JULIAPKG_EXE, then rerun this script."
    ) from None
PY

# --- verify ----------------------------------------------------------------
echo
echo "verification"
python - <<'PY'
import importlib.metadata as md
import pathlib
import sys

PRESENT = ["pysr", "sympy", "scikit-learn", "numpy", "pandas"]
bad = []
for name in PRESENT:
    try:
        print(f"  {name:<14} {md.version(name)}")
    except md.PackageNotFoundError:
        print(f"  {name:<14} MISSING")
        bad.append(name)

print(f"\n  interpreter    {sys.executable}")

# beamfeat must NOT be importable here: this environment exists to score PySR,
# and an installed beamfeat would let a panel run produce beamfeat numbers
# that never passed through the repository's own pinned environment.
try:
    import beamfeat  # noqa: F401

    print("  beamfeat: present  <-- unexpected; this environment should not have it")
except ImportError:
    print("  beamfeat: absent, as intended")

# The panel's data and scoring code must import without beamfeat present.
# The shell has already cd'd into this directory, so benchmarks/ is its parent.
sys.path.insert(0, str(pathlib.Path.cwd().parent))
try:
    from feynman_panel import _exact_form, equations, make_data  # noqa: F401

    rows = equations()
    X, X_test, y, y_test = make_data(rows[0][0], rows[0][1], rows[0][2])
    assert X.shape == (500, 3) and y_test.shape == (500,)
    print(f"  panel import:  {len(rows)} equations, data generators reachable")
except Exception as exc:
    print(f"  panel import:  FAILED  {type(exc).__name__}: {exc}")
    bad.append("feynman_panel")

try:
    import numpy as np
    from pysr import PySRRegressor

    rng = np.random.default_rng(0)
    Xs = rng.uniform(1, 5, (60, 2))
    ys = Xs[:, 0] * Xs[:, 1]
    PySRRegressor(
        binary_operators=["+", "*"], niterations=2, parallelism="serial",
        deterministic=True, random_state=0, progress=False, verbosity=0,
        temp_equation_file=True,
    ).fit(Xs, ys)
    print("  pysr smoke test: passed")
except Exception as exc:
    print(f"  pysr smoke test: FAILED  {type(exc).__name__}: {exc}")
    bad.append("pysr")

if bad:
    sys.exit(f"\nEnvironment is not usable; check: {sorted(set(bad))}")
PY

cat <<MSG

Ready. To run the panel:

  cd $HERE && python run_panel.py | tee results/pysr_panel.log

Budget, operator sets and the front-selection rule are fixed in run_panel.py
and stated in README.md. Do not vary them per equation; a run that does is not
comparable with the beamfeat panel it is joined to.
MSG
