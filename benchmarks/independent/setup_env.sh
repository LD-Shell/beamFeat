#!/usr/bin/env bash
# Prepare an activated environment for the independent comparison study:
# installs the pinned dependencies, beamfeat from this repository, and
# JupyterLab, then registers a kernel and verifies the result.
#
#     conda create -n af315 python=3.11 -y
#     conda activate af315
#     bash benchmarks/independent/setup_env.sh
#
# A plain virtual environment works equally well:
#
#     python -m venv .venv-af315 && source .venv-af315/bin/activate
#     bash benchmarks/independent/setup_env.sh
#
# The script must be run inside an activated environment. It refuses to
# install into a base or system interpreter, because the pins it applies
# (numpy 1.26.4, scikit-learn 1.7.2) would downgrade an everyday environment.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
KERNEL_NAME="${KERNEL_NAME:-af315}"

echo "repository:   $ROOT"

if ! command -v python >/dev/null 2>&1; then
    cat <<'MSG'

No `python` on PATH. An environment is almost certainly not active: a failed
`conda activate` leaves the shell unchanged and every command after it runs
against the base system.

    conda info --envs                       # is the environment there?
    conda create -n af315 python=3.11 -y    # create it if not
    conda activate af315                    # must print (af315) in the prompt
    bash benchmarks/independent/setup_env.sh

MSG
    exit 1
fi

echo "interpreter:  $(command -v python)"

# --- refuse to run outside a dedicated environment ------------------------
python - <<'PY'
import sys, os
prefix = sys.prefix
in_venv = sys.prefix != sys.base_prefix
conda = os.environ.get("CONDA_DEFAULT_ENV", "")
if not in_venv and conda in ("", "base"):
    sys.exit(
        "\nRefusing to install: no dedicated environment is active.\n"
        "These pins (numpy 1.26.4, scikit-learn 1.7.2) would downgrade a\n"
        "general-purpose environment. Create and activate one first:\n"
        "    conda create -n af315 python=3.11 -y && conda activate af315\n"
    )
print(f"environment:  {conda or prefix}")
PY

# --- install ---------------------------------------------------------------
echo
echo "installing pinned dependencies, beamfeat, and JupyterLab..."
cd "$HERE"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

# --- register the kernel ---------------------------------------------------
echo "registering Jupyter kernel '$KERNEL_NAME'..."
python -m ipykernel install --user \
    --name "$KERNEL_NAME" \
    --display-name "Python ($KERNEL_NAME) - beamfeat study"

# --- verify ----------------------------------------------------------------
echo
echo "verification"
python - <<'PY'
import importlib.metadata as md
import sys

EXPECTED = {"numpy": "1.26.4", "scikit-learn": "1.7.2"}
PRESENT = ["numpy", "scipy", "pandas", "matplotlib", "scikit-learn", "lightgbm",
           "autofeat", "openfe", "featuretools", "knockpy", "beamfeat",
           "pint", "sympy", "pytest", "pytest-cov", "ipykernel", "jupyterlab",
           "setuptools"]

bad = []
for name in PRESENT:
    try:
        version = md.version(name)
    except md.PackageNotFoundError:
        print(f"  {name:<14} MISSING")
        bad.append(name)
        continue
    want = EXPECTED.get(name)
    flag = ""
    if want and version != want:
        flag = f"  <-- expected {want}"
        bad.append(name)
    print(f"  {name:<14} {version}{flag}")

print(f"\n  interpreter    {sys.executable}")

try:
    import autofeat, numpy as np
    from autofeat import AutoFeatRegressor
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 6, (120, 3))
    y = X[:, 0] * X[:, 1] + rng.normal(0, 0.1, 120)
    AutoFeatRegressor(feateng_steps=2, verbose=0).fit_transform(X, y)
    print("  autofeat smoke test: passed")
except Exception as exc:
    print(f"  autofeat smoke test: FAILED  {type(exc).__name__}: {exc}")
    bad.append("autofeat")

try:
    import numpy as np
    from knockpy.knockoff_filter import KnockoffFilter
    rng = np.random.default_rng(0)
    F = rng.standard_normal((300, 25))
    beta = np.zeros(25)
    beta[:5] = 3.0
    KnockoffFilter(ksampler="gaussian", fstat="lasso").forward(
        X=F, y=F @ beta + rng.standard_normal(300), fdr=0.2)
    print("  knockpy smoke test:  passed")
except Exception as exc:
    print(f"  knockpy smoke test:  FAILED  {type(exc).__name__}: {exc}")
    bad.append("knockpy")

try:
    import pkg_resources  # noqa: F401  (featuretools imports it at module load)
    print("  featuretools prerequisite: pkg_resources present")
except Exception:
    print("  featuretools prerequisite: MISSING pkg_resources (needs setuptools < 82)")
    bad.append("setuptools")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.plot([0, 1], [0, 1])
    import tempfile, os
    fd, tmp = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    fig.savefig(tmp, format="pdf")
    os.unlink(tmp)
    plt.close(fig)
    print("  matplotlib smoke test: passed (PDF backend)")
except Exception as exc:
    print(f"  matplotlib smoke test: FAILED  {type(exc).__name__}: {exc}")
    bad.append("matplotlib")

if bad:
    sys.exit(f"\nEnvironment is not usable; check: {sorted(set(bad))}")
PY

cat <<MSG

Ready. To run the study:

  Notebook, default mode (loads the archived results, about two minutes):
      cd $HERE && jupyter lab beamfeat_benchmark.ipynb
      then choose the kernel "Python ($KERNEL_NAME) - beamfeat study"

  Notebook, headless:
      cd $HERE && jupyter nbconvert --to notebook --execute --inplace beamfeat_benchmark.ipynb

  A single autofeat cell, for example the friedman1 seeds:
      cd $HERE && DATASETS=friedman1 python bench.py synthetic autofeat 5 results_syn_af3.json
MSG
