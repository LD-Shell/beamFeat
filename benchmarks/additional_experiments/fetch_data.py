"""Fetch and preprocess every dataset of the additional-experiments study.

Run on a machine with normal network access:

    python fetch_data.py            # core five (re-derive from the canonical sources)
    python fetch_data.py --all      # plus the four larger optional sets

The shipped data/ folder already contains the core five, derived from GitHub
mirrors of the canonical files (see PROVENANCE.md). This
script re-derives each CSV from the canonical source and compares md5s against
data/CHECKSUMS.md5, warning loudly on any mismatch instead of failing, so a
formatting difference is visible rather than silent.

Canonical sources:
  communities   UCI: Communities and Crime (Redmond)                 [core]
  superconduct  UCI: Superconductivity (Hamidieh, 2018)              [core]
  tecator       StatLib Tecator via mirror (Borggaard & Thodberg)    [core]
  eyedata       CRAN flare: Scheetz et al. (2006) TRIM32             [core]
  riboflavin    CRAN hdi: Buhlmann, Kalisch & Meier (2014)           [core]
  ct_slices     UCI: Relative location of CT slices                  [--all]
  blogfeedback  UCI: BlogFeedback                                    [--all]
  ujiindoorloc  UCI: UJIIndoorLoc (target: longitude)                [--all]
  geomusic      PMLB 4544_GeographicalOriginalofMusic (pip pmlb)     [--all]
"""
from __future__ import annotations

import hashlib
import io
import pathlib
import sys
import urllib.request
import zipfile

import numpy as np
import pandas as pd

DATA = pathlib.Path(__file__).resolve().parent / "data"
DATA.mkdir(exist_ok=True)
UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases"
UCI_STATIC = "https://archive.ics.uci.edu/static/public"


def md5(path: pathlib.Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def check(path: pathlib.Path) -> None:
    ledger = DATA / "CHECKSUMS.md5"
    if not ledger.exists():
        return
    recorded = dict(
        line.split()[::-1] for line in ledger.read_text().splitlines() if line.strip()
    )
    key = f"data/{path.name}"
    if key in recorded:
        got = md5(path)
        status = "OK" if got == recorded[key] else f"MISMATCH (recorded {recorded[key]}, got {got})"
        print(f"  checksum {path.name}: {status}")
    else:
        print(f"  checksum {path.name}: {md5(path)} (new; append to CHECKSUMS.md5)")


def fetch(url: str, fallback: str | None = None) -> bytes:
    for u in [url] + ([fallback] if fallback else []):
        try:
            print(f"  fetching {u}")
            with urllib.request.urlopen(u, timeout=300) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001 - report and try the fallback
            print(f"    failed: {type(e).__name__}: {e}")
    raise RuntimeError(f"could not fetch {url}")


# ------------------------------------------------------------------ core five
def communities() -> None:
    raw = fetch(f"{UCI}/communities/communities.data")
    df = pd.read_csv(io.BytesIO(raw), header=None, na_values="?")
    assert df.shape == (1994, 128), df.shape
    pred = df.iloc[:, 5:]                      # drop 5 non-predictive id columns
    miss = pred.isna().mean()
    keep = pred.loc[:, (miss == 0) | (miss < 0.01)].dropna()   # drop 22 LEMAS cols, 1 row
    assert keep.shape == (1993, 101), keep.shape               # 100 predictors + target
    keep.to_csv(DATA / "communities_crime_numeric.csv", index=False)
    check(DATA / "communities_crime_numeric.csv")


def superconduct() -> None:
    raw = fetch(f"{UCI}/00464/superconduct.zip", f"{UCI_STATIC}/464/superconductivty+data.zip")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = next(n for n in z.namelist() if n.endswith("train.csv"))
        df = pd.read_csv(z.open(name))
    assert df.shape == (21263, 82), df.shape
    df.to_csv(DATA / "superconductivity.csv", index=False)
    check(DATA / "superconductivity.csv")


def tecator() -> None:
    # Canonical origin is StatLib (lib.stat.cmu.edu/datasets/tecator), a bespoke
    # text format; the shipped CSV came from a GitHub mirror of the standard
    # 240 x (100 absorbances + 22 PCs + 3 targets) layout and is pinned by
    # checksum. This function re-fetches that mirror.
    url = "https://raw.githubusercontent.com/gogorazet/tecator/main/csvtecator_simplified_header.csv"
    alt = "https://raw.githubusercontent.com/gogorazet/tecator/main/csvtecator.csv"
    df = pd.read_csv(io.BytesIO(fetch(url, alt)))
    assert df.shape == (240, 126), df.shape
    # The mirror's variants differ only in header names; normalise positionally
    # to the canonical layout so any variant yields the identical file.
    df.columns = (["id"] + [f"_{i}" for i in range(1, 101)]
                  + [f"principal_component_{i}" for i in range(1, 23)]
                  + ["moisture", "fat", "protein"])
    df.to_csv(DATA / "tecator.csv", index=False)
    check(DATA / "tecator.csv")


def _cran_rda(pkg: str, fname: str) -> dict:
    import rdata  # pip install rdata

    raw = fetch(f"https://codeload.github.com/cran/{pkg}/tar.gz/refs/heads/master")
    import tarfile

    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith(fname))
        blob = tf.extractfile(member).read()
    tmp = DATA / f"_{pathlib.Path(fname).name}"
    tmp.write_bytes(blob)
    try:
        return rdata.read_rda(tmp)
    finally:
        tmp.unlink()


def eyedata() -> None:
    ey = _cran_rda("flare", "data/eyedata.rda")
    x, y = np.asarray(ey["x"]), np.asarray(ey["y"]).ravel()
    df = pd.DataFrame(x, columns=[f"g{i}" for i in range(x.shape[1])])
    df.insert(0, "trim32", y)
    assert df.shape == (120, 201), df.shape
    df.to_csv(DATA / "eyedata.csv", index=False)
    check(DATA / "eyedata.csv")


def riboflavin() -> None:
    # hdi stores riboflavin as a data.frame with a matrix column, which the
    # high-level readers reject; walk the parsed object tree instead.
    import tarfile

    import rdata

    raw = fetch("https://codeload.github.com/cran/hdi/tar.gz/refs/heads/master")
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith("data/riboflavin.RData"))
        blob = tf.extractfile(member).read()
    tmp = DATA / "_riboflavin.RData"
    tmp.write_bytes(blob)
    try:
        parsed = rdata.parser.parse_file(tmp)
    finally:
        tmp.unlink()

    arrays: list[np.ndarray] = []

    def collect(o, depth=0):
        val = getattr(o, "value", None)
        if isinstance(val, np.ndarray) and val.dtype.kind in "fd" and val.size >= 71:
            arrays.append(val)
        if isinstance(val, (list, tuple)):
            for c in val:
                collect(c, depth + 1)
        for extra in ("tag", "attributes"):
            ch = getattr(o, extra, None)
            if ch is not None and depth < 12:
                collect(ch, depth + 1)

    collect(parsed.object)
    y = next(a for a in arrays if a.size == 71).ravel()
    x = next(a for a in arrays if a.size == 71 * 4088)
    x = x.reshape(4088, 71).T if x.ndim == 1 else x  # R stores column-major
    df = pd.DataFrame(x, columns=[f"g{i}" for i in range(4088)])
    df.insert(0, "y_log_prod", y)
    assert df.shape == (71, 4089), df.shape
    df.to_csv(DATA / "riboflavin.csv", index=False)
    check(DATA / "riboflavin.csv")


# ----------------------------------------------------- optional larger sets
def ct_slices() -> None:
    raw = fetch(f"{UCI}/00206/slice_localization_data.zip",
                f"{UCI_STATIC}/206/relative+location+of+ct+slices+on+axial+axis.zip")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        df = pd.read_csv(z.open(name))
    df = df.drop(columns=[c for c in df.columns if c.lower() == "patientid"])
    assert df.shape[1] == 385, df.shape       # 384 features + reference target
    df.to_csv(DATA / "ct_slices.csv", index=False)
    check(DATA / "ct_slices.csv")


def blogfeedback() -> None:
    raw = fetch(f"{UCI}/00304/BlogFeedback.zip", f"{UCI_STATIC}/304/blogfeedback.zip")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        df = pd.read_csv(z.open("blogData_train.csv"), header=None)
    assert df.shape == (52397, 281), df.shape  # 280 features + comment count
    df.to_csv(DATA / "blogfeedback.csv", index=False)
    check(DATA / "blogfeedback.csv")


def ujiindoorloc() -> None:
    raw = fetch(f"{UCI}/00310/UJIndoorLoc.zip", f"{UCI_STATIC}/310/ujiindoorloc.zip")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = next(n for n in z.namelist() if n.endswith("trainingData.csv"))
        df = pd.read_csv(z.open(name))
    wap = [c for c in df.columns if c.startswith("WAP")]
    X = df[wap].replace(100, -105)             # 100 = "not detected" sentinel
    out = X.copy()
    out["longitude"] = df["LONGITUDE"]
    assert out.shape[1] == 521, out.shape      # 520 WAPs + target
    out.to_csv(DATA / "ujiindoorloc.csv", index=False)
    check(DATA / "ujiindoorloc.csv")


def geomusic() -> None:
    from pmlb import fetch_data  # pip install pmlb

    df = fetch_data("4544_GeographicalOriginalofMusic")
    assert df.shape[1] == 118, df.shape        # 117 features + target
    df.to_csv(DATA / "geomusic.csv", index=False)
    check(DATA / "geomusic.csv")


CORE = [communities, superconduct, tecator, eyedata, riboflavin]
EXTRA = [ct_slices, blogfeedback, ujiindoorloc, geomusic]

if __name__ == "__main__":
    jobs = CORE + (EXTRA if "--all" in sys.argv else [])
    for job in jobs:
        print(f"== {job.__name__} ==")
        try:
            job()
        except Exception as e:  # noqa: BLE001 - keep going, report at the end
            print(f"  FAILED: {type(e).__name__}: {e}")
    print("done. Re-run bench.py: newly present files register automatically.")
