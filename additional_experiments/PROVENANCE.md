# Provenance

## How the shipped data/ was built

The five core CSVs were derived from plain-file GitHub mirrors of the
canonical sources and validated against the canonical shapes and value ranges
before use. `fetch_data.py` re-derives every file from the canonical source
(or the pinned mirror where the canonical format is bespoke) and compares
md5s against `data/CHECKSUMS.md5`, warning on mismatch. Run it once so the
archived records descend from the source of record; if a canonical re-fetch
produces a different but shape-identical file, regenerate `CHECKSUMS.md5` and
note it in the revision.

## Datasets

| file | shape (rows x cols incl. target) | canonical source / citation | preprocessing |
|---|---|---|---|
| communities_crime_numeric.csv | 1,993 x 101 | UCI Communities and Crime; Redmond & Baveja (2002) | drop 5 non-predictive id columns; drop the 22 LEMAS columns (~84% missing); drop the single row missing OtherPerCap |
| superconductivity.csv | 21,263 x 82 | UCI Superconductivity; Hamidieh (2018) | none (train.csv verbatim) |
| tecator.csv | 240 x 126 | StatLib Tecator; Borggaard & Thodberg (1992); mirror pinned by checksum | none; the harness uses the 100 absorbance channels and the fat target, ignoring the shipped principal components |
| eyedata.csv | 120 x 201 | CRAN `flare`; Scheetz et al. (2006), PNAS | matrix x + TRIM32 target from eyedata.rda |
| riboflavin.csv | 71 x 4,089 | CRAN `hdi`; Buhlmann, Kalisch & Meier (2014) | y + 4,088-gene matrix from riboflavin.RData (R column-major layout honoured) |
| ct_slices.csv (via --all) | 53,500 x 385 | UCI Relative location of CT slices; Graf et al. (2011) | drop patientId |
| blogfeedback.csv (via --all) | 52,397 x 281 | UCI BlogFeedback; Buza (2014) | train file only |
| ujiindoorloc.csv (via --all) | 19,937 x 521 | UCI UJIIndoorLoc; Torres-Sospedra et al. (2014) | 520 WAP columns, sentinel 100 -> -105 dBm; target = longitude |
| geomusic.csv (via --all) | 1,059 x 118 | PMLB 4544_GeographicalOriginalofMusic; Zhou et al. (2014); Olson et al. (2017) | PMLB file verbatim |

## Result records

`results_dev/` holds records from development runs on a constrained machine
(Linux, Python 3.12, scikit-learn 1.8.0, numpy 2.4.4) at reduced trial and
seed counts. Cross-method ratios there are meaningful; absolute times are
not comparable to the paper's pinned machine. Regenerate all records locally
before citing any number, and note that `bench.py` runs each fit in a
spawned subprocess when `FIT_BUDGET_S` > 0, which adds a small constant to
recorded times.
