# Releases

Prebuilt downloadable packages of `src/flowpredict/` (see `deploy/README.md`
for what's inside and how to deploy either one). Committed as plain blobs,
not Git LFS: `releases/.gitattributes` overrides the repo-wide `*.tar.gz`
LFS routing rule for this directory, since these files are small
(tens of KB, source-only).

| File | Version | Notes |
|---|---|---|
| `flowpredict-0.3.0.zip` / `.tar.gz` | 0.3.0 | Fixes the Vercel deploy failure: `src/flowpredict` no longer imports `scikit-learn` or `scipy` (replaced with pure-numpy equivalents in `src/flowpredict/_stats.py`, and `lightgbm` trained via its native `Booster` API instead of the sklearn-wrapper `LGBMRegressor`, which hard-requires scikit-learn). `numpy+pandas+scipy+scikit-learn+lightgbm` together were ~293MB unzipped, over Vercel's 250MB serverless-function limit — the trimmed `api/requirements.txt` set is ~225MB. |
| ~~`flowpredict-0.2.0.*`~~ | 0.2.0 | Superseded — this is the version that failed to deploy. |
