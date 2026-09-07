# Releases

Prebuilt downloadable packages of `src/flowpredict/` (see `deploy/README.md`
for what's inside and how to deploy either one). Committed as plain blobs,
not Git LFS: `releases/.gitattributes` overrides the repo-wide `*.tar.gz`
LFS routing rule for this directory, since these files are small
(tens of KB, source-only).

| File | Version | Notes |
|---|---|---|
| `flowpredict-0.4.0.zip` / `.tar.gz` | 0.4.0 | Fixes `api/predict.py`'s Vercel build error ("No python entrypoint found in default locations"): switched from a `BaseHTTPRequestHandler` subclass named `handler` to a standard WSGI `app` callable, which Vercel's Python builder looks for by default. |
| ~~`flowpredict-0.3.0.*`~~ | 0.3.0 | Fixed the size-limit deploy failure (dropped `scikit-learn`/`scipy`) but still used the `handler`-class entrypoint, which then failed the build with an entrypoint-detection error. |
| ~~`flowpredict-0.2.0.*`~~ | 0.2.0 | Superseded — over Vercel's 250MB unzipped function-size limit. |
