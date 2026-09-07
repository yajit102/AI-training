# FlowPredict — Deploy Package

Two supported deploy targets for the FlowPredict TAT model pipeline
(`src/flowpredict/`): a Docker container (batch job, run on demand) or
a Vercel serverless API (`GET /api/predict`, on-demand HTTP).

**Vercel cannot run the Docker container** — a bare Vercel deploy of
this repo without `api/` and `vercel.json` returns exactly the
`404: NOT_FOUND` platform error you'd get hitting any route with no
matching function or static file. Use the Vercel section below instead.

## Deploy to Vercel (serverless API)

```bash
npm i -g vercel   # if you don't already have the CLI
vercel login
vercel --prod
```

Or via the Vercel dashboard: import the GitHub repo, leave the root
directory as-is (no framework preset needed — `vercel.json` +
`api/predict.py` are auto-detected as a Python serverless function),
and deploy.

Once live:

```
GET https://<your-project>.vercel.app/api/predict?n_cases=800&seed=42
```

returns the diagnostics report as JSON (P50 MAE, per-quantile coverage,
model feature list). `/` serves a small landing page (`public/index.html`)
linking to it.

Notes:
- `n_cases` is capped at 3000 and defaults to 800 (not the CLI's 8000)
  because a serverless invocation has a hard wall-clock budget —
  `vercel.json` sets `maxDuration: 10` to match the Hobby plan's ceiling.
  If you're on Pro, raise it (up to 60) for larger `n_cases`.
- The endpoint runs with `skip_vif=True` (see `src/flowpredict/pipeline.py`)
  to skip the statsmodels VIF pass, which is the slowest single step and
  not worth re-running on every request — it hardcodes the previously
  established pruned feature set instead.
- `api/requirements.txt` is scoped to only what `api/predict.py` imports
  (no `statsmodels`, no `pytest`/`ruff`) to keep the function bundle small.
- If you still see `404: NOT_FOUND` after deploying, check the Vercel
  build log for the function — it usually means `api/predict.py` failed
  to build (missing dependency, Python version mismatch), not that the
  route is unmapped.

## Deploy as a Docker container (batch job)

### First-time setup (on the deploy machine)

Requires Docker (with the `docker compose` plugin) and git.

```bash
git clone https://github.com/yajit102/AI-training.git
cd AI-training
git checkout claude/flowpredict-model-strengthen-9gjd7z   # or main, once merged

docker compose -f deploy/docker-compose.yml build
docker compose -f deploy/docker-compose.yml run --rm flowpredict --n-cases 8000
```

The pipeline is a batch job, not a long-running server: each run prints a
diagnostics report (P50 MAE, conformal coverage, VIF-pruned features,
critical-path comparison) to stdout and exits. Pass `--n-cases N` /
`--seed S` to change the synthetic run size.

## Pushing a code update to an already-deployed machine

```bash
deploy/update.sh
```

This fetches the latest commits on the current branch (fast-forward
only — it refuses to run over local changes or diverged history rather
than discarding anything), rebuilds the `flowpredict:latest` image, and
re-runs it. Override the branch or run arguments:

```bash
FLOWPREDICT_BRANCH=main FLOWPREDICT_ARGS="--n-cases 20000" deploy/update.sh
```

## Versioning

`deploy/VERSION` is bumped on each packaged release (see
`deploy/build_package.sh`). Check it against `docs/FLOWPREDICT_MODEL_STRENGTHENING.md`'s
history to confirm which model-strengthening changes a given deployed
image includes.

## Building a standalone downloadable package

To produce a `.tar.gz` you can hand to someone without git access:

```bash
deploy/build_package.sh
```

Outputs `dist/flowpredict-<version>.tar.gz` containing the pipeline
source, scripts, docs, requirements, and this `deploy/` directory. The
recipient extracts it and runs the "First-time setup" commands above
(minus `git clone`).
