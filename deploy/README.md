# FlowPredict — Deploy Package

Everything needed to build, run, and update the FlowPredict TAT model
pipeline (`src/flowpredict/`) as a Docker container.

## First-time setup (on the deploy machine)

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
