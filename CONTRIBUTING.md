# Contributing

## Before you commit

1. Run `git lfs install` once per clone (required — see
   [`docs/DATA_MANAGEMENT.md`](docs/DATA_MANAGEMENT.md)).
2. Never `git add -f` anything inside `data/` or `models/checkpoints|final`.
   If you think you need to, you almost certainly need a manifest file
   instead — read the data management doc.
3. Run `scripts/check_large_files.sh` before pushing. CI also runs it and
   will reject the PR if it fails.
4. Run tests and lint:
   ```bash
   pytest
   ruff check src tests
   ```

## Code organization

- All real logic lives under `src/`; `scripts/` files should be thin CLI
  wrappers that parse args and call into `src/`.
- New model architectures go in `src/models/`, training loops in
  `src/training/`, data loading/preprocessing in `src/data/`.
- Every new dataset needs: a manifest (`data/raw/<name>/manifest.yaml`) and
  a card (`datasets/<name>/README.md`).
- Every new released checkpoint needs a card
  (`models/final/<name>/README.md`) with base model, training data version,
  hyperparameters, and eval numbers.

## Commit messages

Use clear, imperative commit messages (`Add dedup step to corpus pipeline`,
not `updates`). Reference the dataset/model version when relevant.

## Notebooks

Strip output cells before committing (`jupyter nbconvert --clear-output`)
to keep diffs reviewable.
