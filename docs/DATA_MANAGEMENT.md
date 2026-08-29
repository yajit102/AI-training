# Data & Model Artifact Management

This repo tracks **code and metadata only**. Datasets and model weights are
too large and too fluid for `git` to manage well, so they follow strict rules.

## The three tiers

| Tier | Where | Tracked in git? | How |
|---|---|---|---|
| Code, configs, small metadata (<1MB) | `src/`, `configs/`, `datasets/*.md` | Yes, normal git | Committed directly |
| Medium binary artifacts you must version (sample data, small eval sets, demo checkpoints) | anywhere | Yes, via **Git LFS** | Matched by `.gitattributes` |
| Raw datasets, full training corpora, production checkpoints | `data/`, `models/` | **No** | External storage + pointer/manifest file |

## Rule 1 — Never commit raw data or full checkpoints

`data/raw/`, `data/interim/`, `data/processed/`, `data/external/`,
`models/checkpoints/`, and `models/final/` are `.gitignore`'d by default
(only `.gitkeep` survives). Do not force-add files into them.

Store the real files in:
- Object storage (S3 / GCS / Azure Blob), or
- A dataset/model registry (Hugging Face Hub, Weights & Biases Artifacts), or
- [DVC](https://dvc.org/) remote storage, if you adopt DVC for full lineage
  tracking (`*.dvc` pointer files ARE committed — they're tiny).

Then document *where* in a manifest file next to the ignored directory, e.g.
`data/raw/manifest.yaml`:

```yaml
dataset: my-finetune-corpus-v3
source: s3://my-bucket/datasets/my-finetune-corpus-v3/
format: jsonl
size_gb: 42.3
num_examples: 1_204_555
checksum_sha256: <sha256sum of the archive>
license: CC-BY-4.0
retrieved: 2026-08-29
notes: Deduplicated against v2; see datasets/CHANGELOG.md
```

## Rule 2 — Anything binary that IS committed goes through Git LFS

`.gitattributes` already routes common binary extensions (`*.pt`, `*.ckpt`,
`*.safetensors`, `*.parquet`, images, audio, archives, etc.) through LFS
automatically. Before committing a new binary file:

```bash
git lfs install          # once per clone
git lfs track "*.newext" # if the extension isn't already covered
git add .gitattributes
git add path/to/file
git commit -m "..."
```

Only use this path for things that genuinely need version history in git —
small eval fixtures, a demo/quantized checkpoint for a README, sample data
for unit tests. Not for full training sets.

## Rule 3 — Every dataset and checkpoint needs a card

Add a short markdown card next to the manifest:
- `datasets/<name>/README.md` — source, license, preprocessing steps,
  splits, known issues.
- `models/final/<name>/README.md` — base model, training data version,
  hyperparameters, eval results, intended use, limitations.

This is what keeps a fine-tuning repo auditable months later.

## Rule 4 — Large-file guardrail

Before pushing, run:

```bash
scripts/check_large_files.sh
```

This fails the commit if a non-LFS file over 5MB is staged, to catch
accidental `git add` of raw data before it hits the remote.

## Suggested workflow for adding a new dataset

1. Land the raw data in external storage.
2. Write `data/raw/<dataset>/manifest.yaml` + `datasets/<dataset>/README.md`.
3. Write a loader in `src/data/` that pulls from the manifest location.
4. Write processed/cached versions into `data/processed/` (still git-ignored)
   via that loader — never by hand.
5. Commit only the manifest, the card, and the loader code.
