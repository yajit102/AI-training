# AI Training

A structured repository for training and fine-tuning ML models — LLMs,
computer vision, or general ML — with strict, explicit rules around how
datasets and model weights are tracked.

## Layout

```
.
├── configs/            # Hyperparameter / run configs (YAML), by domain
│   ├── training/
│   ├── model/
│   └── data/
├── data/               # Git-ignored except manifests — see docs/DATA_MANAGEMENT.md
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
├── datasets/           # Dataset cards (README per dataset) + loader registry
├── models/             # Git-ignored except manifests/cards
│   ├── checkpoints/
│   └── final/
├── notebooks/          # Exploratory notebooks (keep outputs stripped)
├── src/                # Installable package: all real logic lives here
│   ├── data/           # Loading, preprocessing, tokenization/augmentation
│   ├── models/         # Model/architecture definitions
│   ├── training/       # Training loops, trainer configs
│   ├── evaluation/     # Metrics, eval harnesses
│   └── utils/          # Shared utilities
├── scripts/            # Thin CLI entry points that call into src/
├── tests/              # Unit tests for src/
└── docs/
    └── DATA_MANAGEMENT.md   # The rules — read this first
```

## The one rule that matters

**Raw datasets and trained weights never go into normal git.** They're
tracked one of two ways:

1. Small, genuinely-versioned binaries → **Git LFS** (auto-routed by
   `.gitattributes`).
2. Everything else (full datasets, production checkpoints) → external
   storage, referenced by a `manifest.yaml`/`.dvc` pointer file that *is*
   committed.

Full details, examples, and the required manifest schema:
[`docs/DATA_MANAGEMENT.md`](docs/DATA_MANAGEMENT.md).

## Getting started

```bash
# Install Git LFS once per machine
git lfs install

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Verify nothing large is about to be committed
scripts/check_large_files.sh
```

## Adding a dataset or model

See [`docs/DATA_MANAGEMENT.md`](docs/DATA_MANAGEMENT.md) — write a manifest
and a card before touching `data/` or `models/`.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).
