# Datasets

This directory holds **cards**, not data. One subfolder per dataset:

```
datasets/
└── <dataset-name>/
    └── README.md
```

Each card should cover:
- **Source** — where it came from, license, citation
- **Location** — pointer to the manifest at `data/raw/<dataset-name>/manifest.yaml`
- **Preprocessing** — steps applied to go from raw → processed, and which
  script in `src/data/` does it
- **Splits** — train/val/test sizes and how they were made
- **Known issues** — dedup gaps, label noise, license caveats

See `docs/DATA_MANAGEMENT.md` for the full policy and manifest schema.
See `templates/DATASET_CARD_TEMPLATE.md` for a starting point.
