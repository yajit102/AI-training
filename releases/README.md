# Releases

Prebuilt downloadable packages of `src/flowpredict/` (see `deploy/README.md`
for what's inside and how to deploy either one). Tracked via Git LFS per
this repo's `.gitattributes` (`*.tar.gz` is LFS-routed) — `git lfs install`
before cloning if you want the actual archive rather than a pointer file.

| File | Version | Notes |
|---|---|---|
| `flowpredict-0.2.0.tar.gz` | 0.2.0 | Adds the Vercel serverless deploy target (`api/`, `public/`, `vercel.json`) alongside the Docker package. |
