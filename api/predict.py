"""Vercel serverless entry point for the FlowPredict TAT pipeline.

GET /api/predict?n_cases=800&seed=42

Runs the pipeline against a freshly-generated synthetic cohort and
returns the diagnostics report as JSON. `n_cases` is capped well below
the CLI default (8000) because a serverless function has a hard wall-
clock budget (Vercel Hobby: 10s; Pro: configurable up to 60s here via
vercel.json) and lightgbm has non-trivial cold-start import time on top
of the actual training.

Exposes a standard WSGI `app` callable -- Vercel's Python builder looks
for this by default. (An earlier version used a `BaseHTTPRequestHandler`
subclass named `handler`, which some Vercel builder versions only treat
as a fallback/legacy pattern and refuse to auto-select, failing with
"No python entrypoint found in default locations".)
"""
import json
import sys
import traceback
from pathlib import Path
from urllib.parse import parse_qs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MAX_N_CASES = 3000
DEFAULT_N_CASES = 800


def app(environ, start_response):
    query = parse_qs(environ.get("QUERY_STRING", ""))
    try:
        n_cases = min(int(query.get("n_cases", [str(DEFAULT_N_CASES)])[0]), MAX_N_CASES)
        seed = int(query.get("seed", ["42"])[0])

        from src.flowpredict.pipeline import run

        result = run(n_cases=n_cases, seed=seed, verbose=False, skip_vif=True)
        diagnostics = result["diagnostics"]

        body = {
            "status": "ok",
            "n_cases": n_cases,
            "seed": seed,
            "n_open_cases": result["n_open_cases"],
            "n_closed_cases": result["n_closed_cases"],
            "mae_p50_days": diagnostics.attrs["mae_p50"],
            "quantile_crossings": diagnostics.attrs["quantile_crossings"],
            "diagnostics_by_quantile": diagnostics.to_dict(orient="records"),
            "model_features": result["model_features"],
        }
        status = "200 OK"
    except Exception:  # noqa: BLE001 - surface any failure as JSON, not a bare 500 page
        body = {
            "status": "error",
            "message": traceback.format_exc().strip().splitlines()[-1],
            "traceback": traceback.format_exc(),
        }
        status = "500 Internal Server Error"

    payload = json.dumps(body, default=str).encode("utf-8")
    headers = [
        ("Content-Type", "application/json"),
        ("Content-Length", str(len(payload))),
    ]
    start_response(status, headers)
    return [payload]
