"""Vercel serverless entry point for the FlowPredict TAT pipeline.

GET /api/predict?n_cases=800&seed=42

Runs the pipeline against a freshly-generated synthetic cohort and
returns the diagnostics report as JSON. `n_cases` is capped well below
the CLI default (8000) because a serverless function has a hard wall-
clock budget (Vercel Hobby: 10s; Pro: configurable up to 60s here via
vercel.json) and lightgbm/statsmodels have non-trivial cold-start
import time on top of the actual training.
"""
import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MAX_N_CASES = 3000
DEFAULT_N_CASES = 800


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        query = parse_qs(urlparse(self.path).query)
        try:
            n_cases = min(int(query.get("n_cases", [DEFAULT_N_CASES])[0]), MAX_N_CASES)
            seed = int(query.get("seed", [42])[0])

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
            status_code = 200
        except Exception as exc:  # noqa: BLE001 - surface any failure as JSON, not a bare 500 page
            body = {
                "status": "error",
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            status_code = 500

        payload = json.dumps(body, default=str).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
