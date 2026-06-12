#!/usr/bin/env python3
"""Seed the corpus with ~20 arXiv papers on commodity / time-series forecasting.

Runs against a live API (default http://localhost:8000). Brings the graph to life
from day one with a meaningful, methodologically diverse set of papers.

    cd backend && uv run python ../scripts/seed_corpus.py
"""

from __future__ import annotations

import os
import sys
import time

import httpx

API = os.environ.get("LATTICE_API_BASE", "http://localhost:8000")

# A diverse seed set: classical econometrics, ML, deep learning, transformers,
# and surveys across commodity / energy / agricultural price forecasting.
SEED_ARXIV_IDS = [
    "2304.11829",  # transformer time series
    "2310.20496",
    "2201.12740",
    "2211.14730",  # PatchTST
    "2310.10688",  # time series foundation models
    "2106.13008",  # Autoformer
    "2012.07436",  # Informer
    "1912.09363",  # Temporal Fusion Transformer
    "2205.13504",  # are transformers effective for TS
    "2302.11939",
    "2303.06053",
    "2308.11200",
    "2207.01186",
    "2202.07125",  # TS transformers survey
    "2305.18803",
    "2309.06236",
    "2110.13041",
    "2101.02118",
    "2004.13408",
    "2009.04269",
]


def main() -> int:
    token = os.environ.get("LATTICE_AUTH_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    ok = 0
    with httpx.Client(base_url=API, timeout=180.0, headers=headers) as client:
        try:
            client.get("/health").raise_for_status()
        except httpx.HTTPError as exc:
            print(f"API not reachable at {API}: {exc}", file=sys.stderr)
            return 1
        for i, arxiv_id in enumerate(SEED_ARXIV_IDS, 1):
            print(f"[{i}/{len(SEED_ARXIV_IDS)}] ingesting arXiv:{arxiv_id} ...", flush=True)
            try:
                resp = client.post("/ingest/arxiv", json={"arxiv_id": arxiv_id})
                resp.raise_for_status()
                status = resp.json().get("status")
                print(f"    -> {status}")
                ok += 1
            except httpx.HTTPError as exc:
                print(f"    !! failed: {exc}", file=sys.stderr)
            time.sleep(1.0)  # be polite to arXiv
    print(f"\nDone: {ok}/{len(SEED_ARXIV_IDS)} ingested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
