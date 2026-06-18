#!/usr/bin/env python3
"""Run the evaluation harnesses and print reports.

In ``--ci`` mode it enforces the PRD thresholds and exits non-zero on regression.
Deterministic and offline (no LLM keys or services).

    cd backend && uv run python ../scripts/run_eval.py --ci
"""

from __future__ import annotations

import json
import sys

from lattice.eval.runner import run_offline_eval


def main(argv: list[str]) -> int:
    report, ok = run_offline_eval()
    print(json.dumps(report, indent=2))
    if "--ci" in argv:
        if not ok:
            print("EVAL CI CHECK FAILED", file=sys.stderr)
            return 1
        print("\nEVAL CI CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
