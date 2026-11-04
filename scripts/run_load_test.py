"""Sweep Locust over 1, 5, 10, 20 concurrent users and collect p50/p95/throughput.

    python scripts/run_load_test.py --host http://localhost:8000 --target forgegpt
    python scripts/run_load_test.py --host $FUNCTION_URL --target lambda --label aws_lambda

Writes benchmarks/results/load_test_<label>.csv
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--target", default="forgegpt", choices=["forgegpt", "lambda", "llamacpp"])
    p.add_argument("--label", default=None, help="name for the results file")
    p.add_argument("--users", type=int, nargs="+", default=[1, 5, 10, 20])
    p.add_argument("--seconds", type=int, default=45)
    p.add_argument("--prompt", default="short", choices=["short", "long"])
    p.add_argument("--max-tokens", type=int, default=64)
    args = p.parse_args()
    label = args.label or f"{args.target}_{args.prompt}"
    tmp = ROOT / "benchmarks" / "results" / "locust_raw"
    tmp.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "TARGET": args.target, "PROMPT": args.prompt,
           "MAX_TOKENS": str(args.max_tokens)}  # fmt: skip

    rows = []
    for n in args.users:
        prefix = tmp / f"{label}_u{n}"
        print(f"{label}: {n} users for {args.seconds}s...", flush=True)
        subprocess.run(
            [sys.executable, "-m", "locust", "-f", str(ROOT / "benchmarks" / "locustfile.py"),
             "--headless", "-u", str(n), "-r", str(n), "-t", f"{args.seconds}s",
             "--host", args.host, "--csv", str(prefix), "--only-summary"],
            env=env, check=True, capture_output=True,
        )  # fmt: skip
        stats = pd.read_csv(f"{prefix}_stats.csv")
        agg = stats[stats["Name"] == "Aggregated"].iloc[0]
        rows.append(
            {
                "label": label,
                "users": n,
                "requests": int(agg["Request Count"]),
                "failures": int(agg["Failure Count"]),
                "p50_ms": float(agg["50%"]),
                "p95_ms": float(agg["95%"]),
                "mean_ms": round(float(agg["Average Response Time"]), 1),
                "req_per_s": round(float(agg["Requests/s"]), 2),
                "max_tokens": args.max_tokens,
            }
        )
        print(rows[-1], flush=True)
    out = ROOT / "benchmarks" / "results" / f"load_test_{label}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
