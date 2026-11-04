"""Cost per 1,000 requests from measured latency/throughput and on-demand list prices.

    python scripts/cost_model.py

Inputs (whatever exists in benchmarks/results/):
  inference.csv                  local single-request latency of the INT8 ONNX model
  load_test_forgegpt_short.csv   local API throughput at saturation
  load_test_aws_lambda.csv       measured on AWS (after deployment), if present
  load_test_llamacpp_*.csv       measured on EC2 (after deployment), if present
Rows built only from local measurements are marked "projected".

Prices: us-east-1, on-demand list prices before any free tier. Check
https://aws.amazon.com/lambda/pricing/ and EC2 pricing before relying on them.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"

LAMBDA_PER_REQUEST = 0.20 / 1_000_000  # USD
LAMBDA_PER_GB_SECOND = 0.0000166667  # USD, x86, first pricing tier
LAMBDA_MEMORY_GB = 2.0
EC2_M7I_FLEX_LARGE_HOURLY = 0.0958  # USD, Linux on-demand, us-east-1


def lambda_cost_per_1k(latency_ms: float, memory_gb: float = LAMBDA_MEMORY_GB) -> float:
    return 1000 * (LAMBDA_PER_REQUEST + memory_gb * (latency_ms / 1000) * LAMBDA_PER_GB_SECOND)


def ec2_cost_per_1k(req_per_s: float, hourly: float = EC2_M7I_FLEX_LARGE_HOURLY) -> float:
    """Cost per 1k requests when the instance is kept fully busy (best case for EC2)."""
    return 1000 * hourly / 3600 / req_per_s


def main() -> None:
    rows = []
    inf = RESULTS / "inference.csv"
    if inf.exists():
        df = pd.read_csv(inf).set_index("variant")
        lat = float(df.loc["onnx_int8", "latency_ms_p50"])
        n = int(df.loc["onnx_int8", "new_tokens"])
        rows.append(
            {
                "option": "Lambda 2 GB, ForgeGPT INT8 (projected)",
                "basis": f"local p50 latency {lat:.0f} ms for {n} tokens",
                "cost_per_1k_usd": lambda_cost_per_1k(lat),
                "measured_on": "local CPU",
            }
        )
    local = RESULTS / "load_test_forgegpt_short.csv"
    if local.exists():
        lt = pd.read_csv(local)
        best = lt.loc[lt["req_per_s"].idxmax()]
        rows.append(
            {
                "option": "EC2 m7i-flex.large, ForgeGPT INT8 at full load (projected)",
                "basis": f"local max {best.req_per_s:.1f} req/s at {int(best.users)} users",
                "cost_per_1k_usd": ec2_cost_per_1k(best.req_per_s),
                "measured_on": "local CPU (2 vCPU)",
            }
        )
    aws = RESULTS / "load_test_aws_lambda.csv"
    if aws.exists():
        lt = pd.read_csv(aws)
        lat = float(lt.loc[lt["users"] == lt["users"].min(), "p50_ms"].iloc[0])
        rows.append(
            {
                "option": "Lambda 2 GB, ForgeGPT INT8 (measured)",
                "basis": f"AWS p50 {lat:.0f} ms (includes network)",
                "cost_per_1k_usd": lambda_cost_per_1k(lat),
                "measured_on": "AWS",
            }
        )
    for path in sorted(glob.glob(str(RESULTS / "load_test_llamacpp_*.csv"))):
        lt = pd.read_csv(path)
        best = lt.loc[lt["req_per_s"].idxmax()]
        rows.append(
            {
                "option": f"EC2 m7i-flex.large, Qwen2.5-0.5B {Path(path).stem.split('_')[-1]}",
                "basis": f"AWS max throughput {best.req_per_s:.2f} req/s",
                "cost_per_1k_usd": ec2_cost_per_1k(best.req_per_s),
                "measured_on": "AWS",
            }
        )
    if not rows:
        raise SystemExit("no benchmark results found; run the benchmarks first")

    out = pd.DataFrame(rows)
    out["cost_per_1k_usd"] = out["cost_per_1k_usd"].round(5)
    out.to_csv(RESULTS / "cost_per_1k.csv", index=False)
    print(out.to_markdown(index=False))

    # Break-even: EC2 bills every hour it runs, Lambda only per request.
    lam = rows[0]["cost_per_1k_usd"] / 1000
    breakeven = EC2_M7I_FLEX_LARGE_HOURLY / 3600 / lam
    print(
        f"\nAn always-on m7i-flex.large costs ${EC2_M7I_FLEX_LARGE_HOURLY * 730:.0f}/month. "
        f"Lambda is cheaper below ~{breakeven:.1f} requests/s sustained "
        f"(~{breakeven * 86400 / 1e6:.2f}M requests/day)."
    )


if __name__ == "__main__":
    main()
