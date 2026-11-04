"""Summarise training runs (e.g. ablations) into a Markdown table.

python scripts/summarize_runs.py runs/ --out docs/ablations.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rows = []
    for cfg_path in sorted(args.runs_dir.glob("*/config.json")):
        run = cfg_path.parent
        cfg = json.loads(cfg_path.read_text())
        if not (run / "eval.csv").exists():
            continue
        ev = pd.read_csv(run / "eval.csv")
        best = ev["val_loss"].min()
        m, t = cfg["model"], cfg["train"]
        rows.append(
            {
                "run": run.name,
                "layers": m["n_layer"],
                "width": m["d_model"],
                "context": m["block_size"],
                "lr": t["lr"],
                "tokens (M)": round(ev["tokens_seen"].max() / 1e6, 1),
                "best val loss": round(best, 3),
                "val ppl": round(math.exp(best), 1),
            }
        )
    table = pd.DataFrame(rows).sort_values("best val loss").to_markdown(index=False)
    print(table)
    if args.out:
        args.out.write_text(table + "\n")


if __name__ == "__main__":
    main()
