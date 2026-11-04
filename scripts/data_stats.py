"""Small CSV summaries of the corpus, used for the README charts.

    python scripts/data_stats.py

Writes docs/data_stats/by_year.csv (narratives and ALL-CAPS share per received year, from
the raw extract) and docs/data_stats/lengths.csv (cleaned narrative lengths, val split).
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/interim/complaints_raw.parquet")
    p.add_argument("--val", default="data/processed/v1/val.parquet")
    p.add_argument("--tokenizer", default="data/tokenizer/v1/bpe-8k.json")
    p.add_argument("--out", default="docs/data_stats")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    seen: set[int] = set()
    total, caps = Counter(), Counter()
    for batch in pq.ParquetFile(args.raw).iter_batches(
        columns=["narrative", "date_received"], batch_size=200_000
    ):
        cols = batch.to_pydict()
        for text, date in zip(cols["narrative"], cols["date_received"], strict=True):
            if not text or (h := hash(text)) in seen:
                continue
            seen.add(h)
            year = date[:4]
            total[year] += 1
            caps[year] += text.isupper()
    years = sorted(y for y in total if y.isdigit())
    pd.DataFrame(
        {
            "year": years,
            "narratives": [total[y] for y in years],
            "allcaps_share": [round(caps[y] / total[y], 4) for y in years],
        }
    ).to_csv(out / "by_year.csv", index=False)

    texts = pq.read_table(args.val, columns=["text"])["text"].to_pylist()
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)
    n_tokens = [len(e.ids) for e in tok.encode_batch(texts)]
    pd.DataFrame({"chars": [len(t) for t in texts], "tokens": n_tokens}).to_csv(
        out / "lengths.csv", index=False
    )
    print(
        f"by_year: {len(years)} years; lengths: {len(texts):,} narratives, "
        f"median {int(np.median(n_tokens))} tokens"
    )


if __name__ == "__main__":
    main()
