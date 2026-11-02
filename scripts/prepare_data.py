"""Clean, deduplicate and split the complaints into train/val/test Parquet files.

Run from the repo root, after scripts/download_nhtsa.py:
    python scripts/prepare_data.py

Outputs, in data/processed/v1/:
    train.parquet, val.parquet, test.parquet
        columns: text, odino, make, model, year, date_received
    manifest.json
        counts, how often each cleaning rule fired, source checksum
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from forgegpt.clean import MIN_CHARS, RULES, clean_narrative

SPLITS = {"train": 0.90, "val": 0.05, "test": 0.05}
META_COLUMNS = ["odino", "make", "model", "year", "date_received"]
SCHEMA = pa.schema([("text", pa.string())] + [(c, pa.string()) for c in META_COLUMNS])


def digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def split_for(text_digest: bytes) -> str:
    """Pick a split from the text's hash.

    The same narrative always lands in the same split, on any machine and in any
    run, and we never have to shuffle the whole dataset in memory.
    """
    bucket = int.from_bytes(text_digest[:8], "big") % 10_000
    edge = 0.0
    for name, frac in SPLITS.items():
        edge += frac * 10_000
        if bucket < edge:
            return name
    return name  # rounding guard: the last split


class SplitWriter:
    """One Parquet writer per split, fed in batches."""

    def __init__(self, out_dir: Path, batch_size: int = 50_000):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.paths = {s: out_dir / f"{s}.parquet" for s in SPLITS}
        self.writers = {s: pq.ParquetWriter(self._tmp(s), SCHEMA) for s in SPLITS}
        self.buffers: dict[str, list[dict]] = {s: [] for s in SPLITS}
        self.rows = Counter()
        self.chars = Counter()
        self.batch_size = batch_size

    def _tmp(self, split: str) -> Path:
        return self.paths[split].with_name(self.paths[split].name + ".tmp")

    def add(self, split: str, row: dict) -> None:
        self.buffers[split].append(row)
        self.rows[split] += 1
        self.chars[split] += len(row["text"])
        if len(self.buffers[split]) >= self.batch_size:
            self._flush(split)

    def _flush(self, split: str) -> None:
        if self.buffers[split]:
            self.writers[split].write_table(pa.Table.from_pylist(self.buffers[split], SCHEMA))
            self.buffers[split] = []

    def close(self) -> None:
        for s in SPLITS:
            self._flush(s)
            self.writers[s].close()
            self._tmp(s).replace(self.paths[s])  # only now do the final files appear


def sha256_of(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, default=Path("data/interim/complaints_raw.parquet"))
    parser.add_argument("--raw-zip", type=Path, default=Path("data/raw/FLAT_CMPL.zip"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/v1"))
    args = parser.parse_args()

    pf = pq.ParquetFile(args.src)
    writer = SplitWriter(args.out_dir)
    seen_raw: set[bytes] = set()  # skips exact repeats cheaply, before cleaning
    seen_clean: set[bytes] = set()  # the real dedup, on cleaned text
    rule_hits: Counter[str] = Counter()
    n_in = n_empty = n_short = n_dup = 0
    samples: list[str] = []
    rng = random.Random(0)

    with tqdm(total=pf.metadata.num_rows, unit=" rows", desc="cleaning") as bar:
        for batch in pf.iter_batches(columns=["narrative", *META_COLUMNS], batch_size=100_000):
            cols = batch.to_pydict()
            for i, raw in enumerate(cols["narrative"]):
                n_in += 1
                if not raw:
                    n_empty += 1
                    continue
                # Dedup BEFORE counting rule hits, so the counts describe unique text.
                key = digest(raw)
                if key in seen_raw:
                    n_dup += 1
                    continue
                seen_raw.add(key)
                text = clean_narrative(raw, rule_hits)
                if len(text) < MIN_CHARS:
                    n_short += 1
                    continue
                # Second dedup on the cleaned text: narratives that only differed
                # by case, tags or whitespace collapse here.
                key = digest(text)
                if key in seen_clean:
                    n_dup += 1
                    continue
                seen_clean.add(key)
                row = {"text": text, **{c: cols[c][i] for c in META_COLUMNS}}
                writer.add(split_for(key), row)
                kept = sum(writer.rows.values())
                if len(samples) < 5:
                    samples.append(text)
                elif (j := rng.randrange(kept)) < 5:
                    samples[j] = text
            bar.update(batch.num_rows)
    writer.close()

    manifest = {
        "version": args.out_dir.name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_zip_sha256": sha256_of(args.raw_zip),
        "min_chars": MIN_CHARS,
        "split_method": "blake2b(cleaned text) bucket: train 90 / val 5 / test 5",
        "rows_in": n_in,
        "dropped_empty": n_empty,
        "dropped_duplicate": n_dup,
        "dropped_too_short": n_short,
        "rows_out": dict(writer.rows),
        "chars_out": dict(writer.chars),
        "rule_hits_on_unique_raw": {name: rule_hits[name] for name, _ in RULES},
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(json.dumps(manifest, indent=2))
    print("\nrandom cleaned samples (first 300 chars):")
    for k, s in enumerate(samples, 1):
        print(f"[{k}] {s[:300]}")


if __name__ == "__main__":
    main()
