"""Tokenize the train/val/test splits into flat uint16 token files for training.

Run from the repo root, after scripts/train_tokenizer.py:
    python scripts/encode_data.py

Outputs, in data/tokens/v1/:
    train.bin, val.bin, test.bin   token ids, each narrative followed by <|endoftext|>
    meta.json                      tokenizer used (with checksum), counts per split
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pyarrow.parquet as pq

from forgegpt.data import TOKEN_DTYPE, encode_to_bin
from forgegpt.tokenizer import EOT, load_tokenizer

SPLITS = ["train", "val", "test"]


def iter_texts(path: Path):
    for batch in pq.ParquetFile(path).iter_batches(columns=["text"], batch_size=50_000):
        yield from batch.column(0).to_pylist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/v1"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/v1/bpe-8k.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/tokens/v1"))
    parser.add_argument("--splits", nargs="+", default=SPLITS)
    args = parser.parse_args()

    tok = load_tokenizer(args.tokenizer)
    eot_id = tok.token_to_id(EOT)
    meta_path = args.out_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"splits": {}}
    meta.update(
        {
            "tokenizer_file": args.tokenizer.name,
            "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
            "vocab_size": tok.get_vocab_size(),
            "eot_id": eot_id,
            "dtype": TOKEN_DTYPE.__name__,
        }
    )

    for split in args.splits:
        start = time.time()
        n_docs, n_tokens = encode_to_bin(
            iter_texts(args.data_dir / f"{split}.parquet"),
            tok,
            eot_id,
            args.out_dir / f"{split}.bin",
        )
        meta["splits"][split] = {"docs": n_docs, "tokens": n_tokens}
        print(f"{split:<6}{n_docs:>11,} docs{n_tokens:>14,} tokens  ({time.time() - start:.0f}s)")
        meta_path.write_text(json.dumps(meta, indent=2))  # saved after each split

    print(f"\nwrote {args.out_dir}/meta.json")


if __name__ == "__main__":
    main()
