"""Train byte-level BPE tokenizers on the training split and compare them with GPT-2's.

Run from the repo root, after scripts/prepare_data.py:
    python scripts/train_tokenizer.py

Outputs, in data/tokenizer/v1/:
    bpe-8k.json, bpe-16k.json     the trained tokenizers
    report.json                   tokens-per-narrative comparison on the val split
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer

from forgegpt.tokenizer import PLACEHOLDERS, train_tokenizer

D_MODEL = 384  # planned model width, only used to show the embedding-size cost
EXAMPLE = (
    "the transmission shudders when accelerating from a stop and the dealer says "
    "the torque converter is fine. abs and traction control warning lights came on."
)


def iter_texts(path: Path, limit: int | None = None) -> Iterator[str]:
    """Stream the `text` column so the whole split never has to fit in memory."""
    n = 0
    for batch in pq.ParquetFile(path).iter_batches(columns=["text"], batch_size=50_000):
        for text in batch.column(0).to_pylist():
            if limit is not None and n >= limit:
                return
            n += 1
            yield text


def load_gpt2() -> Tokenizer | None:
    try:
        return Tokenizer.from_pretrained("openai-community/gpt2")
    except Exception as e:  # offline, blocked, etc.: comparison is optional
        print(f"[warn] could not load GPT-2 tokenizer, skipping comparison: {e}")
        return None


def evaluate(tok: Tokenizer, texts: list[str]) -> dict:
    counts = [len(enc.ids) for enc in tok.encode_batch(texts)]
    chars = sum(len(t) for t in texts)
    return {
        "vocab_size": tok.get_vocab_size(),
        "embedding_params": tok.get_vocab_size() * D_MODEL,
        "mean_tokens_per_narrative": round(statistics.mean(counts), 1),
        "median_tokens_per_narrative": statistics.median(counts),
        "chars_per_token": round(chars / sum(counts), 2),
        "example_tokens": tok.encode(EXAMPLE).tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/v1"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/tokenizer/v1"))
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[8192, 16384])
    parser.add_argument("--max-docs", type=int, default=None, help="train on a subset")
    parser.add_argument("--eval-docs", type=int, default=None, help="default: all of val")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.data_dir / "train.parquet"
    n_train = pq.ParquetFile(train_path).metadata.num_rows
    if args.max_docs:
        n_train = min(n_train, args.max_docs)
    val_texts = list(iter_texts(args.data_dir / "val.parquet", args.eval_docs))

    results: dict[str, dict] = {}
    for size in args.vocab_sizes:
        name = f"bpe-{size // 1024}k"
        print(f"\ntraining {name} on {n_train:,} narratives...")
        start = time.time()
        tok = train_tokenizer(iter_texts(train_path, args.max_docs), size, length=n_train)
        tok.save(str(args.out_dir / f"{name}.json"))
        results[name] = evaluate(tok, val_texts)
        results[name]["train_seconds"] = round(time.time() - start)
        assert all(tok.token_to_id(p) is not None for p in PLACEHOLDERS)

    gpt2 = load_gpt2()
    if gpt2 is not None:
        results["gpt2"] = evaluate(gpt2, val_texts)

    report = {"train_docs": n_train, "eval_docs": len(val_texts), "results": results}
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))

    print(f"\n{'tokenizer':<10}{'vocab':>8}{'emb params':>12}{'tok/narr':>10}{'chars/tok':>11}")
    for name, r in results.items():
        print(
            f"{name:<10}{r['vocab_size']:>8,}{r['embedding_params']:>12,}"
            f"{r['mean_tokens_per_narrative']:>10}{r['chars_per_token']:>11}"
        )
    print(f"\nexample: {EXAMPLE!r}")
    for name, r in results.items():
        print(f"\n{name} ({len(r['example_tokens'])} tokens):\n  {r['example_tokens']}")


if __name__ == "__main__":
    main()
