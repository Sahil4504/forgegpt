"""Read the NHTSA complaints flat file (FLAT_CMPL.zip) into Parquet.

Format notes, from NHTSA's data dictionary (static.nhtsa.gov/odi/ffdd/cmpl/CMPL.txt):
- Tab-delimited, no header row, one record per line.
- 51 fields today; older records may be shorter. The narrative is field 20 (CDESCR).
- One line per complaint *component*: a complaint that names three components appears
  three times with the same ODINO and the same narrative. Dedup happens later.

We keep only the fields useful for language modelling and analysis. Personal fields
(VIN, city, dealer details, operator name) are never read, so they never reach disk.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Output column -> 0-based position in the flat file (data-dictionary field number - 1).
COLUMNS: dict[str, int] = {
    "cmplid": 0,  # CMPLID: internal id, may change between releases
    "odino": 1,  # ODINO: complaint number, repeats across components
    "make": 3,  # MAKETXT
    "model": 4,  # MODELTXT
    "year": 5,  # YEARTXT (9999 = unknown)
    "component": 11,  # COMPDESC
    "date_received": 16,  # LDATE, YYYYMMDD
    "narrative": 19,  # CDESCR
}
MIN_FIELDS = max(COLUMNS.values()) + 1

SCHEMA = pa.schema([(name, pa.string()) for name in COLUMNS])


def parse_line(line: str) -> dict[str, str] | None:
    """Turn one flat-file line into a record, or None if it has too few fields."""
    fields = line.rstrip("\r\n").split("\t")
    if len(fields) < MIN_FIELDS:
        return None
    return {name: fields[i].strip() for name, i in COLUMNS.items()}


def iter_lines(zip_path: Path, encoding: str = "utf-8") -> Iterator[str]:
    """Stream text lines out of the zip without extracting it to disk."""
    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".txt")]
        if len(members) != 1:
            raise ValueError(f"expected one .txt file in {zip_path}, found {members}")
        with zf.open(members[0]) as raw:
            # newline="\n" splits only on real line ends, so a stray "\r" inside a
            # narrative can't break one record into two. errors="replace" turns
            # undecodable bytes into U+FFFD, which the download script counts.
            text = io.TextIOWrapper(raw, encoding=encoding, errors="replace", newline="\n")
            yield from text


def write_parquet(
    records: Iterable[dict[str, str]], out_path: Path, batch_size: int = 50_000
) -> int:
    """Write records to Parquet in batches, so the full dataset never sits in memory.

    Writes to a temporary file first and renames at the end, so an interrupted run
    never leaves a half-written file that looks complete.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    n_rows = 0
    batch: list[dict[str, str]] = []
    with pq.ParquetWriter(tmp_path, SCHEMA) as writer:
        for rec in records:
            batch.append(rec)
            if len(batch) >= batch_size:
                writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
                n_rows += len(batch)
                batch = []
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
            n_rows += len(batch)
    tmp_path.replace(out_path)
    return n_rows


# --- token files -------------------------------------------------------------------------
# The training loop reads one flat stream of token ids per split: every narrative's ids
# followed by the end-of-text id. Stored as uint16 (2 bytes per token), which works
# because the vocabulary is smaller than 65,536. Reading uses a memory map, so even a
# multi-GB file is never loaded into RAM: the OS pages in only the slices we touch.

TOKEN_DTYPE = np.uint16


def encode_to_bin(
    texts: Iterable[str], tokenizer, eot_id: int, out_path: Path, batch_size: int = 20_000
) -> tuple[int, int]:
    """Tokenize texts in batches and append them to a .bin file. Returns (docs, tokens)."""
    if tokenizer.get_vocab_size() > np.iinfo(TOKEN_DTYPE).max + 1:
        raise ValueError("vocabulary too large for uint16 token files")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    n_docs = n_tokens = 0

    def flush(batch: list[str], f) -> None:
        nonlocal n_docs, n_tokens
        ids: list[int] = []
        for enc in tokenizer.encode_batch(batch):  # runs in parallel, in Rust
            ids.extend(enc.ids)
            ids.append(eot_id)
        arr = np.asarray(ids, dtype=TOKEN_DTYPE)
        arr.tofile(f)
        n_docs += len(batch)
        n_tokens += len(arr)

    with open(tmp_path, "wb") as f:
        batch: list[str] = []
        for text in texts:
            batch.append(text)
            if len(batch) >= batch_size:
                flush(batch, f)
                batch = []
        if batch:
            flush(batch, f)
    tmp_path.replace(out_path)
    return n_docs, n_tokens


def load_tokens(path: Path) -> np.memmap:
    """Open a .bin token file read-only, without loading it into memory."""
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")
