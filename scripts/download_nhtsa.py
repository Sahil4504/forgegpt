"""Download the NHTSA complaints flat file and extract the useful columns to Parquet.

Run from the repo root:
    python scripts/download_nhtsa.py

Outputs:
    data/raw/FLAT_CMPL.zip                  untouched source file (resumable download)
    data/interim/complaints_raw.parquet     selected columns, one row per flat-file line

It also prints a summary of the data, which we use to decide the cleaning rules.
"""

from __future__ import annotations

import argparse
import http.client
import random
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from forgegpt.data import iter_lines, parse_line, write_parquet

URL = "https://static.nhtsa.gov/odi/ffdd/cmpl/FLAT_CMPL.zip"


def download(url: str, dest: Path, retries: int = 5) -> None:
    """Download with resume: progress is kept in a .part file and continued on retry."""
    if dest.exists():
        print(f"{dest} already exists, skipping download")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "forgegpt-data/0.1"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if have and resp.status != 206:  # server ignored the Range header
                    have = 0
                total = (resp.length + have) if resp.length else None
                mode = "ab" if have else "wb"
                with (
                    open(part, mode) as f,
                    tqdm(
                        total=total, initial=have, unit="B", unit_scale=True, desc=dest.name
                    ) as bar,
                ):
                    while chunk := resp.read(1 << 20):
                        f.write(chunk)
                        bar.update(len(chunk))
            part.replace(dest)
            return
        except urllib.error.HTTPError as e:
            if e.code == 416:  # nothing left to fetch: the .part file is complete
                part.replace(dest)
                return
            if 400 <= e.code < 500 and e.code not in (408, 429):
                raise SystemExit(f"download failed with HTTP {e.code}: {url}") from e
            print(f"attempt {attempt}/{retries} failed: HTTP {e.code}")
        except (OSError, http.client.HTTPException) as e:
            print(f"attempt {attempt}/{retries} failed: {e}")
        time.sleep(min(2**attempt, 30))
    raise SystemExit("download failed; rerun the script and it will resume")


class Stats:
    """Summary numbers gathered while streaming, so we never load the whole file."""

    def __init__(self, n_samples: int = 5, seed: int = 0):
        self.lines = 0
        self.malformed = 0
        self.rows = 0
        self.empty = 0
        self.odinos: set[str] = set()
        self.narrative_hashes: set[int] = set()
        self.chars = 0
        self.max_chars = 0
        self.replacement_chars = 0
        self.by_year: Counter[str] = Counter()
        self.upper_by_year: Counter[str] = Counter()
        self.samples: list[str] = []
        self.n_samples = n_samples
        self.rng = random.Random(seed)

    def add(self, rec: dict[str, str]) -> None:
        self.rows += 1
        text = rec["narrative"]
        self.odinos.add(rec["odino"])
        if not text:
            self.empty += 1
            return
        self.narrative_hashes.add(hash(text))
        self.chars += len(text)
        self.max_chars = max(self.max_chars, len(text))
        self.replacement_chars += text.count("�")
        year = rec["date_received"][:4] or "????"
        self.by_year[year] += 1
        if text.isupper():  # every cased letter is uppercase
            self.upper_by_year[year] += 1
        # Reservoir sampling: a uniform random sample without storing everything.
        if len(self.samples) < self.n_samples:
            self.samples.append(text)
        else:
            j = self.rng.randrange(self.rows)
            if j < self.n_samples:
                self.samples[j] = text

    def report(self) -> str:
        nonempty = self.rows - self.empty
        out = [
            "",
            "=== NHTSA complaints: raw summary ===",
            f"lines read                {self.lines:>12,}",
            f"malformed (skipped)       {self.malformed:>12,}",
            f"rows written              {self.rows:>12,}",
            f"unique complaints (ODINO) {len(self.odinos):>12,}",
            f"unique narratives         {len(self.narrative_hashes):>12,}",
            f"empty narratives          {self.empty:>12,}",
            f"avg chars                 {self.chars / max(nonempty, 1):>12,.0f}",
            f"max chars                 {self.max_chars:>12,}",
            f"undecodable chars (U+FFFD){self.replacement_chars:>12,}",
            "",
            "received year   rows      all-caps %",
        ]
        for year in sorted(self.by_year):
            n = self.by_year[year]
            pct = 100 * self.upper_by_year[year] / n
            out.append(f"{year:<12}{n:>10,}   {pct:>8.1f}")
        out += ["", "random samples (first 300 chars):"]
        for i, s in enumerate(self.samples, 1):
            out.append(f"[{i}] {s[:300]}")
        return "\n".join(out)


def records(zip_path: Path, encoding: str, stats: Stats):
    for line in iter_lines(zip_path, encoding=encoding):
        stats.lines += 1
        rec = parse_line(line)
        if rec is None:
            stats.malformed += 1
            continue
        stats.add(rec)
        yield rec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=URL)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/interim/complaints_raw.parquet"))
    parser.add_argument("--encoding", default="utf-8", help="try cp1252 if U+FFFD count is high")
    args = parser.parse_args()

    zip_path = args.raw_dir / "FLAT_CMPL.zip"
    download(args.url, zip_path)

    stats = Stats()
    stream = tqdm(records(zip_path, args.encoding, stats), unit=" rows", desc="parsing")
    n = write_parquet(stream, args.out)
    print(f"\nwrote {n:,} rows to {args.out}")
    print(stats.report())


if __name__ == "__main__":
    main()
