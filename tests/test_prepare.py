"""End-to-end test of scripts/prepare_data.py on a tiny fake dataset."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def run_prepare(tmp_path, narratives):
    n = len(narratives)
    src = tmp_path / "raw.parquet"
    cols = {"narrative": narratives, "odino": [str(i) for i in range(n)]}
    for c in ["make", "model", "year", "date_received"]:
        cols[c] = [""] * n
    pq.write_table(pa.table(cols), src)
    out = tmp_path / "v1"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/prepare_data.py"), "--src", str(src),
         "--raw-zip", str(tmp_path / "missing.zip"), "--out-dir", str(out)],
        check=True, env=env, capture_output=True,
    )  # fmt: skip
    texts = [
        t
        for s in ["train", "val", "test"]
        for t in pq.read_table(out / f"{s}.parquet")["text"].to_pylist()
    ]
    return texts, json.loads((out / "manifest.json").read_text())


def test_dedup_filtering_and_already_clean_text(tmp_path):
    narratives = [
        "the brakes failed on the highway",  # already clean: must be KEPT
        "THE ENGINE STALLED AT A LIGHT.  *AK",
        "The engine stalled at a light.",  # same as above after cleaning
        "THE ENGINE STALLED AT A LIGHT.  *AK",  # exact repeat (component row)
        "TOO SHORT *TR",
        "",
    ]
    texts, manifest = run_prepare(tmp_path, narratives)
    assert sorted(texts) == ["the brakes failed on the highway", "the engine stalled at a light."]
    assert manifest["dropped_duplicate"] == 2
    assert manifest["dropped_too_short"] == 1
    assert manifest["dropped_empty"] == 1
    assert sum(manifest["rows_out"].values()) == 2
