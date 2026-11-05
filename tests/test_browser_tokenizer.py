"""The browser demo's JavaScript tokenizer must produce exactly the same ids as Python's.

Runs serving/browser/forgegpt.js under Node; skipped when Node is not installed.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from forgegpt.tokenizer import train_tokenizer

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "serving" / "browser" / "forgegpt.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

TEXTS = [
    "while driving at 65 mph the brakes failed!!",
    "the contact owns a 2019 ford f-150. the contact stated that the vehicle stalled.",
    "  leading and   multiple   spaces  ",
    "tabs\tand\nnewlines\n\n",
    "it's i'm we're they've you'll he'd",
    "call me at [phone] or [email], vin [vin] see [url] [xxx].",
    "emoji ⚠️ and café 90° → ½",
    "numbers 1234567 and 3.5l v6 2019",
    "<|endoftext|>start of a new complaint",
    "",
]


def test_js_tokenizer_matches_python(tmp_path):
    tok = train_tokenizer(TEXTS * 30, vocab_size=384)
    tok_path = tmp_path / "tok.json"
    tok.save(str(tok_path))
    cases = [
        {"text": t, "ids": tok.encode(t).ids, "decoded": tok.decode(tok.encode(t).ids)}
        for t in TEXTS
    ]
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    script = f"""
import fs from "node:fs";
import {{ Tokenizer }} from {json.dumps(JS.as_uri())};
const tok = new Tokenizer(JSON.parse(fs.readFileSync({json.dumps(str(tok_path))}, "utf8")));
const cases = JSON.parse(fs.readFileSync({json.dumps(str(tmp_path / "cases.json"))}, "utf8"));
const out = cases.map((c) => ({{ ids: tok.encode(c.text), decoded: tok.decode(c.ids) }}));
console.log(JSON.stringify(out));
"""
    res = subprocess.run(
        ["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=True
    )
    js = json.loads(res.stdout)
    for case, got in zip(cases, js, strict=True):
        assert got["ids"] == case["ids"], case["text"]
        assert got["decoded"] == case["decoded"], case["text"]
