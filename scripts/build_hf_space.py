"""Build (and optionally publish) a static Hugging Face Space where ForgeGPT runs in the browser.

    python scripts/build_hf_space.py                                # build hf_space/ only
    python scripts/build_hf_space.py --push <hf-username>/forgegpt  # build + upload

A static Space only serves files, which is free and never sleeps. The page
(serving/browser/index.html) downloads the INT8 ONNX model and runs it in the visitor's
browser with ONNX Runtime Web (WebAssembly); the tokenizer and KV-cached generation loop are
a JavaScript port (serving/browser/forgegpt.js) tested to match the Python versions.

ONNX Runtime Web's three files are downloaded once from jsDelivr, checked against pinned
SHA-256 hashes, cached in data/vendor/, and served from the Space itself (no CDN at runtime).
Uploading needs a Hugging Face write token: run `hf auth login` (or `huggingface-cli login`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "hf_space"
GITHUB = "https://github.com/Sahil4504/forgegpt"

ORT_VERSION = "1.30.0"
ORT_FILES = {
    "ort.wasm.min.js": "faece07e84faa9001f5104bbe4e5c70861d06cb2376a0dea1fa0ad33a67adc81",
    "ort-wasm-simd-threaded.mjs": "e13f7f94fc51b4ca72b12faeb1ee95f4ace6dfbc8939bc718aabdc0a27c4299b",  # noqa: E501
    "ort-wasm-simd-threaded.wasm": "3398c10d07d229bd91b364548e130e0e51a8e5704b88c7c083ebbeb78842dee2",  # noqa: E501
}

SPACE_README = """\
---
title: ForgeGPT
emoji: 🚗
colorFrom: blue
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
short_description: GPT from scratch, trained on NHTSA vehicle complaints
---

# ForgeGPT

A {params}M-parameter GPT written from scratch in PyTorch and trained on 1.6 million NHTSA
vehicle complaints (validation perplexity {ppl}). This demo downloads the INT8-quantized ONNX
export ({model_mb} MB) and runs it **entirely in your browser** with ONNX Runtime Web: start a
complaint and the model writes the rest, token by token.

It continues text in the style of the complaints; it is not a chatbot and it invents details
such as recall numbers. Code, training logs and benchmarks: {github}

ONNX Runtime Web {ort} (MIT license, Microsoft) is included in `ort/`.
"""

# Charts shown in the Benchmarks tab (GUI screenshots and AWS cost projections are left out).
CHARTS = [
    "tokenizer_comparison.png",
    "allcaps_by_year.png",
    "training_curves.png",
    "ablations.png",
    "attention_heads.png",
    "kv_cache_scaling.png",
    "inference_variants.png",
    "load_test.png",
    "narrative_lengths.png",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_ort(ort_dir: Path | None) -> Path:
    """Return a folder holding the pinned ONNX Runtime Web files, downloading if needed."""
    cache = ort_dir or ROOT / "data" / "vendor" / f"onnxruntime-web-{ORT_VERSION}"
    cache.mkdir(parents=True, exist_ok=True)
    for name, digest in ORT_FILES.items():
        path = cache / name
        if path.exists() and sha256(path) == digest:
            continue
        mirrors = [
            f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_VERSION}/dist/{name}",
            f"https://unpkg.com/onnxruntime-web@{ORT_VERSION}/dist/{name}",
        ]
        for attempt in range(1, 7):
            url = mirrors[attempt % 2 == 0]  # alternate between the two mirrors
            try:
                print(f"downloading {url}")
                urllib.request.urlretrieve(url, path)
                break
            except OSError as e:
                if attempt == 6:
                    raise
                print(f"  failed ({e}); retrying in {5 * attempt}s")
                time.sleep(5 * attempt)
        if sha256(path) != digest:
            path.unlink()
            raise SystemExit(f"checksum mismatch for {name}; refusing to use it")
    return cache


def model_info(model: Path, summary: Path, charts: list[str]) -> dict:
    import onnxruntime as ort

    meta = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    meta = meta.get_modelmeta().custom_metadata_map
    s = json.loads(summary.read_text())
    return {
        **{k: int(v) for k, v in meta.items() if v.isdigit()},
        "model_file": "forgegpt.int8.onnx",
        "model_mb": round(model.stat().st_size / 1e6, 1),
        "quantization": "INT8 (dynamic)",
        "val_ppl": round(s["best_val_ppl"], 1),
        "train_tokens": s["tokens_seen"],
        "charts": charts,
    }


def build(model: Path, tokenizer: Path, summary: Path, ort_dir: Path | None) -> Path:
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir()
    shutil.copy(ROOT / "serving" / "browser" / "index.html", OUT / "index.html")
    shutil.copy(ROOT / "serving" / "browser" / "forgegpt.js", OUT / "forgegpt.js")
    shutil.copytree(ROOT / "serving" / "app" / "static" / "fonts", OUT / "fonts")
    shutil.copytree(fetch_ort(ort_dir), OUT / "ort")

    (OUT / "model").mkdir()
    shutil.copy(model, OUT / "model" / "forgegpt.int8.onnx")
    shutil.copy(tokenizer, OUT / "model" / "bpe-8k.json")

    (OUT / "charts").mkdir()
    charts = []
    for chart in CHARTS:
        src = ROOT / "docs" / "images" / chart
        if src.exists():
            shutil.copy(src, OUT / "charts" / chart)
            charts.append(chart)

    info = model_info(model, summary, charts)
    (OUT / "model" / "info.json").write_text(json.dumps(info, indent=2))
    (OUT / "README.md").write_text(
        SPACE_README.format(
            params=round(info["num_params"] / 1e6, 1),
            ppl=info["val_ppl"],
            model_mb=info["model_mb"],
            github=GITHUB,
            ort=ORT_VERSION,
        ),
        encoding="utf-8",
    )
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"built {OUT} ({size / 1e6:.1f} MB)")
    return OUT


def with_retries(step: str, fn, attempts: int = 6):
    """Retry a network step: flaky connections reset some TLS handshakes but not others."""
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:  # connection resets surface as several exception types
            if i == attempts:
                raise
            wait = min(5 * i, 30)
            print(f"{step} failed ({type(e).__name__}); retry {i}/{attempts - 1} in {wait}s")
            time.sleep(wait)


def push(folder: Path, repo_id: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    with_retries(
        "create Space",
        lambda: api.create_repo(repo_id, repo_type="space", space_sdk="static", exist_ok=True),
    )
    # Re-running is safe: files already on the Hub with the same content are not re-sent.
    with_retries(
        "upload",
        lambda: api.upload_folder(
            folder_path=str(folder),
            repo_id=repo_id,
            repo_type="space",
            commit_message="Deploy ForgeGPT in-browser demo",
        ),
    )
    print(f"uploaded: https://huggingface.co/spaces/{repo_id}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=Path, default=ROOT / "artifacts/onnx/forgegpt.int8.onnx")
    p.add_argument("--tokenizer", type=Path, default=ROOT / "data/tokenizer/v1/bpe-8k.json")
    p.add_argument("--summary", type=Path, default=ROOT / "runs/kaggle_t4/summary.json")
    p.add_argument("--ort-dir", type=Path, help="folder that already has the ORT Web files")
    p.add_argument("--push", metavar="USER/SPACE", help="Hugging Face Space to upload to")
    args = p.parse_args()
    folder = build(args.model, args.tokenizer, args.summary, args.ort_dir)
    if args.push:
        push(folder, args.push)


if __name__ == "__main__":
    main()
