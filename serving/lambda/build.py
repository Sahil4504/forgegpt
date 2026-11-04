"""Build the Lambda deployment zip (no Docker needed).

    python serving/lambda/build.py --model artifacts/onnx/forgegpt.int8.onnx \
        --tokenizer data/tokenizer/v1/bpe-8k.json

Downloads Linux wheels for the Lambda runtime (Python 3.12, x86_64) with pip's
--platform option, so it works from Windows or macOS too. Output: serving/lambda/dist/
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="artifacts/onnx/forgegpt.int8.onnx")
    p.add_argument("--tokenizer", default="data/tokenizer/v1/bpe-8k.json")
    p.add_argument("--python", default="3.12")
    # Lambda's Python 3.12 runtime is Amazon Linux 2023 (glibc 2.34): manylinux_2_28 is fine.
    p.add_argument(
        "--platform", nargs="+", default=["manylinux_2_28_x86_64", "manylinux2014_x86_64"]
    )
    args = p.parse_args()

    build = HERE / "build"
    dist = HERE / "dist"
    shutil.rmtree(build, ignore_errors=True)
    build.mkdir(parents=True)
    dist.mkdir(exist_ok=True)

    pip = [sys.executable, "-m", "pip", "install", "--quiet", "--target", str(build),
           *[x for plat in args.platform for x in ("--platform", plat)],
           "--python-version", args.python, "--implementation", "cp",
           "--only-binary=:all:"]  # fmt: skip
    subprocess.run([*pip, "onnxruntime", "numpy"], check=True)
    # tokenizers' own dependencies (huggingface_hub, httpx, ...) are only needed to
    # download tokenizers from the Hub; we load a local file, so skip them.
    subprocess.run([*pip, "--no-deps", "tokenizers"], check=True)
    pkg = build / "forgegpt"
    pkg.mkdir()
    for name in ("__init__.py", "onnx_infer.py", "serving.py"):
        shutil.copy(ROOT / "src" / "forgegpt" / name, pkg / name)
    shutil.copy(HERE / "handler.py", build / "handler.py")
    (build / "model").mkdir()
    shutil.copy(ROOT / args.model, build / "model" / "forgegpt.int8.onnx")
    shutil.copy(ROOT / args.tokenizer, build / "model" / "bpe-8k.json")

    # Trim what Lambda never needs.
    for pattern in ("**/__pycache__", "**/tests", "*.dist-info"):
        for path in build.glob(pattern):
            shutil.rmtree(path, ignore_errors=True)

    out = dist / "forgegpt-lambda.zip"
    unzipped = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(build.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(build))
                unzipped += f.stat().st_size
    print(f"{out}: {out.stat().st_size / 1e6:.1f} MB zipped, {unzipped / 1e6:.1f} MB unzipped")
    if unzipped > 250e6:
        raise SystemExit("over Lambda's 250 MB unzipped limit")


if __name__ == "__main__":
    main()
