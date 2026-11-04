"""CPU inference benchmark: PyTorch vs ONNX Runtime, FP32 vs INT8, with/without KV cache.

    python scripts/benchmark_inference.py --ckpt runs/cpu_demo/ckpt_best.pt

Each variant runs in its own subprocess so memory numbers are not polluted by the others.
Writes benchmarks/results/inference.csv, cache_scaling.csv and machine.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

VARIANTS = ["torch_fp32_nocache", "torch_fp32", "torch_int8", "onnx_fp32", "onnx_int8"]
LABELS = {
    "torch_fp32_nocache": "PyTorch FP32, no KV cache",
    "torch_fp32": "PyTorch FP32 + KV cache",
    "torch_int8": "PyTorch INT8 (dynamic) + KV cache",
    "onnx_fp32": "ONNX Runtime FP32 + KV cache",
    "onnx_int8": "ONNX Runtime INT8 + KV cache",
}


def peak_rss_mb() -> float:
    try:
        import resource  # Linux/macOS

        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return kb / 1024 if sys.platform != "darwin" else kb / 1024**2
    except ImportError:  # Windows
        import psutil

        return psutil.Process().memory_info().peak_wset / 1024**2


def test_windows(test_bin: str, block: int, n: int, seed: int = 0) -> np.ndarray:
    data = np.memmap(test_bin, dtype=np.uint16, mode="r")
    starts = np.random.default_rng(seed).integers(0, len(data) - block - 1, size=n)
    return np.stack([data[s : s + block + 1] for s in starts]).astype(np.int64)


def prompts_from_test(test_bin: str, n: int, length: int) -> list[list[int]]:
    """The first `length` tokens of n test narratives (split on <|endoftext|> = 0)."""
    data = np.asarray(np.memmap(test_bin, dtype=np.uint16, mode="r")[:2_000_000])
    starts = np.flatnonzero(data == 0)[:-1] + 1
    out = [data[s : s + length].tolist() for s in starts if 0 not in data[s : s + length]]
    return out[:: max(1, len(out) // n)][:n]


def worker(args: argparse.Namespace) -> dict:
    """Benchmark one variant in this process and return a result row."""
    variant = args.worker
    t0 = time.perf_counter()
    if variant.startswith("torch"):
        import torch  # imported only here: ONNX variants never load PyTorch

        from forgegpt.generate import SamplingConfig, timed_generation
        from forgegpt.onnx_export import load_checkpoint

        torch.set_num_threads(args.threads)
        model = load_checkpoint(args.ckpt)
        if variant == "torch_int8":
            model = torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        tmp = Path(args.out) / f"_{variant}_weights.pt"
        torch.save(model.state_dict(), tmp)  # weights only (the checkpoint also has Adam state)
        size_mb = tmp.stat().st_size / 1e6
        tmp.unlink()
        block = model.cfg.block_size
        cfg = SamplingConfig(max_new_tokens=args.new_tokens, temperature=0.0, stop_at_eot=False)
        use_cache = variant != "torch_fp32_nocache"

        def run(p):
            return timed_generation(model, p, cfg, use_cache=use_cache)

        @torch.no_grad()
        def window_loss(w):
            x = torch.from_numpy(w[None, :-1])
            y = torch.from_numpy(w[None, 1:])
            return model(x, y)[1].item()
    else:
        from forgegpt.onnx_infer import OnnxGenerator, Sampling

        path = Path(args.onnx_dir) / (
            "forgegpt.onnx" if variant == "onnx_fp32" else "forgegpt.int8.onnx"
        )
        gen = OnnxGenerator(str(path), args.tokenizer, threads=args.threads)
        size_mb = path.stat().st_size / 1e6
        block = gen.block_size
        s = Sampling(max_new_tokens=args.new_tokens, temperature=0.0, ignore_eot=True)

        def run(p):
            start = time.perf_counter()
            first, out = None, []
            for tok in gen.stream_ids(p, s):
                first = first or time.perf_counter()
                out.append(tok)
            end = time.perf_counter()
            return {
                "tokens": out,
                "ttft_ms": 1000 * (first - start),
                "total_ms": 1000 * (end - start),
                "decode_tok_per_s": (len(out) - 1) / (end - first),
            }

        def window_loss(w):
            logits, _ = gen.session.run(
                None, {"input_ids": w[None, :-1], "past": gen._empty_past()}
            )
            logits = logits[0].astype(np.float64)
            logz = np.log(np.exp(logits - logits.max(-1, keepdims=True)).sum(-1)) + logits.max(-1)
            return float(np.mean(logz - logits[np.arange(len(w) - 1), w[1:]]))

    load_s = time.perf_counter() - t0
    prompts = prompts_from_test(args.test_bin, args.n_prompts, args.prompt_len)
    run(prompts[0])  # warm-up
    results = [run(p) for p in prompts]
    windows = test_windows(args.test_bin, block, args.ppl_windows)
    loss = float(np.mean([window_loss(w) for w in windows]))
    lat = np.array([r["total_ms"] for r in results])
    return {
        "variant": variant,
        "label": LABELS[variant],
        "model_mb": round(size_mb, 1),
        "load_s": round(load_s, 2),
        "peak_rss_mb": round(peak_rss_mb(), 0),
        "ttft_ms_p50": round(float(np.median([r["ttft_ms"] for r in results])), 2),
        "decode_tok_per_s": round(float(np.mean([r["decode_tok_per_s"] for r in results])), 1),
        "latency_ms_p50": round(float(np.percentile(lat, 50)), 1),
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 1),
        "test_loss": round(loss, 4),
        "test_ppl": round(float(np.exp(loss)), 2),
        "new_tokens": args.new_tokens,
        "first_completion_ids": results[0]["tokens"][:20],
    }


def cache_scaling(args: argparse.Namespace) -> list[dict]:
    """Per-token latency as the sequence grows, with and without the KV cache."""
    import torch

    from forgegpt.generate import SamplingConfig, stream
    from forgegpt.onnx_export import load_checkpoint

    torch.set_num_threads(args.threads)
    model = load_checkpoint(args.ckpt)
    n_new = model.cfg.block_size - args.prompt_len
    cfg = SamplingConfig(max_new_tokens=n_new, temperature=0.0, stop_at_eot=False)
    prompt = prompts_from_test(args.test_bin, 1, args.prompt_len)[0]
    rows = []
    for use_cache in (False, True):
        for rep in range(3):
            t = time.perf_counter()
            for i, _ in enumerate(stream(model, prompt, cfg, use_cache=use_cache)):
                now = time.perf_counter()
                rows.append(
                    {
                        "kv_cache": use_cache,
                        "rep": rep,
                        "position": args.prompt_len + i + 1,
                        "ms": 1000 * (now - t),
                    }
                )
                t = now
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", default="runs/cpu_demo/ckpt_best.pt")
    p.add_argument("--tokenizer", default="data/tokenizer/v1/bpe-8k.json")
    p.add_argument("--test-bin", default="data/tokens/v1/test.bin")
    p.add_argument("--onnx-dir", default="artifacts/onnx")
    p.add_argument("--out", default="benchmarks/results")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--n-prompts", type=int, default=30)
    p.add_argument("--prompt-len", type=int, default=16)
    p.add_argument("--new-tokens", type=int, default=64)
    p.add_argument("--ppl-windows", type=int, default=200)
    p.add_argument("--worker", choices=VARIANTS + ["cache_scaling"])
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.worker == "cache_scaling":
        print(json.dumps(cache_scaling(args)))
        return
    if args.worker:
        print(json.dumps(worker(args)))
        return

    import pandas as pd

    if not (Path(args.onnx_dir) / "forgegpt.int8.onnx").exists():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "forgegpt.onnx_export",
                "--ckpt",
                args.ckpt,
                "--out-dir",
                args.onnx_dir,
            ],
            check=True,
        )
    base = [sys.executable, __file__] + [
        f"--{k.replace('_', '-')}={v}" for k, v in vars(args).items() if k != "worker"
    ]
    rows = []
    for v in VARIANTS:
        print(f"benchmarking {LABELS[v]}...", flush=True)
        res = subprocess.run(base + [f"--worker={v}"], check=True, capture_output=True, text=True)
        rows.append(json.loads(res.stdout.strip().splitlines()[-1]))
        print(
            {
                k: rows[-1][k]
                for k in ("ttft_ms_p50", "decode_tok_per_s", "latency_ms_p50", "test_ppl")
            }
        )
    df = pd.DataFrame(rows)
    df.drop(columns=["first_completion_ids"]).to_csv(out / "inference.csv", index=False)

    res = subprocess.run(
        base + ["--worker=cache_scaling"], check=True, capture_output=True, text=True
    )
    pd.DataFrame(json.loads(res.stdout.strip().splitlines()[-1])).to_csv(
        out / "cache_scaling.csv", index=False
    )

    machine = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "threads_used": args.threads,
        "python": platform.python_version(),
    }
    (out / "machine.json").write_text(json.dumps(machine, indent=2))
    cols = [
        "label",
        "model_mb",
        "peak_rss_mb",
        "ttft_ms_p50",
        "decode_tok_per_s",
        "latency_ms_p50",
        "test_ppl",
    ]
    print(df[cols].to_markdown(index=False))


if __name__ == "__main__":
    main()
