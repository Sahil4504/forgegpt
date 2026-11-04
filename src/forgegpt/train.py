"""Training loop for ForgeGPT.

Run:
    python -m forgegpt.train --config configs/kaggle_t4.json
    python -m forgegpt.train --config configs/cpu_demo.json --max_iters 50  # override any field

Features: random-crop batches from memory-mapped token files, AdamW with weight decay on
matrices only, linear warmup + cosine decay, gradient accumulation and clipping, mixed
precision on GPU, checkpoint + resume (Kaggle sessions time out), a wall-clock budget,
CSV logs (MLflow optional), and a text sample at every evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch

from forgegpt.generate import SamplingConfig, generate
from forgegpt.model import GPT, GPTConfig


@dataclass
class TrainConfig:
    # data
    data_dir: str = "data/tokens/v1"
    train_file: str = "train.bin"
    val_file: str = "val.bin"
    tokenizer: str = "data/tokenizer/v1/bpe-8k.json"
    out_dir: str = "runs/default"
    # model
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    block_size: int = 256
    dropout: float = 0.0
    # optimisation
    batch_size: int = 64
    grad_accum: int = 1  # effective batch = batch_size * grad_accum sequences
    max_iters: int = 10_000
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_iters: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # bookkeeping
    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 10
    max_minutes: float | None = None  # stop cleanly (with a checkpoint) after this long
    device: str = "auto"  # auto | cpu | cuda
    dtype: str = "auto"  # auto | float32 | bfloat16 | float16
    compile: bool = False
    seed: int = 1337
    resume: bool = True
    mlflow: bool = False
    sample_prompt: str = "the vehicle"
    extra: dict = field(default_factory=dict)  # free-form notes saved with the run


def load_config(path: str | None, overrides: dict) -> TrainConfig:
    values = json.loads(Path(path).read_text()) if path else {}
    values.update({k: v for k, v in overrides.items() if v is not None})
    return TrainConfig(**values)


def resolve_device_dtype(cfg: TrainConfig) -> tuple[str, torch.dtype]:
    device = (
        cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if cfg.dtype != "auto":
        return device, getattr(torch, cfg.dtype)
    if device == "cuda":
        # bf16 needs Ampere+ (A100, L4...). Kaggle's T4/P100 use fp16 + a gradient scaler.
        return device, torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return device, torch.float32


class Batches:
    """Random crops of `block_size + 1` tokens from a memory-mapped token file."""

    def __init__(self, path: Path, block_size: int, batch_size: int, device: str, seed: int):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block, self.batch, self.device = block_size, batch_size, device
        self.rng = np.random.default_rng(seed)

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        starts = self.rng.integers(0, len(self.data) - self.block - 1, size=self.batch)
        chunk = np.stack([self.data[s : s + self.block + 1] for s in starts]).astype(np.int64)
        x = torch.from_numpy(chunk[:, :-1])  # inputs:  tokens 0..T-1
        y = torch.from_numpy(chunk[:, 1:])  # targets: tokens 1..T  (the "next token")
        if self.device == "cuda":
            return x.pin_memory().to("cuda", non_blocking=True), y.pin_memory().to(
                "cuda", non_blocking=True
            )
        return x, y


def lr_at(it: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay from lr down to min_lr at max_iters."""
    if it < cfg.warmup_iters:
        return cfg.lr * (it + 1) / cfg.warmup_iters
    progress = min(1.0, (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters))
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (cfg.lr - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model: GPT, splits: dict[str, Batches], iters: int, autocast) -> dict:
    model.eval()
    out = {}
    for name, batches in splits.items():
        losses = []
        for _ in range(iters):
            x, y = batches.get()
            with autocast:
                _, loss = model(x, y)
            losses.append(loss.item())
        out[name] = float(np.mean(losses))
    model.train()
    return out


def append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def train(cfg: TrainConfig) -> dict:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    device, dtype = resolve_device_dtype(cfg)
    autocast = (
        torch.autocast(device_type=device, dtype=dtype) if dtype != torch.float32 else nullcontext()
    )
    scaler = torch.amp.GradScaler(device, enabled=(dtype == torch.float16))

    data_dir = Path(cfg.data_dir)
    meta = json.loads((data_dir / "meta.json").read_text())
    model_cfg = GPTConfig(
        vocab_size=meta["vocab_size"],
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        d_model=cfg.d_model,
        dropout=cfg.dropout,
    )
    model = GPT(model_cfg).to(device)
    opt = model.configure_optimizer(cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), device)

    start_iter, best_val, tokens_seen = 0, float("inf"), 0
    latest = out / "ckpt_latest.pt"
    if cfg.resume and latest.exists():
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_iter, best_val = ckpt["iter"] + 1, ckpt["best_val"]
        tokens_seen = ckpt.get("tokens_seen", 0)
        print(f"resumed from {latest} at iter {start_iter}")

    (out / "config.json").write_text(
        json.dumps({"train": asdict(cfg), "model": model_cfg.to_dict()}, indent=2)
    )
    print(
        f"device={device} dtype={dtype} params={model.num_params():,} "
        f"(non-embedding {model.num_params(non_embedding=True):,})"
    )

    raw_model = model
    if cfg.compile:
        model = torch.compile(model)

    train_b = Batches(data_dir / cfg.train_file, cfg.block_size, cfg.batch_size, device, cfg.seed)
    val_b = Batches(data_dir / cfg.val_file, cfg.block_size, cfg.batch_size, device, cfg.seed + 1)
    eval_train_b = Batches(
        data_dir / cfg.train_file, cfg.block_size, cfg.batch_size, device, cfg.seed + 2
    )

    tokenizer = None
    if Path(cfg.tokenizer).exists():
        from forgegpt.tokenizer import load_tokenizer

        tokenizer = load_tokenizer(cfg.tokenizer)

    mlflow = None
    if cfg.mlflow:
        import mlflow  # optional dependency

        mlflow.set_experiment("forgegpt")
        mlflow.start_run(run_name=out.name)
        mlflow.log_params({**asdict(cfg), "params": raw_model.num_params()})

    def save(path: Path, it: int) -> None:
        torch.save(
            {
                "model": raw_model.state_dict(),
                "optimizer": opt.state_dict(),
                "model_config": model_cfg.to_dict(),
                "iter": it,
                "best_val": best_val,
                "tokens_seen": tokens_seen,
            },
            path,
        )

    tokens_per_iter = cfg.batch_size * cfg.grad_accum * cfg.block_size
    t0 = time.time()
    t_log = time.time()
    it = start_iter
    model.train()
    x, y = train_b.get()
    for it in range(start_iter, cfg.max_iters):
        lr = lr_at(it, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        last = it == cfg.max_iters - 1
        out_of_time = cfg.max_minutes is not None and (time.time() - t0) / 60 > cfg.max_minutes
        if it % cfg.eval_interval == 0 or last or out_of_time:
            losses = estimate_loss(
                model, {"train": eval_train_b, "val": val_b}, cfg.eval_iters, autocast
            )
            sample = ""
            if tokenizer is not None:
                ids = generate(
                    raw_model,
                    tokenizer.encode(cfg.sample_prompt).ids,
                    SamplingConfig(max_new_tokens=60),
                    seed=0,
                )
                sample = cfg.sample_prompt + tokenizer.decode(ids)
                with open(out / "samples.txt", "a", encoding="utf-8") as f:
                    f.write(f"--- iter {it} | val {losses['val']:.3f}\n{sample}\n\n")
            row = {
                "iter": it,
                "tokens_seen": tokens_seen,
                "train_loss": round(losses["train"], 4),
                "val_loss": round(losses["val"], 4),
                "val_ppl": round(math.exp(losses["val"]), 2),
                "lr": lr,
                "elapsed_min": round((time.time() - t0) / 60, 2),
            }
            append_csv(out / "eval.csv", row)
            if mlflow:
                mlflow.log_metrics({k: v for k, v in row.items() if k != "iter"}, step=it)
            print(
                f"[eval] iter {it}: train {losses['train']:.4f} val {losses['val']:.4f} "
                f"(ppl {row['val_ppl']}) | {sample[:100]!r}"
            )
            if losses["val"] < best_val:
                best_val = losses["val"]
                save(out / "ckpt_best.pt", it)
            save(latest, it)
            if out_of_time:
                print(f"time budget of {cfg.max_minutes} min reached, stopping")
                break

        # One optimisation step, accumulated over grad_accum micro-batches.
        for _ in range(cfg.grad_accum):
            with autocast:
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum
            x, y = train_b.get()  # fetch the next batch while the GPU is busy
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        tokens_seen += tokens_per_iter

        if it % cfg.log_interval == 0:
            dt = time.time() - t_log
            t_log = time.time()
            steps = cfg.log_interval if it > start_iter else 1
            row = {
                "iter": it,
                "loss": round(loss.item() * cfg.grad_accum, 4),
                "lr": lr,
                "grad_norm": round(float(grad_norm), 3),
                "tok_per_s": round(steps * tokens_per_iter / max(dt, 1e-9)),
            }
            append_csv(out / "train_log.csv", row)
            print(
                f"iter {it}: loss {row['loss']:.4f} lr {lr:.2e} "
                f"grad {row['grad_norm']} {row['tok_per_s']:,} tok/s"
            )

    if mlflow:
        mlflow.end_run()
    summary = {
        "best_val_loss": best_val,
        "best_val_ppl": math.exp(best_val),
        "iters": it + 1,
        "tokens_seen": tokens_seen,
        "minutes": round((time.time() - t0) / 60, 1),
        "params": raw_model.num_params(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ForgeGPT")
    parser.add_argument("--config", help="JSON file with TrainConfig fields")
    types = {"int": int, "float": float, "str": str, "float | None": float}
    for f in fields(TrainConfig):
        if f.type == "bool":
            parser.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"))
        elif f.type in types:
            parser.add_argument(f"--{f.name}", type=types[f.type])
    args = vars(parser.parse_args())
    cfg = load_config(args.pop("config"), args)
    if os.environ.get("FORGEGPT_THREADS"):
        torch.set_num_threads(int(os.environ["FORGEGPT_THREADS"]))
    train(cfg)


if __name__ == "__main__":
    main()
