"""Build the README / GUI charts from saved results. Each chart is skipped if its inputs
are missing, so this can be re-run as more results arrive.

    python scripts/make_charts.py --run runs/cpu_demo

Output: docs/images/*.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"

# Validated categorical palette, used in this fixed order (blue, orange, aqua).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.titlepad": 24,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "legend.frameon": False,
        "lines.linewidth": 2,
    }
)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def subtitle(ax, text: str) -> None:
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK_2, fontsize=9.5, va="bottom")


def bar_labels(ax, bars, fmt: str) -> None:
    for b in bars:
        w = b.get_width()
        ax.text(
            w,
            b.get_y() + b.get_height() / 2,
            " " + fmt.format(w),
            va="center",
            color=INK,
            fontsize=9.5,
        )


def tokenizer_comparison() -> None:
    path = ROOT / "data" / "tokenizer" / "v1" / "report.json"
    if not path.exists():
        return
    r = json.loads(path.read_text())["results"]
    names = {"bpe-8k": "ForgeGPT 8k (chosen)", "bpe-16k": "ForgeGPT 16k", "gpt2": "GPT-2 50k"}
    keys = [k for k in ("gpt2", "bpe-16k", "bpe-8k") if k in r]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    bars = ax.barh(
        [names[k] for k in keys],
        [r[k]["mean_tokens_per_narrative"] for k in keys],
        color=BLUE,
        height=0.55,
    )
    bar_labels(ax, bars, "{:.1f}")
    ax.set_xlabel("mean tokens per complaint (validation split)")
    ax.set_xlim(0, max(b.get_width() for b in bars) * 1.15)
    ax.grid(axis="y", visible=False)
    ax.set_title("A domain tokenizer with 1/6 of GPT-2's vocabulary is as compact")
    save(fig, "tokenizer_comparison.png")


def allcaps_by_year() -> None:
    path = ROOT / "docs" / "data_stats" / "by_year.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["year"] >= 1995]
    caps = df["narratives"] * df["allcaps_share"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(df["year"], caps / 1e3, color=BLUE, width=0.8, label="ALL CAPS")
    ax.bar(
        df["year"],
        (df["narratives"] - caps) / 1e3,
        bottom=caps / 1e3,
        color=ORANGE,
        width=0.8,
        label="mixed case",
        edgecolor=SURFACE,
        linewidth=1,
    )
    ax.set_ylabel("unique narratives (thousands)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("Complaints were written in ALL CAPS until 2021, so the corpus is lowercased")
    save(fig, "allcaps_by_year.png")


def narrative_lengths() -> None:
    path = ROOT / "docs" / "data_stats" / "lengths.csv"
    if not path.exists():
        return
    tokens = pd.read_csv(path)["tokens"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.hist(tokens.clip(upper=500), bins=50, color=BLUE, edgecolor=SURFACE, linewidth=1)
    med = tokens.median()
    ax.axvline(med, color=INK, linewidth=1, linestyle="--")
    ax.text(med, ax.get_ylim()[1] * 0.92, f"  median {med:.0f} tokens", color=INK, fontsize=9.5)
    ax.set_xlabel("tokens per narrative (values above 500 shown at 500)")
    ax.set_ylabel("narratives")
    ax.grid(axis="x", visible=False)
    ax.set_title("Most complaints fit in a 256-token context")
    subtitle(ax, f"validation split, {len(tokens):,} narratives, bpe-8k tokenizer")
    save(fig, "narrative_lengths.png")


def training_curves(run: Path) -> None:
    ev = run / "eval.csv"
    if not ev.exists():
        return
    df = pd.read_csv(ev)
    log = pd.read_csv(run / "train_log.csv") if (run / "train_log.csv").exists() else None
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    x = df["tokens_seen"] / 1e6
    ax.plot(x, df["train_loss"], color=BLUE, label="train")
    ax.plot(x, df["val_loss"], color=ORANGE, label="validation")
    best = df.loc[df["val_loss"].idxmin()]
    ax.plot(
        best["tokens_seen"] / 1e6,
        best["val_loss"],
        "o",
        color=ORANGE,
        markersize=8,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
    )
    ax.annotate(
        f"best val loss {best['val_loss']:.2f} (perplexity {np.exp(best['val_loss']):.0f})",
        (best["tokens_seen"] / 1e6, best["val_loss"]),
        xytext=(-10, 28),
        textcoords="offset points",
        ha="right",
        color=INK,
        fontsize=9.5,
    )
    ax.set_ylim(df["val_loss"].min() - 0.3, min(df["val_loss"].max(), 7.5) + 0.2)
    ax.set_xlabel("training tokens seen (millions)")
    ax.set_ylabel("cross-entropy loss")
    ax.legend(loc="upper right")
    cfg = json.loads((run / "config.json").read_text())["model"]
    tok_s = int(log["tok_per_s"].median()) if log is not None else None
    ax.set_title("Training the demo model on CPU")
    desc = f"{cfg['n_layer']} layers, {cfg['n_head']} heads, width {cfg['d_model']}"
    desc += f", context {cfg['block_size']}"
    if tok_s:
        desc += f"; ~{tok_s:,} tokens/s on 2 vCPUs"
    subtitle(ax, desc)
    save(fig, "training_curves.png")


def attention_heads(run: Path, tokenizer: Path) -> None:
    ckpt = run / "ckpt_best.pt"
    if not ckpt.exists() or not tokenizer.exists():
        return
    import torch
    from tokenizers import Tokenizer

    from forgegpt.onnx_export import load_checkpoint

    model = load_checkpoint(ckpt)
    tok = Tokenizer.from_file(str(tokenizer))
    text = "while driving at 45 mph the brake pedal went to the floor and the vehicle did not stop"
    enc = tok.encode(text)
    labels = [t.replace("Ġ", "") for t in enc.tokens]
    last = model.blocks[-1].attn
    last.store_weights = True
    with torch.no_grad():
        model(torch.tensor([enc.ids]))
    w = last.last_weights[0].numpy()  # (H, T, T)
    n = min(4, w.shape[0])
    fig, axes = plt.subplots(1, n, figsize=(4.1 * n, 4.6), constrained_layout=True)
    for h, ax in enumerate(np.atleast_1d(axes)):
        ax.imshow(w[h], cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=8)
        ax.set_yticks(range(len(labels)), labels if h == 0 else [""] * len(labels), fontsize=8)
        ax.grid(False)
        ax.set_title(f"head {h + 1}", fontsize=11, loc="center")
    fig.suptitle(
        "Last-layer attention: each row is a token reading the earlier tokens",
        x=0.01,
        ha="left",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
        y=1.07,
    )
    save(fig, "attention_heads.png")


def kv_cache_scaling() -> None:
    path = ROOT / "benchmarks" / "results" / "cache_scaling.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["position"] > df["position"].min()]  # skip the prefill step
    med = df.groupby(["kv_cache", "position"])["ms"].median().unstack(0)
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.plot(med.index, med[False], color=BLUE, label="without KV cache")
    ax.plot(med.index, med[True], color=ORANGE, label="with KV cache")
    ax.set_xlabel("position of the generated token")
    ax.set_ylabel("ms per token")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    ax.text(
        med.index[-1],
        med[False].iloc[-1],
        f" {med[False].iloc[-1]:.1f} ms",
        va="center",
        color=INK,
        fontsize=9.5,
    )
    ax.text(
        med.index[-1],
        med[True].iloc[-1],
        f" {med[True].iloc[-1]:.1f} ms",
        va="center",
        color=INK,
        fontsize=9.5,
    )
    ax.set_title("The KV cache keeps per-token cost flat as the text grows")
    subtitle(ax, "PyTorch FP32, greedy decoding, median of 3 runs, 2 CPU threads")
    save(fig, "kv_cache_scaling.png")


def inference_variants() -> None:
    path = ROOT / "benchmarks" / "results" / "inference.csv"
    if not path.exists():
        return
    df = pd.read_csv(path).iloc[::-1]
    short = {
        "torch_fp32_nocache": "PyTorch FP32, no cache",
        "torch_fp32": "PyTorch FP32",
        "torch_int8": "PyTorch INT8",
        "onnx_fp32": "ONNX FP32",
        "onnx_int8": "ONNX INT8",
    }
    labels = [short[v] for v in df["variant"]]
    panels = [
        ("decode_tok_per_s", "tokens / second (higher is better)", "{:.0f}"),
        ("model_mb", "model size, MB", "{:.1f}"),
        ("peak_rss_mb", "peak memory, MB", "{:.0f}"),
        ("test_ppl", "test perplexity (lower is better)", "{:.2f}"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), sharey=True)
    for ax, (col, title, fmt) in zip(axes, panels, strict=True):
        bars = ax.barh(labels, df[col], color=BLUE, height=0.6)
        bar_labels(ax, bars, fmt)
        ax.set_xlim(0, df[col].max() * 1.3)
        ax.set_title(title, fontsize=10.5, fontweight="normal", color=INK)
        ax.grid(axis="y", visible=False)
    fig.suptitle(
        "CPU inference: INT8 ONNX is fastest and smallest",
        x=0.01,
        ha="left",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
        y=1.04,
    )
    save(fig, "inference_variants.png")


def load_test() -> None:
    files = sorted((ROOT / "benchmarks" / "results").glob("load_test_*.csv"))
    if not files:
        return
    fig, axes = plt.subplots(1, len(files), figsize=(6.4 * len(files), 3.6), squeeze=False)
    for ax, f in zip(axes[0], files, strict=True):
        df = pd.read_csv(f)
        ax.plot(
            df["users"],
            df["p50_ms"],
            "o-",
            color=BLUE,
            label="p50",
            markersize=7,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )
        ax.plot(
            df["users"],
            df["p95_ms"],
            "o-",
            color=ORANGE,
            label="p95",
            markersize=7,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )
        for _, r in df.iterrows():
            ax.annotate(
                f"{r.req_per_s:.1f} req/s",
                (r.users, r.p95_ms),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color=INK_2,
            )
        ax.set_xticks(df["users"])
        ax.set_xlabel("concurrent users")
        ax.set_ylabel(f"latency, ms ({int(df['max_tokens'].iloc[0])} tokens)")
        ax.set_ylim(0, df["p95_ms"].max() * 1.25)
        ax.legend(loc="upper left")
        names = {
            "forgegpt_short": "Local API, ForgeGPT INT8, short prompts",
            "aws_lambda": "AWS Lambda 2 GB",
            "llamacpp_q4_k_m": "EC2 llama.cpp, Qwen2.5-0.5B Q4_K_M",
            "llamacpp_q8_0": "EC2 llama.cpp, Qwen2.5-0.5B Q8_0",
        }
        key = f.stem.replace("load_test_", "")
        ax.set_title(names.get(key, key.replace("_", " ")), fontsize=11)
        ax.set_xlim(-1, df["users"].max() + 2)
    fig.suptitle(
        "Latency under load: a 2-vCPU server saturates, then requests queue",
        x=0.01,
        ha="left",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
        y=1.05,
    )
    save(fig, "load_test.png")


def cost_per_1k() -> None:
    path = ROOT / "benchmarks" / "results" / "cost_per_1k.csv"
    if not path.exists():
        return
    df = pd.read_csv(path).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.4, 0.75 * len(df) + 1.4))
    bars = ax.barh(df["option"], df["cost_per_1k_usd"] * 100, color=BLUE, height=0.55)
    bar_labels(ax, bars, "{:.3f}¢")
    ax.set_xlim(0, df["cost_per_1k_usd"].max() * 100 * 1.3)
    ax.set_xlabel("US cents per 1,000 requests (on-demand list price, before free tier)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Cost per 1,000 requests")
    save(fig, "cost_per_1k.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/cpu_demo")
    p.add_argument("--tokenizer", default="data/tokenizer/v1/bpe-8k.json")
    args = p.parse_args()
    tokenizer_comparison()
    allcaps_by_year()
    narrative_lengths()
    training_curves(Path(args.run))
    attention_heads(Path(args.run), Path(args.tokenizer))
    kv_cache_scaling()
    inference_variants()
    load_test()
    cost_per_1k()


if __name__ == "__main__":
    main()
