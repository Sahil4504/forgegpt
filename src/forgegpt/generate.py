"""Text generation: sampling strategies and a KV-cached generation loop.

Generation is autoregressive: predict one token, append it, repeat. Without a cache,
step t re-runs the model over all t tokens (quadratic work overall). With a KV cache,
the first call ("prefill") processes the prompt once, and every later call ("decode")
processes only the single newest token, reusing stored keys and values.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

import torch

from forgegpt.model import GPT


@dataclass
class SamplingConfig:
    max_new_tokens: int = 80
    temperature: float = 0.8  # <1 sharper / safer, >1 flatter / more random; 0 = greedy
    top_k: int | None = 50  # keep only the k most likely tokens (None = off)
    top_p: float | None = 0.95  # keep the smallest set whose probability sums to p
    stop_at_eot: bool = True


def sample_next(
    logits: torch.Tensor, cfg: SamplingConfig, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Pick the next token id from a (B, V) row of logits. Returns (B, 1)."""
    if cfg.temperature <= 0:  # greedy: always the single most likely token
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / cfg.temperature
    if cfg.top_k is not None and cfg.top_k < logits.size(-1):
        kth = torch.topk(logits, cfg.top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if cfg.top_p is not None and cfg.top_p < 1.0:
        # Nucleus sampling: sort, accumulate probabilities, cut once the sum passes p.
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cum - torch.softmax(sorted_logits, dim=-1) > cfg.top_p  # always keep the top 1
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


@torch.no_grad()
def stream(
    model: GPT,
    prompt_ids: list[int],
    cfg: SamplingConfig,
    eot_id: int = 0,
    use_cache: bool = True,
    seed: int | None = None,
) -> Iterator[int]:
    """Yield generated token ids one at a time (for streaming to a UI)."""
    model.eval()
    device = next(model.parameters()).device
    block = model.cfg.block_size
    gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    ids = list(prompt_ids) or [eot_id]  # an empty prompt starts from "start of document"
    cache = None

    for _ in range(cfg.max_new_tokens):
        if use_cache and cache is not None and len(ids) <= block:
            # Decode step: feed only the newest token; reuse cached keys/values.
            x = torch.tensor([ids[-1:]], device=device)
            logits, _, cache = model(x, past_kvs=cache, use_cache=True)
        else:
            # Prefill (or no cache, or context overflow): run over the last `block` tokens.
            x = torch.tensor([ids[-block:]], device=device)
            if use_cache:
                logits, _, cache = model(x, use_cache=True)
            else:
                logits, _ = model(x)
            if len(ids) >= block:
                cache = None  # the window will slide next step; positions must be rebuilt
        nxt = int(sample_next(logits[:, -1, :], cfg, gen).item())
        if cfg.stop_at_eot and nxt == eot_id:
            return
        ids.append(nxt)
        yield nxt


def generate(model: GPT, prompt_ids: list[int], cfg: SamplingConfig, **kwargs) -> list[int]:
    return list(stream(model, prompt_ids, cfg, **kwargs))


def timed_generation(model: GPT, prompt_ids: list[int], cfg: SamplingConfig, **kwargs) -> dict:
    """Generate and measure time-to-first-token and decode throughput."""
    start = time.perf_counter()
    first = None
    out: list[int] = []
    for tok in stream(model, prompt_ids, cfg, **kwargs):
        if first is None:
            first = time.perf_counter()
        out.append(tok)
    end = time.perf_counter()
    decode_tokens = max(len(out) - 1, 0)
    return {
        "tokens": out,
        "ttft_ms": 1000 * ((first or end) - start),
        "total_ms": 1000 * (end - start),
        "decode_tok_per_s": decode_tokens / (end - first) if first and end > first else 0.0,
    }
