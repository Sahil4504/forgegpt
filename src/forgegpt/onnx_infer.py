"""Generation with ONNX Runtime and numpy only: no PyTorch needed.

This is what runs inside AWS Lambda, where a ~1 GB PyTorch install does not fit in a zip
deployment but onnxruntime (~20 MB) + numpy + tokenizers do.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

EOT_ID = 0


@dataclass
class Sampling:
    max_new_tokens: int = 80
    temperature: float = 0.8
    top_k: int | None = 50
    top_p: float | None = 0.95
    seed: int | None = None
    ignore_eot: bool = False  # benchmarks: always generate exactly max_new_tokens


def sample_next(logits: np.ndarray, s: Sampling, rng: np.random.Generator) -> int:
    """Same algorithm as forgegpt.generate.sample_next, for a single (V,) row."""
    if s.temperature <= 0:
        return int(np.argmax(logits))
    logits = logits.astype(np.float64) / s.temperature
    if s.top_k is not None and s.top_k < logits.size:
        kth = np.partition(logits, -s.top_k)[-s.top_k]
        logits = np.where(logits < kth, -np.inf, logits)
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    if s.top_p is not None and s.top_p < 1.0:
        order = np.argsort(-probs)
        cum = np.cumsum(probs[order])
        keep = order[(cum - probs[order]) <= s.top_p]
        mask = np.zeros_like(probs, dtype=bool)
        mask[keep] = True
        probs = np.where(mask, probs, 0.0)
        probs /= probs.sum()
    return int(rng.choice(probs.size, p=probs))


class OnnxGenerator:
    def __init__(self, model_path: str, tokenizer_path: str, threads: int | None = None):
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        past = self.session.get_inputs()[1].shape  # [L, 2, batch, H, past_tokens, D]
        self.n_layer, self.n_head, self.head_dim = past[0], past[3], past[5]
        # block_size is not stored in the graph; 256 for the full model, 128 for the demo.
        self.block_size = int(
            self.session.get_modelmeta().custom_metadata_map.get("block_size", 128)
        )

    def _run(self, ids: list[int], past: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits, present = self.session.run(
            None, {"input_ids": np.asarray([ids], dtype=np.int64), "past": past}
        )
        return logits[0, -1], present

    def _empty_past(self) -> np.ndarray:
        return np.zeros((self.n_layer, 2, 1, self.n_head, 0, self.head_dim), dtype=np.float32)

    def stream_ids(self, prompt_ids: list[int], s: Sampling) -> Iterator[int]:
        rng = np.random.default_rng(s.seed)
        ids = list(prompt_ids) or [EOT_ID]
        ids = ids[-self.block_size :]
        logits, past = self._run(ids, self._empty_past())  # prefill
        for _ in range(s.max_new_tokens):
            nxt = sample_next(logits, s, rng)
            if nxt == EOT_ID and not s.ignore_eot:
                return
            yield nxt
            ids.append(nxt)
            if len(ids) > self.block_size:  # context full: slide the window, rebuild cache
                ids = ids[-self.block_size :]
                logits, past = self._run(ids, self._empty_past())
            else:  # decode: one new token + cache
                logits, past = self._run([nxt], past)

    def complete(self, prompt: str, s: Sampling) -> dict:
        """Generate a completion and time it (used by the API and the benchmarks)."""
        prompt_ids = self.tokenizer.encode(prompt.lower()).ids
        start = time.perf_counter()
        first = None
        out: list[int] = []
        for tok in self.stream_ids(prompt_ids, s):
            first = first or time.perf_counter()
            out.append(tok)
        end = time.perf_counter()
        return {
            "completion": self.tokenizer.decode(out),
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(out),
            "ttft_ms": round(1000 * ((first or end) - start), 2),
            "latency_ms": round(1000 * (end - start), 2),
            "tokens_per_s": round(len(out) / (end - start), 1) if end > start else 0.0,
        }
