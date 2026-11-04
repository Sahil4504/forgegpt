"""Export a trained ForgeGPT checkpoint to ONNX (with KV cache) and quantize it to INT8.

One ONNX graph serves both phases of generation:
    inputs : input_ids (B, T) int64, past (L, 2, B, H, P, D) float32
    outputs: logits (B, T, V) float32, present (L, 2, B, H, P+T, D) float32
Prefill passes the whole prompt with an empty cache (P = 0); each decode step passes one
token plus the cache from the previous call.

Run:
    python -m forgegpt.onnx_export --ckpt runs/cpu_demo/ckpt_best.pt --out-dir artifacts/onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from forgegpt.model import GPT, GPTConfig


def load_checkpoint(path: str | Path, device: str = "cpu") -> GPT:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    return model.eval()


class CachedDecoder(nn.Module):
    """Wraps GPT so the per-layer KV cache is one stacked tensor (ONNX-friendly)."""

    def __init__(self, model: GPT):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, past: torch.Tensor):
        past_kvs = [(past[i, 0], past[i, 1]) for i in range(past.size(0))]
        logits, _, presents = self.model(input_ids, past_kvs=past_kvs, use_cache=True)
        present = torch.stack([torch.stack([k, v]) for k, v in presents])
        return logits, present


def empty_past(cfg: GPTConfig, batch: int = 1, length: int = 0) -> torch.Tensor:
    return torch.zeros(cfg.n_layer, 2, batch, cfg.n_head, length, cfg.d_model // cfg.n_head)


def export_onnx(model: GPT, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = CachedDecoder(model).eval()
    # Trace with a non-empty cache so the exporter sees P as a real, variable dimension.
    ids = torch.randint(0, model.cfg.vocab_size, (1, 3))
    past = empty_past(model.cfg, length=2)
    torch.onnx.export(
        wrapper,
        (ids, past),
        str(path),
        input_names=["input_ids", "past"],
        output_names=["logits", "present"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "new_tokens"},
            "past": {2: "batch", 4: "past_tokens"},
            "logits": {0: "batch", 1: "new_tokens"},
            "present": {2: "batch", 4: "total_tokens"},
        },
        opset_version=17,
        dynamo=False,
    )
    add_metadata(path, model.cfg, num_params=model.num_params())
    return path


def add_metadata(path: str | Path, cfg: GPTConfig, **extra) -> None:
    """Store the model config in the ONNX file so the runtime knows block_size etc."""
    import onnx

    m = onnx.load(str(path))
    del m.metadata_props[:]
    for key, value in {**cfg.to_dict(), **extra}.items():
        entry = m.metadata_props.add()
        entry.key, entry.value = key, str(value)
    onnx.save(m, str(path))


def quantize_int8(src: str | Path, dst: str | Path) -> Path:
    """Dynamic INT8: weights are stored as 8-bit integers; activations are quantized on
    the fly per batch. ~4x smaller weights and faster matmuls on CPU, no calibration."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    meta = {p.key: p.value for p in onnx.load(str(src)).metadata_props}
    m = onnx.load(str(dst))
    if not m.metadata_props:  # copy the config over if the quantizer dropped it
        for key, value in meta.items():
            entry = m.metadata_props.add()
            entry.key, entry.value = key, value
        onnx.save(m, str(dst))
    return Path(dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out-dir", default="artifacts/onnx")
    args = parser.parse_args()
    model = load_checkpoint(args.ckpt)
    fp32 = export_onnx(model, Path(args.out_dir) / "forgegpt.onnx")
    int8 = quantize_int8(fp32, Path(args.out_dir) / "forgegpt.int8.onnx")
    for p in (fp32, int8):
        print(f"{p}: {p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
