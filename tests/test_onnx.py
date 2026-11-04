"""ONNX export: outputs must match PyTorch for prefill and cached decode, and the
numpy-only generator must reproduce PyTorch's greedy generation."""

import numpy as np
import pytest
import torch

ort = pytest.importorskip("onnxruntime")

from forgegpt.generate import SamplingConfig, generate  # noqa: E402
from forgegpt.model import GPT, GPTConfig  # noqa: E402
from forgegpt.onnx_export import empty_past, export_onnx, quantize_int8  # noqa: E402
from forgegpt.onnx_infer import OnnxGenerator, Sampling, sample_next  # noqa: E402
from forgegpt.tokenizer import train_tokenizer  # noqa: E402

CFG = GPTConfig(vocab_size=320, block_size=32, n_layer=2, n_head=4, d_model=64)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    d = tmp_path_factory.mktemp("onnx")
    path = export_onnx(model, d / "m.onnx")
    tok = train_tokenizer(["the brakes failed on the highway at speed."] * 100, vocab_size=320)
    tok.save(str(d / "tok.json"))
    return model, path, d


def run(sess, ids, past):
    return sess.run(None, {"input_ids": np.asarray(ids, dtype=np.int64), "past": past})


def test_prefill_and_decode_match_pytorch(exported):
    model, path, _ = exported
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ids = torch.randint(0, CFG.vocab_size, (1, 12))
    full, _ = model(ids)

    logits, present = run(sess, ids[:, :8].numpy(), empty_past(CFG).numpy())  # P = 0
    np.testing.assert_allclose(logits, full[:, :8].detach().numpy(), atol=1e-4)
    for t in range(8, 12):  # decode one token at a time with the cache
        logits, present = run(sess, ids[:, t : t + 1].numpy(), present)
        np.testing.assert_allclose(logits[:, 0], full[:, t].detach().numpy(), atol=1e-4)
    assert present.shape[4] == 12


def test_numpy_generator_matches_pytorch_greedy(exported):
    model, path, d = exported
    gen = OnnxGenerator(str(path), str(d / "tok.json"))
    assert gen.block_size == CFG.block_size
    s = Sampling(max_new_tokens=40, temperature=0.0)  # 40 > block_size: sliding window
    torch_ids = generate(model, [5, 7, 9], SamplingConfig(max_new_tokens=40, temperature=0.0))
    assert list(gen.stream_ids([5, 7, 9], s)) == torch_ids


def test_int8_model_runs_and_is_smaller(exported):
    _, path, d = exported
    q = quantize_int8(path, d / "m.int8.onnx")
    assert q.stat().st_size < path.stat().st_size
    sess = ort.InferenceSession(str(q), providers=["CPUExecutionProvider"])
    logits, present = run(sess, [[1, 2, 3]], empty_past(CFG).numpy())
    assert logits.shape == (1, 3, CFG.vocab_size) and np.isfinite(logits).all()
    assert sess.get_modelmeta().custom_metadata_map["block_size"] == str(CFG.block_size)


def test_numpy_top_k_top_p():
    logits = np.array([5.0, 4.0, 1.0, 0.0, -1.0])
    rng = np.random.default_rng(0)
    assert {
        sample_next(logits, Sampling(temperature=1, top_k=2, top_p=None), rng) for _ in range(200)
    } <= {0, 1}
    assert {
        sample_next(logits, Sampling(temperature=1, top_k=None, top_p=0.5), rng) for _ in range(200)
    } == {0}
