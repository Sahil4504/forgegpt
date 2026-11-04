"""Serving layer: request validation, API key, the Lambda handler and the FastAPI app.

Uses a tiny randomly initialised model exported to ONNX, so no trained checkpoint is needed.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("onnxruntime")

from forgegpt.model import GPT, GPTConfig  # noqa: E402
from forgegpt.onnx_export import export_onnx, quantize_int8  # noqa: E402
from forgegpt.serving import BadRequest, api_key_ok, parse_request  # noqa: E402
from forgegpt.tokenizer import train_tokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    d = tmp_path_factory.mktemp("serve")
    tok = train_tokenizer(["the brakes failed on the highway at speed."] * 100, vocab_size=320)
    tok.save(str(d / "tok.json"))
    torch.manual_seed(0)
    fp32 = export_onnx(
        GPT(GPTConfig(vocab_size=320, block_size=32, n_layer=2, n_head=4, d_model=64)), d / "m.onnx"
    )
    int8 = quantize_int8(fp32, d / "m.int8.onnx")
    return int8, d / "tok.json"


def test_parse_request_defaults_and_caps():
    prompt, s = parse_request({"prompt": "the brakes"})
    assert prompt == "the brakes" and s.max_new_tokens == 80 and s.temperature == 0.8
    with pytest.raises(BadRequest):
        parse_request({"prompt": "x", "max_tokens": 10_000})  # over MAX_TOKENS_CAP
    with pytest.raises(BadRequest):
        parse_request({"prompt": "x", "temperature": "hot"})
    with pytest.raises(BadRequest):
        parse_request({"prompt": 5})
    _, s = parse_request({"max_tokens": "20", "seed": "3"})  # query strings arrive as text
    assert s.max_new_tokens == 20 and s.seed == 3


def test_api_key(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    assert api_key_ok({})
    monkeypatch.setenv("API_KEY", "s3cret")
    assert not api_key_ok({})
    assert not api_key_ok({"x-api-key": "wrong"})
    assert api_key_ok({"X-Api-Key": "s3cret"})


def load_handler(monkeypatch, model, tok):
    monkeypatch.setenv("MODEL_FILE", str(model))
    monkeypatch.setenv("TOKENIZER_FILE", str(tok))
    spec = importlib.util.spec_from_file_location(
        "handler", ROOT / "serving" / "lambda" / "handler.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lambda_handler(monkeypatch, artifacts, capsys):
    monkeypatch.setenv("API_KEY", "k")
    monkeypatch.setenv("EMF", "1")
    h = load_handler(monkeypatch, *artifacts)
    event = {
        "headers": {"x-api-key": "k"},
        "body": json.dumps({"prompt": "the brakes", "max_tokens": 12, "seed": 1}),
    }
    first = h.handler(event, None)
    body = json.loads(first["body"])
    assert first["statusCode"] == 200 and body["cold_start"] is True
    assert 0 < body["generated_tokens"] <= 12 and isinstance(body["completion"], str)
    assert json.loads(h.handler(event, None)["body"])["cold_start"] is False
    log = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert log["_aws"]["CloudWatchMetrics"][0]["Namespace"] == "ForgeGPT"

    assert h.handler({"headers": {}, "body": "{}"}, None)["statusCode"] == 401
    bad = h.handler({"headers": {"x-api-key": "k"}, "body": "not json"}, None)
    assert bad["statusCode"] == 400


def test_fastapi_app(monkeypatch, artifacts, tmp_path):
    from fastapi.testclient import TestClient

    model, tok = artifacts
    monkeypatch.setenv("FORGEGPT_MODEL", str(model))
    monkeypatch.setenv("FORGEGPT_TOKENIZER", str(tok))
    monkeypatch.setenv("FORGEGPT_CHARTS", str(tmp_path))
    monkeypatch.delenv("API_KEY", raising=False)
    sys.path.insert(0, str(ROOT))
    sys.modules.pop("serving.app.main", None)
    from serving.app.main import app

    c = TestClient(app)
    assert c.get("/healthz").json() == {"ok": True}
    assert "ForgeGPT" in c.get("/").text
    info = c.get("/v1/info").json()
    assert info["block_size"] == 32 and info["quantization"] == "INT8 (dynamic)"
    r = c.post("/v1/generate", json={"prompt": "the brakes", "max_tokens": 10, "seed": 0})
    assert r.status_code == 200 and r.json()["generated_tokens"] <= 10
    assert c.post("/v1/generate", json={"max_tokens": -1}).status_code == 400
    events = [
        json.loads(line[6:])
        for line in c.get("/v1/stream?prompt=the&max_tokens=8&seed=0").text.splitlines()
        if line.startswith("data: ")
    ]
    assert events[-1]["done"] is True and len(events) - 1 == events[-1]["generated_tokens"]
    assert c.post("/v1/tokenize", json={"text": "the brakes"}).json()["tokens"][0] == "the"
