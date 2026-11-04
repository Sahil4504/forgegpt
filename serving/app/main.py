"""Local API + web GUI for ForgeGPT (ONNX Runtime, no PyTorch).

    pip install -e ".[serve]"
    uvicorn serving.app.main:app --port 8000      # then open http://localhost:8000

Environment variables:
    FORGEGPT_MODEL      default artifacts/onnx/forgegpt.int8.onnx
    FORGEGPT_TOKENIZER  default data/tokenizer/v1/bpe-8k.json
    FORGEGPT_CHARTS     default docs/images (shown in the Benchmarks tab)
    FORGEGPT_SUMMARY    optional runs/<run>/summary.json (shows validation perplexity)
    API_KEY, MAX_TOKENS_CAP  same meaning as in the Lambda handler
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from forgegpt.onnx_infer import OnnxGenerator
from forgegpt.serving import BadRequest, api_key_ok, log_request, parse_request

HERE = Path(__file__).parent
MODEL = Path(os.environ.get("FORGEGPT_MODEL", "artifacts/onnx/forgegpt.int8.onnx"))
TOKENIZER = os.environ.get("FORGEGPT_TOKENIZER", "data/tokenizer/v1/bpe-8k.json")
CHARTS = Path(os.environ.get("FORGEGPT_CHARTS", "docs/images"))
SUMMARY = os.environ.get("FORGEGPT_SUMMARY")

app = FastAPI(title="ForgeGPT", version="0.1.0")
gen = OnnxGenerator(str(MODEL), TOKENIZER, threads=int(os.environ.get("ORT_THREADS", "2")))
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
if CHARTS.exists():
    app.mount("/charts", StaticFiles(directory=CHARTS), name="charts")


def _auth(request: Request) -> None:
    if not api_key_ok(dict(request.headers)):
        raise HTTPException(401, "missing or invalid x-api-key")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(HERE / "static" / "index.html")


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.get("/v1/info")
def info() -> dict:
    meta = gen.session.get_modelmeta().custom_metadata_map
    out = {
        "model_file": MODEL.name,
        "model_mb": round(MODEL.stat().st_size / 1e6, 1),
        "quantization": "INT8 (dynamic)" if "int8" in MODEL.name else "FP32",
        "runtime": "ONNX Runtime (CPU)",
        **{k: int(v) for k, v in meta.items() if v.isdigit()},
    }
    if SUMMARY and Path(SUMMARY).exists():
        s = json.loads(Path(SUMMARY).read_text())
        out.update(val_ppl=round(s["best_val_ppl"], 1), train_tokens=s["tokens_seen"])
    out["charts"] = sorted(p.name for p in CHARTS.glob("*.png")) if CHARTS.exists() else []
    return out


@app.post("/v1/generate")
async def generate(request: Request) -> dict:
    _auth(request)
    try:
        prompt, sampling = parse_request(await request.json())
    except (ValueError, BadRequest) as e:
        raise HTTPException(400, str(e)) from e
    result = gen.complete(prompt, sampling)
    log_request(str(uuid.uuid4()), result, service="local")
    return result


@app.post("/v1/tokenize")
async def tokenize(request: Request) -> dict:
    text = (await request.json()).get("text", "")
    enc = gen.tokenizer.encode(str(text)[:4000].lower())
    return {"tokens": enc.tokens, "ids": enc.ids}


@app.get("/v1/stream")
def stream(request: Request) -> StreamingResponse:
    """Server-sent events: one event per generated token, then a final metrics event."""
    _auth(request)
    try:
        prompt, sampling = parse_request(dict(request.query_params))
    except BadRequest as e:
        raise HTTPException(400, str(e)) from e

    def events():
        prompt_ids = gen.tokenizer.encode(prompt.lower()).ids
        start = time.perf_counter()
        first, ids, text = None, [], ""
        for tok in gen.stream_ids(prompt_ids, sampling):
            first = first or time.perf_counter()
            ids.append(tok)
            # Decode the whole completion and send only the new characters, so multi-byte
            # characters split across tokens are never sent half-finished.
            full = gen.tokenizer.decode(ids)
            piece, text = full[len(text) :], full
            yield f"data: {json.dumps({'piece': piece, 'id': tok})}\n\n"
        end = time.perf_counter()
        result = {
            "done": True,
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(ids),
            "ttft_ms": round(1000 * ((first or end) - start), 1),
            "latency_ms": round(1000 * (end - start), 1),
            "tokens_per_s": round(len(ids) / (end - start), 1) if end > start else 0.0,
        }
        log_request(str(uuid.uuid4()), result, service="local")
        yield f"data: {json.dumps(result)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
