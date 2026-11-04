"""AWS Lambda handler for ForgeGPT, exposed through a Lambda Function URL.

The model (INT8 ONNX, ~8 MB) and tokenizer are bundled in the deployment zip, so a cold
start only has to create the ONNX Runtime session. Set MODEL_S3_URI to load the model
from S3 at cold start instead (this is the variant compared in the benchmarks).

Request:  POST {"prompt": "the brakes", "max_tokens": 60, "temperature": 0.8,
                "top_k": 50, "top_p": 0.95, "seed": 1}
          header x-api-key: <API_KEY>   (only if the API_KEY env var is set)
Response: {"completion": "...", "generated_tokens": 60, "ttft_ms": ..., "latency_ms": ...}
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

from forgegpt.onnx_infer import OnnxGenerator
from forgegpt.serving import BadRequest, api_key_ok, log_request, parse_request

HERE = Path(__file__).parent
COLD_START = True
INIT_START = time.perf_counter()


def _model_path() -> str:
    uri = os.environ.get("MODEL_S3_URI")
    if not uri:
        return str(HERE / os.environ.get("MODEL_FILE", "model/forgegpt.int8.onnx"))
    import boto3  # available in the Lambda runtime

    bucket, key = uri.removeprefix("s3://").split("/", 1)
    local = Path("/tmp") / Path(key).name
    if not local.exists():
        boto3.client("s3").download_file(bucket, key, str(local))
    return str(local)


# Runs once per execution environment (cold start), outside the handler.
GENERATOR = OnnxGenerator(
    _model_path(),
    str(HERE / os.environ.get("TOKENIZER_FILE", "model/bpe-8k.json")),
    threads=int(os.environ.get("ORT_THREADS", "2")),
)
INIT_MS = round(1000 * (time.perf_counter() - INIT_START), 1)


def _response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(body),
    }


def handler(event, context):
    global COLD_START
    cold, COLD_START = COLD_START, False
    request_id = getattr(context, "aws_request_id", "local")

    if not api_key_ok(event.get("headers") or {}):
        return _response(401, {"error": "missing or invalid x-api-key"})
    if event.get("rawPath") == "/healthz":
        return _response(200, {"ok": True, "cold_start": cold, "init_ms": INIT_MS})

    raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw).decode()
    try:
        prompt, sampling = parse_request(json.loads(raw))
    except (json.JSONDecodeError, BadRequest) as e:
        return _response(400, {"error": str(e)})

    result = GENERATOR.complete(prompt, sampling)
    result.update({"cold_start": cold, "init_ms": INIT_MS if cold else 0.0})
    log_request(request_id, result, service="lambda")
    return _response(200, result)
