"""Request handling shared by the local FastAPI app and the AWS Lambda handler.

Pure Python + the ONNX generator: no web framework or PyTorch imports, so it can be
packaged into a small Lambda zip.
"""

from __future__ import annotations

import hmac
import json
import os
import sys
import time

from forgegpt.onnx_infer import Sampling

MAX_TOKENS_CAP = int(os.environ.get("MAX_TOKENS_CAP", "200"))  # protects compute credits
MAX_PROMPT_CHARS = 2000


class BadRequest(ValueError):
    pass


def _num(body: dict, key: str, default, kind, lo, hi):
    value = body.get(key, default)
    if value is None:
        return None
    try:
        value = kind(value)
    except (TypeError, ValueError) as e:
        raise BadRequest(f"'{key}' must be a {kind.__name__}") from e
    if not lo <= value <= hi:
        raise BadRequest(f"'{key}' must be between {lo} and {hi}")
    return value


def parse_request(body: dict) -> tuple[str, Sampling]:
    """Validate a generation request. Raises BadRequest with a user-facing message."""
    if not isinstance(body, dict):
        raise BadRequest("body must be a JSON object")
    prompt = body.get("prompt", "")
    if not isinstance(prompt, str):
        raise BadRequest("'prompt' must be a string")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise BadRequest(f"'prompt' is limited to {MAX_PROMPT_CHARS} characters")
    sampling = Sampling(
        max_new_tokens=_num(body, "max_tokens", 80, int, 1, MAX_TOKENS_CAP),
        temperature=_num(body, "temperature", 0.8, float, 0.0, 2.0),
        top_k=_num(body, "top_k", 50, int, 1, 8192),
        top_p=_num(body, "top_p", 0.95, float, 0.01, 1.0),
        seed=_num(body, "seed", None, int, 0, 2**31 - 1),
    )
    return prompt, sampling


def api_key_ok(headers: dict) -> bool:
    """If API_KEY is set, require a matching x-api-key header (constant-time compare)."""
    expected = os.environ.get("API_KEY")
    if not expected:
        return True
    given = {k.lower(): v for k, v in (headers or {}).items()}.get("x-api-key", "")
    return hmac.compare_digest(given, expected)


def log_request(request_id: str, result: dict, service: str) -> None:
    """One structured JSON log line per request. With EMF=1 it is also in CloudWatch
    Embedded Metric Format, so CloudWatch turns the fields into metrics with no API calls."""
    record = {
        "request_id": request_id,
        "service": service,
        "prompt_tokens": result.get("prompt_tokens"),
        "generated_tokens": result.get("generated_tokens"),
        "ttft_ms": result.get("ttft_ms"),
        "latency_ms": result.get("latency_ms"),
        "tokens_per_s": result.get("tokens_per_s"),
    }
    if os.environ.get("EMF") == "1":
        record["_aws"] = {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": "ForgeGPT",
                    "Dimensions": [["service"]],
                    "Metrics": [
                        {"Name": "ttft_ms", "Unit": "Milliseconds"},
                        {"Name": "latency_ms", "Unit": "Milliseconds"},
                        {"Name": "tokens_per_s", "Unit": "Count/Second"},
                        {"Name": "generated_tokens", "Unit": "Count"},
                    ],
                }
            ],
        }
    print(json.dumps(record), file=sys.stdout, flush=True)
