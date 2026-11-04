# ForgeGPT

**A GPT written from scratch, trained on 1.6 million vehicle complaints, made fast on CPU,
and packaged for serverless serving on AWS, with every step measured.**

![ForgeGPT playground](docs/images/gui_playground.png)

ForgeGPT is a decoder-only transformer (the GPT-2 architecture) implemented in plain PyTorch:
attention, blocks, embeddings and the training loop are all in this repo, with no
`nn.Transformer`. It is trained on the narratives of the public NHTSA vehicle-complaints
database, optimized for CPU inference (KV cache, ONNX Runtime, INT8), served through a web
GUI and an AWS Lambda function, and benchmarked for latency, throughput and cost.

> **What has been run and what is ready to run.** Everything below was executed and
> measured on a 2-vCPU Linux machine with a **5.3M-parameter demo model trained on CPU**
> (15.4M tokens, 96 minutes). The full-size 14M-parameter run
> (`notebooks/kaggle_train.ipynb`, free Kaggle T4) and the AWS deployment (`infra/`, one
> `cdk deploy`) are scripted and tested locally, but not yet executed: see
> [docs/DEPLOY.md](docs/DEPLOY.md). No number in this README is estimated unless it says so.

## Architecture

![Architecture](docs/images/architecture.png)

| Stage | What | Where |
|---|---|---|
| Data | Download, clean (11 rules), dedupe, split 2.26M rows → 1.62M narratives | `scripts/download_nhtsa.py`, `src/forgegpt/clean.py` |
| Tokenizer | Byte-level BPE, 8,192 tokens, trained on the train split | `src/forgegpt/tokenizer.py` |
| Model | Causal multi-head attention, pre-LN blocks, GELU MLP, tied embeddings, KV cache | `src/forgegpt/model.py` |
| Training | AdamW, warmup + cosine, grad clipping, fp16 on GPU, checkpoint/resume, CSV/MLflow logs | `src/forgegpt/train.py` |
| Inference | Greedy / temperature / top-k / top-p, KV-cached streaming generation | `src/forgegpt/generate.py` |
| Optimization | ONNX export with the KV cache inside the graph, dynamic INT8 quantization | `src/forgegpt/onnx_export.py` |
| Serving | FastAPI + web GUI with token streaming; Lambda handler (ONNX Runtime + numpy only) | `serving/` |
| Open-model baseline | llama.cpp + Qwen2.5-0.5B-Instruct (Q4_K_M / Q8_0) on EC2 m7i-flex.large | `serving/ec2/` |
| Infrastructure | AWS CDK: S3, 2 Lambdas + Function URLs, optional EC2, CloudWatch dashboard, budget | `infra/` |
| Quality | 55 unit tests, ruff, GitHub Actions CI, OIDC deploy workflow | `tests/`, `.github/` |

## Results

### Data and tokenizer

![All caps by year](docs/images/allcaps_by_year.png)

Complaints filed before 2021 are almost entirely upper case and later ones are not, so
the corpus is lowercased; otherwise the model would learn two "dialects" and the
tokenizer would spend vocabulary on `BRAKE` and `brake`. Cleaning also removes NHTSA
analyst tags and legal boilerplate and masks phone numbers, emails, URLs and VINs
(details and counts in [docs/DATA.md](docs/DATA.md)).

![Tokenizer comparison](docs/images/tokenizer_comparison.png)

A custom 8k byte-level BPE encodes complaints in 100.6 tokens on average versus 101.7 for
GPT-2's 50k vocabulary. At width 384 that is a 3.1M-parameter embedding table instead of
19.3M, which matters in a model this size, and a 6x smaller output layer on CPU.

### Training

![Training curves](docs/images/training_curves.png)

The demo model (4 layers, 4 heads, width 256, context 128; 5.3M parameters) trained for 7,500 steps of 16 × 128 tokens, 15.4M tokens in total, in 96 minutes on 2 vCPUs at about 2,900 tokens/s. Validation loss fell from 9.06 (perplexity 8,580: a uniform guess over the 8,192-token vocabulary) to **3.40, a perplexity of 29.8**. Train and validation curves stay together: the model has seen only ~10% of the 149M available training tokens, so it is limited by compute, not by data, and does not overfit. The full 14M-parameter configuration in `configs/kaggle_t4.json` sees about 10× more tokens on a free Kaggle T4.

Full logs, config and per-evaluation samples: [`benchmarks/results/training/cpu_demo/`](benchmarks/results/training/cpu_demo/).

Samples from the final checkpoint (prompt in bold, temperature 0.8, top-k 50, top-p 0.95):

> **while driving on the highway at 65 mph the** car died. after the car turned off and on again, the same thing happened. the car was towed to the nearest chevy dealership and was told there was a recall for this exact problem. the car is still under recall for the same issue. i am concerned that this is a common problem with this car, but my vin is not included in this
>
> **the contact owns a 2019** ford f-150. the contact stated while driving at an undisclosed speed, there was an abnormal sound coming from the vehicle. the contact stated that the failure was intermittent and the check engine warning light was illuminated. the contact stated that the vehicle had been taken to the dealer where it was diagnosed that the engine needed to be replaced.
>
> **the airbag warning light** is on and the dash lights are on. i would have to press the gas pedal and start the car. this has happened twice and it is a safety hazard that could cause a crash.

It has learned both registers of the data: first-person owner complaints and NHTSA's own "the contact stated…" summaries. Longer outputs drift (*"i asked them to bring it back and i could call me back"*), which is expected at this size and training budget. Seeds 1, 3 and 2; the INT8 ONNX model, the same one the GUI and Lambda serve.

![Attention heads](docs/images/attention_heads.png)

Different heads in the last layer learn different jobs. Head 2 parks most of its attention on the first token (a common "resting" pattern). Head 4 keeps returning to *brake* from *and*, *the*, *vehicle* and *not*: it tracks what the complaint is about. Head 1 links *stop* back to *vehicle*, and head 3 links *and* and *vehicle* to the verb *went*.

### CPU inference

| Variant | Model file | Peak RAM | Time to first token | Decode speed | 64-token latency (p50) | Test perplexity |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch FP32, no KV cache | 21.1 MB | 636 MB | 8.7 ms | 94 tok/s | 687 ms | 31.53 |
| PyTorch FP32 + KV cache | 21.1 MB | 636 MB | 8.9 ms | 238 tok/s | 270 ms | 31.53 |
| PyTorch INT8 (dynamic) + KV cache | 13.8 MB | 638 MB | 5.5 ms | 238 tok/s | 271 ms | 31.98 |
| ONNX Runtime FP32 + KV cache | 29.6 MB | 156 MB | 3.1 ms | 500 tok/s | 126 ms | 31.53 |
| **ONNX Runtime INT8 + KV cache** | 7.5 MB | 137 MB | 2.2 ms | 876 tok/s | 72 ms | 31.91 |

30 prompts from the test split, 16-token prompt, 64 generated tokens (greedy), 2 threads; perplexity on 200 random 128-token test windows. Each variant runs in its own process.

![Inference variants](docs/images/inference_variants.png)

- **KV cache: 2.5× faster decoding** (94 → 238 tokens/s) with identical output: it is tested to give the same logits as a full forward pass.
- **ONNX Runtime: another 2.1×** over PyTorch with the same cache, and **4× less memory** (156 MB vs 636 MB) because the serving process never imports PyTorch.
- **INT8 dynamic quantization: 1.75× faster again and 4× smaller** (7.5 MB), for a **1.2% perplexity increase** (31.53 → 31.91).
- End to end: **9.4× faster decoding** than the PyTorch baseline, 64 tokens in 72 ms.
- PyTorch's own dynamic INT8 shrinks the weights but is no faster here; the speedup comes from ONNX Runtime's fused INT8 kernels. The FP32 ONNX file is larger than the PyTorch weights because the export stores the tied embedding and output matrices separately.

![KV cache scaling](docs/images/kv_cache_scaling.png)

Without a cache, generating token *t* re-runs the model over all *t* tokens, so each step gets slower (9.6 → 17.8 ms per token by position 128). With the cache, each step processes only the new token: a flat ~4.5 ms. The gap widens with longer contexts (256 in the full model).

### Serving under load (local API, 2 vCPUs)

| Concurrent users | Requests | Failures | p50 | p95 | Throughput |
|---:|---:|---:|---:|---:|---:|
| 1 | 90 | 0 | 76 ms | 150 ms | 2.3 req/s |
| 5 | 408 | 0 | 120 ms | 270 ms | 10.5 req/s |
| 10 | 588 | 0 | 320 ms | 540 ms | 14.7 req/s |
| 20 | 535 | 0 | 1100 ms | 1500 ms | 13.7 req/s |

Locust, short prompts, up to 64 generated tokens per request, 40 s per level; the API and the load generator share the same 2-vCPU machine.

![Load test](docs/images/load_test.png)

A single 2-vCPU server peaks at about **14.7 requests/s with 10 concurrent users** (p95 540 ms). Beyond that, throughput stays flat and requests queue: at 20 users the p95 reaches 1.5 s. On AWS, Lambda avoids this by running one request per execution environment and scaling out; the cost of that is per-request pricing and cold starts.

### Cost per 1,000 requests

| Option | Basis | Cost per 1,000 requests |
|---|---|---:|
| Lambda 2 GB, ForgeGPT INT8 (projected) | local p50 latency 72 ms for 64 tokens | $0.0026 |
| EC2 m7i-flex.large, ForgeGPT INT8 at full load (projected) | local max 14.7 req/s at 10 users | $0.0018 |

![Cost per 1k requests](docs/images/cost_per_1k.png)

Lambda's cost per request is fixed, while an EC2 instance costs ~$70/month whether busy or idle. With these numbers **Lambda is cheaper below about 10 requests/s sustained (~0.9M requests/day)**, which covers any portfolio or internal-tool workload; above that, a busy instance wins. Both rows are *projected* from local measurements at us-east-1 list prices (Lambda $0.20 per 1M requests + $0.0000166667 per GB-s; m7i-flex.large $0.0958/h). After `cdk deploy`, `scripts/cost_model.py` adds measured AWS rows for Lambda and for Qwen2.5-0.5B on llama.cpp.

## Web GUI

`uvicorn serving.app.main:app` serves a playground over the same ONNX Runtime code path that
runs in Lambda: start a complaint, adjust temperature / top-k / top-p, watch tokens stream
in with time-to-first-token and tokens/s, and switch on **Show tokens** to see exactly how
the BPE tokenizer split the prompt and the output.

| Token view (dark mode) | Model card |
|---|---|
| ![Tokens](docs/images/gui_tokens_dark.png) | ![Model](docs/images/gui_model.png) |

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,train,serve,bench]"

# data -> tokenizer -> token files  (~10 min)
python scripts/download_nhtsa.py
python scripts/prepare_data.py
python scripts/train_tokenizer.py
python scripts/encode_data.py

# train (CPU demo; the full model runs on Kaggle: notebooks/kaggle_train.ipynb)
python -m forgegpt.train --config configs/cpu_demo.json

# optimize, benchmark, chart
python -m forgegpt.onnx_export --ckpt runs/cpu_demo/ckpt_best.pt --out-dir artifacts/onnx
python scripts/benchmark_inference.py --ckpt runs/cpu_demo/ckpt_best.pt
uvicorn serving.app.main:app --port 8000 &                # GUI at http://localhost:8000
python scripts/run_load_test.py --host http://localhost:8000
python scripts/cost_model.py
python scripts/make_charts.py --run runs/cpu_demo

pytest                                                   # 55 tests
```

Deploying to AWS (Free plan) and running the full Kaggle training: [docs/DEPLOY.md](docs/DEPLOY.md).

## Engineering decisions

- **Attention written by hand, checked against PyTorch.** The tests compare it with
  `F.scaled_dot_product_attention`, prove the causal mask (changing future tokens never
  changes earlier outputs), overfit a single batch, and check that KV-cached decoding gives
  the same logits as a full forward pass.
- **One ONNX graph for prefill and decode.** The KV cache is a graph input/output
  (`past` → `present`), so the same file handles the prompt (empty cache) and every
  following token. Tests check it against PyTorch for both phases.
- **Lambda without PyTorch.** The handler needs only ONNX Runtime, numpy and tokenizers:
  a 48 MB zip deployed without a container registry, with a 220–350 ms model
  initialization (measured locally on the Lambda Python 3.12 runtime's wheels).
- **Function URL instead of API Gateway**, with an API-key check, a `max_tokens` cap and
  input validation in code, which keeps serving inside Lambda's always-free tier.
- **Metrics with no extra API calls.** Each request logs one JSON line in CloudWatch
  Embedded Metric Format; CloudWatch turns it into TTFT / latency / tokens-per-second
  metrics for the dashboard.
- **Free-plan guardrails in code.** No NAT gateway, EC2 opt-in and IP-restricted, SSM
  instead of SSH, log retention of one week, budgets at $5/$20/$50, and `cdk destroy`
  removes everything including the bucket.

## Limitations

- The demo model is small and trained for 96 minutes on CPU; it writes fluent
  complaint-style text but drifts on long generations and can repeat. The 14M-parameter
  Kaggle run is the intended model.
- Latency and cost numbers are from a 2-vCPU machine; AWS numbers will differ (Lambda at
  2 GB gets about 1.2 vCPU). Rows marked *projected* are computed from local measurements.
- The model continues text; it is not instruction-tuned and does not answer questions.
- A few free-text street addresses in older complaints are not masked (see docs/DATA.md).

## Project layout

```
src/forgegpt/      data, clean, tokenizer, model, train, generate, onnx_export, onnx_infer, serving
scripts/           data pipeline, benchmarks, load tests, cost model, charts
configs/           training configs (CPU demo, Kaggle T4)
notebooks/         Kaggle training + ablations notebook
serving/app/       FastAPI app + web GUI
serving/lambda/    Lambda handler + zip builder
serving/ec2/       llama.cpp user-data + benchmark script
infra/             AWS CDK app (+ synth tests)
benchmarks/        Locust file + results CSVs
tests/             unit and integration tests
docs/              data card, deployment guide, charts
```
