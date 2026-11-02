# ForgeGPT — Project Brief for Claude

Read this first. It is the full context for this repo; the detailed phase plan is in `docs/PROJECT_PLAN.md`.

## Who I am and why this exists
- Sahil Manikshete, Data Science grad student at the University of Michigan–Dearborn, graduating May 2027.
- Targeting AI Engineer / ML Engineer roles (also Data Engineer) in the US. This is a **portfolio project**: it must teach me the fundamentals AND look production-grade to recruiters.
- Intermediate developer. Explain non-obvious decisions briefly; don't hand me a finished repo in one go.

## What we are building
**ForgeGPT: transformer from scratch → optimized inference → serving on AWS (free plan only).**
1. A small decoder-only GPT (~10–30M params) written from scratch in PyTorch (no `nn.Transformer`).
2. Trained on automotive text: the public **NHTSA vehicle complaints** dataset (narrative field). Domain chosen to tie in with my AutoSense project and Michigan's auto industry.
3. Optimized for CPU inference: KV cache, sampling (greedy/temperature/top-k/top-p), ONNX export, INT8 quantization.
4. Served two ways on AWS: my model on **Lambda (Function URL)**, and an open small LLM (Qwen2.5-0.5B/1.5B-Instruct GGUF) via **llama.cpp on EC2 m7i-flex.large**.
5. Benchmarked: TTFT, tokens/sec, p50/p95 latency, memory, cost per 1k requests (Locust + CloudWatch). Optional rows: vLLM on a Kaggle T4, and a cheap Bedrock model.
6. Reproducible with IaC (AWS CDK in Python, or Terraform) + GitHub Actions CI.

## Hard constraint: AWS FREE PLAN ONLY
- Account is on the AWS **Free plan** (created after July 15, 2025): up to $200 credits, ends after 6 months or when credits run out. **Never suggest switching to the Paid plan.**
- **No GPU instances on AWS.** All GPU training happens on **Kaggle Notebooks** (primary) or Colab free tier.
- Free-plan-eligible EC2 types: t3.micro, t3.small, t4g.micro, t4g.small, c7i-flex.large, m7i-flex.large. Nothing xlarge or above.
- Allowed services we use: S3, Lambda (+ Function URL, avoids API Gateway), EC2 (above types), EBS gp3, CloudWatch, AWS Budgets, optionally Bedrock. If a service is blocked on the free plan, propose a free alternative (e.g. Lambda zip + layer if ECR is restricted).
- Target total spend: ~$10–20 of credits. Stop EC2 after every session; tear down at phase end; one region (us-east-1).
- Never commit AWS keys or secrets. Use local AWS profiles / GitHub OIDC.

## How to work with me
- **Build incrementally, phase by phase.** Finish and verify one phase before starting the next. Ask before jumping ahead.
- At the start of a session, check "Current status" below and continue from there; update it when a phase is done.
- Every phase ends with something verifiable (tests pass, a table of numbers, an endpoint that responds).
- Prefer simple, readable code over clever abstractions. Add tests for the transformer internals (shapes, causal mask, overfit-one-batch, KV-cache equivalence).
- Training code must checkpoint regularly (Kaggle sessions time out) and log runs to MLflow (local) or W&B free tier.
- When I need a document, I prefer clean, condensed Word (.docx) output.

## Planned repo layout
```
src/forgegpt/      model.py, tokenizer.py, data.py, train.py, generate.py
notebooks/         Kaggle training notebooks
serving/lambda/    Lambda handler + Dockerfile / zip packaging
serving/ec2/       llama.cpp user-data script, server config
benchmarks/        Locust files, benchmark scripts, results CSVs, charts
infra/             CDK or Terraform
tests/             pytest unit tests
docs/              plan, architecture diagram, write-ups
```

## Phases (details in docs/PROJECT_PLAN.md)
| # | Phase | Runs on | Week |
|---|-------|---------|------|
| 0 | AWS account + guardrails (MFA, IAM user, Budgets $5/$20/$50, 5 onboarding activities for +$100 credits), repo setup | AWS console, laptop | 0 |
| 1 | Data pipeline + BPE tokenizer (8k–16k vocab), data to S3 | Laptop, Kaggle | 1 |
| 2 | GPT from scratch + training + ablations | Kaggle GPU | 2–3 |
| 3 | Inference engine + CPU optimization (KV cache, ONNX, INT8) | Laptop | 4 |
| 4 | Serverless serving on Lambda + Function URL | AWS | 5 |
| 5 | Open model on EC2 m7i-flex.large with llama.cpp (Q4_K_M vs Q8_0) | AWS | 6 |
| 6 | Load testing + CloudWatch observability + cost per 1k requests | Laptop + AWS | 7 |
| 7 | IaC, CI, README, demo video, final teardown | Laptop + GitHub | 8 |

## Current status
- [ ] Phase 0 — in progress: repo scaffold created 2026-11-01; AWS account setup pending
- [ ] Phase 1 — in progress: download + cleaning + splits done (1.62M narratives, see docs/DATA.md); next: BPE tokenizer, then S3 upload
- AWS account creation date: _fill in_ (free plan ends 6 months after this)

## Decisions log
- 2026-11-01: Lowercase all narratives (pre-2021 data is ALL CAPS). Mask phone/email/URL/VIN with placeholder tokens. Hash-based 90/5/5 split after dedup.
- 2026-11-01: Chose free-plan-only AWS; GPU work moved to Kaggle/Colab. Dataset: NHTSA complaints. Serving: Lambda (own model) + llama.cpp on m7i-flex.large (open model).
