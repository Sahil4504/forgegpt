**ForgeGPT**

*Transformer from scratch → optimized inference → serving on AWS (free
plan only)*

Project plan for Sahil Manikshete · November 2026 · 8 weeks

# 1. What you are building

A small GPT language model written from scratch in PyTorch and trained
on automotive text (NHTSA vehicle complaints). You then optimize it and
an open small LLM for fast CPU inference, serve both on AWS, and publish
a benchmark of speed and cost. The story for interviews is that you
understand transformers end to end, from the attention math to the
latency of each request in production.

**The free-plan rule:** all GPU work happens on Kaggle or Colab (free).
AWS is used only for storage, CPU serving, monitoring and budgets, using
free-plan-eligible resources. The free plan cannot bill you; it simply
stops when the credits or the 6 months run out.

**Why automotive data:** it ties into AutoSense and Michigan’s auto
industry, and the NHTSA complaints dataset is public, large and
domain-specific.

# 2. Architecture at a glance

| **Layer**                   | **Tool**                                                         | **Cost**                                 |
|-----------------------------|------------------------------------------------------------------|------------------------------------------|
| Data + training             | Kaggle Notebooks (T4/P100 GPU), PyTorch, Hugging Face tokenizers | Free (weekly GPU quota)                  |
| Experiment tracking         | MLflow (local) or Weights & Biases free tier                     | Free                                     |
| Artifact storage            | Amazon S3 (model weights, tokenizer, datasets)                   | Credits (pennies at this size)           |
| Serverless serving          | AWS Lambda (container image) + Lambda Function URL               | Always-free tier (1M requests/month)     |
| Open-model serving          | EC2 m7i-flex.large (8 GB RAM) running llama.cpp server           | Credits, about \$0.10/hr; stop when idle |
| Monitoring                  | Amazon CloudWatch logs, custom metrics, one dashboard            | Free-tier limits                         |
| Load testing                | Locust, run from your laptop                                     | Free                                     |
| Infrastructure as code + CI | AWS CDK (Python) or Terraform; GitHub Actions                    | Free                                     |
| Cost guardrail              | AWS Budgets alerts                                               | Free                                     |

# 3. Phase plan

## Phase 0 — Account and guardrails (Week 0, about 2 hours)

**Where it runs:** AWS console + laptop

**Goal:** A safe AWS account before anything else exists.

- **Sign up on the Free plan.** Do not choose Paid. Credits apply only
  to new customers, so if your Cloud Computing course already used an
  AWS account in your name, check eligibility first.

- Turn on MFA for the root user; create an IAM admin user for daily
  work; never create root access keys.

- **Earn the extra \$100:** the five onboarding activities are launching
  an EC2 instance, an RDS database, a Lambda function with a Function
  URL, a Bedrock text prompt, and an AWS Budget. Delete the EC2 and RDS
  resources right after.

- Create budget alerts at \$5, \$20 and \$50 of credit use, emailed to
  you.

- Note your account creation date. The free plan ends 6 months later, so
  all AWS phases must finish before then.

- Create the GitHub repo: src/, notebooks/, serving/, infra/,
  benchmarks/, docs/.

**Done when:** MFA on, budgets live, \$200 in credits showing, repo
pushed.

## Phase 1 — Data pipeline and tokenizer (Week 1)

**Where it runs:** Laptop + Kaggle

**Goal:** A clean, versioned training corpus and your own tokenizer.

- Download the NHTSA complaints flat file; keep the complaint narrative
  text; deduplicate, lowercase or not (decide and document), strip
  PII-like fields.

- Split train/validation/test (90/5/5) and save as Parquet.

- Train a byte-level BPE tokenizer (vocabulary 8k–16k) with Hugging Face
  tokenizers. Compare tokens-per-complaint with GPT-2’s tokenizer, which
  makes a good README chart.

- Upload the corpus and tokenizer to S3 with versioned keys
  (data/v1/...).

**AWS (free plan):** S3 standard bucket with versioning,
block-public-access on. Keep it under a few GB.

**Done when:** Running one script reproduces the dataset and tokenizer
from the raw file.

## Phase 2 — GPT from scratch (Weeks 2–3)

**Where it runs:** Laptop for code, Kaggle GPU for training

**Goal:** A decoder-only transformer you wrote yourself, trained to a
meaningful validation loss.

- Implement token + positional embeddings, causal multi-head
  self-attention, MLP, pre-LayerNorm residual blocks, weight tying and
  the LM head. No nn.Transformer.

- Unit tests: output shapes, the causal mask (a token never sees the
  future), and overfitting a single batch to near-zero loss.

- Training loop: AdamW, warmup + cosine LR schedule, gradient clipping,
  mixed precision (torch.autocast), periodic checkpointing (Kaggle
  sessions time out).

- Target about 10–30M parameters (e.g., 6 layers, 6 heads, 384-dim,
  context 256). It trains in a few Kaggle GPU hours.

- Run 2–3 ablations (e.g., model depth, context length, learning rate)
  and log every run to MLflow/W&B.

- Visualize attention maps for a sample complaint; save generated
  samples at each checkpoint.

- Push the final checkpoint + config to S3 (models/forgegpt/v1/).

**AWS (free plan):** S3 only.

**Done when:** Validation perplexity reported, ablation table filled,
generated text clearly domain-flavored.

## Phase 3 — Inference engine and CPU optimization (Week 4)

**Where it runs:** Laptop (CPU)

**Goal:** Make your model fast on CPU, which is what the free plan gives
you.

- Write generate() with a KV cache, and measure the speedup against no
  cache.

- Add sampling: temperature, top-k, top-p; plus greedy as a baseline.

- Export to ONNX and apply dynamic INT8 quantization; also try
  torch.compile.

- Benchmark script: time to first token, tokens/sec, peak RAM, and
  perplexity change after quantization. Save results as CSV.

**Done when:** A table showing FP32 vs INT8 vs ONNX on speed, memory and
quality.

## Phase 4 — Serverless serving on Lambda (Week 5)

**Where it runs:** AWS

**Goal:** Your from-scratch model behind a public HTTPS endpoint at zero
standing cost.

- Package a FastAPI-style handler (or plain Lambda handler) + ONNX model
  into a container image; pick 2–3 GB memory.

- Load weights at cold start from S3 (or bake them into the image) and
  compare cold-start times for both.

- Expose it with a Lambda Function URL (avoids API Gateway). Add a
  simple API key check and a max_tokens cap to protect your credits.

- Emit structured JSON logs: request id, prompt tokens, generated
  tokens, latency.

**AWS (free plan):** Lambda (always-free tier: 1M requests and 400k
GB-seconds per month), S3, CloudWatch Logs. If container images in ECR
are restricted on your plan, use a zip deployment with a Lambda layer
instead.

**Done when:** curl to the Function URL returns generated text; cold-
and warm-start latency recorded.

## Phase 5 — Open-model serving on EC2 (Week 6)

**Where it runs:** AWS

**Goal:** Serve a real open LLM on the largest free-plan CPU instance
and compare it with your own model.

- Launch m7i-flex.large (2 vCPU, 8 GB), which is free-plan eligible. Use
  Ubuntu, a 20 GB gp3 volume, and a security group open only to your IP.

- Build llama.cpp and run its OpenAI-compatible server with a small
  instruct model in GGUF format (e.g., Qwen2.5-0.5B or 1.5B-Instruct) at
  Q4_K_M and Q8_0.

- Write a user-data script so the whole box rebuilds itself from
  scratch. Then you can terminate it freely.

- **Credit habit:** start it for a benchmarking session, then stop it.
  50 hours of use is roughly \$5 of credits (check current EC2 pricing).

- **Optional GPU comparison:** run the same model with vLLM on a Kaggle
  T4 notebook to get a GPU column for your benchmark table, still at no
  cost.

**AWS (free plan):** EC2 m7i-flex.large (or t3.small for a tiny
always-on demo), EBS gp3.

**Done when:** Both quantization levels serving; tokens/sec and memory
recorded.

## Phase 6 — Load testing and observability (Week 7)

**Where it runs:** Laptop + AWS

**Goal:** Production-style numbers: p50/p95 latency, throughput and cost
per 1,000 requests.

- Locust scenarios: 1, 5, 10 and 20 concurrent users; short vs long
  prompts.

- Publish custom CloudWatch metrics (TTFT, tokens/sec, error rate). Stay
  within the free custom-metric count and build one dashboard.

- Experiment matrix: Lambda vs EC2, your model vs the open model, INT8
  vs FP32, llama.cpp thread count.

- Compute cost per 1k requests from your measured latency and on-demand
  prices. This is the slide recruiters remember.

- **Optional:** call one cheap Bedrock model for the same prompts (a few
  cents of credits) to add a “managed API” row to the comparison.

**AWS (free plan):** CloudWatch metrics, dashboard and logs (free-tier
limits); Bedrock optional.

**Done when:** Benchmark report with charts committed to benchmarks/.

## Phase 7 — Infrastructure as code, CI and portfolio polish (Week 8)

**Where it runs:** Laptop + GitHub

**Goal:** Anyone can recreate and tear down the whole stack with one
command.

- CDK or Terraform for the S3 bucket, Lambda + Function URL, EC2 +
  security group, and CloudWatch dashboard. Include a destroy command
  and use it.

- GitHub Actions: lint, unit tests (shape, mask, KV-cache equivalence),
  and a tiny CPU smoke test of generate().

- README: architecture diagram, results tables, “total AWS credits used:
  \$X”, how to reproduce, limitations.

- Demo: 2–3 minute video plus a write-up (Medium/LinkedIn) on what
  building attention from scratch taught you.

- Final teardown: terminate EC2, delete Lambda/images, empty buckets you
  no longer need.

**Done when:** A fresh clone + one deploy command reproduces the system;
everything is torn down afterwards.

# 4. Expected credit use

These are rough estimates. Check current AWS pricing before you start.

| **Item**                                                   | **Usage**                   | **Approx. credits**      |
|------------------------------------------------------------|-----------------------------|--------------------------|
| Onboarding activities (EC2, RDS, Lambda, Bedrock, Budgets) | Brief, then deleted         | \< \$2                   |
| S3 storage + requests                                      | \< 5 GB for 2 months        | \< \$1                   |
| Lambda serving                                             | Within always-free tier     | \$0                      |
| EC2 m7i-flex.large + 20 GB EBS                             | ~50–80 hours                | \$6–\$10                 |
| CloudWatch metrics, logs, dashboard                        | Small volume                | \$0–\$3                  |
| Bedrock comparison (optional)                              | A few hundred short prompts | \< \$2                   |
| Total                                                      |                             | about \$10–\$20 of \$200 |

# 5. Rules that keep you inside the free plan

- Never switch to the Paid plan. The free plan stops instead of billing
  you.

- Stop EC2 after every session; terminate it when a phase ends
  (user-data rebuilds it).

- Release Elastic IPs and delete unattached EBS volumes and snapshots;
  these cost money while idle.

- Keep everything in one region (e.g., us-east-1) so nothing is
  forgotten elsewhere.

- Check Billing → Free Tier and Credits weekly.

- Keep code, configs and results in GitHub. When the plan ends the
  account closes, and AWS keeps the data for only 90 days.

- Never commit AWS keys; use GitHub OIDC or local profiles.

# 6. Resume bullets (fill in your numbers)

- Implemented a decoder-only transformer (XXM params) from scratch in
  PyTorch, including multi-head causal attention, KV caching and top-p
  sampling; trained on XXk NHTSA vehicle complaints to a validation
  perplexity of XX.

- Cut CPU inference latency by XX% with INT8 quantization and ONNX
  Runtime at \< X% perplexity cost; served the model serverlessly on AWS
  Lambda with XX ms warm p95 latency.

- Benchmarked self-hosted LLM serving (llama.cpp on EC2) against
  serverless and managed APIs on throughput, p95 latency and cost per 1k
  requests; deployed with AWS CDK and GitHub Actions CI for about \$XX
  of compute.

# 7. Skills this covers

Transformers (attention, positional encoding, training dynamics),
tokenization, PyTorch, experiment tracking, ONNX and quantization,
inference optimization, serverless deployment, CPU LLM serving,
observability, load testing, infrastructure as code, CI/CD and cost
engineering.

**Natural next step:** the RAG platform project can reuse this repo’s
serving layer and benchmark harness.
