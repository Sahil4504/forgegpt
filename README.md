# ForgeGPT

A small GPT written from scratch in PyTorch, trained on NHTSA vehicle complaints,
optimized for CPU inference (KV cache, ONNX, INT8), and served on AWS
(Lambda + EC2 with llama.cpp) with latency and cost benchmarks.

> Work in progress. See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) for the roadmap.

## Repo layout

```
src/forgegpt/   model, tokenizer, data, training, generation
scripts/        one-off entry points (data download, preprocessing)
notebooks/      Kaggle training notebooks
serving/        Lambda and EC2 deployment code
benchmarks/     load tests, results, charts
infra/          infrastructure as code
tests/          unit tests
docs/           plan, architecture, write-ups
```

## Data

```bash
python scripts/download_nhtsa.py   # download NHTSA complaints, extract narratives
python scripts/prepare_data.py     # clean, dedupe, split 90/5/5
```

1.62M unique complaint narratives (~780M characters) after cleaning.
See [`docs/DATA.md`](docs/DATA.md) for every cleaning rule and the numbers behind it.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
