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
python scripts/train_tokenizer.py  # byte-level BPE (8k and 16k) + comparison with GPT-2
python scripts/encode_data.py      # token files for training (uint16 .bin)
```

1.62M unique complaint narratives after cleaning, 149M training tokens. A custom 8k BPE
tokenizer encodes complaints slightly more compactly than GPT-2's 50k vocabulary.
See [`docs/DATA.md`](docs/DATA.md) for every cleaning rule and the numbers behind it.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
