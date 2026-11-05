# Portfolio notes

## Resume bullets (every number is measured)

- Implemented a GPT-style decoder-only transformer from scratch in PyTorch (causal multi-head
  attention, pre-LN blocks, weight tying, KV cache, top-k/top-p sampling) and trained a
  13.9M-parameter model on 1.6M deduplicated NHTSA vehicle complaints (163.8M tokens, custom
  8k byte-level BPE) on a Kaggle T4 in 44 minutes, reaching a validation perplexity of 15.7;
  ran ablations on depth, context length and learning rate.
- Exported the model to ONNX with the KV cache inside the graph and applied INT8
  quantization, speeding up CPU generation 11.3x (53 → 606 tokens/s; 64 tokens in 105 ms vs
  1.19 s) and cutting memory 3.1x (676 → 220 MB) at a 0.5% perplexity cost.
- Served the INT8 model through a FastAPI API with token streaming and a web UI; load-tested
  with Locust (13.1 req/s peak, 120 ms single-user p50 latency on 2 vCPUs), backed by 56 unit
  and integration tests in GitHub Actions CI.

## Interview talking points

1. **Why divide attention scores by sqrt(d)?** Dot products of d-dimensional vectors grow
   with d; without scaling, softmax saturates and gradients vanish.
2. **How does the KV cache work and what does it cost?** Keys/values of past tokens never
   change, so they are stored and only the new token is processed: per-token time stays flat
   (~5–6 ms) instead of growing (~15 → ~51 ms by position 256). The cost is memory:
   2 × layers × heads × head_dim × tokens floats per sequence.
3. **How do you know your attention is correct?** It matches
   `F.scaled_dot_product_attention`, changing future tokens never changes earlier outputs,
   the model overfits one batch, and cached decoding equals a full forward pass.
4. **Why was the corpus lowercased?** Pre-2021 complaints are ~100% upper case; keeping case
   would split the data into two dialects and waste tokenizer vocabulary.
5. **Why an 8k vocabulary?** 16k saved only 2.5% of tokens but doubled embedding parameters;
   the 8k domain tokenizer is as compact on complaints as GPT-2's 50k vocabulary.
6. **What did the ablations show?** At an equal 32.8M-token budget: a 3x higher learning rate
   helped most (-0.047 loss), depth helped steadily but 8 layers cost 25% more time for -0.015,
   and context 128 vs 256 made no difference because most complaints are under 128 tokens.
   One seed each, so small gaps are noise.
7. **A bug you found?** The first Kaggle run trained in bf16 on a T4. PyTorch reports bf16 as
   supported there because it can emulate it, which is much slower. I switched the check to
   native support (compute capability 8.0+) so a T4 uses fp16 with a gradient scaler.
8. **Why ONNX Runtime and not PyTorch for serving?** It needs a third of the memory (no
   PyTorch import), and its INT8 kernels gave a 2.6x speedup over FP32, while PyTorch's own
   dynamic INT8 was 2.5x slower than ORT INT8 and lost more quality (perplexity 16.20 vs 16.07).
9. **How would you deploy it?** The repo is Lambda-ready (ONNX Runtime-only zip + AWS CDK) but
   not deployed. Lambda costs per request, EC2 per hour; projecting from local measurements,
   Lambda is cheaper below ~7 req/s sustained, and a busy instance wins above that.
10. **What would you do next?** Train longer with the higher learning rate, deploy and replace
    projected numbers with AWS measurements, and fine-tune on complaint summarization to make
    it useful beyond text continuation.
