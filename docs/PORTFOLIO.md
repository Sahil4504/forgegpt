# Portfolio notes

## Resume bullets (every number is measured; update them after the Kaggle run)

- Implemented a GPT-style decoder-only transformer from scratch in PyTorch (causal multi-head
  attention, pre-LN blocks, weight tying, KV cache, top-k/top-p sampling) and trained it on
  1.6M deduplicated NHTSA vehicle complaints (149M tokens, custom 8k byte-level BPE),
  reaching a validation perplexity of 29.8 with a 5.3M-parameter CPU demo model.
- Exported the model to ONNX with the KV cache inside the graph and applied INT8
  quantization, cutting CPU generation time 9.4x (94 → 876 tokens/s) and memory 4.6x
  (636 → 137 MB) at a 1.2% perplexity cost.
- Served the INT8 model through a FastAPI API with token streaming and a web UI; load-tested
  with Locust (14.7 req/s peak, 76 ms single-user p50 latency on 2 vCPUs), backed by 55 unit
  and integration tests in GitHub Actions CI.

## Interview talking points

1. **Why divide attention scores by sqrt(d)?** Dot products of d-dimensional vectors grow
   with d; without scaling, softmax saturates and gradients vanish.
2. **How does the KV cache work and what does it cost?** Keys/values of past tokens never
   change, so they are stored and only the new token is processed: per-token time stays flat
   (~4.5 ms) instead of growing (9.6 → 17.8 ms by position 128). The cost is memory:
   2 × layers × heads × head_dim × tokens floats per sequence.
3. **How do you know your attention is correct?** It matches
   `F.scaled_dot_product_attention`, changing future tokens never changes earlier outputs,
   the model overfits one batch, and cached decoding equals a full forward pass.
4. **Why was the corpus lowercased?** Pre-2021 complaints are ~100% upper case; keeping case
   would split the data into two dialects and waste tokenizer vocabulary.
5. **Why an 8k vocabulary?** 16k saved only 2.5% of tokens but doubled embedding parameters;
   the 8k domain tokenizer is as compact on complaints as GPT-2's 50k vocabulary.
6. **Why ONNX Runtime and not PyTorch on Lambda?** PyTorch does not fit a zip deployment and
   uses 4x the memory; ORT's INT8 kernels are what give the speedup (PyTorch dynamic INT8 was
   no faster).
7. **How would you deploy it?** The repo is Lambda-ready (ONNX Runtime-only zip + AWS CDK) but
   not deployed. Lambda costs per request, EC2 per hour; projecting from local measurements,
   Lambda is cheaper below ~10 req/s sustained, and a busy instance wins above that.
8. **What would you do next?** Train the 14M model on Kaggle, deploy and replace projected
   numbers with AWS measurements, add response streaming, and fine-tune on complaint
   summarization to make it useful beyond text continuation.
