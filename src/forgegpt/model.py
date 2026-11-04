"""ForgeGPT: a decoder-only transformer, written from scratch in PyTorch.

No nn.Transformer: attention, blocks, embeddings and the LM head are all defined here.
Shape names used throughout:
    B = batch size, T = new tokens in this call, P = tokens already in the KV cache,
    C = d_model (width of a token vector), H = number of heads, D = head_dim = C // H,
    V = vocabulary size, L = number of layers
"""

from __future__ import annotations

import inspect
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# A KV cache entry for one layer: (keys, values), each (B, H, P, D).
LayerCache = tuple[torch.Tensor, torch.Tensor]


@dataclass
class GPTConfig:
    vocab_size: int = 8192  # our bpe-8k tokenizer
    block_size: int = 256  # max context length: how many tokens the model sees at once
    n_layer: int = 6  # number of transformer blocks
    n_head: int = 6  # attention heads per block
    d_model: int = 384  # width of every token vector (384 / 6 heads = 64 per head)
    dropout: float = 0.0  # 0 for pre-training on lots of data; try 0.1 if it overfits
    bias: bool = False  # bias terms in Linear/LayerNorm; modern GPTs drop them

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where each token can only look at itself and the past.

    For every token we compute three vectors:
        query  "what am I looking for?"
        key    "what do I contain?"
        value  "what do I pass on if someone attends to me?"
    The score between token i and token j is how well i's query matches j's key.
    Softmax turns each row of scores into weights that sum to 1, and token i's output is
    the weighted average of the values. With several heads, each head does this in its
    own smaller subspace, so different heads can track different relationships.

    KV cache: when generating, the keys and values of earlier tokens never change, so
    we keep them (`past_kv`) and only compute q, k, v for the new token(s).
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        if cfg.d_model % cfg.n_head:
            raise ValueError(f"d_model ({cfg.d_model}) must be divisible by n_head ({cfg.n_head})")
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head

        # One matrix multiply produces queries, keys and values for all heads at once;
        # it is the same as three separate (C -> C) layers, just faster.
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        # After the heads are concatenated, this mixes their outputs back together.
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Causal mask: lower-triangular True/False matrix, row i is True for columns 0..i.
        # A buffer moves with the model to the GPU but is not a trainable parameter;
        # persistent=False keeps it out of checkpoints (it is rebuilt from the config).
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer(
            "causal_mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False
        )
        # Set to True to keep the last attention weights (for visualisation only).
        self.store_weights = False
        self.last_weights: torch.Tensor | None = None

    def forward(
        self, x: torch.Tensor, past_kv: LayerCache | None = None, use_cache: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, LayerCache]:
        B, T, C = x.shape

        # 1. Project every token to query, key, value.               each (B, T, C)
        q, k, v = self.qkv(x).split(C, dim=2)

        # 2. Split C into H heads of size D and move heads next to batch, so every
        #    head is processed in parallel as if it were a separate batch item.
        #    (B, T, C) -> (B, T, H, D) -> (B, H, T, D)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 2b. KV cache: prepend keys/values of the P earlier tokens.  k, v: (B, H, P+T, D)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        total = k.size(2)
        past = total - T

        # 3. Scores: every query against every key.  (B,H,T,D) @ (B,H,D,P+T) -> (B,H,T,P+T)
        #    Dividing by sqrt(D) keeps the scores' variance around 1. Without it, dot
        #    products of 64-dim vectors get large, softmax saturates to nearly one-hot,
        #    and gradients vanish.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 4. Causal mask: the new tokens sit at positions past..total-1, so take those
        #    rows of the mask. Future positions get -inf, which softmax turns into 0.
        mask = self.causal_mask[:, :, past:total, :total]
        scores = scores.masked_fill(~mask, float("-inf"))

        # 5. Softmax over the last axis: each row becomes attention weights summing to 1.
        weights = F.softmax(scores, dim=-1)
        if self.store_weights:
            self.last_weights = weights.detach()
        weights = self.attn_dropout(weights)

        # 6. Weighted average of values.         (B,H,T,P+T) @ (B,H,P+T,D) -> (B,H,T,D)
        out = weights @ v

        # 7. Merge heads back: (B,H,T,D) -> (B,T,H,D) -> (B,T,C). transpose() only changes
        #    the view of memory, so .contiguous() is needed before view() can reshape it.
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # 8. Mix the heads' outputs together.                                  (B, T, C)
        out = self.resid_dropout(self.proj(out))
        return (out, (k, v)) if use_cache else out


class MLP(nn.Module):
    """Position-wise feed-forward network: widen to 4C, GELU, project back to C.

    Attention moves information *between* tokens; the MLP then processes each token on
    its own. Most of a transformer's parameters (and much of its "knowledge") live here.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x), approximate="tanh")))


class Block(nn.Module):
    """One transformer block, pre-LayerNorm style (as in GPT-2):

        x = x + attention(layernorm(x))
        x = x + mlp(layernorm(x))

    The residual "x +" gives gradients a direct path through all layers, and normalising
    *before* each sub-layer (pre-LN) makes deep stacks train stably without tricks.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(
        self, x: torch.Tensor, past_kv: LayerCache | None = None, use_cache: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, LayerCache]:
        if use_cache:
            a, present = self.attn(self.ln1(x), past_kv=past_kv, use_cache=True)
            x = x + a
            return x + self.mlp(self.ln2(x)), present
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    """Token + position embeddings -> L blocks -> final LayerNorm -> LM head."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)  # token embeddings
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)  # learned position embeddings
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Weight tying: the output layer reuses the input embedding matrix. A token's
        # "meaning" vector is used both to read it in and to score it as the next
        # token. Saves V*C parameters (3.1M here) and usually improves quality.
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # GPT-2 trick: scale the init of layers that write into the residual stream by
        # 1/sqrt(2L), so the sum over 2L residual additions does not grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        """Parameter count. Tied weights are counted once (named_parameters dedups)."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wte.weight.numel() + self.wpe.weight.numel()
        return n

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_kvs: list[LayerCache] | None = None,
        use_cache: bool = False,
    ):
        """idx: (B, T) token ids. Returns (logits, loss) or, with use_cache=True,
        (logits, loss, presents) where presents is the updated per-layer KV cache."""
        B, T = idx.shape
        past = 0 if past_kvs is None else past_kvs[0][0].size(2)
        if past + T > self.cfg.block_size:
            raise ValueError(
                f"sequence of {past + T} tokens exceeds block_size {self.cfg.block_size}"
            )

        pos = torch.arange(past, past + T, device=idx.device)  # positions of the new tokens
        x = self.drop(self.wte(idx) + self.wpe(pos))  # (B, T, C)

        presents: list[LayerCache] = []
        for i, block in enumerate(self.blocks):
            if use_cache:
                x, present = block(
                    x, past_kv=None if past_kvs is None else past_kvs[i], use_cache=True
                )
                presents.append(present)
            else:
                x = block(x)
        logits = self.lm_head(self.ln_f(x))  # (B, T, V): a score for every next token

        loss = None
        if targets is not None:
            # Cross-entropy = -log(probability the model gave to the true next token),
            # averaged over every position in the batch.
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return (logits, loss, presents) if use_cache else (logits, loss)

    def configure_optimizer(
        self, weight_decay: float, lr: float, betas: tuple[float, float], device_type: str
    ) -> torch.optim.AdamW:
        """AdamW with weight decay on matrices only. Biases, LayerNorm gains and other
        1-D parameters are not decayed: shrinking them toward 0 only hurts."""
        params = [p for p in self.parameters() if p.requires_grad]
        groups = [
            {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
            {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
        ]
        fused = device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters
        return torch.optim.AdamW(groups, lr=lr, betas=betas, **({"fused": True} if fused else {}))
