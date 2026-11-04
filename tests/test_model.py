"""Tests for the full GPT: shapes, init, weight tying, learning, and KV-cache equivalence."""

import math

import pytest
import torch

from forgegpt.generate import SamplingConfig, generate, sample_next
from forgegpt.model import GPT, GPTConfig

TINY = GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=4, d_model=64)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return GPT(TINY).eval()


def test_default_config_size():
    """The planned ~14M-parameter model (6 layers, 6 heads, 384 wide, 8k vocab)."""
    n = GPT(GPTConfig()).num_params()
    assert 13_000_000 < n < 15_000_000


def test_weight_tying(model):
    assert model.lm_head.weight is model.wte.weight


def test_logits_shape_and_initial_loss(model):
    idx = torch.randint(0, TINY.vocab_size, (3, 20))
    targets = torch.randint(0, TINY.vocab_size, (3, 20))
    logits, loss = model(idx, targets=targets)
    assert logits.shape == (3, 20, TINY.vocab_size)
    # At init the model should be ~uniform over the vocab: loss close to ln(V).
    assert abs(loss.item() - math.log(TINY.vocab_size)) < 0.3


def test_model_is_causal(model):
    idx = torch.randint(0, TINY.vocab_size, (1, 16))
    changed = idx.clone()
    changed[0, 10:] = (changed[0, 10:] + 1) % TINY.vocab_size
    a, _ = model(idx)
    b, _ = model(changed)
    torch.testing.assert_close(a[:, :10], b[:, :10])


def test_rejects_too_long_sequence(model):
    with pytest.raises(ValueError):
        model(torch.zeros(1, TINY.block_size + 1, dtype=torch.long))


def test_overfits_a_single_batch():
    """If the model + loss + optimizer are wired correctly, it must memorise one batch."""
    torch.manual_seed(0)
    m = GPT(TINY)
    opt = m.configure_optimizer(weight_decay=0.0, lr=3e-3, betas=(0.9, 0.95), device_type="cpu")
    data = torch.randint(0, TINY.vocab_size, (4, TINY.block_size + 1))
    x, y = data[:, :-1], data[:, 1:]
    for _ in range(150):
        _, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_kv_cache_matches_full_forward(model):
    """Prefill + one-token-at-a-time decoding must give the same logits as one full pass."""
    idx = torch.randint(0, TINY.vocab_size, (2, 24))
    full, _ = model(idx)

    logits, _, cache = model(idx[:, :10], use_cache=True)  # prefill 10 tokens
    steps = [logits]
    for t in range(10, 24):  # then decode the rest one token at a time
        logits, _, cache = model(idx[:, t : t + 1], past_kvs=cache, use_cache=True)
        steps.append(logits)
    torch.testing.assert_close(torch.cat(steps, dim=1), full, atol=1e-5, rtol=1e-4)
    assert cache[0][0].shape == (2, TINY.n_head, 24, TINY.d_model // TINY.n_head)


def test_greedy_generation_identical_with_and_without_cache(model):
    cfg = SamplingConfig(max_new_tokens=40, temperature=0.0, stop_at_eot=False)
    prompt = [5, 9, 13]  # 3 + 40 tokens also exceeds block_size: tests the sliding window
    assert generate(model, prompt, cfg, use_cache=True) == generate(
        model, prompt, cfg, use_cache=False
    )


def test_top_k_and_top_p_restrict_choices():
    logits = torch.tensor([[5.0, 4.0, 1.0, 0.0, -1.0]])
    g = torch.Generator().manual_seed(0)
    k2 = SamplingConfig(temperature=1.0, top_k=2, top_p=None)
    assert {sample_next(logits, k2, g).item() for _ in range(200)} <= {0, 1}
    p = SamplingConfig(temperature=1.0, top_k=None, top_p=0.5)  # token 0 alone has ~70%
    assert {sample_next(logits, p, g).item() for _ in range(200)} == {0}


def test_sampling_is_reproducible_with_seed(model):
    cfg = SamplingConfig(max_new_tokens=15, stop_at_eot=False)
    assert generate(model, [1, 2], cfg, seed=7) == generate(model, [1, 2], cfg, seed=7)
