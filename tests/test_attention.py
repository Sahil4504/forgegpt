"""Tests for CausalSelfAttention: shapes, causality, and agreement with PyTorch's own."""

import pytest
import torch
import torch.nn.functional as F

from forgegpt.model import CausalSelfAttention, GPTConfig

CFG = GPTConfig(block_size=16, n_head=4, d_model=32)


@pytest.fixture
def attn():
    torch.manual_seed(0)
    return CausalSelfAttention(CFG).eval()


def test_output_shape_matches_input(attn):
    x = torch.randn(2, 10, CFG.d_model)
    assert attn(x).shape == (2, 10, CFG.d_model)


def test_full_context_and_single_token(attn):
    assert attn(torch.randn(1, CFG.block_size, CFG.d_model)).shape == (1, CFG.block_size, 32)
    assert attn(torch.randn(1, 1, CFG.d_model)).shape == (1, 1, 32)


def test_no_information_flows_from_the_future(attn):
    """Changing tokens 5.. must not change the outputs at positions 0..4."""
    x = torch.randn(1, 8, CFG.d_model)
    x_changed = x.clone()
    x_changed[:, 5:] = torch.randn(1, 3, CFG.d_model)

    out, out_changed = attn(x), attn(x_changed)

    torch.testing.assert_close(out[:, :5], out_changed[:, :5])
    assert not torch.allclose(out[:, 5:], out_changed[:, 5:])


def test_first_token_only_sees_itself(attn):
    """Token 0 has one allowed position (itself), so its weight is 1 and its output
    is just the projection of its own value vector."""
    x = torch.randn(1, 6, CFG.d_model)
    _, _, v = attn.qkv(x).split(CFG.d_model, dim=2)
    torch.testing.assert_close(attn(x)[:, 0], attn.proj(v[:, 0]))


def test_matches_pytorch_reference(attn):
    """Our hand-written math must equal F.scaled_dot_product_attention."""
    B, T, C, H = 2, 12, CFG.d_model, CFG.n_head
    x = torch.randn(B, T, C)
    q, k, v = attn.qkv(x).split(C, dim=2)
    q, k, v = (t.view(B, T, H, C // H).transpose(1, 2) for t in (q, k, v))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = attn.proj(ref.transpose(1, 2).reshape(B, T, C))
    torch.testing.assert_close(attn(x), ref)


def test_gradients_reach_every_parameter(attn):
    attn.train()
    attn(torch.randn(2, 8, CFG.d_model)).sum().backward()
    for name, p in attn.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_rejects_head_count_that_does_not_divide_width():
    with pytest.raises(ValueError):
        CausalSelfAttention(GPTConfig(d_model=30, n_head=4))


def test_mask_is_not_saved_in_checkpoints(attn):
    assert "causal_mask" not in attn.state_dict()
