import os

import torch

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_fuse_ab import fuse_a

logger = init_logger(__name__)

ENABLED = fuse_a.ENABLED and fuse_a.FUSE_B

_attn = {}  # layer_idx -> MultiHeadLatentAttentionWrapper
_qkv_ops = {}


def use_for_layer(vllm_config, config, layer_idx: int) -> bool:
    return (ENABLED and 0 < layer_idx < config.num_hidden_layers
            and fuse_a.use_for_layer(vllm_config, config, layer_idx - 1))


def moe_reduce_results(vllm_config, config, layer_idx: int) -> bool:
    return not (fuse_a.use_for_final(vllm_config, config, layer_idx)
                or use_for_layer(vllm_config, config, layer_idx + 1))


def register_attn(layer_idx: int, attn):
    w = attn.mla_attn
    if w.q_lora_rank is None or w.fused_qkv_a_proj is None:
        raise RuntimeError(f"fuse B expects MLA with q_lora at layer {layer_idx}")
    _attn[layer_idx] = w


def _has_kw(w) -> bool:
    return bool(w.indexer is not None and w.is_sparse and not w.skip_topk)


def _widths(w):
    kw = w.indexer.head_dim + w.indexer.n_head if _has_kw(w) else 0
    return w.q_lora_rank, w.kv_lora_rank, w.qk_rope_head_dim, kw


def _qkv_op(D):
    op = _qkv_ops.get(D)
    if op is None:
        w = next(iter(_attn.values()))
        op = fuse_a._mod.fused_qkv_norm_ag(fuse_a._comm, fuse_a._max_tokens, w.q_lora_rank, w.kv_lora_rank, D,
                                           fuse_a._dtype)
        _qkv_ops[D] = op
    return op


def _rs_norm_qkv(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float,
                 layer_idx: int) -> torch.Tensor:
    w = _attn[layer_idx]
    Q, KV, R, KW = _widths(w)
    T = x.shape[0]
    if not fuse_a._fused(T, residual):
        x = tensor_model_parallel_all_reduce(x)
        z = x.float() + residual.float()
        residual.copy_(z)
        y = (z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)
        qkv = w.fused_qkv_a_proj(y)[0]
        q_c, kv_c, k_pe = qkv.split([Q, KV, R], dim=-1)
        parts = [w.q_a_layernorm(q_c), w.kv_a_layernorm(kv_c), k_pe]
        if KW:
            parts.append(w.indexer.wk_weights_proj(y)[0])
        return torch.cat(parts, dim=-1)
    b, e = fuse_a._comm.row_range(T)
    y = fuse_a._op(x.contiguous(), residual[b:e], weight, eps, out=x.new_empty((e - b, x.shape[1])), gather=False)
    rows = w.fused_qkv_a_proj(y)[0]
    if KW:
        rows = torch.cat([rows, w.indexer.wk_weights_proj(y)[0]], dim=-1)
    D = Q + KV + R + KW
    return _qkv_op(D)(rows.contiguous(), T, w.q_a_layernorm.weight, w.q_a_layernorm.variance_epsilon,
                      w.kv_a_layernorm.weight, w.kv_a_layernorm.variance_epsilon)


def _rs_norm_qkv_fake(x, residual, weight, eps, layer_idx):
    Q, KV, R, KW = _widths(_attn[layer_idx])
    return x.new_empty((x.shape[0], Q + KV + R + KW))


direct_register_custom_op(
    op_name="hcu_fuse_b_rs_norm_qkv",
    op_func=_rs_norm_qkv,
    mutates_args=["residual"],
    fake_impl=_rs_norm_qkv_fake,
)


def input_layernorm(layer, hidden_states, residual):
    norm = layer.input_layernorm
    return torch.ops.vllm.hcu_fuse_b_rs_norm_qkv(hidden_states, residual, norm.weight, norm.variance_epsilon,
                                                  layer.layer_idx)


def mla_forward(w, positions, qkv, llama_4_scaling=None):
    Q, KV, R, KW = _widths(w)
    T = qkv.shape[0]
    q_c = qkv[:, :Q].contiguous()
    kv_c_normed = qkv[:, Q:Q + KV].contiguous()
    k_pe = qkv[:, Q + KV:Q + KV + R]
    q = w.q_b_proj(q_c)[0]
    q = q.view(-1, w.num_heads, w.qk_head_dim)
    k_pe = k_pe.unsqueeze(1)
    if w.rotary_emb is not None:
        q[..., w.qk_nope_head_dim:], k_pe = w.rotary_emb(positions, q[..., w.qk_nope_head_dim:], k_pe)
    if KW:
        w.indexer(qkv, q_c, positions, w.indexer_rope_emb, kw=qkv[:, Q + KV + R:].contiguous())
    if llama_4_scaling is not None:
        q *= llama_4_scaling
    attn_out = w.mla_attn(q, kv_c_normed, k_pe, output_shape=(T, w.num_heads * w.v_head_dim))
    return w.o_proj(attn_out)[0]
