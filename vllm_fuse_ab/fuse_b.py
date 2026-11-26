# 位置 B：MoE 输出的 AR + 下一层 input_layernorm 换成 FlagTree 的 RS + add + RMSNorm（每卡只留自己的行），
# qkv_a GEMM（和 full indexer 层的 wk_weights_proj）只算本卡行，再用 q / kv norm + AG 汇齐 [T, D] 给 attention。
#
#   原生：AR(MoE 部分和) → add + RMSNorm（T 行）→ fused_qkv_a_proj（T 行）→ q_a_layernorm / kv_a_layernorm
#         full indexer 层另有 indexer.wk_weights_proj(hidden_states)（T 行）
#   融合：rs+norm（本卡行）→ fused_qkv_a_proj（本卡行）[→ wk_weights_proj（本卡行），拼在行尾]
#         → q / kv norm + AG（行尾原样传）→ [T, q_lora | kv_lora | rope (| head_dim + n_head)]
#
# 数学和原生逐 token 相同，只是每卡少算 7/8 的行、不再 AG 完整 hidden。residual 只更新本卡 row_range 的行，
# 和位置 A 的约定一致，所以打开 B 时 A 之后不再 AG residual（fuse_a.GATHER_RESIDUAL 默认随之为 0）。
# 用 fuse_a 的 RingComm 和同一套原生判断（T <= NATIVE_MAX_TOKENS 或录 CUDA graph 时走原生）：
# 同一次 fwd 里 A 和 B 的路径一致。
#
# 环境变量：VLLM_HCU_FUSE_B=1 打开（需要 VLLM_HCU_FUSE_A=1）。只作用于前一层是融合 MoE 层的那些层的输入。
import os

import torch

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_fuse_ab import fuse_a

logger = init_logger(__name__)

ENABLED = fuse_a.ENABLED and fuse_a.FUSE_B

_attn = {}  # layer_idx -> MultiHeadLatentAttentionWrapper
_qkv_ops = {}  # 行宽 D -> fused_qkv_norm_ag


def use_for_layer(vllm_config, config, layer_idx: int) -> bool:
    """第 layer_idx 层的 input_layernorm（及其前面上一层 MoE 输出的 AR）是否走位置 B。"""
    return (ENABLED and 0 < layer_idx < config.num_hidden_layers
            and fuse_a.use_for_layer(vllm_config, config, layer_idx - 1))


def moe_reduce_results(vllm_config, config, layer_idx: int) -> bool:
    """第 layer_idx 层 MoE 的 FusedMoE 要不要自己做 AR：最后一层交给 final norm，下一层走 B 时交给 B。"""
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
    """x：上一层 MoE 的部分和 [T, H]；residual：[T, H]，原地更新（融合时只更新本卡行）。
    返回 [T, D] = q_c（已 q_a_layernorm）| kv_c（已 kv_a_layernorm）| k_pe | [indexer wk_weights_proj 输出]。"""
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
    """DecoderLayer 里走 B 的层：替换 input_layernorm(hidden_states, residual)，返回给 mla_forward 的 [T, D]。"""
    norm = layer.input_layernorm
    return torch.ops.vllm.hcu_fuse_b_rs_norm_qkv(hidden_states, residual, norm.weight, norm.variance_epsilon,
                                                  layer.layer_idx)


def mla_forward(w, positions, qkv, llama_4_scaling=None):
    """MultiHeadLatentAttentionWrapper.forward（厂商打过补丁的版本）在 q_lora 分支下的同一流程，
    只是 fused_qkv_a_proj、q_a_layernorm、kv_a_layernorm、indexer 的 wk_weights_proj 已经在 B 里做完。"""
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
        # indexer_op 只用 hidden_states 的行数和 device
        w.indexer(qkv, q_c, positions, w.indexer_rope_emb, kw=qkv[:, Q + KV + R:].contiguous())
    if llama_4_scaling is not None:
        q *= llama_4_scaling
    attn_out = w.mla_attn(q, kv_c_normed, k_pe, output_shape=(T, w.num_heads * w.v_head_dim))
    return w.o_proj(attn_out)[0]
