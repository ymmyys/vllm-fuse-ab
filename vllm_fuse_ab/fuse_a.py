# 位置 A：MoE 层 o_proj 不做 AR，post_attention_layernorm 换成 FlagTree 的 RS + add + RMSNorm + AG。
#
# residual 始终是完整的 [T, H]。走 fused 时只有本卡 row_range(T, W, rank) 那些行有效（kernel 原地更新），
# 走原生时所有行都有效。T <= VLLM_HCU_FUSE_A_NATIVE_MAX_TOKENS 或在录 CUDA graph 时走原生
# （AR + add + RMSNorm）：kernel 的 flag base / parity 由 Python 每次调用算好传入，不能进 graph。
# 判断放在 custom op 里：vLLM 只编译一次图，写在 forward 里的分支会被固定成第一次编译时的结果。
#
# 环境变量：
#   VLLM_HCU_FUSE_A=1                        打开
#   VLLM_HCU_FUSE_A_KERNEL                   FlagTree 的 10-intra-node-rs-norm-ag.py 路径
#   VLLM_HCU_FUSE_A_NATIVE_MAX_TOKENS (128)  不超过它走原生，等于 P 的最大 CUDA graph capture size
#   VLLM_HCU_FUSE_A_GATHER_RESIDUAL          fused 后把 residual AG 回完整行；位置 B 没打开时默认 1（必须），
#                                            VLLM_HCU_FUSE_B=1 时默认 0
#   VLLM_HCU_FUSE_A_FINAL (1)                最后一层 MoE 的 AR + final norm 也换成 RS + add + RMSNorm + AG
#                                            （输出给 lm_head，之后不再用 residual，不做 AG）
#   VLLM_HCU_FUSE_A_QUANT (0)                A 之后的 fp8 量化（per-token e4m3fn，scale = max(amax, 1e-10)/448）：
#     0  AG bf16，MoE 里 shared / routed 各自量化（原来的做法）
#     1  AG bf16，A 之后对完整 T 行量化一次，shared 和 routed 共用；router 仍在 MoE 里用 bf16 算
#     2  kernel 里本卡行量化后 AG fp8 + scale；router（gate + topk）在本卡 T/8 行 bf16 上算，
#        topk_weights / topk_ids 打包做一次 NCCL AG；MoE 收到的 hidden 只是形状占位
import importlib.util
import os

import torch

from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

ENABLED = os.environ.get("VLLM_HCU_FUSE_A", "0") == "1"
KERNEL_PATH = os.environ.get("VLLM_HCU_FUSE_A_KERNEL", "/root/fuse-a/10-intra-node-rs-norm-ag.py")
NATIVE_MAX_TOKENS = int(os.environ.get("VLLM_HCU_FUSE_A_NATIVE_MAX_TOKENS", "128"))
FUSE_B = os.environ.get("VLLM_HCU_FUSE_B", "0") == "1"
# 位置 B（vllm_fuse_ab/fuse_b.py）打开后，B 只读本卡的 residual 行，A 之后不用再 AG residual
GATHER_RESIDUAL = os.environ.get("VLLM_HCU_FUSE_A_GATHER_RESIDUAL", "0" if FUSE_B else "1") == "1"
FINAL = os.environ.get("VLLM_HCU_FUSE_A_FINAL", "1") == "1"

QUANT = int(os.environ.get("VLLM_HCU_FUSE_A_QUANT", "0"))

_comm = None  # 每个进程一个 RingComm，位置 B（fuse_b.py）共用
_op = None
_mod = None  # FlagTree kernel 模块，fuse_b 用它建 fused_qkv_norm_ag
_max_tokens = 0
_dtype = None
_moe = {}  # layer_idx -> DeepseekV2MoE，QUANT=2 时 op 里用它的 gate 和 router


def use_for_layer(vllm_config, config, layer_idx: int) -> bool:
    """第 layer_idx 层是否走位置 A 的融合（只做 MoE 层，dense 层不动）。"""
    if not ENABLED or vllm_config.parallel_config.enable_lightly_cp:
        return False
    if get_tensor_model_parallel_world_size() < 2:
        return False
    moe_layer_freq = getattr(config, "moe_layer_freq", 1)
    return (config.n_routed_experts is not None and layer_idx >= config.first_k_dense_replace
            and layer_idx % moe_layer_freq == 0)


def use_for_final(vllm_config, config, layer_idx: int) -> bool:
    """第 layer_idx 层是否是最后一层、且它的 MoE AR + final norm 走融合。"""
    return (FINAL and use_for_layer(vllm_config, config, layer_idx) and layer_idx == config.num_hidden_layers - 1
            and get_pp_group().world_size == 1)


def init(vllm_config, hidden_size: int, dtype: torch.dtype):
    """模型初始化时调用一次：在 vLLM 估算 KV cache 显存之前分配 RingComm 的共享 buffer。"""
    global _comm, _op, _mod, _max_tokens, _dtype
    if not ENABLED or _comm is not None:
        return
    import torch.distributed as dist

    tp = get_tensor_model_parallel_world_size()
    if dist.get_world_size() != tp:
        # RingComm 在默认进程组上建 IPC：只支持单机纯 TP
        raise RuntimeError(f"VLLM_HCU_FUSE_A needs world size == TP size, got {dist.get_world_size()} vs {tp}")
    spec = importlib.util.spec_from_file_location("flagtree_rs_norm_ag", KERNEL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    pb = mod.fused_rs_add_rms_norm_ag.parity_bytes(tp, max_tokens, hidden_size, dtype)
    if FUSE_B:  # B 的 q / kv norm + AG：行宽最多 q_lora + kv_lora + rope + indexer（head_dim + n_head）
        cfg = vllm_config.model_config.hf_config
        D = (cfg.q_lora_rank + cfg.kv_lora_rank + cfg.qk_rope_head_dim
             + getattr(cfg, "index_head_dim", 0) + getattr(cfg, "index_n_heads", 0))
        pb = max(pb, mod.fused_qkv_norm_ag.parity_bytes(tp, max_tokens, D, dtype))
    _comm = mod.RingComm(pb, peer="ipc", platform="hcu")
    _op = mod.fused_rs_add_rms_norm_ag(_comm, max_tokens, hidden_size, dtype)
    _mod, _max_tokens, _dtype = mod, max_tokens, dtype
    logger.info("fuse A: RingComm %.0f MiB, max_tokens=%d, native when T <= %d, gather residual=%s, final=%s, "
                "quant=%d, fuse B=%s", _comm.buf.numel() / 2**20, max_tokens, NATIVE_MAX_TOKENS, GATHER_RESIDUAL, FINAL,
                QUANT, FUSE_B)


def _native(x, residual, weight, eps):
    x = tensor_model_parallel_all_reduce(x)
    z = x.float() + residual.float()
    residual.copy_(z)
    return (z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


def _gather_residual(residual, b, e):
    # 本卡的有效行 AG 回完整 [T, H]；row_range 每卡 ceil(T/W) 行，末尾几卡可能更少，补齐后再 AG
    T, H = residual.shape
    W = _comm.world_size
    chunk = (T + W - 1) // W
    own = residual.new_zeros((chunk, H))
    own[:e - b] = residual[b:e]
    residual.copy_(tensor_model_parallel_all_gather(own, dim=0)[:T])


def _rs_norm_ag(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float,
                gather_residual: bool) -> torch.Tensor:
    """x：本卡的部分和 [T, H]（o_proj 或最后一层 MoE 的输出）；residual：[T, H]，原地更新；
    返回完整的 norm 输出 [T, H]。gather_residual=False：之后不用 residual（final norm），不做 AG。"""
    T = x.shape[0]
    if (_op is None or T <= NATIVE_MAX_TOKENS or torch.cuda.is_current_stream_capturing()
            or not residual.is_contiguous()):  # kernel 原地写 residual[b:e]，必须连续
        return _native(x, residual, weight, eps)
    b, e = _comm.row_range(T)
    out = torch.empty_like(x)
    dbg = _debug_calls()
    if dbg:
        _dbg(f"call T={T} rows=[{b},{e}) x={_desc(x)} residual={_desc(residual)} weight={_desc(weight)} "
             f"out={_desc(out)} buf=0x{_comm.buf.data_ptr():x}+{_comm.buf.numel()} peers={_comm.peers.tolist()}")
    _op(x.contiguous(), residual[b:e], weight, eps, out=out, gather=True)
    if dbg:
        torch.cuda.synchronize()
        _dbg(f"kernel ok, wait timeouts={_comm.errors()}")
    if GATHER_RESIDUAL and gather_residual:
        _gather_residual(residual, b, e)
        if dbg:
            torch.cuda.synchronize()
            _dbg("gather residual ok")
    return out


_DEBUG_LEFT = int(os.environ.get("VLLM_HCU_FUSE_A_DEBUG", "0"))  # 前 N 次 fused 调用打日志并同步


def _debug_calls():
    global _DEBUG_LEFT
    if _DEBUG_LEFT <= 0:
        return False
    _DEBUG_LEFT -= 1
    return True


def _desc(t):
    return f"{tuple(t.shape)}/{tuple(t.stride())}/{t.dtype}/0x{t.data_ptr():x}"


def _dbg(msg):
    logger.warning("fuse A debug rank%d: %s", _comm.rank, msg)


def _rs_norm_ag_fake(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float,
                     gather_residual: bool) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="hcu_fuse_a_rs_norm_ag",
    op_func=_rs_norm_ag,
    mutates_args=["residual"],
    fake_impl=_rs_norm_ag_fake,
)


# ---- QUANT=1 / 2：A 之后的量化交给 A，MoE 收量化好的输入 ----


def _quant(x):
    # routed（lightop fp8_marlin）自己量化时用的就是它；shared（vLLM QuantFP8）的 scale 下限是
    # 1/(448*512)，只在 amax < 0.002 时和这里不同
    from lightop._lmslim_native.layers.gemm.fp8_utils import per_token_quant_fp8
    return per_token_quant_fp8(x)


def _fused(T, residual):
    return not (_op is None or T <= NATIVE_MAX_TOKENS or torch.cuda.is_current_stream_capturing()
                or not residual.is_contiguous())


def _rs_norm_ag_quant(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                      eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """QUANT=1：返回完整的 norm 输出 bf16 [T, H]（给 router）和它的 fp8 q [T, H]、scale [T, 1]。"""
    out = _rs_norm_ag(x, residual, weight, eps, True)
    q, s = _quant(out)
    return out, q, s


def _rs_norm_ag_quant_fake(x, residual, weight, eps):
    T, H = x.shape
    return (torch.empty_like(x), x.new_empty((T, H), dtype=torch.float8_e4m3fn),
            x.new_empty((T, 1), dtype=torch.float32))


direct_register_custom_op(
    op_name="hcu_fuse_a_rs_norm_ag_quant",
    op_func=_rs_norm_ag_quant,
    mutates_args=["residual"],
    fake_impl=_rs_norm_ag_quant_fake,
)


def register_moe(layer_idx: int, moe):
    _moe[layer_idx] = moe


def _route(moe, y):
    logits, _ = moe.gate(y)
    return moe.experts.runner.router.select_experts(hidden_states=y, router_logits=logits)


def _gather_rows(ts, T):
    """本卡的几组行（行数相同，ceil(T/W) 以内）按字节拼成一份，做一次 AG，拆回完整 T 行。"""
    W = _comm.world_size
    chunk = (T + W - 1) // W
    rows = ts[0].shape[0]
    raw = [t.contiguous().view(torch.uint8).view(rows, -1) for t in ts]
    widths = [r.shape[1] for r in raw]
    own = raw[0].new_zeros((chunk, sum(widths)))
    if rows:
        own[:rows] = torch.cat(raw, dim=1)
    full = tensor_model_parallel_all_gather(own, dim=0)[:T]
    out, c = [], 0
    for t, w in zip(ts, widths):
        out.append(full[:, c:c + w].contiguous().view(t.dtype).view(T, *t.shape[1:]))
        c += w
    return out


def _rs_norm_quant_route(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float,
                         layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """QUANT=2：返回完整 T 行的 q [T, H] fp8、scale [T, 1]、topk_weights、topk_ids。
    融合路径上 gate + topk 只在本卡 T/8 行上算，bf16 不出本卡。"""
    moe = _moe[layer_idx]
    T = x.shape[0]
    if not _fused(T, residual):
        y = _native(x, residual, weight, eps)
        q, s = _quant(y)
        tw, ti = _route(moe, y)
        return q, s, tw, ti
    b, e = _comm.row_range(T)
    y, q, s = _op(x.contiguous(), residual[b:e], weight, eps, out=x.new_empty((e - b, x.shape[1])),
                  quant=_op.QUANT_FP8_AG)
    tw, ti = _route(moe, y)
    tw, ti = _gather_rows([tw, ti], T)
    if GATHER_RESIDUAL:
        _gather_residual(residual, b, e)
    return q, s, tw, ti


def _rs_norm_quant_route_fake(x, residual, weight, eps, layer_idx):
    T, H = x.shape
    k = _moe[layer_idx].experts.top_k if layer_idx in _moe else 8
    return (x.new_empty((T, H), dtype=torch.float8_e4m3fn), x.new_empty((T, 1), dtype=torch.float32),
            x.new_empty((T, k), dtype=torch.float32), x.new_empty((T, k), dtype=torch.int32))


direct_register_custom_op(
    op_name="hcu_fuse_a_rs_norm_quant_route",
    op_func=_rs_norm_quant_route,
    mutates_args=["residual"],
    fake_impl=_rs_norm_quant_route_fake,
)


def post_attention(layer, hidden_states, residual):
    """DecoderLayer 里融合层的 post_attention_layernorm + mlp。"""
    norm = layer.post_attention_layernorm
    if QUANT == 1:
        y, q, s = torch.ops.vllm.hcu_fuse_a_rs_norm_ag_quant(hidden_states, residual, norm.weight,
                                                               norm.variance_epsilon)
        return layer.mlp(y, quanted=(q, s))
    if QUANT == 2:
        q, s, tw, ti = torch.ops.vllm.hcu_fuse_a_rs_norm_quant_route(hidden_states, residual, norm.weight,
                                                                       norm.variance_epsilon, layer.layer_idx)
        # routed 的 fp8_marlin 是 inplace：输出写回 hidden，所以要一块可写的 [T, H] bf16，内容不读
        return layer.mlp(torch.empty_like(hidden_states), quanted=(q, s), topk=(tw, ti))
    y = torch.ops.vllm.hcu_fuse_a_rs_norm_ag(hidden_states, residual, norm.weight, norm.variance_epsilon, True)
    return layer.mlp(y)


def moe_gate(vllm_config, config, layer_idx: int, gate):
    """QUANT=2 时融合层的 FusedMoE 不持有 gate（否则 runner 总会在传入的 hidden 上跑一遍 gate）。"""
    return None if QUANT == 2 and use_for_layer(vllm_config, config, layer_idx) else gate


def moe_experts_kwargs(quanted, topk):
    """DeepseekV2MoE.forward 里传给 self.experts 的量化输入 / 路由结果。"""
    kw = {}
    if quanted is not None:
        kw.update(quanted_hidden_states=quanted[0], scale=quanted[1])
    if topk is not None:
        kw.update(topk_weights=topk[0], topk_ids=topk[1])
    return kw
