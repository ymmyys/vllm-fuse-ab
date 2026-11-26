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
FUSE_B = os.environ.get("VLLM_HCU_FUSE_B", "0") == "1"
GATHER_RESIDUAL = os.environ.get("VLLM_HCU_FUSE_A_GATHER_RESIDUAL", "0" if FUSE_B else "1") == "1"
FINAL = os.environ.get("VLLM_HCU_FUSE_A_FINAL", "1") == "1"

QUANT = int(os.environ.get("VLLM_HCU_FUSE_A_QUANT", "0"))

_comm = None
_op = None
_mod = None
_max_tokens = 0
_dtype = None
_moe = {}


def use_for_layer(vllm_config, config, layer_idx: int) -> bool:
    if not ENABLED or vllm_config.parallel_config.enable_lightly_cp:
        return False
    if get_tensor_model_parallel_world_size() < 2:
        return False
    moe_layer_freq = getattr(config, "moe_layer_freq", 1)
    return (config.n_routed_experts is not None and layer_idx >= config.first_k_dense_replace
            and layer_idx % moe_layer_freq == 0)


def use_for_final(vllm_config, config, layer_idx: int) -> bool:
    return (FINAL and use_for_layer(vllm_config, config, layer_idx) and layer_idx == config.num_hidden_layers - 1
            and get_pp_group().world_size == 1)


def init(vllm_config, hidden_size: int, dtype: torch.dtype):
    global _comm, _op, _mod, _max_tokens, _dtype
    if not ENABLED or _comm is not None:
        return
    import torch.distributed as dist

    tp = get_tensor_model_parallel_world_size()
    if dist.get_world_size() != tp:
        raise RuntimeError(f"VLLM_HCU_FUSE_A needs world size == TP size, got {dist.get_world_size()} vs {tp}")
    spec = importlib.util.spec_from_file_location("flagtree_rs_norm_ag", KERNEL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    pb = mod.fused_rs_add_rms_norm_ag.parity_bytes(tp, max_tokens, hidden_size, dtype)
    if FUSE_B:
        cfg = vllm_config.model_config.hf_config
        D = (cfg.q_lora_rank + cfg.kv_lora_rank + cfg.qk_rope_head_dim
             + getattr(cfg, "index_head_dim", 0) + getattr(cfg, "index_n_heads", 0))
        pb = max(pb, mod.fused_qkv_norm_ag.parity_bytes(tp, max_tokens, D, dtype))
    _comm = mod.RingComm(pb, peer="ipc", platform="hcu", cuda_graph=not vllm_config.model_config.enforce_eager)
    _op = mod.fused_rs_add_rms_norm_ag(_comm, max_tokens, hidden_size, dtype)
    _mod, _max_tokens, _dtype = mod, max_tokens, dtype
    logger.info("fuse A: RingComm %.0f MiB, max_tokens=%d, cuda_graph=%s, gather residual=%s, final=%s, "
                "quant=%d, fuse B=%s", _comm.buf.numel() / 2**20, max_tokens, _comm.cuda_graph, GATHER_RESIDUAL, FINAL,
                QUANT, FUSE_B)


def _native(x, residual, weight, eps):
    x = tensor_model_parallel_all_reduce(x)
    z = x.float() + residual.float()
    residual.copy_(z)
    return (z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


def _gather_residual(residual, b, e):
    T, H = residual.shape
    W = _comm.world_size
    chunk = (T + W - 1) // W
    own = residual.new_zeros((chunk, H))
    own[:e - b] = residual[b:e]
    residual.copy_(tensor_model_parallel_all_gather(own, dim=0)[:T])


def _rs_norm_ag(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float,
                gather_residual: bool) -> torch.Tensor:
    T = x.shape[0]
    if _op is None:
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


_DEBUG_LEFT = int(os.environ.get("VLLM_HCU_FUSE_A_DEBUG", "0"))


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


def _quant(x):
    from lightop._lmslim_native.layers.gemm.fp8_utils import per_token_quant_fp8
    return per_token_quant_fp8(x)


def _fused(T, residual):
    return _op is not None


def _rs_norm_ag_quant(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                      eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    if y.shape[0] == 0:  # T < W: trailing ranks own no rows
        k = moe.experts.top_k
        return y.new_empty((0, k), dtype=torch.float32), y.new_empty((0, k), dtype=torch.int32)
    logits, _ = moe.gate(y)
    return moe.experts.runner.router.select_experts(hidden_states=y, router_logits=logits)


def _gather_rows(ts, T):
    W = _comm.world_size
    chunk = (T + W - 1) // W
    rows = ts[0].shape[0]
    raw = [t.contiguous().view(torch.uint8).view(rows, t.shape[-1] * t.element_size()) for t in ts]
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
    norm = layer.post_attention_layernorm
    if QUANT == 1:
        y, q, s = torch.ops.vllm.hcu_fuse_a_rs_norm_ag_quant(hidden_states, residual, norm.weight,
                                                               norm.variance_epsilon)
        return layer.mlp(y, quanted=(q, s))
    if QUANT == 2:
        q, s, tw, ti = torch.ops.vllm.hcu_fuse_a_rs_norm_quant_route(hidden_states, residual, norm.weight,
                                                                       norm.variance_epsilon, layer.layer_idx)
        return layer.mlp(torch.empty_like(hidden_states), quanted=(q, s), topk=(tw, ti))
    y = torch.ops.vllm.hcu_fuse_a_rs_norm_ag(hidden_states, residual, norm.weight, norm.variance_epsilon, True)
    return layer.mlp(y)


def moe_gate(vllm_config, config, layer_idx: int, gate):
    return None if QUANT == 2 and use_for_layer(vllm_config, config, layer_idx) else gate


def moe_experts_kwargs(quanted, topk):
    kw = {}
    if quanted is not None:
        kw.update(quanted_hidden_states=quanted[0], scale=quanted[1])
    if topk is not None:
        kw.update(topk_weights=topk[0], topk_ids=topk[1])
    return kw
