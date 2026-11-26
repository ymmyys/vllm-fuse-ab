# vllm-fuse-ab

厂商 vLLM 0.21 + vllm_hcu 的 general plugin：把 GLM-5.2 每层两次 TP AllReduce（位置 A：o_proj 之后；位置 B：MoE 之后到下一层 q_b 之前）换成 FlagTree 的 rs+norm+ag ring kernel。不改镜像里的文件，运行时按 `vllm_fuse_ab/edits.py` 改 `vllm_hcu.models.deepseek_v2` 的源码再加载。

安装：

```bash
pip install --no-deps --no-build-isolation /path/to/vllm-fuse-plugin
```

开关（不设时插件什么都不做）：

| 环境变量 | 含义 |
|---|---|
| `VLLM_HCU_FUSE_A=1` | 打开位置 A（也是插件的总开关） |
| `VLLM_HCU_FUSE_A_QUANT=2` | A 的 AG 传 fp8，router 在本卡 1/8 行上算（0 = AG bf16） |
| `VLLM_HCU_FUSE_B=1` | 打开位置 B |
| `VLLM_HCU_FUSE_A_KERNEL` | FlagTree `10-intra-node-rs-norm-ag.py` 的路径 |

其余开关见 `vllm_fuse_ab/fuse_a.py`、`fuse_b.py` 开头的注释。
