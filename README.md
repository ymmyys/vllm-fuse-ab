# vllm-fuse-ab

Changes to vendor vLLM 0.21 + vllm_hcu (`0.21.0+das.dtk2604.torch2100.2608251643.g135f01`) that replace the two TP all-reduces of every GLM-5.2 MoE layer (A: after o_proj; B: after the MoE, up to the next layer's q_b) with FlagTree rs+norm+ag ring kernels.

- `vllm_hcu/fuse_a.py`: position A
- `vllm_hcu/fuse_b.py`: position B
- `vllm_hcu/models/deepseek_v2.py.diff`: changes to `vllm_hcu/models/deepseek_v2.py`

Install into the installed vllm_hcu:

```bash
SITE=$(python3 -c "import os, vllm_hcu; print(os.path.dirname(os.path.dirname(vllm_hcu.__file__)))")
cp vllm_hcu/fuse_a.py vllm_hcu/fuse_b.py $SITE/vllm_hcu/
patch -p1 -d $SITE < vllm_hcu/models/deepseek_v2.py.diff
```

Switches (all unset = vendor behavior):

| Variable | Meaning |
|---|---|
| `VLLM_HCU_FUSE_A=1` | enable A |
| `VLLM_HCU_FUSE_A_QUANT=2` | A all-gathers fp8 + scale, router on this rank's rows (0 = bf16 all-gather) |
| `VLLM_HCU_FUSE_B=1` | enable B |
| `VLLM_HCU_FUSE_A_KERNEL` | path of FlagTree `10-intra-node-rs-norm-ag.py` |
