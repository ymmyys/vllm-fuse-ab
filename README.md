# vllm-fuse-ab

A vLLM general plugin for vendor vLLM 0.21 + vllm_hcu. It replaces the two TP all-reduces of every GLM-5.2 MoE layer (A: after o_proj; B: after the MoE, up to the next layer's q_b) with FlagTree rs+norm+ag ring kernels. No file in the image is modified: at load time the plugin applies `vllm_fuse_ab/edits.py` to `vllm_hcu.models.deepseek_v2` and imports the result.

Install:

```bash
pip install --no-deps --no-build-isolation /path/to/vllm-fuse-ab
```

Switches (the plugin does nothing when unset):

| Variable | Meaning |
|---|---|
| `VLLM_HCU_FUSE_A=1` | enable A (also the plugin's master switch) |
| `VLLM_HCU_FUSE_A_QUANT=2` | A all-gathers fp8 + scale, router on this rank's rows (0 = bf16 all-gather) |
| `VLLM_HCU_FUSE_B=1` | enable B |
| `VLLM_HCU_FUSE_A_KERNEL` | path of FlagTree `10-intra-node-rs-norm-ag.py` |
