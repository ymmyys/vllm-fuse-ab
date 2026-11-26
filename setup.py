from setuptools import setup

setup(
    name="vllm-fuse-ab",
    version="0.1.0",
    description="GLM-5.2 TP prefill: fuse the two TP all-reduces per layer (A, B) with FlagTree rs+norm+ag "
    "ring kernels on vendor vLLM 0.21 + vllm_hcu",
    packages=["vllm_fuse_ab"],
    python_requires=">=3.10",
    entry_points={"vllm.general_plugins": ["fuse_ab = vllm_fuse_ab:register"]},
)
