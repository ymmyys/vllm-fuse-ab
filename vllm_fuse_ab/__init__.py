# vLLM general plugin：在厂商 vLLM 0.21 + vllm_hcu 上接入位置 A / B 的融合（FlagTree rs+norm+ag ring kernel）。
#
# 不改镜像里的文件：vLLM 在每个进程加载模型之前先调 register()，这里给 vllm_hcu.models.deepseek_v2
# 装一个 import 钩子，按 edits.EDITS 改它的源码，把改好的源码写到本包的 _generated/ 下再从那里加载
# （文件真实存在，inspect / torch.compile 的缓存 hash 读到的就是改后的源码）。
#
# 开关（同 fuse_a.py / fuse_b.py）：VLLM_HCU_FUSE_A=1 才装钩子；VLLM_HCU_FUSE_A_QUANT、VLLM_HCU_FUSE_B 等见那两个文件。
# 开关不设时 register() 什么都不做，走的就是厂商原来的代码。
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

TARGET = "vllm_hcu.models.deepseek_v2"
_GEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_generated")


def _patched_source(src: str) -> str:
    from vllm_fuse_ab.edits import EDITS

    for i, (old, new, *count) in enumerate(EDITS, 1):
        want = count[0] if count else 1
        n = src.count(old)
        if n != want:
            raise RuntimeError(f"vllm_fuse_ab: edit {i} anchor found {n} times in {TARGET}, expected {want}")
        src = src.replace(old, new)
    return src


class _Finder(importlib.abc.MetaPathFinder):

    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        orig = importlib.machinery.PathFinder.find_spec(name, path)
        if orig is None or not orig.origin:
            return None
        with open(orig.origin) as f:
            src = _patched_source(f.read())
        os.makedirs(_GEN_DIR, exist_ok=True)
        gen = os.path.join(_GEN_DIR, "deepseek_v2.py")
        tmp = f"{gen}.{os.getpid()}.tmp"  # 多个进程同时写：先写临时文件再原子替换
        with open(tmp, "w") as f:
            f.write(src)
        os.replace(tmp, gen)
        return importlib.util.spec_from_file_location(name, gen)


def register():
    if os.environ.get("VLLM_HCU_FUSE_A", "0") != "1":
        return
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    if TARGET in sys.modules:  # 已经被别的插件提前 import 了：按改后的源码重新加载
        importlib.reload(sys.modules[TARGET])
