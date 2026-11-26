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
        tmp = f"{gen}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            f.write(src)
        os.replace(tmp, gen)
        return importlib.util.spec_from_file_location(name, gen)


def register():
    if os.environ.get("VLLM_HCU_FUSE_A", "0") != "1":
        return
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    if TARGET in sys.modules:
        importlib.reload(sys.modules[TARGET])
