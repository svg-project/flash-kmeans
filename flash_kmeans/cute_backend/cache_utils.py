"""Persistent JIT compilation cache for the CuTe DSL kernels.

Adapted from FlashAttention-4's flash_attn/cute/cache_utils.py (BSD-3-Clause,
Dao-AILab): compiled kernels are exported to object files via the DSL's AOT
support (tvm-ffi + export_to_c) and loaded back in fresh processes with
cute.runtime.load_module, so a kernel for a given problem geometry is only
compiled once across runs instead of once per process.

Layout on disk:
    <cache_dir>/<source_fingerprint>/lloyd/<key_sha256>.o

- <source_fingerprint> hashes every cute_backend/*.py source plus the
  python/cutlass/tvm-ffi/torch/CUDA/quack versions, so any code or dependency
  change automatically invalidates all previously exported kernels.
- <key_sha256> hashes the per-kernel compile key (arch, geometry, K,
  fuse_sums, topj, num_sms, kernel name) — see lloyd_smXX._get_compiled.

Env vars:
    FLASH_KMEANS_CUTE_DSL_CACHE_ENABLED   "1" (default) enables the persistent
                                          cache; "0" falls back to in-memory.
    FLASH_KMEANS_CUTE_DSL_CACHE_DIR       override the cache root
                                          (default /tmp/<user>/flash_kmeans_cute_dsl_cache).
    FLASH_KMEANS_CUTE_DSL_CACHE_VERBOSE   "1" prints cache events to stderr.

Concurrent processes are serialized per artifact with flock(2); exports are
staged and atomically renamed so readers never observe a partial object file.
If apache-tvm-ffi is unavailable or an export/load fails, the cache degrades
to in-memory (per-process) behavior rather than failing the caller.
"""
import fcntl
import hashlib
import os
import pickle
import sys
import tempfile
import time
from functools import lru_cache
from getpass import getuser
from pathlib import Path
from typing import Hashable, TypeAlias

import ctypes

import cutlass
import cutlass.cute as cute

# Pre-load cute DSL runtime libraries with RTLD_GLOBAL so their symbols
# (e.g. _cudaLibraryLoadData) are visible to .o modules later loaded via
# dlopen (mirrors FA4; upstream cute.runtime.load_module loads them without
# RTLD_GLOBAL, which breaks loading cached kernels from disk in some setups).
for _lib_path in cute.runtime.find_runtime_libraries(enable_tvm_ffi=False):
    if Path(_lib_path).exists():
        ctypes.CDLL(_lib_path, mode=ctypes.RTLD_GLOBAL)

try:
    import tvm_ffi
    _TVM_FFI_AVAILABLE = True
except ImportError:
    tvm_ffi = None
    _TVM_FFI_AVAILABLE = False

CompileKeyType: TypeAlias = tuple[Hashable, ...]
# TVMFFIJitCompiledFunction (fresh compile) or tvm_ffi.Function (disk load)
CompiledFnType: TypeAlias = object

_ENABLED = os.getenv("FLASH_KMEANS_CUTE_DSL_CACHE_ENABLED", "1") == "1"
_CACHE_DIR = os.getenv("FLASH_KMEANS_CUTE_DSL_CACHE_DIR", None)


def _parse_verbose(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 1 if value.strip().lower() in ("true", "yes", "on") else 0


_VERBOSE = _parse_verbose(os.getenv("FLASH_KMEANS_CUTE_DSL_CACHE_VERBOSE", "0"))

EXPORT_FUNCTION_PREFIX = "func"


def tvm_ffi_available() -> bool:
    """tvm-ffi is required for both exporting to and loading from disk."""
    return _TVM_FFI_AVAILABLE


def cache_enabled() -> bool:
    return _ENABLED and _TVM_FFI_AVAILABLE


def fk_log(level: int, msg: str) -> None:
    if _VERBOSE >= level:
        print(f"[flash-kmeans-cute cache] {msg}", file=sys.stderr)


def get_cache_path() -> Path:
    if _CACHE_DIR is not None:
        cache_dir = Path(_CACHE_DIR)
    else:
        # getuser() consults the password db and raises in passwd-less
        # containers; fall back to the numeric uid.
        try:
            user = getuser()
        except Exception:
            user = str(os.getuid())
        cache_dir = Path(tempfile.gettempdir()) / user / "flash_kmeans_cute_dsl_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@lru_cache(maxsize=1)
def _compute_source_fingerprint() -> str:
    """Hash package sources + dependency ABI stamps into one fingerprint.

    Changes whenever any cute_backend/*.py file is added/removed/renamed/
    modified, or the python, cutlass, tvm-ffi, torch, CUDA or quack version
    changes (the kernels inline quack helpers, so its version matters too).
    """
    pkg_root = Path(__file__).resolve().parent
    h = hashlib.sha256()

    h.update(f"py{sys.version_info.major}.{sys.version_info.minor}".encode())
    h.update(f"cutlass={cutlass.__version__}".encode())
    h.update(f"tvm_ffi={tvm_ffi.__version__ if tvm_ffi else 'none'}".encode())

    import torch
    h.update(f"torch={torch.__version__}".encode())
    h.update(f"cuda={torch.version.cuda}".encode())
    # The kernels inline quack helpers at trace time, so quack affects the
    # compiled code. Stamp it WITHOUT importing it (its __init__ runs DSL code
    # that raises on quack/DSL mismatches — a broken quack must not be able to
    # break the cache): dist version + the package's own *.py contents, so
    # editable/patched installs invalidate too (bounded so a pathological
    # site-packages can never make this O(worst-case)).
    import importlib.util
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
    spec = importlib.util.find_spec("quack")
    qroot = (Path(spec.submodule_search_locations[0])
             if spec is not None and spec.submodule_search_locations else None)
    try:
        h.update(f"quack={_pkg_version('quack-kernels')}".encode())
    except PackageNotFoundError:
        h.update(b"quack=none")
    if qroot is not None and qroot.is_dir():
        qfiles = sorted(qroot.rglob("*.py"))[:512]
        for src in qfiles:
            if not src.is_file():
                continue
            h.update(src.relative_to(qroot).as_posix().encode())
            content = src.read_bytes()
            h.update(len(content).to_bytes(8, "little"))
            h.update(content)

    for src in sorted(pkg_root.rglob("*.py")):
        if not src.is_file():
            continue
        h.update(src.relative_to(pkg_root).as_posix().encode())
        content = src.read_bytes()
        h.update(len(content).to_bytes(8, "little"))
        h.update(content)

    return h.hexdigest()


class FileLock:
    """fcntl.flock advisory lock (shared or exclusive), blocking with polling."""

    def __init__(self, lock_path: Path, exclusive: bool, timeout: float = 15,
                 label: str = ""):
        self.lock_path = lock_path
        self.exclusive = exclusive
        self.timeout = timeout
        self.label = label
        self._fd = -1

    @property
    def _lock_label(self) -> str:
        kind = "exclusive" if self.exclusive else "shared"
        return f"{kind} {self.label}" if self.label else kind

    def __enter__(self) -> "FileLock":
        open_flags = os.O_WRONLY | os.O_CREAT if self.exclusive else os.O_RDONLY | os.O_CREAT
        lock_type = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH
        self._fd = os.open(str(self.lock_path), open_flags)
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fcntl.flock(self._fd, lock_type | fcntl.LOCK_NB)
                return self
            except OSError:
                if time.monotonic() >= deadline:
                    os.close(self._fd)
                    self._fd = -1
                    raise RuntimeError(
                        f"Timed out after {self.timeout}s waiting for "
                        f"{self._lock_label} lock: {self.lock_path}"
                    )
                time.sleep(0.1)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = -1


class JITCache:
    """In-memory cache for compiled kernels (per-process)."""

    def __init__(self):
        self.cache: dict[CompileKeyType, CompiledFnType] = {}

    def __setitem__(self, key: CompileKeyType, fn: CompiledFnType) -> None:
        self.cache[key] = fn

    def __getitem__(self, key: CompileKeyType) -> CompiledFnType:
        return self.cache[key]

    def __contains__(self, key: CompileKeyType) -> bool:
        return key in self.cache

    def clear(self) -> None:
        self.cache.clear()


class JITPersistentCache(JITCache):
    """In-memory cache backed by on-disk AOT object files (tvm-ffi only)."""

    LOCK_TIMEOUT_SECONDS = 15

    def __init__(self, cache_path: Path):
        super().__init__()
        cache_path.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_path

    def __setitem__(self, key: CompileKeyType, fn: CompiledFnType) -> None:
        JITCache.__setitem__(self, key, fn)
        try:
            self._try_export_to_storage(key, fn)
        except Exception as e:
            # The in-memory entry is already set; a failed export only means
            # the next process will recompile. Never fail the caller.
            fk_log(1, f"Export failed (in-process cache still valid): {e!r}")

    def __getitem__(self, key: CompileKeyType) -> CompiledFnType:
        # __contains__ populates the in-memory cache from disk if needed.
        self.__contains__(key)
        return JITCache.__getitem__(self, key)

    def __contains__(self, key: CompileKeyType) -> bool:
        # True guarantees the in-memory cache holds the entry afterwards.
        if JITCache.__contains__(self, key):
            return True
        return self._try_load_from_storage(key)

    def _try_load_from_storage(self, key: CompileKeyType) -> bool:
        sha256_hex = self._key_to_hash(key)
        obj_path = self.cache_path / f"{sha256_hex}.o"
        corrupt = False
        try:
            with FileLock(self._lock_path(sha256_hex), exclusive=False,
                          timeout=self.LOCK_TIMEOUT_SECONDS, label=sha256_hex):
                if not obj_path.exists():
                    fk_log(1, f"Disk cache miss: {sha256_hex}")
                    return False
                try:
                    m = cute.runtime.load_module(str(obj_path), enable_tvm_ffi=True)
                    fn = getattr(m, EXPORT_FUNCTION_PREFIX)
                    fk_log(1, f"Loaded compiled kernel from disk: {obj_path}")
                except Exception as e:
                    fk_log(1, f"Failed to load {obj_path} ({e!r}); will recompile")
                    corrupt = True
                else:
                    JITCache.__setitem__(self, key, fn)
                    return True
        except Exception as e:
            # Lock acquisition itself failing (stuck writer, EACCES, full
            # disk on lock-file creation) must degrade to an in-process
            # recompile, never fail the caller.
            fk_log(1, f"Disk cache read failed for {sha256_hex} ({e!r}); "
                      "will recompile")
            return False
        # Outside the shared lock before taking the exclusive one: flock on a
        # different fd from the same process conflicts with our own shared
        # lock and would just time out (self-deadlock).
        if corrupt:
            self._unlink_corrupt(sha256_hex)
        return False

    def _unlink_corrupt(self, sha256_hex: str) -> None:
        """Best-effort delete of an unloadable artifact so the recompiled
        replacement can be exported (otherwise the bad file wedges the key:
        export skips it because the path exists)."""
        obj_path = self.cache_path / f"{sha256_hex}.o"
        try:
            with FileLock(self._lock_path(sha256_hex), exclusive=True,
                          timeout=self.LOCK_TIMEOUT_SECONDS, label=sha256_hex):
                if obj_path.exists():
                    fk_log(1, f"Removing unloadable artifact: {obj_path}")
                    obj_path.unlink()
        except Exception:
            pass  # recompile still works; the wedge only costs export-side


    def _try_export_to_storage(self, key: CompileKeyType, fn: CompiledFnType) -> None:
        sha256_hex = self._key_to_hash(key)
        obj_path = self.cache_path / f"{sha256_hex}.o"
        with FileLock(self._lock_path(sha256_hex), exclusive=True,
                      timeout=self.LOCK_TIMEOUT_SECONDS, label=sha256_hex):
            if obj_path.exists():
                fk_log(1, f"Skipping export, already on disk: {obj_path}")
                return
            # Export into a per-process staging dir, then atomically rename,
            # so concurrent readers never see a partially written object.
            staging = self.cache_path / f".staging-{sha256_hex}-{os.getpid()}"
            staging.mkdir(parents=True, exist_ok=True)
            try:
                fk_log(1, f"Exporting compiled kernel to disk: {obj_path}")
                fn.export_to_c(object_file_path=str(staging / f"{sha256_hex}.o"),
                               function_name=EXPORT_FUNCTION_PREFIX)
                os.replace(staging / f"{sha256_hex}.o", obj_path)
                fk_log(1, f"Exported compiled kernel to disk: {obj_path}")
            finally:
                try:
                    (staging / f"{sha256_hex}.o").unlink(missing_ok=True)
                    staging.rmdir()
                except OSError:
                    pass

    def _key_to_hash(self, key: CompileKeyType) -> str:
        return hashlib.sha256(pickle.dumps(key)).hexdigest()

    def _lock_path(self, sha256_hex: str) -> Path:
        return self.cache_path / f"{sha256_hex}.lock"

    def clear(self) -> None:
        """Clear the in-memory cache AND purge the persistent cache."""
        fk_log(1, f"Clearing persistent cache at {self.cache_path}")
        super().clear()
        for child in self.cache_path.iterdir():
            if child.is_file():
                child.unlink()
            elif child.name.startswith(".staging-"):
                import shutil
                shutil.rmtree(child, ignore_errors=True)


_CACHES: dict[str | None, JITCache] = {}


def get_jit_cache(name: str | None = None) -> JITCache:
    """JIT cache factory (one shared instance per `name`).

    `name` groups artifacts into a subdirectory of the fingerprint dir.
    Persistent caching namespaces artifacts under the source fingerprint so
    code or dependency changes automatically invalidate stale entries.
    Falls back to a plain in-memory cache on ANY setup failure (read-only or
    full filesystem, sandboxing, ...) so the caller never breaks.
    """
    if name in _CACHES:
        return _CACHES[name]
    cache: JITCache | None = None
    if cache_enabled():
        try:
            path = get_cache_path() / _compute_source_fingerprint()
            if name:
                path = path / name
            fk_log(1, f"Persistent JIT cache at {path}")
            cache = JITPersistentCache(path)
        except Exception as e:
            fk_log(1, f"Persistent cache setup failed ({e!r}); "
                      "using in-memory JIT cache")
    else:
        if _ENABLED and not _TVM_FFI_AVAILABLE:
            fk_log(1, "apache-tvm-ffi not installed; persistent cache disabled, "
                      "using in-memory JIT cache")
        else:
            fk_log(1, "Persistent cache disabled, using in-memory JIT cache")
    if cache is None:
        cache = JITCache()
    _CACHES[name] = cache
    return cache
