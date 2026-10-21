"""CPU batch accumulator. CDLL releases GIL; Python row packing still holds it.

Each append owns a new contiguous output slab. Returned tensor views retain that
slab, so later appends, slot reuse, or builder deletion cannot overwrite payloads.
No CUDA operations or synchronization. The C++ library is compiled once into a
user-owned cache, or prebuilt with ``python -m easymagpie_vllm_omni.bulk_native``.
"""

import ctypes as C
import fcntl
import hashlib
import os
import platform
import shlex
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from .chunk_schedule import startup_schedule

HAS_FRAME, SKIP_ZERO, TERMINAL, SEGMENT, ABORT = 1, 2, 4, 8, 16
KINDS = {1: "chunk", 2: "empty", 3: "marker", 4: "stale", 5: "aborted"}


@dataclass
class ChunkEvent:
    index: int
    kind: str
    codes: object
    frames: int
    seen: int
    emitted: int
    chunks: int


def build_native_library(cache_dir=None):
    """Build the CPU chunk accumulator without modifying the installed package.

    ``CXX`` may contain a compiler and flags. Commands use an argument vector,
    never a shell. The source, compiler selection and platform identify a cache
    entry; an interprocess lock and atomic rename hide incomplete builds.
    """
    source = Path(__file__).with_name("chunk_builder.cpp")
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or shutil.which(compiler[0]) is None:
        raise RuntimeError("the native chunk builder requires a C++ compiler; set CXX")
    compiler[0] = str(Path(shutil.which(compiler[0])).resolve())
    compiler_stat = Path(compiler[0]).stat()
    flags = ["-std=c++17", "-O3", "-fPIC", "-shared"]
    identity = repr(
        (compiler, flags, compiler_stat.st_size, compiler_stat.st_mtime_ns, platform.system(), platform.machine())
    ).encode()
    digest = hashlib.sha256(source.read_bytes() + identity).hexdigest()
    if cache_dir is None:
        cache_dir = os.environ.get("EASYMAGPIE_NATIVE_CACHE_DIR")
    if cache_dir is None:
        cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "easymagpie" / "native"
    directory = Path(cache_dir).expanduser() / digest
    directory.mkdir(parents=True, exist_ok=True)
    library = directory / "libchunk_builder.so"
    with (directory / "build.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.is_file():
            with tempfile.TemporaryDirectory(prefix="build-", dir=directory) as temporary:
                output = Path(temporary) / library.name
                command = [*compiler, *flags, str(source), "-o", str(output)]
                try:
                    result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
                except (OSError, subprocess.TimeoutExpired) as error:
                    raise RuntimeError("could not compile the native chunk builder") from error
                if result.returncode != 0:
                    raise RuntimeError(f"native chunk builder compilation failed:\n{result.stderr[-8000:]}")
                if not output.is_file() or not output.stat().st_size:
                    raise RuntimeError("native chunk builder compilation produced no library")
                output.replace(library)
    return library


def _load():
    lib = C.CDLL(str(build_native_library()))
    lib.cb_create.argtypes = [C.c_uint32]
    lib.cb_create.restype = C.c_void_p
    lib.cb_create_configured.argtypes = [C.c_uint32, C.c_uint32, C.c_void_p]
    lib.cb_create_configured.restype = C.c_void_p
    lib.cb_destroy.argtypes = [C.c_void_p]
    lib.cb_destroy.restype = None
    lib.cb_reset_batch.argtypes = [C.c_void_p, C.c_uint32, C.c_void_p, C.c_void_p, C.c_void_p]
    lib.cb_reset_batch.restype = C.c_int
    lib.cb_release.argtypes = [C.c_void_p, C.c_uint32, C.c_uint64]
    lib.cb_release.restype = C.c_int
    lib.cb_release_batch.argtypes = [C.c_void_p, C.c_uint32, C.c_void_p, C.c_void_p]
    lib.cb_release_batch.restype = C.c_int
    lib.cb_append.argtypes = [C.c_void_p, C.c_uint32] + [C.c_void_p] * 9
    lib.cb_append.restype = C.c_int
    lib.cb_export.argtypes = [C.c_void_p, C.c_uint32, C.c_void_p, C.c_void_p]
    lib.cb_export.restype = C.c_int
    return lib


class BulkChunkAccumulator:
    """Single logical owner, with locking against accidental concurrent calls."""

    def __init__(self, capacity=2048, width=16, startup_frames=(2, 6)):
        if width != 16 or not 0 < capacity <= 1048576:
            raise ValueError("native builder requires width=16 and positive bounded capacity")
        self.capacity, self.width = int(capacity), int(width)
        self.startup_frames = startup_schedule(startup_frames)
        self.frame_viewless = os.environ.get('EASYMAGPIE_BULK_FRAME_VIEWLESS', '0') == '1'
        self.viewless_batch_calls = 0
        self.viewless_rows = 0
        self._lib = _load()
        schedule = (C.c_uint8 * len(self.startup_frames))(*self.startup_frames)
        self._ptr = self._lib.cb_create_configured(self.capacity, len(schedule), schedule)
        if not self._ptr:
            raise MemoryError("cannot allocate native lane storage")
        self._lock = threading.RLock()
        self._pending_resets = {}
        self._pending_releases = {}
        self._last_generation = {}

    def close(self):
        with self._lock:
            if self._ptr:
                self._lib.cb_destroy(self._ptr)
                self._ptr = None

    def __del__(self):
        if getattr(self, "_ptr", None):
            self.close()

    def _check_slot(self, slot):
        if not isinstance(slot, int) or not 0 <= slot < self.capacity:
            raise ValueError("slot outside fixed capacity")

    def reset(self, slot, generation, delay):
        self._check_slot(slot)
        if not 0 < generation < 2**64 or not 0 <= delay < 2**32:
            raise ValueError("invalid generation or delay")
        with self._lock:
            if generation <= self._last_generation.get(slot, 0):
                raise ValueError("generation must increase on slot reuse")
            # Batch registrations into the next native call to avoid a GIL
            # handoff for every newly admitted request.
            self._last_generation[slot] = generation
            self._pending_resets[slot] = (generation, delay)
            self._pending_releases.pop(slot, None)

    def _flush_resets(self):
        if not self._ptr:
            raise RuntimeError("closed builder")
        if self._pending_resets:
            entries = list(self._pending_resets.items())
            n = len(entries)
            slots = (C.c_int32 * n)(*(slot for slot, _ in entries))
            gens = (C.c_uint64 * n)(*(pair[0] for _, pair in entries))
            delays = (C.c_uint32 * n)(*(pair[1] for _, pair in entries))
            rc = self._lib.cb_reset_batch(self._ptr, n, slots, gens, delays)
            if rc:
                raise ValueError(f"native reset failed: {rc}")
            self._pending_resets.clear()
        if self._pending_releases:
            entries = list(self._pending_releases.items())
            n = len(entries)
            slots = (C.c_int32 * n)(*(slot for slot, _ in entries))
            gens = (C.c_uint64 * n)(*(gen for _, gen in entries))
            rc = self._lib.cb_release_batch(self._ptr, n, slots, gens)
            if rc:
                raise ValueError(f"native release failed: {rc}")
            self._pending_releases.clear()

    def release(self, slot, generation):
        self._check_slot(slot)
        with self._lock:
            if not self._ptr:
                raise RuntimeError("closed builder")
            if self._last_generation.get(slot) != generation:
                return False
            # Release generation ownership immediately in the Python facade;
            # native cleanup is batched before the next operation touches state.
            self._pending_releases[slot] = generation
            return True

    def append_raw(self, count, rows, slots, generations, flags, output, frames, kinds, finished, counters):
        """Pointers/ctypes arrays; caller retains all owning arrays until return.

        Rows are contiguous int64[count,16]; output int64[count,8,16]. Sidecars:
        slots int32[count], generations uint64[count], flags uint8[count]; output
        frames/kinds/finished uint8[count], counters uint64[count,3]. The caller
        must provide correctly sized arrays. The native API validates lane IDs,
        flag masks and stale generations, but cannot validate pointer lengths.
        """
        with self._lock:
            self._flush_resets()
            rc = self._lib.cb_append(
                self._ptr, count, rows, slots, generations, flags, output, frames, kinds, finished, counters
            )
            if rc:
                raise ValueError(f"native append failed: {rc}")

    def append(self, rows, slots, generations, flags):
        import torch

        n = len(slots)
        if not (len(rows) == len(generations) == len(flags) == n):
            raise ValueError("batch sidecar lengths differ")
        if not n:
            return []
        # The sender can retain ordinary owned [1,16] snapshots intact.
        # Concatenate them in one Torch call instead of indexing/reshaping each
        # row, which repeatedly releases/reacquires the GIL under contention.
        if self.frame_viewless and all(
            row is None
            or (
                type(row) is torch.Tensor
                and row.device.type == 'cpu'
                and row.dtype == torch.long
                and tuple(row.shape) == (1, 16)
                and row.is_contiguous()
                and not row.requires_grad
            )
            for row in rows
        ):
            if any(row is None and flag & HAS_FRAME for row, flag in zip(rows, flags)):
                raise ValueError('HAS_FRAME needs a row')
            zero = torch.zeros((1, 16), dtype=torch.long) if any(row is None for row in rows) else None
            slab = torch.cat([zero if row is None else row for row in rows], dim=0)
            self.viewless_batch_calls += 1
            self.viewless_rows += n
            return self.append_tensor(slab, slots, generations, flags)
        zero = None
        normalized = []
        for row, flag in zip(rows, flags):
            if row is None:
                if flag & HAS_FRAME:
                    raise ValueError("HAS_FRAME needs a row")
                if zero is None:
                    zero = torch.zeros(16, dtype=torch.long)
                normalized.append(zero)
            else:
                if row.device.type != "cpu" or row.dtype != torch.long or row.numel() != 16 or row.requires_grad:
                    raise ValueError("rows must be owned CPU int64[16] tensors without grad")
                normalized.append(row if row.ndim == 1 else row.reshape(16))
        slab = torch.stack(normalized, dim=0)
        return self.append_tensor(slab, slots, generations, flags)

    def append_tensor(self, rows, slots, generations, flags):
        """Avoid per-row stack when the producer already owns a contiguous batch."""
        import torch

        n = len(slots)
        if not (len(generations) == len(flags) == n):
            raise ValueError("batch sidecar lengths differ")
        if (
            rows.device.type != "cpu"
            or rows.dtype != torch.long
            or rows.shape != (n, 16)
            or not rows.is_contiguous()
            or rows.requires_grad
        ):
            raise ValueError("rows must be contiguous CPU int64[N,16]")
        if any(not isinstance(x, int) or not 0 <= x < self.capacity for x in slots):
            raise ValueError("invalid slot")
        if any(not isinstance(x, int) or not 0 <= x < 2**64 for x in generations):
            raise ValueError("invalid generation")
        if any(not isinstance(x, int) or not 0 <= x < 32 for x in flags):
            raise ValueError("invalid flags")
        slot_array = (C.c_int32 * n)(*slots)
        generation_array = (C.c_uint64 * n)(*generations)
        flag_array = (C.c_uint8 * n)(*flags)
        output = torch.empty((n, 8, 16), dtype=torch.long)
        frames, kinds, finished = [(C.c_uint8 * n)() for _ in range(3)]
        counters = (C.c_uint64 * (n * 3))()
        self.append_raw(
            n,
            rows.data_ptr(),
            slot_array,
            generation_array,
            flag_array,
            output.data_ptr(),
            frames,
            kinds,
            finished,
            counters,
        )
        events = []
        for i in range(n):
            kind = kinds[i]
            if not kind:
                continue
            if kind == 1:
                physical = 8 if flags[i] & TERMINAL else frames[i]
                codes = output[i, :physical]
            elif kind == 2:
                codes = torch.empty(0, dtype=torch.long)
            else:
                codes = None
            events.append(
                ChunkEvent(i, KINDS[kind], codes, frames[i], counters[i * 3], counters[i * 3 + 1], counters[i * 3 + 2])
            )
        return events

    def export_state(self, slot):
        import torch

        self._check_slot(slot)
        with self._lock:
            self._flush_resets()
            metadata = (C.c_uint64 * 7)()
            rows = torch.empty((8, 16), dtype=torch.long)
            rc = self._lib.cb_export(self._ptr, slot, metadata, rows.data_ptr())
            if rc:
                raise ValueError(f"native export failed: {rc}")
            generation, seen, emitted, chunks, delay, pending, active = metadata
            return {
                "generation": generation,
                "seen": seen,
                "emitted": emitted,
                "chunks": chunks,
                "delay": delay,
                "active": bool(active),
                "buffer": [row.clone() for row in rows[:pending]],
            }


if __name__ == "__main__":
    print(build_native_library())
