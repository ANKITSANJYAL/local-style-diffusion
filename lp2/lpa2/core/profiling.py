from __future__ import annotations
import time
from contextlib import contextmanager
import torch

@contextmanager
def profiler():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    vram_mb = 0.0
    if torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
    yield_data = {"time_sec": round(dt, 3), "peak_mem_mb": round(vram_mb, 1)}
    # expose to caller
    try:
        from contextvars import ContextVar
    except Exception:
        pass
    # simpler: return via caller
    print(f"[PROFILE] time_sec={yield_data['time_sec']} peak_mem_mb={yield_data['peak_mem_mb']}")
