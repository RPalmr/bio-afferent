"""Benchmark helpers: warmup, compile/steady-state split, backend report."""

from __future__ import annotations

import platform
import statistics
import time

import jax


def backend_info() -> dict:
    dev = jax.devices()[0]
    return {
        "backend": dev.platform,
        "device": str(dev),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "x64": bool(jax.config.read("jax_enable_x64")),
    }


def measure(fn, warmup=5, reps=25) -> dict:
    """Time steady-state execution; ``fn`` must be a zero-arg thunk."""
    for _ in range(warmup):
        jax.block_until_ready(fn())
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        samples.append(time.perf_counter() - t0)
    return {
        "median_s": statistics.median(samples),
        "mean_s": statistics.mean(samples),
        "stdev_s": statistics.pstdev(samples),
        "min_s": min(samples),
        "max_s": max(samples),
        "reps": reps,
    }


def report(name: str, stats: dict, batch_desc: str) -> None:
    info = backend_info()
    print(f"[{name}] batch={batch_desc} dtype=x32 backend={info['backend']}")
    med = stats["median_s"] * 1e3
    mean = stats["mean_s"] * 1e3
    std = stats["stdev_s"] * 1e6
    print(f"  median={med:.3f} ms mean={mean:.3f} ms (n={stats['reps']})")
    print(f"  stdev={std:.1f} us min={stats['min_s'] * 1e3:.3f} ms")
