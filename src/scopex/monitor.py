"""Where the time actually went: tracing, lowering, or the backend compiler.

This is the FIRST instrument to reach for, before any attribution. It costs one compile, needs no
flags, and it answers the only question that routes all the others: which stage is slow?

The numbers come from ``jax.monitoring``, which JAX emits itself. They are not sampled and not
estimated -- they matched wall clock to three decimals in every program we checked. Attribution
tooling that runs before you know the stage split is guessing: source attribution explains time in
tracing and in HLO passes, and says nothing at all about time in autotuning.

Two regimes show up repeatedly, and they want different tools:

``PASS-BOUND``     most of the backend time is in HLO passes. Source attribution and the fusion
                   graph apply.
``AUTOTUNE-BOUND`` most of the backend time is in picking kernel configurations. Source attribution
                   will point at code that is not the problem; change autotuning flags instead.

This module reports the split. It does not guess the regime for you unless you ask
:func:`regime`, which is a heuristic and says so.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager

import jax

__all__ = ["record", "Timings", "regime"]

_KEYS = {
    "/jax/core/compile/jaxpr_trace_duration_secs": "trace",
    "/jax/core/compile/jaxpr_to_mlir_module_duration_secs": "lower",
    "/jax/core/compile/backend_compile_duration_secs": "backend",
}


class Timings(dict):
    """Stage durations in seconds, plus wall time and any cache events seen."""

    @property
    def total(self) -> float:
        return sum(self.get(k, 0.0) for k in ("trace", "lower", "backend"))

    @property
    def unaccounted(self) -> float:
        """Wall time minus the three stages. Large values mean the time is somewhere JAX does not
        instrument -- dispatch, host transfer, or your own code."""
        return self.get("wall", 0.0) - self.total

    def __str__(self) -> str:
        w = self.get("wall", 0.0)
        rows = [f"{'stage':10s} {'seconds':>9s} {'share':>7s}"]
        rows.append("-" * len(rows[0]))
        for k in ("trace", "lower", "backend"):
            v = self.get(k, 0.0)
            rows.append(f"{k:10s} {v:9.3f} {100 * v / max(1e-9, w):6.1f}%")
        rows.append(f"{'unaccounted':10s} {self.unaccounted:9.3f} "
                    f"{100 * self.unaccounted / max(1e-9, w):6.1f}%")
        rows.append(f"{'WALL':10s} {w:9.3f}")
        if self.get("cache_events"):
            rows.append(f"cache events: {dict(self['cache_events'])}")
        return "\n".join(rows)


@contextmanager
def _listen(acc, events):
    def cb(name, value, **kw):
        if name in _KEYS:
            acc[_KEYS[name]] += float(value)
        elif "cache" in name:
            events[name] += 1

    jax.monitoring.register_event_duration_secs_listener(cb)
    jax.monitoring.register_event_listener(lambda name, **kw: cb(name, 0.0, **kw))
    try:
        yield
    finally:
        # jax.monitoring has no public deregister; listeners are process-global and additive. That
        # is why `record` warns against calling it many times in one process (see its docstring).
        pass


def record(fn, *args, clear_caches: bool = True, **kwargs) -> Timings:
    """Compile ``fn(*args)`` once and report where the time went.

    ``clear_caches`` makes this a COLD compile, which is almost always what you want to measure --
    a warm compile returns in microseconds and tells you nothing.

    CAVEAT: ``jax.monitoring`` listeners are process-global and cannot be removed in jax 0.10.2, so
    calling this repeatedly in one process accumulates listeners and inflates later readings. For
    a benchmark loop, run each measurement in a fresh subprocess.
    """
    acc: defaultdict = defaultdict(float)
    events: defaultdict = defaultdict(int)
    if clear_caches:
        jax.clear_caches()
    with _listen(acc, events):
        t0 = time.perf_counter()
        jax.jit(fn).lower(*args, **kwargs).compile()
        wall = time.perf_counter() - t0
    t = Timings(acc)
    t["wall"] = wall
    t["cache_events"] = dict(events)
    return t


def regime(t: Timings, *, threshold: float = 0.6) -> str:
    """A HEURISTIC label, not a measurement.

    Returns ``'trace-bound'``, ``'lower-bound'``, ``'backend-bound'`` or ``'mixed'``. Distinguishing
    PASS-BOUND from AUTOTUNE-BOUND inside the backend needs per-pass timing, which this module does
    not collect -- see :mod:`scopex.flags` for the vmodule and dump settings that do.
    """
    w = max(1e-9, t.total)
    for k in ("trace", "lower", "backend"):
        if t.get(k, 0.0) / w >= threshold:
            return f"{k}-bound"
    return "mixed"
