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

# The metric names JAX actually emits, verified by listening on jax 0.10.2 rather than by reading
# documentation. An earlier version of this table appended "_secs" to each -- a suffix JAX does NOT
# use -- so NOTHING ever matched and `record()` returned 0.0 for all three stages while looking
# perfectly healthy. Both spellings are accepted now so a rename degrades to a duplicate rather than
# to a silent zero, and `Timings.matched` makes a total miss impossible to overlook.
_STAGES = {
    "jaxpr_trace_duration": "trace",
    "jaxpr_to_mlir_module_duration": "lower",
    "backend_compile_duration": "backend",
}
_KEYS = {}
for _stem, _lab in _STAGES.items():
    _KEYS[f"/jax/core/compile/{_stem}"] = _lab
    _KEYS[f"/jax/core/compile/{_stem}_secs"] = _lab


class Timings(dict):
    """Stage durations in seconds, plus wall time and any cache events seen."""

    @property
    def total(self) -> float:
        return sum(self.get(k, 0.0) for k in ("trace", "lower", "backend"))

    @property
    def matched(self) -> bool:
        """Did any JAX metric actually match? False means the names moved and every stage reads
        0.0 -- a failure that is otherwise indistinguishable from a genuinely instant compile."""
        return any(self.get(k, 0.0) > 0.0 for k in ("trace", "lower", "backend"))

    @property
    def unaccounted(self) -> float:
        """Wall time minus the three stages. Large values mean the time is somewhere JAX does not
        instrument -- dispatch, host transfer, your own code, or (see
        :meth:`trace_looks_blind`) tracing that JAX declined to time."""
        return self.get("wall", 0.0) - self.total

    @property
    def trace_looks_blind(self) -> bool:
        """True when ``trace`` reads ~0 while most of the wall is unaccounted.

        That combination has ONE common cause and it is not dispatch. JAX emits
        ``jaxpr_trace_duration`` only under ``core.trace_state_clean()``
        (jax/_src/pjit.py:528), so a TOP-LEVEL ``vmap`` or ``grad`` over a jitted callee pushes a
        trace, no enclosing jit event exists, and the metric reads a silent 0.0 while real tracing
        happens. Measured on jax 0.10.2 / CPU with a 10-operand ``einsum(optimize='optimal')``::

            jax.jit(inner).lower(*xs)   wall 2.518 s   trace 2.511 s
            jax.vmap(inner)(*xs)        wall 2.711 s   trace 0.001 s   2.651 s unaccounted

        ``matched`` stays True in the second row -- lower and backend are non-zero -- so nothing
        else in this module notices. :func:`scopex.trace_profile` is oblivious to
        ``trace_state_clean`` (it hooks CPython frames, not pjit) and reports the same 2.6 s.

        :func:`record` itself is SAFE from this, because it always wraps in
        ``jax.jit(fn).lower(...)``. You are exposed when you time your own
        ``jax.vmap(jitted)(...)`` with a listener.
        """
        w = self.get("wall", 0.0)
        return (self.matched and w > 0.05
                and self.get("trace", 0.0) < 0.01 * w
                and self.unaccounted > 0.5 * w)

    def __str__(self) -> str:
        w = self.get("wall", 0.0)
        if not self.matched:
            return (f"NO JAX METRICS MATCHED -- every stage would read 0.0.\n"
                    f"jax.monitoring's metric names have moved; scopex.monitor._KEYS needs "
                    f"updating.\nSaw these names: {sorted(self.get('seen_names', []))}\n"
                    f"WALL {w:.3f} s (the only trustworthy number here)")
        rows = [f"{'stage':10s} {'seconds':>9s} {'share':>7s}"]
        rows.append("-" * len(rows[0]))
        for k in ("trace", "lower", "backend"):
            v = self.get(k, 0.0)
            rows.append(f"{k:10s} {v:9.3f} {100 * v / max(1e-9, w):6.1f}%")
        rows.append(f"{'unaccounted':10s} {self.unaccounted:9.3f} "
                    f"{100 * self.unaccounted / max(1e-9, w):6.1f}%")
        rows.append(f"{'WALL':10s} {w:9.3f}")
        if self.trace_looks_blind:
            rows.append(
                "\ntrace reads ~0.0 while most of the wall is unaccounted. Do NOT read that as\n"
                "dispatch or host transfer. JAX emits jaxpr_trace_duration only under\n"
                "core.trace_state_clean(), so a TOP-LEVEL vmap/grad over a jitted callee makes\n"
                "the metric silently zero while tracing happens (measured: 2.651 s of 2.711 s).\n"
                "  scopex.trace_profile(fn, *args)   hooks CPython frames, so it sees it anyway")
        b = self.get("backend", 0.0)
        if b / max(1e-9, w) > 0.5:
            rows.append(
                "\nbackend dominates, and it is ONE number covering HLO passes, autotuning and\n"
                "codegen -- which want opposite responses. To split it:\n"
                "  scopex.pass_timings(src)   per-XLA-pass seconds (runs a SUBPROCESS: vmodule is\n"
                "                             read at `import jax` and cannot be set after)\n"
                "  scopex.dump()              XLA's own artifacts (must precede the 1st compile;\n"
                "                             setting XLA_FLAGS later is a SILENT no-op)")
        if self.get("cache_events"):
            rows.append(f"cache events: {dict(self['cache_events'])}")
        return "\n".join(rows)


@contextmanager
def _listen(acc, events, seen):
    """Register the two listeners for the duration of the block, then REMOVE them.

    An earlier version of this module stated in two places that "jax.monitoring has no public
    deregister; listeners are process-global and cannot be removed in jax 0.10.2", and `record`'s
    docstring told users to run each measurement in a fresh subprocess because of it. That is not
    true on jax 0.10.2 and may never have been: ``jax.monitoring`` re-exports
    ``unregister_event_duration_listener``, ``unregister_event_listener``,
    ``unregister_event_time_span_listener``, ``unregister_scalar_listener`` and
    ``clear_event_listeners``. Verified by counting ``jax._src.monitoring`` listeners across a
    register/unregister pair: 0 -> 1 -> 0.

    Note the ASYMMETRIC NAMES -- ``register_event_duration_secs_listener`` pairs with
    ``unregister_event_duration_listener`` (no ``_secs``) -- and that the unregister functions
    ``assert callback in <list>`` before removing, so they raise AssertionError (or ValueError under
    ``python -O``, where the assert is stripped and ``list.remove`` raises instead) if the callback
    is already gone. Cleanup must never be able to fail a measurement that succeeded, so both are
    swallowed here.
    """
    def cb(name, value, **kw):
        seen.add(name)
        if name in _KEYS:
            acc[_KEYS[name]] += float(value)
        elif "cache" in name:
            events[name] += 1

    ev = lambda name, **kw: cb(name, 0.0, **kw)                              # noqa: E731
    jax.monitoring.register_event_duration_secs_listener(cb)
    jax.monitoring.register_event_listener(ev)
    try:
        yield
    finally:
        for fname, f in (("unregister_event_duration_listener", cb),
                         ("unregister_event_listener", ev)):
            try:
                getattr(jax.monitoring, fname)(f)
            except Exception:            # missing in a future jax, or already removed
                pass


def record(fn, *args, clear_caches: bool = True, **kwargs) -> Timings:
    """Compile ``fn(*args)`` once and report where the time went.

    ``clear_caches`` makes this a COLD compile, which is almost always what you want to measure --
    a warm compile returns in microseconds and tells you nothing.

    SAFE TO CALL REPEATEDLY IN ONE PROCESS. Its listeners are removed when it returns (see
    :func:`_listen`), and even before that fix they did not inflate later readings, because each
    call's listener writes into its own accumulator -- three consecutive records on one program read
    total/wall 0.92, 0.96, 0.96. What is NOT safe is a hand-rolled timing loop without
    ``jax.clear_caches()``, which measures a warm compile and reads ~0.

    THIS FUNCTION IS ALSO SAFE FROM THE ``trace_state_clean`` BLIND SPOT, because it always wraps in
    ``jax.jit(fn).lower(...)``. Timing your own ``jax.vmap(jitted)(...)`` with a listener is not --
    see :meth:`Timings.trace_looks_blind`.
    """
    acc: defaultdict = defaultdict(float)
    events: defaultdict = defaultdict(int)
    seen: set = set()
    if clear_caches:
        jax.clear_caches()
    with _listen(acc, events, seen):
        t0 = time.perf_counter()
        jax.jit(fn).lower(*args, **kwargs).compile()
        wall = time.perf_counter() - t0
    t = Timings(acc)
    t["wall"] = wall
    t["cache_events"] = dict(events)
    t["seen_names"] = sorted(seen)
    if not t.matched:
        import warnings
        warnings.warn(
            "scopex.record matched no jax.monitoring metrics; all stage times are 0.0. "
            f"jax emitted: {sorted(seen)}. Update scopex.monitor._KEYS.",
            RuntimeWarning, stacklevel=2)
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
