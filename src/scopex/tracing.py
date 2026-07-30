"""WHERE inside the trace stage the seconds went -- the one thing :func:`scopex.record` cannot say.

``record()`` reports ``trace-bound`` and stops. Every other level in this package starts at the
jaxpr, which is the OUTPUT of tracing, so none of them can see a second spent producing it. On the
corpus's einsum case that is 31.5 s against a 0.005 s control -- a 6,332x regression on which all
five scopex attribution views (split / package / site / transform / kind) are IDENTICAL between the
arms. The finding that opened that investigation said scopex "can never say WHO without a sampling
profiler over the tracing call stack". This module is that profiler.

    scopex.trace_profile(fn, *args)          -> frames by SELF time, and sites in YOUR code

Two implementations, and the choice matters:

``method='xplane'`` (default)
    The python tracer that ships inside jaxlib (``ProfileOptions.python_tracer_level = 1``). It
    writes one event per python call onto a single ``/host:CPU`` line named ``python``, with
    ``start_ns``/``duration_ns``. Nesting is by containment, so a stack over start-sorted events
    reconstructs the call tree exactly and SELF time is recursion-safe. It keeps real stacks, so it
    can charge time to the nearest enclosing frame of YOUR code and can separate tracing at nesting
    depth 1 from depth 12.

``method='cprofile'``
    stdlib ``cProfile`` around the same ``make_jaxpr``. Cheaper, no temp files, no protobuf, and it
    records FULL ABSOLUTE PATHS where the xplane tracer records only a basename. It aggregates by
    function and keeps NO stacks, so the per-site table is unavailable.

The two agreeing is the strongest evidence either is right. On the einsum case xplane self-time
reads 4.998 s for ``opt_einsum/paths.py:236 _optimal_iterate`` and cProfile ``tottime`` reads
4.987 s -- two independent instruments, 0.2% apart.

MEASURED
    einsum_optimal_n10 (jax#2583)   4.998 s self, 94% of the traced run, in a THIRD-PARTY frame
                                    containing no JAX code at all. Its ``_pathlit`` control -- same
                                    program, byte-identical HLO, solver skipped -- profiles at
                                    0.020 s with its top real frame at 0.0008 s. ~6,000x.
    retrace_static_40 (jax#4667 family) 61.4% of trace self-time charged to the jitted helper that
                                    misses the cache 40 times, at file:line granularity matching
                                    scopex's existing ``site`` convention.
    jitfib_t20                      39.5% charged to the user's ``fib``, i.e. it attributes tracing
                                    that happens INSIDE nested jits (22 nested ``_trace_for_jit``,
                                    55 ``cache_miss``) where the stage split has one scalar.

IT ALSO SEES WHAT THE STAGE SPLIT STRUCTURALLY CANNOT.
``jaxpr_trace_duration`` is emitted by JAX only under ``core.trace_state_clean()``
(jax/_src/pjit.py:528), so a TOP-LEVEL ``vmap`` or ``grad`` over a jitted callee pushes a trace, no
enclosing jit event exists, and the metric reads a silent **0.0** while real tracing happens.
Measured here on jax 0.10.2 / CPU with a 10-operand ``einsum(optimize='optimal')``::

    jax.jit(inner).lower(*xs)      wall 2.518 s   trace metric 2.511 s
    jax.vmap(inner)(*xs)           wall 2.711 s   trace metric 0.001 s   2.651 s MISSING

``Timings.matched`` stays True in the second row -- lower and backend are non-zero -- so nothing
warns, and 98% of the compile lands in ``unaccounted``. This profiler is oblivious to
``trace_state_clean`` because it hooks CPython frames rather than pjit, and reports the same 2.6 s.
:meth:`scopex.Timings.trace_looks_blind` now flags that combination and points here.
"""

from __future__ import annotations

import glob
import os
import shutil
import sys
import tempfile
import time
from collections import Counter

__all__ = ["trace_profile", "TraceProfile", "TraceProfileError"]

NO_FRAME = "<no-frame>"

# Above this many python events the xplane capture costs more memory than most sessions want; see
# trace_profile's docstring for the measured curve.
EVENT_BUDGET = 4_000_000


class TraceProfileError(RuntimeError):
    """The instrument produced nothing. NEVER return an empty profile instead.

    An empty profile is indistinguishable from an instant trace, which is precisely the silent
    failure shape this package exists to prevent.
    """


# ── frame classification ────────────────────────────────────────────────────────────────────────

def _lib_roots() -> tuple[str, ...]:
    import sysconfig
    roots = set()
    for k in ("purelib", "platlib", "stdlib", "platstdlib"):
        p = sysconfig.get_paths().get(k)
        if p:
            roots.add(os.path.abspath(p) + os.sep)
    return tuple(roots)


_LIB_ROOTS = _lib_roots()


def _lib_basenames() -> set[str]:
    """Basenames of every currently-imported module living under site-packages or the stdlib.

    THE XPLANE PYTHON TRACER RECORDS ONLY A BASENAME -- measured: 0 of 527 distinct frame names in
    one capture contained a ``/``. Resolving a basename back to a full path through ``sys.modules``
    is AMBIGUOUS for 53.7% of them (``core.py``, ``util.py``, ``__init__.py`` ...), so a resolver
    that picked one would be wrong about half the time and would say nothing about being wrong.

    This rule therefore never resolves. A frame is LIBRARY iff its basename belongs to some imported
    library module; otherwise it is USER. The error is ONE-DIRECTIONAL by construction: a user file
    named ``core.py`` is charged to the library bucket (under-reporting user code), and library time
    is NEVER charged to user code. ``method='cprofile'`` has real paths and needs no such rule.
    """
    out = set()
    for mod in list(sys.modules.values()):
        p = getattr(mod, "__file__", None)
        if p and os.path.abspath(p).startswith(_LIB_ROOTS):
            out.add(os.path.basename(p))
    return out


def _is_user(name: str, libnames: set[str]) -> bool:
    if not name.startswith("$"):
        return False
    head = name[1:].split(" ", 1)[0]
    if ":" not in head:
        return False                      # '$builtins len', '$jaxlib.utils safe_map'
    base = head.rsplit(":", 1)[0]
    if os.path.isabs(base):
        return not base.startswith(_LIB_ROOTS)
    return base not in libnames


def _short(name: str) -> str:
    return name[1:] if name.startswith("$") else name


# ── xplane reading ──────────────────────────────────────────────────────────────────────────────

def _python_line(path):
    from jax.profiler import ProfileData
    pd = ProfileData.from_file(path)
    planes = {p.name: p for p in pd.planes}
    host = planes.get("/host:CPU")
    if host is None:
        raise TraceProfileError(
            f"XPlane has no '/host:CPU' plane. planes: {sorted(planes)}. Nothing python-level was "
            f"recorded; refusing to report an empty profile.")
    lines = {ln.name: ln for ln in host.lines}
    if "python" not in lines:
        raise TraceProfileError(
            f"'/host:CPU' has no line named 'python' -- ProfileOptions.python_tracer_level did not "
            f"take effect. lines: {sorted(lines)}. Every per-frame number would read 0.0, which is "
            f"indistinguishable from an instant trace. Refusing.")
    return lines["python"]


def _walk_events(events, libnames):
    """``(self_ns, incl_ns, calls, user_self, user_incl)`` from one containment stack pass.

    ``user_owner`` is the nearest ENCLOSING user frame -- the line of your program responsible for
    this frame running -- which is what makes the per-site table possible at all.
    """
    evs = sorted(((e.start_ns, e.end_ns, e.name) for e in events), key=lambda t: (t[0], -t[1]))
    self_ns, incl_ns, calls = Counter(), Counter(), Counter()
    user_self, user_incl = Counter(), Counter()
    stack: list = []          # [end, name, self_accum, user_owner]
    for start, end, name in evs:
        while stack and stack[-1][0] <= start:
            _e, nm, sf, uo = stack.pop()
            self_ns[nm] += sf
            user_self[uo] += sf
        owner = name if _is_user(name, libnames) else (stack[-1][3] if stack else NO_FRAME)
        if stack:
            stack[-1][2] -= (end - start)
        if _is_user(name, libnames) and not any(_is_user(f[1], libnames) for f in stack):
            user_incl[name] += end - start
        stack.append([end, name, end - start, owner])
        incl_ns[name] += end - start
        calls[name] += 1
    for _e, nm, sf, uo in reversed(stack):
        self_ns[nm] += sf
        user_self[uo] += sf
    return self_ns, incl_ns, calls, user_self, user_incl


# ── result ──────────────────────────────────────────────────────────────────────────────────────

class TraceProfile:
    """Where the trace stage's seconds went. ``method`` is always visible -- read it."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    @property
    def frames(self):
        """``[(frame, self_s, incl_s, calls)]`` by self time, instrument frames dropped."""
        return [r for r in self._frames if "jax/_src/profiler.py" not in r[0]]

    @property
    def sites(self):
        """``[(user file:line func, self_s)]`` -- trace time charged to YOUR code.

        Empty for ``method='cprofile'``, which keeps no stacks.
        """
        return self._sites

    @property
    def top_frame(self):
        """``(frame, self_s)`` for the hottest frame, or ``(None, 0.0)``."""
        f = self.frames
        return (_short(f[0][0]), f[0][1]) if f else (None, 0.0)

    def __str__(self):
        out = [f"trace_profile  {self.traced_wall:.3f} s traced ({self.n_events} events, "
               f"method={self.method}, instrument overhead x{self.overhead:.2f})", "",
               "SELF TIME BY FRAME", f"{'self_s':>9} {'incl_s':>9} {'calls':>9}  frame", "-" * 86]
        for name, s, i, c in self.frames[:15]:
            out.append(f"{s:9.4f} {i:9.4f} {c:9d}  {_short(name)[:58]}")
        if self.method == "cprofile":
            out += ["", "(cprofile keeps no stacks, so the per-site table is unavailable; "
                        "the frame paths above are absolute and unambiguous)"]
            return "\n".join(out)
        out += ["", "SELF TIME CHARGED TO YOUR CODE (nearest enclosing non-library frame)",
                f"{'self_s':>9} {'share':>7}  site", "-" * 86]
        tot = max(1e-9, sum(s for _, s in self.sites))
        for name, s in self.sites[:10]:
            out.append(f"{s:9.4f} {100 * s / tot:6.1f}%  {_short(name)[:60]}")
        return "\n".join(out)


# ── entry point ─────────────────────────────────────────────────────────────────────────────────

def trace_profile(fn, *args, method: str = "xplane", keep: str | None = None,
                  static_argnums=None, **kwargs) -> TraceProfile:
    """Trace ``fn(*args)`` under a python-level profiler and report where the seconds went.

    ONLY TRACING RUNS -- no lowering, no backend compile -- so every second reported here is a
    second :func:`scopex.record` would have put in the ``trace`` row.

    HARD LIMIT, AND IT CORRELATES WITH THE THING BEING MEASURED. The xplane tracer holds one event
    per python call in memory. Measured (traced wall / events / xplane bytes / maxRSS)::

        jitfib_t20          0.044 s      18,911 ev     0.3 MB     529 MB
        retrace_static_40   1.39  s     767,937 ev      12 MB   1,013 MB
        einsum_n10          6.62  s   1,727,007 ev      29 MB   1,588 MB
        retrace_static_320  6.97  s   6,643,988 ev     108 MB   4,665 MB

    That is ~950k events/s and ~0.6 GB RSS per million events, so the LONGER the trace you need
    explained, the likelier the instrument OOMs -- roughly 20 GB for 30 s of dense tracing. Use
    ``method='cprofile'`` above a few seconds of tracing; it costs less (overhead x1.35 vs x1.45 on
    the einsum case) and has real absolute paths. ``EVENT_BUDGET`` is where this module calls the
    xplane route imprudent; exceeding it warns rather than truncating, because a truncated profile
    with no warning is the failure mode this package exists to prevent.

    ``static_argnums`` is forwarded to ``jax.make_jaxpr``. A ``lower``-only profile is NOT offered:
    ``jit().lower()`` re-traces, so the lowering stage cannot be sampled in isolation -- an early
    draft of this instrument reported "LOWER 33.684 s, 99.9% opt_einsum" for a stage that
    ``record()`` measures at 0.031 s, because it had run ``make_jaxpr`` untimed inside the sampled
    window.
    """
    import jax

    if method not in ("xplane", "cprofile"):
        raise ValueError(f"method must be 'xplane' or 'cprofile', not {method!r}")
    mj = {} if static_argnums is None else {"static_argnums": static_argnums}

    jax.clear_caches()
    t0 = time.perf_counter()
    jax.make_jaxpr(fn, **mj)(*args, **kwargs)
    base = time.perf_counter() - t0

    if method == "cprofile":
        import cProfile
        import pstats
        jax.clear_caches()
        pr = cProfile.Profile()
        t0 = time.perf_counter()
        pr.enable()
        try:
            jax.make_jaxpr(fn, **mj)(*args, **kwargs)
        finally:
            pr.disable()
        traced = time.perf_counter() - t0
        st = pstats.Stats(pr).stats
        if not st:
            raise TraceProfileError("cProfile recorded no functions at all; refusing to report an "
                                    "empty profile.")
        rows = sorted(((f"{f}:{ln} {n}", v[2], v[3], v[0]) for (f, ln, n), v in st.items()),
                      key=lambda r: -r[1])
        return TraceProfile(_frames=rows, _sites=[], traced_wall=traced, baseline=base,
                            overhead=traced / max(base, 1e-9),
                            n_events=sum(v[0] for v in st.values()),
                            method="cprofile", residual=0.0)

    from jax.profiler import ProfileOptions
    logdir = keep or tempfile.mkdtemp(prefix="scopex_trace_profile_")
    opts = ProfileOptions()
    opts.python_tracer_level = 1
    opts.host_tracer_level = 2
    jax.clear_caches()
    jax.profiler.start_trace(logdir, profiler_options=opts)
    t1 = time.perf_counter()
    try:
        jax.make_jaxpr(fn, **mj)(*args, **kwargs)
    finally:
        traced = time.perf_counter() - t1
        jax.profiler.stop_trace()

    files = glob.glob(os.path.join(logdir, "**", "*.xplane.pb"), recursive=True)
    if not files:
        raise TraceProfileError(
            f"jax.profiler wrote no *.xplane.pb under {logdir}. Nothing was captured, and a zeroed "
            f"profile would read as an instant trace.")
    newest = sorted(files)[-1]
    size = os.path.getsize(newest)
    events = list(_python_line(newest).events)
    if not events:
        raise TraceProfileError(
            "the 'python' line exists but has zero events -- the tracer attached and recorded "
            "nothing. Refusing to report a profile of an untraced run.")
    if len(events) > EVENT_BUDGET:
        import warnings
        warnings.warn(
            f"{len(events):,} python events (~{0.6 * len(events) / 1e6:.1f} GB peak RSS by the "
            f"measured curve). The profile below is COMPLETE, not truncated, but re-run with "
            f"method='cprofile' if this session is memory-constrained.",
            RuntimeWarning, stacklevel=2)
    self_ns, incl_ns, calls, user_self, _ui = _walk_events(events, _lib_basenames())
    if keep is None:
        shutil.rmtree(logdir, ignore_errors=True)

    frames = sorted(((n, self_ns[n] / 1e9, incl_ns[n] / 1e9, calls[n]) for n in self_ns),
                    key=lambda r: -r[1])
    sites = sorted(((n, v / 1e9) for n, v in user_self.items()), key=lambda r: -r[1])
    return TraceProfile(_frames=frames, _sites=sites, traced_wall=traced, baseline=base,
                        overhead=traced / max(base, 1e-9), n_events=len(events),
                        method="xplane", xplane_bytes=size,
                        residual=(max(e.end_ns for e in events) - min(e.start_ns for e in events)
                                  - sum(self_ns.values())) / 1e9)
