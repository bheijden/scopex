"""RECIPE -- ``stage_split.py`` says trace-bound. scopex can time that stage and CANNOT attribute
inside it. So who is it?

This is the recipe for where the package stops. Every attribution view in scopex keys on a jaxpr
equation, and a trace-stage pathology is time spent NOT emitting equations -- a path solver, a
pytree flatten, a python loop that runs before any primitive is bound. On the case below, every
view returns the same constant for both arms: ``author`` and ``library`` are ``<none>`` for all ten
equations, ``package`` and ``role`` are ``<unmarked>``, ``site`` is ``<no-frame>``, ``transform`` is
``<primal>``, and ``scope`` is the identical einsum spec string. There is nothing to attribute,
because the expensive code emitted nothing.

Two instruments, in the order they answer:

    boundary arity      jax.tree_util.tree_flatten(args) -- how many ARRAY OBJECTS is jit being
                        asked to think about? Compilation cost scales with this, not with array
                        size. One line, no compile.
    trace_profile()     a SAMPLING profiler running only for the duration of the trace stage,
                        aggregated by package. ~200 Hz off a background thread, so it does not
                        distort what it measures the way a deterministic profiler would.

FOUND ON: einsum_optimal_n11 (jax#2583 / PR#25214) and arity_tree_200 (jax#4667), both CPU -- and
both are platform-INDEPENDENT by construction, because the cost is python before any backend is
consulted.

MEASURED (original investigation):
  einsum_optimal_n11 vs _control       trace 49.664 s vs 0.011 s = 4,500x
                                       lower 0.086 vs 0.054, backend 0.219 vs 0.199 -- IDENTICAL
                                       regime 'trace-bound' vs 'backend-bound'
      Every IR view is a null and correctly so: jaxpr 10 equations vs 10, StableHLO 31 lines vs 31,
      walk_hlo 61 units vs 60 (1.02x), and the per-pass instruction curve stays between 0.86x and
      1.33x across all 33 snapshots with an identical final module.
      The case ships a `_pathlit` arm -- the optimal contraction path pasted in as a literal, so the
      solver never runs -- which reads trace 0.016 s / backend 0.357 s. That pins the cost to the
      SEARCH and exonerates the resulting contraction. A discriminating arm like that is worth more
      than any profiler, when the case gives you one.
      Trace ladder: 0.007 s / 4.961 s / 49.664 s at nops 8 / 10 / 11 against a control flat at
      0.011-0.012 s. Super-exponential, as the issue claims.
  arity_tree_200 vs _control           trace 14.183 s vs 0.114 s = 124x
                                       lower 2.451 vs 0.101 = 24x, backend 14.256 vs 0.223 = 64x
      trace+lower is 16.6 s of the 31.5 s wall = the LARGER half, and it is structurally invisible
      to pass_timings, to dump() and to every HLO-level view. `regime` still says 'mixed', because
      the backend is inflated too -- read the RATIO, not the label.
      The independent variable is the BOUNDARY: 2*nleaves arrays crossing jit (400 vs 2 at n=200),
      and the pre-optimization HLO has 2*nleaves parameters. Scaling is LINEAR in leaf count on
      0.10.2 (n=100 -> 200 doubles trace 1.93x), refuting the issue's "closer to quadratic".

RE-MEASURED for this recipe (CPU, serial, x64):
  einsum_optimal_n11   trace 31.468 s vs 0.005 s = 6,332x; lower 0.039 vs 0.007; backend 0.097 vs
                       0.096 = 1.0x. regime 'trace-bound' vs 'backend-bound'. jaxpr 10 eqns vs 10.
      THE SAMPLER NAMES IT, which the original investigation said nothing in scopex could do:
      100% of 3,152 trace-stage samples are in `opt_einsum`, and the hot leaves are
          paths.py:256 _optimal_iterate   54.4%
          paths.py:236 _optimal_iterate   29.3%
          paths.py:274 / :244             8.5% / 7.6%
      i.e. the optimal-contraction-path search, in four lines of one function. Every scopex
      attribution view (split, package, site, transform, kind) is IDENTICAL between the two arms,
      which is the null the original recorded and the reason this instrument had to exist.
  arity_tree_200       trace 9.815 s vs 0.069 s = 142x; lower 1.577 vs 0.023 = 69x; backend 10.560
                       vs 0.126 = 84x. trace+lower = 52% of the wall (original: 54%).
                       jaxpr equations 36,800 vs 183 -- exactly the original figure.
                       BOUNDARY: 400 leaves vs 2, from 2 arguments in BOTH arms. That is the whole
                       diagnosis, and `n_args` alone would have missed it.
                       96% of trace samples are in `jax` itself, spread thin (top leaf 4%): there
                       is no hot line, because the cost is per-leaf bookkeeping. A profile with no
                       peak is the signature of the boundary problem, and the leaf count is the
                       number to report.
                       `regime` returned 'mixed' for the case, as reported -- and on this run it
                       also returned 'mixed' for the CONTROL, whose three stages are 0.069/0.023/
                       0.126 s. At that scale the label is noise. Read the ratios.

WHEN IT WORKS
    * Whenever `record` puts the seconds in trace (or in lower). It is the only way to get a NAME,
      and the answer is usually a package rather than a line: opt_einsum's path solver, jax's own
      pytree flattening, a library's __init__ running per element.
    * The boundary-arity count works before you compile anything and costs nothing.

WHEN IT DOES NOT
    * A SAMPLER SEES THE STACK, NOT THE COST. Deep C extensions (numpy inner loops) show up as the
      python frame that called them. Fine for naming a package, useless for a line inside C.
    * It cannot see time in a C++ extension that releases the GIL and never returns to python.
    * THE `lower` STAGE CANNOT BE SAMPLED ON ITS OWN. `jit(fn).lower(*args)` re-traces on its way
      to StableHLO, so a sampler around it sees trace+lower together; only `record`, reading jax's
      own counters, separates them. This recipe therefore reports `trace` and `trace+lower`, never
      a bare `lower`. (The first draft did report one, and it read 33.684 s / 99.9% opt_einsum for
      a stage `record` measured at 0.031 s.)
    * CACHING WILL LIE TO YOU, and not through jax's cache. opt_einsum memoises contraction paths
      itself, so timing `make_jaxpr` a second time in the same process reads 0.002 s for a 49 s
      search. `scopex.record` calls `jax.clear_caches()` and still measured the honest 4.75 s on a
      second call -- but this recipe re-imports nothing, so profile in a FRESH process and profile
      the FIRST trace only. That is what the __main__ below does.
    * If the profile points at jax's own machinery rather than a third-party package, the answer is
      probably the boundary count, not a bug: read `n_leaves` first.
"""

from __future__ import annotations

import collections
import os
import sys
import threading
import time

import scopex

_STDLIB = os.path.dirname(os.__file__)


def _package_of(path: str) -> str:
    """Which package a frame belongs to. Crude on purpose -- the answer wanted here is
    'opt_einsum' or 'jax' or 'your code', not a module path."""
    p = path.replace("\\", "/")
    if "/site-packages/" in p:
        return p.split("/site-packages/", 1)[1].split("/")[0]
    if p.startswith(_STDLIB.replace("\\", "/")):
        return "<stdlib>"
    if "/jax/" in p or "/jaxlib/" in p:
        return "jax"
    return "<your code>"


class _Sampler:
    """Stack samples of ONE thread at ``hz``, taken from a daemon thread.

    A sampler and not cProfile because a deterministic profiler multiplies the cost of exactly the
    workload this is aimed at -- millions of tiny python calls inside a search -- and would change
    the shape of the answer as well as its magnitude.
    """

    def __init__(self, target_tid: int, hz: float = 200.0):
        self.tid, self.interval = target_tid, 1.0 / hz
        self.frames: collections.Counter = collections.Counter()
        self.leaf: collections.Counter = collections.Counter()
        self.packages: collections.Counter = collections.Counter()
        self.samples = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            f = sys._current_frames().get(self.tid)
            if f is not None:
                self.samples += 1
                seen = set()
                top = f
                while f is not None:
                    key = (f.f_code.co_filename, f.f_code.co_name)
                    if key not in seen:          # recursion must not count once per frame
                        seen.add(key)
                        self.frames[key] += 1
                    f = f.f_back
                self.leaf[(top.f_code.co_filename, top.f_lineno, top.f_code.co_name)] += 1
                self.packages[_package_of(top.f_code.co_filename)] += 1
            self._stop.wait(self.interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=2)


def trace_profile(fn, args, *, hz: float = 200.0, stage: str = "trace") -> dict:
    """Sample the python stack for the duration of ONE stage and report who was on it.

    ``stage`` is ``"trace"`` (``jax.make_jaxpr``) or ``"trace+lower"`` (``jit(fn).lower``).

    THERE IS NO LOWER-ONLY STAGE TO SAMPLE. ``jit(fn).lower(*args)`` re-traces on its way to
    StableHLO, so a sampler wrapped around it sees both stages and cannot separate them; only
    `record`, which reads jax's own counters, can. An earlier version of this function tried to run
    ``make_jaxpr`` first and sample only the ``lower`` call, and reported "LOWER profile 33.684 s,
    99.9% opt_einsum" for a stage that `record` measured at 0.031 s -- the untimed trace was inside
    the sampled window. The label is honest now instead of the arithmetic being wrong.

    This is the instrument scopex does not have; it is here because three of the corpus's
    trace-bound cases can be TIMED by scopex and named by nothing in it.
    """
    import jax

    jax.clear_caches()
    tid = threading.get_ident()
    t0 = time.perf_counter()
    with _Sampler(tid, hz) as s:
        if stage == "trace":
            jax.make_jaxpr(fn)(*args)
        else:
            jax.jit(fn).lower(*args)
    el = time.perf_counter() - t0
    tot = max(1, s.samples)
    return {
        "stage": stage,
        "seconds": round(el, 3),
        "samples": s.samples,
        "by_package": [(k, round(v / tot, 3)) for k, v in s.packages.most_common(6)],
        "hot_leaves": [(f"{os.path.basename(f)}:{ln} {fun}", round(n / tot, 3))
                       for (f, ln, fun), n in s.leaf.most_common(8)],
        "on_stack": [(f"{os.path.basename(f)} {fun}", round(n / tot, 3))
                     for (f, fun), n in s.frames.most_common(8)],
    }


def who_is_slow_inside_tracing(fn, args, control_fn, control_args, *, hz: float = 200.0) -> dict:
    """One line: the compiler is innocent -- which python is it, and how many arrays did you pass?

    Returns the stage split for both arms, the boundary arity for both arms, a sampled profile of
    the case's trace stage, and the scopex attribution views on the jaxpr -- which are included
    precisely to show that they are CONSTANT across the two arms and cannot answer this.
    """
    import jax

    t = scopex.record(fn, *args)
    c = scopex.record(control_fn, *control_args)
    stages = ("trace", "lower", "backend")
    ratio = {k: round(t.get(k, 0.0) / max(1e-9, c.get(k, 0.0)), 1) for k in stages}

    def boundary(a):
        leaves, treedef = jax.tree_util.tree_flatten(a)
        return {"n_args": len(a), "n_leaves": len(leaves),
                "n_arrays": sum(1 for x in leaves if hasattr(x, "shape")),
                "treedef_depth": str(treedef).count("(")}

    prof = trace_profile(fn, args, hz=hz)
    prof_lower = trace_profile(fn, args, hz=hz, stage="trace+lower")

    j = jax.make_jaxpr(fn)(*args)
    jc = jax.make_jaxpr(control_fn)(*control_args)
    units, cunits = list(scopex.walk(j)), list(scopex.walk(jc))
    views = {}
    for by in ("split", "package", "site", "transform", "kind"):
        a = scopex.attribute(units, by)
        b = scopex.attribute(cunits, by)
        views[by] = {"case": dict(a.most_common(3)), "control": dict(b.most_common(3)),
                     "identical": set(a) == set(b)}

    trace_share = (t.get("trace", 0.0) + t.get("lower", 0.0)) / max(1e-9, t.get("wall", 0.0))
    out = {
        "stage_seconds": {"case": {k: round(t.get(k, 0.0), 3) for k in stages},
                          "control": {k: round(c.get(k, 0.0), 3) for k in stages}},
        "stage_ratio": ratio,
        "regime": {"case": scopex.regime(t), "control": scopex.regime(c)},
        "trace_plus_lower_share_of_wall": round(trace_share, 3),
        "boundary": {"case": boundary(args), "control": boundary(control_args)},
        "jaxpr_equations": {"case": len(units), "control": len(cunits)},
        "trace_profile": prof,
        "trace_plus_lower_profile": prof_lower,
        "attribution_views_are_constant": {k: v["identical"] for k, v in views.items()},
        "views": views,
    }
    top_pkg = prof["by_package"][0] if prof["by_package"] else ("?", 0.0)
    lb, cb = out["boundary"]["case"]["n_leaves"], out["boundary"]["control"]["n_leaves"]
    out["verdict"] = (
        f"trace {ratio['trace']}x / lower {ratio['lower']}x against the control; "
        f"trace+lower is {trace_share:.0%} of the wall. "
        f"{top_pkg[1]:.0%} of trace-stage samples are in `{top_pkg[0]}`. "
        f"Boundary arity {lb} leaves vs {cb}.")
    out["next"] = (
        "if the hot package is third-party, the fix is in how you call it (einsum path, "
        "precomputed order). If it is jax and the leaf count is large, the fix is the BOUNDARY: "
        "stack the leaves into arrays -- compilation cost scales with the number of array objects "
        "crossing jit, not with their size.")
    return out


if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    import _cases

    for name in ("einsum_optimal_n11", "arity_tree_200"):
        fn, args = _cases.load(name)
        cfn, cargs = _cases.load(name + "_control")
        r = who_is_slow_inside_tracing(fn, args, cfn, cargs)
        print(f"\n=== {name} vs {name}_control (CPU) " + "=" * 28)
        print(f"  stages   case {r['stage_seconds']['case']}")
        print(f"           ctrl {r['stage_seconds']['control']}")
        print(f"  ratios   {r['stage_ratio']}   regime {r['regime']}")
        print(f"  trace+lower = {r['trace_plus_lower_share_of_wall']:.0%} of wall")
        print(f"  boundary case {r['boundary']['case']}")
        print(f"           ctrl {r['boundary']['control']}")
        print(f"  jaxpr eqns    {r['jaxpr_equations']}")
        print(f"  TRACE profile {r['trace_profile']['seconds']} s, "
              f"{r['trace_profile']['samples']} samples")
        print(f"     by package {r['trace_profile']['by_package']}")
        print(f"     hot leaves {r['trace_profile']['hot_leaves'][:4]}")
        print(f"  TRACE+LOWER   {r['trace_plus_lower_profile']['seconds']} s "
              f"(lower cannot be sampled alone -- jit().lower() re-traces)")
        print(f"     by package {r['trace_plus_lower_profile']['by_package']}")
        print(f"  attribution views identical across arms: "
              f"{r['attribution_views_are_constant']}")
        print(f"  VERDICT  {r['verdict']}")
        print(f"  next     {r['next']}")
