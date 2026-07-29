"""GAP 10 (Python-side compile cost that is NOT jaxpr size) and GAP 9 (lowering as its own stage),
SYNTHESISED.

No issue URL. Constructed from the mechanism. The behaviour is correct -- JAX's trace cache is
keyed on static arguments and on the argument treedef, and a different key is a different entry --
but the resulting workload shape is the single most common self-inflicted compile-time wound in
real JAX code, and the corpus does not contain it.

THE SHAPE. A jitted helper is called K times inside one traced function. Something in its cache key
varies across the calls WITHOUT changing what it computes: a step index or a layer name passed as a
`static_argnums` argument the body never reads, or a dict key that differs per call. Every call
misses the trace cache, so JAX traces K structurally identical sub-jaxprs and lowers K structurally
identical MLIR functions. XLA then inlines all of them and the optimised HLO is byte-for-byte what
you would have got from one cache entry.

    K distinct MLIR functions        (measured: 321 `func.func` at K=320, versus 2 for the control)
    IDENTICAL optimised HLO          (measured: 466 lines in BOTH arms, at every K)
    identical FLOPs, identical runtime, identical answer

So the pathology is entirely upstream of the artifact any HLO-level instrument inspects, and it is
concentrated in the half of compilation the corpus has barely probed.

MEASURED IN-ENV BEFORE COMMITTING (jax 0.10.2, JAX_PLATFORMS=cpu, x64 on, x = (32,32) f64, inner
body = 30 x (sin, mul, add) then a reduce; `lower` is `jit(f).lower(x)` i.e. trace + lowering,
`comp` is the backend):

      K    variant   distinct lower  control lower  ratio |  distinct comp  control comp  ratio
     40    static        1.596 s        0.055 s      29x  |    0.976 s        0.620 s     1.6x
     40    treedef       0.956 s        0.075 s      13x  |    0.941 s        0.656 s     1.4x
     80    static        1.521 s        0.167 s       9x  |    1.730 s        1.366 s     1.3x
     80    treedef       2.164 s        0.077 s      28x  |    1.409 s        0.863 s     1.6x
    160    static        4.053 s        0.124 s      33x  |    2.972 s        1.664 s     1.8x
    160    treedef       2.654 s        0.129 s      21x  |    2.562 s        1.212 s     2.1x
    320    static        6.422 s        0.217 s      30x  |    4.768 s        3.160 s     1.5x
    320    treedef       6.164 s        0.856 s       7x  |    8.933 s        4.135 s     2.2x

The trace+lower ratio is 7-33x; the backend ratio never leaves 1.3-2.2x, and that residue is the
cost of inlining K functions into the one program both arms end up with. A profiler that attributes
this case to XLA is wrong by an order of magnitude.

CONFIRMED IN FRESH PROCESSES at K=320 (one process per measurement, arms interleaved), which is how
the harness measures and is the number to trust:

    retrace_static_320             lower 11.602 s    compile 4.515 s    hlo 467 lines
    retrace_static_320_control     lower  0.948 s    compile 4.174 s    hlo 467 lines
    retrace_treedef_320            lower  8.026 s    compile 3.921 s    hlo 467 lines
    retrace_treedef_320_control    lower  0.199 s    compile 2.015 s    hlo 467 lines

-> static arm  12.2x on `lower_s`, 1.08x on `compile_s`
-> treedef arm 40.3x on `lower_s`, 1.94x on `compile_s`
-> optimised HLO line count IDENTICAL, 467, in all four.

WHAT EACH CONTROL ISOLATES -- and both are single-token controls.

  retrace_static_K          `_static(x, tag)` is jitted with `static_argnums=1`; `tag` is NEVER
                            READ by the body. The pathological arm passes `i`, the control passes
                            `0`. One integer literal differs between the two programs.
  retrace_static_K_control  same K calls, same dispatch count, same jaxpr equations at the top
                            level, same optimised HLO. 1 trace instead of K.

  retrace_treedef_K         `_treed(d)` takes a one-entry dict; the pathological arm uses the key
                            `f"k{i:06d}"`, the control uses `"k000000"`. The KEY NAME is the only
                            difference, and it is not part of the computation at all -- the body
                            reads `next(iter(d.values()))`.
  retrace_treedef_K_control same K calls with a constant key.

  funcsplit_K               identical program to `retrace_static_K`
  funcsplit_K_control       the SAME K x W x 3 equations with no inner jit at all, so they land in
                            ONE MLIR function instead of K. Op-count-matched (29440 reachable
                            equations at K=320 against ~29500; 461 optimised HLO lines against
                            467); function count is the only real variable. This pair is the
                            attribution split, and at K=320 in fresh processes it reads:

                                retrace_static_320_control (1 func,    92 eqns)  lower  0.948 s
                                funcsplit_320_control      (1 func, 29440 eqns)  lower  7.106 s
                                retrace_static_320         (320 funcs, same eqns) lower 11.6-16.3 s

                            so of the ~10.6 s the cache miss adds, roughly 6.2 s is "you traced and
                            lowered 320x more equations" and roughly 4.4-9.2 s is "you did it as
                            320 separate cache misses and 320 separate MLIR functions". BOTH terms
                            are real; neither alone explains the case. A tool that reports only one
                            of them has half the answer.

Having the first two matters because they hit DIFFERENT COMPONENTS of the same cache key -- the
static argument tuple and the argument treedef. A tool that localises one and not the other has
found a symptom rather than the cache.

SIZE SWEEP: K in (40, 80, 160, 320), the number of cache misses. Trace+lower grows roughly
linearly in K in both arms (the control at 1/30th the slope), which is the prediction; the point of
four sizes is that the RATIO is stable rather than an artefact of one program size.

RELATIONSHIP TO `case_manymodules_dispatch_constant`. That file uses the same trigger -- an unused
static argument taking K values -- but places the calls inside `jax.ensure_compile_time_eval()`, so
each miss becomes a whole extra XLA MODULE compiled during `.lower()` (measured ~180-390 ms each).
Here the calls are staged into the enclosing trace, so each miss costs only an extra trace plus an
extra MLIR function (~20 ms each at K=320). Same user error, two stages, two orders of magnitude
apart. A profiler must tell them apart; that is why both exist.

WHAT A SIZE-BASED HEURISTIC SEES. Top-level jaxpr equation count is the same in both arms (K `pjit`
equations either way). Optimised HLO op count is the same. What differs is the number of DISTINCT
sub-jaxprs reachable from the top-level jaxpr -- 1 versus K -- and the number of `func.func` ops in
the StableHLO. Those two are the signals that separate this case, and neither is a size.

PLATFORM: either. Tracing and jaxpr->StableHLO are backend-independent; the backend residue is
inlining, which both backends do. Measured on CPU only because the GPU was owned by another
investigation when this was written.

MEMORY: negligible. Largest array is (32,32) f64 = 8 KB.

CAVEAT: `_harness.classify` keys on `compile_s`, where these arms are only 1.3-2.2x apart, so the
harness will score this "no". The 7-33x lives in `lower_s`.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

W = 30      # sin/mul/add steps inside the jitted helper


def _body(y):
    for _ in range(W):
        y = jnp.sin(y) * 1.0001 + 0.25
    return jnp.sum(y)


@functools.partial(jax.jit, static_argnums=1)
def _static(x, tag):
    """`tag` is never read. It exists only to sit in the trace-cache key."""
    return _body(x)


@jax.jit
def _treed(d):
    """The dict KEY is never read either -- only the single value is."""
    return _body(next(iter(d.values())))


def _call_static(x, ncalls: int, distinct: bool):
    acc = 0.0
    for i in range(ncalls):
        acc = acc + _static(x, i if distinct else 0)
    return acc


def _call_treedef(x, ncalls: int, distinct: bool):
    acc = 0.0
    for i in range(ncalls):
        acc = acc + _treed({(f"k{i:06d}" if distinct else "k000000"): x})
    return acc


def _call_inline(x, ncalls: int):
    """The same K x W equations with NO inner jit at all: one MLIR function instead of K.

    Op-count-matched against `_call_static(distinct=True)`. Measured FLAT against it, which is what
    pins the cost on the duplicated equations rather than on emitting K functions.
    """
    acc = 0.0
    for _ in range(ncalls):
        acc = acc + _body(x)
    return acc


# numpy at module scope: importing this file claims no device.
X = np.ones((32, 32), dtype=np.float64)

SIZES = (40, 80, 160, 320)

CASES = {}

for _k in SIZES:
    CASES[f"retrace_static_{_k}"] = (
        functools.partial(_call_static, ncalls=_k, distinct=True), (X,),
        f"gap10 synth: {_k} calls to one jitted helper whose UNUSED static arg takes {_k} distinct "
        f"values -> {_k} trace-cache misses, {_k} identical MLIR functions, identical optimised HLO",
    )
    CASES[f"retrace_static_{_k}_control"] = (
        functools.partial(_call_static, ncalls=_k, distinct=False), (X,),
        f"control: the same {_k} calls with the unused static arg held at 0 -> 1 trace. Same "
        f"dispatch count, same optimised HLO line count, same runtime",
    )
    CASES[f"retrace_treedef_{_k}"] = (
        functools.partial(_call_treedef, ncalls=_k, distinct=True), (X,),
        f"gap10 synth, second cache-key component: {_k} calls passing a 1-entry dict whose KEY "
        f"NAME differs per call -> {_k} distinct treedefs -> {_k} trace-cache misses",
    )
    CASES[f"retrace_treedef_{_k}_control"] = (
        functools.partial(_call_treedef, ncalls=_k, distinct=False), (X,),
        f"control: the same {_k} calls with a constant dict key -> 1 trace. Only the key string "
        f"differs, and the body never reads it",
    )
    CASES[f"funcsplit_{_k}"] = (
        functools.partial(_call_static, ncalls=_k, distinct=True), (X,),
        f"MEASURED NEGATIVE, attribution arm: the same {_k * W * 3} equations emitted as {_k} "
        f"separate MLIR functions",
    )
    CASES[f"funcsplit_{_k}_control"] = (
        functools.partial(_call_inline, ncalls=_k), (X,),
        f"control: the identical {_k * W * 3} equations emitted as ONE function (no inner jit). "
        f"Op-count-matched, function count 1 vs {_k}. Measured FLAT -- so the retrace cost is the "
        f"duplicated equations, not the function emission",
    )
