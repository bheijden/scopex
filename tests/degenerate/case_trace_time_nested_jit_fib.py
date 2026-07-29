"""jax#22385 -- doubly-recursive jit-with-static_argnums costs phi^t; single recursion costs t.

    https://github.com/jax-ml/jax/issues/22385         OPEN (filed 2024-07-11, jax 0.4.30)

REPORTED (jax 0.4.30, macOS arm64). `fib(t)` under `functools.partial(jax.jit, static_argnums=(0,))`
recursing as `fib(t-1) + fib(t-2)`. The reporter's print statements fire exactly once per distinct
`t`, so Python-level tracing is LINEAR -- each `t` is traced once and the jit cache is doing its
job. Yet wall time grows exponentially. The self-diagnosing pair the reporter gives is:

    fib.lower(t).as_text()              grows LINEARLY    (StableHLO has subroutines: `call @fib_0`)
    jax.make_jaxpr(fib, static_argnums=(0,))(t)   grows EXPONENTIALLY

i.e. the diamond-shaped call DAG (fib(t-1) and fib(t-2) both reach fib(t-3)) is expanded into a
TREE of size O(phi^t) somewhere below the jaxpr. @abadams put the wall clock in `_jaxpr_forwarding`.

WHAT WE MEASURED ON 0.10.2 -- THE PATHOLOGY HAS MOVED, WHICH IS THE REASON TO KEEP THIS CASE.
An indicative CPU probe (JAX_PLATFORMS=cpu, single process, not a benchmark run -- the harness owns
the real numbers) says the jaxpr side is now FIXED and the exponent has migrated into XLA:

    t            8      12      16      18      20
    lower_s   0.17    0.12    0.21    0.22    0.26     <- FLAT. trace+lower is no longer the cost.
    compile_s 0.10    0.32    1.13    2.12    5.37     <- x2.53 per +2 in t, i.e. phi^t exactly.
    jaxpr chars 1119  1705    2324    2632    2940     <- LINEAR. make_jaxpr is ~10 ms throughout.
    hlo chars  1748   2582    3426    3848    4270     <- LINEAR. the `call @fib_0` subroutines
                                                          survive all the way to StableHLO.
    control (single recursion) compile_s stays 0.07-0.21 at every t.

So on 0.10.2 the linear-IR / exponential-cost discrepancy is REAL BUT RELOCATED: the exponential is
now XLA inlining the HLO call graph, not JAX flattening the jaxpr. That makes this case a sharper
scopex test than when it was filed, because a tool that attributes by IR SIZE sees nothing wrong at
any stage -- every artifact it can measure is linear -- while the cost is quadratic-plus in a graph
whose textual form never grows. Attribution here has to come from the call structure, not the byte
count.

WHAT THE CONTROL ISOLATES. One token: `fib(t-1) + fib(t-2)` becomes `fib(t-1) + 1`. Identical
decorator, identical static_argnums, identical recursion DEPTH, identical number of distinct traces,
identical number of nested pjit calls. The only difference is that the call DAG becomes a path
instead of a tree. Everything that differs between the arms is the DAG-vs-tree expansion and
nothing else.

Both arms take NO arguments, exactly as filed. The whole computation is therefore a compile-time
constant, so `runtime_s` is ~dispatch overhead and the compile/runtime ratio is enormous for both
arms; the ratio gate is uninformative here and the case/control gate is the one that matters.

PLATFORM. The probe above is CPU. This is a call-graph-inlining pathology, so it is expected to
appear on both backends, but that is a prediction, not a measurement -- run both.

SIZES. t=26 is ~96 s of compile extrapolated from the CPU probe (x2.618 per +2); t=28 would be
~250 s and is left out so one arm cannot eat the harness timeout. Deep Python recursion: 24 nested
`fib` frames each carry a jit-dispatch frame stack, so the module raises the recursion limit at
import (raising it only, never lowering).
"""

from __future__ import annotations

import functools
import sys

import jax

# 24 levels of `fib` x ~50 frames of jit dispatch per level overruns CPython's default 1000-frame
# limit before t=26 is reached. Raise, never lower -- and only if it is currently lower.
if sys.getrecursionlimit() < 20_000:
    sys.setrecursionlimit(20_000)


def _build_fib(double: bool):
    """A FRESH jitted `fib` per case entry, so no entry warms another entry's trace cache."""

    @functools.partial(jax.jit, static_argnums=(0,))
    def fib(t):
        if t <= 2:
            return 1
        # double: the call DAG is a diamond, expanded as a tree -> O(phi^t)
        # single: the call DAG is a path                        -> O(t)
        return fib(t - 1) + fib(t - 2) if double else fib(t - 1) + 1

    return fib


def _mk(double: bool, t: int):
    fib = _build_fib(double)

    def fn():
        return fib(t)

    kind = "double recursion fib(t-1)+fib(t-2)" if double else "control: single recursion fib(t-1)+1"
    return fn, (), f"jax#22385 t={t}, {kind}"


# The claim is about SCALING -- one point cannot separate "exponential in t" from "big program".
# t is the only thing that varies within an arm, and `double` is the only thing that varies across.
CASES = {}
for _t in (16, 20, 22, 24, 26):
    CASES[f"jitfib_t{_t}"] = _mk(True, _t)
    CASES[f"jitfib_t{_t}_control"] = _mk(False, _t)
