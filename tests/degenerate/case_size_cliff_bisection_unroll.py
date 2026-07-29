"""jax#10621 -- m=95 does not finish, m=96 compiles in ~10 s. A compiler heuristic that flips at a
size boundary, non-monotonically.

    https://github.com/jax-ml/jax/issues/10621    "scan unroll + nested scan + vmap cause dead loop?"

A tridiagonal-bisection eigenvalue solver. An inner `lax.scan(unroll=48)` over the m Sturm-sequence
counts, sitting inside an outer `lax.scan(length=24, unroll=3)` doing the bisection, the whole
thing under `jax.vmap` twice (once over the m eigenvalue indices inside, once over the batch of n
matrices outside). The reporter's verdict, verbatim:

    "If n=1 or m>=96 or smaller unroll for scan in count, everything is fine."
    "It might be not a dead loop, but at least takes much longer time (>5min) than
     m>=96 case (~10s)."

and XLA's own `slow_operation_alarm` fires on the bad arm, which is a second opinion from the
compiler rather than an inference from a stopwatch.

WHY THIS CASE EARNS ITS PLACE, AND IT IS NOT THE UNROLLING. The corpus already has unrolled chains;
"I unrolled 48 iterations and the program got big" is not news. What is new is that the blowup is
NOT MONOTONE IN PROBLEM SIZE. m=95 is catastrophic and m=96 -- one column wider, strictly more
work, strictly more FLOPs, strictly more eigenvalues to find -- is fine. Monotone blowups are
findable without any tool: sweep the size, watch the number climb, point at the thing that got
bigger. That is the standard diagnostic, and this case defeats it, because the sweep says the
program got FASTER as it got bigger. Non-monotonicity is also positive evidence about mechanism: a
cost that jumps at a boundary and comes back down is a tiling / vectorization / fusion heuristic
selecting a different code path, not a program that merely grew. No amount of reading the Python
explains it, because the Python is identical modulo one integer.

The reporter also establishes WHERE the cost is not: `jax.make_jaxpr` completes fine on the bad
arm ("Tracing Ok" in the original script). Tracing and jaxpr construction are exonerated; whatever
is happening is downstream, in lowering or in XLA. That is a free bisection the issue hands us and
this file preserves it -- the harness's separate `lower_s` and `compile_s` should show the damage
landing entirely in `compile_s`, and if it does not, that is a finding about jax 0.10.2.

WHAT THE CONTROLS ISOLATE. Three, all from the reporter, all against a byte-identical program:

  (a) `bisect_m95_control` is m=96. One integer, one column, ~1% more arithmetic. This is the
      sharpest control in the whole corpus if it survives -- there is no structural difference at
      all between the arms, not one op, not one nesting level.
  (b) `bisect_m95_n1` drops the outer vmap axis from 2 to 1. Same inner program, one fewer batch
      dimension. Tests whether the outer vmap is required to trigger it.
  (c) `bisect_m95_u8 .. _u64` vary only the inner scan's `unroll`. This is the mechanistically
      informative axis: `unroll` is the one knob that changes how much straight-line code the
      vectorizer is handed without changing the mathematics at all.

HIGH RISK, STATED UP FRONT. Filed 2022 against jax 0.3.x, and a heuristic threshold is exactly the
kind of thing that moves between releases -- the constant that put the cliff at 95 in 2022 has no
obligation to be the same constant in 0.10.2, and the cliff may have been fixed, or moved, or
inverted. THIS IS WHY THE FILE IS A SWEEP AND NOT A POINT. If m=95 is fast, the case is not dead
until the whole grid has been walked: m across 64..256 including both sides of every power of two
and of 95/96 itself, and unroll across 3..64. What matters is whether a cliff exists ANYWHERE in
that grid, not whether it is still at 95. A clean monotone surface across the entire sweep is a
publishable negative and would say the heuristic was made size-independent.

READING THE RESULT. The harness's `_control` convention pairs `bisect_m95` with `bisect_m95_control`
(= m=96) and that pair is the headline. Every other entry is a point on a curve and must be read as
one: the m arms are one line, the unroll arms another. A single entry's YES/no verdict means very
little here -- a CLIFF between two ADJACENT entries is the entire finding. Note that m=96 appears
only as `bisect_m95_control`, deliberately, so it is measured once and serves both roles.

FLOAT32 IS MANDATORY, not a performance choice. `signbit` is implemented as
`bitcast_convert_type(x, int32) >> 31`, which is only meaningful for a 32-bit float -- bitcasting
f64 to int32 yields a trailing dimension of 2 and the algorithm falls apart. The reporter says so
("only works for fp32 now"). The harness enables x64 globally, so every array and every scalar
constant in this file is pinned to float32/int32 explicitly. That pinning is the only substantive
change from the 2022 source; the loop structure, the `unroll` values, the `length=24`, the
`jnp.pad(b2, (1, 0))` closing over the enclosing `b2`, and the two levels of vmap are verbatim.

RUNTIME IS TINY BY CONSTRUCTION -- 24 bisection steps over m counts on an (n, m) matrix, a few
hundred KB at the largest size here -- so `compile/runtime` should be enormous on any arm that
compiles slowly at all, and the harness's ratio test is meaningful even on the sweep entries that
have no paired control.

VERIFIED ON CPU before committing (jax 0.10.2, x64 on, trace and lower only -- no execution, no
GPU). All 17 arms trace under the double vmap with x64 enabled and every intermediate holds at
float32 through the bitcast. The m=95 and m=96 jaxprs are structurally identical -- 24 equations
each, same primitives, same nesting -- differing only in the operands' trailing dimension:

    bisect_m95           in float32[2,95], float32[2,94]   24 eqns   out float32[2,95]
    bisect_m95_control   in float32[2,96], float32[2,95]   24 eqns   out float32[2,96]

There is nothing in the program for a source-level profiler to point at, which is the whole
difficulty. Lowering is cheap and small on the arms that are safe to lower here: 162 MLIR lines at
m=95/unroll=3 (0.65 s) and 457 at m=64/unroll=48 (0.26 s). Combined with the reporter's "Tracing
Ok" that puts everything upstream of XLA in the clear, and it means the file's headline arms are
worth lowering separately from compiling -- a large `lower_s` on the bad arm would be a genuine
surprise and would refute the reading above.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

_F32 = np.float32
_I32 = np.int32

# Bisection steps in the outer scan, and its unroll.  Both verbatim from the issue.
OUTER_LENGTH = 24
OUTER_UNROLL = 3


def _signbit(x):
    # Verbatim from the issue.  Only meaningful for float32 -- see the module docstring.
    return lax.shift_right_logical(lax.bitcast_convert_type(x, jnp.int32), _I32(31))


def _eigvalsh_tridiagonal_bisection(a, b, unroll):
    """a: (m,) diagonal, b: (m-1,) sub-diagonal.  Verbatim apart from float32 pinning."""
    b2 = b ** 2

    def count(x):
        # Thanks to IEEE-754, no pivmin is needed: https://epubs.siam.org/doi/epdf/10.1137/050641624
        # (see also "Faster numerical algorithms via exception handling").  Assumes b2i != 0, i > 0.
        def scan_f(carry, data):
            q, c = carry
            ai, b2i = data
            q = (ai - x) - (b2i * lax.reciprocal(q))
            return (q, lax.add(c, _signbit(q))), None

        # `b2` here closes over the enclosing scope, as written in the issue.
        return lax.scan(scan_f, (_F32(1.0), _I32(0)),
                        (a, jnp.pad(b2, (1, 0))), unroll=unroll)[0][1]

    b_abs = lax.abs(b)
    r = jnp.pad(b_abs, (1, 0)) + jnp.pad(b_abs, (0, 1))
    emax = jnp.max(a + r)
    emin = jnp.min(a - r)
    norm = lax.max(lax.abs(emax), lax.abs(emin))
    m = a.size
    upper0 = emax + norm * _F32(3e-7) * _F32(m)
    lower0 = emin - norm * _F32(3e-7) * _F32(m)

    @jax.vmap
    def bisection(cnt):
        def step(carry, _):
            lower, upper = carry
            mid = (lower + upper) / _F32(2.0)
            pred = count(mid) <= cnt
            lower = lax.select(pred, mid, lower)
            upper = lax.select(pred, upper, mid)
            return (lower, upper), None

        # only works for fp32 now
        lower, upper = lax.scan(step, (lower0, upper0), None,
                                length=OUTER_LENGTH, unroll=OUTER_UNROLL)[0]
        return (lower + upper) / _F32(2.0)

    return bisection(jnp.arange(m, dtype=jnp.int32))


def _prog(a, b, unroll):
    return jax.vmap(functools.partial(_eigvalsh_tridiagonal_bisection, unroll=unroll))(a, b)


# ------------------------------------------------------------------------------------------------
# Inputs.  numpy at module scope: importing this file touches no device.  The reporter's
# gamma-distributed sub-diagonal with shape parameter arange(m-1, 0, -1) is reproduced; the values
# cannot affect compile time but they keep the bisection doing representative work at runtime.
# ------------------------------------------------------------------------------------------------
def _args(n: int, m: int):
    rng = np.random.default_rng(10621)
    a = rng.standard_normal((n, m)).astype(_F32)
    b2 = rng.gamma(np.arange(m - 1, 0, -1, dtype=np.float64), size=(n, m - 1))
    b = np.sqrt(b2).astype(_F32)
    return (a, b)


def _mk(n: int, m: int, unroll: int, note: str):
    return functools.partial(_prog, unroll=unroll), _args(n, m), note


_U = 48   # the reported inner unroll
_N = 2    # the reported outer batch (vmap) size

CASES = {
    # --- the reported point, and control (a): one column wider ---------------------------------
    "bisect_m95": _mk(_N, 95, _U,
                      "jax#10621 as reported: n=2, m=95, inner scan unroll=48 -- >5 min / hang"),
    "bisect_m95_control": _mk(_N, 96, _U,
                              "control (a): m=96, ONE column wider and strictly more work -- "
                              "reported to compile in ~10 s. The sharpest control in the corpus"),

    # --- control (b): drop one vmap axis -------------------------------------------------------
    "bisect_m95_n1": _mk(1, 95, _U,
                         "control (b): n=1, outer vmap axis dropped; reporter says this is fine"),

    # --- control (c): the unroll axis, the mechanistically informative one ---------------------
    "bisect_m95_u3": _mk(_N, 95, 3, "unroll sweep at m=95: inner unroll=3 (matches outer)"),
    "bisect_m95_u8": _mk(_N, 95, 8, "unroll sweep at m=95: inner unroll=8"),
    "bisect_m95_u16": _mk(_N, 95, 16, "unroll sweep at m=95: inner unroll=16"),
    "bisect_m95_u24": _mk(_N, 95, 24, "unroll sweep at m=95: inner unroll=24"),
    "bisect_m95_u32": _mk(_N, 95, 32, "unroll sweep at m=95: inner unroll=32"),
    "bisect_m95_u64": _mk(_N, 95, 64, "unroll sweep at m=95: inner unroll=64 (> m, full unroll)"),

    # --- the m axis at the reported unroll.  Looking for a CLIFF between adjacent points, not a
    # --- trend.  Both sides of 64/128/256 and of the reported 95/96 boundary are covered; m=96 is
    # --- deliberately absent because it is bisect_m95_control above.
    "bisect_m64": _mk(_N, 64, _U, "m sweep, unroll=48: m=64 (power of two)"),
    "bisect_m65": _mk(_N, 65, _U, "m sweep, unroll=48: m=65 (just past a power of two)"),
    "bisect_m80": _mk(_N, 80, _U, "m sweep, unroll=48: m=80"),
    "bisect_m94": _mk(_N, 94, _U, "m sweep, unroll=48: m=94 (just BELOW the reported bad point)"),
    "bisect_m112": _mk(_N, 112, _U, "m sweep, unroll=48: m=112"),
    "bisect_m128": _mk(_N, 128, _U, "m sweep, unroll=48: m=128 (power of two)"),
    "bisect_m192": _mk(_N, 192, _U, "m sweep, unroll=48: m=192"),
    "bisect_m256": _mk(_N, 256, _U, "m sweep, unroll=48: m=256 (power of two)"),
}
