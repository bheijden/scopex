"""jax#4667 -- compile cost is set by the NUMBER of arrays crossing the jit boundary, and most of
it is spent BEFORE the backend compiler ever runs.

    https://github.com/jax-ml/jax/issues/4667      opened by mfkasim1-style user, answered by jakevdp

The reporter wrote the same discrete dynamical system three ways and compared the pre-optimisation
HLO he got out of each:

    state as a dict of 100 leaves                     up to 26 MB of HLO
    state as one stacked (horizon, dim) array         up to 50 KB of HLO          (520x smaller)
    arrays at the boundary, dict rebuilt INSIDE       up to 1.1 MB of HLO         (in between)

Identical dynamics, identical total FLOPs, identical answer. jakevdp's answer is the whole
mechanism and is worth quoting because it is a standing statement about JAX, not a bug triage:

    "the issue is not with pytrees per se, but rather with defining hundreds of array arguments,
    and compiling functions which take hundreds of array arguments. ... compilation costs should
    not scale with the size of the individual arrays being operated on, but we do expect them to
    scale with the number of array objects being passed to the function. In the best case, you
    might achieve linear scaling - but I suspect in reality you'll see closer to quadratic."

WHY THIS CASE EARNS ITS PLACE. Not because "big pytrees are slow" -- that much is folklore. Because
of WHERE the seconds land. Measured in-env before this file was written (jax 0.10.2, CPU backend --
the GPU was busy, and the trace/lower half of this is backend-independent by construction since it
happens entirely in Python and MLIR), at nleaves=50, dim=100, 30 steps, under jax.grad:

                          trace+lower      backend compile      StableHLO lines
    dict of 50 leaves       10.265 s            6.108 s              21105
    one (50,100) array       0.237 s            0.411 s                427
    ratio                     43.3x              14.9x                 49x

The pathology is 1.7x LARGER in trace+lower than in backend compile, and trace+lower is the half
that no compiler-side instrument sees. A profiler wrapped around `backend_compile` reports 6.1 s
and silently drops 10.3 s on the floor; a user staring at that profile concludes XLA is slow and
starts tuning XLA flags. Every one of the 53 cases in the corpus so far is dominated by the backend
stage, so nothing has yet exercised scopex's route 2c ("lower dominates" -- distinct xla_metadata
value count, computation count, shared-computation census). This case is that route's test input,
and it is the reason the harness records `lower_s` separately from `compile_s`: on this case those
two numbers tell different stories and only reading both gets the attribution right.

Note also that 50 leaves is 2x below the issue's 100 and 160x below the "hundreds" jakevdp is
talking about. The effect does not need an extreme program.

WHAT THE CONTROL ISOLATES. The container, and only the container. Same 30 steps, same 0.999/1e-3
affine update, same sum-of-squares objective, same jax.grad, same total element count
(nleaves x dim), same dtype. `arity_tree_N` carries the state as a dict of N arrays of shape
(dim,); `arity_tree_N_control` carries it as one array of shape (N, dim). Nothing else differs.
Neither arm's runtime should move much -- the FLOPs are identical -- so whatever separates them is
pure compiler-facing structure.

THE MIDDLE ARM is the issue's own third variant and it is what makes this more than a two-point
comparison. `arity_mixed_N` passes ARRAYS across the jit boundary (arity 2) but rebuilds the dict
inside the loop body on every step, so the jaxpr still contains N separate leaves of work while the
boundary sees two arrays. It bisects jakevdp's claim:

    if mixed is FAST     -> the cost is arity AT THE BOUNDARY (pytree flattening, per-argument
                            lowering bookkeeping, avals, donation/layout logic per leaf)
    if mixed is SLOW     -> the cost is the leaf count INSIDE (equation count in the jaxpr and
                            op count in the emitted StableHLO), and the boundary is incidental

The issue's own HLO sizes (26 MB / 1.1 MB / 50 KB) say the honest answer is "both, unequally":
mixed is 22x better than the dict but still 22x worse than the array. Which of the two halves
dominates on jax 0.10.2 is not something the 2020 issue can tell us, and it is a different answer
for trace+lower than for backend compile. That is precisely the attribution question scopex exists
to answer, so the arm is here.

THE SCAN ARM removes the last confound. In the unrolled arms the dict version's jaxpr is literally
N times longer than the array version's, so a sceptic can say "of course it is slower, it is a
bigger program". `arity_scan_100` wraps the identical 30 steps in `lax.scan(length=30)`, whose body
is traced ONCE. Program length is then O(1) in steps for both arms and the only surviving
difference is how many leaves the carry has. If the gap survives the scan, arity is doing the work
on its own; if the gap collapses, the unrolled arms were measuring program length wearing a pytree
costume. Either outcome is a real result about what scopex should point at.

SIZES. nleaves in (25, 50, 100, 200) at fixed dim=100, so the leaf count moves 8x while the total
element count moves 8x with it and the per-leaf size stays put. jakevdp predicts "closer to
quadratic"; four points on a log-log line is enough to say whether that holds on 0.10.2. The
extrapolation from the measured nleaves=50 point, if quadratic, is roughly 65 s of trace+lower at
nleaves=200 -- inside the harness's 900 s timeout but not by a large margin, and if 200 times out
that is itself the measurement (superquadratic), not a broken case.

MEMORY is a non-issue by design: the largest array in the file is 200 x 100 float64 = 160 KB. This
case consumes compile time, not device memory, which is exactly why it is a clean probe.

ARGS ARE PYTREES, deliberately. The corpus convention is a tuple of concrete arrays; here the
tuple's elements are dicts of arrays, because the number of arrays at the boundary IS the
independent variable. Flattening them into positional arguments would be the same thing to jit --
which is the point jakevdp is making -- but would hide the variable being swept.

VERIFIED ON CPU before committing (jax 0.10.2). All 14 arms trace. At nleaves=25:

    jaxpr equations      tree 4600      mixed 22335 (at n=50)      scan 902      array control 183
    StableHLO lines      tree 10555                                              array control 427

and every arm's gradient is BIT-IDENTICAL to the array control's (max relative difference 0.0 for
tree, mixed and scan alike). The four programs compute exactly the same function; the only thing
that changes is how many array objects the compiler is asked to think about. Note that the
control's 427 MLIR lines are the same 427 measured at nleaves=50 -- the control is flat in N while
the tree arm went 10555 -> 21105 over the same doubling.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

DIM = 100      # elements per leaf; held fixed so only the LEAF COUNT varies
STEPS = 30     # unrolled dynamics steps, as in the in-env verification


def _key(i: int) -> str:
    return f"k{i}"


# ------------------------------------------------------------------------------------------------
# Arm 1: dict of N leaves at the boundary and inside.  The reporter's original.
# ------------------------------------------------------------------------------------------------
def _loss_tree(state, params):
    for _ in range(STEPS):
        state = jax.tree.map(lambda a, b: 0.999 * a + 1e-3 * b, state, params)
    return sum(jnp.sum(v ** 2) for v in jax.tree.leaves(state))


def _grad_tree(state, params):
    return jax.grad(_loss_tree, argnums=1)(state, params)


# ------------------------------------------------------------------------------------------------
# Arm 2 (control): the identical dynamics on one stacked (N, DIM) array.
# ------------------------------------------------------------------------------------------------
def _loss_array(state, params):
    for _ in range(STEPS):
        state = 0.999 * state + 1e-3 * params
    return jnp.sum(state ** 2)


def _grad_array(state, params):
    return jax.grad(_loss_array, argnums=1)(state, params)


# ------------------------------------------------------------------------------------------------
# Arm 3 (middle): arrays at the boundary, dict rebuilt inside the innermost function every step.
# The issue's third variant.  Boundary arity 2, internal leaf count N.
# ------------------------------------------------------------------------------------------------
def _loss_mixed(state, params, nleaves):
    names = [_key(i) for i in range(nleaves)]
    for _ in range(STEPS):
        sd = {n: state[i] for i, n in enumerate(names)}
        pd = {n: params[i] for i, n in enumerate(names)}
        nd = {n: 0.999 * sd[n] + 1e-3 * pd[n] for n in names}
        state = jnp.stack([nd[n] for n in names])
    return jnp.sum(state ** 2)


def _grad_mixed(state, params, nleaves):
    return jax.grad(_loss_mixed, argnums=1)(state, params, nleaves)


# ------------------------------------------------------------------------------------------------
# Arm 4: the same comparison with the step loop rolled into lax.scan, so the jaxpr is O(1) in
# STEPS on BOTH arms and only the carry's leaf count differs.
# ------------------------------------------------------------------------------------------------
def _loss_tree_scan(state, params):
    def body(carry, _):
        return jax.tree.map(lambda a, b: 0.999 * a + 1e-3 * b, carry, params), None

    state = lax.scan(body, state, xs=None, length=STEPS)[0]
    return sum(jnp.sum(v ** 2) for v in jax.tree.leaves(state))


def _grad_tree_scan(state, params):
    return jax.grad(_loss_tree_scan, argnums=1)(state, params)


def _loss_array_scan(state, params):
    def body(carry, _):
        return 0.999 * carry + 1e-3 * params, None

    state = lax.scan(body, state, xs=None, length=STEPS)[0]
    return jnp.sum(state ** 2)


def _grad_array_scan(state, params):
    return jax.grad(_loss_array_scan, argnums=1)(state, params)


# ------------------------------------------------------------------------------------------------
# Inputs.  numpy at module scope: importing this file touches no device.
# ------------------------------------------------------------------------------------------------
def _tree_args(nleaves: int):
    state = {_key(i): np.zeros(DIM, dtype=np.float64) for i in range(nleaves)}
    params = {_key(i): np.ones(DIM, dtype=np.float64) for i in range(nleaves)}
    return (state, params)


def _array_args(nleaves: int):
    return (np.zeros((nleaves, DIM), dtype=np.float64),
            np.ones((nleaves, DIM), dtype=np.float64))


NLEAVES = (25, 50, 100, 200)
# The middle arm at the two SMALL sizes on purpose: rebuilding the dict inside every one of the 30
# steps makes its jaxpr the longest in the file (22335 equations at n=50, ~5x the tree arm's), so
# n=100 would risk the harness timeout for a comparison that n=25 and n=50 already answer.
MIXED_AT = (25, 50)
SCAN_AT = (100,)          # the program-length confound removed

CASES = {}

for _n in NLEAVES:
    CASES[f"arity_tree_{_n}"] = (
        _grad_tree, _tree_args(_n),
        f"jax#4667: grad of {STEPS}-step dynamics, state as a dict of {_n} leaves of shape "
        f"({DIM},) -- {_n} arrays at the jit boundary; expect the cost to land in trace+lower",
    )
    CASES[f"arity_tree_{_n}_control"] = (
        _grad_array, _array_args(_n),
        f"control: bit-for-bit the same dynamics and the same {_n * DIM} elements as one "
        f"({_n}, {DIM}) array -- 1 array at the boundary; only the container differs",
    )

for _n in MIXED_AT:
    CASES[f"arity_mixed_{_n}"] = (
        functools.partial(_grad_mixed, nleaves=_n), _array_args(_n),
        f"middle arm (issue's 3rd variant): arrays at the boundary (arity 2), dict of {_n} leaves "
        f"rebuilt inside every step -- separates boundary arity from internal leaf count",
    )
    CASES[f"arity_mixed_{_n}_control"] = (
        _grad_array, _array_args(_n),
        f"control for the middle arm: same array boundary, no dict anywhere, n={_n}",
    )

for _n in SCAN_AT:
    CASES[f"arity_scan_{_n}"] = (
        _grad_tree_scan, _tree_args(_n),
        f"scan arm: the {STEPS} steps rolled into lax.scan so the jaxpr is O(1) in steps; carry is "
        f"a dict of {_n} leaves. Isolates arity from program length",
    )
    CASES[f"arity_scan_{_n}_control"] = (
        _grad_array_scan, _array_args(_n),
        f"control: same lax.scan, carry is one ({_n}, {DIM}) array",
    )
