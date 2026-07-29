"""jax#2777 -- exponential compile time in python-unrolled loop length, diagnosed BELOW XLA, in
LLVM's global value numbering during PTX codegen.

    https://github.com/jax-ml/jax/issues/2777          CLOSED as addressed, 2020, jax 0.1.x

Reported GPU compile times for `sample(key, n, scan=False)`:

    n           1       2       3       4       5       6
    unrolled  59.1ms  96.4ms   233ms  1.07 s  6.39 s  28.1 s        ~5x per added iteration
    scan=True                        flat, milliseconds, all n

Reported flat on TPU, exponential on CPU and GPU. The maintainer's profile put the time in
`llvm::GVN::propagateEquality` and neighbouring global-value-numbering passes -- LLVM's
dominance-tree value-equality propagation going superlinear as the unrolled basic block grows.

WHY THIS CASE IS WORTH A SLOT DESPITE BEING SIX YEARS OLD AND CLOSED. Every other case in the
corpus puts its time somewhere scopex can in principle see from the jaxpr or from XLA's own pass
timing: gather simplification, scan partial-eval, scatter lowering. This one does not. The jaxpr is
linear in n. The HLO is linear in n. The XLA HLO pass timing is linear in n. The cost appears only
after HLO, inside the LLVM pipeline that produces PTX, and it is a different attribution target
from anything else we have -- an attribution tool that reports "your program has 6x more ops than
at n=1" when the compile time went up 500x has said something true and useless.

THE CONTROL IS BUILT INTO THE REPRO. `scan` is a static boolean argument of the reporter's own
`sample`. `scan=True` runs the identical body as `lax.scan`, `scan=False` unrolls it in python.
Same body, same n, same numerics, same runtime, one boolean. Nothing else in the corpus has a
control that cheap.

IF IT REPRODUCES, the immediate follow-up is to confirm the STAGE, not just the effect: compare
total compile against XLA's own HLO pass timing (and dump with XLA_FLAGS=--xla_gpu_dump_llvmir).
A cost that lives in LLVM rather than in an HLO pass is the finding; a cost that lives in an HLO
pass is a different, more ordinary case.

EXPECTATION MANAGEMENT -- this is the least likely of its batch to be alive, and that is fine, a
non-reproduction here is a result. Since 2020: `jax.random` was rewritten around partitionable
threefry, the whole PJRT/GPU compilation path was replaced, and the issue was closed as addressed.
The most likely outcome is a flat curve on both arms. The reason to spend the slot anyway is that
it is the ONLY probe we have for the LLVM/PTX stage, so if it is alive it is worth more than its
rank.

The program body is verbatim from the issue, including the nested `@jax.jit` on `inner_loop`: that
nesting is part of what was measured (it becomes a pjit boundary per unrolled iteration in modern
jax) and removing it would be reconstructing a different program. `n` and `scan` were the
reporter's static argnums; here they are closed over with functools.partial instead, so the
harness's `jax.jit(fn).lower(*args)` sees exactly the same specialisation.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

NUM_OBS = 8


@jax.jit
def _inner_loop(key, ra, rb):
    """Verbatim from the issue."""
    gate_shape = (NUM_OBS, 1, 1, 1)
    key, shard = jax.random.split(key)
    gate_forward = jax.random.uniform(
        shard, gate_shape, dtype=jnp.float32, minval=0.0, maxval=1.0)
    gate_forward = gate_forward.reshape([-1, 1])
    new_ra = 0.5 * (ra + gate_forward * ra + (1 - gate_forward) * rb)
    new_rb = 0.5 * (rb + gate_forward * ra + (1 - gate_forward) * rb)
    return key, new_ra, new_rb


def _sample(key, n, use_scan):
    """Verbatim from the issue; `n` and `use_scan` were its static argnums."""
    # float32 spelled out. The issue predates this harness's global x64, under which these literals
    # would come out float64 and, mixing with the explicitly-float32 gate, promote the entire
    # computation to f64 -- different LLVM codegen from the f32 the reported curve was measured on,
    # for a case whose whole claim is about codegen.
    ra = jnp.array([[1.0, 0.0] for _ in range(NUM_OBS)], dtype=jnp.float32)
    rb = jnp.array([[0.0, 1.0] for _ in range(NUM_OBS)], dtype=jnp.float32)
    if use_scan:
        (key, ra, rb), () = jax.lax.scan(
            lambda c, _: (_inner_loop(*c), ()), (key, ra, rb), (), length=n)
    else:
        for _ in range(n):
            key, ra, rb = _inner_loop(key, ra, rb)
    return ra


# A raw uint32[2] threefry key as plain numpy: device-free at import, and it is what
# jax.random.PRNGKey(0) produces, so the traced program is unchanged. Building the key with
# jax.random.PRNGKey at module scope would claim the GPU merely to import this file.
_KEY = np.array([0, 0], dtype=np.uint32)


def _mk(n, use_scan):
    fn = functools.partial(_sample, n=n, use_scan=use_scan)
    if use_scan:
        return fn, (_KEY,), f"control: identical body under lax.scan, n={n}"
    return fn, (_KEY,), f"jax#2777 python-unrolled sampling loop, n={n}"


# The reported curve turns over between n=4 (1.07 s) and n=6 (28.1 s). If the ~5x/iteration growth
# is still there, n=7 is roughly two minutes and n=8 would blow the harness timeout, so the sweep
# stops at 7. If the curve is flat at n=7 the case is dead and the whole sweep cost seconds.
CASES = {}
for _n in (4, 5, 6, 7):
    CASES[f"unrolled_sample_{_n}"] = _mk(_n, False)
    CASES[f"unrolled_sample_{_n}_control"] = _mk(_n, True)
