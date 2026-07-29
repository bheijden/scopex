"""SYNTHESISED (gap 13, linalg) -- TWO findings, and the negative is the more useful one.

  (A) NEGATIVE: every linalg PRIMITIVE is compile-time free on CPU. cholesky, qr, svd, eigh, lu
      and triangular_solve all compile in 0.26-0.48 s, their gradients in 0.77-1.37 s, and mixing
      six distinct decompositions in one module costs no more than repeating one six times.
      Gap 13 asked for linalg coverage; on CPU there is nothing there, and this file says so with
      numbers.

  (B) POSITIVE, 10.1x: the linalg call that is NOT a primitive. ``jax.scipy.linalg.expm`` is a
      scaling-and-squaring Pade approximant written in JAX, so a 32-call chain expands to 22 343
      HLO lines and 6.7 s where an op-count-matched ``jnp.linalg.matrix_power`` chain expands to
      1 313 lines and 0.66 s -- from the SAME number of top-level jaxpr equations.

NOT MINED FROM AN ISSUE. Constructed and then measured.

MECHANISM. ``jnp.linalg.cholesky`` and friends bind a primitive that lowers to one LAPACK-backed
custom call: the compiler never looks inside, and compile time is independent of the matrix size
and of how many different decompositions appear. ``jax.scipy.linalg.expm`` binds nothing -- it is
ordinary JAX code implementing Higham's algorithm: an L1-norm estimate, a selection among Pade
orders 3/5/7/9/13, the matrix polynomials for the chosen order, a triangular solve, and then a
squaring loop. All of that is staged out and lands in the HLO of the caller. The two look
identical at the call site and are three orders of magnitude apart in emitted instructions.

So the control variable is "does this linalg-looking API call bind a primitive or unroll", which
is invisible at the source level and invisible in the top-level jaxpr.

MEASURED IN THIS ENVIRONMENT (JAX_PLATFORMS=cpu, jax/jaxlib 0.10.2, x64 on, 32x32 f32 matrices,
compile seconds, one fresh subprocess per measurement). D chained calls:

    D    expm      matrix_power   cholesky    expm/matpow    jaxpr eqns       HLO lines
                                                             (expm/matpow)   (expm/matpow)
     1   0.678 s     0.136 s      0.605 s        5.0x           3 /  3          736 /   73
     4   2.858 s     0.191 s      0.401 s       15.0x           9 /  9         2827 /  193
    16   2.261 s     0.208 s      0.764 s       10.9x          33 / 33        11191 /  673
    32   6.736 s     0.664 s      0.917 s       10.1x          65 / 65        22343 / 1313

The jaxpr equation counts are IDENTICAL between the expm and matrix_power arms at every depth
(3/9/33/65), because both APIs keep their bodies inside a single closed call at the top level.
The HLO line counts differ by 17x. That is the attribution test: a tool that ranks by jaxpr size
sees a tie; a tool that ranks by HLO size sees the answer; a tool that maps HLO back to source
lines lands inside ``jax/_src/scipy/linalg.py``, not in the user's program, which is the correct
answer and an awkward one to report.

RE-MEASURED ON A QUIETER BOX (same environment, ``jax.jit(fn).lower()`` then ``.compile()``):
``linalg_expm_d32`` 3.618 s compile against its matrix_power control's 0.184 s, i.e. **19.7x** --
nearly double the 10.1x seen at load ~30, because the control shrinks faster than the case when
the machine frees up. The table above is from the loaded run and its absolutes are UPPER BOUNDS.

The cholesky column is the second control and it is the one that makes this a linalg finding
rather than a "big program" finding: the cholesky chain has MORE jaxpr equations than the expm
chain at every depth (137 versus 65 at D=32) and compiles 7.3x faster, because every one of its
equations is a custom call.

--- (A) THE NEGATIVE, IN FULL ------------------------------------------------------------------

Measured on 48x48 f32, single calls, so that later work does not repeat these:

    forward:  cholesky 0.463  qr 0.406  svd 0.334  eigh 0.258  lu 0.484  triangular_solve 0.409
    grad:     svd 0.820  eigh 0.772  cholesky 1.371  qr 1.144
    batched:  grad of svd over a batch of 32   0.407

    TARGET DIVERSITY (k distinct decompositions in one module vs k copies of one):
        k=2  diverse 1.434 / uniform 1.077      k=4  diverse 1.403 / uniform 0.626
        k=6  diverse 0.919 / uniform 1.046

Nothing separates. On CPU each of these is one LAPACK custom call; the number of DISTINCT
custom-call targets in a module does not cost anything measurable either, which is a useful
negative for gap 6 as well. All of it is below the 3 s floor and none of it is worth a slot.

WHAT THE CONTROLS ISOLATE.

  * ``*_control`` (the tight one): ``jax.scipy.linalg.expm(y)`` -> ``jnp.linalg.matrix_power(y, 13)``
    at the same depth. Same call shape, same matrix, same trailing ``* 0.1``, same top-level
    equation count, and 13 is the Pade order expm itself selects, so the FLOP scale is comparable.
    What changes is whether the body is a polynomial-plus-solve-plus-squaring or a binary
    exponentiation.
  * ``linalg_chol_d32``: the same chain length built from a genuine primitive. Establishes that
    depth alone is not the variable -- more equations, 7.3x less time.
  * ``linalg_expm_grad_d16``: 3.609 s against 2.261 s forward at the same depth, i.e. AD roughly
    doubles it rather than exploding it. Recorded because it rules out "the gradient of expm is
    the real problem" -- the forward expansion is.
  * ``linalg_expm_batch_d4``: a batch of 64 16x16 matrices, 2.412 s against 2.858 s unbatched at
    the same depth. Compile cost tracks the DEPTH of the chain, not the amount of data, which is
    what an unrolled-polynomial mechanism predicts.

PLATFORM: **CPU (verified here).** Expected on GPU too -- ``expm``'s expansion happens in jax
before any backend is chosen -- but the GPU was off-limits when this file was written, so the GPU
arm is unverified. The negative in part (A) is more backend-dependent than the positive: GPU
linalg goes through cuSOLVER and its own batching rules, and a GPU re-run of the ``diverse``
versus ``uniform`` idea is the cheap way to check whether target diversity costs anything there.

MEMORY. Matrices are 32x32 f32; the largest arm holds 64 of them. This case is large in
INSTRUCTION count, not in bytes. Inputs are small random values scaled by 0.1 so that the chained
matrix exponentials do not overflow at runtime -- runtime is half the harness's statistic and an
inf-filled result would make it meaningless. Values cannot affect compile time.

NOTE ON THE BOX. Load average was ~30 on 20 cores while these numbers were taken; absolute seconds
are UPPER BOUNDS and the paired ratios are the statistic to trust.
"""

from __future__ import annotations

import functools

import numpy as np

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl

_RNG = np.random.default_rng(0)

# NUMPY at module scope. Scaled down so the chained exponentials stay finite at runtime.
_X32 = (_RNG.standard_normal((32, 32)) * 0.1).astype(np.float32)
_XB = (_RNG.standard_normal((64, 16, 16)) * 0.1).astype(np.float32)
_X48 = (_RNG.standard_normal((48, 48)) * 0.1).astype(np.float32)


def _expm_chain(x, depth: int):
    """D calls to a linalg function that UNROLLS: Pade order selection + squaring, all in HLO."""
    y = x
    for _ in range(depth):
        y = jsl.expm(y) * 0.1
    return y.sum()


def _matpow_chain(x, depth: int, p: int = 13):
    """D calls to a linalg function that does NOT unroll a Pade table. 13 = expm's own top order."""
    y = x
    for _ in range(depth):
        y = jnp.linalg.matrix_power(y, p) * 0.1
    return y.sum()


def _chol_chain(x, depth: int):
    """D calls to a genuine PRIMITIVE -- one LAPACK custom call each, more equations, less time."""
    y = x @ x.T + 32.0 * jnp.eye(x.shape[0], dtype=x.dtype)
    acc = 0.0
    for i in range(depth):
        acc = acc + jnp.linalg.cholesky(y + float(i)).sum()
    return acc


def _expm_grad(x, depth: int):
    return jax.grad(functools.partial(_expm_chain, depth=depth))(x)


def _diverse(x, k: int):
    """k DISTINCT LAPACK targets in one module (the negative)."""
    spd = x @ x.T + 48.0 * jnp.eye(x.shape[0], dtype=x.dtype)
    fns = [lambda z: jnp.linalg.cholesky(z).sum(),
           lambda z: jnp.linalg.qr(z)[0].sum(),
           lambda z: jnp.linalg.svd(z, full_matrices=False)[1].sum(),
           lambda z: jnp.linalg.eigh(z)[0].sum(),
           lambda z: jsl.lu(z)[2].sum(),
           lambda z: jax.lax.linalg.triangular_solve(z, z, left_side=True, lower=True).sum()]
    return sum(fns[i % len(fns)](spd + float(i)) for i in range(k))


def _uniform(x, k: int):
    """k copies of ONE target. Same op count."""
    spd = x @ x.T + 48.0 * jnp.eye(x.shape[0], dtype=x.dtype)
    return sum(jnp.linalg.cholesky(spd + float(i)).sum() for i in range(k))


CASES = {}

# --- (B) the positive: unrolled linalg vs a primitive-backed one, swept over chain depth -----
for _d in (4, 16, 32, 64):
    CASES[f"linalg_expm_d{_d}"] = (
        functools.partial(_expm_chain, depth=_d), (_X32,),
        f"synthesised: chain of {_d} jax.scipy.linalg.expm calls. Not a primitive -- Pade order "
        f"selection plus squaring is staged into the caller's HLO. Measured 6.7 s at D=32 with "
        f"22343 HLO lines from 65 jaxpr equations")
    CASES[f"linalg_expm_d{_d}_control"] = (
        functools.partial(_matpow_chain, depth=_d), (_X32,),
        f"control: the same chain with expm -> jnp.linalg.matrix_power(y, 13), depth {_d}. Same "
        f"top-level jaxpr equation count, comparable FLOPs, 17x fewer HLO lines. Measured 0.66 s "
        f"at D=32")

# --- second control: a genuine primitive at the same chain depth ------------------------------
CASES["linalg_chol_d32"] = (
    functools.partial(_chol_chain, depth=32), (_X32,),
    "second control: 32 chained jnp.linalg.cholesky calls -- 137 jaxpr equations (MORE than the "
    "expm arm's 65) and 0.917 s (7.3x faster). Depth is not the variable; unrolling is")

# --- AD is not the story: forward expansion already is ---------------------------------------
CASES["linalg_expm_grad_d16"] = (
    functools.partial(_expm_grad, depth=16), (_X32,),
    "probe: jax.grad of the depth-16 expm chain -- 3.609 s against 2.261 s forward, i.e. AD "
    "roughly doubles it. Rules out 'the expm gradient is the problem'")
CASES["linalg_expm_grad_d16_control"] = (
    functools.partial(_expm_chain, depth=16), (_X32,),
    "control: the same depth-16 expm chain with no jax.grad")

# --- data volume is not the variable; chain depth is ------------------------------------------
CASES["linalg_expm_batch_d4"] = (
    functools.partial(_expm_chain, depth=4), (_XB,),
    "probe: depth-4 expm chain over a BATCH of 64 16x16 matrices -- 2.412 s against 2.858 s for "
    "one 32x32 matrix at the same depth. Compile tracks depth, not data")

# --- (A) the negative: custom-call TARGET DIVERSITY costs nothing on CPU ----------------------
for _k in (4, 6):
    CASES[f"linalg_diverse_k{_k}"] = (
        functools.partial(_diverse, k=_k), (_X48,),
        f"NEGATIVE: {_k} DISTINCT LAPACK custom-call targets (cholesky/qr/svd/eigh/lu/"
        f"triangular_solve) in one module. Measured 1.403 s at k=4, 0.919 s at k=6 -- flat")
    CASES[f"linalg_diverse_k{_k}_control"] = (
        functools.partial(_uniform, k=_k), (_X48,),
        f"control: {_k} copies of ONE target (cholesky), same op count. Measured 0.626 s at k=4, "
        f"1.046 s at k=6 -- i.e. the 'control' is sometimes slower. No effect either way")
