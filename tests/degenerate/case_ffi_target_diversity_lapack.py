"""SYNTHESISED, AND IT DOES **NOT** REPRODUCE (gap 6: custom call / FFI). The number of DISTINCT
FFI custom-call targets in one module is not a compile-time variable in jax 0.10.2 on CPU. Kept
because a bounded negative is the other half of `case_ffi_customcall_count`: that file shows the
custom-call COUNT is superlinear, this one shows target DIVERSITY is free.

No issue URL -- constructed from the hypothesis, not mined. The hypothesis was that each distinct
custom-call target costs something at lowering or compile time: a registry lookup, a handler
resolution, a per-target lowering rule in jax, a distinct backend_config schema. jaxlib's CPU
LAPACK bindings give six genuinely different FFI targets for free -- `cholesky`, `lu_factor`, `qr`,
`svd`, `eigh` and `solve_triangular` each lower to their own `lapack_*_ffi` custom call -- so the
diversity can be swept without writing any C++.

MEASURED IN THIS ENVIRONMENT (JAX_PLATFORMS=cpu, jax/jaxlib 0.10.2, x64 on, 16x16 float32, one
fresh process per point, single shot):

    K calls    6 distinct targets        1 target, K times        no custom call at all
              lower   compile   HLO     lower  compile   HLO     lower  compile    HLO
      6       1.225    0.509     287    0.734   0.385     294    0.464   0.382     117
     24       1.211    0.556     977    1.178   0.880   1,068    0.451   0.331     369
     96       1.233    1.350   3,737    1.372   0.796   4,164    0.653   0.576   1,377
    384       4.866    7.892  14,777    2.855   8.160  16,548    1.310   2.474   5,409

The diverse arm and the single-target arm agree to within run-to-run noise at every K -- 0.97x at
K=384, and the single-target arm is actually SLOWER at K=24. Target diversity buys nothing. Note
also that the diverse arm's *lowering* time is FLAT from K=6 to K=96 (1.225 / 1.211 / 1.233 s);
that ~0.75 s is the one-off python import cost of `jax.scipy.linalg` and friends, not a per-call
cost, which is a trap worth writing down: a naive read of "lowering is 2.7x the plain arm" at K=6
attributes an import to the FFI machinery.

WHAT THIS DOES AND DOES NOT BOUND. It bounds "many distinct FFI targets" on CPU, at up to 384 calls
and 6 targets, for targets that jaxlib itself registers. It does NOT bound: GPU (different targets,
different registry); hand-registered targets via `jax.ffi.register_ffi_target` (untested here --
none can be registered from pure python, which is itself why this shape was chosen); custom calls
carrying large ATTRIBUTE payloads, which is a separate hypothesis that was never reached; or the
number of distinct targets going into the thousands.

THE ARMS ARE STILL WORTH RUNNING even though the verdict is negative, for two reasons. (a) The
`lap_*` arms are the corpus's only coverage of LAPACK primitives as primitives -- `cholesky`, `lu`,
`qr`, `svd`, `eigh`, `triangular_solve` all appear, which gap 13 names as uncovered. (b) The second
axis, `lap_onetarget_{k}` against a `tanh`-in-place-of-`cholesky` control, shows that the linalg
arms cost ~3.3x the plain arm at k=384 while carrying ~3x the HLO instructions -- so per-instruction
cost is comparable and the linalg EXPANSION, not the custom-call boundary, is where the extra work
comes from. A profiler that blames the custom call here is wrong, and this axis is how you catch it.

CONSTRUCTION. Each step applies one linalg op to the running matrix and then re-conditions it with
`y @ y.T + M*I`, which keeps every arm numerically well-posed (positive definite for `cholesky`,
non-singular for `lu`/`qr`) and keeps the shape at 16x16 throughout, so all three arms have the same
number of steps and the same matrix size. The re-conditioning is identical in every arm, so it
cancels out of the comparison.

PLATFORM: CPU (measured). The same construction on GPU exercises the cuSOLVER FFI targets instead
and is a genuinely different registry, so a GPU run of this file is NOT a re-measurement -- it is a
new one. GPU was off-limits when this was written.

NUMPY at module scope; no device is touched at import.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

M = 16

# Re-conditioner. float32 spelled out: the harness enables x64 globally, and f64 selects the
# `d`-prefixed LAPACK targets instead of the `s`-prefixed ones -- a different set of custom calls,
# which would silently change what the file is measuring.
_EYE = (np.eye(M, dtype=np.float32) * np.float32(M))


def _chol(y):
    return jnp.linalg.cholesky(y)


def _lu(y):
    return jax.scipy.linalg.lu_factor(y)[0]


def _qr(y):
    return jnp.linalg.qr(y)[1]


def _svd(y):
    return jnp.diag(jnp.linalg.svd(y, compute_uv=False))


def _eigh(y):
    return jnp.diag(jnp.linalg.eigh(y)[0])


def _tri(y):
    return jax.scipy.linalg.solve_triangular(jnp.tril(y) + _EYE, y, lower=True)


# Six distinct jaxlib FFI targets on CPU (lapack_spotrf_ffi, lapack_sgetrf_ffi, lapack_sgeqrf_ffi,
# lapack_sgesdd_ffi, lapack_ssyevd_ffi, and the trsm path).
DIVERSE = (_chol, _lu, _qr, _svd, _eigh, _tri)
SINGLE = (_chol,)


def _linalg_chain(x, k, ops):
    y = x
    for i in range(k):
        y = ops[i % len(ops)](y)
        y = y @ y.T + _EYE
    return y


def _plain_chain(x, k):
    """CONTROL: same step count, same matmul, same re-conditioning, no custom call anywhere."""
    y = x
    for _ in range(k):
        y = jnp.tanh(y)
        y = y @ y.T + _EYE
    return y


_X = (np.eye(M, dtype=np.float32) * np.float32(2.0))

# 6 is one pass over the six targets; 384 is 64 passes and the only size that clears the harness's
# 3 s floor. The sweep is what makes "flat" a measurement rather than an impression.
KS = (6, 24, 96, 384)

CASES = {}
for _k in KS:
    CASES[f"lap_div_{_k}"] = (
        functools.partial(_linalg_chain, k=_k, ops=DIVERSE), (_X,),
        f"synthesised gap-6, NEGATIVE RESULT: {_k} FFI custom calls cycling 6 DISTINCT jaxlib "
        f"LAPACK targets -- indistinguishable from the same count of one target",
    )
    CASES[f"lap_div_{_k}_control"] = (
        functools.partial(_linalg_chain, k=_k, ops=SINGLE), (_X,),
        f"control: the same {_k} FFI custom calls, all ONE target (cholesky) -- same call count, "
        f"same matrix size, same re-conditioning; only target diversity differs",
    )

    CASES[f"lap_onetarget_{_k}"] = (
        functools.partial(_linalg_chain, k=_k, ops=SINGLE), (_X,),
        f"second axis: {_k} FFI custom calls to ONE target against a chain with no custom call at "
        f"all -- ~3.3x at k=384, but with ~3x the HLO instructions, so the cost tracks the linalg "
        f"expansion rather than the custom-call boundary",
    )
    CASES[f"lap_onetarget_{_k}_control"] = (
        functools.partial(_plain_chain, k=_k), (_X,),
        f"control: the same {_k} steps with tanh in place of cholesky -- same step count, same "
        f"matmul, same re-conditioning, zero custom calls",
    )
