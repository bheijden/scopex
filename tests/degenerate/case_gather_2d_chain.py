"""jax#32704 -- chained 2D fancy-indexing compiles exponentially ON CPU; the flattened form does not.

PLATFORM: **CPU-ONLY.** This case does not exist on GPU, and that is the point of keeping it.

    CPU (measured here, N=1000, nsamples=1e6, ncycles=9):  218.666 s vs 0.882 s control = 248x,
                                                           and XLA prints its own "Very slow
                                                           compile?" warning
    GPU (measured here, ncycles 4..9):                     flat ~1 s, both arms, 1.02x

An automated triage pass measured only the GPU arm, wrote "MEASURED IN-ENV, DOES NOT REPRODUCE...
the report was CPU / jax 0.7.2; the GPU gather lowering does not have this behaviour", and dropped
the case -- having named the platform split in its own reasoning. It is retained deliberately.
A pathology that exists on one backend and not another is not a weaker case than one that exists on
both; it is a SHARPER one, because the backend is then a control in its own right.

    https://github.com/jax-ml/jax/issues/32704          OPEN, assigned to jakevdp

Reported on jax 0.7.2, CPU, N=1000, nsamples=1e6. Compile seconds by chain length:

    ncycles       4      5      6      7      8       9
    2D index  0.069  0.086  0.243  0.997  4.144  17.216
    flattened 0.063  0.040  0.035  0.036  0.037   0.037

Roughly 4x per added link against a flat control. Run time is identical for both (0.003-0.010 s),
so this is purely compile-side.

WHY THIS CASE EARNS ITS PLACE IN THE CORPUS. It is a distinct mechanism from the scatter chain we
already have, on three axes:

  * READ side (gather), not write side (scatter).
  * NO TRAILING REDUCTION. The scatter pathology needs a `jnp.sum` to appear at all -- an MWE that
    omitted it reproduced nothing. This one blows up on its own.
  * The control is a pure INDEXING-FORM change: `data[r, c]` versus `data_flat[r * N + c]`. Same
    data, same chain length, same number of gathers, identical results. Anything that differs
    between the two arms is the pathology and nothing else.

The suspected cause is XLA gather simplification / index canonicalisation composing badly across
chained multi-dimensional gathers, but the issue carries no maintainer diagnosis, so that is a
hypothesis and this file does not assert it.

Sizes are cut from the report (N=1000, nsamples=1e6) to N=500, nsamples=2e5 so a full sweep fits in
a test run. If the pathology is exponential in ncycles and only linear in nsamples, that trade
keeps the effect and cuts the constant -- which is itself worth confirming, so both sizes are here.
"""

from __future__ import annotations

import functools

import numpy as np

N = 500
NSAMPLES = 200_000

# Plain NUMPY at module scope: device-free, so merely importing this file to discover its CASES
# dict never claims an accelerator. The first draft used `jnp.asarray` here and failed with
# "no supported devices found for platform CUDA" when another process held the card. jax.jit
# accepts numpy arrays and commits them to whichever device the harness chose.
_rng = np.random.default_rng(0)
_DATA = _rng.integers(0, N, (N, N))
_DATA_FLAT = _DATA.reshape(-1)
_IX = _rng.integers(0, N, NSAMPLES)


def _unflatten(data, rows, cols, ncycles):
    """The pathological form: chained 2D fancy indexing."""
    for _ in range(ncycles):
        cols = data[rows, cols]
    return cols


def _flatten(data_flat, rows, cols, ncycles):
    """The control: same chain, computed on a flattened array."""
    for _ in range(ncycles):
        cols = data_flat[rows * N + cols]
    return cols


def _mk(kind, ncycles):
    if kind == "2d":
        fn = functools.partial(_unflatten, ncycles=ncycles)
        return fn, (_DATA, _IX, _IX), f"jax#32704 chained 2D gather, ncycles={ncycles}"
    fn = functools.partial(_flatten, ncycles=ncycles)
    return fn, (_DATA_FLAT, _IX, _IX), f"control: flattened gather, ncycles={ncycles}"


# A sweep, because the claim is about SCALING. A single point cannot distinguish "exponential in
# chain length" from "this program is simply big".
CASES = {}
for _n in (4, 6, 8, 9, 10):
    CASES[f"gather2d_{_n}"] = _mk("2d", _n)
    CASES[f"gather2d_{_n}_control"] = _mk("flat", _n)
