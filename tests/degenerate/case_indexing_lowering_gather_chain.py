"""jax#32704 -- a serial chain of 2D gathers compiles exponentially; the same chain written as 1D
gathers on the flattened array compiles flat.

    https://github.com/jax-ml/jax/issues/32704          OPEN, assigned to jakevdp

Reported compile seconds, N=1000, nsamples=1e6, one gather per link, index chained through:

    ncycles       4      5      6      7      8       9
    2D  index  0.069  0.086  0.243  0.997  4.144  17.216      ~4x per added link
    flattened  0.063  0.040  0.035  0.036  0.037   0.037      flat
    runtime    identical for both arms, ~3-10 ms

WHAT THE CONTROL ISOLATES. The control is the reporter's *corrected* `sample_flatten` from the
follow-up comment -- the first version in the issue body dropped the serial dependence and so was
not a control at all. In the corrected form `cols = flat[rows * N + cols]` the output of link i is
still the index of link i+1, so both arms have:

  * the same chain DEPTH (ncycles serial, unparallelisable gathers),
  * the same OP COUNT (one gather per link either way),
  * the same data dependence graph, and the same numerical result,
  * the same runtime.

The only thing that moves is the STRUCTURE of the gather's index: `data[rows, cols]` lowers to a
gather whose `start_index_map` covers two dimensions and whose start indices are assembled by a
concatenate of two index vectors, while `flat[rows * N + cols]` is a single-dimension gather off one
precomputed index vector. That is the whole delta, and it is why this case is worth having next to
the scatter-chain flagship: there the knob is a trailing reduction and chain depth is what grows;
here chain depth is HELD CONSTANT between the arms and only index rank/structure moves. "Long serial
chain = slow compile" cannot explain a 355x gap between two programs with identical chain depth.

The issue carries no maintainer diagnosis, so the suspected cause (XLA gather simplification /
index canonicalisation composing badly across chained multi-dimensional gathers) is a hypothesis
this file does not assert.

RELATIONSHIP TO `case_gather_2d_chain.py`. That file measures the same issue at N=500 /
nsamples=2e5 with int64 indices (x64 is on globally, so numpy's default integer dtype survives into
the gather) and reproduced NOTHING on this GPU: compile sat at 0.7-1.3 s and did not grow at all
from ncycles 4 to 9, against a reported 0.069 -> 17.2 s over the same span. This file restores the
two things that run cut, so a non-reproduction can be attributed to the backend rather than to the
scaling-down:

  * FULL reported sizes, N=1000 and nsamples=1e6 (still only ~8 MB per array), and
  * int32 data and indices, which is what the reporter's environment produced by default with x64
    off, and which is the dtype the reported curve was measured at.

and it pushes the sweep to ncycles=11, two links past where the report already showed 17 s. If this
GPU is still flat at ncycles=11 with the reported shapes and dtypes, the honest conclusion is that
jax#32704 is a CPU-backend pathology and does not exist on CUDA -- which is a RESULT, and one that
is only defensible because the sizes match the report.
"""

from __future__ import annotations

import functools

import numpy as np

N = 1000
NSAMPLES = 1_000_000

# Plain numpy at module scope: device-free, so importing this file to discover CASES never claims an
# accelerator. jax.jit accepts numpy arrays and commits them to whichever device the harness picked.
# int32 on purpose -- see the docstring; the harness forces x64 on, which would otherwise silently
# make every index in this file int64 and change the gather that XLA sees.
_rng = np.random.default_rng(0)
_DATA = _rng.integers(0, N, (N, N), dtype=np.int32)
_DATA_FLAT = np.ascontiguousarray(_DATA.reshape(-1))
_IX = _rng.integers(0, N, NSAMPLES, dtype=np.int32)


def _chain_2d(data, rows, cols, ncycles):
    """Pathological arm: each link is a 2-dimensional gather `data[rows, cols]`."""
    for _ in range(ncycles):
        cols = data[rows, cols]
    return cols


def _chain_flat(data_flat, rows, cols, ncycles):
    """Control arm: identical chain, identical depth, one-dimensional gathers.

    `rows * N + cols` keeps the serial dependence -- `cols` is still the output of the previous
    link. Dropping that dependence (the version in the issue body) makes the chain parallel and
    stops being a control.
    """
    for _ in range(ncycles):
        cols = data_flat[rows * N + cols]
    return cols


def _mk(kind, ncycles):
    if kind == "2d":
        return (functools.partial(_chain_2d, ncycles=ncycles),
                (_DATA, _IX, _IX),
                f"jax#32704 serial chain of {ncycles} two-dimensional gathers, N={N}")
    return (functools.partial(_chain_flat, ncycles=ncycles),
            (_DATA_FLAT, _IX, _IX),
            f"control: same {ncycles}-deep serial chain as 1D gathers on the flattened array")


# A sweep, because the claim is about SCALING in chain length. A single point cannot separate
# "exponential in depth" from "this program is simply large". 8 and 9 are the two points the report
# measured at 4.1 s and 17.2 s; 10 and 11 are past the end of the reported curve.
CASES = {}
for _n in (8, 9, 10, 11):
    CASES[f"gatherchain2d_{_n}"] = _mk("2d", _n)
    CASES[f"gatherchain2d_{_n}_control"] = _mk("flat", _n)
