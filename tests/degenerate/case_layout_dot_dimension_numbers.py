"""SYNTHESISED (gap 1: LAYOUT ASSIGNMENT / TRANSPOSE FOLDING). Byte-identical jaxpr, different
`dimension_numbers`, different HLO. WEAK ON CPU (1.3x-2.5x) and shipped as a BOUND, not a win.

    PLATFORM: EITHER, and the two backends are expected to disagree sharply.
    CPU: measured here (jax 0.10.2 / jaxlib 0.10.2, JAX_PLATFORMS=cpu). 2.0-2.5x at small K,
         decaying to ~1.3x at large K. Real but modest, and it does not clear the harness's
         MIN_VS_CONTROL of 10.
    GPU: UNMEASURED -- the GPU was owned by another investigation when this was written. GPU is
         where this axis is expected to matter, because XLA:GPU runs a real LayoutAssignment with
         cost-driven heuristics plus TransposeFolding plus an NCHW/NHWC conversion pass, whereas
         XLA:CPU's CpuLayoutAssignment::AddBackendConstraints simply forces row-major on every
         operand of every instruction (cpu_layout_assignment.cc, the final `else` branch), so
         there is almost no layout DECISION on CPU to be expensive.

No issue URL. Constructed from the audit's gap 1, which names exactly this artifact: "Same program
with permuted `dimension_numbers`; identical FLOPs and op count, different layout decisions."

WHY IT IS HERE AT ALL, GIVEN THE MODEST CPU NUMBERS. Gap 1 had ZERO coverage in the corpus -- an
entire HLO pipeline stage with nothing pointing at it. This file is the smallest honest thing that
does, and it has one property nothing else in the corpus has:

    THE JAXPR IS IDENTICAL AND THE HLO IS NOT.

Measured equation counts, all three arms, at every K: 16 / 64 / 128 / 256 -- equal. Same primitive
(`dot_general`), same operand shapes up to permutation, same output shape (8, 96, 96) in all three
arms, same FLOPs, same dtype. What differs is one tuple of integers passed to `lax.dot_general`.
Below the jaxpr, XLA's DotDecomposer has to normalise the non-canonical arms into a batch-major
form and inserts transposes to do it, so the HLO line count diverges:

    K=8    HLO 63 (major)   vs 77 (mid)   vs 77 (minor)
    K=32   HLO 135          vs 228        vs 228
    K=64   HLO 468          vs 548        vs 548
    K=128  HLO 1108         vs 1188       vs 1188

A profiler that works off the jaxpr sees three identical programs. A profiler that works off the
optimised HLO sees three different ones. That disagreement is the discriminator, and it is worth
having in the corpus even though the seconds are small.

`lax.dot_general` is used deliberately instead of `jnp.einsum`. einsum with a non-canonical spec
inserts transposes IN JAX, before the jaxpr closes: the same comparison written with
`jnp.einsum('ibj,jbk->ibk', ...)` gives 13 equations against 9 for the canonical spec (measured),
which moves part of the cost above the jaxpr and destroys the isolate. dot_general takes
`dimension_numbers` verbatim and traces to exactly one equation whatever they are.

THE THREE ARMS. `a` and `w` hold the same numbers in all three arms, transposed at module scope
with numpy so no work happens at trace time:

    major (control)   a:(b,i,j)  w:(b,j,k)   batch dim 0    -- canonical, batch-major
    mid               a:(i,b,j)  w:(j,b,k)   batch dim 1    -- batch dim in the middle
    minor             a:(i,j,b)  w:(j,k,b)   batch dim 2    -- batch dim minormost

All three contract over a 96-length axis and produce (8, 96, 96). The K independent dots are
summed, so nothing about chaining or output ordering differs between arms.

MEASURED, JAX_PLATFORMS=cpu, jax 0.10.2, a = (8,96,96) f32, one process per arm:

    K      major (control)   mid       minor     mid/major   minor/major
      8       0.137 s      0.275 s   0.346 s      2.01x        2.53x
     32       0.405 s      1.017 s   0.851 s      2.51x        2.10x
     64       2.553 s      3.343 s   2.341 s      1.31x        0.92x
    128       2.891 s      3.919 s   3.811 s      1.36x        1.32x

READ THE SWEEP, NOT ANY ONE ROW. The ratio DECAYS with K, and at K=64 the `minor` arm came in
below its own control. That is the honest result: on CPU the per-dot transpose is a fixed, small
cost that is swamped once the module is large enough for other per-instruction costs to dominate.
A single measurement at K=8 would have reported "2.5x" and been misleading; the sweep is what
shows the effect is a constant, not a scaling law. This is exactly the failure the corpus's
size-sweep rule exists to prevent.

WHAT THIS BOUNDS. On XLA:CPU, permuted dot `dimension_numbers` cost a small constant per dot and
nothing more -- there is no superlinear layout blowup hiding here at K up to 128. It says nothing
about GPU, and it says nothing about convolution, whose CPU layout story is a genuine cliff and is
covered separately in case_layout_conv_spatial_rank.py and
case_layout_conv_nested_computation.py in this directory.

HOW THE HARNESS WILL SCORE IT. K=64 and K=128 clear the 3 s floor; the ratios are ~1.3x, so expect
"no (1.3x control)". That is the correct verdict for this file and printing it is the file doing
its job.
"""

from __future__ import annotations

import functools

import numpy as np
from jax import lax

# NUMPY at module scope. A jax array here would claim a device at import, before the harness has
# chosen one.
_BB, _II, _JJ = 8, 96, 96
_rng = np.random.default_rng(0)

_A_bij = (_rng.normal(size=(_BB, _II, _JJ)) * 0.1).astype(np.float32)
_A_ibj = np.ascontiguousarray(_A_bij.transpose(1, 0, 2))
_A_ijb = np.ascontiguousarray(_A_bij.transpose(1, 2, 0))

_W_bjk = [(_rng.normal(size=(_BB, _JJ, _JJ)) * 0.05).astype(np.float32) for _ in range(128)]
_W_jbk = [np.ascontiguousarray(w.transpose(1, 0, 2)) for w in _W_bjk]
_W_jkb = [np.ascontiguousarray(w.transpose(1, 2, 0)) for w in _W_bjk]

# ((contracting lhs, contracting rhs), (batch lhs, batch rhs)). All three give a (8, 96, 96)
# result from a 96-length contraction -- the arithmetic is the same, only the axis order moves.
_DN_MAJOR = (((2,), (1,)), ((0,), (0,)))
_DN_MID = (((2,), (0,)), ((1,), (1,)))
_DN_MINOR = (((1,), (0,)), ((2,), (2,)))

_KS = (8, 32, 64, 128)


def _indep_dots(a, K: int, dn, ws):
    """K independent dot_generals over the same lhs, summed. Exactly K dot_general equations."""
    acc = None
    for i in range(K):
        r = lax.dot_general(a, ws[i], dimension_numbers=dn)
        acc = r if acc is None else acc + r
    return acc.sum()


CASES = {}
for _k in _KS:
    CASES[f"dotdims_mid_k{_k}"] = (
        functools.partial(_indep_dots, K=_k, dn=_DN_MID, ws=_W_jbk),
        (_A_ibj,),
        f"gap 1: K={_k} dot_generals with the batch dim in the MIDDLE; identical jaxpr to the "
        "control, DotDecomposer inserts transposes below it",
    )
    CASES[f"dotdims_mid_k{_k}_control"] = (
        functools.partial(_indep_dots, K=_k, dn=_DN_MAJOR, ws=_W_bjk),
        (_A_bij,),
        f"canonical batch-major dimension_numbers, same K={_k} equations, same shapes, same FLOPs",
    )
    CASES[f"dotdims_minor_k{_k}"] = (
        functools.partial(_indep_dots, K=_k, dn=_DN_MINOR, ws=_W_jkb),
        (_A_ijb,),
        f"gap 1: K={_k} dot_generals with the batch dim MINORMOST -- the most permuted arm",
    )
    CASES[f"dotdims_minor_k{_k}_control"] = (
        functools.partial(_indep_dots, K=_k, dn=_DN_MAJOR, ws=_W_bjk),
        (_A_bij,),
        f"canonical batch-major dimension_numbers, same K={_k} equations, same shapes, same FLOPs",
    )
