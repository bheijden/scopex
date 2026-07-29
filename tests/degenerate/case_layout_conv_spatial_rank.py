"""SYNTHESISED (gap 1: LAYOUT ASSIGNMENT / dimension_numbers). Two extra size-1 spatial dimensions
in the convolution's `dimension_numbers` -- adding ZERO arithmetic -- cost 10x compile time, at an
IDENTICAL jaxpr equation count AND an IDENTICAL optimised HLO line count.

    PLATFORM: CPU-ONLY BY CONSTRUCTION, and measured here (jax 0.10.2 / jaxlib 0.10.2,
    JAX_PLATFORMS=cpu). The gate being crossed is a hard-coded rule in the XLA:CPU backend; the
    GPU backend has no equivalent 3-spatial-dim limit, so a GPU run of this file is expected to
    be flat and that flatness would be a result about the CPU rule, not about the case.

No issue URL. Constructed from XLA source, not mined. The gate is named in two places and both
were read in the tree that matches this jaxlib:

    xla/service/cpu/ir_emission_utils.cc, PotentiallyImplementedAsEigenConvolution():

        // Only 1D through 3D convolutions are supported at the moment.
        const int64_t num_spatial_dims = dnums.output_spatial_dimensions_size();
        if (num_spatial_dims < 1 || num_spatial_dims > 3) {
          return false;
        }

    xla/service/cpu/thunk_emitter.cc, EmitConvolutionThunk():

        if (PotentiallyImplementedAsEigenConvolution(...)) { ... ConvolutionThunk ... }
        // This is a completely un-optimized version of convolution just to
        // have an early version that works. E.g. the input index and
        // padding calculation is not hoisted out of the inner loop.
        VLOG(2) << "Falling back to unoptimized convolution: " << ...
        return EmitElementalKernelThunk(instruction);

So a convolution whose `dimension_numbers` declare 1-3 spatial dims becomes ONE call into the
Eigen runtime -- a handful of LLVM instructions that push arguments and call a symbol. A
convolution whose `dimension_numbers` declare 4 becomes a hand-emitted loop nest with the index
and padding arithmetic INSIDE the innermost loop, which LLVM then has to optimise. Same
arithmetic, different emitter, and the choice is made purely on the shape of the dimension
numbers.

Note what this is NOT. It is not `ConvCanonicalization` (xla/service/cpu/conv_canonicalization.cc)
failing: that pass rewrites a non-NHWC/HWIO permutation into the canonical one by inserting three
transposes, and it fires happily on both arms here. Permutation is fixable; spatial-dim COUNT is
not, because no transpose can turn four spatial dims into three. That is why the permutation axis
(NCHW vs NHWC) is worth only ~1.2-1.8x on CPU -- measured, see below -- and this axis is worth
10x.

THE ONE VARIABLE, AND WHY IT ADDS NO ARITHMETIC. The control is a plain 2D convolution:

    input  (B, H, W, C)          kernel (3, 3, C, C)        dnums ('NHWC',   'HWIO',   'NHWC')

The case reshapes the SAME buffers -- same element count, same bytes, same values -- to declare
two extra spatial axes of extent 1, with kernel extent 1 along each:

    input  (B, H, W, 1, 1, C)    kernel (3, 3, 1, 1, C, C)  dnums ('NHWXYC', 'HWXYIO', 'NHWXYC')

A convolution over an axis of extent 1 with a kernel of extent 1 is the identity on that axis. The
multiply-accumulate count is bit-for-bit the same: B*H*W*C*C*9 in both arms. The reshape is done
in numpy at module scope, so it is not even an operation in the traced program.

MEASURED EQUAL, not assumed -- K chained convs, each followed by tanh, reduced with sum:

    K       jaxpr eqns (case / control)     optimised HLO lines (case / control)
     32           65 / 65                           291 / 291
     64          129 / 129                          547 / 547
    128          257 / 257                         1059 / 1059
    256          513 / 513                         2083 / 2083

Equal equation count AND equal HLO line count. Every graph-size metric the corpus has says these
are the same program.

MEASURED, JAX_PLATFORMS=cpu, jax 0.10.2, B=2 C=16 H=W=16 kernel 3x3 f32, one process per arm,
arms run back-to-back:

    K       case (4 spatial dims)   control (2 spatial dims)   ratio
     32          0.616 s                  0.181 s             3.41x
     64          2.955 s                  0.336 s             8.79x
    128          3.762 s                  0.411 s             9.16x
    256          7.810 s                  0.778 s            10.03x

The ratio grows with K and the case arm's absolute cost is close to linear in K above K=64 (2.955
-> 7.810 for a 4x change in K is sublinear, because a fixed setup cost dominates at K=64). THAT IS
THE CORRECT READING: this is a large CONSTANT per convolution, not a superlinear blowup. The sweep
is what establishes that, and it is why the file ships four sizes rather than one -- a single
measurement at K=256 would have been reported as "10x" with no way to tell a constant from a
scaling law, and a single measurement at K=32 would have reported 3.4x and looked like noise.

The K=32 row was taken while the machine was carrying a load average near 20 from other
compilations; the RATIO survives contention because both arms meet the same load back to back,
but its absolute seconds are an upper bound. The same caveat applies wherever this file is
re-measured.

FOR CONTEXT, the permutation axis on the same program, measured in the same session, so the two
can be compared directly. NCHW convolutions in the entry computation, against NHWC:

    K=64    0.338 s vs 0.319 s   1.06x    (HLO 554 vs 547 -- only SEVEN extra lines, total)
    K=256   0.912 s vs 0.762 s   1.20x    (HLO 2090 vs 2083 -- again seven)

Seven extra HLO lines for 256 convolutions, because ConvCanonicalization's per-conv transposes
cancel pairwise between adjacent convs and only the first input and the last output survive. That
cancellation is transpose folding doing its job, and blocking it with an `optimization_barrier`
after each conv raises the cost to 1.173 s vs 0.661 s (1.77x) with 3119 vs 2085 HLO lines at
K=256. Both of those are real gap-1 effects and both are small. The spatial-rank arm is an order
of magnitude bigger because it changes which EMITTER runs, not how many copies get inserted.

HOW THE HARNESS WILL SCORE IT. K=128 and K=256 clear the 3 s floor and K=256 lands at 10.03x
against its control, right at MIN_VS_CONTROL. Expect a "YES" at K=256 and a "no (9.2x control)" at
K=128 -- a boundary case, deliberately, because the interesting number here is the whole column
and not the verdict on any one row.
"""

from __future__ import annotations

import functools

import jax.numpy as jnp
import numpy as np
from jax import lax

# NUMPY at module scope: a jax array here would claim a device at import, before the harness has
# chosen one. jax.jit accepts numpy arrays.
_B, _C, _H, _W, _KS = 2, 16, 16, 16, 3
_rng = np.random.default_rng(0)

_X_NHWC = _rng.normal(size=(_B, _H, _W, _C)).astype(np.float32)
_KERNS_HWIO = [(_rng.normal(size=(_KS, _KS, _C, _C)) * 0.05).astype(np.float32)
               for _ in range(256)]

# Same buffers, same bytes, same values -- reshaped in numpy to declare two extra spatial axes of
# extent 1, with kernel extent 1 along each. Adds zero multiply-accumulates.
_X_R6 = _X_NHWC.reshape(_B, _H, _W, 1, 1, _C)
_KERNS_R6 = [k.reshape(_KS, _KS, 1, 1, _C, _C) for k in _KERNS_HWIO]

# 2 spatial dims -> Eigen runtime call. 4 spatial dims -> EmitElementalKernelThunk loop nest.
_DN_2D = ("NHWC", "HWIO", "NHWC")
_DN_4D = ("NHWXYC", "HWXYIO", "NHWXYC")

_KS_SWEEP = (32, 64, 128, 256)


def _chain(x, K: int, dn, kerns, strides):
    """K convolutions, each followed by tanh, reduced to a scalar. Exactly 2K+1 equations."""
    for i in range(K):
        x = lax.conv_general_dilated(x, kerns[i], strides, "SAME", dimension_numbers=dn)
        x = jnp.tanh(x)
    return x.sum()


CASES = {}
for _k in _KS_SWEEP:
    CASES[f"convrank_k{_k}"] = (
        functools.partial(_chain, K=_k, dn=_DN_4D, kerns=_KERNS_R6, strides=(1, 1, 1, 1)),
        (_X_R6,),
        f"gap 1: K={_k} convs declaring 4 spatial dims (two of extent 1) -- off the Eigen path, "
        "elemental loop nest per conv; identical eqn and HLO line counts to the control",
    )
    CASES[f"convrank_k{_k}_control"] = (
        functools.partial(_chain, K=_k, dn=_DN_2D, kerns=_KERNS_HWIO, strides=(1, 1)),
        (_X_NHWC,),
        f"same K={_k} convs, same buffers, same multiply-accumulate count, 2 spatial dims -- one "
        "Eigen runtime call per conv",
    )
