"""SYNTHESISED (gap 1: LAYOUT ASSIGNMENT / TRANSPOSE FOLDING). The CPU layout-canonicalisation pass
walks ONLY the entry computation, so the same NCHW convolution costs 1.06x in the entry
computation and 8x inside a `lax.scan` body. Two-equation jaxpr, identical HLO line count.

    PLATFORM: CPU-ONLY BY CONSTRUCTION, measured here (jax 0.10.2 / jaxlib 0.10.2,
    JAX_PLATFORMS=cpu). The pass being skipped is `xla::cpu::ConvCanonicalization`, which exists
    only in the CPU backend. A GPU run is expected to be flat; that would be a result about the
    CPU pass, not about the case.

No issue URL. Constructed from XLA source. One line is the whole mechanism --
xla/service/cpu/conv_canonicalization.cc, ConvCanonicalization::RunImpl():

    for (HloInstruction* hlo :
         module->entry_computation()->MakeInstructionPostOrder()) {
      if (hlo->opcode() == HloOpcode::kConvolution &&
          !PotentiallyImplementedAsEigenConvolution(*hlo, target_machine_features_)) {

`module->entry_computation()`. Not `module->computations()`. A convolution that is not in
NHWC/HWIO/NHWC form gets three transposes inserted and keeps its fast path IF AND ONLY IF it sits
in the entry computation. The identical convolution inside a while body -- which is what
`lax.scan`, `lax.fori_loop` and `lax.while_loop` all produce -- is never visited, still fails
`PotentiallyImplementedAsEigenConvolution` at emission time, and falls to
`EmitElementalKernelThunk` in xla/service/cpu/thunk_emitter.cc: "a completely un-optimized version
of convolution ... the input index and padding calculation is not hoisted out of the inner loop".

WHY THIS EARNS A SLOT NEXT TO case_layout_conv_spatial_rank.py. That file changes the
dimension_numbers so that NO pass can fix them. This one leaves the dimension_numbers perfectly
fixable and changes WHERE THE INSTRUCTION LIVES. The pair separates "these dimension numbers are
unsupported" from "this pass did not look here", which are different findings for a user and
should be different findings for a profiler. Measured in the same session, same convolutions, same
sizes, only entry-vs-scan differing:

    NCHW vs NHWC, K=64 convs UNROLLED IN THE ENTRY COMPUTATION   0.338 s / 0.319 s  = 1.06x
    NCHW vs NHWC, K=64 convs INSIDE A lax.scan BODY              3.418 s / 0.423 s  = 8.08x

Same convolutions. Same dimension_numbers. 1.06x or 8.08x depending only on whether they are
inside a loop body.

THE ONE VARIABLE. Both arms are the same `lax.scan` over 4 steps whose body contains D chained
convolutions, each followed by tanh. Only the dimension_numbers token differs, with the module-
scope numpy inputs permuted to match:

    case      dimension_numbers = ('NCHW', 'OIHW', 'NCHW')   input (B, C, H, W)
    control   dimension_numbers = ('NHWC', 'HWIO', 'NHWC')   input (B, H, W, C)

Same element count, same bytes, same values (the arrays are numpy transposes of one another), same
multiply-accumulate count, same trip count, same body length.

MEASURED EQUAL, not assumed:

    D       jaxpr eqns (case / control)     optimised HLO lines (case / control)
     16            2 / 2                            209 / 209
     32            2 / 2                            337 / 337
     64            2 / 2                            593 / 593
    128            2 / 2                           1105 / 1105

TWO jaxpr equations in both arms -- the scan and the reduction. The entire pathology is invisible
above the jaxpr, and the optimised HLO line counts are equal as well, which is the sharper
statement: unlike the entry-computation form (where NCHW adds exactly seven HLO lines, total, for
the surviving input and output transposes), here XLA adds NOTHING because the pass never ran. Two
identical-looking modules, 8x apart in compile time.

MEASURED, JAX_PLATFORMS=cpu, jax 0.10.2, B=2 C=16 H=W=16 kernel 3x3 f32, scan length 4, one
process per arm, arms run back-to-back:

    D       case (NCHW in scan)   control (NHWC in scan)   ratio
      1          0.515 s               0.220 s            2.35x
      2          0.444 s               0.260 s            1.71x
      4          0.617 s               0.226 s            2.73x
      8          0.573 s               0.249 s            2.30x
     16          1.318 s               0.329 s            4.00x
     32          2.019 s               0.425 s            4.75x
     64          3.418 s               0.423 s            8.08x
    128          7.831 s               0.397 s           19.71x

Read the whole column. Below D=8 the ratio bounces between 1.7x and 2.7x on sub-second compiles --
noise. From D=16 the case arm grows roughly linearly in D (1.318 -> 7.831 is 5.9x for an 8x change
in D) while the control arm is FLAT (0.329 -> 0.397), because in the control the loop body
contains D calls to a runtime symbol and calls are cheap to emit no matter how many there are. The
ratio therefore grows without the mechanism being superlinear: it is a large constant per
convolution in one arm against a near-zero constant in the other. That flat control is what makes
the ratio keep climbing, and it is why the sweep runs to D=128.

The D=128 row was taken while the machine was carrying a load average near 20 from other
compilations, so its absolute seconds are an upper bound; the ratio survives contention because
both arms meet the same load back to back.

THE SCAN LENGTH IS 4 AND DOES NOT MATTER. A while body is compiled once regardless of trip count,
so the compile-time knob is D (the body's convolution count), not the number of steps. The length
is kept small and nonzero purely so the runtime is real and the harness's compile/runtime ratio is
meaningful.

HOW THE HARNESS WILL SCORE IT. D=64 clears the 3 s floor at 8.08x, just under MIN_VS_CONTROL of
10, so expect "no (8.1x control)" there; D=128 clears both at 19.71x and should report "YES". The
column is the finding and the verdict on any one row is not -- a suite that shipped only D=64
would have recorded this mechanism as absent.
"""

from __future__ import annotations

import functools

import jax.numpy as jnp
import numpy as np
from jax import lax

# NUMPY at module scope: a jax array here would claim a device at import, before the harness has
# chosen one.
_B, _C, _H, _W, _KS = 2, 16, 16, 16, 3
_rng = np.random.default_rng(0)

_X_NHWC = _rng.normal(size=(_B, _H, _W, _C)).astype(np.float32)
# The same numbers, transposed in numpy so the traced program is identical in both arms.
_X_NCHW = np.ascontiguousarray(_X_NHWC.transpose(0, 3, 1, 2))

_KERNS_HWIO = [(_rng.normal(size=(_KS, _KS, _C, _C)) * 0.05).astype(np.float32)
               for _ in range(128)]
_KERNS_OIHW = [np.ascontiguousarray(k.transpose(3, 2, 0, 1)) for k in _KERNS_HWIO]

_DN_NCHW = ("NCHW", "OIHW", "NCHW")   # fixable by ConvCanonicalization -- but only in the entry
_DN_NHWC = ("NHWC", "HWIO", "NHWC")   # already canonical, so nothing to fix anywhere

# Trip count is irrelevant to compile cost (a while body is compiled once); kept small so the
# runtime measurement stays cheap and real.
_SCAN_LEN = 4
_DS = (16, 32, 64, 128)


def _convs_in_scan(x, D: int, dn, kerns):
    """A lax.scan whose body holds D chained convolutions. Exactly 2 jaxpr equations."""

    def body(carry, _):
        h = carry
        for i in range(D):
            h = lax.conv_general_dilated(h, kerns[i], (1, 1), "SAME", dimension_numbers=dn)
            h = jnp.tanh(h)
        return h, None

    out, _ = lax.scan(body, x, None, length=_SCAN_LEN)
    return out.sum()


def _convs_in_entry(x, D: int, dn, kerns):
    """The same D convolutions unrolled in the entry computation, where the pass DOES run."""
    h = x
    for i in range(D):
        h = lax.conv_general_dilated(h, kerns[i], (1, 1), "SAME", dimension_numbers=dn)
        h = jnp.tanh(h)
    return h.sum()


CASES = {}
for _d in _DS:
    CASES[f"convscan_nchw_d{_d}"] = (
        functools.partial(_convs_in_scan, D=_d, dn=_DN_NCHW, kerns=_KERNS_OIHW),
        (_X_NCHW,),
        f"gap 1: D={_d} NCHW convs inside a lax.scan body -- ConvCanonicalization never visits a "
        "non-entry computation, so they fall to the elemental emitter",
    )
    CASES[f"convscan_nchw_d{_d}_control"] = (
        functools.partial(_convs_in_scan, D=_d, dn=_DN_NHWC, kerns=_KERNS_HWIO),
        (_X_NHWC,),
        f"same scan, same D={_d} convs, same buffers and FLOPs, dimension_numbers already "
        "canonical so no pass is needed; identical jaxpr eqn and HLO line counts",
    )

# The SAME convolutions in the entry computation, where the pass does run. This pair is the
# location control: it holds dimension_numbers fixed and moves only where the instruction lives.
for _d in (64,):
    CASES[f"conventry_nchw_d{_d}"] = (
        functools.partial(_convs_in_entry, D=_d, dn=_DN_NCHW, kerns=_KERNS_OIHW),
        (_X_NCHW,),
        f"location control: the same D={_d} NCHW convs UNROLLED IN THE ENTRY COMPUTATION, where "
        "ConvCanonicalization fixes them for seven HLO lines total -- measured 1.06x, not 8x",
    )
    CASES[f"conventry_nchw_d{_d}_control"] = (
        functools.partial(_convs_in_entry, D=_d, dn=_DN_NHWC, kerns=_KERNS_HWIO),
        (_X_NHWC,),
        f"same D={_d} convs in the entry computation with canonical dimension_numbers",
    )
