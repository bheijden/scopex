"""SYNTHESISED (gaps 11 + 2, landing in gap 4). Fan-out topology alone moves compile time 4x with
a BYTE-IDENTICAL jaxpr, a BYTE-IDENTICAL optimised HLO module, and BYTE-IDENTICAL unoptimised
LLVM IR. The entire divergence happens inside LLVM.

    PLATFORM: CPU (measured here, jax 0.10.2 / jaxlib 0.10.2, JAX_PLATFORMS=cpu).
    GPU UNMEASURED -- the GPU was owned by another investigation. On GPU the same shape is the
    documented trigger for multi-output-fusion sibling merging, so the mechanism there is likely
    to be a different one with the same knob; read the CPU numbers as CPU only.

No issue URL. Constructed from the audit's gap 11 ("nothing is WIDE: thousands of mutually
independent ops") and gap 2 (fan-out graphs as the natural scheduling trigger). What it actually
found is gap 4 -- compile time spent BELOW XLA -- which is why it earns a slot: it is a width case
whose cost is invisible to every HLO-level metric there is.

WHY THIS IS THE ADVERSARIAL CASE FOR AN HLO-LEVEL PROFILER. Between the two arms, ALL of the
following were measured EQUAL, not assumed:

    jaxpr equation count            512 / 1024 / 2048 / 4096   equal at every P
    optimised HLO line count        930 / 1826 / 3618 / 7202   equal at every P
    optimised HLO opcode histogram  identical at P=256 (512 constant, 512 broadcast,
                                    256 maximum, 256 add, 2 parameter, 1 fusion)
    XLA pass sequence               identical at P=128: 32 dump files under
                                    --xla_dump_hlo_pass_re=.*, same names, same order,
                                    i.e. the same passes changed the module in both arms
    UNOPTIMISED LLVM IR             1605 lines in BOTH arms at P=512

and these diverged:

    optimised LLVM IR    P=512    1597 lines (case)   vs   1083 lines (control)
    object file bytes    P=512    257,904 (case)      vs    40,688 (control)   = 6.3x
    compile seconds      P=512      11.109            vs      2.901            = 3.83x

A profiler that sums HLO pass durations, counts instructions, counts fusions, or diffs the
optimised module will report that these two programs are the same program. They are not: one of
them makes LLVM emit six times as much machine code.

THE ONE VARIABLE. Both arms build P chained producers and P consumers:

    p[0] = barrier(max(x,      c_0))
    p[i] = barrier(max(p[i-1], c_i))                    i = 1 .. P-1
    q[i] = atan2( <<HERE>>, d_i )                       i = 0 .. P-1
    out  = p[P-1] + q[0] + q[1] + ... + q[P-1]

    case     <<HERE>> = p[i]     -- every producer has exactly 2 users; fan-out is SPREAD
    control  <<HERE>> = p[0]     -- p[0] has P+1 users, every other producer has 1;
                                    fan-out is CONCENTRATED on one node

Same number of equations, same primitives, same shapes, same dtype, same constants, same output
shape. The only thing that changes is which value the first operand of each `atan2` names -- an
integer index, not an op.

THE MECHANISM. XLA fuses the whole program into ONE kernel in both arms (measured: exactly one
`fusion` instruction, `temp_size_in_bytes` 0, in both). The elemental emitter then writes that
fusion out as a loop nest, and the two loop bodies have the SAME unoptimised IR size because they
have the same instruction count. LLVM is where they part company. In the control every `atan2`
reads the same SSA value, so its argument is loop-invariant and common to all P calls; LLVM hoists
and CSEs it and the polynomial expansion of `atan2` collapses. In the case every `atan2` reads a
different value, nothing is common, and LLVM's optimiser, vectoriser and instruction selector each
have to chew through P independent transcendental expansions. Compile cost tracks the object code
LLVM ends up emitting, which is 6.3x larger.

That makes this case the mirror image of case_fusion_pass_barrier_chain.py in this directory. That
one has MORE HLO and compiles faster; this one has EXACTLY THE SAME HLO and compiles 4x slower.
Neither is explicable by size.

RELATION TO case_compilemem_peak_live_fanout.py, which is also about use counts. That file varies
whether a value has one user or two in order to move XLA's fusion decision, and its artifact is
`temp_size_in_bytes`. Here BOTH arms fuse into a single kernel and BOTH report zero temp bytes;
use count does not change the fusion structure at all, and the artifact is object-code size and
wall-clock. The two files should be attributed to different stages by any tool that is working.

MEASURED, JAX_PLATFORMS=cpu, jax 0.10.2, x = (64,) f32, one process per arm, arms run
back-to-back:

    P        case (spread)   control (concentrated)   ratio    eqns (both)   HLO lines (both)
     128        1.056 s            0.526 s            2.01x        512             930
     256        3.914 s            1.090 s            3.59x       1024            1826
     512       11.109 s            2.901 s            3.83x       2048            3618
    1024       21.708 s            5.501 s            3.95x       4096            7202

The ratio rises and then flattens near 4x, and the case arm's own growth is superlinear in P
(1.056 -> 21.708 is 20.6x for an 8x change in P, exponent ~1.45) while the control's is close to
linear (0.526 -> 5.501, 10.5x for 8x, exponent ~1.13). So the sweep separates "this program is
bigger" from "this program is shaped worse": the two arms are the same size at every P.

WHY THE SWEEP IS NOT OPTIONAL. At P=128 the gap is 2x and the absolute compile is half a second --
below the harness floor and easily written off as noise. Only the sweep shows that the gap widens
and that the exponents differ. P is capped at 1024 because 2048 would put the case arm near 90 s
per measurement for no new information.

The `barrier` in the producer chain is `lax.optimization_barrier`, and it is present in BOTH arms
in identical count. It is not part of the mechanism -- it was in the program when these numbers
were taken and the optimised HLO shows XLA removes every one of them before codegen (no
`opt-barrier` opcode survives in either arm) -- so it is kept only so the shipped program is the
program that was measured.
"""

from __future__ import annotations

import functools

import jax.numpy as jnp
import numpy as np
from jax import lax

# NUMPY at module scope: materialising a jax array at import would claim a device before the
# harness has chosen one. jax.jit accepts numpy arrays.
_M = 64
_X = np.linspace(0.1, 0.9, _M).astype(np.float32)

# Distinct per-op constants. In the producer chain they stop the algebraic simplifier telescoping
# the maxima; in the consumers they stop CSE collapsing the P atan2 calls in the CONTROL arm,
# which would otherwise make the control trivially cheap for the wrong reason.
_CP = np.linspace(0.5, 1.5, 4096 + 8).astype(np.float32)
_CQ = np.linspace(1.5, 2.5, 4096 + 8).astype(np.float32)

_PS = (128, 256, 512, 1024)


def _fanout(x, P: int, spread: bool):
    """P producers, P consumers. `spread` selects which producer each consumer reads."""
    ps = []
    t = x
    for i in range(P):
        t = lax.optimization_barrier(jnp.maximum(t, float(_CP[i])))
        ps.append(t)
    # THE ONE VARIABLE: ps[i] (every producer used twice) vs ps[0] (one producer used P+1 times).
    qs = [jnp.arctan2(ps[i] if spread else ps[0], float(_CQ[i])) for i in range(P)]
    acc = ps[-1]
    for q in qs:
        acc = acc + q
    return acc


CASES = {}
for _p in _PS:
    CASES[f"fanout_spread_p{_p}"] = (
        functools.partial(_fanout, P=_p, spread=True),
        (_X,),
        f"gap 11/4: P={_p} producers each read TWICE -- spread fan-out; identical HLO to the "
        "control, ~6x more machine code out of LLVM",
    )
    CASES[f"fanout_spread_p{_p}_control"] = (
        functools.partial(_fanout, P=_p, spread=False),
        (_X,),
        f"same P={_p} equations, same shapes, same HLO line count and opcode histogram; every "
        "consumer reads ps[0] instead of ps[i], so LLVM can CSE the shared operand",
    )
