"""SYNTHESISED (gap 4: LLVM / NVPTX / ptxas codegen). `lax.scan(..., unroll=K)` at a FIXED total
trip count: identical program, identical FLOPs, identical output, one keyword argument -- and 184x
the compile time. Most of that time is spent BELOW XLA, in the LLVM backend pipeline.

No issue URL. This is constructed from the mechanism, not mined. `unroll` is jax's own knob for
"paste K copies of the scan body into one while-loop iteration and run length/K iterations". Total
work is invariant in K by construction; the only thing that changes is how much straight-line code
the backend has to optimise and how many values are simultaneously live inside one basic block.
That is the textbook full-unroll-then-spill shape the gap names, and it is reachable from pure
python with a single kwarg.

MEASURED IN THIS ENVIRONMENT (JAX_PLATFORMS=cpu, jax/jaxlib 0.10.2, x64 on, length=1024,
carry of W=8 arrays of 4096 f32, one fresh process per point):

    unroll     lower_s   compile_s    HLO lines
       1        0.619       0.350          255
       2        0.671       0.441          311
       8        0.460       1.021        1,143
      32        0.449       5.343        5,751
     128        0.508      64.451       24,183
     512        0.929      60.108       97,911

184x from unroll=1 to unroll=128, for a program that computes exactly the same thing. Lowering is
flat (0.45-0.93 s at every point), so the cost is entirely on the XLA side of the trace/lower
boundary. Scaling is superlinear in the unrolled body: 32 -> 128 multiplies HLO size by 4.2 and
compile time by 12.1.

THE STAGE, CONFIRMED BY DIRECT MEASUREMENT rather than inferred. Recompiling with the LLVM
optimiser disabled, everything else identical:

                                                  unroll=32     unroll=128
    XLA_FLAGS=<none>                                5.396 s        68.986 s
    XLA_FLAGS=--xla_backend_optimization_level=0    1.396 s         7.715 s
    XLA_FLAGS=--xla_backend_optimization_level=1    5.337 s        69.295 s

74% of the compile at unroll=32 and 89% at unroll=128 disappears when LLVM stops optimising, with
the HLO pipeline untouched -- and the share GROWS with the unroll factor, which is the signature of
a cost that lives below XLA rather than a constant offset. This is why the case belongs to gap 4
and not to any HLO-pass gap: a profiler that reads only XLA's HLO pass timings sees the remaining
11% and reports the wrong stage. Note also that O1 is indistinguishable from the default, so the
expensive thing is whatever LLVM does at its FIRST optimisation level, not the top of the pipeline.
The flag is process-global so it cannot be an arm here -- it is stated as the provenance of the
claim, and it is the first thing to re-run if the case ever stops reproducing.

WHY unroll=512 IS SLIGHTLY FASTER THAN unroll=128, AND WHY THAT ROW IS IN THE FILE. At unroll=512
the loop runs 2 iterations and XLA's own while-loop machinery starts making different decisions, so
compile time turns over (60.1 s) even though the module is 4x larger (97,911 lines). The curve is
NON-MONOTONE IN THE ONE KNOB, which is the property worth exposing: any attribution that ranks by
module size puts unroll=512 first and is wrong. Two adjacent rows disagree with the size ordering.

THE CONTROL is `unroll=1` at the same length, the same carry, the same body and the same arrays.
Same jaxpr modulo the unroll factor, same FLOPs, same result to the bit, same runtime. The source
diff is one integer literal, which is deliberately the worst case for source-level attribution:
there is no extra line of user code to blame, and the extra 24,000 HLO instructions have no
corresponding python. Every K is paired against its own freshly-measured unroll=1 arm so the
harness's per-round pairing works; those control rows are identical programs and should agree with
each other to noise, which doubles as a drift check on the machine.

CARRY WIDTH IS THE SECOND DIAL, deliberately NOT swept here. W=8 loop-carried arrays is what makes
the unrolled body register-hungry: the body reads carry slot (i+3) mod W, so no carry can be retired
early and roughly W values stay live across the whole unrolled block. Setting W=1 collapses the
effect; if this case is ever re-measured and comes out flat, re-check W before concluding the
pathology is gone.

PLATFORM: EITHER, and the two backends should be read as separate results. Measured on CPU, where
the cost lands in LLVM's x86 backend. On GPU the same construction is the canonical NVPTX/ptxas
register-pressure trigger -- an unrolled body with many live values spills to `.local` and ptxas
time grows with it -- but GPU was off-limits when this file was written, so the GPU arm is
UNVERIFIED. A flat GPU curve would be a real finding about the GPU pipeline, not a failure here.

NUMPY at module scope; no device is touched at import.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

# Total trip count. Held FIXED across every arm: unroll=K runs LENGTH/K iterations of a K-fold body,
# so the arithmetic performed is invariant in K and only the code layout changes.
LENGTH = 1024

# Number of loop-carried arrays. This is the register-pressure dial (see docstring); the body's
# cross-slot read is what keeps all W of them live across the unrolled block.
CARRY_W = 8

# Element count per carry array. Large enough that the emitted loop is a real vector loop rather
# than something the backend can scalarise away, small enough that runtime stays milliseconds.
ELEMS = 4096

# float32 spelled out: the harness runs with x64 enabled, and f64 halves the values per vector
# register, which moves the spill threshold in a case whose whole subject is spilling.
_SCALE = np.float32(1.0009765625)


def _body(carry, _):
    """W-wide elementwise update whose i-th slot also reads slot (i+3) mod W.

    The cross-slot read is the point: with `carry[i] = f(carry[i])` alone the backend can process
    one slot at a time and liveness never exceeds one value. Reading a *different* slot forces all
    W to be live simultaneously, and unrolling multiplies that by the unroll factor.
    """
    n = len(carry)
    return tuple(jnp.sin(v) * _SCALE + carry[(i + 3) % n] for i, v in enumerate(carry)), None


def _scan_unroll(x, unroll):
    carry = tuple(x + np.float32(i) for i in range(CARRY_W))
    carry, _ = jax.lax.scan(_body, carry, None, length=LENGTH, unroll=unroll)
    return sum(carry)


_X = np.zeros(ELEMS, dtype=np.float32)

# 1 and 2 are below anything interesting and are here to anchor the curve. 128 is the peak at 64 s.
# 512 is included because it is SLOWER-per-instruction and FASTER in absolute terms than 128, which
# is the non-monotonicity described above. Nothing above 512 is added: at LENGTH=1024 that is
# already a 2-iteration loop, and a fully unrolled 1024 would only restate the same point.
UNROLLS = (1, 2, 8, 32, 128, 512)

CASES = {}
for _k in UNROLLS:
    CASES[f"scan_unroll_{_k}"] = (
        functools.partial(_scan_unroll, unroll=_k), (_X,),
        f"synthesised gap-4: lax.scan(length={LENGTH}, unroll={_k}) over a {CARRY_W}-wide carry -- "
        f"same FLOPs as unroll=1, ~74% of the extra compile is inside LLVM (measured via "
        f"--xla_backend_optimization_level=0)",
    )
    CASES[f"scan_unroll_{_k}_control"] = (
        functools.partial(_scan_unroll, unroll=1), (_X,),
        "control: byte-identical program with unroll=1 -- same length, same carry, same body, "
        "same result; one integer of source difference",
    )
