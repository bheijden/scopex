"""SYNTHESISED (gap 2: HLO SCHEDULING / BUFFER ASSIGNMENT). Written to test gap 2's hypothesis and
it FALSIFIED it: at a byte-identical HLO line count, making N simultaneously-live buffers all
DIFFERENT sizes instead of all the SAME size leaves buffer assignment's work bit-for-bit unchanged
(1540 allocations, peak temp within 1.2%) and instead produces 2x as many kernel modules, 5.3x as
much optimised LLVM IR and 2.7x as much machine code.

    PLATFORM: CPU (measured here, jax 0.10.2 / jaxlib 0.10.2, JAX_PLATFORMS=cpu).
    GPU UNMEASURED -- the GPU was owned by another investigation when this was written.

No issue URL. This file started as a probe for two mined XLA commits about quadratic behaviour in
buffer assignment -- openxla/xla 3b6b7903 (BufferIntervalTree rebalanced as a treap; landed
2026-07-28, ELEVEN DAYS after the jax 0.10.2 branch cut, so the unbalanced BST is live in this
jaxlib) and 172ec6fd ("the default O(N * M) naive allocation scan ... leading to O(N^2)
worst-case"). Neither reproduced. What the probe found instead is a different stage entirely, and
the negative half is as useful as the positive half.

THE DESIGN. N parts, each a slice of one parameter passed through `atan2` and an
`optimization_barrier`, all consumed by ONE N-ary `concatenate` -- so all N buffers are
simultaneously live, by construction, in every arm. Three arms differ ONLY in how long each part
is:

    equal     (control)   every part is 512 elements
    bitrev    (case)      part lengths spread over 512 +/- 256, assigned in BIT-REVERSED
                          program order -- the arrangement predicted to scramble the heap
                          simulator's insertion order
    monotone  (case)      the same set of lengths, assigned in INCREASING program order --
                          the arrangement predicted to make insertion order degenerate

Total bytes are equal by construction (the lengths are symmetric about 512). Slice offsets differ
per part in every arm so CSE cannot collapse the parts in the equal arm for the wrong reason.

MEASURED EQUAL ACROSS ALL THREE ARMS, at N=256 -- these are deterministic, not timings:

    jaxpr equations                770
    optimised HLO lines           9470
    BufferAllocation count        1540
    memory_analysis().temp_size_in_bytes   524,352 (equal) / 530,464 (both distinct arms)

Buffer assignment produces the SAME NUMBER OF ALLOCATIONS and essentially the same peak. Gap 2's
premise -- that peak-live-buffer structure at fixed op count is what costs -- is false here, and
that is a measurement, not an opinion.

MEASURED DIFFERENT, same N=256, same deterministic dumps (--xla_dump_to):

                              equal        bitrev        monotone
    kernel modules             259           530             530
    unoptimised LLVM IR      18,968        45,816          45,811   lines
    optimised LLVM IR        19,765       104,743         104,846   lines
    object file bytes       471,336     1,293,832       1,293,976

bitrev and monotone agree to within 0.1% on every one of these. INSERTION ORDER DOES NOT MATTER;
SIZE DIVERSITY DOES. That is the clean refutation of the interval-tree hypothesis, and it is why
both distinct-size arms are shipped rather than just one -- with only one of them the file could
not tell the two hypotheses apart.

THE MECHANISM. XLA deduplicates identical fusion COMPUTATIONS. When all N parts have the same
shape, the N per-part fusions are structurally identical and collapse: 256 parts produce 259
kernel modules. When every part has a different length, every fusion has a unique shape, nothing
collapses, and 256 parts produce 530 kernel modules whose loops have 256 distinct trip counts --
so LLVM makes 256 separate unroll and vectorisation decisions instead of reusing one. The 5.3x in
optimised IR against only 2.4x in unoptimised IR is that second effect: it is not just that there
is more code, it is that LLVM cannot amortise its work across shapes.

So the compile-time knob here is SHAPE DIVERSITY AT FIXED BUFFER COUNT, which nothing else in the
corpus varies, and the cost lands below HLO. Together with case_width_fanout_llvm_codegen.py in
this directory -- a completely different construction that also holds HLO fixed and also lands in
LLVM -- it makes the same point twice by independent routes: on XLA:CPU, graph-structure changes
at fixed HLO size are paid for in codegen, not in scheduling or buffer assignment.

WALL CLOCK, AND WHY IT IS THE WEAKER HALF OF THIS FILE. Two runs, JAX_PLATFORMS=cpu, jax 0.10.2,
one process per arm:

    run A (quiet box)      N=128   equal 1.551 s   bitrev 3.646 s   2.35x
                           N=256   equal 3.217 s   bitrev 7.155 s   2.22x
                           N=512   equal 12.646 s  bitrev 16.306 s  1.29x

    run B (box contended   N=128   equal 2.852 s   bitrev 4.081 s   monotone 7.688 s
    by other compiles)     N=256   equal 8.518 s   bitrev 8.381 s   monotone 9.078 s

Run B's absolute numbers are 2-3x run A's for the same programs, and its N=256 ratios collapse to
1.0x. The seconds in this file are therefore NOISE-LIMITED and should not be quoted; the
deterministic artifacts above should. The N=512 ratio of 1.29x in run A also suggests the
wall-clock effect decays as other per-instruction costs grow, even though the object-size ratio
does not. Anyone re-measuring this file should check `xla_dump` object bytes first and treat the
timings as corroboration.

HOW THE HARNESS WILL SCORE IT. The case arms clear the 3 s floor from N=256 but the ratio is
~2.2x at best, well under MIN_VS_CONTROL of 10, so expect "no (2.2x control)" or worse under
contention. THAT IS THE CORRECT VERDICT for the wall-clock artifact. The file's value is the
falsified buffer-assignment hypothesis and the deterministic codegen numbers, both of which live
in this docstring and in `--xla_dump_to` output rather than in the harness's stopwatch.
"""

from __future__ import annotations

import functools

import jax.numpy as jnp
import numpy as np
from jax import lax

# NUMPY at module scope: a jax array here would claim a device at import, before the harness has
# chosen one.
_XLEN = 4096
_X = np.linspace(0.1, 0.9, _XLEN).astype(np.float32)
# Distinct per-part constants so the parts cannot be CSE'd together in the equal-size arm, which
# would make that arm cheap for a reason other than the one under test.
_C = np.linspace(0.5, 1.5, 8192).astype(np.float32)

_L = 512          # base part length; the equal arm uses this for every part
_SPAN = 256       # distinct arms spread lengths over _L +/- _SPAN, symmetric so bytes match

_NS = (128, 256, 512)


def _bitrev(i: int, n: int) -> int:
    """Bit-reverse i within [0, n). n must be a power of two."""
    bits = n.bit_length() - 1
    return int(format(i, f"0{bits}b")[::-1], 2)


def _parts_concat(x, N: int, mode: str):
    """N slices -> atan2 -> barrier, all consumed by ONE N-ary concatenate.

    Every arm has the same equation count, the same HLO line count and the same number of
    simultaneously live buffers. `mode` changes only how long each part is.
    """
    step = 2 * _SPAN // N
    parts = []
    for i in range(N):
        if mode == "equal":
            ln = _L
        elif mode == "bitrev":
            ln = _L + (_bitrev(i, N) - N // 2) * step
        else:  # "monotone"
            ln = _L + (i - N // 2) * step
        # Distinct offsets in every arm, so the equal arm's parts are not literally identical.
        off = (i * 7) % (_XLEN - ln - 1)
        p = jnp.arctan2(lax.dynamic_slice(x, (off,), (ln,)), float(_C[i]))
        parts.append(lax.optimization_barrier(p))
    return jnp.concatenate(parts, axis=0).sum()


CASES = {}
for _n in _NS:
    CASES[f"shapediv_bitrev_n{_n}"] = (
        functools.partial(_parts_concat, N=_n, mode="bitrev"),
        (_X,),
        f"gap 2 probe, N={_n}: {_n} distinct part lengths in bit-reversed order -- same HLO lines "
        "and same allocation count as the control, 5.3x the optimised LLVM IR",
    )
    CASES[f"shapediv_bitrev_n{_n}_control"] = (
        functools.partial(_parts_concat, N=_n, mode="equal"),
        (_X,),
        f"N={_n} parts of one identical length; same total bytes, same eqn count, same HLO line "
        "count, same allocation count -- the N per-part fusions dedup to ~1 kernel shape",
    )
    CASES[f"shapediv_monotone_n{_n}"] = (
        functools.partial(_parts_concat, N=_n, mode="monotone"),
        (_X,),
        f"gap 2 probe, N={_n}: the SAME {_n} distinct lengths in increasing order -- matches the "
        "bitrev arm to within 0.1% on every deterministic artifact, so insertion order is not "
        "the variable",
    )
    CASES[f"shapediv_monotone_n{_n}_control"] = (
        functools.partial(_parts_concat, N=_n, mode="equal"),
        (_X,),
        f"N={_n} parts of one identical length (same control as the bitrev arm)",
    )
