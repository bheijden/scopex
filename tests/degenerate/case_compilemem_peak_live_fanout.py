"""SYNTHESISED (gap 15). Peak buffer memory, not wall clock, is the artifact: one operand token
moves compiler-reported peak temp memory by 50x at byte-identical jaxpr size.

    PLATFORM: CPU (measured here, jax 0.10.2 / jaxlib 0.10.2). Should also hold on GPU by the
    same fusion rule, unverified -- the GPU was off-limits when this was written.

No issue URL. This is constructed from the mechanism, not mined.

WHAT THE ARTIFACT IS. Not seconds. The number this case exists to move is

    jax.jit(fn).lower(*args).compile().memory_analysis().temp_size_in_bytes

i.e. the size of the scratch allocation XLA's buffer assignment reserves for the module. The
harness measures wall clock and will therefore score every arm here as "no" (the compile-time gap
is only ~2.5x, well under MIN_VS_CONTROL). THAT IS THE POINT OF THE FILE. Gap 15 in the audit --
"compile-time memory as the artifact" -- names exactly this: four mined candidates were dropped
because their cost is memory rather than time, and nothing in the corpus lets a profiler be scored
on a memory signal. A tool that only reads pass durations reports nothing at all here.

THE MECHANISM. XLA will not vertically fuse an elementwise producer that has more than one user,
because doing so duplicates the producer's code into every consumer (the same multi-use guard that
`kAllowedCodeDuplication = 15` and "do not fuse elementwise ops with more than one user" encode).
A value with ONE user is fused into its consumer and never materialises; a value with TWO users is
materialised into its own buffer. So use-count -- a property invisible to op counting, FLOP
counting and jaxpr size -- decides how many buffers the module needs, and therefore the peak.

    pathological   h = sin(h) ; acc = acc + h        h_i is read by sin_{i+1} AND by add_i
                                                     -> 2 users -> no fusion -> N buffers
    control        h = sin(h) ; h  = h   + x         h_i is read by add_i only
                                                     -> 1 user  -> full fusion -> 1 buffer

Both arms are N `sin`, N vector `add`, one `reduce_sum`, all f32, all shape (M,), identical FLOPs.
The only difference is which value the second operand of the add names: the running accumulator
`acc` (pathological) or the function parameter `x` (control). Verified identical jaxpr equation
counts and identical primitive sets:

    N=8   17 eqns / 17 eqns     N=32  65 / 65     N=128  257 / 257
    primitives both arms: {add, reduce_sum, sin}

MEASURED, JAX_PLATFORMS=cpu, jax 0.10.2, M = 2**18 f32 (1.0 MiB per buffer):

    N      temp_size_in_bytes            compile s
           acc (case)   par (control)    acc      par
      8       8.0 MiB       1.0 MiB      0.61     0.16
     16      16.0 MiB       1.0 MiB      0.40     0.22
     32      32.0 MiB       1.0 MiB      0.55     0.21
     64      50.0 MiB       1.0 MiB      1.11     0.36
    128      50.0 MiB       1.0 MiB      2.11     0.82
    192      50.0 MiB       1.0 MiB      3.44     1.46
    256      50.0 MiB       1.0 MiB      5.71     2.29

Peak memory: 1x -> 50x. Wall clock: 1x -> 2.5x. The two signals disagree by a factor of 20, which
is the discriminator this file hands the profiler.

RE-MEASURED under the harness's own configuration (JAX_ENABLE_X64=1, fresh subprocess per arm,
`.lower()` and `.compile()` timed separately). Buffers are pinned f32 so x64 does not change any
byte count, only the wall clock:

    arm                            lower s  compile s   temp        run s
    peaklive_fanout_n8               0.024     0.25      8.0 MiB    0.047
    peaklive_fanout_n8_control       0.047     0.18      1.0 MiB    0.018
    peaklive_fanout_n64              0.049     2.20     50.0 MiB    0.920
    peaklive_fanout_n64_control      0.038     0.35      1.0 MiB    0.040
    peaklive_fanout_n256             0.092     6.31     50.0 MiB    3.536
    peaklive_fanout_n256_control     0.141     1.25      1.0 MiB    0.133
    peaklive_p50_n48                 0.057     1.04     48.0 MiB    0.456
    peaklive_p50_n50                 0.049     1.09     50.0 MiB    0.527
    peaklive_p50_n52                 0.051     1.20     50.0 MiB    0.484
    peaklive_fanout_big_n64          0.042     1.19    200.0 MiB    2.457
    peaklive_fanout_big_n64_control  0.043     0.35      4.0 MiB    0.150

Note `p50_n48 -> n50 -> n52`: 48.0 -> 50.0 -> 50.0 MiB. The cap is hit exactly at 50 and the
compile time does not step with it, so a wall-clock reading cannot find the discontinuity at all.
Note also the RUNTIME column: the fan-out arm is 20x slower to execute as well, so the harness's
compile/runtime ratio heuristic is useless here -- only the control comparison means anything, and
only in bytes.

THE 50-BUFFER PLATEAU IS A SEPARATE, DATABLE FINDING. The saturation above is not 50 MiB, it is
FIFTY BUFFERS, exactly, independent of buffer size -- verified by re-running the sweep at four
different M:

    per-buffer   N=16        N=32        N=64        N=128       N=256
     0.125 MiB   2.00 (16)   4.00 (32)   6.25 (50)   6.25 (50)   6.25 (50)
     0.500 MiB   8.00 (16)  16.00 (32)  25.00 (50)  25.00 (50)  25.00 (50)
     1.000 MiB  16.00 (16)  32.00 (32)  50.00 (50)  50.00 (50)  50.00 (50)
     4.000 MiB  64.00 (16) 128.00 (32) 200.00 (50) 200.00 (50) 200.00 (50)
                MiB (buffer count in parentheses)

Linear in N up to 50 buffers, then flat forever. That is a hard structural cap in XLA:CPU, not a
smooth heuristic, and a size sweep is the only way to see it -- one data point at N=32 says
"linear", one at N=256 says "constant". `_p50_*` below brackets it at N=48/50/52/56 so a profiler
can be asked to localise a discontinuity rather than a slope.

The plateau is NOT the memory-fitting machinery: sweeping `jax_memory_fitting_level` over
O0/O1/O2/O3 crossed with `jax_memory_fitting_effort` in {-1.0, 0.0, +1.0} leaves it at exactly
50 MiB in all twelve combinations on CPU. Those dials are inert here, which is consistent with the
CPU pipeline having no rematerialization pass at all (see case_remat_threshold_activation_chain.py,
where that absence is verified pass-by-pass).

WHY THIS SHAPE AND NOT `optimization_barrier`. A barrier forces liveness by fiat and changes the
instruction mix. This forces it through the fusion rule alone, so the two arms remain
op-count-identical. An early draft used `lax.optimization_barrier` on both arms and both went to
N buffers -- the barrier, not the fan-out, was doing the work, and the control isolated nothing.

RELATIONSHIP TO OTHER FILES. This is the CPU-measurable generator for the GPU threshold case in
case_remat_threshold_activation_chain.py: it is the cheapest known way to make peak buffer memory
grow at fixed program size, which is precisely what pushes a GPU module across
HloRematerialization's memory limit.

Memory at import: zero. Args are `np.zeros`, which is calloc-backed -- virtual pages only until
something reads them.
"""

from __future__ import annotations

import functools

import jax.numpy as jnp
import numpy as np

# Elements per activation buffer. 2**18 f32 = 1.0 MiB, the size the numbers above were taken at.
M_1MIB = 1 << 18
# A second buffer size, to prove the plateau is a COUNT and not a byte budget.
M_4MIB = 1 << 20

# Chain lengths. Spread over the linear region and well past the plateau.
DEPTHS = (8, 16, 32, 64, 128, 256)

# Bracket of the 50-buffer cap itself. Below / at / just above / clear of.
PLATEAU_DEPTHS = (48, 50, 52, 56)


def _fanout(n: int, x):
    """PATHOLOGICAL. Every intermediate has two users, so none of them fuse away.

    `h` is the chain, `acc` the accumulator. `h_i` is read by the next `sin` and by `acc + h_i`,
    which is two users, which forbids vertical fusion, which materialises `h_i` into its own
    buffer. Peak temp is therefore n buffers (up to the cap), not one.
    """
    h = x
    acc = x
    for _ in range(n):
        h = jnp.sin(h)
        acc = acc + h
    return acc.sum()


def _inline(n: int, x):
    """CONTROL. Identical op count, shapes, dtypes and FLOPs; one operand token differs.

    The add's second operand is the parameter `x` instead of the accumulator, so every `sin`
    output has exactly ONE user, the whole chain fuses into a single loop, and exactly one
    temporary exists no matter how long the chain is. `x` is a parameter and lives in the argument
    allocation, not in temp, so it does not contribute to `temp_size_in_bytes`.
    """
    h = x
    for _ in range(n):
        h = jnp.sin(h)
        h = h + x
    return h.sum()


def _arg(m: int):
    # calloc-backed: costs no physical memory at import, only when jax touches it.
    return np.zeros(m, dtype=np.float32)


CASES: dict = {}

# --- main sweep: peak temp linear in chain length, at 1 MiB per buffer ------------------------
for _n in DEPTHS:
    CASES[f"peaklive_fanout_n{_n}"] = (
        functools.partial(_fanout, _n),
        (_arg(M_1MIB),),
        f"gap15 SYNTH: n={_n} two-user chain, 1 MiB buffers -- ARTIFACT IS "
        f"memory_analysis().temp_size_in_bytes (measured {min(_n, 50)} MiB), not seconds; "
        f"harness will score this 'no' because wall clock only moves ~2.5x",
    )
    CASES[f"peaklive_fanout_n{_n}_control"] = (
        functools.partial(_inline, _n),
        (_arg(M_1MIB),),
        f"control: n={_n}, identical eqn count/prims/FLOPs, add's operand is the parameter "
        f"instead of the accumulator -> one user per value -> temp stays 1.0 MiB",
    )

# --- same sweep at 4x the buffer size: separates 'fifty buffers' from 'fifty megabytes' -------
for _n in (16, 32, 64, 256):
    CASES[f"peaklive_fanout_big_n{_n}"] = (
        functools.partial(_fanout, _n),
        (_arg(M_4MIB),),
        f"gap15 SYNTH: n={_n} two-user chain at 4 MiB per buffer -- temp measured "
        f"{min(_n, 50) * 4} MiB, i.e. the cap is 50 BUFFERS not 50 MiB",
    )
    CASES[f"peaklive_fanout_big_n{_n}_control"] = (
        functools.partial(_inline, _n),
        (_arg(M_4MIB),),
        f"control: n={_n} at 4 MiB per buffer, one user per value -> temp stays 4.0 MiB",
    )

# --- the discontinuity itself ----------------------------------------------------------------
for _n in PLATEAU_DEPTHS:
    CASES[f"peaklive_p50_n{_n}"] = (
        functools.partial(_fanout, _n),
        (_arg(M_1MIB),),
        f"gap15 SYNTH: n={_n}, brackets the exact 50-buffer plateau in XLA:CPU buffer "
        f"assignment -- temp is linear below it and flat at and above it",
    )
    CASES[f"peaklive_p50_n{_n}_control"] = (
        functools.partial(_inline, _n),
        (_arg(M_1MIB),),
        f"control: n={_n}, one user per value, temp 1.0 MiB on both sides of the plateau",
    )
