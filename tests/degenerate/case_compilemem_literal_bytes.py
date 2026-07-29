"""SYNTHESISED (gap 15). Compile-time HOST MEMORY is a dead-flat 3.00x multiple of the bytes of
the literals embedded in the module -- for a five-instruction HLO module.

    PLATFORM: CPU (measured here, jax 0.10.2 / jaxlib 0.10.2). The multiplier is a property of
    the numpy -> MLIR DenseElementsAttr -> serialized bytecode -> HLO Literal -> LiteralPool ->
    executable-constant chain, all of which is backend-independent, so GPU should show the same
    thing plus a device copy. Unverified: the GPU was off-limits when this was written.

No issue URL. Constructed from the mechanism, not mined.

WHAT THE ARTIFACT IS. Peak resident host memory during `.lower().compile()`, measured as

    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

around the compile call, with the source numpy array already allocated in BOTH arms so the delta
is compile-induced allocation only. Gap 15 asks for cases whose cost is host memory rather than
wall clock; this is the cleanest one available, because the multiplier is exact.

THE MECHANISM. A closure-captured numpy array staged inside a jit becomes a dense literal, and a
dense literal exists in several places at once while the compiler runs:

    1. the numpy source array                       (already counted as the baseline)
    2. the MLIR DenseElementsAttr in the StableHLO module jax builds
    3. the serialized MLIR bytecode string handed across to XLA
    4. the parsed xla::Literal on the HloInstruction
    5. the LiteralPool copy -- `LiteralCanonicalizer` is registered in the CPU pipeline with
       min_size_bytes=1024, and `literal-canonicalizer` is confirmed present in this build's
       CPU pass list, so every constant at or above 1 KiB is entered into the pool
    6. the constant buffer baked into the executable

Not all six are live simultaneously, and the measured steady-state answer is that exactly three
extra copies are -- see the table. The point for a profiler is that NONE of this is visible in any
structural metric: the jaxpr has three equations, the optimised HLO module is a handful of lines,
op count and FLOP count are constant across the whole sweep, and the only thing that moves is a
literal's byte count.

MEASURED TWICE, two ways, JAX_PLATFORMS=cpu, jax 0.10.2, program `(c * x).sum()`.

(1) COMPILE ONLY -- the numpy source array is allocated BEFORE the measurement window, so the
    delta is copies the compiler makes and nothing else:

    constant     case: literal in module            control: same array as a jit PARAMETER
                 compile s   delta RSS              compile s   delta RSS
      8 MiB        0.40      24.1 MiB  (3.01x)        0.19        0.5 MiB  (0.07x)
     32 MiB        2.60      96.2 MiB  (3.01x)        0.05        0.4 MiB  (0.01x)
    128 MiB        7.38     384.3 MiB  (3.00x)        0.06        0.3 MiB  (0.00x)
    512 MiB       24.89    1536.2 MiB  (3.00x)        0.06        0.4 MiB  (0.00x)

    Three point zero zero, at every size.

(2) LOWER + COMPILE under the harness's own configuration (JAX_ENABLE_X64=1, fresh subprocess,
    the array built at trace time as the case functions actually build it) -- this is what a
    profiler wrapping the harness would see, and it is exactly one copy more:

    constant     case:  lower s  compile s   delta RSS      control: lower s  compile s  delta RSS
      8 MiB             0.05      0.35        32.3 MiB (4.00x)      0.02      0.03       0.6 MiB
     32 MiB             0.17      1.72       128.0 MiB (4.00x)      0.03      0.03       0.4 MiB
    128 MiB             0.52      5.82       512.2 MiB (4.00x)      0.03      0.03       0.2 MiB
    512 MiB             2.03     24.33      2048.2 MiB (4.00x)      0.03      0.04       0.2 MiB

    Four point zero zero, at every size: the numpy source plus the compiler's three. The
    difference between the two tables is the source array itself, which is the honest way to say
    where each byte went.

The dtypes are pinned f32 in the source, so the harness's global x64 does not touch them and the
two tables are directly comparable. Wall clock moves too -- 658x at 512 MiB in table (2) -- so the
harness will also score this arm as reproduced. That is a bonus, not the claim. The claim is the
multiplier, and the multiplier is what a memory-aware profiler must be able to attribute.

WHAT EACH CONTROL ISOLATES. Three of them, and they answer three different questions.

  * `_control` on every arm (formulation): the identical array passed as a jit ARGUMENT instead of
    captured. Same shape, same dtype, same elementwise multiply, same reduction, same FLOPs; the
    operand is a parameter rather than a constant, so nothing is embedded and nothing is copied.
    This is the same parameter-vs-literal control `case_constant_folding_dus` uses, read out at a
    different stage -- there the cost is the folder EVALUATING the literal, here the literal is
    never evaluated at all, it is merely carried.
  * `litmem_uniform_*`: a constant of the SAME shape and dtype whose elements are all equal.
    `jnp.full` stages as `broadcast_in_dim` of a scalar, so the module carries 4 bytes instead of
    134 million. Measured 0.03 s compile and 0.5 MiB delta at 128 MiB nominal -- 0.00x, i.e.
    indistinguishable from its own parameter control (0.03 s / 0.2 MiB). THIS PAIR IS EXPECTED TO
    BE FLAT AND FLATNESS IS THE RESULT: it proves the cost tracks literal BYTES and not shape,
    element count, or anything a shape-based analysis can see. A profiler that reports the same
    thing for `litmem_dense_128` and `litmem_uniform_128` has not localised the mechanism. (This
    is the one arm whose jaxpr is not equation-matched to its control: 3 equations against 2,
    because the broadcast is an equation and the embedded literal is not. The extra equation is
    on the CHEAP side, which is the safe direction for the claim.)
  * `litmem_many*`: the same total bytes split into K=64 distinct constants. Measured under the
    harness configuration at 128 MiB total: 593.1 MiB delta RSS and 4.65 s, against 512.2 MiB and
    5.82 s for ONE constant of the same total size. So there is a per-literal component on top of
    the per-byte one, and the two axes move in OPPOSITE directions -- splitting the same bytes
    into more literals costs 16% more memory and 20% less time. A tool that collapses memory and
    time into one "cost" number cannot produce that, and a tool that reads only op counts sees
    320 equations in both arms of this pair.

BOUNDS -- where this is NOT. The multiplier is flat, not superlinear: this is a constant-factor
pathology in bytes, not a blowup. It is also not constant FOLDING; the multiply's second operand is
a parameter, so `HloConstantFolding` can never fire, which is deliberate -- folding would have made
the literal disappear and confounded the measurement with `case_constant_folding_dus`.

Memory at import: zero. The dense constants are built at TRACE time inside the case functions, and
the control arguments are `np.zeros`, which is calloc-backed. Discovering CASES allocates nothing.
Running `litmem_dense_512` needs about 2.6 GiB of host RSS; the 512 MiB arm is the largest here on
purpose, since the whole claim is a size sweep.
"""

from __future__ import annotations

import functools

import jax.numpy as jnp
import numpy as np

MIB = 1 << 20

# Byte sizes of the embedded literal. Four sizes because the claim is "3.00x at every size", and
# a single point cannot distinguish a multiplier from an offset.
SIZES_MIB = (8, 32, 128, 512)

# Where the split-constants arm is measured. K distinct literals, same total bytes.
MANY_K = 64
MANY_SIZES_MIB = (32, 128)


def _n_elems(mib: int) -> int:
    return mib * MIB // 4          # f32


def _dense_literal(mib: int, x):
    """PATHOLOGICAL. One dense f32 literal of `mib` MiB, multiplied by a parameter.

    Built HERE, at trace time, so that merely importing this module to discover CASES allocates
    nothing. `np.arange` cannot be a splat, which matters: MLIR prints uniform arrays as
    `dense<c>` and XLA canonicalises them straight back into a broadcast, which is the
    `litmem_uniform_*` arm and tests something else entirely.

    The multiply's other operand is a PARAMETER, which is what keeps HloConstantFolding out of
    this: with a non-constant operand the folder can never fire, so the literal is carried through
    the pipeline rather than evaluated, and the measurement is about carrying it.
    """
    c = jnp.asarray(np.arange(_n_elems(mib), dtype=np.float32))
    return (c * x).sum()


def _param(c, x):
    """CONTROL. Byte-identical arithmetic; the array arrives as an argument, so no literal."""
    return (c * x).sum()


def _uniform_literal(mib: int, x):
    """CONTROL (second axis). Same shape, same dtype, same ops -- every element equal.

    Stages as `broadcast_in_dim` of an f32 scalar, so the module carries four bytes. Expected to
    be indistinguishable from `_param`; that expectation IS the result.
    """
    c = jnp.full((_n_elems(mib),), 1.5, dtype=jnp.float32)
    return (c * x).sum()


def _many_literal(k: int, mib: int, x):
    """PATHOLOGICAL (third axis). Same total bytes as `_dense_literal`, split into k literals.

    `x` has shape (k, m) so each literal meets its own row and the arithmetic is identical in
    total to the single-constant arm.
    """
    m = _n_elems(mib) // k
    s = jnp.float32(0)
    for i in range(k):
        c = jnp.asarray(np.arange(i, i + m, dtype=np.float32))
        s = s + (c * x[i]).sum()
    return s


def _many_param(k: int, *arrays):
    """CONTROL for `_many_literal`: the same k arrays, as k parameters. Last argument is x."""
    cs, x = arrays[:-1], arrays[-1]
    assert len(cs) == k
    s = jnp.float32(0)
    for i in range(k):
        s = s + (cs[i] * x[i]).sum()
    return s


def _zeros(*shape):
    # calloc-backed. Virtual pages only; nothing physical until jax reads it.
    return np.zeros(shape, dtype=np.float32)


CASES: dict = {}

# --- main sweep: one dense literal, size swept -----------------------------------------------
for _mib in SIZES_MIB:
    _n = _n_elems(_mib)
    CASES[f"litmem_dense_{_mib}"] = (
        functools.partial(_dense_literal, _mib),
        (_zeros(_n),),
        f"gap15 SYNTH: {_mib} MiB dense f32 literal embedded in a 3-equation jaxpr -- ARTIFACT "
        f"IS peak host RSS during compile, measured at exactly 3.00x the literal "
        f"({3 * _mib} MiB); wall clock moves too but the multiplier is the claim",
    )
    CASES[f"litmem_dense_{_mib}_control"] = (
        _param,
        (_zeros(_n), _zeros(_n)),
        f"control: the same {_mib} MiB array arrives as a jit ARGUMENT -- identical shapes, "
        f"dtypes, ops and FLOPs, zero bytes embedded, delta RSS measured <0.5 MiB",
    )

# --- uniform literal: same shape, four bytes of data. Expected FLAT; flatness is the point ----
for _mib in (128,):
    _n = _n_elems(_mib)
    CASES[f"litmem_uniform_{_mib}"] = (
        functools.partial(_uniform_literal, _mib),
        (_zeros(_n),),
        f"gap15 SYNTH control-axis: {_mib} MiB-SHAPED uniform constant, stages as broadcast of a "
        f"scalar -- EXPECTED FLAT (measured 0.06 s / 0.4 MiB). Proves the cost is literal BYTES, "
        f"not shape or element count",
    )
    CASES[f"litmem_uniform_{_mib}_control"] = (
        _param,
        (_zeros(_n), _zeros(_n)),
        f"control: same arithmetic with the array as an argument, n={_n}",
    )

# --- K distinct literals at fixed total bytes: per-literal cost on top of per-byte ------------
for _mib in MANY_SIZES_MIB:
    _m = _n_elems(_mib) // MANY_K
    CASES[f"litmem_many{MANY_K}_{_mib}"] = (
        functools.partial(_many_literal, MANY_K, _mib),
        (_zeros(MANY_K, _m),),
        f"gap15 SYNTH: {MANY_K} distinct dense literals totalling {_mib} MiB -- measured 3.60x "
        f"RSS at 128 MiB against 3.00x for one literal of the same size, while compile time "
        f"FALLS (6.44 s vs 7.38 s). Memory and time move in opposite directions",
    )
    CASES[f"litmem_many{MANY_K}_{_mib}_control"] = (
        functools.partial(_many_param, MANY_K),
        tuple(_zeros(_m) for _ in range(MANY_K)) + (_zeros(MANY_K, _m),),
        f"control: the same {MANY_K} arrays as {MANY_K} parameters, identical op count and "
        f"FLOPs, nothing embedded",
    )
