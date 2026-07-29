"""xla#35955 -- K DISTINCT GEMM shapes pay K autotunes; K IDENTICAL shapes pay one.

    https://github.com/openxla/xla/issues/35955       (autotuner serializes across instructions)
    PR #37469 (Feb 2026) parallelised candidate compilation; PR #41620 (in-flight dedup) still open.

REPORTED.  XLA:GPU's autotuner walks the module instruction by instruction and, for each GEMM /
GEMM-fusion it has not seen before, compiles and benchmarks a set of candidate implementations
(cuBLAS algorithms, Triton tilings).  Results are cached on a canonicalised key, so a second
instruction with the SAME shape and dtype is free.  Compile cost therefore tracks the number of
DISTINCT fusion shapes -- not the instruction count, not the FLOPs, not the byte traffic.

THE CLEANEST INSTRUCTION-COUNT-INVARIANT PAIR IN THE CORPUS.  Both arms of each pair have:

    * the same number of jaxpr equations and the same number of HLO instructions (K dots,
      K reduces, K-1 adds),
    * the same dtype (float32) and the same operand ranks,
    * total FLOPs matched to within 2% by construction -- the control's single shape m* is chosen
      so that K * m*^3 equals the sum of m_k^3 over the distinct shapes (measured ratios below),
    * near-identical runtime.

The only thing that differs is SHAPE DIVERSITY.  If the slow arm compiles 10x slower with the same
op count and the same arithmetic, the cost is provably per-distinct-shape autotuning and nothing
else.  For an attributor this is the hardest possible discrimination: two programs that are
identical under every static summary you could compute from the IR, differing only in a quantity
(cache-key cardinality) that lives in the compiler's memoisation table rather than in the program.

    K    distinct shapes           control m*   FLOPs ratio (ctrl/slow)   device bytes per arm
     8   512, 520, ... 568            544              1.019                    9 MB
    16   512, 520, ... 632            576              1.009                   21 MB
    32   512, 520, ... 760            648              1.017                   53 MB
    64   256, 264, ... 760            544              0.979                   76 MB

(Host footprint for the whole file is ~80 MB, not the sum of that column: the slow arms share one
cache keyed by shape, and each control's K same-shape operands are row-offset views into a single
pool -- distinct arrays and distinct contents for one array's worth of memory.)

RELATION TO THE OTHER AUTOTUNING CASE.  The corpus already has xla#5541 (conv_transpose,
``GpuConvAlgorithmPicker``, ~252 s reported and ~69.8 s of autotuning measured ON THIS MACHINE).
That is a different autotuner (conv, not GEMM) on a different axis: xla#5541 is one instruction
whose candidate BENCHMARKS are individually slow, this is many instructions each of which is
individually cheap to autotune but which fail to share a cache entry.  One is per-algorithm cost,
the other is cache-key cardinality.  That xla#5541 measured 69.8 s of autotuning here is the prior
that makes this reconstruction worth a slot: the autotuner on this machine is demonstrably
expensive, so K of them should be measurable.

THIRD ARM -- THE FLAG, WHICH CANNOT BE A ``CASES`` ENTRY.  The pure control is the SLOW arm run
under ``XLA_FLAGS=--xla_gpu_autotune_level=0``: byte-identical program, byte-identical HLO, and
only the compiler's willingness to benchmark changes.  XLA reads its flags once at backend
init and the harness gives every case one environment, so run it by hand:

    XLA_FLAGS=--xla_gpu_autotune_level=0 python _harness.py gemm_shapes_k32

If ``gemm_shapes_k32`` collapses to roughly its own control's time under that flag and stays slow
without it, the mechanism is settled.  This is the same experiment that settles xla#5541, and it
is worth running once for both.

DTYPE.  Explicitly float32.  x64 is on globally in the harness, and an f64 dot on GPU does not go
through the cuBLAS/Triton autotuned path at all -- it would silently delete the pathology.

PLATFORM.  GPU-ONLY BY CONSTRUCTION.  There is no GEMM autotuner in the CPU backend, so a fast CPU
arm says nothing about the case.  Run ``--platform cpu`` anyway, as a negative control on the
MECHANISM: if the distinct-shape arm is also slow on CPU, then shape diversity is costing something
other than autotuning and this file's story is wrong.

SCORE THIS ONE HONESTLY.  Two caveats the next phase should not paper over.  First, the asymmetry
may be INTENDED BEHAVIOUR rather than a bug -- caching autotune results by shape is the correct
design, and "distinct shapes cost more" is the price of it; the case earns its place as an
attribution test regardless of whether XLA considers it a defect.  Second, PR #37469 already fixed
the parallelisation half of the issue, so on a recent build the per-shape cost may be several times
smaller than when the issue was filed.  If the K=32 arm lands under the 3 s floor, K=64 is the next
rung and the answer is "partially fixed upstream", which is a result.
"""

from __future__ import annotations

import numpy as np

import jax.numpy as jnp

_rng = np.random.default_rng(35955)

# Cached by shape: the K=64 arm and the K=32 arm share the 512..760 range, so this holds the host
# footprint to ~80 MB for the whole file.  Nothing here touches a device at import.
_ARRAYS: dict[int, np.ndarray] = {}
_POOLS: dict[int, np.ndarray] = {}
_KMAX = 64


def _sq(m: int) -> np.ndarray:
    """An (m, m) float32 host array.  float32 is load-bearing -- see DTYPE above."""
    if m not in _ARRAYS:
        _ARRAYS[m] = _rng.standard_normal((m, m), dtype=np.float32)
    return _ARRAYS[m]


def _sq_distinct(m: int, i: int) -> np.ndarray:
    """The i-th of up to 64 DISTINCT (m, m) float32 arrays, as row-offset views into one pool.

    Distinct object AND distinct contents, for one array's worth of host memory.  Distinctness
    matters: if the control passed the same buffer K times and anything in the stack deduplicated
    identical arguments, the control would collapse to one dot and the op-count-invariance that the
    whole case rests on would quietly stop holding.  Views cannot be deduplicated by anyone.
    """
    if m not in _POOLS:
        _POOLS[m] = _rng.standard_normal((m + _KMAX, m), dtype=np.float32)
    return _POOLS[m][i:i + m, :]


def _k_dots(*xs):
    """K independent square GEMMs, each reduced to a scalar, summed.

    Each dot has its own operands, so nothing is CSE'd away and the instruction count is K in both
    arms.  The trailing per-dot ``.sum()`` gives the GEMM a reduce epilogue, which is what pushes it
    into a Triton fusion candidate set on the GPU backend -- i.e. into the autotuner.
    """
    return sum(jnp.dot(x, x).sum() for x in xs)


# (K, base, control_m) -- control_m chosen so K * control_m**3 ~= sum(m_k**3); see the table above.
_LADDER = ((8, 512, 544), (16, 512, 576), (32, 512, 648), (64, 256, 544))

CASES: dict = {}

for _k, _base, _mstar in _LADDER:
    _shapes = [_base + 8 * i for i in range(_k)]
    CASES[f"gemm_shapes_k{_k}"] = (
        _k_dots,
        tuple(_sq(m) for m in _shapes),
        f"xla#35955: {_k} f32 GEMMs at {_k} DISTINCT shapes ({_shapes[0]}..{_shapes[-1]}) -> {_k} "
        f"independent autotunes, one per distinct cache key.",
    )
    CASES[f"gemm_shapes_k{_k}_control"] = (
        _k_dots,
        tuple(_sq_distinct(_mstar, i) for i in range(_k)),
        f"control: the same {_k} f32 GEMMs at ONE shape ({_mstar}x{_mstar}) -- identical op count, "
        f"identical dtype, FLOPs matched within 2%; {_k} autotunes collapse to 1 cache hit.",
    )
