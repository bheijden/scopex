"""openxla/xla#35587 -- argsort on float32 compiles ~25x slower than on int32, because a single
HLO PASS declines to fire.

    https://github.com/openxla/xla/issues/35587

VERIFIED IN THIS ENVIRONMENT (jax 0.10.2, CUDA GPU, x64 on), n = 1e7, byte-identical call::

    jax.jit(lambda x: jnp.argsort(x)[:8].sum())

    dtype      compile      runtime     compile/runtime
    int32       0.583 s     5.01 ms          116
    uint32      0.462 s     4.91 ms           94
    float32    14.626 s    35.70 ms          410       <- 25.1x int32, 31.7x uint32

MECHANISM. XLA:GPU's ``SortRewriter`` replaces a ``sort`` with a CUB radix-sort custom call only
when the comparator region is a "simple compare computation" -- essentially a bare ``compare`` of
the two key parameters. For integers JAX emits exactly that, the rewriter fires, and the whole sort
becomes one custom call whose body the compiler never has to look inside. For floats JAX cannot
emit a bare compare: IEEE-754 ordering is not the bit ordering, so ``lax.sort`` builds a comparator
containing a bitcast to integer, a sign-magnitude fixup, NaN detection and negative-zero
normalisation. The rewriter's pattern match fails, it bails, and the sort falls back to the generic
bitonic lowering, which INLINES that whole comparator region into every one of the O(log^2 n)
bitonic stages. The compile-time cost is (comparator size) x (number of stages); the integer arm
pays neither factor.

WHY THIS CASE EARNS A SLOT. The corpus has nothing where the control variable flips WHICH PASS
FIRES. Every case we have makes the program bigger, or more nested, or a different shape, so an
attribution tool that counts jaxpr equations or HLO instructions has a fighting chance. Here the
two arms have the same jaxpr shape, the same instruction count, the same op, the same sizes; one
token of dtype decides whether ``SortRewriter`` accepts or rejects the comparator, and everything
downstream follows from that one decision. Naming the responsible PASS -- not the responsible line
-- is the only correct answer, which is precisely what a per-pass vlog instrument exists to
produce.

WHAT THE CONTROLS ISOLATE.

  * ``_control`` (the tight one): the same lambda on an int32 array. One dtype token. Identical
    source, identical op, identical shapes, identical output dtype.
  * ``argsort_u32_1e7``: a second integer dtype, to show the fast arm is not an int32 special case.
  * ``argsort_bitcast_i32_1e6``: the SAME float32 BYTES, bitcast to int32 before sorting, so the
    data is float data but the comparator is the simple integer one. If this arm is fast, the
    variable is the COMPARATOR and not the data or the dtype of the buffer -- which is the claim.
  * ``sort_values_f32_1e6``: single-operand ``jnp.sort`` instead of two-operand ``argsort``, to
    check whether the bail is about the comparator (expected: same story) or about the sort having
    a payload operand (would show up as the values-only float arm being fast).

THE SIZE SWEEP EXISTS TO CLEAR THE compile/runtime BAR. As measured at n=1e7 the ratio is only 410,
because the float sort is also genuinely slower AT RUNTIME (35.7 ms vs 5.0 ms) -- that part is real
and is what the issue was actually filed about. Compile cost here scales with the NUMBER OF BITONIC
STAGES, i.e. O(log^2 n), while runtime scales with O(n log n). Shrinking n therefore collapses
runtime far faster than compile: at n=1e5 there are ~150 stages against ~300 at n=1e7, so compile
should fall by roughly half while runtime falls ~100x. The prediction is that the small-n arms
clear 1000x comfortably and the large-n arm does not. If compile instead falls off proportionally
with n, the mechanism is not stage-count-driven and this reading is wrong -- which is worth knowing.

CONFIRMING ATTRIBUTION, once measured. Re-run one arm under::

    TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE=sort_rewriter=2 ...

and look for the rewriter declining the float comparator. Both variables are needed; setting
TF_CPP_VMODULE without lowering TF_CPP_MIN_LOG_LEVEL prints nothing, which is this repo's trap #1.

VERIFIED AT TRACE TIME (CPU, jax 0.10.2, no execution). The float and int arms produce STRUCTURALLY
IDENTICAL jaxprs -- ``iota`` then ``sort[dimension=0 is_stable=True num_keys=1]`` over two operands,
then slice, then reduce_sum -- differing in exactly one token, the key operand's dtype. The index
payload is i64 in both arms under the harness's global x64, so payload width is held constant and
is not a confounder (and CUB accepts it: the int32 arm compiles in 0.583 s). There is nothing in
the program for an instruction-counting attribution to point at; the comparator region that
actually differs is generated during lowering and never appears in the jaxpr at all.

MEMORY. Largest array is 1e7 float32 = 40 MB; the file allocates ~110 MB of host numpy at import.
Inputs are RANDOM, not zeros: values cannot affect compile time, but an all-ties input makes the
runtime leg -- half the statistic here -- measure something unrepresentative.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

_rng = np.random.default_rng(35587)


def _floats(n: int, dtype) -> np.ndarray:
    return _rng.standard_normal(n).astype(dtype)


def _ints(n: int, dtype) -> np.ndarray:
    # Small magnitudes so the trailing .sum() of int keys in the values-only probe cannot overflow.
    # The magnitudes are irrelevant to compile time.
    return _rng.integers(0, 1_000_000, size=n).astype(dtype)


def _argsort_head(x):
    """The issue's repro verbatim. The [:8] keeps the output tiny; the sort is the whole cost."""
    return jnp.argsort(x)[:8].sum()


def _argsort_full(x):
    """Same sort, no slice -- guards against XLA's slice-of-sort -> top-k rewrite muddying things."""
    return jnp.argsort(x).sum()


def _argsort_bitcast_i32(x):
    """float32 bytes, integer comparator. Isolates comparator complexity from buffer contents."""
    return jnp.argsort(jax.lax.bitcast_convert_type(x, jnp.int32))[:8].sum()


def _sort_head(x):
    """Single-operand sort: no index payload, so a different CUB entry point."""
    return jnp.sort(x)[:8].sum()


CASES = {}

# --- the reported flip, swept over n. Compile ~ O(log^2 n) stages, runtime ~ O(n log n). ---------
for _n, _tag in ((int(1e5), "1e5"), (int(1e6), "1e6"), (int(1e7), "1e7")):
    CASES[f"argsort_f32_{_tag}"] = (
        _argsort_head, (_floats(_n, np.float32),),
        f"xla#35587: jnp.argsort on float32, n={_tag} -- SortRewriter rejects the IEEE-754 "
        f"comparator, generic bitonic lowering inlines it into every stage",
    )
    CASES[f"argsort_f32_{_tag}_control"] = (
        _argsort_head, (_ints(_n, np.int32),),
        f"control: identical call on int32, n={_tag} -- one dtype token, comparator is a bare "
        f"compare, SortRewriter swaps in the CUB custom call",
    )

# --- second integer dtype: the fast arm is not an int32 special case -----------------------------
CASES["argsort_u32_1e7"] = (
    _argsort_head, (_ints(int(1e7), np.uint32),),
    "probe: uint32 argsort, n=1e7 -- measured 0.462 s, i.e. even faster than int32",
)

# --- wider float: does comparator cost grow with element width? ----------------------------------
CASES["argsort_f64_1e6"] = (
    _argsort_head, (_floats(int(1e6), np.float64),),
    "probe: float64 argsort, n=1e6 -- 64-bit bitcast fixup, also outside CUB's supported types",
)

# --- float DATA through an integer COMPARATOR: the discriminating probe ---------------------------
CASES["argsort_bitcast_i32_1e6"] = (
    _argsort_bitcast_i32, (_floats(int(1e6), np.float32),),
    "probe: same float32 bytes bitcast to int32 before argsort -- if fast, the variable is the "
    "comparator, not the data (NB bitcast ordering != float ordering for negatives; irrelevant "
    "to compile time)",
)

# --- no slice: rules out the slice-of-sort -> top-k rewrite as the real actor ---------------------
CASES["argsort_f32_noslice_1e6"] = (
    _argsort_full, (_floats(int(1e6), np.float32),),
    "probe: float32 argsort with no [:8] slice, n=1e6 -- checks TopkRewriter is not what differs",
)
CASES["argsort_f32_noslice_1e6_control"] = (
    _argsort_full, (_ints(int(1e6), np.int32),),
    "control: same, int32",
)

# --- values-only sort: is the bail about the comparator or about the payload operand? ------------
CASES["sort_values_f32_1e6"] = (
    _sort_head, (_floats(int(1e6), np.float32),),
    "probe: single-operand jnp.sort on float32, n=1e6 -- no index payload",
)
CASES["sort_values_f32_1e6_control"] = (
    _sort_head, (_ints(int(1e6), np.int32),),
    "control: single-operand jnp.sort on int32, n=1e6",
)
