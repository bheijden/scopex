"""jax#12789 -- XLA constant-folds a whole N^3 buffer at compile time to serve one scalar store.

    https://github.com/jax-ml/jax/issues/12789          assigned to mattjj, no maintainer diagnosis

Reported on the issue's jax (2022), CPU. The repro is four lines:

    @jax.jit
    def foo():
        x = jax.numpy.ones((500, 500, 500))
        return x.at[(100, 100, 100)].set(2)

    first call  3.31   s      <- compile
    second call 0.00235 s     <- run

and XLA's own slow_operation_alarm names the guilty instruction verbatim:

    Constant folding an instruction is taking > 1s: dynamic-update-slice.1
    ... took 2.22s

i.e. 2.22 of the 3.31 s is ONE pass on ONE instruction. That is the cleanest ground truth an
attribution tool can be handed: there is a single correct answer and XLA already knows it.

WHY THIS CASE EARNS ITS PLACE. The corpus exercises no constant folding at all. Every case we have
is "the program XLA was given is too big / too awkward"; this one is "the program is tiny and XLA
chose to *execute* part of it at compile time". Compile cost is proportional to a LITERAL's size,
not to instruction count, not to FLOPs, not to jaxpr depth -- so instruction-counting attribution
points at a 6-line HLO module and shrugs. Also note the runtime is genuinely ~0: once folded, the
executable returns a constant and does no work, so the compile/runtime ratio is enormous.

WHAT THE CONTROLS ISOLATE.

  * `_control` (formulation): hoist the buffer out of the jit so the scatter's operand is a
    PARAMETER instead of a literal. `lambda x: x.at[(1,1,1)].set(2.0)` -- byte-identical HLO shape,
    identical semantics, one scalar store. The folder cannot fire on a parameter. Everything that
    differs between the arms is constant folding and nothing else.
  * the SIZE SWEEP (128/256/352/400) is the second control axis: the claim is that cost tracks the
    LITERAL's element count, not the program's size, and the program is the same six instructions
    at every n. 352^3 = 43.6M elements and 400^3 = 64.0M elements deliberately bracket 45M, which
    is the `kMaximumConstantSizeElements` cut-off XLA's HloConstantFolding has carried for some
    years -- so if the guard is live we expect the curve to climb to 352 and then FALL OFF A CLIFF
    at 400. A cliff is a positive result, not a failure: it dates the guard.

TWO SHAPES OF LITERAL, because they may not behave the same and we cannot tell without measuring:

  * `constfold_ones_*` reproduces the issue verbatim. `jnp.ones` inside a jit stages out as
    `broadcast_in_dim(1.0)` (verified: the jaxpr keeps the broadcast, it is not folded by jax), so
    the scatter's operand is a BROADCAST, not a kConstant. Recent XLA explicitly refuses to
    constant-fold broadcasts ("broadcasts dramatically increase the size of constants"), which
    would mean the scatter's operands are never all-constant and the 2022 pathology is dead.
  * `constfold_literal_*` closes over a real, NON-UNIFORM numpy array instead. Verified by
    lowering: this emits a genuine dense `stablehlo.constant` and `main()` still takes no arguments,
    so the scatter's operand IS a kConstant and the broadcast exemption cannot apply. If the
    mechanism survives at all in jax 0.10.2 / current XLA, it survives here.

If both arms come out flat, that is the result: the folder's guards now cover this and the case is
retired. Run one arm under XLA_FLAGS=--xla_dump_to=... and grep stderr for "Constant folding an
instruction is taking" to confirm attribution against XLA's own alarm.

Memory: sizes are f64 under the harness's global x64, so n=400 is a 512 MB output and the paired
control is ~1 GB live. Host RSS at import is ~0 -- the control's input is np.zeros (lazily mapped
pages, physical only once jax transfers it) and the literal arm builds its numpy array at TRACE
time inside the function, so merely importing this file to discover CASES costs nothing.
"""

from __future__ import annotations

import functools

import jax.numpy as jnp
import numpy as np

# The store site. Any single element does; the issue used (100, 100, 100), which does not exist for
# n=128, so use a corner-adjacent index valid at every size in the sweep.
IDX = (1, 1, 1)

# Verbatim-repro sizes. 352^3 = 43,614,208 elements and 400^3 = 64,000,000 elements bracket the
# 45,000,000-element ceiling in XLA's HloConstantFolding.
ONES_SIZES = (128, 256, 352, 400)

# Dense-literal sizes are smaller: this array is materialised in host RAM at trace time and copied
# into the MLIR module as a DenseElementsAttr, so it is paid for twice on the host.
LITERAL_SIZES = (128, 256)


def _ones_internal(n):
    """jax#12789 verbatim: the buffer is born inside the jit, so it is a literal to XLA."""
    x = jnp.ones((n, n, n))
    return x.at[IDX].set(2.0)


def _literal_internal(n):
    """Same store, but over a real dense constant rather than a broadcast of a scalar.

    The array is built HERE, at trace time, not at import time -- discovering CASES must not
    allocate. jnp.asarray of a numpy array inside a trace lowers to a dense stablehlo.constant and
    `main()` still takes no arguments (verified by lowering), which is what the constant folder
    needs to see for the 2022 behaviour to apply.

    It must be NON-UNIFORM. A first draft used np.zeros and MLIR printed it as a splat
    (`dense<0.0>`), which XLA is free to canonicalise straight back into a broadcast -- landing in
    exactly the case the ones-arm already covers and testing nothing new. arange cannot be a splat.
    """
    x = jnp.asarray(np.arange(n**3, dtype=np.float64).reshape(n, n, n))
    return x.at[IDX].set(2.0)


def _param(x):
    """CONTROL: identical store, operand is a parameter, so there is nothing to fold."""
    return x.at[IDX].set(2.0)


def _zeros(n):
    # np.zeros is calloc-backed: virtual pages only, no physical RAM until something reads it.
    # That keeps `import case_constant_folding_dus` free even though a 400^3 f64 arg is 512 MB.
    return np.zeros((n, n, n), dtype=np.float64)


CASES = {}

for _n in ONES_SIZES:
    _mb = _n**3 * 8 / 1e6
    CASES[f"constfold_ones_{_n}"] = (
        functools.partial(_ones_internal, _n),
        (),
        f"jax#12789 verbatim: jnp.ones(({_n},)*3) inside jit, one .at[].set() "
        f"-- {_n**3 / 1e6:.1f}M elem / {_mb:.0f} MB literal; operand is a BROADCAST, "
        f"which current XLA may refuse to fold",
    )
    CASES[f"constfold_ones_{_n}_control"] = (
        _param,
        (_zeros(_n),),
        f"control: same store, buffer hoisted out of the jit to a parameter, n={_n}",
    )

for _n in LITERAL_SIZES:
    _mb = _n**3 * 8 / 1e6
    CASES[f"constfold_literal_{_n}"] = (
        functools.partial(_literal_internal, _n),
        (),
        f"jax#12789 with a genuine dense constant, n={_n} "
        f"({_n**3 / 1e6:.1f}M elem / {_mb:.0f} MB) -- broadcast exemption cannot apply here",
    )
    CASES[f"constfold_literal_{_n}_control"] = (
        _param,
        (_zeros(_n),),
        f"control: same store on a parameter, n={_n}",
    )
