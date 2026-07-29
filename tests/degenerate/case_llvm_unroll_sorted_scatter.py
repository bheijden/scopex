"""xla#22233 -- one integer in a scatter's update width (34 vs 35) flips LLVM into full loop
unrolling and register spills.

    https://github.com/openxla/xla/issues/22233

The sorted-scatter emitter accumulates a slice of updates in REGISTERS across a loop whose trip
count is the update-slice width, then writes the accumulator out once per output row. Two
maintainer comments are the whole diagnosis:

    jreiffers: "the initial MLIR is nearly identical ... the IR starts to diverge in
                LoopFullUnrollPass"
    pifon2a:   "related to the large number of iter_args in the generated loops"

Past a width threshold LLVM decides the loop is fully unrollable, produces a body with hundreds or
thousands of live loop-carried values, runs out of registers and spills to ``.local``. The reporter
measured 38 ms against 0.4 ms of RUNTIME at the cliff, and observed the threshold moving with the
machine ("Try 1097, 1103, 1109 if it's not slow").

READ THIS BEFORE SCORING THE RESULT. **The issue is a RUNTIME report. The compile-time angle is an
inference, not a claim anyone made.** It is a well-founded inference -- fully unrolling a
~1000-iteration loop carrying many values is a textbook NVPTX compile-time blowup, and it would put
the cost in the LLVM/PTX stage where this corpus has almost nothing -- but it is still an
inference. The honest expected outcome is that COMPILE TIME IS FLAT ACROSS BOTH ARMS while runtime
shows a cliff. **That is a result to keep, not a failure.** A 95x runtime cliff with zero
compile-time signal is exactly the discrimination scopex should be able to demonstrate: a tool that
reports a compile-time hotspot here would be hallucinating one.

WHY THIS CASE EARNS A SLOT. Everything else in the corpus lives at or above the HLO level, where an
attribution tool can in principle read the jaxpr or the pass timings and reason. This one lives
underneath: identical jaxpr, identical HLO, identical MLIR, and the divergence happens inside a
single named LLVM pass. If it does move compile time, it is the only probe we have that can only be
answered below XLA. If it does not, it is the negative control for that entire layer.

TWO CONTROLS, both razor-thin, and they answer different questions.

  * ``_control`` -- the FLAG control, and the better one for scopex. The same ``.at[idx].add()``
    with ``indices_are_sorted=False`` instead of ``True``. Semantically identical program (the
    indices really are sorted in both arms, so the results agree), same shapes, same dtypes, same
    op -- but XLA selects a DIFFERENT EMITTER. The source difference is one keyword argument, so
    any source-level attribution has almost nothing to point at.
  * the WIDTH control -- read ACROSS entries, not within a pair: 34 against 35, and 1090 against
    1091. One integer, everything else byte-identical. The harness pairs only ``name`` with
    ``name_control``, so this axis is read off the results table rather than computed by
    ``classify``; the widths are chosen so adjacent rows are the comparison.

THE SWEEP. 34/35 and 1090/1091 are the reporter's own numbers; 1097/1103/1109 are their suggested
fallbacks for machines whose threshold sits elsewhere. 4096 and 8192 are ours: if full unrolling
ever costs compile time, a loop that long is where it must show, and if those are flat too then the
unroller is bounded and the compile-time hypothesis is dead in one measurement. The large-width
entries use fewer update rows so the arrays stay small -- width, not update count, is the variable.

SHAPES. operand ``(rows, W)`` float32, indices ``(n_updates,)`` int32 SORTED, updates
``(n_updates, W)``. This yields a scatter with ``update_window_dims=(1,)`` of extent W, which is
the loop whose trip count the emitter unrolls. float32 is spelled out: under the harness's global
x64 these would come out f64, halving the values per register and moving whatever threshold exists.
Buffers are calloc-backed zeros -- XLA cannot see a parameter's values, so scatter-add of zeros
compiles and runs identically, and importing this file costs no physical RAM.

NO TRAILING REDUCTION on purpose. The corpus's largest finding is that a trailing ``jnp.sum`` after
a scatter chain is itself the pathology; adding one here would fuse the scatter into a reduction,
change the emitter, and measure that finding again instead of this one.

VERIFIED AT TRACE TIME (CPU, jax 0.10.2, no execution). All 18 arms are 5-equation jaxprs. The
``scatter-add`` equation carries ``indices_are_sorted=True`` in the case arms and ``False`` in the
controls with everything else identical, and ``update_window_dims=(1,)`` -- confirming the window
whose extent is W really is the emitter's inner loop, which is the object the whole hypothesis is
about.
"""

from __future__ import annotations

import numpy as np

_rng = np.random.default_rng(22233)

ROWS = 256

# The reporter's cliff pair, their second pair, their machine-dependent fallbacks, then two widths
# far past anything a bounded unroller would touch.
WIDTHS = (34, 35, 1090, 1091, 1097, 1103, 1109)
WIDE_WIDTHS = (4096, 8192)

N_UPDATES = 2048
N_UPDATES_WIDE = 512

# Which other row in the results table each width is meant to be read against. 34/35 and 1090/1091
# are the reporter's adjacent-integer pairs; the three fallbacks are a cluster to be read against
# each other and against 1091.
PARTNER = {34: 35, 35: 34, 1090: 1091, 1091: 1090,
           1097: 1091, 1103: 1091, 1109: 1091}


def _sorted_scatter(x, idx, upd):
    """The pathological arm: indices_are_sorted=True selects the register-accumulating emitter."""
    return x.at[idx].add(upd, indices_are_sorted=True, unique_indices=False)


def _unsorted_scatter(x, idx, upd):
    """CONTROL: one keyword argument different. Indices are still sorted, so results agree."""
    return x.at[idx].add(upd, indices_are_sorted=False, unique_indices=False)


def _args(width: int, n_updates: int):
    # Indices genuinely sorted: indices_are_sorted=True is a promise, and a false promise would
    # make the two arms compute different things and the comparison meaningless.
    idx = np.sort(_rng.integers(0, ROWS, size=n_updates)).astype(np.int32)
    return (np.zeros((ROWS, width), dtype=np.float32),
            idx,
            np.zeros((n_updates, width), dtype=np.float32))


CASES = {}

for _w in WIDTHS:
    _a = _args(_w, N_UPDATES)
    CASES[f"sorted_scatter_w{_w}"] = (
        _sorted_scatter, _a,
        f"xla#22233: .at[sorted].add() with update width {_w} -- LoopFullUnrollPass is the "
        f"reported divergence point; read this against sorted_scatter_w{PARTNER[_w]} in the same "
        f"table as well as against its own control",
    )
    CASES[f"sorted_scatter_w{_w}_control"] = (
        _unsorted_scatter, _a,
        f"control: identical scatter, width {_w}, indices_are_sorted=False -- one keyword, "
        f"different emitter, same semantics",
    )

for _w in WIDE_WIDTHS:
    _a = _args(_w, N_UPDATES_WIDE)
    CASES[f"sorted_scatter_w{_w}"] = (
        _sorted_scatter, _a,
        f"probe past the reported range: update width {_w} with {N_UPDATES_WIDE} updates -- if "
        f"full unrolling ever costs compile time it must show here; flat means the unroller is "
        f"bounded and the compile-time hypothesis is dead",
    )
    CASES[f"sorted_scatter_w{_w}_control"] = (
        _unsorted_scatter, _a,
        f"control: same, width {_w}, indices_are_sorted=False",
    )
