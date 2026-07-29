"""jax#12789 -- one scalar store into a jit-internal constant costs seconds inside
HloConstantFolding, and XLA names the guilty instruction itself.

    https://github.com/jax-ml/jax/issues/12789        assigned to mattjj, no maintainer diagnosis

    @jax.jit
    def foo():
        x = jax.numpy.ones((500, 500, 500))
        return x.at[(100, 100, 100)].set(2)     # reported 3.31 s compile, 2.35 ms run (2022, CPU)

OVERLAP, STATED UP FRONT. `case_constant_folding_dus.py` in this directory already covers this
issue -- verbatim shape (no reduction), a broadcast-vs-dense-literal split, and a size sweep to
400 that brackets XLA's 45M-element folding ceiling. Nothing here duplicates a case name from it
and this file is NOT a replacement. It exists for ONE reason: that file's control is the
parameter-hoisted arm, which was measured in this environment at 1.749 s against the case's
5.846 s -- only 3.3x, UNDER the corpus's 10x-vs-control bar. On that control the case scores "no"
despite compiling for 5.8 s with a 0.86 ms runtime. The control is wrong, not the case. Hoisting
the buffer to a parameter removes the folding but KEEPS a 216 MB allocation and a 216 MB
reduction, and compiling those costs ~1.7 s all by itself. This file supplies controls that do not
pay that 1.7 s, so the measurement isolates the folding instead of hiding it.

MEASURED IN THIS ENVIRONMENT (jax 0.10.2, CUDA, x64) before the case was written:

    n=300, in-jit constant + .at[].set() + .sum()    compile 5.846 s   run 0.864 ms   ratio 6765
    n=200, same                                      compile 2.232 s
    n=300, buffer hoisted to a parameter             compile 1.749 s   <- the weak control, 3.3x

and XLA volunteers the answer on stderr while it does it:

    Constant folding an instruction is taking > 1s: %get-tuple-element.9 = f64[300,300,300] ...
        metadata={stack_frame_id=3}
    ... The operation took 1.770 s

WHY THIS EARNS A SLOT. Not for the family -- the corpus has constant folding via jax#14655 and via
`case_const_fold_fft_capture.py`. It earns it because of that stderr line. This is the ONLY case in
the corpus that ships GROUND TRUTH for per-instruction attribution: XLA names the exact instruction
AND attaches a `stack_frame_id` pointing back into the user's source. scopex's answer can be
checked against the compiler's own, not against a human's guess. Everywhere else in the corpus we
are inferring the culprit; here we can grade.

The second reason is minimality: one line, one edit, no nesting, no AD, no control flow, no shape
games. If attribution cannot get this one right there is no point running it on the 8-link scatter
chain.

WHAT EACH CONTROL ISOLATES.

  * `_control` (the paired one) is the same in-jit constant and the same reduction with the
    `.at[].set()` DELETED: `jnp.ones((n,n,n)).sum()`. Same source, same buffer, same jit boundary,
    one method call removed. It is fast because `reduce(broadcast(1.0))` collapses algebraically
    and no literal is ever materialised -- which is the point. Everything that differs between the
    arms is "XLA ran the program at compile time to serve one scalar store". Note what this control
    deliberately does NOT hold fixed: it removes the 216 MB materialisation as well as the fold,
    because in the slow arm those are the same event.
  * `dusfold_dynidx_300` is the tighter, more conservative arm and it is here as a hedge. It keeps
    every op that costs anything -- the constant, the dynamic-update-slice, the reduction -- and
    moves only the store INDEX from a compile-time constant to a runtime argument, which is enough
    to disqualify the instruction from folding. All the tensor-shaped work and all the runtime
    FLOPs are identical to the slow arm; the jaxpr is 16 equations against 7, and every one of the
    nine extra is a SCALAR index clamp/convert that the dynamic path requires. PREDICTION: it lands near the parameter arm's ~1.7 s, i.e. ~3.4x, under the bar --
    for the same reason the parameter arm does, since it too materialises and reduces 216 MB for
    real. If it instead comes out fast, it is the better control and the paired one should be
    switched to it. Either way the pair of numbers separates "cost of folding" from "cost of
    compiling a big reduce", which is the thing the weak control conflated.
  * The SIZE SWEEP (200/300/350) is the third axis. The program is the same six instructions at
    every n; only the literal's element count changes. 2.232 s at n=200 and 5.846 s at n=300 is
    super-linear in elements (3.4x elements would be 7.5 s if linear... it is 2.6x for 3.4x, so it
    is close to linear here) -- the sweep is what settles that, and n=350 (42.9M elements) sits
    just under the 45M-element ceiling XLA's HloConstantFolding has carried for years. If n=350 is
    FASTER than n=300, the guard fired and that is a dated, publishable fact rather than a failure.

DIFFERENCE FROM THE SIBLING FILE, in one line: every arm here ends in `.sum()`. That is what the
in-env verification used, and it matters twice -- it drops the output from 216 MB to 8 bytes, so
`runtime_s` measures the store rather than a device-to-host copy and the compile/runtime ratio is
meaningful (6765) rather than transfer-dominated; and it keeps the folded result from being the
program's output, which is a different XLA path.

GROUND-TRUTH CHECK, worth doing once by hand: run one arm and grep stderr for
"Constant folding an instruction is taking". The instruction name and its `stack_frame_id` are the
reference answer. The harness's `xla_slow_warning` flag only looks for "Very slow compile?" and
will NOT catch this alarm, so it must be read from the subprocess's stderr directly.

MEMORY. f64 under the harness's global x64: n=200 is 64 MB, n=300 is 216 MB, n=350 is 343 MB, and
the folder holds a host-side copy while it works. Import cost is zero -- every buffer is built
inside the traced function, and the only module-level array is one int32 index.
"""

from __future__ import annotations

import functools

import numpy as np

import jax.numpy as jnp

# Sizes in elements: 200^3 = 8.0M, 300^3 = 27.0M, 350^3 = 42.9M. The last one sits just below the
# 45,000,000-element ceiling in XLA's HloConstantFolding, so it is the arm that dates the guard.
SIZES = (200, 300, 350)


def _fold_sum(n):
    """SLOW: the buffer is born inside the jit, so its literal is the scatter's operand.

    Verbatim in-env repro. The store index is a compile-time constant, which is what lets XLA
    decide it can materialise the whole n^3 buffer and rewrite it rather than emit a kernel.
    """
    return jnp.ones((n, n, n)).at[(n // 2,) * 3].set(2.0).sum()


def _nofold_sum(n):
    """CONTROL: same constant, same reduction, `.at[].set()` deleted. Nothing to fold."""
    return jnp.ones((n, n, n)).sum()


def _dynidx_sum(k, n):
    """HEDGE CONTROL: all of the slow arm's tensor work, but the store index arrives at runtime.

    A dynamic-update-slice with a non-constant index has a non-constant operand set, so the folder
    cannot fire on it -- while the emitted program still allocates n^3, stores one element and
    reduces the whole thing, exactly as the slow arm's *runtime* does.
    """
    return jnp.ones((n, n, n)).at[k, k, k].set(2.0).sum()


CASES = {}

for _n in SIZES:
    _elem, _mb = _n ** 3 / 1e6, _n ** 3 * 8 / 1e6
    CASES[f"dusfold_sum_{_n}"] = (
        functools.partial(_fold_sum, _n), (),
        f"jax#12789: one .at[].set() into a jit-internal {_n}^3 f64 constant, then .sum() "
        f"-- {_elem:.1f}M elem / {_mb:.0f} MB folded at compile time "
        f"(in-env n=300: 5.846 s compile, 0.864 ms run)",
    )
    CASES[f"dusfold_sum_{_n}_control"] = (
        functools.partial(_nofold_sum, _n), (),
        f"control: identical constant and reduction with the .at[].set() removed, n={_n} "
        f"-- no literal is ever materialised",
    )

CASES["dusfold_dynidx_300"] = (
    functools.partial(_dynidx_sum, n=300), (np.int32(150),),
    "hedge control, unpaired: all of dusfold_sum_300's tensor work kept, store index moved to a "
    "runtime "
    "argument so folding is disqualified -- PREDICTED ~1.7 s (only ~3.4x), which if true is the "
    "measurement proving the weak parameter control's 1.7 s is the reduce, not the fold",
)
