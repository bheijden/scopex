"""jax#19653 -- compiling ``lax.top_k`` blows up at particular input LENGTHS; n+1 is fine.

    https://github.com/jax-ml/jax/issues/19653

Reported on jax 0.4.23, GPU: ``jax.jit(partial(jax.lax.top_k, k=k))`` applied to
``jnp.zeros(1 << 20)`` exhausted host RAM during COMPILATION, while the same call on
``jnp.zeros((1 << 20) + 1)`` compiled without incident.  The reporter observed the same failure
at ``n = 2**20 + j*1024`` and not at intermediate lengths.  The executable, once built, is fine
-- the blowup is entirely compiler-side.

WHY THIS CASE EARNS ITS PLACE IN THE CORPUS.  Two properties nothing else in the corpus has:

  * The pathology is a KNIFE EDGE on a single integer.  Every other case scales -- more links,
    more nodes, bigger dtype -- so a profiler can find the cause by watching a number grow.  Here
    n and n+1 produce the same primitive, the same k, the same dtype, the same op count, the same
    runtime, and wildly different compile behaviour.  The only signal is which branch the GPU
    emitter took, and the alignment of one dimension is what selects it.
  * The resource consumed is compile-time HOST MEMORY, not wall clock.  A tool that only
    instruments pass durations will see a short profile and then a dead process.  Note that this
    harness measures wall clock only: an arm that genuinely OOMs shows up as ERROR in results,
    and that IS the reproduction, not a failed measurement.  Anyone chasing this should watch
    ``ru_maxrss`` of the child alongside the timings.

MECHANISM (as far as the issue establishes it).  XLA:GPU chooses a partition/bitonic strategy for
top_k from the input length.  At exact powers of two, and at 1024-aligned offsets from one, the
emitted code path expands enormously at compile time.  Gating on ALIGNMENT rather than MAGNITUDE
is the tell: it points at a tiling/padding decision in the emitter, not at "the problem got
bigger".  No maintainer diagnosis is attached to the issue, so this is a reading of the evidence,
not an assertion.

CONTROL.  One character: ``n = (1 << 20) + 1`` instead of ``n = 1 << 20``.  Same primitive, same
k, same dtype, output shapes differing by nothing that matters.  The reporter explicitly confirms
the +1 length compiles fine.  This is the tightest control in the batch, and each ``_control``
entry below is exactly that.

THE SWEEP.  The emitter was reworked after 0.4.23 (CUB-backed paths landed later), so the exact
constants from the issue are not trustworthy on jax 0.10.2.  The entries therefore probe three
independent axes rather than one point:

  * k, at 8 / 128 / 1024 -- small k may take a radix path, large k a sort path.
  * ALIGNMENT at fixed n, via ``+512`` / ``+1024`` / ``+2048`` offsets from 2**20.  If the cliff
    is alignment-gated, +1024 and +2048 misbehave and +512 and +1 do not.
  * MAGNITUDE, via 2**19 / 2**21 / 2**22 at fixed k.  If those are all fine and only 2**20
    misbehaves, the cliff is not about size at all.

Inputs are float32 (x64 is on globally in the harness and f64 would change the emitter's
element-width decisions).  Largest array here is 2**22 float32 = 16 MB, so the whole file is
cheap to hold; the expense, if any, is entirely in the compiler.

PLATFORM.  GPU-only by construction: the reported blowup is in the XLA:GPU top_k emitter, and the
CPU backend lowers top_k through an unrelated path.  Run ``--platform cpu`` as a negative control
on the mechanism -- a cliff that also appears on CPU is not the emitter and would refute this
file's reading.

VERIFIED AT TRACE TIME (CPU, jax 0.10.2, no execution): all twelve arms are single-equation
jaxprs, ``n=2**20`` and ``n=2**20+1`` differing only in the operand's leading dimension
(1048576 vs 1048577) with identical k, dtype and output shapes.  There is nothing in the program
for a source-level profiler to point at.

UNCERTAINTY.  Honest expectation: this most likely does NOT reproduce on jax 0.10.2 -- the
reported failure is two years and one emitter rewrite old.  A clean negative across the whole
sweep is a publishable result and is why the sweep is broad rather than a single point.
"""

from __future__ import annotations

import functools

import numpy as np

from jax import lax

_rng = np.random.default_rng(19653)

_M = 1 << 20


def _mk(n: int, k: int, note: str):
    # Random rather than zeros: values cannot affect compile time, but an all-ties input makes
    # the runtime leg of the harness measure something unrepresentative.
    x = _rng.standard_normal(n, dtype=np.float32)
    return functools.partial(lax.top_k, k=k), (x,), note


CASES = {
    # --- the reported cliff, at three values of k. n = 2**20 vs n = 2**20 + 1 -------------------
    "topk_pow20_k8": _mk(_M, 8, "jax#19653: top_k k=8 on n=2**20 (the reported OOM length)"),
    "topk_pow20_k8_control": _mk(_M + 1, 8, "control: n=2**20+1, one element longer, k=8"),

    "topk_pow20_k128": _mk(_M, 128, "jax#19653: top_k k=128 on n=2**20"),
    "topk_pow20_k128_control": _mk(_M + 1, 128, "control: n=2**20+1, k=128"),

    "topk_pow20_k1024": _mk(_M, 1024, "jax#19653: top_k k=1024 on n=2**20"),
    "topk_pow20_k1024_control": _mk(_M + 1, 1024, "control: n=2**20+1, k=1024"),

    # --- alignment axis: the issue claims 2**20 + j*1024 also fails ----------------------------
    "topk_pow20p512_k128": _mk(
        _M + 512, 128, "alignment probe: n=2**20+512 (NOT 1024-aligned), k=128"),
    "topk_pow20p1024_k128": _mk(
        _M + 1024, 128, "alignment probe: n=2**20+1024 (1024-aligned, reported to fail), k=128"),
    "topk_pow20p2048_k128": _mk(
        _M + 2048, 128, "alignment probe: n=2**20+2048 (1024-aligned), k=128"),

    # --- magnitude axis: is anything special about 2**20 at all? -------------------------------
    "topk_pow19_k128": _mk(1 << 19, 128, "magnitude probe: n=2**19, k=128"),
    "topk_pow21_k128": _mk(1 << 21, 128, "magnitude probe: n=2**21, k=128"),
    "topk_pow22_k128": _mk(1 << 22, 128, "magnitude probe: n=2**22, k=128"),
}
