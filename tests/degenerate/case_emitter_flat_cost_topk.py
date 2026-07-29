"""jax#19653 -- ``lax.top_k`` costs a flat multi-second compile at EVERY size; ``jnp.sort`` does not.

    https://github.com/jax-ml/jax/issues/19653

WHAT THE ISSUE REPORTED, AND WHY THAT PART IS DEAD.  On jax 0.4.23 / GPU the reporter saw
``jax.jit(partial(lax.top_k, k=k))`` exhaust host RAM while compiling for ``n = 1 << 20``, with
``n = (1 << 20) + 1`` compiling fine, i.e. a SIZE CLIFF gated on alignment.  That claim is probed
directly and separately in ``case_size_cliff_topk.py``.  It does not survive on jax 0.10.2: a sweep
of n and k in this environment found no cliff at 2**20, at 2**20 + j*1024, or anywhere else.

WHAT IS ACTUALLY THERE, WHICH APPEARS UNREPORTED.  ``lax.top_k`` costs 4.2-6.2 s to compile at
EVERY size, and the number is essentially independent of both n and k:

    n = 2**12   6.193 s        k = 1      5.267 s      (k ladder measured at n = 2**20)
    n = 2**14   4.414 s        k = 8      5.279 s
    n = 2**16   4.200 s        k = 128    5.855 s
    n = 2**18   4.470 s        k = 1024   4.342 s
    n = 2**20   5.396 s

while the mathematically identical ``jnp.sort(x)[-k:]`` on the same array compiles in 0.404 s.  The
headline pair, measured in this environment before the file was written:

    SLOW     lax.top_k(x, 8)[0].sum(),  n = 2**20 f64    compile 5.279 s   run 1.878 ms   ratio 2812
    CONTROL  jnp.sort(x)[-8:].sum(),    same array       compile 0.404 s   run 1.169 ms

13.1x, for the same answer in the same output shape at ~the same runtime.  Both bars are cleared
unaided: 5.279 s > 3.0 s floor, ratio 2812 > 1000, 13.1x > 10x control.

WHY THIS CASE EARNS ITS PLACE IN THE CORPUS.  Two properties, and the corpus has neither.

  * **The cost is FLAT.**  Every other case in the corpus is found by turning a knob and watching a
    number grow -- chain length, matrix size, nesting depth, dtype width, batch.  Here every knob is
    dead.  n moves by 256x and compile time moves by 1.4x, in the wrong direction.  k moves by 1024x
    and compile time moves by 1.3x, also in the wrong direction.  A profiler that attributes by
    correlating compile time against problem dimensions has nothing to correlate against, and a
    bisect-the-input strategy converges on nothing.  Only per-INSTRUCTION or per-PASS attribution
    can name the cause, which is exactly the capability scopex claims.  That the flatness is the
    finding is why the ladders below are in the file at all -- they are not a scaling study, they
    are the evidence that there is no scaling.
  * **It is a fixed per-primitive toll.**  ~5 s is spent on one HLO instruction in a program that
    otherwise contains a reduce.  Cost per instruction here is ~10^4 x the corpus median.  Any
    attribution scheme that spreads compile time over instructions by count, or by FLOPs, or by
    output bytes, will mis-assign essentially all of it.

MECHANISM (a reading of the evidence, not a maintainer diagnosis).  A cost that does not move with
the data is a cost paid to EMIT rather than to process: XLA:GPU builds top_k from a fixed-shape
partitioned bitonic/radix kernel whose codegen -- template expansion, then LLVM/PTX for the emitted
module -- is the same work whatever n and k are, and the shape only enters as a launch dimension.
``jnp.sort`` on the same array takes the sort emitter's ordinary path and costs 0.4 s.  If that
reading is right the time lands in LLVM/PTX emission for one fusion, and scopex should be able to
say so; if instead it lands in an XLA pass over the whole module, the reading is wrong and that too
is a result.  Run one arm with ``XLA_FLAGS=--xla_dump_to=...`` and compare pass timings between the
top_k and sort arms to settle it.

WHAT THE CONTROL ISOLATES.  ``jnp.sort(v)[-k:]`` against ``lax.top_k(v, k)[0]``: same input array,
same dtype, same output shape, same value set, and the trailing ``.sum()`` makes the two arms
return the identical scalar (order-invariant, so the descending/ascending difference cannot leak
in).  The ONLY difference is which primitive jax emits -- ``top_k`` vs ``sort`` + ``slice``.  Note
the control is the LARGER program (sort of the whole array is asymptotically more work at runtime,
and the runtimes above confirm it: 1.17 ms vs 1.88 ms), so this is a case where the fast-compiling
arm is the one that does MORE work.  Any heuristic of the form "big compile means big program" gets
the sign wrong here.

THE NO-SUM PAIR.  The corpus's flagship case (8-link scatter chain) is pathological only WITH a
trailing ``jnp.sum`` and vanishes without it, so a trailing reduction is a known confounder in this
corpus.  ``topkflat_n2p20_k8_nosum`` returns the top-k values directly, no reduction.  If the ~5 s
survives, the reduction is irrelevant here and the two families are unrelated; if it collapses,
this case is a second instance of the reduction pathology rather than a new one.  Either answer is
worth a measurement slot.

DTYPE.  f64 throughout, matching the verification run above (x64 is on globally in the harness, so
``jnp.zeros(n)`` is f64 there too).  f32 may take a different emitter branch; that is a follow-up,
not this file.

PLATFORM.  GPU is where this was measured and where the mechanism is claimed.  ``--platform cpu``
is the negative control on the mechanism: the CPU backend lowers top_k through an unrelated path,
so a flat ~5 s on CPU as well would refute the GPU-emitter reading.

MEMORY.  Largest array is 2**20 f64 = 8 MB.  Nothing here is big; the expense is entirely
compiler-side, which is the point.
"""

from __future__ import annotations

import functools

import numpy as np

import jax.numpy as jnp
from jax import lax

# n ladder at fixed k=8, spanning 256x.  The finding is that compile time does NOT move.
N_LADDER = (1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20)

# k ladder at fixed n=2**20, spanning 1024x.  Same finding.
K_LADDER = (1, 128, 1024)

_N_BIG = 1 << 20


def _topk_sum(v, k):
    """The pathological arm.  ``[0]`` drops the indices; ``.sum()`` makes the arms comparable."""
    return lax.top_k(v, k)[0].sum()


def _sort_sum(v, k):
    """CONTROL: same k values, same scalar answer, via the sort primitive instead of top_k."""
    return jnp.sort(v)[-k:].sum()


def _topk_raw(v, k):
    """No-reduction variant -- is the trailing sum load-bearing here as it is for scatter chains?"""
    return lax.top_k(v, k)[0]


def _sort_raw(v, k):
    return jnp.sort(v)[-k:]


_XCACHE: dict[int, np.ndarray] = {}


def _x(n: int):
    # Random, not zeros: values cannot change compile time, but an all-ties input would make the
    # runtime leg of the harness measure something unrepresentative of a real top_k.  Cached so
    # that merely importing this file to discover CASES costs one 8 MB array, not five.
    if n not in _XCACHE:
        _XCACHE[n] = np.random.default_rng(19653).standard_normal(n)
    return _XCACHE[n]


def _pair(tag: str, n: int, k: int, note_slow: str, note_ctl: str, raw: bool = False):
    slow, ctl = (_topk_raw, _sort_raw) if raw else (_topk_sum, _sort_sum)
    x = _x(n)
    return {
        tag: (functools.partial(slow, k=k), (x,), note_slow),
        f"{tag}_control": (functools.partial(ctl, k=k), (x,), note_ctl),
    }


CASES: dict = {}

# --- the n ladder at k=8: compile time measured 6.19 / 4.41 / 4.20 / 4.47 / 5.40 s -----------------
for _n in N_LADDER:
    _e = _n.bit_length() - 1                      # 4096 -> 12, ..., 1048576 -> 20
    CASES.update(_pair(
        f"topkflat_n2p{_e}_k8", _n, 8,
        f"jax#19653 (flat-cost reading): lax.top_k(x,8) at n=2**{_e} f64 -- ~5 s at EVERY n",
        f"control: jnp.sort(x)[-8:] on the identical array, n=2**{_e} -- same scalar, ~0.4 s"))

# --- the k ladder at n=2**20: compile time measured 5.27 / 5.86 / 4.34 s for k=1/128/1024 ---------
for _k in K_LADDER:
    CASES.update(_pair(
        f"topkflat_n2p20_k{_k}", _N_BIG, _k,
        f"jax#19653 (flat-cost reading): k={_k} at n=2**20 -- k moves 1024x, compile does not",
        f"control: jnp.sort(x)[-{_k}:] at n=2**20"))

# --- is the trailing reduction load-bearing, as it is for the corpus's scatter-chain case? --------
CASES.update(_pair(
    "topkflat_n2p20_k8_nosum", _N_BIG, 8,
    "jax#19653 with NO trailing reduction: top_k values returned directly. If ~5 s survives, this "
    "case is unrelated to the corpus's reduction pathology; if it collapses, it is the same thing",
    "control: jnp.sort(x)[-8:] returned directly, no reduction",
    raw=True))
