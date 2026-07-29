"""SYNTHESISED (gap 13, FFT) -- 256 FFTs of 256 DISTINCT transform lengths compile 2.7x slower
than 256 FFTs of ONE length, with an IDENTICAL jaxpr equation count and an IDENTICAL HLO
instruction count. 13.7 s against 5.0 s.

NOT MINED FROM AN ISSUE. Constructed and then measured. The corpus's only existing FFT entry
(``case_const_fold_fft_capture.py``) is about a captured constant, not about the FFT primitive
itself; this file varies the transform length and holds everything else fixed.

MECHANISM UNDER TEST. ``jnp.fft.fft`` lowers on CPU to a custom call into XLA's FFT runtime, keyed
by the transform length. Two things then scale with the number of DISTINCT lengths in a module
rather than with the number of FFTs:

  * nothing can be shared between calls of different lengths -- no CSE, no shared descriptor, no
    shared twiddle setup, and each output shape is a distinct buffer size that allocation reuse
    cannot fold together;
  * on the jax side every shape-keyed cache in the tracing and lowering path (shape rules,
    aval interning, the lowering cache) misses once per distinct length instead of hitting 255
    times out of 256.

Both arms in this file perform 256 FFTs on the same input buffer, each perturbed by a different
scalar so that CSE cannot collapse either arm. Both arms TRUNCATE a length-4096 input down to the
transform length, so both emit the same primitives in the same order. The only difference is
whether the 256 lengths are ``2048, 2049, ... 2303`` or ``2048`` two hundred and fifty-six times.

MEASURED IN THIS ENVIRONMENT (JAX_PLATFORMS=cpu, jax/jaxlib 0.10.2, x64 on, x = 4096 f32,
seconds, one fresh subprocess per measurement):

    K      distinct lengths                  one length                    ratios
           trace  lower  compile      trace  lower  compile      trace  compile  end-to-end
     16    0.219  0.866    2.310      0.126  0.624    0.963       1.7x    2.4x      2.2x
     64    0.583  0.705    2.458      0.183  0.587    0.838       3.2x    2.9x      2.4x
    128    1.305  1.245    4.291      0.313  0.619    2.021       4.2x    2.1x      2.2x
    256    3.683  1.922   13.747      0.792  0.991    5.034       4.6x    2.7x      2.8x

jaxpr equation counts are IDENTICAL between the arms at every K (112 / 448 / 896 / 1792) and so
are the unoptimised HLO line counts (923 / 3707 / 7419 / 14843). There is no structural difference
between case and control at all -- same ops, same order, same instruction count, same total bytes
to within 256 elements out of 524 288. Only the integers in the shapes differ.

That is what makes this worth a slot: it is the corpus's cleanest SHAPE-DIVERSITY case. Every
size-based or count-based attribution metric reports the two arms as equal.

RE-MEASURED ON A QUIETER BOX (same environment, ``jax.jit(fn).lower()`` then ``.compile()``):
``fftlen_distinct_k128`` lower 2.006 s / compile 3.110 s against its control's 0.609 s / 1.381 s
-- 3.3x on lowering and 2.3x on compilation, 2.0x end to end. The table above is from a run at
load ~30 and its absolutes are UPPER BOUNDS; the ratios are stable.

THE COST IS SPLIT ACROSS TWO STAGES, AND THE FILE EXPOSES BOTH. Tracing is 4.6x apart and
compilation is 2.7x apart at K=256. A profiler that instruments only XLA will find 2.7x and miss
more than a third of the wall clock; one that instruments only Python will find 4.6x on a much
smaller number. The right answer names both.

WHAT THE CONTROLS ISOLATE.

  * ``*_control`` (the tight one): the identical loop with ``n=BASE+i`` replaced by ``n=BASE``.
    One expression. Everything else -- the perturbation, the ``jnp.abs``, the reduction, the input
    array, the op count -- is shared code, not a parallel copy, so the arms cannot drift.
  * ``fftlen_slicediv_k256`` / ``fftlen_redwindiv_k256``: the SAME shape-diversity pattern with
    the FFT replaced by a plain ``lax.slice`` and by a ``lax.reduce_window``. **These were run and
    they answer the question: the effect is NOT about FFT.** At K=256, again with identical
    equation counts and identical HLO line counts inside each pair:

        op                distinct   one shape   ratio    HLO lines (both arms)
        jnp.fft.fft        6.526 s     2.938 s    2.2x         14843
        lax.slice          5.108 s     2.436 s    2.1x         12793
        lax.reduce_window  7.877 s     2.685 s    2.9x         16117

    ``lax.slice`` is the cheapest op in the instruction set and it pays the same 2.1x. So the
    variable is SHAPE DIVERSITY AT FIXED INSTRUCTION COUNT, and FFT is merely the entry point
    named by gap 13. That is a more valuable finding than an FFT-specific one, and it is why the
    discriminators are shipped rather than deleted: without them this file would have published
    the wrong mechanism.

    (A fourth arm, 256 dots of distinct sizes, is deliberately NOT included: CSE collapses the
    equal-size control down to 319 HLO lines against the diverse arm's 7970, so that pair is not
    op-count matched and its 4.3x is uninterpretable.)

WHAT DID **NOT** REPRODUCE, recorded so the slot is not spent again. All measured on CPU:

  * transform-length FACTORISATION: 65536 (2^16) 0.218 s, 65537 (prime) 0.255 s, 65520 (smooth)
    0.168 s, 104729 (prime) 0.241 s. Flat -- the CPU FFT does not pay for an awkward length at
    compile time.
  * RANK of ``fftn`` at a fixed 2^18 elements: 1-D 0.233 s, 2-D 0.292 s, 3-D 0.297 s, 5-D 0.345 s.
    Nearly flat.
  * transform AXIS: axis 0 of a (2048, 2048) array 0.631 s against axis -1 0.228 s. A 2.8x
    difference, but it comes from the explicit transposes jax inserts, i.e. it is a layout case
    (gap 1), not an FFT case, and it does not clear the floor.
  * ``jax.grad`` of an FFT: 0.188 s at n=65536. Nothing.

PLATFORM: **CPU (verified here).** GPU is the arm that should be STRONGER and is unverified: cuFFT
builds a plan per distinct transform length, and plan construction is exactly the kind of per-shape
setup this case multiplies by 256. The GPU was off-limits when this file was written. A flat GPU
result would be a surprise worth chasing.

MEMORY. Input is 4096 f32 = 16 KB. Each FFT produces a complex64 output of ~2 K elements and it is
reduced immediately, so peak live memory is small; the K=256 arms are large in INSTRUCTION count,
not in bytes. Runtime is milliseconds, so the compile/runtime ratio should be large.

NOTE ON THE BOX. Load average was ~30 on 20 cores while these numbers were taken; absolute seconds
are UPPER BOUNDS and the paired ratios are the statistic to trust.
"""

from __future__ import annotations

import functools

import numpy as np

import jax.numpy as jnp
from jax import lax

L = 4096          # input length; every transform TRUNCATES it, in both arms
BASE = 2048       # the shared transform length


def _ffts(x, k: int, diverse: bool):
    """``k`` FFTs. ``diverse`` decides only whether the lengths are BASE+i or BASE."""
    acc = 0.0
    for i in range(k):
        acc = acc + jnp.abs(jnp.fft.fft(x + float(i), n=BASE + (i if diverse else 0))).sum()
    return acc


def _slices(x, k: int, diverse: bool):
    """Discriminator: the same shape-diversity pattern with the cheapest possible op."""
    acc = 0.0
    for i in range(k):
        acc = acc + lax.slice(x + float(i), (0,), (BASE + (i if diverse else 0),)).sum()
    return acc


def _redwins(x, k: int, diverse: bool):
    """Discriminator: shape diversity through the WINDOW rather than the operand."""
    acc = 0.0
    for i in range(k):
        w = 3 + (i if diverse else 0)
        acc = acc + lax.reduce_window(x + float(i), -jnp.inf, lax.max, (w,), (1,), "VALID").sum()
    return acc


# NUMPY at module scope. Random rather than zeros: an all-zero FFT is degenerate at runtime and
# runtime is half the harness's statistic. Values cannot affect compile time.
_X = np.random.default_rng(10596).random(L).astype(np.float32)

CASES = {}

# --- the pair, swept over the number of FFTs -------------------------------------------------
for _k in (16, 64, 128, 256):
    CASES[f"fftlen_distinct_k{_k}"] = (
        functools.partial(_ffts, k=_k, diverse=True), (_X,),
        f"synthesised: {_k} FFTs of {_k} DISTINCT transform lengths (2048..{2048 + _k - 1}). "
        f"Measured 13.7 s at K=256 with 1792 jaxpr equations and 14843 HLO lines")
    CASES[f"fftlen_distinct_k{_k}_control"] = (
        functools.partial(_ffts, k=_k, diverse=False), (_X,),
        f"control: {_k} FFTs of ONE transform length (2048). Identical equation count, identical "
        f"HLO line count, identical op order; only the integers in the shapes differ. Measured "
        f"5.0 s at K=256")

# --- discriminators: is the variable FFT, or shape diversity in general? ----------------------
CASES["fftlen_slicediv_k256"] = (
    functools.partial(_slices, k=256, diverse=True), (_X,),
    "discriminator, MEASURED 5.108 s: 256 lax.slices of 256 distinct lengths instead of 256 "
    "FFTs. The cheapest op in the instruction set pays the same 2.1x, so the mechanism is shape "
    "diversity at fixed instruction count, not FFT")
CASES["fftlen_slicediv_k256_control"] = (
    functools.partial(_slices, k=256, diverse=False), (_X,),
    "control: 256 lax.slices of ONE length -- measured 2.436 s, identical 12793 HLO lines")

CASES["fftlen_redwindiv_k256"] = (
    functools.partial(_redwins, k=256, diverse=True), (_X,),
    "discriminator, MEASURED 7.877 s: shape diversity moved into the reduce_window WINDOW size "
    "rather than the operand shape -- a second, independent way to make 256 distinct shapes, and "
    "the strongest arm in the file at 2.9x")
CASES["fftlen_redwindiv_k256_control"] = (
    functools.partial(_redwins, k=256, diverse=False), (_X,),
    "control: 256 reduce_windows of ONE window size -- measured 2.685 s, identical 16117 HLO "
    "lines")
