"""jax#18221 -- associative_scan compiles far slower for complex than for real, from one line.
This file adds the two axes the sibling file does not have: dtype WIDTH, and a FLOP-matched control.

    https://github.com/jax-ml/jax/issues/18221

Reported: the identical training step, differing only by ``x = jax.lax.complex(x, -x)`` inside
``apply``, took 23.8 s versus 3.3 s -- 7.2x -- at OBS_SIZE=32, BATCH_SIZE=50000, on CPU / jax
0.4.19. Shapes, combiner, ``value_and_grad`` wrapper and the ``jnp.abs`` in the loss are identical
between the arms.

OVERLAP, STATED UP FRONT. ``case_dtype_expansion_assoc_scan.py`` in this directory already covers
this issue: the one-line toggle, ``value_and_grad``, and a scan-LENGTH sweep (8k/50k/131k) at
float32 with paired real controls. No case name here collides with it and this is not a
replacement. It exists because of a specific risk that file cannot cover from inside its own
design, and because of one control it does not have:

  RISK. 7.2x is UNDER the corpus's 10x-vs-control bar, and that 7.2x was measured on CPU, on jax
  0.4.19, with tracing/lowering/compilation/first-run all conflated into one number. Separated and
  moved to a GPU on 0.10.2 it can only get smaller. The sibling's only lever is scan length, which
  changes the number of tree levels but not what happens inside one. This file's lever is dtype
  WIDTH -- complex128-over-float64 instead of complex64-over-float32 -- which doubles the operand
  size of every one of the 4 multiplies and 2 adds that XLA's complex decomposition emits per
  complex multiply-add, at every level of the Blelloch tree AND in the transpose that
  ``value_and_grad`` adds. If dtype is a live axis on this build at all, this is the arm most
  likely to clear 10x. If BOTH files come out under the bar, the two of them together say
  something much stronger than either alone: the effect is real but sub-bar across two widths and
  three lengths, i.e. jax 0.10.2 / current XLA has largely fixed it. That is a result.

  MISSING CONTROL. The real-dtype control does roughly a QUARTER of the runtime FLOPs, because a
  complex multiply is 4 real multiplies. So a 7x compile gap that arrives with a 4x runtime gap is
  a much weaker claim than a 7x compile gap at equal work, and neither file's paired control can
  tell those apart. ``assocscan64_realwide_32k`` fixes that: a REAL arm with the feature width
  raised 32 -> 128, which makes its multiply count per scanned element equal to the complex arm's
  (4 x 32 = 128) while leaving the tree, the combiner, the transpose and the dtype alone. If the
  complex arm is still much slower than that, the gap is structural and not workload. If it is
  not, the "7x" is mostly just arithmetic and the case should be retired.

WHY THE CASE IS INTERESTING AT ALL. Nothing else in the corpus varies DTYPE. The traced program is
the same length in both arms -- verified at trace time on CPU, jaxpr equation counts are within
~10% (642 vs 586 at length 8192), and those extra ~60 equations are the ``lax.complex``/``abs``
plumbing, NOT the 4x multiply expansion. The expansion does not exist in the jaxpr. It is
manufactured downstream, when XLA decomposes complex arithmetic after lowering. So any profiler
that attributes compile cost by counting jaxpr nodes, or by mapping HLO back to source lines, sees
two indistinguishable programs and cannot explain any gap at all. And both arms have the same one
obvious hot region -- the scan -- so "the scan is expensive" is not an answer here; the answer is
one line above the scan, and its cost lands somewhere else entirely.

WHAT THE PAIRED CONTROL ISOLATES. ``complexify=False`` deletes the single ``jax.lax.complex(x,
-x)`` call and changes nothing else -- the ``jnp.repeat``, the combiner, the ``value_and_grad``,
the ``jnp.abs``, the shapes and the inputs are shared code, not a parallel copy, so the arms cannot
drift.

READING THE RESULT. Watch the compile/runtime ratio, not just the compile gap: the complex arm does
~4x the runtime FLOPs of its paired control, so a compile ratio near 4x is consistent with "it is
simply a bigger computation" and only a ratio well clear of that is evidence for the mechanism.
``lower_s`` is reported separately by the harness -- if the gap shows up there rather than in
``compile_s``, the case moves from "XLA pass cost" to "jax tracing cost", which is a different
subject for a profiler and is worth recording as such.

MEMORY AND DTYPE. Inputs are explicitly float64 here (the harness enables x64 globally), so the
complexified arm is complex128 at 16 bytes/element -- deliberately the opposite choice from the
sibling file's float32. At length 65536 with 32 features that is 33.5 MB per intermediate across
~17 tree levels plus the reverse pass, so ``assocscan64_complex_65k`` is the entry that could
plausibly OOM a small GPU; the two shorter lengths exist so a result survives if it does.

PLATFORM. Complex decomposition happens in XLA's algebraic simplification on CPU and GPU alike, so
this is expected to behave similarly on both. A large divergence would be surprising and worth
chasing -- the issue was reported on CPU and the corpus's harness defaults elsewhere.
"""

from __future__ import annotations

import functools

import numpy as np

import jax
import jax.numpy as jnp

OBS = 32          # the issue's feature width; the scan LENGTH is the batch dimension


def _combiner(carry, inc):
    """The issue's combiner, verbatim -- (p, x*p + y). Reproduced as written, not as corrected."""
    _, x = carry
    p, y = inc
    return p, x * p + y


def _apply(params, x, batch, complexify):
    bp = jnp.repeat(jnp.expand_dims(params, 0), batch, axis=0)
    if complexify:
        x = jax.lax.complex(x, -x)        # <-- THE ONLY DIFFERENCE BETWEEN THE ARMS
    return jax.lax.associative_scan(_combiner, (bp, x))[1]


def _loss(params, x, y, batch, complexify):
    return jnp.mean((jnp.abs(_apply(params, x, batch, complexify)) - y) ** 2)


def _step(params, x, y, batch, complexify):
    """value_and_grad of the loss: the forward scan plus its transpose, which is the point."""
    f = functools.partial(_loss, batch=batch, complexify=complexify)
    return jax.value_and_grad(f)(params, x, y)


def _mk(batch, complexify, note, obs=OBS):
    """Inputs are float64, so `complexify` produces complex128 -- the width axis of this file."""
    rng = np.random.default_rng(18221)
    params = np.ones(obs, dtype=np.float64)
    x = rng.standard_normal((batch, obs))
    y = rng.standard_normal((batch, 1))
    return functools.partial(_step, batch=batch, complexify=complexify), (params, x, y), note


CASES = {
    "assocscan64_complex_8k": _mk(
        8_192, True,
        "jax#18221 at complex128 (f64 inputs), scan length 8192 -- ~13 Blelloch tree levels"),
    "assocscan64_complex_8k_control": _mk(
        8_192, False,
        "control: the single lax.complex line removed, float64, length 8192"),

    "assocscan64_complex_32k": _mk(
        32_768, True,
        "jax#18221 at complex128, scan length 32768 -- ~15 tree levels"),
    "assocscan64_complex_32k_control": _mk(
        32_768, False,
        "control: float64 at length 32768"),

    "assocscan64_complex_65k": _mk(
        65_536, True,
        "jax#18221 at complex128, scan length 65536 -- ~17 tree levels, 33.5 MB per intermediate, "
        "the arm most likely to OOM a small GPU"),
    "assocscan64_complex_65k_control": _mk(
        65_536, False,
        "control: float64 at length 65536"),

    # Unpaired on purpose: this is a REFERENCE arm, not a case, so a "below floor" verdict from the
    # harness is the expected reading. Compare its compile time by hand against
    # assocscan64_complex_32k, whose real-multiply count per scanned element it matches (4 x 32).
    "assocscan64_realwide_32k": _mk(
        32_768, False,
        "FLOP-matched reference (unpaired): real float64 with 128 features instead of 32, so its "
        "multiply count equals complex128-at-32 -- if the complex arm is still much slower than "
        "this, the gap is structural rather than arithmetic",
        obs=4 * OBS),
}
