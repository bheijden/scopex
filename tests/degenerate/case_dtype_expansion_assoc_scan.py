"""jax#18221 -- associative_scan compiles ~7x slower for complex than for real, from one line.

    https://github.com/jax-ml/jax/issues/18221

Reported: the identical training step, differing only by ``x = jax.lax.complex(x, -x)`` inside
``apply``, took 23.8 s versus 3.3 s to get through jit.  Shapes, combiner, ``value_and_grad``
wrapper and the ``jnp.abs`` in the loss are byte-identical between the two arms.

WHY THIS CASE EARNS ITS PLACE IN THE CORPUS.  Every other case varies a knob a profiler can
already see: chain length, matrix size, nesting depth, op count.  This one varies DTYPE, and
nothing else in the corpus does.  The traced program is the same length; the jaxpr has the same
number of ``lax`` calls; the shapes are the same.  What changes is that XLA's complex
decomposition rewrites each complex multiply-add into 4 real multiplies and 2 adds -- at EVERY
level of the log-depth Blelloch tree that ``associative_scan`` builds, and again in the
transposed scan that ``value_and_grad`` adds.  A ~4x change in primitive count buys a ~7x change
in compile time, so the superlinearity is real and it is indexed by something a source-level
node counter cannot see.

For scopex this is the "attribute to the right AXIS" test.  Both arms have one obvious hot
region (the scan).  Pointing at the scan is not an answer; the useful answer names the dtype
promotion, which happens one line above and whose cost lands somewhere else entirely.

MEASURED AT TRACE TIME (CPU, jax 0.10.2, no execution), jaxpr equation counts, complex vs real:

    length   8192     642  vs  586    (+9.6%)
    length  50000     723  vs  659    (+9.7%)
    length 131072     834  vs  762    (+9.4%)

That is the sharpest thing about this case and it was worth checking before spending a
measurement slot on it.  The complex arm is NOT a bigger program at the jaxpr level -- the extra
~60 equations are the complex/abs plumbing, not the 4x multiply expansion.  So whatever compile
gap shows up is manufactured downstream, by XLA decomposing complex arithmetic after lowering.
Any profiler that attributes compile cost by counting jaxpr nodes, or by mapping HLO back to
source lines, will find the two arms indistinguishable and will be unable to explain a 7x gap.
The counts also confirm the intended scaling knob: equations grow with scan LENGTH (642 -> 723
-> 834, i.e. with the log-depth tree) exactly as the mechanism predicts.

CONTROL.  The issue is literally written as a one-line toggle, and that is what the ``_control``
entries are: ``complexify=False`` removes the single ``lax.complex`` call.  Everything else --
the ``jnp.repeat``, the combiner, the ``value_and_grad``, the ``jnp.abs``, the shapes -- is
shared code, not a parallel copy, so the arms cannot drift.

TWO CORRECTIONS TO THE ISSUE'S NUMBERS.

  * The reported 23.8 s / 3.3 s conflate tracing, lowering, compilation and the first execution.
    The harness separates them (``lower`` then ``compile`` then timed runs), so expect the
    compile-only figures to be smaller than the issue's.  If the effect lives in tracing rather
    than in XLA, it will show up in ``lower_s`` and not in ``compile_s`` -- which is itself a
    finding worth recording, because it moves the case from "XLA pass cost" to "jax tracing
    cost" and those are different subjects for a profiler.
  * The complex arm also does roughly 4x the runtime FLOPs.  So the ``compile/runtime >= 1000``
    gate, not the raw compile ratio, is the discriminating test here: a 7x compile ratio that
    comes with a 4x runtime ratio is far less interesting than one that does not.

SCALING.  The knob that should matter is the scan LENGTH, because that sets the depth of the
Blelloch tree and hence the number of distinct sliced sub-computations in the traced program:
8192 -> 13 levels, 50000 -> 16, 131072 -> 17.  OBS (the trailing feature dimension) is deliberately
held at 32, the issue's value: growing it adds FLOPs but no nodes, so if compile time tracks OBS
the mechanism is not what this docstring claims.  Both arms of every length are present so the
scaling can be read off the ratios rather than asserted.

DTYPE AND MEMORY.  Inputs are explicitly float32 -- x64 is on globally in the harness, and f64
inputs would make the complex arm complex128 and double an already-large residual set.  At
length 131072 the complex arm holds roughly 33 MB per intermediate across ~17 tree levels plus
the reverse pass, so it is the one entry here that could plausibly OOM a small GPU; that is why
the two smaller lengths exist below it.

PLATFORM.  Nothing in this mechanism is backend-specific -- complex decomposition happens in
XLA's algebraic simplification on CPU and GPU alike -- so this case is expected to behave
similarly on both, unlike the two GPU-emitter cases in this batch.  A large CPU/GPU divergence
here would be surprising and worth chasing.
"""

from __future__ import annotations

import functools

import numpy as np

import jax
import jax.numpy as jnp

OBS = 32          # the issue's value; held fixed, see SCALING above


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
    """value_and_grad of the loss -- the forward scan plus its transpose, which is the point."""
    f = functools.partial(_loss, batch=batch, complexify=complexify)
    return jax.value_and_grad(f)(params, x, y)


def _mk(batch: int, complexify: bool, note: str):
    rng = np.random.default_rng(18221)
    params = np.ones(OBS, dtype=np.float32)
    x = rng.standard_normal((batch, OBS), dtype=np.float32)
    y = rng.standard_normal((batch, 1), dtype=np.float32)
    fn = functools.partial(_step, batch=batch, complexify=complexify)
    return fn, (params, x, y), note


CASES = {
    "assocscan_complex_8k": _mk(
        8_192, True,
        "jax#18221: complex associative_scan under value_and_grad, length 8192 (~13 tree levels)"),
    "assocscan_complex_8k_control": _mk(
        8_192, False,
        "control: identical program with the single lax.complex line removed, length 8192"),

    "assocscan_complex_50k": _mk(
        50_000, True,
        "jax#18221 at the reported length 50000 (~16 tree levels); reported 23.8 s vs 3.3 s"),
    "assocscan_complex_50k_control": _mk(
        50_000, False,
        "control: real dtype at the reported length 50000"),

    "assocscan_complex_131k": _mk(
        131_072, True,
        "scaled: length 131072 (~17 tree levels); largest arm, the one that may OOM a small GPU"),
    "assocscan_complex_131k_control": _mk(
        131_072, False,
        "control: real dtype at length 131072"),
}
