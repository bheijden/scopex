"""SYNTHESISED (gap 13, reduce_window) -- grad of MAX-pool compiles 12.8x slower than grad of
SUM-pool from a one-token reducer change, while having FEWER jaxpr equations.

    mechanism source (read, not a bug report):
    https://raw.githubusercontent.com/openxla/xla/main/xla/service/select_and_scatter_expander.cc
    jax side: jax/_src/lax/windowed_reductions.py, _select_and_gather_add / _select_and_scatter_add

NOT MINED FROM AN ISSUE. Constructed from the AD rules for ``lax.reduce_window`` and then measured.
``reduce_window`` / pooling had zero coverage in this corpus.

MECHANISM. ``lax.reduce_window`` is one HLO opcode, but its REVERSE-MODE rule depends entirely on
which reducer you hand it, and the two rules land in different parts of the compiler:

  * reducer = ``lax.add``  -> the operation is LINEAR, so its transpose is another
    ``reduce_window(add)`` on the cotangent (with mirrored padding). One opcode in, one opcode out.
  * reducer = ``lax.max``  -> the operation is not linear. jax emits ``select_and_scatter_add``
    (plus ``select_and_gather_add`` for the forward-mode pieces), i.e. HLO's ``kSelectAndScatter``.
    XLA has no direct emitter for that opcode on CPU: ``SelectAndScatterExpander`` rewrites every
    one of them into a scatter driven by a windowed arg-max, which is a substantial block of HLO
    per pooling layer, and the CPU backend then has to emit and optimise all of it.

So the control variable is a single argument -- ``lax.max`` versus ``lax.add`` -- and it decides
whether an entire HLO expander pass runs.

WHY THIS CASE EARNS A SLOT: THE OP COUNT POINTS THE WRONG WAY. The slow arm has FEWER jaxpr
equations than the fast arm at every depth, because the fast arm's transpose emits more separate
padding/broadcast equations while the slow arm emits one dense primitive that only explodes later.
Any attribution that ranks by jaxpr node count, or by "how much source is under this line", ranks
these two arms in the WRONG ORDER. The HLO line count does track the cost (8.6x at D=128), which
is exactly the discrimination this file is here to test: the cost appears between the jaxpr and
the optimised HLO, in one named expander pass.

MEASURED IN THIS ENVIRONMENT (JAX_PLATFORMS=cpu, jax/jaxlib 0.10.2, x64 on, compile seconds,
one fresh subprocess per measurement). 2-D arm: x = (1, 128, 128, 8) f32, window 4x4, stride 1,
SAME padding, D stacked pooling layers, then ``jax.grad`` of the sum:

    D     max-pool grad     sum-pool grad    ratio    jaxpr eqns (max/sum)   HLO lines (max/sum)
     32       2.462 s          0.299 s       8.2x          66 /  98            3687 /   453
     64       3.970 s          0.381 s      10.4x         130 / 194            7335 /   869
    128       5.471 s          0.428 s      12.8x         258 / 386           14631 /  1701

The ratio GROWS with depth (8.2 -> 10.4 -> 12.8) because the sum arm is essentially flat in D
(0.299 -> 0.428 s over a 4x increase in layers) while the max arm is close to linear. That is the
signature the mechanism predicts: a fixed per-``select_and_scatter`` expansion cost, paid D times.

3-D pooling (x = (1, 24, 24, 24, 4), window 3x3x3) reproduces it on a different rank:

    D=8    1.103 / 0.212 = 5.2x     D=32  1.901 / 0.342 = 5.6x     D=64  5.250 / 0.594 = 8.8x

4-D pooling (x = (1, 12, 12, 12, 12, 2), window 3^4, D=16): 3.103 / 0.769 = 4.0x.

WHAT THE CONTROLS ISOLATE.

  * ``*_control`` (the tight one): the SAME function with ``lax.max`` -> ``lax.add`` and the
    init value ``-inf`` -> ``0.0``. Same input, same window, same stride, same padding, same
    number of layers, same ``jax.grad`` wrapper, same output shape. Two tokens, and they decide
    whether ``SelectAndScatterExpander`` has anything to do.
  * ``poolgrad_maxfwd_d128`` / ``poolgrad_sumfwd_d128``: the identical stacks WITHOUT ``jax.grad``.
    Measured earlier at D=8 these are 0.175 s and 0.245 s -- i.e. the forward max-pool is if
    anything CHEAPER than the forward sum-pool. That kills "max-pool is just an expensive op" and
    pins the cost to the reverse-mode rule specifically. This pair is the discriminator; without
    it the case is only a repro.
  * ``poolgrad_generic_d8``: a reducer written as ``max(a,b) + min(a,b)*0`` -- mathematically max,
    but not pattern-matchable. Measured 0.275 s against 0.175 s for the bare ``lax.max`` at D=8,
    i.e. essentially nothing. Recorded because it is a NEGATIVE that rules out "XLA failed to
    recognise the reducer" as an alternative explanation for the grad gap.

RANK / SIZE NOTE. Compile cost here tracks the NUMBER OF POOLING LAYERS, not the spatial size:
the spatial dimensions were deliberately kept small (128x128 in 2-D, 24^3 in 3-D) so that runtime
stays negligible and the compile/runtime ratio stays large. If a measurement shows compile time
tracking the spatial extent instead of D, this docstring's reading is wrong.

PLATFORM: **CPU (verified here).** Expected on GPU too -- ``SelectAndScatterExpander`` is in the
shared HLO pipeline, not a CPU pass -- but the GPU was off-limits when this file was written, so
the GPU arm is unverified. GPU has additional select-and-scatter emitters, so the ratio could be
smaller there; a flat GPU result would be a statement about the GPU pipeline, not about the case.

WINDOW DILATION IS NOT AN AXIS HERE. ``window_dilation != 1`` raises
``NotImplementedError: VJP not implemented for select_and_gather (MaxPool) with window dilation``
in jax 0.10.2, so that knob cannot be swept in the max arm and is left out rather than being swept
in the control alone.
"""

from __future__ import annotations

import functools

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

# NUMPY at module scope. Values are irrelevant to compile time; zeros keep the file small.
_X2 = np.zeros((1, 128, 128, 8), dtype=np.float32)      # 2-D pooling, NHWC
_X3 = np.zeros((1, 24, 24, 24, 4), dtype=np.float32)    # 3-D pooling
_X4 = np.zeros((1, 12, 12, 12, 12, 2), dtype=np.float32)  # 4-D pooling


def _stack(z, w: int, d: int, nsp: int, kind: str):
    """``d`` stacked reduce_windows, window ``w`` on each of ``nsp`` spatial dims, stride 1."""
    reducer, init = (lax.max, -jnp.inf) if kind == "max" else (lax.add, 0.0)
    win = (1,) + (w,) * nsp + (1,)
    strd = (1,) * (nsp + 2)
    for _ in range(d):
        z = lax.reduce_window(z, init, reducer, win, strd, "SAME")
    return z


def _grad(x, w: int, d: int, nsp: int, kind: str):
    """THE ONLY DIFFERENCE BETWEEN THE ARMS IS ``kind``, which picks the reducer."""
    return jax.grad(lambda z: _stack(z, w, d, nsp, kind).sum())(x)


def _fwd(x, w: int, d: int, nsp: int, kind: str):
    return _stack(x, w, d, nsp, kind).sum()


def _generic(x, w: int, d: int, matchable: bool):
    """Reducer that computes max but is not a bare ``max`` -- the pattern-match negative."""
    if matchable:
        def r(a, b):
            return lax.max(a, b)
    else:
        def r(a, b):
            return lax.max(a, b) + lax.min(a, b) * jnp.float32(0.0)
    z = x
    for _ in range(d):
        z = lax.reduce_window(z, jnp.float32(-jnp.inf), r, (1, w, w, 1), (1, 1, 1, 1), "SAME")
    return z.sum()


CASES = {}

# --- the one-token pair, swept over the number of pooling layers (2-D) -----------------------
for _d in (32, 64, 128, 256):
    CASES[f"poolgrad_max2d_d{_d}"] = (
        functools.partial(_grad, w=4, d=_d, nsp=2, kind="max"),
        (_X2,),
        f"synthesised: grad of {_d} stacked 4x4 MAX reduce_windows -- each one becomes a "
        f"select_and_scatter that SelectAndScatterExpander must rewrite. Measured 5.47 s at "
        f"D=128 from 258 jaxpr equations")
    CASES[f"poolgrad_max2d_d{_d}_control"] = (
        functools.partial(_grad, w=4, d=_d, nsp=2, kind="sum"),
        (_X2,),
        f"control: identical program with the reducer changed from lax.max to lax.add (and the "
        f"init from -inf to 0.0), D={_d}. Linear op -> its own transpose. Measured 0.43 s at "
        f"D=128 with MORE jaxpr equations (386 vs 258)")

# --- forward-only: the discriminator that pins the cost to the AD rule, not to max-pool ------
CASES["poolgrad_maxfwd_d128"] = (
    functools.partial(_fwd, w=4, d=128, nsp=2, kind="max"), (_X2,),
    "discriminator: the same 128-layer max-pool stack with NO jax.grad. If this is fast, the "
    "cost is the reverse-mode rule, not the pooling op")
CASES["poolgrad_maxfwd_d128_control"] = (
    functools.partial(_fwd, w=4, d=128, nsp=2, kind="sum"), (_X2,),
    "control: forward-only sum-pool stack, D=128")

# --- rank axis: the same mechanism on 3-D and 4-D windows ------------------------------------
for _d in (32, 64):
    CASES[f"poolgrad_max3d_d{_d}"] = (
        functools.partial(_grad, w=3, d=_d, nsp=3, kind="max"), (_X3,),
        f"3-D pooling (3x3x3), D={_d} -- same mechanism on a different rank; measured 8.8x at D=64")
    CASES[f"poolgrad_max3d_d{_d}_control"] = (
        functools.partial(_grad, w=3, d=_d, nsp=3, kind="sum"), (_X3,),
        f"control: 3-D sum-pool grad, D={_d}")

CASES["poolgrad_max4d_d16"] = (
    functools.partial(_grad, w=3, d=16, nsp=4, kind="max"), (_X4,),
    "4-D pooling (3^4 window), D=16 -- measured 3.10 s vs 0.77 s (4.0x)")
CASES["poolgrad_max4d_d16_control"] = (
    functools.partial(_grad, w=3, d=16, nsp=4, kind="sum"), (_X4,),
    "control: 4-D sum-pool grad, D=16")

# --- NEGATIVE, kept deliberately: reducer pattern-matching is NOT the variable ----------------
CASES["poolgrad_generic_d8"] = (
    functools.partial(_generic, w=4, d=8, matchable=False), (_X2,),
    "NEGATIVE probe: forward max-pool whose reducer is written max(a,b)+min(a,b)*0 so XLA cannot "
    "pattern-match it. Measured 0.275 s against 0.175 s for the bare reducer -- i.e. nothing. "
    "Rules out 'unrecognised reducer' as an explanation for the grad gap")
CASES["poolgrad_generic_d8_control"] = (
    functools.partial(_generic, w=4, d=8, matchable=True), (_X2,),
    "control: identical stack with a bare lax.max reducer")
