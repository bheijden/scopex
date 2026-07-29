"""SYNTHESISED (gap 14) -- f8_e4m3fn compiles 3.1x slower than f8_e5m2 from a ONE-TOKEN dtype
change, with a byte-identical jaxpr, identical shapes and an identical HLO instruction count.

    source of the mechanism (read, not a bug report):
    https://raw.githubusercontent.com/openxla/xla/main/xla/hlo/transforms/simplifiers/float_normalization.cc

NOT MINED FROM AN ISSUE. There is no bug filed for this; it is constructed from the source of
``FloatNormalization`` plus the CPU elemental emitter, and then measured. Everything below with a
number attached was measured in this environment.

MECHANISM -- THREE STAGES IN ONE PROGRAM, WHICH IS THE POINT.

  (1) ``FloatNormalization`` is a single-sweep HLO pass (no fixpoint). For every instruction whose
      operand or result type the backend does not support natively it inserts one ``convert`` per
      unsupported operand plus one for the result. XLA:CPU supports nothing narrower than f16, so a
      D-step elementwise chain written in f8 becomes ~2D converts wrapped around 2D f32 ops. On its
      own that is a constant factor of about 3.

  (2) The CPU **elemental IR emitter** then has to emit each of those converts. ``f8_e5m2`` has the
      same exponent field as f16 (5 exponent bits, 2 mantissa bits, bias 15, real infinities), so
      f32<->e5m2 lowers to a shift-and-round. ``f8_e4m3fn`` is the *finite* variant: 4 exponent
      bits, 3 mantissa bits, bias 7, **no infinities**, NaN only at the all-ones pattern, and a
      saturating finite max of 448. Every f32<->e4m3fn convert therefore lowers to a long branchy
      bit-manipulation sequence -- clamp, saturate, renormalise subnormals, special-case NaN.

  (3) LLVM then has to optimise that sequence, once per convert, i.e. O(D) times.

So the cost is NOT monotone in dtype width, which is the finding. Measured here (below), f16 is
sometimes FASTER than f32, bf16 lands between them, and the two 8-bit types -- same width, same
op count, same HLO line count -- differ from each other by up to 3.2x.

MEASURED IN THIS ENVIRONMENT (JAX_PLATFORMS=cpu, jax/jaxlib 0.10.2, x64 on, x=(256,256) f32,
compile seconds only, one fresh subprocess per measurement):

    D    f32     f16    bf16   f8_e5m2   f8_e4m3fn   e4m3fn/e5m2   e4m3fn/f32
     8  0.066   0.168   0.117    0.504      0.941        1.87x        14.3x
    16  0.061   0.244   0.269    0.527      1.411        2.68x        23.1x
    32  0.342   0.286   0.284    1.021      3.253        3.19x         9.5x
    48  0.279   0.443   0.505    1.738      5.436        3.13x        19.5x

jaxpr equation counts at those depths: f32 33/65/129/193, every other dtype 35/67/131/195 (two
extra equations, the in and out converts). The two f8 arms are IDENTICAL on every structural
metric: same equation count, same shapes, same primitives, and the same number of unoptimised HLO
lines (99 / 147 / 243 / 339 at D = 8/16/32/48). The ONLY difference between case and control is
the six characters of the dtype name.

Growth is close to linear in D with a large constant (e4m3fn 0.94 -> 1.41 -> 3.25 -> 5.44 for
D = 8 -> 48), so this case is about a per-op CONSTANT, not a superlinear blowup. The depth sweep
is present because the claim "cost is per convert" predicts linearity, and a measured linearity is
what distinguishes this from a pass that is quadratic in module size.

WHAT EACH CONTROL ISOLATES -- there are two, and the pair is the reason this earns a slot.

  * ``*_control`` = **f8_e5m2 at the same depth**. Identical jaxpr, identical shapes, identical op
    count, identical bit width, identical number of inserted converts. This rules out both "narrow
    floats are slow" and "FloatNormalization inserted a lot of converts", because the control pays
    exactly the same amount of both. What is left is the e4m3fn conversion sequence specifically,
    i.e. stage (2) and (3) above.
  * ``f8chain_f32_d*`` = **float32 at the same depth**. Same source, no converts inserted at all.
    This isolates stage (1), FloatNormalization plus convert insertion, as a whole.

If a profiler attributes the e5m2-vs-e4m3fn gap and the f32-vs-e5m2 gap to the same place, it is
wrong: they are different stages of the pipeline. That discrimination is the test this file poses.

WHY THIS BELONGS IN THE CORPUS. Gap 14 says every existing case runs under x64 and that whether
these pathologies survive at f32/bf16 is unmeasured. This file makes dtype the ONLY independent
variable, and finds a 3.1x compile swing with the program held byte-identical. It is also one of
the few cases where the cost provably lands below the HLO layer: no HLO pass timing can explain a
gap between two modules that have the same instruction count and the same opcodes.

CONFIRMING ATTRIBUTION, once measured. Re-run one arm under

    TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE=float_normalization=2 ...

to see the converts being inserted, and dump the LLVM module
(``XLA_FLAGS=--xla_dump_to=... --xla_dump_hlo_pass_re=.*``) to see the e4m3fn conversion sequence.
Setting TF_CPP_VMODULE without lowering TF_CPP_MIN_LOG_LEVEL prints nothing.

PLATFORM: **CPU (verified here).** Almost certainly also GPU -- FloatNormalization is
backend-independent and no NVIDIA GPU before Hopper has native f8 arithmetic -- but the GPU was
off-limits when this file was written, so the GPU arm is unverified. On a Hopper-class device the
e4m3fn arm could invert (native f8 support would remove the converts entirely), which would itself
be a strong result.

MEMORY / RUNTIME. One (256, 256) f32 input, 256 KiB. Runtime is microseconds in every arm; this is
a pure compile-time case and the compile/runtime ratio should be enormous. Inputs are zeros --
values cannot affect compile time, and f8 has so little range that random inputs would saturate.
"""

from __future__ import annotations

import functools

import numpy as np

import jax
import jax.numpy as jnp

SHAPE = (256, 256)

# Exactly representable in every dtype tested, so no arm gets a different constant-folding
# opportunity from the others. 1 + 2^-10.
_C = 1.0009765625


def _chain(x, depth: int, dt):
    """``depth`` steps of (y*c + c ; max(y, y*c)) carried out ENTIRELY in ``dt``.

    Written with ``lax`` primitives rather than ``jnp`` operators so that no implicit promotion
    can sneak an f32 step in and silently change the op count between arms. Verified: equation
    counts are identical for every non-f32 dtype at every depth.
    """
    y = jax.lax.convert_element_type(x, dt)
    c = jnp.asarray(_C, dtype=dt)
    for _ in range(depth):
        y = jax.lax.add(jax.lax.mul(y, c), c)
        y = jax.lax.max(y, jax.lax.mul(y, c))
    return jax.lax.convert_element_type(y, jnp.float32).sum()


def _mk(depth: int, dt, note: str):
    return functools.partial(_chain, depth=depth, dt=dt), (_X,), note


# NUMPY at module scope: importing this file to discover CASES must never claim a device.
_X = np.zeros(SHAPE, dtype=np.float32)

CASES = {}

# --- the one-token pair, swept over depth. Case = e4m3fn, control = e5m2. --------------------
for _d in (16, 32, 48, 96):
    CASES[f"f8chain_e4m3fn_d{_d}"] = _mk(
        _d, jnp.float8_e4m3fn,
        f"synthesised: depth-{_d} elementwise chain in float8_e4m3fn -- non-IEEE bias, no "
        f"infinities, saturating max, so every inserted convert is a branchy bit sequence")
    CASES[f"f8chain_e4m3fn_d{_d}_control"] = _mk(
        _d, jnp.float8_e5m2,
        f"control: byte-identical program in float8_e5m2 at depth {_d} -- same width, same "
        f"equation count, same HLO line count, same inserted converts; only the dtype name "
        f"differs. Measured 3.1x faster at D=48")

# --- second control axis: no converts inserted at all (isolates FloatNormalization itself) ----
for _d in (16, 48):
    CASES[f"f8chain_f32_d{_d}"] = _mk(
        _d, jnp.float32,
        f"second control: the same chain in float32 at depth {_d} -- natively supported, "
        f"FloatNormalization inserts nothing. Isolates convert INSERTION from convert CODEGEN")

# --- non-monotonicity in width: the finding that dtype cost is not ordered by bit count -------
CASES["f8chain_f16_d48"] = _mk(
    48, jnp.float16,
    "probe: float16 at depth 48 -- measured 0.443 s, i.e. only 1.6x float32 and 12x faster than "
    "e4m3fn despite being narrower than f32")
CASES["f8chain_bf16_d48"] = _mk(
    48, jnp.bfloat16,
    "probe: bfloat16 at depth 48 -- measured 0.505 s. Same width as f16, different exponent "
    "field, and both are far cheaper than either 8-bit type")
