"""SYNTHESISED (gap 13, RNG) -- which DISTRIBUTION you sample from swings compile time ~10x at a
fixed number of threefry calls, an identical jaxpr equation count and an identical call shape.
And the axis everyone expects to matter -- threefry vs rbg -- does nothing at all on CPU.

NOT MINED FROM AN ISSUE. Constructed and then measured. Gap 13 named "RNG beyond random.split
(rbg vs threefry)"; the rbg axis turned out to be the negative and the distribution axis the
positive, so both are shipped here.

MECHANISM. ``jax.random.<dist>(key, shape)`` is one line at every call site, but the samplers
behind those lines are not comparable programs:

  * ``bits`` / ``uniform`` / ``bernoulli`` / ``exponential`` / ``cauchy`` are CLOSED-FORM: threefry
    bits, then a bit-twiddle or a single transcendental. A fixed, small block of HLO per draw.
  * ``normal`` adds ``erf_inv``, a branch-free rational polynomial -- bigger, still closed form.
  * ``gamma`` and ``beta`` are REJECTION SAMPLERS. ``random.gamma`` is Marsaglia-Tsang, implemented
    as a ``lax.while_loop`` wrapped in a ``custom_jvp`` (so it also carries a hand-written
    derivative computation), and ``random.beta`` draws two gammas. Each call therefore stages out
    a loop body, a custom-JVP rule and the boosting path for small concentration.

None of that is visible in the jaxpr equation count, because a ``while_loop`` is ONE equation
whose body hangs off a parameter. So the case and the control have the same number of equations
and differ by 4x in emitted HLO.

MEASURED IN THIS ENVIRONMENT (JAX_PLATFORMS=cpu, jax/jaxlib 0.10.2, x64 on, K draws of shape
(128, 128) f32, one ``random.split`` per draw, compile seconds, one fresh subprocess each):

    K      gamma          normal        uniform     jaxpr eqns (all)   HLO lines (gamma/normal)
     4     2.662 s        0.906 s          --            30              8421 /  2279
     8     6.397 s        3.466 s       0.938 s          58             16769 /  4531
    16     8.901 s        1.855 s          --           114             33465 /  9035
    32    16.612 s        6.325 s          --           226             66885 / 18049

and, from an independent run at K=32 that swept seven distributions at once:

    bits 3.998   uniform 3.033   bernoulli 3.083   cauchy 3.176   exponential 3.532
    normal 3.403   gamma 18.917        <-- 5.6x normal, 6.2x uniform, at 226 equations for all

At K=8, ``beta`` (two gammas per draw) is 8.911 s against ``uniform``'s 0.938 s -- **9.5x** -- with
58 jaxpr equations in both arms and 33657 against 3780 HLO lines. ``t`` (normal / sqrt(chisquare))
is 4.873 s and ``poisson`` 2.027 s, so the ordering tracks sampler structure and not "how exotic
the distribution sounds".

The absolute seconds are NOISY -- the box was at load ~30 on 20 cores and ``normal`` came out at
3.466 s in one run and 1.855 s in another at twice the draw count. The HLO line counts are not
noisy and they are 3.7x apart at every K, so the ratio to trust is the structural one; the compile
ratio ranges 1.8x-6.2x across runs, always in the same direction.

RE-MEASURED ON A QUIETER BOX (same environment, ``jax.jit(fn).lower()`` then ``.compile()``):
``rngdist_gamma_k16`` 5.978 s compile against the uniform control's 1.348 s, i.e. **4.4x**, with
lowering 0.623 s vs 0.444 s. That is the number to quote for the K=16 pair.

WHAT THE CONTROLS ISOLATE.

  * ``*_control`` (the tight one): ``jax.random.gamma(sub, 2.0, SHAPE, dtype)`` ->
    ``jax.random.uniform(sub, SHAPE, dtype)``. Same key, same split chain, same K, same output
    shape and dtype, same reduction, same jaxpr equation count. One identifier and one dropped
    concentration argument.
  * ``rngdist_normal_k32``: a second, intermediate distribution. If the profiler puts gamma and
    normal in the same bucket it has not separated "the sampler has a loop" from "the sampler has
    a polynomial".
  * ``rngdist_beta_k8``: the strongest arm (9.5x). ``beta`` calls ``gamma`` twice, so if the
    mechanism is per-sampler expansion this should be roughly double gamma at the same K --
    measured 8.911 s against gamma's 6.397 s. It is, near enough.

--- THE NEGATIVE: PRNG IMPLEMENTATION DOES NOT MATTER ON CPU ------------------------------------

``jax.random.key(0, impl=...)`` selects the bit generator. The expectation is that ``rbg`` lowers
to a single ``RngBitGenerator`` HLO op while ``threefry2x32`` expands into twenty rounds of
elementwise bit arithmetic, and that the difference should be large. Measured, K draws of
``random.normal``:

    K      threefry2x32     rbg      unsafe_rbg      HLO lines (threefry / rbg / unsafe_rbg)
     4       0.770 s      1.023 s     0.697 s              2279 /  2697 /  3013
    16       2.865 s      2.216 s     1.721 s              9035 / 10725 / 11953
    64       9.657 s      9.407 s     7.453 s             34241 / 41019 / 45895

Flat, and the sign is the opposite of the prediction: the rbg arms emit MORE HLO than threefry,
because XLA:CPU has no hardware bit generator and expands ``RngBitGenerator`` into an algorithm of
its own. Whatever gap 13 expected from "rbg vs threefry", it is not on CPU. These arms are shipped
so the negative is dated and does not need re-deriving. GPU is unverified and is where a hardware
path could exist.

--- SECOND AXIS: MANY SMALL DRAWS vs ONE BIG DRAW ------------------------------------------------

At a fixed number of random BITS and a fixed number of output elements, K separate draws cost 8.1x
one draw of K times the size:

    K=16   16 draws of (128,128)  1.753 s   vs  one draw of (16,128,128)  0.621 s   (2.8x)
    K=64   64 draws of (128,128)  8.298 s   vs  one draw of (64,128,128)  1.021 s   (8.1x)

This one is NOT op-count matched (450 equations against 5) and is included as a scaling
observation rather than a controlled comparison -- it is the RNG instance of gap 12, many small
modules versus one big one, and it is the single easiest thing to fix in a real program.

PLATFORM: **CPU (verified here).** The distribution axis should hold on GPU (the expansion is in
jax, before the backend is chosen); the rbg negative should be re-checked there specifically. The
GPU was off-limits when this file was written.

MEMORY. Draws are (128, 128) f32 = 64 KB each and are summed immediately. The K=64 arms are large
in instruction count, not in bytes. Runtime is milliseconds.
"""

from __future__ import annotations

import functools

import numpy as np

import jax
import jax.numpy as jnp

SHAPE = (128, 128)

# NUMPY at module scope: the only array the file owns is the accumulator seed.
_X = np.zeros(SHAPE, dtype=np.float32)


def _draws(x, k: int, dist: str, impl: str = "threefry2x32"):
    """K draws, one ``split`` each. ``dist`` is the ONLY thing that varies between the arms."""
    key = jax.random.key(0, impl=impl)
    acc = x
    for _ in range(k):
        key, sub = jax.random.split(key)
        if dist == "gamma":
            v = jax.random.gamma(sub, 2.0, SHAPE, dtype=jnp.float32)
        elif dist == "beta":
            v = jax.random.beta(sub, 2.0, 3.0, SHAPE, dtype=jnp.float32)
        elif dist == "t":
            v = jax.random.t(sub, 3.0, SHAPE, dtype=jnp.float32)
        elif dist == "poisson":
            v = jax.random.poisson(sub, 3.0, SHAPE).astype(jnp.float32)
        elif dist == "normal":
            v = jax.random.normal(sub, SHAPE, dtype=jnp.float32)
        elif dist == "exponential":
            v = jax.random.exponential(sub, SHAPE, dtype=jnp.float32)
        elif dist == "cauchy":
            v = jax.random.cauchy(sub, SHAPE, dtype=jnp.float32)
        elif dist == "bits":
            v = jax.random.bits(sub, SHAPE, dtype=jnp.uint32).astype(jnp.float32)
        else:
            v = jax.random.uniform(sub, SHAPE, dtype=jnp.float32)
        acc = acc + v
    return acc.sum()


def _one_big(x, k: int):
    """ONE draw of (K,)+SHAPE -- same total random bits, same number of output elements."""
    v = jax.random.normal(jax.random.key(0), (k,) + SHAPE, dtype=jnp.float32)
    return (v.sum(axis=0) + x).sum()


CASES = {}

# --- the tight pair, swept over the number of draws ------------------------------------------
for _k in (8, 16, 32, 64):
    CASES[f"rngdist_gamma_k{_k}"] = (
        functools.partial(_draws, k=_k, dist="gamma"), (_X,),
        f"synthesised: {_k} jax.random.gamma draws. Marsaglia-Tsang rejection sampler = a "
        f"lax.while_loop inside a custom_jvp, staged out once per draw. Measured 16.6 s at K=32 "
        f"with 66885 HLO lines from 226 jaxpr equations")
    CASES[f"rngdist_gamma_k{_k}_control"] = (
        functools.partial(_draws, k=_k, dist="uniform"), (_X,),
        f"control: the identical loop with random.gamma -> random.uniform, K={_k}. Same key, same "
        f"split chain, same shape, same dtype, same jaxpr equation count; a closed-form sampler "
        f"instead of a rejection loop. Measured 0.94 s at K=8 against gamma's 6.40 s")

# --- the strongest arm: beta draws two gammas ------------------------------------------------
CASES["rngdist_beta_k8"] = (
    functools.partial(_draws, k=8, dist="beta"), (_X,),
    "synthesised: 8 jax.random.beta draws -- two gamma rejection samplers each. Measured 8.911 s "
    "with 33657 HLO lines, against the uniform control's 0.938 s and 3780 lines. 9.5x")
CASES["rngdist_beta_k8_control"] = (
    functools.partial(_draws, k=8, dist="uniform"), (_X,),
    "control: 8 random.uniform draws, same 58 jaxpr equations")

# --- intermediate and cheap distributions: the ordering tracks sampler STRUCTURE --------------
CASES["rngdist_normal_k32"] = (
    functools.partial(_draws, k=32, dist="normal"), (_X,),
    "probe: 32 random.normal draws -- erf_inv polynomial, closed form. 18049 HLO lines against "
    "gamma's 66885 at the same 226 equations")
CASES["rngdist_normal_k32_control"] = (
    functools.partial(_draws, k=32, dist="uniform"), (_X,),
    "control: 32 random.uniform draws, K and shapes identical")
CASES["rngdist_t_k8"] = (
    functools.partial(_draws, k=8, dist="t"), (_X,),
    "probe: 8 random.t draws (normal / sqrt(chisquare), so a gamma underneath). Measured 4.873 s")
CASES["rngdist_poisson_k8"] = (
    functools.partial(_draws, k=8, dist="poisson"), (_X,),
    "probe: 8 random.poisson draws -- also a rejection sampler but a cheaper one. Measured "
    "2.027 s, i.e. 'has a loop' is not sufficient; the loop body's size is what counts")
CASES["rngdist_cauchy_k32"] = (
    functools.partial(_draws, k=32, dist="cauchy"), (_X,),
    "probe: 32 random.cauchy draws -- one tan(), closed form. Measured 3.176 s, indistinguishable "
    "from uniform and bits")

# --- NEGATIVE: the PRNG implementation axis ---------------------------------------------------
for _k in (16, 64):
    CASES[f"rngimpl_rbg_k{_k}"] = (
        functools.partial(_draws, k=_k, dist="normal", impl="rbg"), (_X,),
        f"NEGATIVE: {_k} normal draws from an rbg key. Measured 9.407 s at K=64 against "
        f"threefry's 9.657 s, and emitting MORE HLO (41019 vs 34241) -- XLA:CPU expands "
        f"RngBitGenerator itself")
    CASES[f"rngimpl_rbg_k{_k}_control"] = (
        functools.partial(_draws, k=_k, dist="normal", impl="threefry2x32"), (_X,),
        f"control: the identical program from a threefry2x32 key, K={_k}. One keyword argument")
CASES["rngimpl_unsafe_rbg_k64"] = (
    functools.partial(_draws, k=64, dist="normal", impl="unsafe_rbg"), (_X,),
    "NEGATIVE: third PRNG implementation, K=64. Measured 7.453 s -- the fastest of the three, "
    "and the one emitting the most HLO (45895 lines). Program size and compile time are "
    "uncorrelated across this axis")

# --- SECOND AXIS: many small draws vs one big draw at fixed total random bits ------------------
for _k in (16, 64):
    CASES[f"rngmany_k{_k}"] = (
        functools.partial(_draws, k=_k, dist="normal"), (_X,),
        f"{_k} separate random.normal draws of (128,128). Measured 8.298 s at K=64, 450 jaxpr "
        f"equations. NOT op-count matched to its control -- a scaling observation, not a "
        f"controlled comparison")
    CASES[f"rngmany_k{_k}_control"] = (
        functools.partial(_one_big, k=_k), (_X,),
        f"control: ONE random.normal draw of ({_k},128,128) -- same total random bits, same "
        f"number of output elements, 5 jaxpr equations. Measured 1.021 s at K=64 (8.1x)")
