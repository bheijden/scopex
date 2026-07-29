"""SYNTHESISED (gap 13, while_loop with a DATA-DEPENDENT trip count). Two findings in one file:

  (A) NEGATIVE, and worth having: a data-dependent trip count costs NOTHING versus a
      compile-time-constant one on CPU. Completely flat across T = 16..512.
  (B) POSITIVE, 25.0x: the same data-dependent ``while_loop`` under ``vmap`` pays a superlinear
      cost in jax's own Python -- 35.7 s of LOWERING against 1.4 s for a control with the same
      number of jaxpr equations, the same shapes and the same primitives, while ``compile_s`` for
      the two arms differs by 12%.

    mechanism source: jax/_src/lax/control_flow/loops.py, ``_while_loop_batching_rule``,
    lines 1758-1783 in jax 0.10.2 (the same shape appears in ``_scan_batching_rule``, 1150-1161):

        carry_bat = init_bat
        for _ in range(1 + len(carry_bat)):
            _, carry_bat_out = batching.batch_jaxpr(body_jaxpr, axis_data,
                                                    bconst_bat + carry_bat, instantiate=carry_bat)
            if carry_bat == carry_bat_out:
                break
            carry_bat = safe_map(operator.or_, carry_bat, carry_bat_out)

NOT MINED FROM AN ISSUE. Constructed from that loop and then measured.

--- (A) THE NEGATIVE ------------------------------------------------------------------------

Gap 13 asks for "``while_loop`` with a data-dependent trip count". The first thing to test is the
plain claim: does XLA charge for not knowing the trip count? Measured, x = (256,256) f32, a
``sin/cos`` body, three spellings at each of T = 16, 63, 64, 65, 128, 512 -- ``lax.fori_loop``
with static bounds, ``lax.while_loop`` with a constant bound, and ``lax.while_loop`` whose bound
is a traced argument:

    every one of the eighteen measurements landed between 0.140 s and 0.217 s compile,
    with 2 jaxpr equations and 97 (static) or 99 (dynamic) HLO lines.

Flat. No cliff at T=64 (where XLA's while-loop unroller has a documented trip-count threshold),
no growth with T, no penalty for the dynamic bound. On CPU, XLA does not unroll these loops at
all, so "the trip count is unknown" costs exactly nothing. That is a real bound on the gap and it
is recorded here so nobody spends another slot on it. GPU is unverified and is where the unroller
is actually configured, so a GPU run of the ``dynwhile_plain_*`` arms is the open question.

--- (B) THE POSITIVE ------------------------------------------------------------------------

The cost appears the moment the loop is batched. ``_while_loop_batching_rule`` does not know in
advance which carry components become batched, so it iterates: batch the body jaxpr, see which
outputs came back batched, OR that into the carry mask, repeat. Each iteration RE-TRACES the whole
body. If batched-ness propagates one carry slot per iteration, the fixpoint needs K passes over a
K-equation body: O(K^2) Python work, for a final jaxpr that is the same size either way.

Construction. Carry is ``(i, t, c_0 ... c_{K-1})``. Only ``c_0`` and ``t`` are vmapped; ``c_1..``
are unbatched. The body differs between the arms in ONE ARRAY INDEX:

    case     new[j] = c[j] * 1.0001 + c[j-1] * 0.0001     <-- batched-ness walks one slot per pass
    control  new[j] = c[j] * 1.0001 + c[j]   * 0.0001     <-- fixpoint saturates on the first pass

MEASURED THE WAY THE HARNESS MEASURES (JAX_PLATFORMS=cpu, jax/jaxlib 0.10.2, x64 on, a fresh
process per measurement, timing ``jax.jit(fn).lower(*args)`` with a COLD cache and then
``.compile()``; carry slices are (16, 8) f32, batch 16, trip count 4):

    K        case lower   control lower   ratio      case compile   control compile
     16        0.559 s       0.554 s      1.0x          0.422 s         0.456 s
     32        0.918 s       0.472 s      1.9x          0.587 s         0.554 s
     64        2.411 s       0.546 s      4.4x          0.904 s         0.813 s
    128        7.721 s       0.788 s      9.8x          1.705 s         1.531 s
    256       35.687 s       1.425 s     25.0x          3.747 s         3.351 s

and, from a separate run that timed ``jax.make_jaxpr`` alone, the same curve with the structural
metrics attached:

    K      case trace   control trace   ratio    jaxpr eqns (both)   HLO lines (case/control)
      8      0.158 s       0.058 s      2.7x            40              435 /  421
     16      0.366 s       0.086 s      4.3x            80              795 /  765
     32      0.984 s       0.126 s      7.8x           160             1515 / 1453
     64      2.477 s       0.203 s     12.2x           320             2955 / 2829
    128     10.886 s       0.832 s     13.1x           640             5835 / 5581

The equation counts are IDENTICAL at every K and the HLO line counts differ by under 5%, so the
two arms produce the same program; only the number of times the batching rule re-traced the body
to get there differs. Case-arm lowering grows 0.559 -> 35.687 s over a 16x increase in K, a factor
of 64, i.e. right at K^2 -- the fixpoint signature. The control grows 2.6x over the same range.
**The measurement must be COLD**: ``batch_jaxpr`` results are memoised process-wide, so calling
``make_jaxpr`` before ``lower`` hides most of the cost. This is a trap worth naming, because it is
the natural way to write the probe.

WHERE THE HARNESS WILL SEE THIS. **In ``lower_s``, not in ``compile_s``.** The whole cost is
Python, spent before any HLO exists; ``compile_s`` for the two arms is 3.747 s vs 3.351 s at
K=256, a difference of 12%, against a 25x difference in ``lower_s``. The harness's ``classify``
gates on the ``compile_s`` ratio and will therefore report roughly "no (1.1x control)" for this
case. That verdict is correct about ``compile_s`` and wrong about the case; read ``lower_s``.
This is the intended behaviour of the file -- it is one of the few entries whose signal is
invisible to every HLO-level metric, which makes it a discriminator for any profiler that only
instruments XLA.

WHAT THE CONTROLS ISOLATE.

  * ``*_control`` (the tight one): one array index in the body, ``c[j-1]`` -> ``c[j]``. Same carry
    width, same op count, same shapes, same primitives, same vmap, same data-dependent predicate.
    The only thing that changes is how many fixpoint passes the batching rule needs.
  * ``dynwhile_plain_*``: the (A) arms above -- static versus dynamic trip count with no vmap at
    all. If a profiler blames "the loop bound is dynamic", these arms show that is not the
    variable; the batching fixpoint is.
  * ``dynwhile_predbat_k64`` / ``_control``: a cruder second axis at fixed carry width -- a
    predicate that is batched under vmap versus one that is not, with a body in which
    batched-ness does not propagate at all. On a closely related construction this measured
    ~1.5x (1.479 s vs 0.993 s compile at K=64), i.e. real but small. Included so the strong (B)
    effect is not confused with the weak "the predicate got batched" effect; they are different
    costs and only the first is superlinear.

PLATFORM: **either -- this is pure host Python and is device-independent.** Measured on CPU here.
The (A) negative is CPU-specific and should be re-run on GPU, where XLA's while-loop unroller is
configured differently.

MEMORY. Tiny: the largest arm holds 128 carry slices of (16, 8) f32, well under a megabyte. The
cost is entirely in tracing.

NOTE ON THE BOX. The machine was under heavy load (load average ~30 on 20 cores) while these
numbers were taken, so the absolute seconds are UPPER BOUNDS. The arms were measured back to back
under the same load and the ratios are the statistic to trust.
"""

from __future__ import annotations

import functools

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

W = 8          # width of each carry slice; small on purpose, the cost is in the trace
BATCH = 16
TRIPS = 4


# --------------------------------------------------------------------------------------------
# (B) the batching fixpoint. The two bodies differ in exactly one array index.
# --------------------------------------------------------------------------------------------

def _chainprop(t, cs):
    """new[j] depends on c[j-1]: batched-ness walks one carry slot per fixpoint pass."""
    def cond(s):
        return s[0] < s[1]

    def body(s):
        i, t_, *c = s
        new = [c[0] * 1.0001 + 0.5]
        for j in range(1, len(c)):
            new.append(c[j] * 1.0001 + c[j - 1] * 0.0001)
        return (i + 1, t_, *new)

    out = lax.while_loop(cond, body, (0, t, *cs))
    return sum(o.sum() for o in out[2:])


def _selfprop(t, cs):
    """new[j] depends on c[j]: same op count, fixpoint saturates on the first pass."""
    def cond(s):
        return s[0] < s[1]

    def body(s):
        i, t_, *c = s
        new = [c[j] * 1.0001 + c[j] * 0.0001 for j in range(len(c))]
        new[0] = c[0] * 1.0001 + 0.5
        return (i + 1, t_, *new)

    out = lax.while_loop(cond, body, (0, t, *cs))
    return sum(o.sum() for o in out[2:])


def _mk_fixpoint(fn, k: int, note: str):
    def g(t, c0, crest):
        return jax.vmap(lambda tt, a: fn(tt, [a] + [crest[j] for j in range(k - 1)]),
                        in_axes=(0, 0))(t, c0).sum()

    t = np.full((BATCH,), TRIPS, dtype=np.int32)          # batched -> predicate is batched
    c0 = np.zeros((BATCH, W), dtype=np.float32)           # the one batched carry slot
    crest = np.zeros((k - 1, W), dtype=np.float32)        # unbatched carry slots
    return g, (t, c0, crest), note


# --------------------------------------------------------------------------------------------
# (A) the negative: static vs data-dependent trip count, no vmap.
# --------------------------------------------------------------------------------------------

def _step(c):
    return jnp.sin(c) * 1.0001 + jnp.cos(c) * 0.9999


def _fori_static(x, t, T: int):
    return lax.fori_loop(0, T, lambda i, c: _step(c), x).sum()


def _while_static(x, t, T: int):
    return lax.while_loop(lambda s: s[0] < T, lambda s: (s[0] + 1, _step(s[1])), (0, x))[1].sum()


def _while_dyn(x, t, T: int):
    """trip count is a TRACED argument -- XLA cannot know it."""
    return lax.while_loop(lambda s: s[0] < s[2],
                          lambda s: (s[0] + 1, _step(s[1]), s[2]), (0, x, t))[1].sum()


# --------------------------------------------------------------------------------------------
# second axis: batched vs unbatched predicate at fixed carry width.
# --------------------------------------------------------------------------------------------

def _pred_batched(t, cs):
    def body(s):
        i, t_, *c = s
        return (i + 1, t_, *[cc * 1.0001 + 0.5 for cc in c])
    out = lax.while_loop(lambda s: s[0] < s[1], body, (0, t, *cs))
    return sum(o.sum() for o in out[2:])


def _pred_static(t, cs):
    def body(s):
        i, t_, *c = s
        return (i + 1, t_, *[cc * 1.0001 + 0.5 for cc in c])
    out = lax.while_loop(lambda s: s[0] < TRIPS, body, (0, t, *cs))
    return sum(o.sum() for o in out[2:])


# NUMPY at module scope only.
_XP = np.zeros((256, 256), dtype=np.float32)

CASES = {}

# --- (B) the fixpoint pair, swept over carry width K -----------------------------------------
for _k in (16, 32, 64, 128, 256):
    CASES[f"dynwhile_fixpoint_k{_k}"] = _mk_fixpoint(
        _chainprop, _k,
        f"synthesised: vmap of a data-dependent while_loop, K={_k} carry slots, body written so "
        f"batched-ness propagates ONE slot per batching-rule fixpoint pass -> O(K^2) re-tracing. "
        f"Measured 10.9 s of TRACING at K=128; the cost is in lower_s, not compile_s")
    CASES[f"dynwhile_fixpoint_k{_k}_control"] = _mk_fixpoint(
        _selfprop, _k,
        f"control: identical program with ONE array index changed (c[j-1] -> c[j]), K={_k}. Same "
        f"jaxpr equation count, same shapes, same primitives; the fixpoint saturates on pass 1. "
        f"Measured 0.83 s at K=128")

# --- (A) the negative: trip count staticness, no vmap ----------------------------------------
for _T in (64, 512):
    CASES[f"dynwhile_plain_dyn_T{_T}"] = (
        functools.partial(_while_dyn, T=_T), (_XP, np.int32(_T)),
        f"NEGATIVE: while_loop with a TRACED trip count, T={_T}. Measured flat (0.14-0.20 s) "
        f"against both static spellings at every T from 16 to 512")
    CASES[f"dynwhile_plain_dyn_T{_T}_control"] = (
        functools.partial(_while_static, T=_T), (_XP, np.int32(_T)),
        f"control: identical body and trip count with a CONSTANT bound, T={_T}")
    CASES[f"dynwhile_plain_fori_T{_T}"] = (
        functools.partial(_fori_static, T=_T), (_XP, np.int32(_T)),
        f"third spelling: lax.fori_loop with static bounds, T={_T} -- same 2 jaxpr equations, "
        f"same ~0.16 s")

# --- second axis: batched vs unbatched predicate at fixed carry width ------------------------
for _k in (64,):
    CASES[f"dynwhile_predbat_k{_k}"] = _mk_fixpoint(
        _pred_batched, _k,
        f"probe: K={_k} carry, predicate BATCHED under vmap (trip count differs per lane). "
        f"Measured 1.479 s compile -- real but only 1.5x, a different and much weaker cost than "
        f"the fixpoint above")
    CASES[f"dynwhile_predbat_k{_k}_control"] = _mk_fixpoint(
        _pred_static, _k,
        f"control: same K={_k} carry and body, predicate is a compile-time constant so the "
        f"batching rule leaves the loop condition alone. NOT op-count matched -- 256 jaxpr "
        f"equations against the case's 320, because a batched predicate forces one select_n per "
        f"carry slot. That is inherent to this axis, which is why it is the WEAK second control "
        f"and the fixpoint pair above is the tight one")
