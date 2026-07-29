"""SYNTHESISED (gap 13, cumulative ops) -- ``lax.cumlogsumexp`` costs 13.2x its own sibling
``lax.cumsum`` to compile under ``jax.grad``, from a one-token change, because only ONE of the five
cumulative primitives has a closed-form AD rule registered.

    mechanism source: jax/_src/lax/control_flow/loops.py, jax 0.10.2, lines 3065-3092:

        cumsum_p        = _cumulative_reduction_primitive("cumsum", lax.add, ...)
        ad.deflinear2(cumsum_p, _cumsum_transpose_rule)          # <-- ONLY cumsum gets this
        cumlogsumexp_p  = _cumulative_reduction_primitive("cumlogsumexp", logaddexp, ...)
        ...
        def _cumulative_jvp_rule(primals, tangents, *, axis, reverse, combine_fn):
            return api.jvp(partial(associative_scan, combine_fn, axis=axis, reverse=reverse),
                           primals, tangents)
        ad.primitive_jvps[cumlogsumexp_p] = partial(_cumulative_jvp_rule, combine_fn=logaddexp)
        ad.primitive_jvps[cumprod_p]      = partial(_cumulative_jvp_rule, combine_fn=lax.mul)
        ad.primitive_jvps[cummin_p]       = partial(_cumulative_jvp_rule, combine_fn=lax.min)
        ad.primitive_jvps[cummax_p]       = partial(_cumulative_jvp_rule, combine_fn=lax.max)

NOT MINED FROM AN ISSUE. Constructed from those registrations and then measured.

MECHANISM. All five cumulative primitives lower identically in the forward direction -- a single
``reduce_window``, 2 jaxpr equations, whatever the reducer. They diverge only under
differentiation:

  * ``cumsum`` is linear, so ``ad.deflinear2`` gives it a closed-form transpose: the cotangent
    goes through one more ``cumsum`` with ``reverse=True``. The reverse-mode program stays at a
    handful of equations however long the axis is.
  * ``cumlogsumexp`` / ``cumprod`` / ``cummax`` / ``cummin`` are not linear and have no transpose
    rule. They fall back to ``_cumulative_jvp_rule``, which throws the primitive away and
    RE-TRACES the whole thing as ``associative_scan`` -- a Blelloch prefix-scan tree of
    ceil(log2 N) up-sweep levels plus the same again down-sweep, each level a slice / combine /
    concatenate triple, and then differentiates THAT. The reverse-mode program therefore grows
    with log(N) levels of slabs, each carrying the full combiner.

So the compile-time knob is which AD rule is registered for a primitive that is otherwise
identical, and the growth axis is the length of the scanned dimension.

MEASURED IN THIS ENVIRONMENT (JAX_PLATFORMS=cpu, jax/jaxlib 0.10.2, x64 on, x = (N, 4) f32,
compile seconds, one fresh subprocess per measurement):

    N          cumlogsumexp grad   cumsum grad   ratio   jaxpr eqns (lse/sum)  HLO lines
       65 536       5.453 s          0.516 s     10.6x        982 / 8          7810 / 228
      262 144       5.565 s          0.587 s      9.5x       1104 / 8          9118 / 280
    1 048 576      14.456 s          1.095 s     13.2x       1226 / 8         10474 / 280

Forward-only, same sizes, no ``jax.grad``: cumlogsumexp 0.595 s against cumsum 0.264 s at
N = 1 048 576, i.e. 2.3x (2.305 s vs 0.938 s, the same 2.5x, on a more heavily loaded run of the
same box). The forward arms are 2 jaxpr equations each. That is the whole point -- the two
primitives are nearly indistinguishable until they are differentiated, and then one of them
becomes a 1226-equation program and the gap goes to 8-13x.

RE-MEASURED ON A QUIETER BOX (same environment, ``jax.jit(fn).lower()`` then ``.compile()``):
``cumad_lse_n262144`` 3.668 s against its control's 0.446 s, i.e. 8.2x. The absolute seconds in
the table above are from a run at load ~30 and are UPPER BOUNDS; the ratio is stable at 8-13x.

Equation count grows by exactly 122 per 4x increase in N in the cumlogsumexp arm (982 -> 1104 ->
1226), i.e. ~61 equations per extra prefix-scan level, which is the log2(N) signature the
mechanism predicts. The cumsum arm stays at 8 equations at every N.

--- A TRAP THIS FILE EXISTS TO DOCUMENT ---------------------------------------------------------

The obvious way to write this case is ``jax.grad(lambda z: op(z, axis=0).sum())``. **Do not.** The
cotangent of that loss is a constant, so the whole reverse scan has constant operands and XLA's
constant folder evaluates it in the HLO interpreter. Measured with that (wrong) loss:

    N = 1 048 576, cumsum:   grad of  sum(cumsum(z))       40.617 s   <-- constant folding
                             grad of  sum(cumsum(z) * z)    1.095 s   <-- what this file uses

A 37x effect that has nothing to do with cumulative ops and everything to do with a mechanism the
corpus already covers (``case_constant_folding_cumsum_seqmask.py``). Every loss in this file is
therefore ``(op(z) * z).sum()``, which keeps the cotangent data-dependent, and the numbers above
are from that form. This is recorded because the wrong version reproduces a LARGER number and
would have been filed as a cumulative-op finding.

WHAT THE CONTROLS ISOLATE.

  * ``*_control`` (the tight one): the identical function with ``lax.cumlogsumexp`` replaced by
    ``lax.cumsum``. Same axis, same shape, same loss, same ``jax.grad``, same everything. One
    identifier. It flips which AD rule is registered and nothing else.
  * ``cumad_fwd_*``: the same two primitives with no ``jax.grad`` at all -- 2 equations each,
    1.3x apart. Pins the cost to the AD rule rather than to the reducer or the lowering.
  * ``cumad_cummax_*`` and ``cumad_cumprod_*``: two more primitives that share
    ``_cumulative_jvp_rule``. If the cost were about ``logaddexp`` being an expensive combiner,
    these would be cheap; measured at N = 65536 with the live-cotangent loss they are 2.546 s
    (6.9x) and 1.780 s (4.8x) against the cumsum control's 0.369 s, so the variable is the RULE,
    not the combiner. ``lax.cummax`` produces 1046 jaxpr equations and ``lax.cumprod`` 470,
    against the control's 8.

--- THE DTYPE AXIS (gap 14's question, answered) ------------------------------------------------

Gap 14 asks whether these pathologies survive, vanish or invert at f32/bf16, and says it is
entirely unmeasured. For this case the answer is: **it survives unchanged.** Same program at
N = 262 144 with the whole computation cast to each dtype:

    dtype      cumlogsumexp grad   cumsum grad   ratio   jaxpr eqns (lse)   HLO lines (lse)
    float32         3.193 s          0.397 s     8.0x         1104               9118
    float64         3.401 s          0.428 s     7.9x         1107               9194
    bfloat16        3.142 s          0.446 s     7.0x         1110               5430

Dtype moves nothing: 7.0-8.0x across a 4x range of element width. The bfloat16 arm is worth
noting on its own -- it emits 40% FEWER HLO lines than float32 and compiles in the same time,
which is a second instance of the "program size is not compile time" observation this corpus keeps
running into. These arms are shipped as ``cumad_dtype_*`` so the invariance is dated rather than
assumed.

WHY THIS EARNS A SLOT. Gap 13 named cumulative ops as having no coverage. Beyond that, this is a
"which rule fires" case on the JAX side rather than the XLA side: the divergence happens in
Python, in a dictionary lookup on ``ad.primitive_jvps``, and it multiplies a 2-equation program by
600x before XLA ever sees it. A profiler that starts at the jaxpr will correctly see a big program
and will have nothing to say about WHY it is big; the answer is one missing ``ad.deflinear2``
line, attributable only above the jaxpr.

PLATFORM: **either.** Measured on CPU here. On GPU ``cumsum`` additionally gets a CUB-backed
lowering (``_cumred_gpu_lowering``, registered for ``platform='gpu'`` and only for ``cumsum_p``),
which should widen the forward-direction gap as well -- so the GPU arm is expected to be at least
as strong, and its forward arms are the interesting difference. Unverified; the GPU was off-limits
when this file was written.

MEMORY. The largest arm is (1 048 576, 4) f32 = 16 MB in, and the prefix-scan tree holds ~2x that
across its levels. The two smaller lengths exist so the case survives on a small device.

NOTE ON THE BOX. Load average was ~30 on 20 cores while these numbers were taken; absolute seconds
are UPPER BOUNDS and the paired ratios are the statistic to trust.
"""

from __future__ import annotations

import functools

import numpy as np

import jax.numpy as jnp
from jax import grad
from jax import lax

# NUMPY at module scope. Ones, not random: the cumulative ops here are numerically delicate in f32
# at N = 1e6 and values cannot affect compile time.
_XS = {n: np.ones((n, 4), dtype=np.float32) for n in (65_536, 262_144, 1_048_576)}


def _live_loss(z, op, axis: int = 0):
    """``* z`` keeps the cotangent DATA-DEPENDENT. Without it this measures constant folding."""
    return (op(z, axis=axis) * z).sum()


def _grad_of(x, op):
    return grad(functools.partial(_live_loss, op=op))(x)


def _fwd_of(x, op):
    return op(x, axis=0).sum()


def _grad_dt(x, op, dt):
    """The same case, with the whole computation cast to ``dt``. Gap 14's axis on gap 13's case."""
    def f(z):
        z = lax.convert_element_type(z, dt)
        return lax.convert_element_type((op(z, axis=0) * z).sum(), jnp.float32)
    return grad(f)(x)


CASES = {}

# --- the one-token pair, swept over the scanned length ---------------------------------------
for _n in (65_536, 262_144, 1_048_576):
    CASES[f"cumad_lse_n{_n}"] = (
        functools.partial(_grad_of, op=lax.cumlogsumexp), (_XS[_n],),
        f"synthesised: grad of lax.cumlogsumexp over a length-{_n} axis. No transpose rule, so "
        f"_cumulative_jvp_rule re-traces the whole thing as an associative_scan tree and "
        f"differentiates that -- 1226 jaxpr equations at N=1e6 against the control's 8")
    CASES[f"cumad_lse_n{_n}_control"] = (
        functools.partial(_grad_of, op=lax.cumsum), (_XS[_n],),
        f"control: identical program with lax.cumlogsumexp -> lax.cumsum, length {_n}. One "
        f"identifier. cumsum is the only cumulative primitive with ad.deflinear2 registered, so "
        f"its reverse pass stays at 8 equations however long the axis")

# --- forward-only: pins the cost to the AD rule, not the primitive or its lowering ------------
CASES["cumad_fwd_lse_n1048576"] = (
    functools.partial(_fwd_of, op=lax.cumlogsumexp), (_XS[1_048_576],),
    "discriminator: lax.cumlogsumexp with NO jax.grad, N=1e6 -- 2 jaxpr equations, measured "
    "0.595 s. Identical structure to the cumsum forward arm")
CASES["cumad_fwd_lse_n1048576_control"] = (
    functools.partial(_fwd_of, op=lax.cumsum), (_XS[1_048_576],),
    "control: lax.cumsum forward, N=1e6 -- 2 jaxpr equations, measured 0.264 s. The two "
    "primitives are 2.3x apart until they are differentiated, against 8-13x after")

# --- two more primitives sharing the same rule: the variable is the RULE, not logaddexp -------
CASES["cumad_cummax_n65536"] = (
    functools.partial(_grad_of, op=lax.cummax), (_XS[65_536],),
    "probe: grad of lax.cummax, N=65536 -- also has no transpose rule, also routes through "
    "_cumulative_jvp_rule. 1046 jaxpr equations, measured 2.546 s against the shared cumsum "
    "control's 0.369 s (6.9x)")
CASES["cumad_cumprod_n65536"] = (
    functools.partial(_grad_of, op=lax.cumprod), (_XS[65_536],),
    "probe: grad of lax.cumprod, N=65536 -- same rule again, a cheap combiner. 470 jaxpr "
    "equations, measured 1.780 s (4.8x). Rules out 'logaddexp is expensive' as the explanation")
CASES["cumad_cummax_n65536_control"] = (
    functools.partial(_grad_of, op=lax.cumsum), (_XS[65_536],),
    "control shared by the two probes above: grad of lax.cumsum at N=65536, 8 jaxpr equations, "
    "measured 0.369 s")

# --- gap 14's axis on gap 13's case: does the pathology survive a dtype change? ---------------
for _dt, _tag, _secs in ((jnp.float32, "f32", "3.193"), (jnp.float64, "f64", "3.401"),
                         (jnp.bfloat16, "bf16", "3.142")):
    CASES[f"cumad_dtype_{_tag}"] = (
        functools.partial(_grad_dt, op=lax.cumlogsumexp, dt=_dt), (_XS[262_144],),
        f"dtype axis: the N=262144 case with the whole computation cast to {_tag}. Measured "
        f"{_secs} s -- the pathology SURVIVES unchanged across a 4x range of element width "
        f"(7.0-8.0x against its control in every dtype)")
    CASES[f"cumad_dtype_{_tag}_control"] = (
        functools.partial(_grad_dt, op=lax.cumsum, dt=_dt), (_XS[262_144],),
        f"control: the same {_tag} program with cumlogsumexp -> cumsum. Measured 0.397 / 0.428 / "
        f"0.446 s for f32 / f64 / bf16 -- also flat in dtype")
