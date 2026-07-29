"""mlcommons/algorithmic-efficiency#877 -- ``jnp.cumsum`` over an ALL-CONSTANT array is evaluated by
the compiler, not by the GPU.

    https://github.com/mlcommons/algorithmic-efficiency/issues/877

The reported program is the ``sequence_mask`` helper that the JAX speech-model ecosystem copied out
of lingvo and that appears verbatim in several conformer/deepspeech implementations::

    b = jnp.cumsum(jnp.ones([batch, maxlen]), axis=-1)
    return b <= jnp.expand_dims(lengths, -1)

The operand of the cumsum is a broadcast of the literal 1.0, so every instruction the cumsum lowers
to has all-constant operands, and XLA's constant folder evaluates the whole ladder in the HLO
interpreter at compile time to produce a literal it could have written down in closed form.

MEASURED IN THIS ENVIRONMENT (jax 0.10.2, CUDA GPU, x64 on), batch=64::

    maxlen        500     2000     5000        broadcast-arange control @ 5000
    compile     0.866 s  1.347 s  1.860 s              0.685 s      (2.7x)

That is under both the 3.0 s floor and the 10x bar, which is why THIS FILE PUSHES THE SIZES. Note
what the growth curve says: subtracting the ~0.69 s fixed cost leaves 0.18 / 0.66 / 1.17 s at
maxlen 500 / 2000 / 5000, which is near-LINEAR in maxlen, not quadratic. That is a correction to
the issue's own framing and it matters. ``jnp.cumsum`` only lowers to ``reduce_window`` on TPU; on
CPU and GPU ``lax.cumsum`` lowers via ``associative_scan``, i.e. log2(maxlen) levels of
pad/slice/add, so the folded work is O(batch * maxlen * log maxlen) and not the O(batch * maxlen^2)
an interpreted reduce-window would cost. Extrapolating linearly, maxlen=20000 at batch=64 should
land near 6 s and batch=256 near 21 s.

Because the reported opcode (reduce-window) is NOT what fires on this backend, this file carries a
second group of entries that construct the reduce-window explicitly via ``lax.reduce_window``, so
the opcode the issue names is actually exercised on the hardware we have. Those are deliberately
kept SMALL (maxlen 512 and 1024): interpreted reduce-window really is quadratic, and at maxlen=5000
the folder would be asked to perform ~1.6e9 interpreted element operations.

WHY THIS CASE EARNS A SLOT ALONGSIDE jax#12789. Same family, different opcode, and -- the reason it
is worth a separate slot -- the constant-ness is INCIDENTAL. jax#12789 is a synthetic four-liner
where a human wrote a 500^3 literal on purpose. Here nobody intended to fold anything: an author
wrote a masking helper whose cumsum happens to have a static operand because the lengths come in
separately. The provenance is what a profiler will actually meet in the wild, and the guilty
subexpression is three tokens long inside an idiomatic library function.

WHAT THE CONTROLS ISOLATE.

  * ``_control`` (the tight one, and the reason to prefer this over the arange control): keep the
    cumsum, keep the shape, keep the dtype, keep the op count -- only make the operand a PARAMETER
    instead of a literal. The two arms differ by one argument. Identical HLO modulo where the
    operand comes from; the folder can fire on one and not the other, and nothing else differs.
    The parameter is ``np.zeros`` rather than ``np.ones`` on purpose: XLA cannot see a parameter's
    values, so runtime and compile are unaffected, and calloc-backed zeros cost no physical host
    RAM at import.
  * ``seqmask_arange_*``: the ORIGINAL fix -- replace the cumsum with
    ``jnp.broadcast_to(jnp.arange(1, maxlen+1), (batch, maxlen))``, which is bit-identical output
    with no scan ladder at all. Looser as a control (the op graph changes) but it is what a user
    would actually write, and current XLA explicitly refuses to fold broadcasts, so it should stay
    cheap at every size.
  * ``redwin_const_*`` / ``_control``: the issue's named opcode, forced. Same parameter-vs-literal
    split.

IF THE CURVE IS FLAT the folder's size guards now cover this and that is the result; run one arm
with ``TF_CPP_MIN_LOG_LEVEL=0`` and grep stderr for "Constant folding an instruction is taking",
which is XLA naming the guilty instruction itself and is the ground truth any attribution should
be scored against.

VERIFIED AT TRACE TIME (CPU, jax 0.10.2, no execution). The case arm stages out as
``broadcast_in_dim 1.0`` feeding ``cumsum[axis=1]`` -- jax does NOT constant-fold it itself, so the
constant really is handed to XLA, which is the precondition for the whole mechanism. The control's
jaxpr is the same chain with the broadcast replaced by a parameter: 6 equations against 5, and the
one that differs is the operand's origin.

MEMORY. The largest parameter is 256x20000 float64 = 41 MB, all calloc-backed at import. The
compile-time cost, if any, is the folder's own host-side literals.
"""

from __future__ import annotations

import functools

import jax.numpy as jnp
import numpy as np
from jax import lax

# (batch, maxlen). 5000 is the point already measured (1.860 s); the rest extrapolate the
# near-linear curve upward past the 3.0 s floor along both axes independently.
SEQMASK_SIZES = ((64, 5000), (64, 20000), (256, 20000), (64, 50000))

# Interpreted reduce-window is genuinely O(batch * maxlen^2), so these stay small on purpose.
REDWIN_SIZES = ((64, 512), (64, 1024))


def _seqmask_const(lengths, batch: int, maxlen: int):
    """Verbatim from the issue: the cumsum operand is a broadcast of the literal 1.0."""
    b = jnp.cumsum(jnp.ones([batch, maxlen]), axis=-1)
    return (b <= jnp.expand_dims(lengths, -1)).astype(jnp.float32).sum()


def _seqmask_param(lengths, ones):
    """CONTROL: same cumsum, same shape, same dtype -- operand is a parameter, so nothing folds."""
    b = jnp.cumsum(ones, axis=-1)
    return (b <= jnp.expand_dims(lengths, -1)).astype(jnp.float32).sum()


def _seqmask_arange(lengths, batch: int, maxlen: int):
    """CONTROL (the user-facing fix): bit-identical output, no scan ladder to fold."""
    b = jnp.broadcast_to(jnp.arange(1, maxlen + 1, dtype=jnp.float64), (batch, maxlen))
    return (b <= jnp.expand_dims(lengths, -1)).astype(jnp.float32).sum()


def _cumulative_reduce_window(x, maxlen: int):
    """A cumsum spelled as the opcode the issue names, which is what TPU lowering would emit."""
    return lax.reduce_window(
        x, jnp.array(0.0, x.dtype), lax.add,
        window_dimensions=(1, maxlen), window_strides=(1, 1),
        padding=((0, 0), (maxlen - 1, 0)))


def _redwin_const(batch: int, maxlen: int):
    """All-constant operand: the folder is invited to interpret the whole window reduction."""
    return _cumulative_reduce_window(jnp.ones([batch, maxlen]), maxlen).sum()


def _redwin_param(x, maxlen: int):
    """CONTROL: identical reduce-window on a parameter."""
    return _cumulative_reduce_window(x, maxlen).sum()


def _lengths(batch: int) -> np.ndarray:
    return np.arange(batch, dtype=np.float64)


def _zeros(batch: int, maxlen: int) -> np.ndarray:
    # calloc-backed: virtual only, so importing this file to discover CASES allocates nothing.
    # A parameter's VALUES are invisible to XLA, so zeros and ones compile and run identically.
    return np.zeros((batch, maxlen), dtype=np.float64)


CASES = {}

for _b, _m in SEQMASK_SIZES:
    _k = f"seqmask_cumsum_b{_b}_m{_m}"
    CASES[_k] = (
        functools.partial(_seqmask_const, batch=_b, maxlen=_m), (_lengths(_b),),
        f"algorithmic-efficiency#877: cumsum over a constant [{_b},{_m}] ones array -- "
        f"{_b * _m / 1e6:.2f}M elements x log2({_m}) associative-scan levels, all folded at "
        f"compile time",
    )
    CASES[f"{_k}_control"] = (
        _seqmask_param, (_lengths(_b), _zeros(_b, _m)),
        f"control: same cumsum, same [{_b},{_m}] shape and dtype, operand is a PARAMETER -- "
        f"one argument's provenance is the only difference",
    )

# The user-facing fix, at the largest size only: closed-form arange broadcast, no ladder at all.
_b, _m = SEQMASK_SIZES[-1]
CASES[f"seqmask_arange_b{_b}_m{_m}"] = (
    functools.partial(_seqmask_arange, batch=_b, maxlen=_m), (_lengths(_b),),
    f"control B: broadcast_to(arange(1,{_m + 1})) instead of cumsum -- bit-identical output, "
    f"XLA refuses to fold broadcasts, so this should stay flat at any size",
)

for _b, _m in REDWIN_SIZES:
    _k = f"redwin_const_b{_b}_m{_m}"
    CASES[_k] = (
        functools.partial(_redwin_const, batch=_b, maxlen=_m), (),
        f"the issue's named opcode, forced: lax.reduce_window(add) over a constant [{_b},{_m}] "
        f"ones array -- interpreted folding here is O(batch*maxlen^2) = {_b * _m * _m / 1e6:.0f}M "
        f"element ops, hence the small sizes",
    )
    CASES[f"{_k}_control"] = (
        functools.partial(_redwin_param, maxlen=_m), (_zeros(_b, _m),),
        f"control: identical reduce-window on a parameter, [{_b},{_m}]",
    )
