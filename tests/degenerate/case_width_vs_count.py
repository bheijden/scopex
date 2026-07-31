"""SYNTHESISED. Does compile cost track the NUMBER of equations, or the WIDTH of the arrays?

Every compile-time win measured on the dflux examples moved both at once. Collapsing a per-species
python comprehension into one vector expression took `psat_vec` from 145 equations of mean width 1.1
to 13 equations of mean width 11.4 -- and bought a 0.763-0.869 compile ratio. But the total
arithmetic barely moved (160 output elements -> 148), and separately, DELETING 785 already-wide
duplicate equations bit-identically bought nothing (0.972).

So two explanations fit every observation so far and they have not been separated:

    (count)  cost is per-equation, roughly independent of shape -- 145 tiny instructions cost more
             to analyse than 13 large ones, so collapsing wins and the arithmetic is irrelevant
    (width)  cost tracks total elements, and the wins came from somewhere else entirely

This file separates them with a 2x2. Each axis moves ONE variable and pins the other.

    AXIS A -- same total elements, different equation count
        narrow_N:  N equations of width 1      (a python loop over lanes)
        wide_N:    ~2 equations of width N     (one vector expression)
      Both compute exactly N exps and N sins. Only the packaging differs.

    AXIS B -- same equation count, different width
        thin_K:    K equations of width 8
        fat_K:     K equations of width 8192
      Both emit K tanh/mul pairs. Only the array size differs.

PREDICTIONS, stated before measuring so the result can falsify one:
    if COUNT drives it   -> axis A is a large effect, axis B is flat
    if WIDTH drives it   -> axis A is flat, axis B is large
    if BOTH              -> both move, and their ratio says which dominates

MEASURED (3 rounds, arms rotated, paired within round; jax 0.10.2):

    axis          what varies              what is held        CPU        GPU
    A (count)     4 -> 7,170 equations     same arithmetic     59.1x       --
    B (width)     8 -> 8,188 elements      2,049 equations      3.4x     0.52x

BOTH PREDICTIONS WERE WRONG, and the interesting half is axis B. Count dominates by ~17x, so it is
the lever. But width is NOT flat, and on GPU IT CHANGES SIGN: a 1,024x wider array compiles 3.4x
SLOWER on CPU and 1.9x FASTER on GPU, at an identical equation count. Both sizes agree (the 256 pair
gives 2.6x CPU / 0.46x GPU), so it is not a single-point artifact.

Consequences, both of which cost this project a wrong conclusion first:
  * "cost tracks elements-per-equation" (r = 0.812 over 7 real programs) was measuring a PROXY --
    in model code, narrow programs also have many equations, and only a synthetic can separate them.
  * any advice of the form "prefer wide arrays" is right on GPU and wrong by 3.4x on CPU. The
    statement that survives on both backends is about COUNT, not width.

The `_control` in each pair is the CHEAP arm by the count hypothesis, so a ratio > 1 means the
count hypothesis holds.

PLATFORM: both. The mechanism (HLO pass iteration vs element count) is not backend-specific, but
XLA:GPU adds autotuning that XLA:CPU does not, so run both and report separately.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

_X8 = np.linspace(0.1, 0.9, 8)
_X8K = np.linspace(0.1, 0.9, 8192)


# ── AXIS A: same arithmetic, N narrow equations versus ~2 wide ones ─────────────────────────────
def _narrow(x, n):
    """N separate scalar chains, then one stack. Mean output width 1."""
    return jnp.stack([jnp.exp(x[i]) * jnp.sin(x[i]) for i in range(n)]).sum()


def _wide(x, n):
    """The identical arithmetic as one vector expression. Two equations of width N."""
    return (jnp.exp(x[:n]) * jnp.sin(x[:n])).sum()


# ── AXIS B: same equation count, width 8 versus width 8192 ─────────────────────────────────────
def _chain(x, k):
    """K tanh/mul pairs. Equation count is K regardless of how wide x is."""
    for _ in range(k):
        x = jnp.tanh(x) * 1.0001
    return x.sum()


def _mk_a(n, wide):
    fn = (lambda x, n=n: _wide(x, n)) if wide else (lambda x, n=n: _narrow(x, n))
    x = np.linspace(0.1, 0.9, max(n, 8))
    return fn, (x,), f"axis A n={n} {'wide (~2 eqns)' if wide else 'narrow (N eqns)'}"


def _mk_b(k, fat):
    fn = lambda x, k=k: _chain(x, k)                                          # noqa: E731
    return fn, (_X8K if fat else _X8,), f"axis B k={k} width={'8192' if fat else '8'}"


CASES = {}
# A: the narrow arm is the CASE, the wide arm its control -- ratio > 1 supports `count`
for _n in (64, 256, 1024):
    CASES[f"narrow_{_n}"] = _mk_a(_n, wide=False)
    CASES[f"narrow_{_n}_control"] = _mk_a(_n, wide=True)
# B: the fat arm is the CASE, the thin arm its control -- ratio > 1 supports `width`
for _k in (64, 256, 1024):
    CASES[f"fat_{_k}"] = _mk_b(_k, fat=True)
    CASES[f"fat_{_k}_control"] = _mk_b(_k, fat=False)
