"""diffrax#606 -- ``jnp.stack([y] * N)`` on a never-taken branch: ONE instruction with N operands.

    https://github.com/patrick-kidger/diffrax/issues/606      (fixed in diffrax 0.7.2 by PR #618)

REPORTED.  A diffrax user found compile time exploding with the number of save points.  Patrick
Kidger bisected it to a single block: a Python list repetition, ``jnp.stack([y] * N)``, sitting
inside a doubly-nested ``lax.cond`` on a branch that is essentially never taken.  Reporter's CPU
numbers: ~1.9 s at N=3000 save points, ~35 s at N=30000.  The fix was to replace the stack with a
``dynamic_update_slice``, which is O(1) in HLO.

WHY THIS IS NOT JUST ANOTHER BIG PROGRAM.  Two properties separate it from every reproduced case
in the corpus.

  (1) THE BLOWUP IS OPERAND ARITY, NOT INSTRUCTION COUNT.  The scatter-chain case, the unrolled-LU
      case and the gather-chain case all make the graph BIGGER -- more nodes, more edges, more
      passes to run over more IR.  This one leaves the graph at FOUR jaxpr equations and makes a
      single instruction enormous.  Verified at trace time on this machine (CPU, jax 0.10.2, x64,
      no execution):

          N        eqns  jaxpr chars   make_jaxpr
          1000       4        2 872      0.075 s
          3000       4        7 698      0.126 s
         10000       4       24 585      0.474 s
          control    4          498      0.008 s    at every N

      jax 0.10.2 keeps this as one ``stack[axis=0] h h h h ...`` equation, which lowers to one
      concatenate with N operands.  So an attributor that counts equations, or that ranks
      instructions by name, sees a four-line jaxpr for a program that (reportedly) takes 35 s to
      compile.  The size is in the OPERAND LIST of one node, which most IR summaries never print.

  (2) THE COST IS ON DEAD CODE.  The arms are called with ``p = -1.0``, so the predicate is false
      and the expensive branch NEVER EXECUTES.  A runtime profile of this program is empty.  Any
      attribution keyed on executed work, on kernel time, or on a trace collected from a real run
      is structurally incapable of pointing at the cause.  Only a compile-side instrument can.

WHAT THE CONTROL ISOLATES.  ``jnp.stack([v] * N)`` becomes ``jnp.broadcast_to(v, (N,) + v.shape)``.
These are SEMANTICALLY IDENTICAL -- both produce N copies of ``v`` stacked along a new leading
axis, elementwise equal, same shape, same dtype.  Same cond, same predicate, same false branch,
same trailing ``.sum()``, same arguments.  The only difference is that one spells it as an
N-operand concatenate and the other as a single broadcast_in_dim.  Everything that differs between
the arms is operand arity and nothing else.

WHY THE LIBRARY-FREE FORM.  The in-diffrax reproduction is version-locked (present 0.6.1-0.7.0,
fixed 0.7.2) and would pin an install into the corpus.  This distillation is version-independent
and depends on nothing but jax, at the cost of losing the ecosystem provenance -- which the corpus
does not need, since it is testing scopex and not diffrax.

PLATFORM.  The reporter's numbers are CPU.  Concatenate simplification is a platform-independent
HLO pass, so it should appear on both backends, but the GPU backend also has to emit a kernel for
an N-way concatenate and may take a different (possibly worse) route.  Run both; a divergence here
is interesting rather than disappointing.

SIZES.  N = 3000, 10000, 30000 is the measured-in-the-issue ladder.  ``stackcond_n100000`` is a
STRETCH ARM: if the cost is superlinear in N it may exceed the 900 s harness timeout, and if it
does, run the ladder without it.  It is included because the shape of the curve between 30k and
100k is exactly what distinguishes "linear in operands" from "quadratic in operands", and that
distinction is the mechanism.

UNCERTAINTY.  Not yet measured on this machine at compile time -- the GPU was held by another
tenant.  The trace-time numbers above are real; the compile-time numbers are the reporter's, from
a different jax version on CPU.  XLA has had concatenate-simplification work since, so if the 30k
arm lands below the 3 s floor, that is a genuine "fixed upstream" result and the 100k arm is the
place to look next.
"""

from __future__ import annotations

import functools

import numpy as np

import jax.numpy as jnp
from jax import lax

_WIDTH = 4          # the per-save-point state, as in the issue (a small ODE state vector)


def _slow(y, p, n: int):
    """N-operand concatenate, built by Python list repetition, on the untaken branch."""
    return lax.cond(
        p > 0,
        lambda v: jnp.stack([v] * n),                              # <- ONE instruction, N operands
        lambda v: jnp.zeros((n,) + v.shape, v.dtype),
        y,
    ).sum()


def _fast(y, p, n: int):
    """Semantically identical, spelled as one broadcast_in_dim: O(1) HLO."""
    return lax.cond(
        p > 0,
        lambda v: jnp.broadcast_to(v, (n,) + v.shape),             # <- ONE instruction, 1 operand
        lambda v: jnp.zeros((n,) + v.shape, v.dtype),
        y,
    ).sum()


_rng = np.random.default_rng(606)
# p < 0 so the predicate is FALSE and the expensive branch is dead at runtime. That is the point:
# the compile cost is paid for code that never runs.
_ARGS = (_rng.standard_normal(_WIDTH), np.float64(-1.0))


def _mk(fn, n: int, note: str):
    return functools.partial(fn, n=n), _ARGS, note


CASES: dict = {}

for _n in (3000, 10000, 30000):
    CASES[f"stackcond_n{_n}"] = _mk(
        _slow, _n,
        f"diffrax#606: jnp.stack([y]*{_n}) inside an untaken lax.cond -> one concatenate with "
        f"{_n} operands; 4 jaxpr eqns total, so IR-size attribution sees nothing.",
    )
    CASES[f"stackcond_n{_n}_control"] = _mk(
        _fast, _n,
        f"control: jnp.broadcast_to(y, ({_n},4)) -- elementwise-identical result, same cond, same "
        f"args, one operand instead of {_n}.",
    )

# Stretch rung: tells linear-in-operands apart from quadratic-in-operands. May exceed the timeout.
CASES["stackcond_n100000"] = _mk(
    _slow, 100_000,
    "stretch arm: 100k-operand concatenate. Included to shape the N-curve; may exceed the 900 s "
    "harness timeout, in which case drop it and report the 3k/10k/30k ladder.",
)
CASES["stackcond_n100000_control"] = _mk(
    _fast, 100_000,
    "control for the stretch arm: same (100000,4) result from one broadcast_in_dim.",
)
