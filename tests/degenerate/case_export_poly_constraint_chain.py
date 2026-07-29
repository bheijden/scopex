"""SYNTHESISED (gap 8): `jax.export` symbolic-shape CONSTRAINTS -- compile cost exponential in the
DEPTH of a transitive `>=` chain, at a constraint COUNT the control matches exactly.

No upstream issue.  Constructed from the source.  `jax/_src/export/shape_poly_decision.py`
implements `_DecisionByElimination`: the user's constraints are turned into a set of "combinations",
and `combine_term_with_existing` pairs each new constraint against every combination already
derived, emitting the eliminated result as a further combination.  A set of constraints that
COMPOSE -- `v0 >= v1`, `v1 >= v2`, ..., `v_{K-2} >= v_{K-1}`, `v_{K-1} >= 2` -- lets every derived
combination pair with every other, so the derived set grows multiplicatively in the chain depth.
Constraints that do NOT compose derive nothing new and the set stays at its initial size.

This is the SIBLING of `case_export_poly_monomial_blowup.py` and the two are deliberately
disjoint.  That file uses ZERO constraints and varies the number of monomials in the dimension;
this one uses a K-monomial dimension in EVERY arm and varies only the constraint structure.  Term
count and constraint structure are therefore separated, and a profiler that collapses both into
"shape polymorphism is slow" has not localised either.

WHAT THE CONTROL ISOLATES -- exactly one thing: the RIGHT-HAND SIDE of K-1 constraint strings.

    pathological  ["v0 >= v1", "v1 >= v2", ..., "v_{K-2} >= v_{K-1}", "v_{K-1} >= 2"]
    control       ["v0 >= 2",  "v1 >= 2",  ..., "v_{K-2} >= 2",       "v_{K-1} >= 2"]

Both lists have EXACTLY K entries.  Both mention exactly the same K variables.  Both are satisfiable
and non-redundant.  Both are passed to the same `export.symbolic_shape` call, followed by the same
`export.export` of the same inner function over the same two `ShapeDtypeStruct`s, with the same
symbolic dimension `d = v0 + v1 + ... + v_{K-1}`.  The only difference is that in the case arm the
right-hand side of K-1 constraints is another VARIABLE (so they chain) and in the control arm it is
a CONSTANT (so they do not).

MEASURED IN-ENV (JAX_PLATFORMS=cpu, jax 0.10.2, x64 on, one fresh process per point).  Seconds
inside `export.export(...)`, i.e. what this file puts into the harness's `lower_s`:

    K    constraints   case (chain)   control (flat)   ratio
     6         6          0.598 s        ~0.60 s        1.0x
     7         7          0.630 s        ~0.60 s        1.0x
     8         8          0.921 s         0.731 s       1.3x
     9         9          4.423 s         0.872 s       5.1x
    10        10         40.838 s         0.620 s      65.9x

The case arm multiplies by ~4.8x then ~9.2x for each +1 in K while the control is FLAT at
~0.6-0.9 s across the whole range and across K=11 as well (0.600 s).  Extrapolating the last ratio
puts K=11 near 400 s; it is deliberately not exposed, since 40 s at K=10 already pins the exponent
and the suite has to stay runnable.

TWO NEGATIVE CONTROLS, both of which matter because they kill the obvious wrong explanation.

  (a) CONSTRAINT COUNT IS NOT THE DRIVER.  A DENSE constraint set -- every ordered pair
      `v_i >= v_j`, i.e. K(K-1)/2 + 1 constraints -- is barely worse than the K-constraint chain:

          K     dense (count)      dense time     chain (K constraints)
           6      16                 0.613 s          0.598 s
           7      22                 0.751 s          0.630 s
           8      29                 1.299 s          0.921 s
           9      37                 5.309 s          4.423 s
          10      46                45.309 s         40.838 s

      At K=10 the dense arm carries 4.6x the constraints for 1.11x the time.  Anything that ranks
      by "number of constraints" is reading the wrong number; the dense set is expensive because it
      CONTAINS the chain, not because it is large.

  (b) THE CHAIN, NOT THE RELATIONAL FORM.  A `pairs` set of the same size K, made of DISJOINT
      2-element chains (`v0 >= v1`, `v2 >= v3`, ... plus constant bounds on the odd variables), is
      relational in exactly the same way but does not compose past depth 2:

          K=8   0.969 s      K=9   0.601 s      K=10  0.779 s      K=11  1.351 s

      Flat, like the constant-bound control.  So it is neither "constraints exist", nor "constraints
      count", nor "constraints relate two variables" -- it is specifically the DEPTH of transitive
      composition.  Both negative controls are exposed as cases so the claim can be re-checked
      rather than taken on trust.

PLATFORM: **either**.  Pure host Python in `jax/_src/export/shape_poly_decision.py`, above the
jaxpr and above StableHLO, with no backend involved.  Measured on CPU; a GPU run would measure the
same Python.

WHY THIS IS A PROFILER TEST.  As with its sibling, **all arms produce a byte-identical jaxpr, a
byte-identical StableHLO module and a byte-identical optimised HLO module** -- `fn` returns
`jnp.sum(x * 2.0)` over a 4-element array, 2 equations, ~15 lines of HLO.  A 66x wall-clock gap
with zero difference in every artefact the compiler produces.  The distinguishing signal is a
Python-level profile, and the honest answer names one file in jax that emits no HLO at all.

HARNESS SHAPE.  The harness measures `jax.jit(fn).lower(*args)` then `.compile()`; the polymorphic
export is not on that path, so `fn` performs it inside itself at trace time and discards the result.
That is where the cost genuinely falls for `jax.export` / `jax2tf` users -- their `export()` call is
the slow line.  Consequence, stated up front:

    THE HARNESS WILL PRINT "no (below floor)" FOR EVERY ARM OF THIS FILE.

`classify()` gates on `compile_s`, which is ~0.02 s here in every arm.  The 66x is in `lower_s`,
which the harness records but does not classify on.  Read `lower_s`.

WHY THE EXTRA `solver` ARGUMENT EXISTS.  jax refuses a signature whose dimension variables it
cannot solve for.  The inner function takes a rank-K argument of shape `(v0, ..., v_{K-1})`, from
which every variable is solvable, plus a rank-1 argument of shape `(d,)` with
`d = v0 + ... + v_{K-1}`.  Both arguments are IDENTICAL in every arm of this file; only the
`constraints=` list differs.

SIZES: K in (6, 7, 8, 9, 10).  Five points spanning 1.0x to 66x.  One point could not distinguish
"exponential in chain depth" from "this constraint set is big", which is exactly what negative
control (a) shows is the wrong reading.
"""

from __future__ import annotations

import functools

import numpy as np

import jax
import jax.numpy as jnp
from jax import export

# numpy at module scope, never jnp: importing this file claims no device.
_X = np.arange(4, dtype=np.float32)


def _constraints(nvars: int, mode: str) -> list[str]:
    """All four modes return a constraint list over the same `nvars` variables.

    chain  K   entries, transitively composing:  v0 >= v1 >= ... >= v_{K-1} >= 2
    flat   K   entries, none composing:          v_i >= 2 for every i          <- the control
    pairs  K   entries, composing only to depth 2 (negative control b)
    dense  K(K-1)/2 + 1 entries, contains the chain (negative control a)
    """
    if mode == "chain":
        return [f"v{i} >= v{i + 1}" for i in range(nvars - 1)] + [f"v{nvars - 1} >= 2"]
    if mode == "flat":
        return [f"v{i} >= 2" for i in range(nvars)]
    if mode == "pairs":
        return ([f"v{i} >= v{i + 1}" for i in range(0, nvars - 1, 2)]
                + [f"v{i} >= 2" for i in range(1, nvars, 2)])
    if mode == "dense":
        return ([f"v{i} >= v{j}" for i in range(nvars) for j in range(i + 1, nvars)]
                + [f"v{nvars - 1} >= 2"])
    raise ValueError(mode)


def _inner(solver, x):
    """The polymorphic function being exported.  Identical in every arm.

    `x[1:]` forces a bounds query on the symbolic dimension, which is what sends the decision
    procedure into the constraint set.
    """
    return jnp.sum(x[1:]) + jnp.sum(solver)


def _fn(x, nvars: int, mode: str):
    """Traced by the harness.  The export happens here, in Python, and its result is discarded.

    Everything after the `export.export` line is identical in every arm, so the jaxpr, the
    StableHLO and the optimised HLO are identical in every arm.
    """
    vs = export.symbolic_shape(",".join(f"v{i}" for i in range(nvars)),
                               constraints=_constraints(nvars, mode))
    d = vs[0]
    for v in vs[1:]:
        d = d + v
    export.export(jax.jit(_inner))(
        jax.ShapeDtypeStruct(tuple(vs), np.float32),   # makes every v_i solvable
        jax.ShapeDtypeStruct((d,), np.float32),        # d = v0 + ... + v_{K-1}
    )
    return jnp.sum(x * 2.0)


NVARS = (6, 7, 8, 9, 10)

CASES = {}

for _k in NVARS:
    CASES[f"exportpoly_chain_k{_k}"] = (
        functools.partial(_fn, nvars=_k, mode="chain"), (_X,),
        f"synthesised gap 8: {_k} symbolic-shape constraints forming ONE transitive chain "
        f"v0>=v1>=...>=v{_k - 1}>=2. _DecisionByElimination composes them pairwise, so cost is "
        f"exponential in chain depth ({_k}=10 measured at 40.8 s). All of it Python at TRACE time "
        f"-- read lower_s; compile_s is ~0.02 s and the harness will say 'below floor'",
    )
    CASES[f"exportpoly_chain_k{_k}_control"] = (
        functools.partial(_fn, nvars=_k, mode="flat"), (_X,),
        f"control: the SAME {_k} constraints over the SAME {_k} variables with the same symbolic "
        f"dimension -- only the right-hand side of {_k - 1} of them changes from a variable to the "
        f"constant 2, so nothing composes. Flat at ~0.6-0.9 s for every K",
    )

# ---- negative control (a): constraint COUNT is not the driver ---------------------------------
for _k in (8, 10):
    CASES[f"exportpoly_dense_k{_k}"] = (
        functools.partial(_fn, nvars=_k, mode="dense"), (_X,),
        f"negative control (a): all {_k * (_k - 1) // 2 + 1} ordered pairs v_i>=v_j instead of the "
        f"{_k}-entry chain -- {(_k * (_k - 1) // 2 + 1) / _k:.1f}x the constraints for ~1.1x the "
        f"time (45.3 s vs 40.8 s at K=10). Ranking by constraint count reads the wrong number",
    )
    CASES[f"exportpoly_dense_k{_k}_control"] = (
        functools.partial(_fn, nvars=_k, mode="flat"), (_X,),
        f"control: same {_k} variables, {_k} non-composing constant bounds",
    )

# ---- negative control (b): relational form alone is not enough, depth is ----------------------
for _k in (10,):
    CASES[f"exportpoly_pairs_k{_k}"] = (
        functools.partial(_fn, nvars=_k, mode="pairs"), (_X,),
        f"negative control (b): {_k} constraints that ARE relational but form disjoint 2-chains, so "
        f"composition stops at depth 2. Measured 0.78 s against the chain's 40.8 s at the same K "
        f"and the same count -- the driver is transitive DEPTH, not relational form",
    )
    CASES[f"exportpoly_pairs_k{_k}_control"] = (
        functools.partial(_fn, nvars=_k, mode="flat"), (_X,),
        f"control: same {_k} variables, {_k} non-composing constant bounds",
    )
