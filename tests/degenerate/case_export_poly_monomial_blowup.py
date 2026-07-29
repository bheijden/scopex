"""SYNTHESISED (gap 8): `jax.export` shape polymorphism -- compile cost that is EXPONENTIAL in the
number of MONOMIALS of a symbolic dimension, paid entirely in pure Python before any lowering runs.

No upstream issue.  Constructed from the source: `jax/_src/export/shape_poly.py` represents a
symbolic dimension as a sorted list of (monomial, coefficient) pairs, and every arithmetic
operation on two such expressions goes through `_DimExpr._linear_combination_sorted_pairs`, which
walks both operand lists and calls `_DimExpr._syntactic_cmp` per pair.  On top of that,
`jax/_src/export/shape_poly_decision.py` runs a decision procedure (`_DecisionByElimination`) that
must bound each symbolic dimension; `combine_term_with_existing` builds ONE derived combination PER
TERM of the expression being bounded, and each of those combinations is itself a full symbolic
expression that gets `__sub__`'d and re-sorted.  So bounding an expression of T monomials costs
O(T) combinations x O(T) work each = O(T^2), and T itself can be exponential in the number of
dimension variables.

MECHANISM, precisely.  Build a symbolic dimension as a PRODUCT OF SUMS:

    d = (v0 + 1) * (v1 + 1) * ... * (v_{K-1} + 1)          -- expands to 2^K monomials

`_DimExpr.__mul__` distributes eagerly.  There is no factored representation and no lazy form: the
2^K-monomial normal form is materialised the moment the last multiply happens, and every later
touch of that dimension -- the dimension-variable solver, the shape-assertion generation, each
bounds query raised by `x[1:]` -- pays for all 2^K terms.

PLATFORM: **either**.  This is pure host Python in jax, above the jaxpr and above StableHLO.  It
runs identically on CPU and GPU because no backend is involved at all.  Measured on CPU here.

WHAT THE CONTROL ISOLATES -- exactly one thing: the number of MONOMIALS.

    pathological   d = (v0 + 1) * (v1 + 1) * ... * (v_{K-1} + 1)      2^K monomials
    control        d = v0       * v1       * ... * v_{K-1}            1  monomial

Both arms call `export.symbolic_shape` with the SAME K variable names, perform the SAME K-1 Python
multiplications, build the SAME two `ShapeDtypeStruct`s, and export the SAME inner function.  The
only difference is the `+ 1` on each factor, which is what turns one monomial of degree K into 2^K
monomials of degree <= K.  Total polynomial DEGREE is K in both arms, so degree is ruled out; only
term count differs.

MEASURED IN-ENV (JAX_PLATFORMS=cpu, jax 0.10.2, x64 on, one fresh process per point).  Seconds
inside `export.export(...)`, i.e. the number this file puts into the harness's `lower_s`:

    K    monomials      pathological       control      ratio
    3          8           0.453 s         0.591 s       0.8x     <- below the noise floor
    4         16           0.776 s         0.894 s       0.9x     <- still nothing
    5         32           0.758 s         0.596 s       1.3x
    6         64           2.492 s         0.586 s       4.3x
    7        128           9.113 s         0.649 s      14.0x
    8        256          38.555 s         0.894 s      43.1x
    9        512         255.745 s        (~0.6 s)     ~426x
   10       1024        RecursionError    (~0.6 s)       --

Baseline-subtracted excess: 2.0 / 8.6 / 38.1 / 255.2 s for 64 / 128 / 256 / 512 monomials -- a
factor of ~4.3, ~4.4, ~6.7 per DOUBLING of the term count.  That is O(T^2) tending to worse, on top
of T = 2^K, i.e. O(4^K) in the variable count.  The control is FLAT at ~0.6-0.9 s across the entire
range, so the ratio is the whole story and no drift correction is needed to see it.

THE K=10 ENTRY IS NOT A TIMEOUT, IT IS A CRASH.  At 1024 monomials the decision procedure blows the
Python recursion limit outright:

    File ".../jax/_src/export/shape_poly_decision.py", line 209, in combine_term_with_existing
      acc.append((Comparator.GEQ, _DimExpr(((t, 1),), scope) - int(t_lb), ...
    File ".../jax/_src/export/shape_poly.py", line 610, in _linear_combination_sorted_pairs
    File ".../jax/_src/export/shape_poly.py", line 332, in _syntactic_cmp
    RecursionError: maximum recursion depth exceeded

so K=10 is deliberately NOT exposed as a case -- it is a hard failure, not a slow compile, and the
harness would record it as an ERROR rather than as a measurement.  It is recorded here because the
crash is itself the sharpest available evidence for the mechanism: the recursion is inside the
term-comparison of the decision procedure, exactly where this docstring claims the cost lives.

WHY THIS IS THE PROFILER TEST IT IS.  **The two arms produce a byte-identical jaxpr, a
byte-identical StableHLO module and a byte-identical optimised HLO module.**  `fn` returns
`jnp.sum(x * 2.0)` on a 4-element array in both arms -- 2 jaxpr equations, ~15 lines of HLO.  Every
instruction-counting, HLO-diffing, pass-timing or XLA-side instrument sees two identical programs
and cannot account for a 43x wall-clock difference.  The entire gap sits in Python, in
`shape_poly.py` / `shape_poly_decision.py`, during tracing.  A profiler that only instruments
`backend_compile` reports 0.0 s of difference here; one that instruments tracing but attributes by
jaxpr equation reports "2 equations, nothing to see".  The correct answer names a Python module
that never appears in any HLO.

HARNESS SHAPE, AND WHY IT LOOKS ODD.  The harness measures `jax.jit(fn).lower(*args)` then
`.compile()`.  The polymorphic export is not on that path, so `fn` performs the export INSIDE
itself, at trace time, and discards the result.  This is not a trick to inflate a number: it is
where the cost genuinely falls for anyone using `jax.export` / `jax2tf` with polymorphic shapes --
their `export()` call is the slow line -- and nesting it under the outer trace is the only way to
put a pure-Python pre-lowering cost in front of an instrument that starts at `lower()`.  The
consequence to expect in the results table:

    THE HARNESS WILL PRINT "no (below floor)" FOR EVERY ARM OF THIS FILE.

`classify()` gates on `compile_s >= MIN_COMPILE_S`, and `compile_s` here is ~0.02 s in both arms
because the compiled program is `sum(x * 2)`.  The 43x lives in `lower_s`, which the harness
records but does not classify on.  Read `lower_s`.  That is the point of the case, not a defect in
it -- the same blind spot `case_lowering_arity_pytree.py` documents from the other direction.

NO CONSTRAINTS ARE USED HERE, deliberately.  `export.symbolic_shape` is called with no
`constraints=` argument in both arms, so the `_DecisionByElimination` constraint set is empty and
cannot be confused for the driver.  The sibling file
`case_export_poly_constraint_chain.py` holds the dimension trivial and varies the constraint
structure instead; between the two, term count and constraint structure are separated.

WHY THE EXTRA `solver` ARGUMENT EXISTS.  jax refuses to export a signature whose dimension
variables it cannot solve for: `v0*v1*v2*v3` alone raises "Cannot solve for values of dimension
variables ... We can only solve linear uni-variate constraints".  So the inner function takes a
rank-K argument of shape `(v0, ..., v_{K-1})`, from which every variable is solvable, plus the
rank-1 argument of shape `(d,)` that carries the polynomial.  The solver argument is IDENTICAL in
both arms; only `d` differs.

SIZES.  K in (5, 6, 7, 8, 9) -> 32, 64, 128, 256, 512 monomials.  Five points spanning 0.8x to
~426x is enough to fit the exponent rather than assert it.  The K=9 arm takes ~256 s per
measurement and is the expensive one in the file; it is kept because it is the point that
distinguishes O(T^2) from O(T^3) at the top of the range, and it is still well inside the harness's
900 s per-case timeout.

MEMORY is a non-issue: the largest concrete array in the file is 4 float32 values.  The cost is
Python objects -- a few hundred thousand small tuples -- not device memory.
"""

from __future__ import annotations

import functools

import numpy as np

import jax
import jax.numpy as jnp
from jax import export

# The concrete array the OUTER jitted function actually operates on.  Four elements, identical in
# every arm, so the emitted program is identical in every arm.  numpy at module scope: importing
# this file claims no device.
_X = np.arange(4, dtype=np.float32)


def _inner(solver, x):
    """The polymorphic function being exported.  Identical in both arms.

    `x[1:]` is deliberate: slicing a symbolically-sized dimension forces the decision procedure to
    prove `d >= 1`, which is the bounds query that walks all 2^K monomials.  Without a shape query
    of some kind the solver has less to chew on.
    """
    return jnp.sum(x[1:]) + jnp.sum(solver)


def _build_symbolic_dim(nvars: int, expand: bool):
    """Return (vars, d).  `expand` is the ONLY difference between the two arms.

    expand=True   d = (v0 + 1) * ... * (v_{K-1} + 1)   -> 2^K monomials, degree K
    expand=False  d = v0       * ... * v_{K-1}         -> 1   monomial,  degree K
    """
    vs = export.symbolic_shape(",".join(f"v{i}" for i in range(nvars)))
    d = (vs[0] + 1) if expand else vs[0]
    for v in vs[1:]:
        d = d * ((v + 1) if expand else v)
    return vs, d


def _fn(x, nvars: int, expand: bool):
    """Traced by the harness.  The export happens here, in Python, and its result is discarded.

    Everything after the `export.export` line is identical in both arms, so the jaxpr, the
    StableHLO and the optimised HLO are identical in both arms.
    """
    vs, d = _build_symbolic_dim(nvars, expand)
    export.export(jax.jit(_inner))(
        jax.ShapeDtypeStruct(tuple(vs), np.float32),   # makes every v_i solvable
        jax.ShapeDtypeStruct((d,), np.float32),        # carries the polynomial
    )
    return jnp.sum(x * 2.0)


NVARS = (5, 6, 7, 8, 9)          # -> 32, 64, 128, 256, 512 monomials

CASES = {}

for _k in NVARS:
    _terms = 2 ** _k
    CASES[f"exportpoly_monomials_{_terms}"] = (
        functools.partial(_fn, nvars=_k, expand=True), (_X,),
        f"synthesised gap 8: jax.export symbolic dim (v0+1)*...*(v{_k - 1}+1) = {_terms} monomials; "
        f"cost is O(monomials^2) inside shape_poly's decision procedure, all in Python at TRACE "
        f"time -- read lower_s, not compile_s (compile_s is ~0.02 s and the harness will say "
        f"'below floor')",
    )
    CASES[f"exportpoly_monomials_{_terms}_control"] = (
        functools.partial(_fn, nvars=_k, expand=False), (_X,),
        f"control: same {_k} dimension variables, same {_k - 1} multiplications, same degree {_k}, "
        f"same exported signature -- but d = v0*...*v{_k - 1} is ONE monomial instead of {_terms}. "
        f"Only the '+ 1' on each factor differs; jaxpr and HLO are byte-identical to the case arm",
    )
