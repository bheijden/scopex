"""optax#1498 -- does CRITICAL-PATH DEPTH cost compile time when op count, op kind, dtype and total
bytes are all held fixed?

    https://github.com/google-deepmind/optax/issues/1498

Reported: swapping ``jax.tree.reduce(operator.add, tree)`` for ``jax.tree.reduce_associative(
operator.add, tree)`` inside optax took a model's compile time from ~23 s to ~18 s. No leaf count,
no model, no hardware, no jax version in the thread, and rdyro's reply pushes back explicitly --
nobody confirmed with an XLA engineer that DEPTH is what moved. So the reported effect is 1.28x
against a corpus bar of 10x, and this file exists to CONSTRUCT the experiment the issue only
gestured at, at sizes where the effect can actually show up.

WHY THE VARIABLE IS WORTH A SLOT. ``jax.tree.reduce`` is ``functools.reduce``: N-1 adds in a
LEFT FOLD, so the dependency chain is N-1 long. ``jax.tree.reduce_associative`` emits the SAME N-1
adds as a balanced binary tree, depth ceil(log2 N). Same primitive, same operand shapes, same
dtype, same FLOPs, same number of jaxpr equations, same number of HLO instructions -- only the
shape of the DAG differs. At N=50000 that is depth 49999 against depth 16, a 3000x swing in
critical path at a byte-for-byte identical op multiset. Nothing else in the corpus isolates depth:
the flagship scatter chain varies depth, the scatter primitive AND the trailing reduction together.

If compile time tracks the critical path, that is a first-class attribution fact -- it means a
profiler cannot explain a compile from op counts alone. If it does NOT, that is equally worth
owning: "depth at fixed op count does not move compile time" retires a plausible hypothesis, and
the corpus should contain the negative rather than leave it as folklore. **This case is expected to
be a NEGATIVE and is written to be trustworthy as one.**

WHAT THE CONTROL ISOLATES. The two arms differ in exactly one call: ``jax.tree.reduce`` versus
``jax.tree.reduce_associative``, over the identical pytree of the identical numpy leaves. There is
no size difference to explain away, so a null result here is a clean null and a positive result is
attributable to graph shape and nothing else.

MEASURED CONFOUNDS, DELIBERATELY MINIMISED
--------------------------------------------------------------------------------------------------
Leaves are passed as a pytree of N separate arrays, not sliced out of one big array inside the
function. Slicing would add 2N indexing ops to BOTH arms and bury the N-1 adds under ops that have
nothing to do with the hypothesis. The cost of this choice is that the HLO gets N parameters, which
is itself superlinear-ish work in some XLA passes -- but it is identical in both arms, so it
inflates both compile times and only ever makes the ratio look SMALLER. A positive result survives
it; a null result could in principle be a positive drowned in parameter-handling cost, which is why
the sweep runs to N=50000 where the depth difference is at its most extreme.

The chain arm's runtime is not comparable to a real workload either: N tiny device arrays cost more
in dispatch than the adds cost in FLOPs, so ``compile/runtime`` here says nothing. That is fine --
the harness ranks the control comparison above the ratio precisely for cases like this one.

VERIFIED AT TRACE TIME (CPU, jax 0.10.2, tracing only, no device work), via
``dependency_depth()`` below, which reads the longest path off the actual jaxpr instead of trusting
this docstring:

    N        arm                      eqns    depth        control eqns   control depth
    1000     treereduce_chain_n1000     999      999                999              10
    10000    treereduce_chain_n10000   9999     9999               9999              14
    50000    treereduce_chain_n50000  49999    49999              49999              16
    10000    treereduce_chain_vec10000 9999     9999               9999              14

i.e. op count identical to the equation, depth differing by 100x, 700x and 3100x. The control
variable is real and it is the only one moving. Tracing the 50k arms costs ~11 s each on CPU before
XLA sees anything, which is worth knowing when reading `lower_s` in the results table -- and it is
the same ~11 s in both arms, so it cannot manufacture a difference.

UNCERTAIN, SAID PLAINLY. (1) The reported gap is 1.28x on an unknown program; this reconstruction
may show nothing at all at any N. (2) The N=50000 arms produce a ~50k instruction module in BOTH
arms and may be slow to compile everywhere, which would be a size effect and not a depth effect --
read the pair, never the single number. (3) The brief that produced this file flags jax#4667 as a
verified, stronger case in the same neighbourhood; if slots are scarce, run that first and treat
this file as the controlled follow-up that says whether DEPTH is the operative variable there.
"""

from __future__ import annotations

import functools
import operator

import jax
import numpy as np

# Plain numpy at module scope: importing this file to discover CASES must never claim a device.
_rng = np.random.default_rng(1498)


def _scalar_tree(n: int) -> tuple[np.ndarray, ...]:
    """N separate float64 scalars -- one pytree, N leaves, N HLO parameters, zero extra ops."""
    return tuple(np.float64(v) for v in _rng.standard_normal(n))


def _vector_tree(n: int, width: int) -> tuple[np.ndarray, ...]:
    """N separate (width,) float64 leaves, so the adds are real vector ops rather than scalars."""
    return tuple(_rng.standard_normal(width) for _ in range(n))


# ---------------------------------------------------------------------------------------------
# The two arms. One call apart.
# ---------------------------------------------------------------------------------------------

def _fold_sequential(tree):
    """jax.tree.reduce == functools.reduce: N-1 adds, dependency depth N-1."""
    return jax.tree.reduce(operator.add, tree)


def _fold_associative(tree):
    """jax.tree.reduce_associative: the SAME N-1 adds, dependency depth ceil(log2 N)."""
    return jax.tree.reduce_associative(operator.add, tree)


# Backups that do not depend on the jax version exposing `reduce_associative`. They are not used as
# CASES entries -- `jax.tree.reduce_associative` exists in jax 0.10.2 and using the real API keeps
# the case faithful to the issue -- but they document what the two arms mean in plain terms, and
# `dependency_depth` can be pointed at them if the API ever moves.
def _manual_chain(leaves):
    return functools.reduce(operator.add, leaves)


def _manual_tree(leaves):
    xs = list(leaves)
    while len(xs) > 1:
        it = iter(xs)
        xs = [a + b for a, b in zip(it, it)] + ([xs[-1]] if len(xs) % 2 else [])
    return xs[0]


# ---------------------------------------------------------------------------------------------

_SCALAR_SIZES = (1_000, 10_000, 50_000)
_TREES: dict[int, tuple[np.ndarray, ...]] = {n: _scalar_tree(n) for n in _SCALAR_SIZES}
_VEC_TREE = _vector_tree(10_000, 16)

CASES: dict[str, tuple] = {}

for _n in _SCALAR_SIZES:
    _t = _TREES[_n]
    CASES[f"treereduce_chain_n{_n}"] = (
        _fold_sequential, (_t,),
        f"optax#1498: jax.tree.reduce over {_n} scalar leaves -- {_n - 1} adds, dependency depth "
        f"{_n - 1} (expected NEGATIVE; the issue reports only 1.28x)",
    )
    CASES[f"treereduce_chain_n{_n}_control"] = (
        _fold_associative, (_t,),
        f"control: jax.tree.reduce_associative over the SAME {_n} leaves -- same {_n - 1} adds, "
        f"same dtype, same bytes, depth ~{_n.bit_length()}",
    )

# One non-scalar pair, in case XLA handles a chain of rank-0 adds by a path that never sees the
# dependency structure (scalar folding, constant-ish treatment). Same experiment, leaves of width 16.
CASES["treereduce_chain_vec10000"] = (
    _fold_sequential, (_VEC_TREE,),
    "optax#1498 with (16,)-shaped leaves: 9999 vector adds, depth 9999 -- guards against the "
    "scalar-only arms being special-cased",
)
CASES["treereduce_chain_vec10000_control"] = (
    _fold_associative, (_VEC_TREE,),
    "control: same 9999 vector adds over the same leaves, depth 14",
)


def dependency_depth(name: str) -> tuple[int, int]:
    """``(n_eqns, longest path)`` for an arm, read off its jaxpr. The control variable, MEASURED.

    Tracing only -- no compilation, no device. Expect equal ``n_eqns`` and wildly different depth
    between an arm and its ``_control``; if the equation counts ever differ, the pair has stopped
    being a controlled experiment and the results table must not be read as one.
    """
    fn, args, _ = CASES[name]
    jaxpr = jax.make_jaxpr(fn)(*args).jaxpr
    depth: dict = {}
    best = 0
    for eqn in jaxpr.eqns:
        d = 1 + max((depth.get(v, 0) for v in eqn.invars if hasattr(v, "aval")), default=0)
        for v in eqn.outvars:
            depth[v] = d
        best = max(best, d)
    return len(jaxpr.eqns), best
