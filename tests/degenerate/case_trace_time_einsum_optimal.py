"""jax#2583 / jax PR#25214 -- ``jnp.einsum(optimize='optimal')`` burns 10 s in opt_einsum and 0 s in XLA.

    https://github.com/jax-ml/jax/pull/25214          (jakevdp, merged: default flipped to 'auto')
    https://github.com/jax-ml/jax/issues/2583         (the original "einsum is slow to trace" report)

REPORTED.  ``jnp.einsum`` calls ``opt_einsum.contract_path`` while TRACING, before a single line of
HLO exists.  The ``'optimal'`` solver is an exhaustive DFS over contraction orders and its cost
grows super-exponentially in the operand count.  jakevdp changed the JAX-wide default from
``'optimal'`` to ``'auto'`` in 0.5.0 for exactly this reason -- but ``'optimal'`` is still a
supported public value of the ``optimize`` kwarg, so on 0.10.2 the pathology is one keyword away
and fully live.  Confirmed in this environment (jax 0.10.2, opt_einsum 3.4.0, CPU, path search
only, no jax involved):

    nops   contract_path(optimize='auto')   contract_path(optimize='optimal')
      6              0.005 s                            0.007 s
      8              0.005 s                            0.405 s
     10              0.002 s                            8.517 s      = ~4000x
     11                  --                           62.419 s       (x7.3 for one more operand)
     12                  --                     extrapolates to 450-800 s; not measured

THIS IS THE CORPUS'S ADVERSARIAL CASE AND THE HARNESS WILL SCORE IT "no (below floor)".
That verdict is the finding, not a failure.  ``classify()`` reads ``compile_s`` only, and
``compile_s`` here is a rounding error: the module is a handful of 3x3 dot_generals.  The 10 s
lives in ``lower_s``, inside a third-party Python library, on a stack frame that contains no JAX
code at all.  Read the ``lower_s`` column, not the verdict column:

    an instrument hooked to ``backend_compile`` sees ~0.05 s and reports a healthy program,
    while first-call wall clock is ten seconds and climbing with operand count.

Every other trace-stage case in the corpus (jax#22385 nested-jit fib, jax#1172 random.split) burns
its time inside JAX's own tracer building a jaxpr.  This one burns it in ``opt_einsum.paths``.
If scopex attributes by jaxpr equation, HLO instruction, or XLA pass, it is blind here by
construction -- all three artifacts are tiny and are produced quickly.  Attribution has to come
from the Python stack during tracing, and it has to be willing to charge time to a non-JAX frame.

WHAT THE CONTROLS ISOLATE.  Two, and they isolate different halves of the claim.

  (1) ``_control`` -- ``optimize='auto'``.  ONE KEYWORD.  Same spec, same operands, same dtypes,
      same shapes, same jit.  For nops > 8 opt_einsum's 'auto' dispatches to the greedy solver, so
      the search becomes O(n^3)-ish instead of exhaustive.  Everything that differs between the two
      arms is which path-finding algorithm runs; the contraction that is finally emitted may differ
      slightly in ORDER but is the same amount of arithmetic on 3x3 operands.

  (2) ``_pathlit`` -- the literal optimal path, precomputed and pasted in as a constant, passed as
      ``optimize=[(0, 7), (0, 6), ...]``.  This is the sharper control: jax passes an explicit path
      list straight through to ``contract_path``, which skips the solver entirely and emits BYTE-
      IDENTICAL HLO to the 'optimal' arm.  If ``_pathlit`` traces instantly and the 'optimal' arm
      takes 10 s while both compile to the same module, the cost is provably the SEARCH and not the
      contraction, the operands, or anything XLA does.  ``_pathlit`` is not named ``_control`` on
      purpose -- the harness auto-pairs on that suffix and only one arm can hold the slot.

BY CONSTRUCTION THE RATIO IS ENORMOUS.  Every operand is 3x3, so the runtime is a few microseconds
no matter what the path is; nothing about the shapes contributes to the compile cost.  The program
is deliberately trivial in every dimension except operand COUNT.

PLATFORM.  Platform-independent -- the cost is pure CPU-side Python in a third-party library and
happens before the backend is consulted.  It should be identical on cpu and cuda, and that
invariance is itself worth recording: it is the only case in the corpus for which the two platform
columns are predicted to agree to within noise.

SIZES.  nops = 8, 10, 11 -- a measured ladder of 0.4 s / 8.5 s / 62 s of pure Python.  nops=12 is
deliberately omitted: the per-operand factor is 21x then 7.3x, which puts 12 somewhere between
450 s and 800 s, and one arm that eats the 900 s harness timeout costs more than the data point is
worth.  If the machine turns out to be much faster than the probe, 12 is the next rung.
"""

from __future__ import annotations

import functools

import numpy as np

from jax import numpy as jnp

_LETTERS = "abcdefgh"


def _spec(nops: int) -> str:
    """A cyclic contraction over 7 indices: every index is shared by several operands, which is
    what denies the solver any easy pruning and makes the search space genuinely factorial."""
    return ",".join(_LETTERS[i % 7] + _LETTERS[(i + 3) % 7] for i in range(nops)) + "->a"


# Precomputed on this machine with opt_einsum 3.4.0 by
#   opt_einsum.contract_path(_spec(n), *[np.zeros((3, 3), np.float32)] * n, optimize='optimal')[0]
# Pasted as literals ON PURPOSE: computing them at import would put an 8.5 s solver run inside
# `discover()`, which imports every case file in the corpus before measuring anything.
_OPTIMAL_PATHS: dict[int, list[tuple[int, ...]]] = {
    8: [(1, 5), (1, 4), (1, 5), (3, 4), (1, 3), (0, 2), (0, 1)],
    10: [(0, 7), (0, 6), (0, 5), (1, 4), (2, 4), (0, 4), (0, 3), (1, 2), (0, 1)],
    11: [(0, 7), (1, 7), (1, 6), (1, 5), (1, 4), (0, 2), (3, 4), (2, 3), (1, 2), (0, 1)],
}

_rng = np.random.default_rng(2583)


def _ops(nops: int) -> tuple[np.ndarray, ...]:
    """nops distinct 3x3 float32 operands.  144 bytes each -- the whole point is that the DATA is
    irrelevant and only the operand count matters."""
    return tuple(_rng.standard_normal((3, 3), dtype=np.float32) for _ in range(nops))


def _mk(nops: int, optimize, note: str):
    fn = functools.partial(jnp.einsum, _spec(nops), optimize=optimize)
    return fn, _ops(nops), note


CASES: dict = {}

for _n in (8, 10, 11):
    CASES[f"einsum_optimal_n{_n}"] = _mk(
        _n, "optimal",
        f"jax#2583: nops={_n}, optimize='optimal' -> exhaustive contraction-path DFS inside "
        f"opt_einsum during TRACING. Cost lands in lower_s; compile_s stays ~0 so the harness "
        f"verdict will read 'below floor' -- that IS the result.",
    )
    CASES[f"einsum_optimal_n{_n}_control"] = _mk(
        _n, "auto",
        f"control: nops={_n}, identical spec/operands/dtypes, optimize='auto' -> greedy solver. "
        f"One keyword is the only difference.",
    )

for _n, _path in _OPTIMAL_PATHS.items():
    CASES[f"einsum_optimal_n{_n}_pathlit"] = _mk(
        _n, _path,
        f"discriminator (not auto-paired): nops={_n} with the optimal path passed as a literal, so "
        f"the solver never runs but the HLO is identical to the 'optimal' arm -- isolates SEARCH "
        f"from CONTRACTION.",
    )
