"""jax#17335 -- gradient compile is SUPER-linear in N while the jaxpr stays exactly linear in N.

    https://github.com/jax-ml/jax/issues/17335

WHAT THE ISSUE REPORTS.  An unrolled Kalman-style sequential update of length N, written in numpyro.
Compiling the forward model "takes seconds" and scales tamely; compiling ``jax.grad`` of the same
source is super-linear in N.  The decisive detail, which the reporter measured and stated
explicitly, is that THE GRADIENT JAXPR'S LINE COUNT STAYS LINEAR IN N.  The program does not grow
faster than N; the compile does.

WHY THIS CASE EARNS ITS PLACE IN THE CORPUS.  It is the most direct available test of the single
proposition scopex has to defend -- that jaxpr size is a poor proxy for compile cost -- because the
size measurement has already been done and comes out flat.  Everything else in the corpus can be
argued away as "the program got bigger somewhere you were not looking".  Here the two arms are the
same source, the size ratio between them is a CONSTANT, and the compile ratio is not.

MEASURED IN THIS ENVIRONMENT AT TRACE TIME (CPU, jax 0.10.2, no execution).  This is the load-
bearing measurement of the file and it was taken before the file was written:

    N        fwd eqns   per step   grad eqns   per step   grad/fwd   fwd HLO   grad HLO   ratio
    125        1 501      12.0       2 750       22.0       1.83      1 757      3 005     1.71
    250        3 001      12.0       5 500       22.0       1.83      3 507      6 005     1.71
    500        6 001      12.0      11 000       22.0       1.83      7 007     12 005     1.71
    1000      12 001      12.0      22 000       22.0       1.83     14 007     24 005     1.71

Perfectly linear in N, in both arms, at both the jaxpr and the stablehlo level, with a CONSTANT
1.83x / 1.71x size penalty for the gradient.  So if the grad arm's compile time grows faster than
N, or its compile-per-instruction is worse than the forward arm's at the same N, neither jaxpr node
counting nor HLO instruction counting can explain it.  Trace time is also recorded here because it
is a rival explanation that must be excluded, and the harness reports it separately as ``lower_s``:
grad tracing took 1.27 / 3.08 / 4.86 / 10.24 s at N = 125 / 250 / 500 / 1000, i.e. also linear.  If
the blowup shows up in ``lower_s`` rather than ``compile_s`` this case moves from "XLA scheduling"
to "jax tracing" -- a different subject for a profiler, and a finding either way.

MECHANISM (a reading of the evidence; the issue carries no maintainer diagnosis).  The driver is
not program size but the SHAPE of the reverse-mode dependence graph.  Forward, the program is a
chain: each step reads one carry and writes one carry, so the dataflow graph has width 1 and any
scheduler can walk it in one pass with two live values.  Reverse, every one of the N residual terms
is live back to the same two roots ``a`` and ``b``, so the transposed graph is a chain of length N
whose N cotangent contributions all accumulate into two scalars.  That is a width-N fan-in over a
depth-N chain: the live range of the accumulators spans the whole program, no two accumulation
steps are independent, and the fusion/scheduling passes that are near-linear on a chain are not
near-linear on this.  Compile cost is indexed by graph topology, which is invisible to every
size-based proxy.

THE THREE CONTROLS, each a single edit, in decreasing tightness.

  * ``_control`` (auto-paired by the harness): THE SAME SOURCE WITH NO ``grad``.  ``seqgrad_N500``
    vs ``seqgrad_N500_control`` is ``jax.grad(model)`` vs ``model``, identical loop, identical
    arrays, identical N.  The size ratio between them is the constant 1.83x above, so any compile
    ratio materially above 1.83x is attributable to the reverse pass and to nothing else.
  * The N LADDER (125 / 250 / 500 / 1000, each arm at each N) measures the EXPONENT.  A straight
    line through log(compile) vs log(N) with slope > 1 for the grad arm and slope ~1 for the forward
    arm is the whole claim, and both arms are present at every N so it can be read off rather than
    asserted.
  * ``_scan``: the same computation rewritten as ``lax.scan``, still under ``grad``.  Verified: the
    scan gradient's jaxpr is 7 equations at EVERY N (against 22 000 at N = 1000) and returns
    numerically identical values (checked against the unrolled version at N = 500 to 1e-12).  This
    is the rewrite a user would actually be told to make, and it collapses the graph to a fixed size
    -- so it is both a control and the fix.

RECONSTRUCTION NOTES.  numpyro is dropped deliberately: ``dist.Normal(m, s).log_prob(v)`` is
``-0.5*((v-m)/s)**2 - log(s) - 0.5*log(2*pi)`` and with s fixed the last two terms are additive
constants that change nothing, so the body here is ``lp -= 0.5*(x_i - m)**2``.  That removes any
doubt about numpyro contributing ops of its own, and it is why the per-step equation counts above
are clean round numbers.  ``xs`` is a traced ARGUMENT rather than a captured numpy constant: as a
constant, ``xs[i]`` would be a python float and the indexing ops would vanish from the jaxpr, which
would flatter the size measurement that this file's whole argument rests on.

N was capped at 1000.  The harness's per-measurement timeout is 900 s; at N = 2000 the grad arm
traces for ~20 s before XLA even starts and, if the super-linearity is real, would risk timing out.
If the exponent comes out clearly above 1, N = 2000 is the natural follow-up and is one edit to
``N_LADDER``.

MEMORY.  ``xs`` is N f64 -- 8 kB at the largest N.  Nothing here is about data size.

PLATFORM.  Nothing in this mechanism is backend-specific; scheduling a wide-fan-in accumulation DAG
is XLA-common work.  Expect it on CPU and GPU alike, and treat a large divergence as a finding.
"""

from __future__ import annotations

import functools

import numpy as np

import jax
import jax.numpy as jnp

# Each rung has BOTH arms so the exponent can be read off, not asserted.
N_LADDER = (125, 250, 500, 1000)

DECAY = 0.9          # the AR(1) coefficient of the state update; fixed, not a knob


def _model(a, b, xs, n: int):
    """Unrolled sequential update: N steps, two roots, one scalar out.

    ``a`` seeds the state, ``b`` scales the input, and every step contributes one residual term to
    the running log-density.  Forward this is a width-1 chain; transposed it is a width-N fan-in
    into ``a`` and ``b``.  Written as a python ``for`` so it is UNROLLED -- that is the case.
    """
    lp = 0.0
    m = a
    for i in range(n):
        m = DECAY * m + b * xs[i]
        lp = lp - 0.5 * (xs[i] - m) ** 2
    return lp


def _scan_model(a, b, xs):
    """REWRITE CONTROL: identical numerics, rolled into lax.scan -- 7 jaxpr equations at any N."""

    def step(m, x):
        m = DECAY * m + b * x
        return m, -0.5 * (x - m) ** 2

    _, terms = jax.lax.scan(step, a, xs)
    return jnp.sum(terms)


def _grad_unrolled(a, b, xs, n: int):
    return jax.grad(functools.partial(_model, n=n), argnums=(0, 1))(a, b, xs)


def _grad_scan(a, b, xs):
    return jax.grad(_scan_model, argnums=(0, 1))(a, b, xs)


def _args(n: int):
    # Deterministic, O(1) memory, and bounded away from zero so no term degenerates.
    xs = np.linspace(0.1, 1.0, n)
    return (0.3, 0.7, xs)


CASES: dict = {}

for _n in N_LADDER:
    CASES[f"seqgrad_N{_n}"] = (
        functools.partial(_grad_unrolled, n=_n),
        _args(_n),
        f"jax#17335: grad of an unrolled {_n}-step sequential update -- {22 * _n} jaxpr eqns, "
        f"exactly linear in N; compile is reported super-linear",
    )
    CASES[f"seqgrad_N{_n}_control"] = (
        functools.partial(_model, n=_n),
        _args(_n),
        f"control: the SAME source with no grad, N={_n} -- {12 * _n + 1} eqns, a constant 1.83x "
        f"smaller than the grad arm at every N",
    )

# The rewrite control, at the two largest N only: its jaxpr is 7 equations either way, so more
# rungs would measure nothing.
for _n in (500, 1000):
    CASES[f"seqgrad_N{_n}_scan"] = (
        _grad_scan,
        _args(_n),
        f"rewrite control: identical numerics via lax.scan under grad, N={_n} -- 7 jaxpr eqns at "
        f"any N. Not auto-paired by the harness; compare against seqgrad_N{_n} by hand",
    )
