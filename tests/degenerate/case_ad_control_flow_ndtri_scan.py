"""jax#2609 -- reverse-mode AD through `ndtri` (which still has no derivative rule) inside
`lax.scan`: the scan's partial-eval fixpoint replicates a branch-heavy derivative that forward mode
never builds.

    https://github.com/jax-ml/jax/issues/2609          OPEN, P1, labelled NVIDIA-GPU

Reported (CPU, 2020, scan trip count ONE):

    lax.scan   jacfwd 1.07 s      jacrev 5.76 s       reverse mode 5.4x forward
    python loop jacfwd 185 ms     jacrev 198 ms       reverse mode 1.07x forward

so the scan is a ~29x amplifier on REVERSE MODE SPECIFICALLY (198 ms -> 5.76 s), while costing
forward mode ~6x. Post-compile runtimes are equal between the two modes -- the reporter says so
explicitly -- which is what makes this a compile-time case rather than an efficiency one.

MECHANISM, AND WHY IT IS STILL LIVE. `jax.scipy.special.ndtri` has no `custom_jvp`/`custom_vjp` in
jax 0.10.2 -- verified against the installed source: `log_ndtr`, `logit`, `xlogy`, `expi`, `sici`,
`hyp1f1` and ten others carry `@custom_derivatives.custom_jvp`; `def ndtri` at special.py:1484 has
no decorator. So AD differentiates straight through the piecewise Cephes rational approximation:
two polynomial evaluations selected by `lax.select` on the input range. Forward mode pushes one
tangent through that and stops. Reverse mode has to keep residuals for every branch and every
polynomial term, and `_scan_partial_eval_custom`'s fixpoint then splits the body into
jaxpr_known / jaxpr_staged / jaxpr_known_hoist, replicating the branch structure per residual.

WHAT EACH CONTROL ISOLATES -- three of them, in increasing purity:

  `..._control`   ndtri wrapped in a `custom_vjp` whose backward is the analytic derivative
                  1/pdf(ndtri(p)) = sqrt(2*pi) * exp(ndtri(p)^2 / 2). Still reverse mode, still
                  inside lax.scan, same trip count, same arithmetic, same numerical answer. The
                  ONLY difference is that one primitive now has a derivative rule instead of being
                  differentiated through. This is the sharpest control in the file: whatever
                  compile time survives it is not attributable to the missing rule.
  `ndtri_scan_jacfwd_*`   same program, `jacrev` -> `jacfwd`. One token. Isolates AD direction.
  `ndtri_loop_jacrev_*`   same program, `lax.scan` -> python `for` of the same trip count.
                  Isolates the scan amplifier from the cost of differentiating ndtri at all: this
                  arm still differentiates the rational approximation in reverse mode, it just does
                  not route it through scan's partial-eval machinery.

TWO DELIBERATE DEPARTURES FROM THE ISSUE, both because the issue as written cannot clear a 3 s bar:

 1. THE SIZE KNOB IS BODY DEPTH, NOT SCAN LENGTH. The obvious way to scale this up -- raise the
    scan's trip count -- does not work and would have wasted a measurement slot. `lax.scan` lowers
    to a while loop; the trip count is a constant in the HLO, not an unroll, so the jaxpr and the
    HLO are the SAME SIZE at length 8 and at length 512 and compile time is flat in it. What does
    grow the program is the number of chained copula links inside the body, so `depth` is the knob
    and the trip count is pinned low (8) to keep runtime small enough for the compile/runtime
    ratio to mean something.
 2. Trip count 8, not the reporter's 1, so there is a real loop to differentiate.

READ `lower_s`, NOT ONLY `compile_s`, WHEN JUDGING THIS ONE. Measured here on CPU, tracing only
(`jax.make_jaxpr`, no XLA involved at all), x64 on, jax 0.10.2:

    depth                                 4       16      32
    jacrev through ndtri, in scan      3.23 s  23.77 s  39.19 s     149 / 557 / 1101 eqns
    same, ndtri given a custom_vjp             9.62 s               557 eqns, 2.5x cheaper to trace
    jacfwd                                             15.09 s      12 eqns

Two things follow. First, the mechanism is REAL and still live in 2026: at identical equation count
(557 both arms, depth 16) reverse mode through the rule-less rational approximation costs 2.5x the
tracing of reverse mode with the rule, and the jaxpr it produces is 6187 lines against 2812. Second
-- and this is the part that will mislead a reader who only looks at the harness's verdict column
-- most of that cost lands in `lower()`, not in `compile()`. `_scan_partial_eval_custom`'s fixpoint
is python-level partial evaluation; it runs while tracing, before XLA sees anything. The harness
records `lower_s` but classifies on `compile_s` alone, so this case can perfectly well come back
"no (below floor)" while carrying a 40 s lowering time. That is not a non-reproduction, it is a
pathology in a different stage, and for a tool whose job is attributing compilation artifacts to
source it is arguably the more interesting stage: nothing in the HLO explains where the time went.

UNCERTAIN, AND SAID PLAINLY: 5.76 s was CPU on jax 0.1.x in 2020. Six years of scan partial-eval
work sit between that and jax 0.10.2, and the absolute number here may land under the 3 s floor
even at depth 32. If it does, the RATIO between the arms is still the finding -- a 20x
jacrev/custom_vjp gap at 1 s is the same mechanism as at 6 s and is still something scopex must be
able to attribute to a missing derivative rule on one primitive. Ignore the 2022 non-repro comment
in the thread: it compares post-compile `%timeit` runtimes, which the reporter already said are
equal, and its "loop" arm uses `range(1)`.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.lax import scan
from jax.scipy.special import erfc, ndtri

# Trip count of the scan. Pinned LOW on purpose -- see departure (1) in the docstring. Compile time
# does not depend on it; runtime does, and runtime in the denominator of compile/runtime does.
TRIP = 8

_SQRT2 = np.sqrt(2.0)
_SQRT_2PI = np.sqrt(2.0 * np.pi)


# --------------------------------------------------------------------------------------------
# The control's one and only difference from the pathological arm: ndtri gets a derivative rule.
# --------------------------------------------------------------------------------------------
@jax.custom_vjp
def _ndtri_with_rule(p):
    return ndtri(p)


def _ndtri_fwd(p):
    x = ndtri(p)
    return x, x


def _ndtri_bwd(x, g):
    # d/dp ndtri(p) = 1 / normal_pdf(ndtri(p)) = sqrt(2*pi) * exp(ndtri(p)**2 / 2)
    return (g * _SQRT_2PI * jnp.exp(0.5 * x * x),)


_ndtri_with_rule.defvjp(_ndtri_fwd, _ndtri_bwd)


def _link(cop_dist, rho, ndtri_fn):
    """One Gaussian-copula link, verbatim from the issue's `norm_copula_distribution`."""
    pu = ndtri_fn(cop_dist)
    pv = ndtri_fn(0.9 * cop_dist)
    z = (pu - rho * pv) / jnp.sqrt(1.0 - rho ** 2)
    return erfc(-z / _SQRT2) / 2


def _f_scan(cop_dist, rho, depth, ndtri_fn, trip=TRIP):
    """Pathological structure: the copula chain lives inside lax.scan."""
    def body(carry, _):
        cd, r = carry
        for _ in range(depth):
            cd = _link(cd, r, ndtri_fn)
        return (cd, r), None

    (cd, _), _ = scan(body, (cop_dist, rho), None, length=trip)
    return cd


def _f_loop(cop_dist, rho, depth, ndtri_fn):
    """Control structure: the identical chain as a python loop of the same trip count."""
    cd = cop_dist
    for _ in range(TRIP):
        for _ in range(depth):
            cd = _link(cd, rho, ndtri_fn)
    return cd


def _mk(structure, mode, depth, ndtri_fn, **kw):
    f = functools.partial(structure, depth=depth, ndtri_fn=ndtri_fn, **kw)
    jac = jax.jacrev if mode == "jacrev" else jax.jacfwd
    # differentiate wrt rho (argument 1), as in the issue
    return jac(f, 1)


# float64 numpy scalars: no device work at import, and x64 is on globally so these stay f64.
_ARGS = (np.float64(0.5), np.float64(0.5))

CASES = {}
for _d in (4, 16, 32):
    CASES[f"ndtri_scan_jacrev_d{_d}"] = (
        _mk(_f_scan, "jacrev", _d, ndtri), _ARGS,
        f"jax#2609 jacrev through {_d} chained ndtri links inside lax.scan (trip {TRIP}); "
        "expect the cost in lower_s, not compile_s -- scan partial-eval is a tracing-time fixpoint")
    CASES[f"ndtri_scan_jacrev_d{_d}_control"] = (
        _mk(_f_scan, "jacrev", _d, _ndtri_with_rule), _ARGS,
        f"control: byte-identical, ndtri given a custom_vjp derivative rule, depth {_d}")
    CASES[f"ndtri_scan_jacfwd_d{_d}"] = (
        _mk(_f_scan, "jacfwd", _d, ndtri), _ARGS,
        f"control (AD direction): same scan program under jacfwd, depth {_d}")

# The python-loop arm computes the SAME TOTAL WORK as the scan arm, which means TRIP x more program
# text (all TRIP iterations unrolled) -- it is not size-matched to the scan body and is not meant to
# be. It answers one question only: does routing this through scan's partial-eval fixpoint cost more
# than compiling TRIP times as much straight-line reverse-mode code? If yes, that is the amplifier
# the issue reports.
#
# Depth 4 ONLY. At depth 4 this arm is already 9806 jaxpr equations (against 149 for the scan arm at
# the same depth) and traces in 10 s; depth 16 would be ~40k equations, whose trace alone did not
# finish in 150 s of CPU budget and whose XLA compile would plausibly exceed the harness's 900 s
# timeout. One depth answers the question; the second would spend a long slot confirming that a 40k
# equation straight-line program is slow to compile, which nobody doubts.
# The trip-count claim in departure (1) above is an ARGUMENT ("scan lowers to a while loop, so the
# HLO is the same size at length 8 and at length 512"), and the issue's own suggestion is to scale
# the scan length. An argument that decides how a case is built should be measured, not believed, so
# this arm is ndtri_scan_jacrev_d16 with the trip count raised 8 -> 256 and NOTHING else changed.
# Deliberately unpaired: its comparison is the d16 arm already in the table, not a control of its
# own. If the two compile times match, trip count is confirmed irrelevant, the issue's advice is a
# dead end, and body depth is the only knob -- and the corpus has that in writing instead of in a
# comment. If they do NOT match, this file's whole scaling strategy is wrong and the d* arms are
# under-sized.
CASES["ndtri_scan_jacrev_d16_trip256"] = (
    _mk(_f_scan, "jacrev", 16, ndtri, trip=256), _ARGS,
    "probe (unpaired): ndtri_scan_jacrev_d16 with scan length 256 instead of 8 -- tests whether "
    "trip count moves compile time at all; compare against ndtri_scan_jacrev_d16")

CASES["ndtri_loop_jacrev_d4"] = (
    _mk(_f_loop, "jacrev", 4, ndtri), _ARGS,
    "control (scan amplifier): same total work unrolled as a python loop, depth 4")
