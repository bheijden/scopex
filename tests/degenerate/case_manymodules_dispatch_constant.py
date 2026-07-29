"""GAP 12 (many small modules vs one big one -- the PER-MODULE constant is the cost), SYNTHESISED.

No issue URL. Constructed from the mechanism. Nothing is claimed to be a bug: XLA charging a fixed
setup cost per module it compiles is correct behaviour. The pathology is a workload SHAPE, and it
is one the whole corpus has so far been structurally unable to express, because the harness
compiles exactly one module per case.

HOW ONE HARNESS CASE PRODUCES HUNDREDS OF MODULES. `jax.ensure_compile_time_eval()` (public API,
`jax.ensure_compile_time_eval`) drops out of the enclosing trace, so a `jax.jit`-ed call on
CONCRETE numpy inputs made inside it is genuinely compiled and executed right there, and its result
is folded into the outer jaxpr as a literal. So `fn` -- the plain callable the harness jits -- can
trigger K real end-to-end XLA compilations while the harness is inside `.lower()`. The harness
attributes all of it to `lower_s`, which is exactly right: it happened before the measured module
existed.

This is not a trick for its own sake. It is the standard shape of an eager/dispatch-dominated JAX
workload -- constants and lookup tables built at trace time, per-layer setup, op-by-op numpy-style
code -- where no single module is large and no single compiler pass is hot, and the entire bill is
K times a fixed per-module constant.

MEASURED IN-ENV BEFORE COMMITTING (jax 0.10.2, JAX_PLATFORMS=cpu, x64 on, each inner module is a
20-step sin/mul/add chain on a (4,4) f64 array, i.e. ~61 HLO instructions before fusion). ONE FRESH
PROCESS PER MEASUREMENT, arms interleaved, K=50:

    arm                     lower                compile          optimised HLO
    manymod_50            26.957 / 21.088 s    0.124 / 0.120 s      38 lines
    manymod_50_control     2.265 /  1.836 s    0.313 / 0.128 s      38 lines
    onebigmod_50           2.236 s             0.114 s              38 lines
    onebigmod_50_control   0.902 s             0.122 s              38 lines
    ragged_50             21.730 / 14.194 s   0.095 / 0.077 s       38 lines
    ragged_50_control      1.225 /  1.062 s   0.300 / 0.124 s       38 lines
    manymod_100           37.829 s            0.101 s               38 lines
    manymod_100_control    0.686 s            0.116 s               38 lines

-> manymod vs its control: 11.5x on `lower_s`, 1.0x on `compile_s`.
-> ragged vs its control: 15.6x on `lower_s`, 1.0x on `compile_s`.
-> manymod vs onebigmod (the SAME total instruction count in one module): 9.4x.

THE BACKEND COMPILE OF THE MEASURED MODULE IS ~0.12 s IN EVERY ARM AND ITS OPTIMISED HLO IS 38
LINES IN EVERY ARM. The final program is `sum(x) * <one f64 literal>` no matter which arm produced
the literal. There is nothing for an HLO-pass-timing profiler to look at: the module it is handed
compiles in a tenth of a second, after 21 seconds have already been spent.

Implied per-module constant at K=50: (21.09 - 1.84 - 1.34) / 49 = ~365 ms per 61-instruction
module, where 1.84 s is the process/warmup baseline and 1.34 s is the marginal cost of the same
work done inside one module (onebigmod_50 minus onebigmod_50_control). The K=100 fresh-process
point (37.8 s) gives ~370 ms per module by the same arithmetic, so the scaling looks LINEAR in the
number of modules over 50 -> 100, with the constant, not any exponent, being the whole story.

Note the two `_control` figures at K=50 (1.8-2.3 s) versus K=100 (0.69 s): the K=50 pair was
measured with a script that did not pre-warm the backend, so its controls carry ~1.2 s of process
startup. The pathological arms are unaffected at this magnitude, and the harness pre-warms nothing
either, so treat 1-2 s as the floor for every control in this file.

A SECOND, WARM, IN-PROCESS sweep over K gave 15.06 / 21.04 / 29.46 / 85.19 s for K = 50 / 100 /
200 / 400 against 0.042 / 0.065 / 0.120 / 0.285 s for the control and 2.81 / 4.52 / 6.47 / 13.18 s
for onebigmod. Those K-numbers are NOT comparable across rows and are recorded here only so nobody
re-derives them and thinks they mean something: the inner jit caches persist inside one process, so
the K=100 row paid for 50 new modules, not 100. Per NEW module it works out at 0.29-0.43 s, which
agrees with the fresh-process figure above. The harness, which forks per measurement, is what
produces the honest sweep.

WHAT EACH ARM ISOLATES.

  manymod_K          K calls to a jitted function with an UNUSED `static_argnums` tag taking K
                     distinct values -> K trace-cache misses -> K distinct modules compiled.
  manymod_K_control  The same K calls to the same jitted function with the tag held CONSTANT -> 1
                     compile and K-1 cache hits. Same number of Python calls, same number of
                     dispatches, same eager FLOPs, same returned values, bit-identical final jaxpr.
                     The single variable is the value of an argument the body never reads.
  ragged_K           The same bill arrived at LEGITIMATELY: K eager calls on K genuinely different
                     shapes (4, 4+i). No unused argument, no engineered cache key -- just ragged
                     data, which is how this actually happens to people.
  ragged_K_control   The standard fix: bucket every input to one shape (4, 4+K) so a single module
                     serves all K calls. It does strictly MORE runtime work and far less compile
                     work, which is the trade the case exists to price.
  onebigmod_K        ONE module containing all K bodies -- same total instruction count as the K
                     small modules put together, compiled once. Subtracting this arm from
                     `manymod_K` is what turns "many modules are slow" into a per-module constant.
  onebigmod_K_control  The same single module with ONE body. Gives the fixed cost of the one-big
                     -module arm so the K-scaling of `onebigmod_K` can be read off.

So the file answers three questions with one program: `manymod` vs its control says what a
gratuitous cache miss costs, `ragged` vs its control says what an unavoidable one costs and what
padding buys, and `manymod` vs `onebigmod` says how much of either is the per-module constant
rather than the work in the module.

SIZE SWEEP: K in (50, 100, 200, 400), which is the number of modules. The prediction is that
`manymod` and `ragged` are linear in K with a per-module constant of a few hundred milliseconds
while both controls stay flat, and the sweep is there to check the linearity claim rather than to
establish the ratio, which is already 11x at the smallest size.

PLATFORM: either, but note what "either" means for THIS case. The K inner compiles run on whatever
backend jax has selected, so on a GPU run this file performs K GPU compilations and K device
round-trips during `.lower()`. It was written and measured under JAX_PLATFORMS=cpu and the GPU arm
is deliberately unverified -- the GPU was owned by another investigation. The per-module constant
is expected to be LARGER on GPU (ptxas is in the loop), so if anything CPU is the conservative arm.

MEMORY: negligible. Every array in the file is (4,4) or (8,8) f64.

CAVEAT, STATED PLAINLY: `_harness.classify` renders its verdict on `compile_s`, which is ~0.1 s in
every arm here, so this case will be scored "no (below floor)". That is the harness measuring the
wrong column for this pathology, not the case failing. Read `lower_s`.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

W = 20          # sin/mul/add steps per small module -- ~61 HLO instructions before fusion


def _body(y):
    for _ in range(W):
        y = jnp.sin(y) * 1.0001 + 0.25
    return jnp.sum(y)


@functools.partial(jax.jit, static_argnums=1)
def _small(c, tag):
    """`tag` is never read. It exists only to key the trace cache."""
    return _body(c)


@functools.partial(jax.jit, static_argnums=1)
def _big(c, nbodies):
    tot = jnp.float64(0.0)
    for _ in range(nbodies):
        tot = tot + _body(c)
    return tot


@jax.jit
def _shaped(c):
    """No static argument at all -- the SHAPE of `c` is what varies the cache key."""
    return _body(c)


# numpy at module scope: importing this file claims no device.
C = np.full((4, 4), 0.5, dtype=np.float64)      # the eagerly-consumed constant
X = np.ones((8, 8), dtype=np.float64)           # the harness's actual argument


def _manymod(x, nmods: int, distinct: bool):
    with jax.ensure_compile_time_eval():
        acc = jnp.float64(0.0)
        for i in range(nmods):
            acc = acc + _small(C, i if distinct else 0)
    return jnp.sum(x) * acc


def _onebigmod(x, nbodies: int):
    with jax.ensure_compile_time_eval():
        acc = _big(C, nbodies)
    return jnp.sum(x) * acc


def _ragged(x, nmods: int, pad: bool):
    """The legitimate version of the same bill: the K modules genuinely differ (by shape).

    `pad=True` is the standard fix -- bucket every input to one shape, pay MORE flops, compile
    ONE module.
    """
    with jax.ensure_compile_time_eval():
        acc = jnp.float64(0.0)
        for i in range(nmods):
            width = 4 + nmods if pad else 4 + i
            acc = acc + _shaped(np.full((4, width), 0.5, dtype=np.float64))
    return jnp.sum(x) * acc


SIZES = (50, 100, 200, 400)

CASES = {}

for _k in SIZES:
    CASES[f"manymod_{_k}"] = (
        functools.partial(_manymod, nmods=_k, distinct=True), (X,),
        f"gap12 synth: {_k} DISTINCT tiny modules compiled during .lower() (unused static tag "
        f"takes {_k} values -> {_k} trace-cache misses). All cost lands in lower_s; the measured "
        f"module compiles in ~0.1 s",
    )
    CASES[f"manymod_{_k}_control"] = (
        functools.partial(_manymod, nmods=_k, distinct=False), (X,),
        f"control: the same {_k} calls, same dispatch count, same eager FLOPs, bit-identical final "
        f"jaxpr -- the unused static tag is held constant so it is 1 compile + {_k - 1} cache hits",
    )
    CASES[f"ragged_{_k}"] = (
        functools.partial(_ragged, nmods=_k, pad=False), (X,),
        f"gap12 synth, REALISTIC arm: {_k} eager calls on {_k} DISTINCT SHAPES (4, 4..{4 + _k}) -> "
        f"{_k} legitimate cache misses, {_k} modules. No unused argument anywhere",
    )
    CASES[f"ragged_{_k}_control"] = (
        functools.partial(_ragged, nmods=_k, pad=True), (X,),
        f"control: the standard fix -- every input bucketed to the single shape (4, {4 + _k}), so "
        f"1 module serves all {_k} calls. Strictly MORE runtime flops, far less compile",
    )
    CASES[f"onebigmod_{_k}"] = (
        functools.partial(_onebigmod, nbodies=_k), (X,),
        f"partition control: the SAME total instruction count as manymod_{_k}, emitted as ONE "
        f"module compiled once. manymod minus onebigmod is the per-module constant x {_k}",
    )
    CASES[f"onebigmod_{_k}_control"] = (
        functools.partial(_onebigmod, nbodies=1), (X,),
        f"control for the partition arm: the identical single module with 1 body instead of {_k}, "
        f"giving the fixed cost to subtract",
    )
