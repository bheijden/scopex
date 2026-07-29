"""xla#7971 / jax#18787 -- fusion is superlinear in unrolled chain length, and ADDING an
`optimization_barrier` makes it ~16x faster.

    https://github.com/openxla/xla/issues/7971         (downstream: jax-ml/jax#18787)

Reported numbers (author diagnosed by stack-sampling the compile thread; hawkinsp's reply was
"O(n^2) algorithms in compilation are not acceptable"):

    n         no barrier      with barrier
    2000        86.3 s            5.3 s

Reported mechanism: `InstructionFusion` visits O(n) fusion candidates, and for each candidate calls
`xla::gpu::SharedMemoryUsage`, which is itself O(n^2) in the size of any `kFusion` it walks because
it runs an un-memoised `FindNonTrivialHero` BFS per fused instruction. The whole elementwise chain
fuses into ONE growing fusion that reaches full program size, so the total is cubic. Lowering stays
fast, so the cost is provably on the XLA side of the boundary, not in tracing -- the harness times
`lower` and `compile` separately, which is exactly the discriminator.

WHY THIS CASE EARNS ITS SLOT: THE CONTROL IS INVERTED.
The payload is the dullest possible elementwise chain, so this case is not here for its ops. It is
here because `<case>_control` -- the arm with `jax.lax.optimization_barrier(x)` in the loop body --
has STRICTLY MORE HLO INSTRUCTIONS than the pathological arm and compiles an order of magnitude
FASTER, because the barrier caps how large a single fusion can grow and so drops the complexity
class. Every "attribute compile time by instruction count / by op count / by jaxpr size" heuristic
ranks this pair exactly backwards. If scopex says the barrier arm should be the slow one, the
attribution is measuring size, not cost.

Secondary control, unpaired: `fusion_rolled_n*` computes the same accumulation as a rolled
`lax.fori_loop`. The loop body is O(1) HLO regardless of n, so it should be sub-second at every n
and will register as "no (below floor)" -- which is the correct reading for a control arm.

WHAT IS UNCERTAIN, AND WHY THE SWEEP IS THE POINT.
A partial fix landed Aug 2024 (a `SharedMemoryUsage` cache in `PriorityFusion`, commit 5b0e626).
Nobody re-verified this repro afterwards, so the measurement on jax 0.10.2 is itself new
information. The fix may have degraded cubic to quadratic, in which case the absolute seconds at
n=2000 will be well under the reported 86 s and the case may miss the 3.0 s floor at small n. THE
SIGNAL TO READ OFF THIS FILE IS THE FITTED EXPONENT of compile_s vs n within each arm, and the
ratio between the arms at fixed n -- not any single number. Both arms are swept over identical n so
the exponents are directly comparable.

n is capped at 4000. If the original cubic scaling survived, n=8000 would be ~1400 s and blow the
harness's 900 s timeout; 4000 at cubic is ~340 s, which fits.
"""

from __future__ import annotations

import functools

import jax
import numpy as np

# Plain numpy at module scope: importing this file to discover CASES must never claim a device.
# jax.jit accepts numpy arrays and commits them to whichever device the harness chose.
# Shapes are 1x1 exactly as reported -- the pathology is in the number of fusion candidates, not in
# any tensor being large. Runtime is therefore microseconds in every arm, which is what makes the
# compile/runtime ratio test meaningful.
_X = np.ones((1, 1), dtype=np.float32)
_Y = np.ones((1, 1), dtype=np.float32)

# n values swept identically in both arms so the two fitted exponents are comparable.
_NS = (250, 500, 1000, 2000, 4000)


def _chain(x, y, n: int, barrier: bool):
    """The reported program. `barrier=True` is the INVERTED control: more HLO, less compile time."""
    t = x * y
    for _ in range(n):
        if barrier:
            # Public API on jax 0.10.2 (verified: jax.lax.optimization_barrier exists and traces to
            # an `optimization_barrier` primitive). It is semantically the identity; it exists only
            # to stop the optimiser fusing across it.
            x = jax.lax.optimization_barrier(x)
        t = t + x * y
    return t


def _rolled(x, y, n: int):
    """Secondary control: same accumulation, rolled. HLO size is O(1) in n."""

    def body(_, t):
        return t + x * y

    return jax.lax.fori_loop(0, n, body, x * y)


CASES = {}
for _n in _NS:
    CASES[f"fusion_chain_n{_n}"] = (
        functools.partial(_chain, n=_n, barrier=False),
        (_X, _Y),
        f"xla#7971 unrolled elementwise chain, n={_n}, NO optimization_barrier",
    )
    CASES[f"fusion_chain_n{_n}_control"] = (
        functools.partial(_chain, n=_n, barrier=True),
        (_X, _Y),
        f"inverted control: same chain n={_n} PLUS {_n} optimization_barriers -- more HLO, expected"
        " ~16x faster to compile",
    )

for _n in (2000, 4000):
    CASES[f"fusion_rolled_n{_n}"] = (
        functools.partial(_rolled, n=_n),
        (_X, _Y),
        f"secondary control: same accumulation as a rolled lax.fori_loop, n={_n}, O(1) HLO",
    )
