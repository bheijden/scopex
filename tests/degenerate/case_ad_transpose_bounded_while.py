"""jax#8239 -- an O(1) forward pass whose gradient costs O(max_steps) to trace, lower and compile.

    https://github.com/jax-ml/jax/issues/8239           opened by patrick-kidger, no maintainer fix

`bounded_while_loop` is Patrick Kidger's recursive-halving while loop (the ancestor of the one in
diffrax): a `max_steps`-step loop built as log2(max_steps) nested levels, each level a
`lax.scan(length=2)` whose body is a `lax.cond` that either recurses into the next level or is the
identity. Forward, the jaxpr is O(log max_steps) deep and, because `cond_fun` here stops at step 8
for every `max_steps`, EXECUTION short-circuits after 8 real iterations no matter how large the
bound is. Runtime is constant in `max_steps` by construction.

Backward it cannot short-circuit. Reverse-mode has to linearise and transpose EVERY branch of every
`cond` independently -- a transposed `cond` runs neither branch, it *contains* both -- and each
branch is itself another scan+cond level. Partial-eval of the nested scans also has to split each
level into a residual-producing forward half and a transposed half, and that split composes down
the whole recursion. The reporter's numbers (2021, CPU, the second call, i.e. RUN time):

    max_steps      8       16       32
    grad         0.0377   0.0594   0.1024      -- doubling max_steps roughly doubles it

WHAT IS AND IS NOT ESTABLISHED. The issue's published numbers are runtimes, and they scale linearly
in `max_steps`. The title claims compile time does too, and the compile side is what we care about,
but the issue body never tabulates it. So the compile-time claim is exactly what this case exists to
test, and it is genuinely open. Two ways it can land, both informative:

  * If the jaxpr grows multiplicatively down the recursion (each level's partial-eval duplicating
    the level below), trace+lower+compile all scale with `max_steps` and this is a strong case:
    compile cost driven by a STATIC BOUND that the runtime provably ignores.
  * If the jaxpr stays O(log max_steps) and only the RESIDUAL TENSORS grow (shape (2,2,...,2,width),
    i.e. max_steps elements per hidden unit), then the HLO instruction count stays flat while
    buffers grow, backward runtime grows with max_steps, and the compile/runtime ratio may fall
    below the corpus bar even though compile time itself climbs. That would still be a novel
    attribution target -- "compile cost concentrated in a handful of instructions whose SHAPES blew
    up" -- but it would not clear the >=1000 ratio, and this note is the warning.

Inspecting the lowered StableHLO on CPU before committing this file settled which of the two it is,
and the answer is the second one, so read the numbers accordingly:

    max_steps                    8      64     512
    MLIR lines, grad arm       611    1340    2231     <- O(log max_steps), NOT O(max_steps)
    MLIR lines, control arm     88     151      --
    largest f64 buffer         (2,)*log2(max_steps) + (WIDTH, WIDTH)

The instruction count really is logarithmic, and it is logarithmic on BOTH arms; grad just costs
~9x more per level. What is linear in `max_steps` is the BUFFERS. That largest residual is the
hidden weight MATRIX -- a loop-invariant constant -- stacked once per potential step by every scan
level in the recursion. A loop that runs 8 iterations materialises `max_steps` copies of its own
weights. (Measured at WIDTH=512 it was 2^6 x 512 x 512 = 134 MB at max_steps=64 and 2^9 x 512 x 512
= 1074 MB at 512, which is what forced the width down; see below.)

So the attribution target here is not "many instructions", it is "a handful of instructions whose
SHAPES were decided by a bound the program never reaches". The corpus bar may not like it: backward
runtime grows with the same factor as the buffers, so compile/runtime can stay below 1000 even
while compile time itself climbs. That is the uncertainty this case is carrying, stated up front.

The `WIDTH x WIDTH` in that formula is why the payload is 128 wide here and not the reporter's 1024:
residual memory is `max_steps * WIDTH**2 * 8` bytes, so WIDTH=512 is already 1 GB by max_steps=512
and would OOM long before the sweep got interesting. WIDTH=128 keeps max_steps=2048 at ~268 MB and
buys three more octaves of the axis the finding actually lives on. The MLP is still a real matmul
chain in every duplicated branch, which is what it was there for, and the MLIR line counts above
are width-independent (they were identical at 512 and at 128).

Either way the interesting number is the SLOPE, so this file sweeps `max_steps` and the harness's
per-case compile numbers are meant to be read as a curve, not as four independent verdicts.

WHAT THE CONTROL ISOLATES. One decorator. `bwl_grad_*` is `jax.grad(loss)`; `bwl_*_control` is
`loss` itself. Same `bounded_while_loop`, same MLP, same `max_steps`, same input, same everything.
The forward curve should be FLAT in `max_steps` (the jaxpr really is O(log n) and nothing about the
program grows), and the difference between the two slopes IS the finding. A single grad timing
proves nothing here -- a 4096-bound loop is a big program however you look at it; the control is
what makes the bound's irrelevance visible.

PORTED from the 2021 source. Two changes, both mechanical: `jax.tree_map` -> `jax.tree.map`, and
`jax.experimental.stax` (removed) -> a hand-rolled 2-layer tanh MLP standing in for
Dense(1024)/tanh/Dense(1024)/tanh/Dense(1). The loop structure -- the part that matters -- is
verbatim. `max_steps` must be a power of two (the recursion asserts it); 2048 is 11 nested levels.

Verified on CPU before committing: forward loss and gradient come out bit-identical at max_steps=8
and max_steps=64 (3.091151589860118e-14 and -8.25685617e-14). That is the precondition the whole
case rests on -- the bound genuinely does not change what the program computes, so every second of
compile time it buys is spent on nothing.
"""

from __future__ import annotations

import functools

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

WIDTH = 128
IN_DIM = 1
STOP_AT = 8  # cond_fun stops here for EVERY max_steps -- this is what makes runtime constant


# --------------------------------------------------------------------------------------------
# Verbatim from the issue, apart from jax.tree_map -> jax.tree.map.
# --------------------------------------------------------------------------------------------
def bounded_while_loop(cond_fun, body_fun, init_val, max_steps):
    """API as `lax.while_loop`, except that it takes an integer `max_steps` argument."""
    if not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError("max_steps must be a non-negative integer")
    if max_steps == 0:
        return init_val
    if max_steps & (max_steps - 1) != 0:
        raise ValueError("max_steps must be a power of two")

    init_data = (cond_fun(init_val), init_val)
    _, val = _while_loop(cond_fun, body_fun, init_data, max_steps)
    return val


def _while_loop(cond_fun, body_fun, data, max_steps):
    if max_steps == 1:
        pred, val = data
        new_val = body_fun(val)
        keep = lambda a, b: lax.select(pred, a, b)  # noqa: E731
        new_val = jax.tree.map(keep, new_val, val)
        return cond_fun(new_val), new_val
    else:

        def _call(_data):
            return _while_loop(cond_fun, body_fun, _data, max_steps // 2)

        def _scan_fn(_data, _):
            _pred, _ = _data
            return lax.cond(_pred, _call, lambda x: x, _data), None

        return lax.scan(_scan_fn, data, xs=None, length=2)[0]


# --------------------------------------------------------------------------------------------
# Payload. Hand-rolled stand-in for the removed stax MLP: IN_DIM -> WIDTH -> WIDTH -> IN_DIM.
# Built with numpy at module scope (2 MB, no device touched by importing this file).
# --------------------------------------------------------------------------------------------
def _init_params(seed=0):
    rng = np.random.default_rng(seed)

    def dense(i, o):
        return (rng.standard_normal((i, o)) / np.sqrt(i), np.zeros(o))

    return (dense(IN_DIM, WIDTH), dense(WIDTH, WIDTH), dense(WIDTH, IN_DIM))


_PARAMS = _init_params()
_X0 = np.ones((IN_DIM,), dtype=np.float64)


def _mlp(params, x):
    (w1, b1), (w2, b2), (w3, b3) = params
    x = jnp.tanh(x @ w1 + b1)
    x = jnp.tanh(x @ w2 + b2)
    return x @ w3 + b3


def _cond_fun(val):
    _, step = val
    return step < STOP_AT


def _loss(x0, params, max_steps):
    """sum(bounded_while_loop(...)); the loop runs STOP_AT=8 iterations for every max_steps."""

    def body_fun(val):
        x, step = val
        return (_mlp(params, x), step + 1)

    init = (x0, jnp.asarray(0, dtype=jnp.int64))
    out = bounded_while_loop(_cond_fun, body_fun, init, max_steps)
    return jnp.sum(out[0])


def _grad_loss(x0, params, max_steps):
    return jax.grad(_loss, argnums=0)(x0, params, max_steps)


# Powers of two only (the recursion asserts it). 8 is the degenerate case where the bound equals the
# real iteration count, so it doubles as the y-intercept for both curves. The ceiling is set by
# backward residual memory -- see the module docstring: max_steps * WIDTH**2 * 8 bytes, so 2048 at
# WIDTH=128 is ~268 MB and 4096 would be ~537 MB.
MAX_STEPS = (8, 64, 512, 2048)

CASES = {}
for _m in MAX_STEPS:
    CASES[f"bwl_grad_{_m}"] = (
        functools.partial(_grad_loss, max_steps=_m),
        (_X0, _PARAMS),
        f"jax#8239 grad(bounded_while_loop) max_steps={_m} (log2={_m.bit_length() - 1} nested "
        f"scan+cond levels); loop still runs only {STOP_AT} iterations",
    )
    CASES[f"bwl_grad_{_m}_control"] = (
        functools.partial(_loss, max_steps=_m),
        (_X0, _PARAMS),
        f"control: byte-identical program with jax.grad removed, max_steps={_m}; "
        f"this curve should be FLAT in max_steps",
    )
