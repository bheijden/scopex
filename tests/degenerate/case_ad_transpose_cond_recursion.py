"""jax#8239 -- reverse-mode through a recursively-halved `lax.cond` tree, where the recursion is
NOT shared by a scan and every level therefore inlines both of its children.

    https://github.com/jax-ml/jax/issues/8239          opened by patrick-kidger, no maintainer fix

READ THIS FIRST: THIS FILE IS THE SECOND HALF OF A PAIR.
`case_ad_transpose_bounded_while.py` already covers jax#8239 in its literal published form, where
each recursion level is a `lax.scan(length=2)` whose body is a `lax.cond`. That sharing is load
bearing: one scan body covers both children, so the jaxpr is O(log max_steps) and that file
measured exactly that -- 611 / 1340 / 2231 MLIR lines at max_steps 8 / 64 / 512, logarithmic on
both arms. Its docstring names the alternative it could not rule in or out: "if the jaxpr grows
multiplicatively down the recursion (each level's partial-eval duplicating the level below), trace,
lower and compile all scale with max_steps". This file is that alternative, built deliberately:

    def bwl(cond_fun, body_fun, val, max_steps):
        if max_steps == 1:
            return lax.cond(cond_fun(val), body_fun, lambda v: v, val)
        def go(v):
            v = bwl(cond_fun, body_fun, v, max_steps // 2)     # child 1, inlined
            return bwl(cond_fun, body_fun, v, max_steps // 2)  # child 2, inlined SEPARATELY
        return lax.cond(cond_fun(val), go, lambda v: v, val)

T(n) = 2 T(n/2) + O(1), so the FORWARD jaxpr already holds 2*max_steps - 1 `cond` equations. This
is the shape a user reaches for when they write a bounded loop by hand without knowing that the
scan trick exists, and it is the shape in which the AD pathology is loudest, because now the two
failure modes are separable:

    forward  is O(max_steps)                    -- unavoidable, it is an unrolled program
    backward is O(max_steps) x SOMETHING MORE   -- and that "something more" is the finding

WHERE THE EXTRA FACTOR COMES FROM. Reverse mode cannot short-circuit a `cond` the way execution
can. A transposed `cond` does not run one branch -- it CONTAINS both, transposed independently, so
the identity branch (free at runtime, and the branch actually taken for every step past the 8th)
still costs full program text. On top of that, partial-eval has to split each level into a
residual-producing forward half and a transposed backward half, and every residual produced
anywhere in a subtree must be plumbed out through the cond at the root of that subtree as an extra
output. A subtree of size k contributes O(k) residuals to its parent's output tuple, its parent's
parent's, and so on to the root: O(max_steps log max_steps) residual slots threaded through
O(max_steps) conds. So the prediction this file tests is a LOG FACTOR ON TOP OF THE UNROLL --
grad's curve should not merely sit above forward's, it should bend away from it.

THE PREDICATE IS THE POINT. `cond_fun` stops at step 8 for EVERY value of `max_steps`. The program
computes the same number at max_steps=8 and max_steps=512; the bound is arithmetically inert. Every
second of compile time above the max_steps=8 arm is therefore spent on program text that provably
cannot affect the answer. That is the attribution target: not "this line is slow" but "this STATIC
CONSTANT, which the program never reaches, is what you are paying for".

WHAT THE CONTROL ISOLATES. One decorator, nothing else. `condrec_grad_M` is `jax.grad(loss)`;
`condrec_grad_M_control` is `loss`. Same recursion, same max_steps, same MLP, same weights, same
input. HONEST CAVEAT, and it is the difference between this file and its sibling: here the control
is NOT expected to be flat in max_steps -- both arms unroll, so both grow. The corpus bar
(>=10x the control) may therefore not be cleared even when the mechanism is real and visible,
because the control is growing underneath it. THE STATISTIC TO READ IS THE RATIO'S TREND ACROSS THE
max_steps SWEEP, not any single arm's verdict. A grad/forward ratio that is flat in max_steps means
cond-transpose costs a constant multiple and there is no separate AD pathology here; a ratio that
climbs is the log factor above, and that is the result worth having. Recording that expectation
before the measurement is the whole reason this note exists.

SIZES. max_steps in (8, 32, 128, 512), powers of two (the recursion asserts it). 8 is the
degenerate point where the bound equals the real iteration count -- the y-intercept for both
curves, and the only arm where no work is wasted. 512 puts 1023 `cond` equations in the forward
jaxpr and is the arm most likely to exhaust the harness's 900 s timeout; a timeout there is a
measurement (superlinear growth), not a broken case, and the three surviving points still give a
slope.

PAYLOAD AND MEMORY. A hand-rolled 2-layer tanh MLP, IN_DIM -> WIDTH -> WIDTH -> IN_DIM at
WIDTH=128, matching the sibling file so the two constructions are directly comparable. The
reporter's `jax.experimental.stax` Dense(1024) stack no longer exists in jax 0.10.2 and 1024 would
be absurd here anyway: unlike the scan version, this construction inlines the payload
`2*max_steps - 1` times, so the matmul is what fills every duplicated branch with real HLO and its
width multiplies the program text directly. Unlike the scan version there is NO residual stacking
-- conds do not stack, they alias -- so the weights are a single 128x128 constant referenced by all
1023 branches and peak memory stays trivial (well under 10 MB at max_steps=512). Memory is not the
axis here; program text is.

PORTED from the 2021 issue: `jax.tree_map` -> `jax.tree.map`, stax -> the hand-rolled MLP. The
recursion itself is the cond-only variant described above, which is a deliberate deviation from the
published scan version and is the entire reason this file exists alongside its sibling.

VERIFIED ON CPU before committing (jax 0.10.2, no GPU touched). Both halves of the prediction hold:

    max_steps                      8       32      128
    top-level jaxpr equations      4        4        4     forward   <- CONSTANT
                                   6        6        6     grad      <- CONSTANT
    StableHLO lines              207      831     3327     forward   <- 4.0x per 4x, exactly linear
                                 882     4570    22394     grad      <- 5.2x, then 4.9x per 4x
    grad / forward line ratio    4.26     5.50     6.73                <- CLIMBING
    loss  2.591550082700e-01 at every max_steps  (max abs difference 0.0)
    grad  identical at every max_steps           (max abs difference 0.0)

Two things to take from that table. First, the bound is inert to the last bit: 16x more program
text, same answer, so every line above the max_steps=8 arm is provably wasted. Second, the grad arm
does NOT merely sit above the forward arm at a fixed multiple -- the ratio moves 4.26 -> 5.50 ->
6.73 across two doublings-squared, which is the shape of the residual-plumbing log factor argued
for above, and it is the thing the sibling scan-shared file could not see because sharing hides it.

The third row is the attribution lesson and is why this belongs in the corpus at all: the top-level
jaxpr has FOUR equations at max_steps=8 and FOUR at max_steps=128. Everything that grew, grew
inside nested `cond` branch jaxprs. Any instrument that counts equations at the top level of the
jaxpr -- or attributes by top-level primitive -- reports that these programs are the same size, at
every size, while the emitted StableHLO grows 16x. Descending into branch closures is not optional
here; it is the only place the finding lives.

What the harness still has to settle is whether growth in program TEXT becomes growth in compile
TIME at the same rate. It need not: XLA's cost on 1023 nested conditionals is not obliged to be
linear in their count, and it could equally be worse.
"""

from __future__ import annotations

import functools

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

WIDTH = 128
IN_DIM = 1
STOP_AT = 8  # cond_fun goes false here for EVERY max_steps -- this is what makes the bound inert


# --------------------------------------------------------------------------------------------
# The cond-only recursion.  Two inlined children per level, no scan to share them.
# --------------------------------------------------------------------------------------------
def bounded_while_loop(cond_fun, body_fun, val, max_steps):
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if max_steps & (max_steps - 1) != 0:
        raise ValueError("max_steps must be a power of two")

    if max_steps == 1:
        return lax.cond(cond_fun(val), body_fun, lambda v: v, val)

    def go(v):
        v = bounded_while_loop(cond_fun, body_fun, v, max_steps // 2)
        return bounded_while_loop(cond_fun, body_fun, v, max_steps // 2)

    return lax.cond(cond_fun(val), go, lambda v: v, val)


# --------------------------------------------------------------------------------------------
# Payload.  Stand-in for the removed stax MLP.  numpy at module scope: import touches no device.
# --------------------------------------------------------------------------------------------
def _init_params(seed=8239):
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
    """The loop runs STOP_AT=8 iterations whatever `max_steps` is; only the TEXT grows."""

    def body_fun(val):
        x, step = val
        return (_mlp(params, x), step + 1)

    init = (x0, jnp.asarray(0, dtype=jnp.int64))
    out = bounded_while_loop(_cond_fun, body_fun, init, max_steps)
    return jnp.sum(out[0])


def _grad_loss(x0, params, max_steps):
    return jax.grad(_loss, argnums=0)(x0, params, max_steps)


# 8 is the y-intercept (bound == real iteration count, nothing wasted).  512 inlines 1023 conds and
# is the arm most likely to hit the harness timeout -- see the module docstring.
MAX_STEPS = (8, 32, 128, 512)

CASES = {}
for _m in MAX_STEPS:
    CASES[f"condrec_grad_{_m}"] = (
        functools.partial(_grad_loss, max_steps=_m),
        (_X0, _PARAMS),
        f"jax#8239 cond-only variant: grad through {2 * _m - 1} inlined lax.cond levels, "
        f"max_steps={_m}, loop still executes only {STOP_AT} iterations",
    )
    CASES[f"condrec_grad_{_m}_control"] = (
        functools.partial(_loss, max_steps=_m),
        (_X0, _PARAMS),
        f"control: identical program with jax.grad removed, max_steps={_m}; NOT expected to be "
        f"flat (both arms unroll) -- read the grad/control RATIO across the sweep, not one point",
    )
