"""A small framework that marks itself, and a user model written against it.

This is the example the blueprint quotes. It is deliberately NOT a toy in one respect: the library
runs `jax.jacrev` over the user's hook while building its operator, which is how reverse-mode
machinery ends up inside a program the caller only asked to `solve` -- the interaction the
`library x transform` crosstab exists to surface.

Run it:  python examples/marked_framework.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import scopex

jax.config.update("jax_enable_x64", True)


# ══════════════════════════════════════════════════════════════════════════════════════════════
# THE FRAMEWORK.  Depends on `jax`. Does not import scopex -- the two `named_scope` calls below
# are the entire contract, and `mark_framework` is optional sugar over the same thing.
# ══════════════════════════════════════════════════════════════════════════════════════════════
@scopex.mark_framework("mylib", ("forward", "residual"))
class Block:
    """Users subclass this and implement `forward` and `residual`."""

    def forward(self, x):
        raise NotImplementedError

    def residual(self, x):
        raise NotImplementedError


def assemble(block, x):
    """Build the operator. Needs the Jacobian of the user's residual, hence `jacrev` -- and hence
    reverse-mode machinery inside what the caller experiences as a forward solve."""
    with jax.named_scope("mylib:lib.assemble"):
        return jax.jacrev(block.residual)(x)


def step(block, x):
    with jax.named_scope("mylib:lib.step"):
        J = assemble(block, x)
        r = block.residual(x)
        return x - jnp.linalg.solve(J + 1e-3 * jnp.eye(x.shape[0]), r)


def solve(block, x, iters: int = 3):
    with jax.named_scope("mylib:lib.solve"):
        for _ in range(iters):
            x = step(block, x)
        return block.forward(x)


# ══════════════════════════════════════════════════════════════════════════════════════════════
# THE USER'S MODEL.  A different package as far as the contract is concerned, so its hooks are
# marked `mylib:user.*` automatically. The user wrote no scopex code and no decorators.
# ══════════════════════════════════════════════════════════════════════════════════════════════
class MyModel(Block):
    def __init__(self, w):
        self.w = w

    def forward(self, x):
        return jnp.sum(jnp.tanh(self.w @ x) ** 2)

    def residual(self, x):
        h = jnp.tanh(self.w @ x)
        return h * jnp.sin(x) + 0.1 * x**3 - 0.5


N = 24
MODEL = MyModel(jnp.linspace(-1.0, 1.0, N * N).reshape(N, N))
X0 = jnp.linspace(0.1, 0.9, N)


def program(x):
    return solve(MODEL, x)


if __name__ == "__main__":
    units = list(scopex.walk(jax.make_jaxpr(program)(X0)))
    ours, theirs, ok = scopex.verify_parity(jax.make_jaxpr(program)(X0))
    print(f"{len(units)} equations, parity vs jaxpr_util.all_eqns: {ours}=={theirs} -> {ok}\n")

    print(scopex.table(scopex.attribute(units, "split"), label="split"), "\n")
    print(scopex.table(scopex.attribute(units, "library"), label="mylib subsystem", top=6), "\n")
    print(scopex.table(scopex.attribute(units, "author"), label="user hook", top=6), "\n")

    print("library x transform -- where reverse mode reaches user code:")
    for row, cols in scopex.crosstab(units, rows="library", cols="transform").items():
        for c, n in cols.items():
            if "transpose" in c:
                print(f"  {n:5d}  {row}  x  {c}")
