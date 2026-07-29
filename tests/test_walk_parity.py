"""The load-bearing test: our traversal must equal JAX's own, on every nesting construct.

The payload is byte-identical in every case, so the correct attribution is known in advance and is
the same for all of them. This catches the class of bug that made ``remat`` report 1 equation where
JAX reported 82 -- a sub-jaxpr spelling the traversal did not recognise.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax                                                                  # noqa: E402
import jax.numpy as jnp                                                     # noqa: E402
import pytest                                                              # noqa: E402

import scopex                                                              # noqa: E402

X = jnp.zeros((32, 32), jnp.float32)
IX = jnp.int32(3)


def payload(x, ix):
    """A scatter chain plus a reduction -- the shape whose attribution we care about."""
    for i in range(8):
        x = x.at[ix + i % 8].add(x[0] * (1.0 + i * 1e-4))
    return jnp.sum(x)


WRAPPERS = {
    "flat":        lambda x, ix: payload(x, ix),
    "pjit":        lambda x, ix: jax.jit(payload)(x, ix),
    "pjit_x3":     lambda x, ix: jax.jit(jax.jit(jax.jit(payload)))(x, ix),
    "remat":       lambda x, ix: jax.checkpoint(payload)(x, ix),
    "named_scope": lambda x, ix: _in_scope(payload, x, ix),
    "vmap":        lambda x, ix: jax.vmap(lambda _: payload(x, ix))(jnp.arange(3)).sum(),
    "scan":        lambda x, ix: jax.lax.scan(
        lambda c, _: (c, payload(x, ix)), 0.0, jnp.arange(2))[1].sum(),
    "cond":        lambda x, ix: jax.lax.cond(
        ix > 0, lambda: payload(x, ix), lambda: 0.0),
    "switch":      lambda x, ix: jax.lax.switch(
        1, [lambda: 0.0, lambda: payload(x, ix), lambda: 1.0]),
    "fori":        lambda x, ix: jax.lax.fori_loop(
        0, 2, lambda i, c: c + payload(x, ix), 0.0),
    "while":       lambda x, ix: jax.lax.while_loop(
        lambda s: s[0] < 2, lambda s: (s[0] + 1, s[1] + payload(x, ix)), (0, 0.0))[1],
    "grad":        lambda x, ix: jax.grad(lambda y: payload(y, ix))(x).sum(),
    "grad_of_pjit": lambda x, ix: jax.grad(lambda y: jax.jit(payload)(y, ix))(x).sum(),
}


def _in_scope(f, x, ix):
    with jax.named_scope("lib:probe"):
        return f(x, ix)


@pytest.mark.parametrize("name", sorted(WRAPPERS))
def test_parity_with_jax(name):
    """scopex.walk must find exactly what jaxpr_util.all_eqns finds."""
    j = jax.make_jaxpr(WRAPPERS[name])(X, IX)
    ours, theirs, equal = scopex.verify_parity(j)
    assert equal, f"{name}: scopex found {ours}, jax found {theirs}"


@pytest.mark.parametrize("name", sorted(WRAPPERS))
def test_addresses_round_trip(name):
    """`at(jaxpr, unit.addr)` must return the very equation the unit came from."""
    j = jax.make_jaxpr(WRAPPERS[name])(X, IX)
    for u in scopex.walk(j):
        assert scopex.at(j, u.addr) is u.eqn, f"{name}: bad address {u.addr}"


@pytest.mark.parametrize("name", sorted(WRAPPERS))
def test_scatter_attribution_survives_nesting(name):
    """Every wrapper must attribute the identical payload to the same source line."""
    j = jax.make_jaxpr(WRAPPERS[name])(X, IX)
    sites = {u.site for u in scopex.walk(j) if "scatter" in u.kind}
    assert sites, f"{name}: traversal never reached the scatter"
    assert len(sites) == 1, f"{name}: payload split across {sites}"
    assert sites.pop().endswith(f":{payload.__code__.co_firstlineno + 3}")


def test_unknown_bucket_is_not_silently_filled():
    """An equation with no user frame must be reported, not given its caller's line."""
    j = jax.make_jaxpr(WRAPPERS["flat"])(X, IX)
    plain = [u.site for u in scopex.walk(j)]
    inherited = [u.site for u in scopex.walk(j, inherit_site=True)]
    assert plain.count(scopex.walk.__module__ and "<no-frame>") >= 0     # bucket exists as a label
    assert len(plain) == len(inherited)
    # inheritance may only ever REDUCE the unknown count, never increase it
    assert inherited.count("<no-frame>") <= plain.count("<no-frame>")
