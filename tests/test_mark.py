"""Regression tests for the marking contract.

Every test here corresponds to a defect that was MEASURED, not imagined. Five of them crashed or
silently misattributed in the first implementation of this module; two more were found by
adversarially reviewing competing designs and then reproduced here.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax                                                                  # noqa: E402
import jax.numpy as jnp                                                     # noqa: E402
import pytest                                                              # noqa: E402

import scopex                                                              # noqa: E402

X = jnp.ones((8, 8))


# ── the contract itself ──────────────────────────────────────────────────────────────────────────
def test_slash_is_rejected():
    """A '/' in a name is indistinguishable from nesting below the jaxpr, so it must not compile."""
    with pytest.raises(scopex.mark.InvalidScopeName):
        scopex.scope("pkg", scopex.USER, "a/b")


def test_round_trip():
    assert scopex.parse(scopex.scope("dflux", scopex.USER, "Col.residual")) == \
        ("dflux", "user", "Col.residual")
    assert scopex.parse("not a mark") is None
    assert scopex.parse("pkg:notarole.x") is None


# ── __init_subclass__ mechanics: all four of these crashed before ────────────────────────────────
def test_base_without_parent_hook():
    """`super(cls, cls)` on the SUBCLASS resolves back to the wrapper and recurses forever."""
    @scopex.mark_framework("mylib", ("f",))
    class Block:
        pass

    class Kid(Block):
        def f(self, x):
            return x * 2

    assert Kid.f._scopex_marked


def test_ancestor_hook_fires_exactly_once():
    """Calling both the base's own hook AND super() fires an ancestor's hook twice per subclass,
    which silently double-applies registries and dataclass transforms."""
    seen = []

    class Root:
        def __init_subclass__(cls, **kw):
            super().__init_subclass__(**kw)
            seen.append(cls.__name__)

    @scopex.mark_framework("mylib", ("f",))
    class Block(Root):
        pass

    class Kid(Block):
        def f(self, x):
            return x * 2

    assert seen.count("Kid") == 1, f"ancestor hook fired {seen.count('Kid')}x"


def test_bases_own_hook_is_preserved():
    seen = []

    @scopex.mark_framework("mylib", ("f",))
    class Block:
        def __init_subclass__(cls, **kw):
            super().__init_subclass__(**kw)
            seen.append(cls.__name__)

    class Kid(Block):
        def f(self, x):
            return x * 2

    assert seen == ["Kid"]
    assert Kid.f._scopex_marked


def test_class_keyword_arguments():
    """`class Kid(Base, flavor='x')` is how flax/equinox-adjacent bases are parameterised."""
    @scopex.mark_framework("mylib", ("f",))
    class Block:
        def __init_subclass__(cls, **kw):
            super().__init_subclass__()

    class Kid(Block, flavor="x"):
        def f(self, x):
            return x * 2

    assert Kid.f._scopex_marked


def test_own_package_subclass_is_not_marked():
    """A subclass defined INSIDE the framework's own package is library code, not a user's."""
    own_root = __name__.split(".")[0]          # this test module IS the 'framework' here

    @scopex.mark_framework(own_root, ("f",))
    class Block:
        pass

    class Sibling(Block):
        def f(self, x):
            return x * 2

    assert not getattr(Sibling.f, "_scopex_marked", False)

    foreign = type("Foreign", (Block,), {"f": lambda s, x: x, "__module__": "somebody_else.mod"})
    assert foreign.f._scopex_marked


# ── the binary split is binary, and says so ──────────────────────────────────────────────────────
def test_multi_root_marking_warns():
    """The user/library axis is binary. Ecosystem middleware (flaxformer under flax, say) is
    'foreign' and so reads as user-authored. That is a real limit; it must be loud, not silent."""
    @scopex.mark_framework("mylib", ("f",))
    class Block:
        pass

    type("Mid", (Block,), {"f": lambda s, x: x, "__module__": "middleware.layers"})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        type("Usr", (Block,), {"f": lambda s, x: x, "__module__": "__main__"})
    assert any("more than one package root" in str(x.message) for x in w)


# ── two frameworks at once (requirement: namespaces must not overwrite each other) ───────────────
def test_two_frameworks_both_survive():
    """Designs carrying ownership in ONE metadata key lose the outer namespace when frameworks
    nest. Nested named scopes plus a full-sequence accessor do not."""
    @scopex.mark_framework("flax", ("__call__",))
    class Module:
        pass

    @scopex.mark_framework("dflux", ("residual",))
    class Block:
        pass

    class UserBlock(Block):
        def residual(self, x):
            return jnp.tanh(x)

    class UserMod(Module):
        def __call__(self, x):
            with jax.named_scope("dflux:lib.solve"):
                return UserBlock().residual(x)

    def prog(x):
        with jax.named_scope("flax:lib.apply"):
            return jnp.sum(UserMod()(x))

    deep = [u for u in scopex.walk(jax.make_jaxpr(prog)(X)) if len(u.marks) > 1]
    assert deep, "no unit carried marks from both frameworks"
    assert set(deep[0].packages) == {"flax", "dflux"}
    assert [r for _, r, _ in deep[0].marks] == ["lib", "user", "lib", "user"]


# ── the class-less consumer ──────────────────────────────────────────────────────────────────────
def user_vector_field(x):
    return jnp.sin(x) * jnp.cos(x)


def _solve(vf, x, mark: bool):
    if mark:
        vf = scopex.mark_callable(vf, "diffrax", "vector_field")
    with jax.named_scope("diffrax:lib.solve"):
        return jnp.sum(vf(x))


def test_unmarked_callable_is_credited_to_the_framework():
    """The defect, stated as a test so it cannot be forgotten: a library of plain functions that
    ingests a user callable reports the USER's code as library-authored. Not 'unknown' -- WRONG."""
    u = list(scopex.walk(jax.make_jaxpr(lambda x: _solve(user_vector_field, x, False))(X)))
    split = scopex.attribute(u, "split")
    assert split.get("user", 0) == 0
    assert split["library"] > 0


def test_mark_callable_fixes_it():
    u = list(scopex.walk(jax.make_jaxpr(lambda x: _solve(user_vector_field, x, True))(X)))
    split = scopex.attribute(u, "split")
    assert split["user"] > 0, "mark_callable did not attribute the user's function to the user"


# ── the zero-dependency path: stock jax only, no scopex import on the framework side ─────────────
def test_hand_written_convention_is_read_identically():
    """The README's fifteen-line version must produce exactly what the sugar produces."""
    class HandBlock:
        def __init_subclass__(cls, **kw):
            super().__init_subclass__(**kw)
            if cls.__module__.split(".")[0] != "mylib":
                for name in ("f",):
                    fn = cls.__dict__.get(name)
                    if fn is None:
                        continue

                    def wrap(fn=fn, tag=f"mylib:user.{cls.__qualname__}.{name}"):
                        def marked(*a, **k):
                            with jax.named_scope(tag):
                                return fn(*a, **k)
                        return marked
                    setattr(cls, name, wrap())

    class Kid(HandBlock):
        def f(self, x):
            return jnp.tanh(x)

    u = list(scopex.walk(jax.make_jaxpr(lambda x: jnp.sum(Kid().f(x)))(X)))
    assert any(y.authored for y in u), "hand-written convention produced no readable mark"
    assert scopex.attribute(u, "split")["user"] > 0


# ── the package's own API surface: three defects found by running it, not by reading it ──────────
def test_record_is_the_function_not_the_module():
    """`scopex.record` was the SUBMODULE. `from .records import ...` ran after
    `from .monitor import record`, and importing a submodule binds it on the parent package --
    so the headline API call in the README was uncallable. Same shadowing class as the
    Op/Ins collision this project hit in dflux."""
    assert callable(scopex.record), f"scopex.record is {type(scopex.record).__name__}"


def test_monitor_metric_names_actually_match_jax():
    """`_KEYS` once appended '_secs' to every metric name -- a suffix jax does not emit -- so
    record() matched nothing and reported 0.0 for all three stages while looking healthy."""
    seen = set()
    jax.monitoring.register_event_duration_secs_listener(lambda n, v, **k: seen.add(n))
    jax.jit(lambda x: jnp.sum(jnp.tanh(x))).lower(X).compile()
    from scopex import monitor
    assert seen & set(monitor._KEYS), (
        f"no jax metric matches scopex.monitor._KEYS.\n  jax emits: {sorted(seen)}\n"
        f"  scopex expects: {sorted(monitor._KEYS)}")


def test_timings_shouts_when_nothing_matched():
    """A silent zero is indistinguishable from an instant compile. It must not be silent."""
    from scopex.monitor import Timings
    t = Timings({"wall": 1.0, "seen_names": ["/jax/core/compile/something_renamed"]})
    assert not t.matched
    assert "NO JAX METRICS MATCHED" in str(t)


def test_levels_are_exported():
    for name in ("walk_hlo", "walk_stablehlo", "hlo_instructions"):
        assert hasattr(scopex, name), f"scopex.{name} is not exported"
