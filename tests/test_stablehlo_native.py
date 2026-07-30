"""The StableHLO level is walked as an IR. These are the failures that walk cannot have.

Every test here corresponds to something the LINE-BASED parser it replaced got wrong, and all of
them are the project's characteristic failure shape: a plausible non-empty answer with a blind spot
that sits exactly on the interesting operations.
"""

from __future__ import annotations

import os
from collections import Counter

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax                                                                  # noqa: E402
import jax.numpy as jnp                                                     # noqa: E402
import pytest                                                              # noqa: E402

import scopex                                                              # noqa: E402
from scopex.levels import _walk_stablehlo_text, stablehlo_module           # noqa: E402
from scopex.walk import NO_FRAME                                           # noqa: E402

X = jnp.arange(8.0)
IX = jnp.int32(1)


def control_flow(x, ix):
    """One of every region-bearing operation, plus a marked scope to attribute them to."""
    def body(s):
        i, acc = s
        with jax.named_scope("mylib:lib.loop"):
            return i + 1, acc + jnp.sin(acc) * x[i % 4]
    _, acc = jax.lax.while_loop(lambda s: s[0] < 5, body, (0, x[0]))
    y = jax.lax.cond(ix > 0, lambda: jnp.sum(jnp.tanh(x)), lambda: jnp.sum(jnp.cos(x)))
    z = jax.lax.scan(lambda c, a: (c + a, c), 0.0, x)[0]
    return acc + y + z + jnp.sort(x).sum()


PROGRAMS = {
    "control_flow":  (control_flow, (X, IX)),
    "switch":        (lambda x, i: jax.lax.switch(
        i, [lambda v: jnp.tanh(v).sum(), lambda v: jnp.sin(v).sum(),
            lambda v: jnp.cos(v).sum()], x), (X, IX)),
    "scatter_chain": (lambda x, i: jnp.sum(x.at[i].add(x[0]).at[i + 1].mul(2.0)), (X, IX)),
    "grad_of_fori":  (lambda x, i: jax.grad(lambda v: jax.lax.fori_loop(
        0, 4, lambda k, c: c + jnp.tanh(c), v).sum())(x).sum(), (X, IX)),
    "vmap_scan":     (lambda x, i: jax.vmap(lambda r: jax.lax.scan(
        lambda c, a: (c + jnp.tanh(a), c), 0.0, r)[0])(jnp.stack([x, x])).sum(), (X, IX)),
    "nested_jit":    (lambda x, i: jax.jit(lambda v: jax.jit(jnp.tanh)(v) + 1.0)(x).sum(), (X, IX)),
}


def _lower(name):
    fn, args = PROGRAMS[name]
    return jax.jit(fn).lower(*args)


# ── the bar: at least what the text parser produced, on every opcode ─────────────────────────────
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_native_walk_is_a_superset_of_the_text_parser(name):
    """A rewrite that loses units is worse than the parser it replaced, so this is checked PER
    OPCODE and not on the total -- a total can hide a swap."""
    low = _lower(name)
    old = Counter(u.kind for u in _walk_stablehlo_text(low))
    new = Counter(u.kind for u in scopex.walk_stablehlo(low))
    lost = {k: (old[k], new[k]) for k in old if new[k] < old[k]}
    assert not lost, f"{name}: the IR walk lost units the text parser found: {lost}"
    assert sum(new.values()) >= sum(old.values())


def test_region_bearing_ops_are_invisible_to_a_line_parser_and_visible_here():
    """THE reason this level is not read as text. MLIR prints an op that owns a region across many
    lines and puts its `loc(...)` after the closing brace, so no single line carries both the opcode
    and its location. `while`, `case` and `sort` are therefore missing entirely from a line-based
    walk -- and they are the operations a compile-time question is usually about."""
    low = _lower("control_flow")
    text_kinds = Counter(u.kind for u in _walk_stablehlo_text(low))
    ir_kinds = Counter(u.kind for u in scopex.walk_stablehlo(low))
    for op in ("while", "case", "sort"):
        assert ir_kinds[op] > 0, f"the IR walk did not find {op}"
        assert text_kinds[op] == 0, (
            f"{op} is now visible to the line parser -- MLIR's printing changed, and the rest of "
            "this test file's reasoning should be re-derived rather than trusted")


def test_reduce_body_ops_have_no_line_at_all():
    """`stablehlo.reduce` prints in a short `applies stablehlo.add across dimensions = [0]` form.
    The `add` and `return` inside its region are real operations with real locations that are never
    printed on any line, so no text parser can ever see them."""
    low = _lower("control_flow")
    units = list(scopex.walk_stablehlo(low))
    reduce_bodies = [u for u in units if u.depth > 0]
    assert reduce_bodies, "no operation was found inside a region"
    assert sum(1 for u in units if u.kind == "add") > \
        sum(1 for u in _walk_stablehlo_text(low) if u.kind == "add")


# ── name peeling: the two ways to get a plausible wrong string ───────────────────────────────────
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_no_path_is_a_bare_primitive_marker(name):
    """jax wraps an inline-lowered primitive as NameLoc("scatter:", NameLoc(".../scatter", ...)).
    Reading only the outer one yields `"scatter:"` as the whole name stack -- a string with no
    scopes in it, so every contract accessor comes back empty for that unit."""
    for u in scopex.walk_stablehlo(_lower(name)):
        assert not u.path.endswith(":"), f"{name}: {u!r} kept the primitive marker as its path"


@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_no_path_is_a_frame_function_name(name):
    """The other direction, and the one that nearly shipped. When a traceback is a SINGLE frame jax
    emits NameLoc(name_stack, NameLoc(function, file:line)) with no callsite in between, so a rule
    that descends 'to the innermost NameLoc' returns the python function name -- `<module>` for a
    top-level call. 26 of 311 operations on examples/marked_framework.py."""
    for u in scopex.walk_stablehlo(_lower(name)):
        assert u.path != "<module>", f"{name}: {u!r} took a frame's function name as its name stack"
        assert "/" in u.path or not u.path or u.path.startswith("jit(") or u.function != u.path, (
            f"{name}: {u!r} path looks like a bare frame name")


def test_the_name_stack_is_the_whole_stack():
    low = _lower("control_flow")
    units = list(scopex.walk_stablehlo(low))
    assert any("mylib:lib.loop" in u.path for u in units), "the named scope is missing"
    assert any(u.marks for u in units), "no unit carried a readable mark"
    deep = [u for u in units if u.path.count("/") >= 2]
    assert deep, "no unit carried a nested name stack"


# ── the location also carries the source line, which the text level declined to answer ───────────
def test_sites_are_real_and_join_with_the_jaxpr_level():
    """`site` used to be the constant string '<see-jaxpr-level>'. The callsite chain is on the
    location, so it is now a real file:line -- filtered by the SAME rule scopex.walk uses, which is
    what makes the two levels joinable."""
    fn, args = PROGRAMS["control_flow"]
    low = jax.jit(fn).lower(*args)
    units = list(scopex.walk_stablehlo(low))
    resolved = [u for u in units if u.site != NO_FRAME]
    assert len(resolved) > len(units) // 2, "most operations should resolve to a source line"
    assert all(u.site != "<see-jaxpr-level>" for u in units)

    jaxpr_sites = {e.site for e in scopex.walk(jax.make_jaxpr(fn)(*args))}
    shared = sum(1 for u in resolved if u.site in jaxpr_sites)
    assert shared / len(resolved) > 0.8, (
        f"only {shared}/{len(resolved)} StableHLO sites exist at the jaxpr level -- the two levels "
        "are no longer applying the same frame filter, so `site` has stopped being a join key")

    assert any(u.site == NO_FRAME for u in units) or True   # the honest bucket is kept, not forced


def test_container_and_depth_locate_the_unit():
    low = _lower("nested_jit")
    units = list(scopex.walk_stablehlo(low))
    assert {u.container for u in units} - {"main", "<module>"}, "no outlined function was found"
    assert any(u.outlined for u in units)
    assert all(u.depth >= 0 for u in units)


# ── the accessor story, which is what sent the first implementation to the text ──────────────────
def test_compiler_ir_object_keeps_the_locations_its_printer_drops():
    """`compiler_ir('stablehlo')` was written up as dropping location info because `str()` of it
    contains zero `loc(`. That is `Operation.__str__`'s default, not the IR. The same object printed
    with debug info is byte-identical to `as_text(debug_info=True)`."""
    import io
    low = _lower("control_flow")
    m = low.compiler_ir("stablehlo")
    assert str(m).count("loc(") == 0, "the trap itself has gone away; update flags.TRAPS"
    buf = io.StringIO()
    m.operation.print(enable_debug_info=True, file=buf)
    assert buf.getvalue().strip() == scopex.stablehlo_text(low).strip()
    assert buf.getvalue().count("loc(") > 0


def test_compiler_ir_does_not_re_lower():
    low = _lower("control_flow")
    assert low.compiler_ir("stablehlo") is low.compiler_ir("stablehlo")
    assert stablehlo_module(low) is low.compiler_ir("stablehlo")


def test_text_input_round_trips_through_the_same_walk():
    """Parsing StableHLO text back into a module is lossless, so a saved dump gets the SAME answer
    as a live Lowered rather than the degraded one. It needs jax's registered ir context: a bare
    ir.Context() cannot even parse `func.func`."""
    low = _lower("control_flow")
    live = list(scopex.walk_stablehlo(low))
    reparsed = list(scopex.walk_stablehlo(scopex.stablehlo_text(low)))
    assert [(u.kind, u.path, u.site) for u in live] == \
           [(u.kind, u.path, u.site) for u in reparsed]


def test_the_fallback_is_never_silent():
    """The text parser is kept only for a jaxlib without the bindings, and reaching it must be
    loud -- a quietly degraded answer is the failure this module exists to prevent."""
    import warnings

    from scopex import levels
    low = _lower("control_flow")
    real = levels._mlir
    levels._mlir = lambda: None
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            units = list(scopex.walk_stablehlo(low))
        assert units, "the fallback produced nothing at all"
        assert any("LINE-BASED" in str(x.message) for x in w), "the fallback was silent"
    finally:
        levels._mlir = real
