"""Regression tests for the NATIVE optimized-HLO walk and the fusion decision dump.

Every test here corresponds to a defect that was measured on jaxlib 0.10.2, not imagined.

The headline one: ``scopex.levels._INSTR`` captured an instruction's shape as ``\\S+``, and a TUPLE
shape contains a space -- ``(s32[], f32[8]{0})``. So the parser silently skipped every tuple-shaped
instruction. Measured against the native walk over 2,811 real per-pass dump snapshots, it
UNDERCOUNTED on 895 of them (31.8%), never overcounted, and missed 1,208 instructions. The missing
ones were not a random sample: they were ``while``, ``call``, ``tuple``, ``custom-call`` with a
scratch output, and every ``parameter`` of a control-flow body.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax                                                                  # noqa: E402
import jax.numpy as jnp                                                     # noqa: E402
import pytest                                                              # noqa: E402

import scopex                                                              # noqa: E402
from scopex import fusion as F                                             # noqa: E402
from scopex import levels as L                                             # noqa: E402


def _while_compiled():
    return jax.jit(lambda x: jax.lax.while_loop(
        lambda s: s[0] < 5, lambda s: (s[0] + 1, s[1] * 2), (0, x))[1]).lower(jnp.ones(8)).compile()


# ── the walk must actually run ───────────────────────────────────────────────────────────────────
def test_walk_hlo_is_not_dead_code():
    """``walk_hlo`` called ``frame_tables``/``hlo_sites``/``metadata``, which were declared in
    ``__all__`` and defined NOWHERE. Every call raised NameError, and the suite was green because
    nothing exercised it. An exported function that cannot run once is the same class of defect as
    a parser that returns nothing."""
    for name in ("frame_tables", "hlo_sites", "metadata", "hlo_module"):
        assert callable(getattr(L, name)), f"scopex.levels.{name} is missing"
    assert list(scopex.walk_hlo(_while_compiled()))


# ── the tuple-shape blind spot ───────────────────────────────────────────────────────────────────
def test_tuple_shaped_instructions_are_not_dropped():
    """A tuple shape has a space in it, so `\\S+` cannot match one. These are exactly the
    control-flow and library-call instructions that carry the attribution."""
    kinds = {i.kind for i in scopex.walk_hlo(_while_compiled())}
    # NOT `call`: whether CallInliner leaves one behind depends on flags the BACKEND was built
    # with, and `scopex.dump()` anywhere earlier in the process changes that permanently (see its
    # docstring). These three are structural to a while_loop and survive either way.
    for op in ("while", "tuple", "parameter"):
        assert op in kinds, f"{op} missing; tuple-shaped instructions are being dropped again"


def test_native_walk_is_a_superset_of_the_line_parser():
    """The old parser's answer must be reproducible as a SUBSET of the native one -- if the native
    walk ever loses a unit the line parser found, that is a regression the other way."""
    old = __import__("re").compile(
        r"^\s*(?:ROOT\s+)?%?(?P<name>[\w.\-]+)\s*=\s*(?P<shape>\S+)\s+[a-z][\w-]*\(")
    for fn, args in ((lambda x: jax.lax.while_loop(lambda s: s[0] < 5,
                                                   lambda s: (s[0] + 1, s[1] * 2), (0, x))[1],
                      (jnp.ones(8),)),
                     (lambda x, y: jax.lax.scan(lambda c, z: (c + z, c * z), y, x)[1],
                      (jnp.ones((8, 4)), jnp.ones(4)))):
        co = jax.jit(fn).lower(*args).compile()
        native = {i.unit for i in scopex.walk_hlo(co)}
        text = scopex.hlo_text(co)
        line = {m.group("name") for m in (old.match(ln) for ln in text.splitlines()) if m}
        assert line <= native, f"native walk lost units the line parser saw: {line - native}"
        assert native - line, "expected the native walk to find units the line parser could not"


# ── metadata: mixed quoting, which is how source resolution was lost once already ────────────────
def test_metadata_keeps_unquoted_values():
    """``op_name`` is quoted, ``stack_frame_id`` is not. A quoted-only pattern drops the int and
    the optimized module then reads as carrying no source location at all. It does carry one.

    The second assertion guards the reintroduction of the same bug one layer down: ``re.findall``
    reports a non-participating group as ``""`` rather than ``None``, so picking the quoted
    alternative with ``or``/``is not None`` on a findall tuple yields ``""`` for every unquoted
    value -- which is the identical failure with a different cause."""
    m = L.metadata('%f = f32[] fusion(%a), metadata={op_name="jit(f)/tanh" stack_frame_id=3}')
    assert m["op_name"] == "jit(f)/tanh"
    assert m["stack_frame_id"] == "3", "unquoted metadata value was dropped or blanked"
    assert L.metadata("%a = f32[] add(%b, %c)") == {}


def test_source_lines_resolve_through_the_frame_tables():
    """``stack_frame_id`` indexes four module-level tables. Following that indirection is the only
    way to a source line at this level, and the walk must land on the caller's real file."""
    def f(x, w):
        y = jnp.tanh(x @ w)                      # noqa: F841  -- this line is the assertion
        return jnp.sin(y).sum()

    co = jax.jit(f).lower(jnp.ones((16, 16)), jnp.ones((16, 16))).compile()
    tab = L.frame_tables(scopex.hlo_text(co))
    assert tab["files"] and tab["frames"], "frame tables did not parse"
    sites = {i.site for i in scopex.walk_hlo(co) if i.site != "<no-frame>"}
    assert sites, "no instruction resolved to a source line"
    assert any(s.split(":")[0].endswith(os.path.basename(__file__)) for s in sites), sites


def test_frame_walk_terminates_on_a_self_parent():
    """jaxlib 0.10.2 writes a root frame whose ``parent_frame_id`` is its OWN id. An unguarded
    parent walk never returns."""
    tab = {"files": {1: "/x/y.py"}, "functions": {1: "g"},
           "locations": {1: {"file_name_id": 1, "function_name_id": 1, "line": 7}},
           "frames": {1: {"file_location_id": 1, "parent_frame_id": 1}}}
    assert L.hlo_sites(1, tab) == ("/x/y.py:7", "g")


# ── opcodes come from the enum, and the enum is checked against what XLA prints ──────────────────
def test_opcode_comes_from_the_enum_and_matches_what_xla_prints():
    """``opcode`` is derived from the ``HloOpcode`` enum, which cannot fail to parse. Six enum
    names print differently (``kExp`` -> ``exponential``); any OTHER divergence must warn rather
    than silently rename an opcode."""
    co = jax.jit(lambda x: jnp.exp(x) + jnp.log1p(x) + jnp.sin(x)).lower(jnp.ones(8)).compile()
    L._WARNED.clear()                        # the warning is once-per-opcode process-wide
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ops = {r["opcode"] for r in L.hlo_instructions(co)}
    assert not [x for x in w if "OPCODE_TEXT" in str(x.message)], [str(x.message) for x in w]
    assert ops & {"exponential", "log-plus-one", "sine"}, ops


def test_an_unlisted_opcode_divergence_warns():
    """The self-check must fire, or the table above is just an assertion about the past."""
    L._WARNED.clear()
    saved = L.OPCODE_TEXT.pop("exp")
    try:
        co = jax.jit(lambda x: jnp.exp(x)).lower(jnp.ones(8)).compile()
        with pytest.warns(RuntimeWarning, match="OPCODE_TEXT"):
            list(L.hlo_instructions(co))
    finally:
        L.OPCODE_TEXT["exp"] = saved
        L._WARNED.clear()


# ── the fusion decision dump, read as a proto ────────────────────────────────────────────────────
SAMPLE = '''
fusion_steps {
  producer_ineligible { producer_name: "transpose.2" reason: "the consumer is not fusible" }
}
fusion_steps {
  update_priority { producer_name: "pad.1.0" consumer_names: "transpose.2"
                    us_fused: 993.886 us_unfused: 2973.37695 }
}
fusion_steps {
  fusion { fusion_name: "fusion.1" producer_name: "pad.1.0" consumer_name: "transpose.2" }
}
fusion_steps {
  some_step_added_upstream { producer_name: "x.1" reason: "a kind this code has never seen" }
}
gpu_device_info { name: "NVIDIA GeForce RTX 4090 Laptop GPU"
                  cuda_compute_capability { major: 8 minor: 9 feature_extension: NONE } }
hlo_module_before_fusion: "HloModule m\\n\\nENTRY main {\\n  ROOT c = f32[] constant(1)\\n}\\n"
'''


def test_textproto_parses_scalars_by_type():
    d = F.parse_textproto(SAMPLE)
    up = d["fusion_steps"][1]["update_priority"]
    assert up["us_fused"] == pytest.approx(993.886)
    assert d["gpu_device_info"]["cuda_compute_capability"]["major"] == 8
    assert d["gpu_device_info"]["cuda_compute_capability"]["feature_extension"] == "NONE"
    assert "\n" in d["hlo_module_before_fusion"], "escapes in a quoted string were not decoded"


def test_an_unknown_step_kind_survives_instead_of_vanishing():
    """THE reason this is a proto parser and not a regex. A grep for the three step kinds that
    exist today returns a shorter, entirely plausible list when XLA adds a fourth. Text-proto is
    self-describing, so the kind is READ OFF the file and an unknown one is visible."""
    kinds = {s.kind for s in F.fusion_steps(F.parse_textproto(SAMPLE))}
    assert "some_step_added_upstream" in kinds, kinds
    assert F.fusion_summary(F.parse_textproto(SAMPLE))["kinds"]["some_step_added_upstream"] == 1


def test_fusion_summary_reports_decisions():
    s = F.fusion_summary(F.parse_textproto(SAMPLE))
    assert s["fusions"] == [("pad.1.0", "transpose.2", "fusion.1")]
    assert s["refusals"] == {"the consumer is not fusible": 1}
    assert s["device"].startswith("NVIDIA")


def test_malformed_textproto_raises_rather_than_returning_empty():
    """An empty dict reads as 'this compile made no fusion decisions'. It must not be reachable
    from a parse failure."""
    for bad in ("fusion_steps {", "fusion_steps { fusion {", "a: ", "} {"):
        with pytest.raises(F.TextProtoError):
            F.parse_textproto(bad)


def test_the_embedded_pre_fusion_module_is_real_hlo():
    """``hlo_module_before_fusion`` is a whole HLO module, so it composes with the native parser --
    the dump can be read against the module it describes without a second compile."""
    d = F.parse_textproto(SAMPLE)
    m = L.hlo_module(d["hlo_module_before_fusion"])
    assert [r["opcode"] for r in L.hlo_instructions(m)] == ["constant"]


def test_hlo_module_raises_on_input_that_is_not_hlo():
    """``hlo_module_from_text`` is XLA's own parser and it FAILS on a non-HLO dump file
    (thunk_metadata.txt, buffer-assignment.txt) instead of returning an empty module."""
    with pytest.raises(Exception):
        L.hlo_module("thunk_metadata {\n  thunk_info {\n    profile_annotation: \"copy.1\"\n}\n}")
