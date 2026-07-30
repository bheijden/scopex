"""The quarantine, tested by REINTRODUCING each bug it exists to catch.

A guard that has never fired is a guard nobody has tested. Every test below monkey-patches
``scopex._parse`` back into a shape this package actually shipped, and asserts that a
:class:`ParseError` comes out instead of a plausible answer.

No compiles here: every test runs against the verbatim samples embedded in ``scopex._parse``, which
is the point of embedding them.
"""

from __future__ import annotations

import os
import pathlib
import re

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest                                                              # noqa: E402

from scopex import _parse                                                  # noqa: E402
from scopex._parse import ParseError                                       # noqa: E402


# ── the conformance suite itself ────────────────────────────────────────────────────────────────

def test_conformance_passes_on_its_own_samples():
    r = _parse.conformance()
    assert r["ok"], r["failures"]
    assert not r["failures"]


def test_every_parser_is_exercised_by_conformance():
    """Every public parser in _parse must appear in PARSERS, or it is unguarded by construction."""
    covered = {p.name.split("[")[0] for p in _parse.PARSERS}
    # Record types and helpers, not parsers. `PassSplit` is what `pass_leaf_split` RETURNS -- the
    # parser itself is in PARSERS, and a NamedTuple is callable only because it is a class.
    infra = {"ParseError", "expect", "conformance", "Parser", "PassTime", "Frame", "hlo_site",
             "hlo_shape_and_opcode", "stablehlo_loc_aliases", "pass_pipeline_headers",
             "MlirPassDump", "PassSplit"}
    public = {n for n in _parse.__all__
              if callable(getattr(_parse, n)) and n not in infra}
    assert public <= covered, f"parsers with no conformance entry: {sorted(public - covered)}"


# ── the invariant ───────────────────────────────────────────────────────────────────────────────

def test_expect_allows_empty_result_from_empty_input():
    assert _parse.expect("p", [], "") == 0
    assert _parse.expect("p", [], "   \n ") == 0


def test_expect_rejects_empty_result_from_nonempty_input():
    with pytest.raises(ParseError, match="BROKEN PARSER"):
        _parse.expect("p", [], "some real output\n")


def test_expect_counts_the_witness_rather_than_merely_asking_for_one():
    """1 result from 3,214 witnesses is the shape of bug #1, and `n > 0` does not catch it."""
    text = "\n".join(f"  %{i} = stablehlo.add %a, %b : tensor<f32> loc(#loc{i})" for i in range(50))
    with pytest.raises(ParseError, match="returned 1 .* at least 50"):
        _parse.expect("p", ["one"], text, witness=_parse._MLIR_SSA)
    assert _parse.expect("p", ["x"] * 50, text, witness=_parse._MLIR_SSA) == 50


def test_expect_allow_empty_still_fires_when_the_witness_is_present():
    assert _parse.expect("p", [], "no custom calls here\n", witness=r"custom_call_target=",
                         allow_empty=True) == 0
    with pytest.raises(ParseError):
        _parse.expect("p", [], 'x custom_call_target="a"\n', witness=r"custom_call_target=",
                      allow_empty=True)


# ── bug #1: the StableHLO location indirection ──────────────────────────────────────────────────

def test_bug1_inline_loc_only_pattern_now_raises(monkeypatch):
    """The shipped version matched only ``loc("name")`` and returned 1 unit from real modules."""
    inline_only = re.compile(
        r'^\s*(?:%[\w#$.-]+\s*=\s*)?(?P<op>[a-z_][\w.]*\.[\w.]+|return|func\.func)'
        r'.*?\bloc\((?P<loc>"(?:[^"\\]|\\.)*")\)')
    monkeypatch.setattr(_parse, "_MLIR_OP", inline_only)
    with pytest.raises(ParseError, match="operations"):
        _parse.stablehlo_op_lines(_parse.SAMPLE_STABLEHLO)


def test_bug1_broken_alias_definitions_now_raise(monkeypatch):
    """If the alias-DEFINITION syntax moves, every name silently blanks and the count still looks
    perfect. The dangling-reference check is what sees it."""
    monkeypatch.setattr(_parse, "_LOC_ALIAS", re.compile(r"^(?P<alias>#nope)=(?P<body>.*)$", re.M))
    with pytest.raises(ParseError, match="no definition"):
        _parse.stablehlo_op_lines(_parse.SAMPLE_STABLEHLO)


def test_stablehlo_never_returns_a_file_path_as_a_name():
    """FileLineColLoc aliases begin with a quoted string too -- a path, not a name stack."""
    ops = _parse.stablehlo_op_lines(_parse.SAMPLE_STABLEHLO)
    assert ops and not any(name.startswith("/") for _, name in ops)
    assert any("mylib:user.MyModel.residual" in name for _, name in ops)


# ── bug #2: quoted-only metadata ────────────────────────────────────────────────────────────────

def test_bug2_unquoted_values_are_read():
    md = _parse.hlo_metadata(
        '%t = f32[8,8]{1,0} tanh(%p), metadata={op_name="jit(f)/tanh" stack_frame_id=6}')
    assert md["op_name"] == "jit(f)/tanh"
    assert md["stack_frame_id"] == "6"


def test_bug2_quoted_only_pattern_is_caught_by_conformance(monkeypatch):
    monkeypatch.setattr(_parse, "_META_KV", re.compile(r'(?P<k>\w+)="(?P<qv>[^"]*)"(?P<uv>)'))
    with pytest.raises(ParseError, match="stack_frame_id|conformance FAILED"):
        _parse.conformance()


def test_bug2_is_invisible_per_instruction_and_caught_per_module(monkeypatch):
    """The point of check_metadata_coverage. With a quoted-only pattern every instruction still
    returns a non-empty dict -- op_name is quoted -- so nothing looks wrong until you count the
    stack_frame_ids the MODULE says it has against the ones that came out."""
    monkeypatch.setattr(_parse, "_META_KV", re.compile(r'(?P<k>\w+)="(?P<qv>[^"]*)"(?P<uv>)'))
    line = ('%t = f32[8,8]{1,0} tanh(%p), metadata={op_name="jit(f)/tanh" stack_frame_id=6}')
    assert _parse.hlo_metadata(line) == {"op_name": "jit(f)/tanh"}      # looks perfectly fine
    with pytest.raises(ParseError, match="stack_frame_id"):
        _parse.check_metadata_coverage([_parse.hlo_metadata(line)], line)


def test_metadata_coverage_passes_when_nothing_is_dropped():
    _parse.check_metadata_coverage(
        [_parse.hlo_metadata(ln) for ln in _parse.SAMPLE_HLO.splitlines() if " = " in ln],
        _parse.SAMPLE_HLO)


def test_metadata_block_present_but_unreadable_raises(monkeypatch):
    monkeypatch.setattr(_parse, "_META_BLOCK", re.compile(r"metadataXX=\{(?P<body>.*?)\}"))
    with pytest.raises(ParseError, match="could not read"):
        _parse.hlo_metadata('%t = f32[] tanh(%p), metadata={op_name="a"}')


# ── bug #3: the time unit XLA switches to for the slowest pass ──────────────────────────────────

def test_bug3_min_is_converted_and_ranks_first():
    times = {p.name: p.seconds for p in _parse.pass_timing_lines(_parse.SAMPLE_PASS_LOG)}
    assert times["autotuner"] == pytest.approx(71.651421, abs=1e-4)
    assert max(times, key=times.get) == "autotuner"


def test_bug3_unknown_unit_raises_instead_of_dropping_the_slowest_pass(monkeypatch):
    monkeypatch.setattr(_parse, "UNITS", {"us": 1e-6})
    log = "HLO pass: autotuner time: 1.19 min\nHLO pass: cheap time: 3 us\n"
    with pytest.raises(ParseError, match="cannot convert time units"):
        _parse.pass_timing_lines(log)


def test_pass_names_containing_spaces_are_not_dropped():
    """Found by the witness check on the first live log it saw: 384 lines said `HLO pass: ` and 378
    parsed, because `(?P<name>\\S+)` cannot match `simplification after layout assignment`."""
    times = {p.name for p in _parse.pass_timing_lines(_parse.SAMPLE_PASS_LOG)}
    assert "simplification after layout assignment" in times


def test_pass_timing_witness_catches_an_undercount(monkeypatch):
    monkeypatch.setattr(_parse, "_PASS_LINE", re.compile(
        r"HLO pass:\s+(?P<name>\S+)\s+time:\s+(?P<val>[\d.]+)\s*(?P<unit>[a-z]+)"
        r"(?:\s*\((?P<us>\d+)\s*us\))?"))
    with pytest.raises(ParseError, match="pass timings"):
        _parse.pass_timing_lines(_parse.SAMPLE_PASS_LOG)


def test_pass_log_keeps_module_attribution():
    mods = {m for m, _ in _parse.pass_pipeline_headers(_parse.SAMPLE_PASS_LOG)}
    assert mods == {"jit_convert_element_type", "jit_top"}


# ── the stack-frame parent link ─────────────────────────────────────────────────────────────────

def test_parent_offset_is_derived_and_is_one_on_jaxlib_0_10_2():
    tab = _parse.hlo_frame_tables(_parse.SAMPLE_HLO)
    assert tab["parent_offset"] == 1


def test_frame_stack_resolves_the_whole_python_stack():
    tab = _parse.hlo_frame_tables(_parse.SAMPLE_HLO)
    assert [(f.function, f.line) for f in _parse.hlo_frame_stack(6, tab)] == [
        ("leaf", 6), ("residual", 10), ("solve", 14), ("top", 17), ("<module>", 19)]


def test_the_literal_parent_reading_would_truncate_every_stack():
    """Why the offset is derived. Reading parent_frame_id as printed makes leaf frames self-parent,
    so a cycle-guarded walk returns ONE frame and the innermost line still looks right."""
    tab = dict(_parse.hlo_frame_tables(_parse.SAMPLE_HLO))
    tab["parent_offset"] = 0
    assert len(_parse.hlo_frame_stack(6, tab)) == 1


def test_same_line_reached_two_ways_gets_two_different_stacks():
    """leaf() is called from residual() and from solve(); frames 6 and 8 share a file location and
    must not share a caller. A flat read of the table gives them the same one."""
    tab = _parse.hlo_frame_tables(_parse.SAMPLE_HLO)
    a, b = _parse.hlo_frame_stack(6, tab), _parse.hlo_frame_stack(8, tab)
    assert a[0] == b[0] and a[1] != b[1]


def test_unreadable_parent_links_raise_rather_than_guess(monkeypatch):
    frames = {1: {"file_location_id": 1, "parent_frame_id": 2},
              2: {"file_location_id": 2, "parent_frame_id": 1}}          # a 2-cycle either way
    with pytest.raises(ParseError, match="acyclic"):
        _parse._parent_offset(frames)


def test_frame_tables_absent_is_not_an_error():
    """Per-pass dump snapshots carry stack_frame_id but no tables. That is a real state."""
    tab = _parse.hlo_frame_tables("HloModule m\n\nENTRY %main () -> f32[] {\n  ROOT %c = f32[] "
                                  "constant(0)\n}\n")
    assert tab["frames"] == {} and tab["parent_offset"] == 0


# ── the tuple-shape blind spot ──────────────────────────────────────────────────────────────────

def test_tuple_shapes_survive():
    assert _parse.hlo_shape(_parse.SAMPLE_HLO_TUPLE_LINES[0]) == "(s32[], f32[8]{0})"
    assert _parse.hlo_shape(_parse.SAMPLE_HLO_TUPLE_LINES[2]) == "(f32[8]{0}, s32[])"


def test_shape_pattern_that_cannot_match_a_tuple_now_raises(monkeypatch):
    monkeypatch.setattr(_parse, "_INSTR", re.compile(
        r"^\s*(?:ROOT\s+)?%?(?P<name>[\w.\-]+)\s*=\s*(?P<shape>\S+)\s+(?P<opcode>[a-z][\w-]*)\("))
    with pytest.raises(ParseError, match="could not read the shape"):
        _parse.hlo_shape(_parse.SAMPLE_HLO_TUPLE_LINES[0])


# ── the quarantine boundary ─────────────────────────────────────────────────────────────────────

# Patterns that are NOT parsers of compiler output, and why they are allowed to live elsewhere.
_ALLOWED = {
    ("levels.py", "_CAMEL"): "splits a python enum NAME (kExpMinusOne), not compiler output; every "
                             "divergence from XLA's printed spelling is warned about",
    ("fusion.py", "_TOKEN"): "a complete proto3-text-format tokenizer over a SELF-DESCRIBING "
                             "format -- an unknown field arrives as an unknown key, not as silence",
    ("raw.py", "pat"): "compiles the CALLER's pattern -- Raw.grep and Raw.verify search the "
                       "artifact for whatever the user asks for. scopex authors none of these, "
                       "so there is no scopex parser here to quarantine",
    ("raw.py", "n_wit"): "counts the caller-supplied witness in the caller-supplied text; same "
                         "reason as ('raw.py', 'pat')",
}


def test_no_other_module_parses_compiler_output_with_a_regex():
    """The quarantine, enforced. If this fails, a pattern has escaped back into a caller."""
    src = pathlib.Path(_parse.__file__).parent
    escaped = []
    for f in sorted(src.glob("*.py")):
        if f.name == "_parse.py":
            continue
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if re.search(r"\bre\.(compile|search|match|finditer|findall|sub|split)\b", line):
                # `^` and not `^\s*` once meant an INDENTED assignment could not be named,
                # so a local could only be exempted by widening the rule to the whole file.
                name = (re.findall(r"^\s*(\w+)\s*=", line) or ["?"])[0]
                if (f.name, name) in _ALLOWED:
                    continue
                if f.name == "fusion.py" and "re.match" in line:
                    continue                       # inside the tokenizer's escape handling
                escaped.append(f"{f.name}:{i}: {line.strip()[:90]}")
    assert not escaped, (
        "regexes over compiler output must live in scopex/_parse.py, with the component that "
        "prints them named and a verbatim sample attached:\n  " + "\n  ".join(escaped))
