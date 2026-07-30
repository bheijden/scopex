"""`pass_conservation` and `boundary_diff`: does the check FAIL when it should?

An instrument that reports "consistent" is worth exactly as much as the faults it was shown to
catch. Every fault injected below is one this package actually shipped, or one a real dump actually
contains:

  * the ``\\S+`` shape group that could not match a TUPLE shape, which undercounted every
    ``while`` / ``call`` / ``custom-call`` by 31.8% and biased every curve toward the instructions a
    control-flow pathology is made of;
  * a snapshot the filename grammar cannot read, which drops a pass boundary out of the curve and
    credits its work to the neighbour -- REAL, and found by this check: XLA writes copy-insertion's
    three intra-pass stages with no ``.before_`` field and the shipped grammar dropped all three on
    every CPU compile;
  * a curve that stops before the compile does, on which every internal check still passes.

These build dump directories from hand-written HLO text and compile nothing. The end-to-end
validation is `bisect_m94` vs its m=96 control (see the module docstring in `scopex/artifacts.py`),
which is 969 MB and ninety seconds of CPU and does not belong in a unit test.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from scopex import _parse
from scopex.artifacts import (
    PassStep,
    boundary_diff,
    diverge,
    pass_conservation,
    pass_growth,
    resolve_boundary,
)

STEM = "module_0000.jit_probe"


def _hlo(n_add: int, *, n_tuple: int = 0) -> str:
    """A parseable module with a known instruction count: 3 + n_add + 2 * n_tuple."""
    body = "\n".join(f"  %a.{i} = f32[4]{{0}} add(%p, %p)" for i in range(n_add))
    tup = "\n".join(f"  %t.{i} = (f32[4]{{0}}, s32[]) tuple(%p, %c0)\n"
                    f"  %g.{i} = f32[4]{{0}} get-tuple-element(%t.{i}), index=0"
                    for i in range(n_tuple))
    return ("HloModule m\n\nENTRY %main (p: f32[4]) -> f32[4] {\n"
            "  %p = f32[4]{0} parameter(0)\n  %c0 = s32[] constant(0)\n"
            + (body + "\n" if body else "") + (tup + "\n" if tup else "")
            + "  ROOT %r = f32[4]{0} add(%p, %p)\n}\n")


def _dump(tmp_path, counts, *, stem=STEM, n_tuple=0, before=None, after=None, name=None):
    """A dump directory whose per-pass curve is exactly ``counts`` add-instructions long."""
    d = tmp_path / (name or "dump")
    d.mkdir(exist_ok=True)
    for i, n in enumerate(counts):
        (d / f"{stem}.{i:04d}.pipe.after_pass{i}.before_pass{i + 1}.txt").write_text(
            _hlo(n, n_tuple=n_tuple))
    (d / f"{stem}.before_optimizations.txt").write_text(
        _hlo(counts[0] if before is None else before, n_tuple=n_tuple))
    (d / f"{stem}.cpu_after_optimizations.txt").write_text(
        _hlo(counts[-1] if after is None else after, n_tuple=n_tuple))
    return str(d)


# ── the clean case, so the faults below mean something ──────────────────────────────────────────

def test_a_well_formed_dump_conserves(tmp_path):
    r = pass_conservation(_dump(tmp_path, [1, 5, 5, 40]))
    assert r["counting_consistent"] and r["covers_whole_compile"] and r["complaints"] == []
    assert r["anchors"]["head_gap"] == 0 and r["anchors"]["tail_gap"] == 0
    assert r["coverage"]["fraction"] == 1.0
    assert r["chain"]["residual"] == 0
    assert r["counters"]["disagreements"] == 0
    assert r["counters"]["cross_checked"] == 4          # every snapshot read by BOTH counters
    assert r["indices"]["missing"] == []


# ── fault 1: the counter itself ─────────────────────────────────────────────────────────────────

def test_the_tuple_shape_regex_is_caught_by_the_second_counter(tmp_path, monkeypatch):
    """THE bug this check exists for. `\\S+` cannot match `(f32[4]{0}, s32[])`, so the line counter
    silently drops every tuple-shaped instruction -- and the curve it builds looks perfectly fine.
    """
    d = _dump(tmp_path, [1, 5, 40], n_tuple=4)
    assert pass_conservation(d)["counting_consistent"]                     # clean beforehand

    monkeypatch.setattr(_parse, "_INSTR", re.compile(
        r"^\s*(?:ROOT\s+)?%?(?P<name>[\w.\-]+)\s*=\s*(?P<shape>\S+)\s+(?P<opcode>[a-z][\w-]*)\("))
    r = pass_conservation(d)
    assert not r["counting_consistent"]
    assert r["counters"]["disagreements"] == 3
    pass_name, native, regex, shortfall = r["counters"]["worst"]
    assert regex < native and shortfall > 0
    assert "tuple" in " ".join(r["complaints"])
    # and it is the TUPLE instructions that went missing, all four of them, in every snapshot
    assert all(s.instrs - s.instrs_regex == 4 for s in pass_growth(d))


def test_a_uniform_undercount_is_invisible_to_every_other_check(tmp_path, monkeypatch):
    """Why the second counter is not redundant with the endpoints: a counter that is wrong the SAME
    WAY everywhere leaves the anchors, the chain and the coverage all perfectly satisfied."""
    d = _dump(tmp_path, [1, 5, 40], n_tuple=4)
    monkeypatch.setattr(_parse, "_INSTR", re.compile(
        r"^\s*(?:ROOT\s+)?%?(?P<name>[\w.\-]+)\s*=\s*(?P<shape>\S+)\s+(?P<opcode>[a-z][\w-]*)\("))
    r = pass_conservation(d)
    assert r["anchors"]["head_gap"] == 0                # anchors: fine
    assert r["chain"]["residual"] == 0                  # chain: fine
    assert r["coverage"]["fraction"] == 1.0             # coverage: fine
    assert r["counters"]["disagreements"] == 3          # only this one sees it
    assert not r["counting_consistent"]


# ── fault 2: a snapshot that does not reach the curve ───────────────────────────────────────────

def test_a_missing_snapshot_index_is_reported(tmp_path):
    d = pathlib.Path(_dump(tmp_path, [1, 5, 9, 40]))
    victim = next(p for p in d.iterdir() if ".0002." in p.name)
    victim.rename(d / victim.name.replace(".0002.", ".0002x."))
    r = pass_conservation(str(d))
    assert not r["counting_consistent"]
    assert r["indices"]["missing"] == [2]
    assert "uncounted" in " ".join(r["complaints"])


def test_the_copy_insertion_stage_snapshots_are_readable():
    """REGRESSION. XLA writes these with an `after_` field and NO `.before_` field; the shipped
    grammar returned None for all three, so `pass_growth` skipped three consecutive boundaries on
    every CPU compile and no view said so."""
    m = _parse.dump_snapshot_name(
        "module_0004.jit_f.0019.copy-insertion.after_adding_copies_to_resolve_interference.txt")
    assert m is not None
    assert m["index"] == 19 and m["pipeline"] == "copy-insertion"
    assert m["after_pass"] == "adding_copies_to_resolve_interference"
    assert m["before_pass"] == ""
    # ... and the ordinary two-field form still splits the same way it always did
    n = _parse.dump_snapshot_name(
        "module_0004.jit_f.0009.HLO_passes.after_layout-assignment.before_sub-byte-size-setter.txt")
    assert n["after_pass"] == "layout-assignment" and n["before_pass"] == "sub-byte-size-setter"
    # ... and nothing that is not a snapshot has started parsing as one
    for other in ("module_0004.jit_f.cpu_after_optimizations.txt",
                  "module_0004.jit_f.before_optimizations.txt",
                  "module_0004.jit_f.obj-file.wrapped_tanh.o"):
        assert _parse.dump_snapshot_name(other) is None


def test_stage_snapshots_get_distinct_names(tmp_path):
    """Three snapshots that all lack a `before_pass` must not collapse to one name -- `diverge`
    keys its lockstep walk on the name."""
    d = tmp_path / "stages"
    d.mkdir()
    for i, stage in enumerate(("adding_copies", "removing_copies", "special_case_copies")):
        (d / f"{STEM}.{i:04d}.copy-insertion.after_{stage}.txt").write_text(_hlo(i))
    names = [s.name for s in pass_growth(str(d))]
    assert names == ["copy-insertion/after_adding_copies",
                     "copy-insertion/after_removing_copies",
                     "copy-insertion/after_special_case_copies"]


# ── fault 3: a curve that does not span the compile ─────────────────────────────────────────────

def test_a_truncated_curve_passes_every_internal_check_and_fails_coverage(tmp_path):
    """The failure no self-consistency check can see. Counting is perfect; the curve describes 10%
    of the compile. On `bisect_m94` truncating one pass before `fusion` gives 0.8%."""
    r = pass_conservation(_dump(tmp_path, [1, 3, 5], after=41))
    assert r["counting_consistent"]                    # the numbers are the numbers
    assert r["chain"]["residual"] == 0
    assert r["counters"]["disagreements"] == 0
    assert not r["covers_whole_compile"]               # and they cover a tenth of the compile
    assert r["coverage"]["fraction"] == pytest.approx(4 / 40)
    assert r["anchors"]["tail_gap"] == 36
    assert "localisation within" in " ".join(r["complaints"])


def test_coverage_and_counting_are_separate_flags(tmp_path):
    """A dump taken with a narrow `passes=` regex legitimately covers a slice. If that tripped the
    counting flag, the flag would fail on ordinary use and get ignored."""
    r = pass_conservation(_dump(tmp_path, [10, 12], before=1, after=99))
    assert r["counting_consistent"] is True
    assert r["covers_whole_compile"] is False


def test_the_head_anchor_catches_a_curve_that_starts_late(tmp_path):
    r = pass_conservation(_dump(tmp_path, [10, 20], before=1))
    assert r["counting_consistent"]                    # the counting is fine
    assert not r["covers_whole_compile"]               # the curve is not
    assert r["anchors"]["head_gap"] == 9
    assert "does not start where the compile did" in " ".join(r["complaints"])


# ── the check ships WITH the answer ─────────────────────────────────────────────────────────────

def test_diverge_carries_the_conservation_of_both_arms(tmp_path):
    case = _dump(tmp_path, [1, 5, 400], name="case")
    ctrl = _dump(tmp_path, [1, 5, 9], name="control")
    d = diverge(case, ctrl)
    assert d["diverges_at"]["pass"] == "pipe/pass3"
    assert d["conserves"] is True and d["covers_whole_compile"] is True
    assert set(d["conservation"]) == {"case", "control"}
    assert d["complaints"] == []

    # break ONE arm and the answer must say so while still coming back
    victim = next(p for p in pathlib.Path(case).iterdir() if ".0001." in p.name)
    victim.rename(pathlib.Path(case) / victim.name.replace(".0001.", ".0001x."))
    d = diverge(case, ctrl)
    assert d["diverges_at"] is not None                # still answers
    assert d["conserves"] is False                     # and still says it should not be believed
    assert any(m.startswith("case:") for m in d["complaints"])


def test_conservation_accepts_precomputed_steps(tmp_path):
    """Parsing a 969 MB dump twice is a minute of wall time; `diverge` must not spend it."""
    d = _dump(tmp_path, [1, 5, 40])
    steps = pass_growth(d)
    assert pass_conservation(d, steps=steps) == pass_conservation(d)
    assert not pass_conservation(d, steps=steps[:2])["covers_whole_compile"]


# ── boundary_diff ───────────────────────────────────────────────────────────────────────────────

def test_boundary_diff_totals_agree_four_ways(tmp_path):
    """The opcode census, the computation sizes and the arity histogram are three partitions of one
    population; the line pattern is a fourth, independent reading. All four or the report is short.
    """
    a = _dump(tmp_path, [1, 5, 40], n_tuple=3, name="case")
    b = _dump(tmp_path, [1, 5, 9], name="control")
    r = boundary_diff(a, b, at="after")
    for arm in ("case", "control"):
        sc = r["self_check"][arm]
        assert sc["agree"], sc
        assert len({sc["instructions"], sc["opcode_census"], sc["computation_sizes"],
                    sc["arity_histogram"], sc["line_pattern"]}) == 1
    assert r["instrs"]["case"] == 3 + 40 + 6
    assert r["instrs"]["control"] == 3 + 9


def test_boundary_diff_names_what_appeared_and_what_vanished(tmp_path):
    """The shape of the real finding: `slice` 46,268 vs 570 while `dynamic-slice` goes 294 -> 0."""
    a = _dump(tmp_path, [1, 40], n_tuple=3, name="case")       # has tuple / get-tuple-element
    b = _dump(tmp_path, [1, 9], name="control")                # has neither
    r = boundary_diff(a, b, at="after")
    appeared = dict(r["opcodes"]["appeared"])
    assert appeared == {"tuple": 3, "get-tuple-element": 3}
    assert r["opcodes"]["vanished"] == []
    top = dict((k, (ca, co)) for k, ca, co, _ in r["opcodes"]["delta"])
    assert top["add"] == (41, 10)
    assert r["lineage"] is None
    assert any("renames freely and records no lineage" in c for c in r["caveats"])


def test_boundary_diff_reports_operand_arity(tmp_path):
    a = _dump(tmp_path, [1, 5], n_tuple=2, name="case")
    b = _dump(tmp_path, [1, 5], name="control")
    r = boundary_diff(a, b, at="after")
    assert r["arity"]["case_max"] == 2 and r["arity"]["control_max"] == 2
    # every instruction lands in exactly one bucket, in both arms
    assert sum(ca for _, ca, _, _ in r["arity"]["histogram"]) == r["instrs"]["case"]
    assert sum(co for _, _, co, _ in r["arity"]["histogram"]) == r["instrs"]["control"]
    assert r["arity"]["operand_edges"]["case"] > r["arity"]["operand_edges"]["control"]


def test_boundary_diff_proves_both_arms_were_read_at_the_same_pass(tmp_path):
    """A cross-arm delta read at two different pipeline points is not a delta. The resolution is
    returned rather than assumed, and a mismatch is the first caveat."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    for d, passes in ((a, ("fusion", "cse")), (b, ("cse", "fusion"))):
        d.mkdir()
        for i, p in enumerate(passes):
            (d / f"{STEM}.{i:04d}.pipe.after_{p}.before_next{i}.txt").write_text(_hlo(i + 1))
        (d / f"{STEM}.before_optimizations.txt").write_text(_hlo(1))
        (d / f"{STEM}.cpu_after_optimizations.txt").write_text(_hlo(2))
    r = boundary_diff(str(a), str(b), at="next0")
    assert r["resolved"]["case"]["after_pass"] == "fusion"
    assert r["resolved"]["control"]["after_pass"] == "cse"
    assert "DIFFERENT PASSES" in r["caveats"][0]


def test_resolve_boundary_says_which_pass_and_how(tmp_path):
    """A snapshot is `after_A.before_B`; asking for B gives you the module AFTER A. The returned
    dict names both, so the caller cannot mistake one for the other."""
    d = _dump(tmp_path, [1, 5, 9])
    r = resolve_boundary(d, "pass2")
    assert r["after_pass"] == "pass1" and r["before_pass"] == "pass2"
    assert r["how"] in ("exact-name", "substring-of-name")
    assert resolve_boundary(d, "after")["how"] == "optimization-boundary"
    with pytest.raises(KeyError, match="no pass boundary matching"):
        resolve_boundary(d, "a-pass-that-does-not-exist")


def test_generated_computation_names_do_not_swallow_the_programs_own():
    """The name lists in `boundary_diff` are discounted by this predicate, so an over-broad version
    would tell the reader to ignore the computations that came from their code."""
    gen = _parse.generated_computation_name
    assert all(map(gen, ("fused_computation", "fused_computation.174", "compare_select_fusion.1",
                         "region_3.8.clone", "wide.region_2.11.clone",
                         "wrapped_reduce-window_computation")))
    assert not any(map(gen, ("main.6", "region_0.1", "region_3.8", "main")))


def test_pass_step_stays_backward_compatible():
    """`instrs_regex` is appended LAST and defaults, so existing positional construction still
    works and `.how` has not moved."""
    s = PassStep(0, "pipe", "a", "b", 10, 2, "/x", 1.0)
    assert s.how == "native" and s.instrs_regex == -1 and s.name == "pipe/b"


def test_diverge_names_the_pass_that_caused_the_jump_not_the_next_one(tmp_path):
    """A snapshot is `after_A.before_B`; the jump INTO it was done by A. Reporting B -- which is
    what `PassStep.name` and `diverges_at["pass"]` say -- names an innocent pass downstream, and it
    is also the wrong string to hand to `boundary_diff`."""
    case = _dump(tmp_path, [1, 2, 400, 400], name="case")
    ctrl = _dump(tmp_path, [1, 2, 4, 4], name="control")
    d = diverge(case, ctrl)
    b = d["biggest_step"]
    assert b["caused_by"] == "pass2"                    # the pass that ran
    assert b["snapshot_named"] == "pipe/pass3"          # the pass that had not run yet
    assert (b["before"], b["after"]) == (5, 403)
    assert b["control_same_pass"] == {"before": 5, "after": 7, "ratio": 1.4}
    # and `boundary` is directly usable: it must resolve, in BOTH arms, to a snapshot after pass2
    for arm in (case, ctrl):
        assert resolve_boundary(arm, b["boundary"])["after_pass"] == "pass2"
    assert boundary_diff(case, ctrl, at=b["boundary"])["instrs"] == {
        "case": 403, "control": 7, "ratio": 57.57}
