"""The raw-artifact handover and the pass -> source pointer.

Neither of these produces a number, so neither can be checked by "is the number right". They are
checked by the two things that CAN go wrong with a pointer: it can point at the wrong file, and it
can go on pointing after the thing it points at has moved. Both have happened during this build:
an earlier generator mapped ``float_normalization`` into ``third_party/tsl``, and a per-platform
row lost its implementation file because two dicts keyed differently were indexed with each
other's keys.
"""

from __future__ import annotations

import pathlib

import pytest

from scopex import _parse, passmap
from scopex.raw import raw_of

# ══════════════════════════════════════════════════════════════════════════════════════════════
# THE HANDOVER
# ══════════════════════════════════════════════════════════════════════════════════════════════


def test_raw_round_trips_and_verifies(tmp_path):
    p = tmp_path / "sample.vlog"
    p.write_text(_parse.SAMPLE_PASS_LOG)
    r = raw_of(p, "vlog", witness=r"HLO pass:\s",
               parsed_count=len(_parse.pass_timing_lines(_parse.SAMPLE_PASS_LOG)))
    assert r.witness_count == 4          # the frozen sample has four `HLO pass:` lines
    assert r.text() == _parse.SAMPLE_PASS_LOG
    v = r.verify()
    assert v["ok"], v["problems"]


def test_verify_catches_the_file_changing_under_the_numbers(tmp_path):
    """The one failure the handover exists to make impossible: text that is not what was parsed."""
    p = tmp_path / "sample.vlog"
    p.write_text(_parse.SAMPLE_PASS_LOG)
    r = raw_of(p, "vlog", witness=r"HLO pass:\s", parsed_count=4)
    p.write_text(_parse.SAMPLE_PASS_LOG + "I0729 x hlo_pass_pipeline.cc:176] HLO pass: late "
                                          "time: 9 s (9000000 us)\n")
    v = r.verify()
    assert not v["ok"]
    assert any("sha256 changed" in x for x in v["problems"]), v["problems"]


def test_verify_reports_a_vanished_artifact_rather_than_raising(tmp_path):
    p = tmp_path / "gone.vlog"
    p.write_text("x\n")
    r = raw_of(p, "vlog")
    p.unlink()
    v = r.verify()
    assert not v["ok"] and "is gone" in v["problems"][0]


def test_verify_catches_an_undercounting_parse(tmp_path):
    """A parse that returns fewer results than the witness count is exactly bug #3's shape."""
    p = tmp_path / "sample.vlog"
    p.write_text(_parse.SAMPLE_PASS_LOG)
    r = raw_of(p, "vlog", witness=r"HLO pass:\s", parsed_count=3)   # pretend one line was dropped
    v = r.verify()
    assert not v["ok"]
    assert any("visibly contains" in x or "occurs 4 times" in x for x in v["problems"]), \
        v["problems"]


def test_grep_finds_the_line_that_broke_the_parser(tmp_path):
    """`min` is the unit that once hid 98.8% of a compile. A user must be able to look for it."""
    p = tmp_path / "sample.vlog"
    p.write_text(_parse.SAMPLE_PASS_LOG)
    r = raw_of(p, "vlog")
    hits = r.grep(r"time: [\d.]+ min")
    assert len(hits) == 1 and "autotuner" in hits[0][1]


def test_raw_is_small_no_matter_how_big_the_artifact(tmp_path):
    """The design claim: a handle costs bytes, not megabytes. `pass_growth` reads 78 MB of
    snapshots on a modest program, and holding them was never an option."""
    p = tmp_path / "big.txt"
    p.write_text(("x" * 63 + "\n") * (1 << 16))          # 4 MB, 65,536 lines
    r = raw_of(p, "hlo-snapshot")
    assert r.size_bytes == (1 << 16) * 64
    assert len(repr(r)) < 400
    # head/tail/grep must not materialise the file: each returns bounded text from a lazy read.
    assert len(r.head(3)) < 300 and len(r.tail(3)) < 300
    assert len(r.grep("^x", limit=5)) == 5


# ══════════════════════════════════════════════════════════════════════════════════════════════
# THE POINTER
# ══════════════════════════════════════════════════════════════════════════════════════════════

# Names whose FILE cannot be guessed from the name. These are the rows that justify the table --
# 30 of 213 do not echo their file name -- so they are the rows pinned against a regeneration
# that silently starts guessing.
UNGUESSABLE = {
    "triton-gemm-rewriter": "xla/backends/gpu/transforms/gemm_fusion",
    "rename-instructions": "xla/backends/gpu/transforms/add_tracking_suffix_to_instruction_names",
    "cse_barrier_expander": "xla/hlo/transforms/expanders/optimization_barrier_expander",
    "associative-scan-rewriter": "xla/hlo/transforms/simplifiers/reduce_window_rewriter",
    "permutation_sort_simplifier": "xla/hlo/transforms/expanders/permutation_sort_expander",
    "algsimp": "xla/hlo/transforms/simplifiers/algebraic_simplifier",
    "topk-decomposer": "xla/service/topk_rewriter",
    "convolution-canonicalization": "xla/service/cpu/conv_canonicalization",
}


@pytest.mark.parametrize("name,stem", sorted(UNGUESSABLE.items()))
def test_unguessable_names_point_where_the_source_says(name, stem):
    s = passmap.pass_source(name)
    assert s is not None, f"{name} fell out of the table"
    assert s.file.startswith(stem), (name, s.file)
    assert f'"{name}"' in s.source_line or name[:12] in s.source_line


def test_the_brief_s_own_example():
    s = passmap.pass_source("copy-insertion")
    assert s.read == "xla/service/copy_insertion.cc"
    assert s.kind == "pass"


def test_unknown_name_returns_none_and_never_guesses():
    assert passmap.pass_source("no-such-pass-ever") is None
    assert passmap.pass_source("copy_insertion") is None      # near-miss spelling, still None
    assert passmap.pass_source("") is None


def test_pipelines_are_flagged_as_not_passes():
    s = passmap.pass_source("simplification", platform="cpu")
    assert s.kind == "pipeline"
    assert "cpu_compiler.cc" in s.file
    assert "NOT A PASS" in str(s)
    # and the pipeline rows are exactly the ones with no implementation file to open
    assert s.impl is None


def test_platform_split_rows_keep_their_implementation_file():
    """The bug this pins: `impl` is keyed by BACKEND and `where` by BACKEND, and an earlier
    version indexed one with a path taken from the other, so the CPU arm silently lost its .cc."""
    gpu = passmap.pass_source("fusion-wrapper", platform="gpu")
    cpu = passmap.pass_source("fusion-wrapper", platform="cpu")
    assert gpu.file != cpu.file
    assert gpu.impl and gpu.impl.endswith(".cc") and "backends/gpu" in gpu.impl
    assert cpu.impl and cpu.impl.endswith(".cc") and "service/cpu" in cpu.impl


def test_asking_without_a_platform_says_it_is_backend_specific():
    s = passmap.pass_source("fusion-wrapper")
    assert s.ambiguous_platforms and set(s.ambiguous_platforms) == {"cpu", "gpu"}
    assert "BACKEND-SPECIFIC" in str(s)


def test_same_name_wrapper_is_reported_and_is_a_gpu_fact():
    gpu = passmap.pass_source("sanitize-constant-names", platform="gpu")
    assert gpu.wrapper_pipeline and gpu.wrapper_pipeline[0].endswith("gpu_compiler.cc")
    assert "counts it twice" in str(gpu)
    # The wrappers are all built in gpu_compiler.cc, so a CPU answer must not carry the warning.
    cpu = passmap.pass_source("sanitize-constant-names", platform="cpu")
    assert cpu.wrapper_pipeline is None


def test_no_row_points_where_a_wrong_pointer_was_actually_produced():
    """Two families of wrong pointer have really been generated by this project's own tooling:
    into third_party (a StrCat prefix collision) and into runtime/ (a Thunk's name(), not a
    pass's). Neither may ever appear in a shipped row."""
    for name, (where, impl, *_rest) in passmap.PASSES.items():
        files = [f for f, _ in (list(where.values()) if isinstance(where, dict) else [where])]
        if isinstance(impl, dict):
            files += [v for v in impl.values() if v]
        elif impl:
            files.append(impl)
        for f in files:
            assert not f.startswith("third_party"), (name, f)
            assert "/runtime/" not in f, (name, f)
            assert f.endswith((".h", ".cc")), (name, f)


def test_every_row_is_well_formed():
    for name, row in passmap.PASSES.items():
        where, impl, kind, plats, wrap, ev = row
        assert kind in ("pass", "pipeline"), (name, kind)
        assert plats and set(plats) <= {"cpu", "gpu"}, (name, plats)
        places = list(where.values()) if isinstance(where, dict) else [where]
        for f, ln in places:
            assert isinstance(ln, int) and ln > 0, (name, f, ln)
        assert ev.strip(), name
        if wrap:
            assert wrap[0].endswith(".cc") and isinstance(wrap[1], int)


def test_pass_sources_preserves_order_and_keeps_unresolvable_names():
    fake = {"passes": {"copy-insertion": 1.0, "not-a-real-pass": 0.5, "algsimp": 0.25}}
    got = passmap.pass_sources(fake, platform="cpu")
    assert [n for n, _, _ in got] == ["copy-insertion", "not-a-real-pass", "algsimp"]
    assert got[1][2] is None            # kept, not dropped
    assert got[0][2].read.endswith("copy_insertion.cc")


def test_pipelines_in_reports_what_is_not_a_pass():
    fake = {"passes": {"simplification": 1.0, "copy-insertion": 0.5, "zzz-unknown": 0.1}}
    r = passmap.pipelines_in(fake, platform="cpu")
    assert "simplification" in r["pipelines"]
    assert "copy-insertion" not in r["pipelines"]
    assert r["unmapped"] == ["zzz-unknown"]


def test_verify_pass_map_on_a_tree_that_is_not_xla_reports_rather_than_raises(tmp_path):
    r = passmap.verify_pass_map(tmp_path, limit=3)
    assert not r["ok"] and r["confirmed"] == 0
    assert all("is not in this tree" in p for p in r["problems"])
    assert r["commit_expected"] == passmap.XLA_COMMIT


def test_verify_pass_map_confirms_a_row_when_the_file_is_there(tmp_path):
    """The check is not "the file exists" but "the NAME is still at that line" -- the failure a
    file-existence check cannot see is a row that has drifted onto somebody else's code."""
    name = "copy-insertion"
    s = passmap.pass_source(name)
    f = tmp_path / s.file
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("\n" * (s.line - 1) + s.source_line + "\n")
    r = passmap.verify_pass_map(tmp_path, limit=None)
    assert any(p.startswith(name + ":") for p in r["problems"]) is False
    # and a row whose line has drifted is caught
    f.write_text("\n" * (s.line - 1) + "// something else entirely\n")
    r2 = passmap.verify_pass_map(tmp_path)
    assert any(p.startswith(name + ":") and "no longer contains" in p for p in r2["problems"])


def test_cross_check_needs_the_raw_log_and_says_so_when_absent():
    r = passmap.cross_check({"passes": {"copy-insertion": 1.0}})
    assert not r["ok"] and "no raw log" in r["why"]


def test_the_table_and_the_generated_data_agree_on_the_commit():
    assert len(passmap.XLA_COMMIT) == 40
    assert len(passmap.XLA_SHA256) == 64
    assert passmap.unmapped() == {} or all(
        "reason" in v for v in passmap.unmapped().values())


def test_tools_that_generated_the_table_are_shipped():
    """The table is only checkable because the thing that built it is readable."""
    root = pathlib.Path(__file__).resolve().parent.parent
    assert (root / "tools" / "build_passmap.py").is_file()
    assert (root / "tools" / "probe_passes.py").is_file()
    assert (root / "tools" / "pass_names.json").is_file()
