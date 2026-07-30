"""Every check in :mod:`scopex.autotune`, shown FIRING on a real captured GPU autotune dump.

A check that has never been observed to fail is not evidence. Each test here degrades exactly one
part of the reader -- in the way it was actually written wrong during development, or in the way the
package's own history says it goes wrong -- and asserts the corresponding check catches it while the
healthy reader is silent on the same bytes.

The two fixtures are the whole argument for this module existing. Both produce the same headline
from ``pass_timings``: ``autotuner`` is ~98% of the compile, coverage says PASS-BOUND, fidelity
~1.0. Their causes are opposite, and only ``kernel_share`` separates them.
"""

from __future__ import annotations

import gzip
import json
import os

import pytest

from scopex import autotune as A

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def _load(name):
    with gzip.open(os.path.join(FIX, name), "rt") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def conv():
    """``convT64_dilate16`` -- the arm whose compile really is spent running kernels."""
    return _load("autotune_conv_gpu.json.gz")


@pytest.fixture(scope="module")
def gemm():
    """``gemm_shapes_k16`` -- the arm whose autotuner compiles candidates instead."""
    return _load("autotune_gemm_gpu.json.gz")


def report(blob):
    return A.autotune_report(blob["logs"], blob["results"], vlog=blob["vlog"],
                             backend_s=blob["backend_s"])


# ── the healthy readings ─────────────────────────────────────────────────────────────────────────

def test_conv_arm_is_kernel_bound_and_every_check_passes(conv):
    r = report(conv)
    assert not r.broken, r.verdict
    assert r.winners_ok and r.keys_ok and r.names_ok and r.share_ok
    assert r.argmin_ok
    # 25.79 s of candidate run_time inside a 51.86 s pass inside a 52.02 s compile.
    assert r.kernel_share == pytest.approx(0.497, abs=0.01)
    assert r.compile_share == pytest.approx(0.496, abs=0.01)
    assert r["n_candidates"] == 24 and r["n_instructions"] == 5


def test_gemm_arm_is_compile_bound_on_the_same_pass_name(gemm):
    r = report(gemm)
    assert not r.broken, r.verdict
    # The autotuner is ~98% of this compile too -- and 0.1% of it is kernel execution.
    assert r["pass_s"] / r["backend_s"] > 0.95
    assert r.kernel_share < 0.01
    assert r["n_candidates"] == 456


def test_the_two_arms_are_indistinguishable_by_pass_share_and_900x_apart_by_kernel_share(conv, gemm):
    """The claim this module is built on, as an assertion rather than a docstring."""
    c, g = report(conv), report(gemm)
    assert c["pass_s"] / c["backend_s"] > 0.95
    assert g["pass_s"] / g["backend_s"] > 0.95          # identical headline
    assert c.kernel_share / g.kernel_share > 400        # opposite cause


def test_half_the_conv_arms_autotuner_pass_is_not_accounted_for_by_the_dump(conv):
    """The residual is part of the answer and is reported, not hidden.

    18 cuDNN algorithms are benchmarked for 25.79 s inside a 51.86 s pass. The VLOG independently
    shows exactly 18 candidate sub-module compiles for `cudnn-conv.2`, so the other ~26 s is
    autotuner overhead -- compiling those 18 sub-modules, allocating and redzone-checking the
    134 MB buffers -- and NOT more kernel time. An earlier reading of this arm put kernel_share at
    98% and it was an artefact of a reused dump directory; see `autotune_cost`.
    """
    r = report(conv)
    assert r["n_log_entries"] == 5 and r["n_instructions"] == 5
    assert r.redundant_s == 0.0
    unexplained = r["pass_s"] - r["candidate_s"]
    assert unexplained == pytest.approx(26.07, abs=0.5)


# ── negative controls: each check, made to fire ──────────────────────────────────────────────────

def test_dropping_the_seconds_field_of_a_duration_is_caught(conv, monkeypatch):
    """THE ``min``-UNITS BUG IN A NEW COSTUME.

    ``run_time`` is a ``google.protobuf.Duration`` and proto3 omits whichever of ``seconds`` /
    ``nanos`` is zero. A reader that only ever saw sub-second candidates reads ``nanos`` alone and
    is right on every fast arm and catastrophically wrong on exactly the slow one -- which is the
    shape of the twelve-character regex that cost this package four arms.
    """
    healthy = report(conv)
    monkeypatch.setattr(A, "_duration_s",
                        lambda d: float((d or {}).get("nanos", 0) or 0) / 1e9
                        if isinstance(d, dict) else 0.0)
    sick = report(conv)

    # It is silent in the total: still a plausible number, just three times too small.
    assert sick["candidate_s"] < healthy["candidate_s"] / 2.5
    # And it reorders the candidates, which is what the winner check sees.
    assert sick.broken
    assert sick.winner_slowdown > 3.0            # vs 1.000-1.038 for XLA's selection policy
    assert healthy.winner_slowdown <= healthy.SELECTION_TOLERANCE
    assert not healthy.broken


def test_an_enumerating_candidate_key_is_caught_by_the_identity_check(gemm, monkeypatch):
    """The first draft of ``_key`` knew cuDNN's ``algorithm`` and the emitters' ``other`` and
    returned one ``("none",)`` for everything else, so all 25 Triton configs of an instruction
    became the same candidate. The winner check CANNOT see this -- the recorded winner collapses
    too -- which is why the identity check exists."""
    assert A.key_collisions(A.autotune_logs(gemm["logs"])) == 0

    def enumerating(msg):
        if not isinstance(msg, dict):
            return ("none",)
        a = msg.get("algorithm")
        if isinstance(a, dict):
            return ("algo", a.get("algo_id"))
        o = msg.get("other")
        if isinstance(o, dict):
            return ("other", o.get("name"))
        return ("none",)                       # every Triton and cuBLAS config lands here

    monkeypatch.setattr(A, "_key", enumerating)
    sick = report(gemm)
    assert sick["key_collisions"] > 0, "the collapse must be visible"
    assert not sick.keys_ok and sick.broken
    assert "share one identity" in sick.verdict


def test_dropping_candidates_is_caught_by_the_winner_containment_check(conv, monkeypatch):
    """Silently losing candidates keeps the report's shape and shrinks its numbers."""
    real = A.autotune_logs

    def lossy(src):
        return [a._replace(candidates=[c for c in a.candidates if "cudnn algo 48" not in c.label])
                for a in real(src)]

    monkeypatch.setattr(A, "autotune_logs", lossy)
    sick = report(conv)
    assert sick.winners_ok is False
    assert sick.broken and "not among the candidates" in sick.verdict


def test_wrong_field_numbers_in_the_instruction_proto_are_caught_by_the_name_check(conv,
                                                                                   monkeypatch):
    """The one guess in this module. If XLA renumbers ``HloInstructionProto``, the decoded strings
    stop being instruction names -- and the VLOG, which shares nothing with the proto, says so."""
    assert report(conv)["names_confirmed"] == 1.0

    def shifted(entry):
        try:
            f = A.proto_fields(A._payload(entry.get("instr")))
        except Exception:
            return ("", "")
        v = f.get(3, [b""])[0]                 # field 3 is the shape, not the name
        return (v.decode("utf-8", "replace") if isinstance(v, bytes) else "", "")

    monkeypatch.setattr(A, "instr_name", shifted)
    sick = report(conv)
    assert sick["names_confirmed"] == 0.0
    assert sick.names_ok is False and sick.broken
    assert "field numbers may have moved" in sick.verdict


def test_candidates_cannot_outlast_the_pass_that_ran_them(conv):
    r = A.autotune_report(conv["logs"], conv["results"], vlog=conv["vlog"],
                          backend_s=conv["backend_s"], pass_s=1.0)
    assert not r.share_ok and r.broken
    assert "the pass that ran them" in r.verdict


def test_disqualified_candidates_do_not_win(gemm):
    """A DISQUALIFIED Triton config has an empty ``run_time``, so it reads as 0 s -- the fastest
    thing in the list. 56 of 456 on this arm."""
    logs = A.autotune_logs(gemm["logs"])
    failed = [c for a in logs for c in a.candidates if c.failed]
    assert len(failed) == 56
    assert all(c.seconds == 0.0 for c in failed)
    for a in logs:
        assert all(not c.failed for c in a.contenders)
    assert report(gemm).argmin_ok


# ── the textproto reader this module stands on ───────────────────────────────────────────────────

@pytest.mark.parametrize("src,want", [
    ('v: "\\021"', "\x11"),      # a length prefix inside an embedded proto
    ('v: "\\000"', "\x00"),
    ('v: "\\012"', "\n"),
    ('v: "\\077"', "?"),         # last byte of the range that used to decode as three characters
    ('v: "\\100"', "@"),         # first byte that always worked
    ('v: "\\377"', chr(255)),
    ('v: "\\n"', "\n"),
    ('v: "\\x11"', "\x11"),
])
def test_octal_escapes_decode_to_one_byte(src, want):
    """``_ESCAPES`` has a ``"0"`` key for the ``\\0`` spelling of NUL, and it used to be consulted
    before the octal branch. Proto text format always writes THREE octal digits, so ``\\021`` matched
    it, emitted NUL and left ``21`` as literal text: every byte in ``\\000``-``\\077`` decoded as
    three characters instead of one. Invisible on an ASCII-only dump, fatal on an embedded ``Any``.
    """
    from scopex.fusion import parse_textproto
    got = parse_textproto(src)["v"]
    assert got == want and len(got) == 1


# ── the priority-fusion decision log ─────────────────────────────────────────────────────────────

def test_fusion_consistency_is_closed_and_matches_the_pipelines_own_snapshot():
    """A hand-built log: the causal-closure check, and the negative control for it.

    Real GPU evidence for this lives in the session notes -- ``xtile_issue``, 63 dumps, 129 steps,
    20 fusions created, 0 forward references, and an embedded module whose 184 instruction names
    equal the pipeline's own pre-pass snapshot exactly. What is pinned here is the LOGIC, because
    a 35 KB fixture per dump is not worth the bytes.
    """
    from scopex.fusion import fusion_consistency

    good = '''
      hlo_module_before_fusion: "ENTRY main {\\n  a = f32[4] parameter(0)\\n  b = f32[4] parameter(1)\\n  c = f32[4] add(a, b)\\n}"
      fusion_steps { fusion { producer_name: "a" consumer_name: "c" fusion_name: "fusion.1" } }
      fusion_steps { producer_ineligible { producer_name: "fusion.1" reason: "only bitcast users" } }
    '''
    r = fusion_consistency(__import__("scopex.fusion", fromlist=["x"]).parse_textproto(good))
    assert r["consistent"] is True
    assert r["closed"] and r["forward_references"] == []
    assert r["fusions_created"] == 1 and r["start_instructions"] == 3

    # NEGATIVE CONTROL: the same two steps in the other order. `fusion.1` is now referenced before
    # anything created it, which is what a lost or reordered step looks like.
    bad = '''
      hlo_module_before_fusion: "ENTRY main {\\n  a = f32[4] parameter(0)\\n  b = f32[4] parameter(1)\\n  c = f32[4] add(a, b)\\n}"
      fusion_steps { producer_ineligible { producer_name: "fusion.1" reason: "only bitcast users" } }
      fusion_steps { fusion { producer_name: "a" consumer_name: "c" fusion_name: "fusion.1" } }
    '''
    r2 = fusion_consistency(__import__("scopex.fusion", fromlist=["x"]).parse_textproto(bad))
    assert r2["consistent"] is False
    assert not r2["closed"]
    assert [f[2] for f in r2["forward_references"]] == ["fusion.1"]


def test_fusion_consistency_reports_none_not_true_when_it_could_not_check():
    """No embedded module means the check did not run. It must not read as a pass."""
    from scopex.fusion import fusion_consistency, parse_textproto
    r = fusion_consistency(parse_textproto(
        'fusion_steps { producer_ineligible { producer_name: "x" reason: "r" } }'))
    assert r["consistent"] is None
    assert r["has_start_module"] is False
