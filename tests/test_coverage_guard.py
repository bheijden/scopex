"""THE GUARD, SHOWN FIRING. A guard that has never been demonstrated is not a guard.

`scopex.Coverage` exists because of one specific failure: `pass_timings` returned a plausible dict
topped by `remat-pipeline: 0.1196` for a compile that took 72.5 s, because XLA printed the
autotuner's time as `1.19 min` and the parser knew only `us`, `ms` and `s`. One line of 640, and it
was 98.8% of the compile. The tool did not fail to answer. It answered with the opposite of the
truth and said nothing.

These tests reintroduce that exact parser -- a COPY, verbatim in its broken form, never imported by
the package -- and run both parsers over REAL captured logs. The point is not that the broken parser
is broken; that is known. The point is that `Coverage` SAYS SO, in three independent ways, one of
which involves no arithmetic and no unit table at all.

WHAT THE FIXTURES ARE. Raw stderr from `scopex.pass_timings`' own child process, captured on this
machine (jax/jaxlib 0.10.2, x64, RTX 4090 Laptop, nvidia-smi showing no other compute process),
stored with the `jax.monitoring` numbers from the SAME compile. Regenerate with
`tests/fixtures/README.md`.

AND ONE THING THE CAPTURE TAUGHT US, WHICH IS WHY `convT64_dilate16` IS NOT THE FIXTURE THE BRIEF
ASKED FOR. XLA switches the printed unit to `min` above 60 s. Re-measured on a QUIET GPU, that arm's
autotuner takes 50.7 s and prints `50.7 s` -- so the historical parser reads it correctly and the
bug does not fire at all. The published 72.5 s reading was taken with ten foreign GPU processes at
100% utilisation. **The same program, the same code, the same compiler: the bug is catastrophic on a
loaded machine and invisible on an idle one.** A regression test pinned to that arm would have gone
green on this box and stayed green through the entire lifetime of the bug. So the min-unit fixture
is an arm whose slowest pass clears 60 s with margin, and `convT64_dilate16` is kept alongside as
the positive control: a genuinely autotune-bound compile where coverage is near 1 and the ranking
IS the answer.
"""

from __future__ import annotations

import gzip
import json
import pathlib
import re

import pytest

from scopex import _parse
from scopex.coverage import Coverage

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    p = FIXTURES / f"{name}.json.gz"
    if not p.exists():                                                       # pragma: no cover
        pytest.skip(f"fixture {p} not present")
    return json.loads(gzip.decompress(p.read_bytes()).decode())


# ══════════════════════════════════════════════════════════════════════════════════════════════
# THE PARSER AS IT SHIPPED, BUG AND ALL
#
# Verbatim from the failure: three units, and no fallback to the parenthesised microseconds that
# XLA prints regardless of the headline unit. Nothing in `scopex` imports this. It is here so the
# guard can be tested against the thing it was built to catch rather than against a simulation of
# it -- and so that anyone who "simplifies" the unit table later meets this file.
# ══════════════════════════════════════════════════════════════════════════════════════════════
_HISTORICAL_LINE = re.compile(
    r"HLO pass:\s+(?P<name>.+?)\s+time:\s+(?P<val>[\d.]+)\s*(?P<unit>us|ms|s)")
_HISTORICAL_UNITS = {"us": 1e-6, "ms": 1e-3, "s": 1.0}


def historical_pass_timing_lines(log: str) -> list[_parse.PassTime]:
    """bug #3, reconstructed. Silently skips any line whose unit is not one of three."""
    out = []
    for m in _HISTORICAL_LINE.finditer(log):
        u = m.group("unit")
        if u not in _HISTORICAL_UNITS:                                       # pragma: no cover
            continue
        out.append(_parse.PassTime(m.group("name"),
                                   float(m.group("val")) * _HISTORICAL_UNITS[u], u, False))
    return out


def coverage_of(log: str, backend_s: float, *, parse_lines=None, split: bool = True) -> Coverage:
    """Build a :class:`Coverage` from a log the way ``pass_timings`` does.

    ``parse_lines=historical_pass_timing_lines`` swaps in the broken parser and turns the leaf split
    off, because the instrument that shipped the bug had no leaf split either -- its coverage was
    the naive sum over every line, which is what the published 0.6% figure was computed from.
    """
    parse_lines = parse_lines or _parse.pass_timing_lines
    every = parse_lines(log)
    tot = _parse.pass_log_totals(log)
    s = _parse.pass_leaf_split(log)
    if split:
        leaf, agg, n_leaf, n_agg = (sum(t.seconds for t in s.leaves),
                                    sum(t.seconds for t in s.aggregates),
                                    len(s.leaves), len(s.aggregates))
        unmatched = s.unmatched_closes
    else:
        leaf, agg, n_leaf, n_agg = sum(t.seconds for t in every), 0.0, len(every), 0
        unmatched = 0
    return Coverage(
        parsed_passes=len(every),
        parsed_seconds=sum(t.seconds for t in every),
        parsed_max_s=max((t.seconds for t in every), default=0.0),
        leaf_seconds=leaf, aggregate_seconds=agg, n_leaves=n_leaf,
        n_aggregates=n_agg, unmatched_pipelines=unmatched, log_threads=s.threads,
        xla_pass_count=tot["n_called"], xla_cumulative_s=tot["cumulative_s"],
        xla_max_pass_s=tot["max_pass_s"], tolerance=tot["tolerance"],
        counter_monotone=tot["monotone"], backend_s=backend_s, why_no_backend="")


def ranking(log: str, parse_lines) -> list[tuple[str, float]]:
    agg: dict[str, float] = {}
    for t in parse_lines(log):
        agg[t.name] = agg.get(t.name, 0.0) + t.seconds
    return sorted(agg.items(), key=lambda kv: -kv[1])


# ══════════════════════════════════════════════════════════════════════════════════════════════
# THE FIXTURE MUST BE ABLE TO EXPRESS THE BUG
# ══════════════════════════════════════════════════════════════════════════════════════════════

def test_min_unit_fixture_actually_contains_a_min_unit_line():
    """If it does not, every test below passes for the wrong reason.

    This is the assertion the original bug most needed and never had. Whether a `min` line appears
    at all depends on machine speed and load, so a fixture regenerated on a faster box can silently
    stop exercising the bug -- exactly how the bug survived. Fail loudly instead.
    """
    d = _load("min_unit_gpu")
    hits = [ln for ln in d["log"].splitlines()
            if "HLO pass: " in ln and re.search(r"time:\s+[\d.]+\s+min\b", ln)]
    assert hits, ("the min-unit fixture no longer contains a pass printed in `min`, so it cannot "
                  "reproduce the bug. Recapture on a slower/busier machine or a larger arm.")
    assert d["backend_s"] > 60.0


# ══════════════════════════════════════════════════════════════════════════════════════════════
# THE GUARD FIRES
# ══════════════════════════════════════════════════════════════════════════════════════════════

def test_broken_parser_collapses_coverage_and_the_guard_says_so():
    d = _load("min_unit_gpu")
    good = coverage_of(d["log"], d["backend_s"])
    bad = coverage_of(d["log"], d["backend_s"],
                      parse_lines=historical_pass_timing_lines, split=False)

    # The parse loses exactly the lines XLA printed in a unit it does not know.
    assert bad.lines_lost is not None and bad.lines_lost > 0
    assert bad["parsed_passes"] == good["parsed_passes"] - bad.lines_lost

    # ...and coverage collapses by orders of magnitude. This is the number that would have screamed.
    assert bad.coverage is not None and good.coverage is not None
    assert bad.coverage < 0.05, bad.coverage
    assert good.coverage > 0.5, good.coverage
    assert good.coverage / bad.coverage > 20

    # All three checks fire, and one of them involves no seconds at all.
    assert bad.broken is True
    assert bad.biggest_pass_lost is True
    assert bad.fidelity < 0.1
    assert bad.verdict.startswith("PARSE BROKEN")
    assert "LOST" in bad.verdict


def test_the_broken_parser_still_returns_a_plausible_ranking():
    """The reason a coverage number was needed at all.

    The broken parser does not crash, return empty, or look suspicious. It returns a sorted dict of
    real pass names with real seconds, and the entry on top is a real pass that really ran. Nothing
    about the ranking itself betrays that the pass holding 98% of the compile is missing from it.
    """
    d = _load("min_unit_gpu")
    bad_rank = ranking(d["log"], historical_pass_timing_lines)
    good_rank = ranking(d["log"], _parse.pass_timing_lines)

    assert len(bad_rank) > 20                       # a full, healthy-looking profile
    assert bad_rank[0][1] > 0.0
    assert bad_rank[0][0] != good_rank[0][0], (
        "the fixture's slowest pass survives the broken parser, so this fixture does not "
        "reproduce the failure mode")
    # The truth: the pass the broken ranking omits from the top is most of the compile.
    assert good_rank[0][1] / d["backend_s"] > 0.5
    assert bad_rank[0][1] / d["backend_s"] < 0.2


def test_healthy_parser_does_not_cry_wolf():
    """The other half of a guard: it must be quiet when nothing is wrong.

    Run over every fixture, because a check that fires on a correct parse is worse than no check --
    it trains the reader to ignore it.
    """
    for name in ("min_unit_gpu", "convT64_dilate16_gpu"):
        d = _load(name)
        c = coverage_of(d["log"], d["backend_s"])
        assert c.broken is False, f"{name}: {c.verdict}"
        assert c.lines_lost == 0, name
        assert 0.98 < c.fidelity < 1.02, f"{name}: fidelity {c.fidelity}"
        assert c.split_ok, name
        assert c["counter_monotone"] is True, name
        assert not c.verdict.startswith("PARSE BROKEN"), name


# ══════════════════════════════════════════════════════════════════════════════════════════════
# THE POSITIVE CONTROL: A COMPILE THE PASSES REALLY DO EXPLAIN
# ══════════════════════════════════════════════════════════════════════════════════════════════

def test_conv_autotune_arm_reads_as_pass_bound():
    """`convT64_dilate16` is where every structural instrument in this package returns a null --
    identical opcode histograms, identical PTX counts, identical thunk counts -- because the cost is
    running candidate cuDNN kernels on real buffers, not transforming IR. The pass timer is the one
    instrument that sees it, and coverage is what licenses believing it."""
    d = _load("convT64_dilate16_gpu")
    c = coverage_of(d["log"], d["backend_s"])
    assert c.coverage > 0.9, c.coverage
    assert c.band.startswith("PASS-BOUND")
    top = ranking(d["log"], _parse.pass_timing_lines)[0]
    assert top[0] == "autotuner"
    assert top[1] / d["backend_s"] > 0.9


def test_leaf_split_adds_back_up_on_every_fixture():
    """The identity that makes the leaf/aggregate split checkable instead of merely plausible."""
    for name in ("min_unit_gpu", "convT64_dilate16_gpu"):
        d = _load(name)
        s = _parse.pass_leaf_split(d["log"])
        every = _parse.pass_timing_lines(d["log"])
        assert len(s.leaves) + len(s.aggregates) == len(every), name
        assert s.unmatched_closes == 0, name
        assert abs(sum(t.seconds for t in s.leaves) + sum(t.seconds for t in s.aggregates)
                   - sum(t.seconds for t in every)) < 1e-9, name


def test_xla_own_totals_agree_with_the_parse_on_every_fixture():
    """The cross-check itself, on real logs: XLA's `#called`, `cumulative:` and `max:` against
    scopex's count, sum and maximum over the same lines."""
    for name in ("min_unit_gpu", "convT64_dilate16_gpu"):
        d = _load(name)
        tot = _parse.pass_log_totals(d["log"])
        ts = _parse.pass_timing_lines(d["log"])
        assert tot["n_called"] == len(ts), name
        assert abs(tot["cumulative_s"] - sum(t.seconds for t in ts)) <= 0.02 * tot["cumulative_s"]
        assert abs(tot["max_pass_s"] - max(t.seconds for t in ts)) <= 0.02 * tot["max_pass_s"]
