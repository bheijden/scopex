"""THE SHIP CALL, PINNED. What is exported, what is deliberately not, and what "marked" means.

Three separate things get a test here, because all three have already gone wrong once in this
package's short life:

1. **The export list is a promise and it grows silently.** Five instruments were built in one
   session and eleven names appeared in ``__all__`` before anybody counted. A top-level name is a
   commitment to keep working across jax releases; a table pinned to one XLA commit cannot make
   that commitment however good it is. So the list is pinned literally, and adding to it is a
   deliberate edit to this file with a reason.

2. **``pass_timeline`` must be the VALIDATED implementation.** The old one -- subtract consecutive
   mtimes, scan for a ``.ll``, call the gap LLVM -- returned the same shape of answer with nothing
   checked. Two implementations of one instrument is how the unvalidated one comes back.

3. **A "SHIP MARKED" instrument must be branchable IN THE DATA.** Prose in a verdict string is not
   a marker. ``Coverage.checked`` is False exactly when the self-check could not run, and that is
   the difference between "verified" and "unverifiable" that ``broken is False`` cannot express.
"""

from __future__ import annotations

import importlib

import pytest

import scopex

# ══════════════════════════════════════════════════════════════════════════════════════════════
# 1. THE BAR
# ══════════════════════════════════════════════════════════════════════════════════════════════

# The five names added by the hardening round, and why each cleared a bar that `pass_source`,
# `boundary_diff` and `autotune_cost` did not: every one of them CHECKS a name that was already
# exported, or IS one of those names in validated form.
PROMOTED = {
    "Coverage": "checks pass_timings' ranking against XLA's own (cumulative:, max:, #called:)",
    "pass_conservation": "checks pass_growth / diverge's curves against the endpoint files",
    "Raw": "hands back the bytes a number was parsed from, hashed as parsed",
    "raw_step": "the same, for a pass_growth step",
    "timeline_agreement": "the only way to get a pass_timeline that has been checked at all",
}

# Built, validated, and deliberately NOT promoted. Each must stay importable from its own module --
# they are used by examples/recipes/ -- and must stay OUT of the top-level namespace.
NOT_PROMOTED = {
    "scopex.passmap": ("pass_source", "pass_sources", "PassSource", "pipelines_in",
                       "cross_check", "verify_pass_map"),
    "scopex.artifacts": ("boundary_diff", "opcode_delta", "resolve_boundary", "boundaries_in"),
    "scopex.autotune": ("autotune_cost", "autotune_report", "Autotune"),
    "scopex.fusion": ("fusion_steps", "fusion_consistency"),
    "scopex.phases": ("backend_split",),
    "scopex.emitters": ("emitter_growth",),
    "scopex.tracing": ("trace_profile",),
    "scopex.sharing": ("jaxpr_sharing",),
}


def test_export_count_did_not_drift():
    """45 before this round, 50 after. If this number moves, the ship call moved with it."""
    assert len(scopex.__all__) == 50, sorted(scopex.__all__)
    assert len(set(scopex.__all__)) == len(scopex.__all__), "a name is listed twice"


def test_every_promoted_name_is_a_check_on_something_already_exported():
    for name, why in PROMOTED.items():
        assert name in scopex.__all__, f"{name} was promoted for: {why}"
        assert hasattr(scopex, name)


@pytest.mark.parametrize("module,names", sorted(NOT_PROMOTED.items()))
def test_unpromoted_instruments_are_reachable_but_not_top_level(module, names):
    """Reachable from their own module, absent from the top level. Both halves matter.

    Absent from the top level, because a top-level name is a promise across jax releases.
    Reachable, because deleting a validated instrument to protect an export count is the other
    failure -- and `examples/recipes/` imports these by module path.
    """
    mod = importlib.import_module(module)
    for n in names:
        assert hasattr(mod, n), f"{module}.{n} disappeared -- recipes import it"
        assert n not in scopex.__all__, f"{n} was promoted without a reason in PROMOTED"


def test_the_negative_lineage_result_is_not_an_importable_package_module():
    """`scopex.lineage` measured that name-based instruction lineage is 49-62% correct at exactly
    the passes anyone would ask about, and concluded DO NOT BUILD IT. A package module whose
    headline is 96.7% and whose conclusion is "do not use this" is a trap with a docstring in front
    of it, so the argument moved to examples/recipes/why_no_instruction_lineage.py."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("scopex.lineage")


# ══════════════════════════════════════════════════════════════════════════════════════════════
# 2. ONE IMPLEMENTATION OF THE TIMELINE, AND IT IS THE CHECKED ONE
# ══════════════════════════════════════════════════════════════════════════════════════════════

def test_pass_timeline_is_the_validated_implementation():
    import scopex.artifacts
    import scopex.timeline

    assert scopex.pass_timeline is scopex.timeline.pass_timeline
    assert scopex.artifacts.pass_timeline is scopex.timeline.pass_timeline
    # The signature is what makes it checkable: `log=` is the second clock.
    import inspect
    assert "log" in inspect.signature(scopex.pass_timeline).parameters


def test_an_unchecked_timeline_says_so_rather_than_reading_like_a_checked_one(tmp_path):
    """No log supplied -> `.agreement is None` and the verdict opens UNVALIDATED. The old
    implementation returned the same intervals with nothing to distinguish the two states."""
    tl = scopex.pass_timeline(tmp_path)
    assert tl.agreement is None
    assert not tl.tail_usable
    assert "UNVALIDATED" in tl.verdict or "no HLO snapshots" in tl.verdict


def test_the_verdict_names_a_function_that_exists():
    """The UNVALIDATED verdict routes the reader to `scopex.timeline_agreement()`. That is the
    whole reason it is exported: a verdict that names a symbol which is not there is worse than no
    verdict."""
    tl = scopex.pass_timeline.__doc__ or ""
    assert "timeline_agreement" in tl
    assert callable(scopex.timeline_agreement)


# ══════════════════════════════════════════════════════════════════════════════════════════════
# 3. WHAT "MARKED" MEANS: A CALLER CAN BRANCH
# ══════════════════════════════════════════════════════════════════════════════════════════════

def _cov(**kw):
    base = {"parsed_passes": 10, "parsed_seconds": 1.0, "parsed_max_s": 0.5,
            "leaf_seconds": 0.8, "aggregate_seconds": 0.2, "unmatched_pipelines": 0,
            "backend_s": 2.0, "log_threads": 1}
    base.update(kw)
    return scopex.Coverage(base)


def test_checked_is_false_exactly_when_the_self_check_could_not_run():
    """The deficit this property exists for: if a future XLA stops printing its own totals,
    `fidelity`, `lines_lost` and `biggest_pass_lost` all go None and `broken` goes quiet. Nothing
    raises. An unverified ranking then looks identical to a verified one."""
    unchecked = _cov()                                  # no xla_pass_count
    assert unchecked.checked is False
    assert unchecked.broken is False                    # <- and this is the trap
    assert unchecked.fidelity is None
    assert "UNCHECKED" in unchecked.verdict
    assert "Coverage.checked" in unchecked.verdict

    checked = _cov(xla_pass_count=10, xla_cumulative_s=1.0, xla_max_pass_s=0.5)
    assert checked.checked is True
    assert checked.broken is False


def test_checked_is_independent_of_where_the_seconds_went():
    """`checked` must not consult `coverage`. Low coverage is a result about the compile; the two
    were confused once already and the confusion is what shipped a wrong pass name."""
    for backend in (0.001, 1000.0):
        c = _cov(backend_s=backend, xla_pass_count=10, xla_cumulative_s=1.0, xla_max_pass_s=0.5)
        assert c.checked is True


def test_over_unity_fires_on_the_gpu_autotuning_arm_that_produced_it():
    """gemm_shapes_k16, measured: leaf 18.3103 s against a backend of 18.1094 s = 1.0111, with
    fidelity 0.9996 and split_ok True. A perfect parse whose ratio still exceeded 1, because the
    autotuner compiles candidate sub-modules concurrently across 21 glog threads and concurrent
    seconds sum past wall clock. Documented as unobserved by the previous round; observed now."""
    c = _cov(leaf_seconds=18.3103, aggregate_seconds=0.0, parsed_seconds=18.3103,
             parsed_max_s=18.0, backend_s=18.1094, log_threads=21,
             xla_pass_count=10, xla_cumulative_s=18.3176, xla_max_pass_s=18.0)
    assert c.over_unity is True
    assert c.split_ok is True
    assert c.broken is False                    # the PARSE is fine; the ratio is not a fraction
    assert "concurrent" in c.verdict
    assert "21 log threads" in c.verdict
    assert _cov(xla_pass_count=10, xla_cumulative_s=1.0,
                xla_max_pass_s=0.5).over_unity is False


def test_a_missing_tolerance_does_not_turn_the_check_into_a_false_alarm():
    """FOUND BY WRITING THE TEST ABOVE. `broken` computed its fidelity threshold as
    ``max(FIDELITY_FLOOR, 1.0 - (tolerance or 0)*2)``. With `tolerance` absent that is
    ``max(0.90, 1.0)`` = 1.0, so ANY fidelity below exactly 1.0 read as PARSE BROKEN -- and
    measured fidelity is never exactly 1.0; it spans [0.9978, 1.0015] across 20 arms. Every one of
    them would have cried wolf on a Coverage built without that key.

    A check that fires on healthy data gets ignored, and takes the real firing with it. That is the
    same failure class as a check that stays silent, and it is why this is pinned.
    """
    healthy = dict(xla_pass_count=10, xla_cumulative_s=1.0, xla_max_pass_s=0.5)
    for parsed in (0.9978, 0.9996, 1.0012, 1.0015):        # the measured spread, both sides of 1
        assert _cov(parsed_seconds=parsed, leaf_seconds=parsed, aggregate_seconds=0.0,
                    **healthy).broken is False, parsed


def test_fidelity_still_fires_in_both_directions_at_the_measured_distances():
    """The slack is 10% either way and the observed spread is 0.22%, so the band is 45x wider than
    anything healthy and still catches both real defects."""
    healthy = dict(xla_pass_count=10, xla_cumulative_s=1.0, xla_max_pass_s=0.5, tolerance=0.0014)
    # BELOW: the historical min-unit parse, reconstructed on a real GPU log -> fidelity 0.0111.
    lost = _cov(parsed_seconds=0.0111, leaf_seconds=0.0111, aggregate_seconds=0.0,
                parsed_max_s=0.0111, **healthy)
    assert lost.broken is True
    assert "PARSE BROKEN" in lost.verdict
    # ABOVE: scopex cannot legitimately total MORE seconds than XLA's own running total over
    # exactly the same lines. Counting a line twice is the way that happens.
    dup = _cov(parsed_seconds=2.0, leaf_seconds=2.0, aggregate_seconds=0.0, parsed_max_s=0.5,
               **healthy)
    assert dup.broken is True


def test_the_top_band_warns_about_the_one_pass_it_cannot_discriminate():
    """convT64_dilate16 and gemm_shapes_k16 are indistinguishable to this instrument -- both ~98%
    `autotuner`, both PASS-BOUND, both fidelity ~1.0 -- and 450x apart on what the seconds are.
    The band text has to say so, because the number cannot."""
    c = _cov(leaf_seconds=1.9, backend_s=2.0, xla_pass_count=10, xla_cumulative_s=1.0,
             xla_max_pass_s=0.5)
    assert "PASS-BOUND" in c.band
    assert "autotuner" in c.band and "autotune_cost" in c.band
