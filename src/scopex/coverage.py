"""COVERAGE -- what fraction of its parent does a timing view actually explain?

Every timing view in this package reports a number that is a PART of some larger number, and until
now none of them said which part or how big a part. That gap is not cosmetic. It is where this
package's worst bug lived undetected: ``pass_timings`` returned a plausible ranking topped by
``remat-pipeline: 0.1196`` for a compile that took 72.5 s, and nothing in the return value could
have told anybody that 0.12 s of a 72.5 s compile is not an explanation of anything.

TWO RATIOS, AND CONFUSING THEM IS ITS OWN FAILURE MODE
======================================================

They have different denominators, different normal ranges, and opposite meanings when low.

``fidelity``   = ``sum(parsed pass seconds) / XLA's own cumulative total``

    A SELF-CHECK. XLA prints its own running total on every pass line -- ``(cumulative: 3.5 ms,
    max: 236 us, #called: 384)`` -- so the compiler has already done the arithmetic scopex is
    doing, over exactly the same lines. The two must agree. Measured on a trivial CPU compile:
    parsed 384 lines summing to 3.496 ms against XLA's own 384 and 3.5 ms.

    **Anything other than ~1.0 means the parser is broken.** There is no program for which this
    number is legitimately 0.5. It does not depend on the backend, the program, the machine load,
    or what the compiler spends its time on.

``coverage``   = ``sum(LEAF pass seconds) / jax.monitoring's backend_compile_duration``

    A MEASUREMENT, not a check, and its whole range is meaningful. HLO passes are one part of what
    the backend compiler does; emitters, LLVM, autotuning kernels on real buffers and linking are
    not passes and are not in the pass log at all. Measured across this package's corpus, this
    number runs from 0.0002 to about 1.0 and every value in that range is a true statement about
    where the seconds went.

    LEAF passes, not every ``time:`` line, because XLA registers some pipelines AS passes and prints
    an aggregate line alongside the passes it contains. Summing every line double-counts, and by
    enough to matter: the naive sum reads 187% of the backend on ``adconst_idx_2p22`` and
    ``dusfold_sum_200``. ``fidelity`` above deliberately keeps the NAIVE sum, because XLA's own
    ``cumulative:`` is a naive running total too and the check only works if both sides count the
    same things. Both numbers are carried: ``leaf_seconds`` and ``aggregate_seconds``, and their sum
    is asserted equal to ``parsed_seconds`` on every call.

    **A low value here is a RESULT, not a bug.** On a trivial CPU program most of the backend is
    outside the HLO passes and always was. Reading that as a broken instrument sends you to fix a
    parser that is working; reading a broken parser as "the time is elsewhere" sends you to a
    profiler for a phase that was never slow. The two ratios exist so that you never have to guess
    which one you are looking at.

WHY THE SELF-CHECK IS NOT REDUNDANT WITH THE PARSERS
====================================================

``_parse.expect`` already counts ``HLO pass: `` literals in the log and refuses to return fewer
results than that. This is a different check with a different failure mode: it uses XLA's OWN
counter and XLA's OWN sum, so it survives a log whose lines were mangled, interleaved or partially
lost, and -- the part that matters -- it reports the damage IN SECONDS. ``expect`` can say "you
dropped one line of 640". Only this can say "the line you dropped was 98.8% of the compile".

The count check is also the only one in the package that is entirely unit-free. The historical bug
was a unit table with three entries; a guard built on the same unit table shares its blind spot.
``639 != 640`` does not.

AND THE ONE THING TO BRANCH ON BEFORE EITHER
============================================

``Coverage.checked``. All three of ``fidelity``, ``lines_lost`` and ``biggest_pass_lost`` rest on
XLA continuing to print ``(cumulative:, max:, #called:)``. If a future XLA stops printing them the
guard does not fail -- it goes SILENT, ``broken`` stays False, and an unverified ranking becomes
indistinguishable from a verified one. ``checked`` is False exactly then. It is the reason this
instrument ships marked rather than plain: a caller can branch on it, and prose in ``verdict``
cannot be branched on.
"""

from __future__ import annotations

__all__ = ["Coverage", "FIDELITY_FLOOR", "COVERAGE_BANDS"]

# ══════════════════════════════════════════════════════════════════════════════════════════════
# THE TWO THRESHOLDS, AND THE MEASUREMENTS THEY COME FROM
#
# Fifteen arms, each a cold compile in its own subprocess, serial, on an idle machine
# (jax/jaxlib 0.10.2, x64; CPU rows JAX_PLATFORMS=cpu, GPU rows RTX 4090 Laptop with nvidia-smi
# showing no other compute process). `naive` is what summing every `time:` line would have given.
#
#   arm                       platform   coverage    naive   fidelity
#   ----------------------------------------------------------------
#   gatherchain2d_9              cpu        0.01%    0.02%   1.0002
#   gather2d_8                   cpu        0.06%    0.08%   1.0005
#   convrank_k256                cpu        1.68%    1.98%   1.0002
#   tanh*sin over 256x256        cpu        2.35%    3.39%   0.9995
#   scan_unroll_32               cpu        6.24%    6.64%   0.9994
#   spmd_reshard_d128            cpu        9.81%   10.67%   0.9992
#   poolgrad_max2d_d128          cpu       16.42%   18.11%   0.9994
#   ----------------------------------------------------------------  <- 16.4 .. 50.2 is EMPTY
#   linalg_expm_d32              cpu       50.23%   65.72%   0.9978
#   arity_tree_100               cpu       50.29%   73.65%   0.9984
#   ----------------------------------------------------------------  <- 50.3 .. 91.6 is EMPTY
#   adconst_idx_2p22             cpu       91.59%  183.18%   1.0012
#   dusfold_sum_200              cpu       92.83%  185.58%   1.0011
#   switch_ident_512             cpu       96.18%   96.26%   0.9997
#   jitfib_t22                   cpu       96.25%  105.16%   0.9998
#   switch_ident_1024            gpu       96.40%   96.58%   0.9990
#   convT64_dilate16             gpu       99.67%   99.89%   1.0007
#
# TWO THINGS THAT TABLE SETTLES.
#
# 1. FIDELITY IS FLAT AND COVERAGE IS NOT. Fidelity spans [0.9978, 1.0012] -- 0.22% -- across five
#    orders of magnitude of coverage, two platforms, and compiles from 0.1 s to 76 s. It does not
#    depend on the program, the backend, the machine or where the time went. Coverage depends on
#    all of them. That is the whole argument for keeping them as two numbers: one is a check with a
#    correct value, the other is a measurement with no correct value.
#
# 2. THE BAND EDGES SIT IN GAPS, not in the middle of a cloud. Nothing measured falls between 16%
#    and 50%, or between 50% and 92%. 0.25 and 0.75 are the midpoints of two empty intervals, which
#    is the most a threshold on this quantity can honestly claim.
#
# AND HOW REPEATABLE IS EACH? The sweep was run twice, once before and once after the leaf split, on
# a machine whose load changed in between. The backend denominator moved by 0.78x to 1.08x per arm
# (`dusfold_sum_200` 0.292 -> 0.229 s, `adconst_idx_2p22` 19.85 -> 16.62 s). Fidelity moved by at
# most 0.34% and did not correlate with any of it -- both of its numbers come from the same log, so
# there is nothing for machine load to skew. So: read `coverage` to a band, never to two decimal
# places, and read `fidelity` as an absolute. The band edges survive this because the gaps around
# them are 34 and 41 percentage points wide.
#
# LOW COVERAGE IS NOT A BROKEN PARSE. `gatherchain2d_9` reads 0.01% with fidelity 1.0002: 24.7 s of
# backend, 3.3 ms of it in HLO passes, and the parse is perfect. The seconds are in the CPU loop
# fusion emitter, which XLA neither times as a pass nor snapshots. `Coverage.broken` is the only
# thing here that ever means "the instrument failed", and it never consults coverage to decide.
# ══════════════════════════════════════════════════════════════════════════════════════════════

# The floor below which `fidelity` is a broken parser rather than a rounding artifact.
#
# NOT a guess. The only legitimate source of disagreement is that XLA prints `cumulative:` to three
# significant figures while the per-pass times it sums are exact microseconds, so the error is
# bounded by half an ulp of the printed total -- 1.4% for `3.5 ms`, 4.2% for `1.2 min`, and the
# tolerance actually used is computed from the digits XLA printed (see `_parse._ulp`). This floor is
# the backstop for the case where that computation is unavailable. It sits below the worst printable
# ulp (5%), far above the worst observed deviation (0.22%, table above), and far above the value a
# real loss produces: the min-unit bug reconstructed on `switch_ident_1024` scores 0.0111.
FIDELITY_FLOOR = 0.90

# WHAT THE TOP BAND CANNOT SAY, MEASURED. `coverage` answers "were the seconds inside a pass
# timer" and never "were the seconds spent transforming IR". On CPU those coincide. On GPU they come
# apart, because the autotuner is REGISTERED AS AN HLO PASS: `convT64_dilate16` and
# `gemm_shapes_k16` are indistinguishable here -- both ~98% `autotuner`, both PASS-BOUND, both
# fidelity ~1.0 -- and their causes are opposite. On the first, 49.7% of the pass is cuDNN kernels
# executing on real 134 MB buffers; on the second, 0.11% is kernel time and the seconds are spent
# COMPILING 456 Triton candidates. 450x apart, identical in everything this module reports. So the
# top band routes correctly to "rank the passes" and then stops being able to help, which is why
# its text names the one pass where the ranking is the beginning of the question rather than the
# end of it.
COVERAGE_BANDS = (
    (0.75, "PASS-BOUND -- the HLO passes ARE the compile. Rank them; the top entry is the answer. "
           "ONE EXCEPTION, and on GPU it is the common one: if the top entry is `autotuner`, this "
           "band has told you only that the seconds were inside a pass timer. The autotuner is "
           "registered as a pass and its seconds are kernels running and candidates compiling, "
           "not IR being transformed -- use scopex.autotune.autotune_cost to split those two."),
    (0.25, "MIXED -- the passes are a real but partial account. The top pass is a lead, not a "
           "conclusion; check what is left over before reporting it."),
    (0.00, "ELSEWHERE -- most of the backend is somewhere the pass timer cannot see: the emitter, "
           "LLVM, linking, or autotuning that XLA does not run as a pass. Do NOT report the top "
           "pass. This is a result about your compile, not a failure of the instrument."),
)


class Coverage(dict):
    """What a pass profile explains, and whether it was read correctly. See the module docstring.

    Two independent denominators, so the two questions never get answered by the same number:

    * ``fidelity``  -- scopex's sum against XLA's own. ~1.0 or the parse is broken.
    * ``coverage``  -- scopex's sum against ``jax.monitoring``'s backend seconds. Any value.
    """

    # ── the self-check ───────────────────────────────────────────────────────────────────────
    @property
    def fidelity(self) -> float | None:
        """Parsed seconds / XLA's own cumulative. ``None`` when XLA printed no totals."""
        c = self.get("xla_cumulative_s")
        if c is None:
            return None
        if c <= 0.0:
            return 1.0 if self.get("parsed_seconds", 0.0) <= 0.0 else float("inf")
        return self.get("parsed_seconds", 0.0) / c

    @property
    def lines_lost(self) -> int | None:
        """How many pass lines XLA counted that scopex did not. ``None`` when unknown.

        The unit-free check, and the one that would have caught the historical bug on its own:
        XLA's ``#called`` reached 640, the parser returned 639.
        """
        n = self.get("xla_pass_count")
        return None if n is None else n - self.get("parsed_passes", 0)

    @property
    def biggest_pass_lost(self) -> bool | None:
        """True when XLA's ``max:`` field reports a single pass LONGER than anything scopex parsed.

        The sharpest possible statement of the historical failure: XLA said one pass took 1.19 min
        and the parsed profile's largest entry was 0.12 s. A ranking whose top entry is smaller than
        the compiler's own reported maximum is not a ranking of anything.
        """
        m = self.get("xla_max_pass_s")
        if m is None:
            return None
        return m > self.get("parsed_max_s", 0.0) * 1.05 + 1e-9

    @property
    def split_ok(self) -> bool:
        """Do the leaf passes and the pipeline aggregates add back up to every line parsed?

        An exact identity by construction, so any failure is a bug in the leaf/aggregate walk --
        which is the one derived quantity here that a reader could not otherwise check. Without it
        ``coverage`` would be a number produced by a stack machine that nobody audits, which is the
        shape of every bug in this package's history.
        """
        want = self.get("parsed_seconds", 0.0)
        got = self.get("leaf_seconds", 0.0) + self.get("aggregate_seconds", 0.0)
        if abs(got - want) > 1e-6 + 1e-9 * max(1.0, abs(want)):
            return False
        # A pipeline that opened and never closed means the walk lost its place. Found this way:
        # 19 unclosed pipelines on a 21-thread GPU autotuning log, because the walk was reading one
        # interleaved stream instead of one stream per thread.
        return not self.get("unmatched_pipelines", 0)

    @property
    def checked(self) -> bool:
        """Did the cross-check actually RUN? Branch on this before believing ``broken is False``.

        ``broken`` is False in two completely different situations: the parse was checked against
        XLA's own arithmetic and agreed, or there was nothing to check it against. Every one of
        ``fidelity``, ``lines_lost`` and ``biggest_pass_lost`` depends on XLA continuing to print
        ``(cumulative: ..., max: ..., #called: ...)`` beside each pass. If a future XLA stops, all
        three go ``None``, ``broken`` goes quiet, and the ranking is exactly as unverified as it was
        before this module existed -- while looking identical to a verified one.

        This property is the difference. ``verdict`` says "parse UNCHECKED" in that case and
        ``__str__`` prints it, but a program cannot branch on prose. False here means: the only
        remaining guard is ``_parse.expect``'s witness count, which is weaker and reads the same
        log.

        ``split_ok`` is deliberately NOT folded in -- it is an internal identity that holds with or
        without XLA's totals, so it is checked separately and always.
        """
        return self.get("xla_pass_count") is not None

    @property
    def over_unity(self) -> bool:
        """Did the leaf passes sum to MORE than the backend interval they ran inside?

        Not impossible, and not necessarily a defect: XLA's GPU autotuner compiles candidate
        sub-modules CONCURRENTLY across ~21 threads, each candidate module running its own pass
        pipeline whose leaf passes are logged and summed. Concurrent seconds sum past wall clock.
        Observed on ``gemm_shapes_k16``: leaf 18.3103 s against backend 18.1094 s = 1.0111, with
        ``fidelity`` 0.9996 and ``split_ok`` True -- i.e. the parse was perfect and the ratio still
        exceeded 1.

        So when this is True, ``coverage`` is a ratio of CPU-seconds-across-threads to a wall-clock
        denominator and is not a fraction of anything. Read it as "the passes are the compile" and
        stop; do not read the value. A large excess (say > 1.2) with ``split_ok`` True and
        ``log_threads`` == 1 is a different animal and means the leaf/aggregate walk is wrong.
        """
        c = self.coverage
        return c is not None and c > 1.0 + 1e-9

    @property
    def broken(self) -> bool:
        """Is the PARSE demonstrably wrong? Independent of where the compile spent its time.

        False does NOT mean "checked and fine" -- see :attr:`checked`.
        """
        if not self.split_ok:
            return True
        if self.get("xla_pass_count") is None:
            return False
        if (self.lines_lost or 0) > 0:
            return True
        if self.biggest_pass_lost:
            return True
        f = self.fidelity
        if f is None:
            return False
        # THE SLACK, AND WHY IT IS NOT `1.0 - tolerance*2` ON ITS OWN. `tolerance` is the half-width
        # of the interval XLA's printed `cumulative:` stands for, derived from the digits it
        # printed. When it is absent -- a producer that did not set it, a hand-built Coverage, a
        # future `pass_log_totals` that returns None -- the old expression collapsed to
        # `f < max(0.90, 1.0)`, i.e. ANY fidelity below exactly 1.0 read as PARSE BROKEN. Measured
        # fidelity is never exactly 1.0 (it spans [0.9978, 1.0015] over 20 arms), so a missing
        # tolerance turned this check into a false-alarm generator on every healthy compile. A
        # check that cries wolf gets ignored, taking the real firing with it, which is the same
        # class of failure as a check that stays silent.
        slack = max(1.0 - FIDELITY_FLOOR, (self.get("tolerance") or 0.0) * 2)
        # SYMMETRIC, because both directions are defects and only one of them was being looked at.
        # Below: lines dropped, the historical bug (0.0111 on the reconstructed min-unit parse).
        # Above: lines counted twice -- scopex cannot legitimately total MORE seconds than XLA's own
        # running total over exactly the same lines. Both sit 45x outside the observed spread.
        return abs(f - 1.0) > slack

    # ── the measurement ──────────────────────────────────────────────────────────────────────
    @property
    def coverage(self) -> float | None:
        """LEAF pass seconds / backend seconds, both from the SAME compile. ``None`` if unknown.

        Both numbers come from ONE child process and one compile. That is the whole reason this
        lives in ``pass_timings`` instead of in a recipe: taking the denominator from a second
        compile in the parent made the ratio a comparison of two runs on a drifting machine, and
        the recipe that did so measured 0.88, 0.95, 1.04, 1.11 and 1.52 for a quantity that cannot
        exceed 1 by that mechanism.

        Can still exceed 1.0 slightly, from clock skew between XLA's per-pass timers and jax's
        stage timer, or from a pass whose work is counted in two pipelines. Well above 1.0 means
        the leaf/aggregate split failed -- check ``unmatched_pipelines``.
        """
        b = self.get("backend_s")
        if not b:
            return None
        return self.get("leaf_seconds", self.get("parsed_seconds", 0.0)) / b

    @property
    def naive_coverage(self) -> float | None:
        """What ``sum(result["passes"].values()) / backend`` would have given -- double-counted.

        Kept visible because it is what every hand-rolled version of this computation produces, and
        a reader who has one of those in a notebook needs to be able to see why it disagrees.
        """
        b = self.get("backend_s")
        return None if not b else self.get("parsed_seconds", 0.0) / b

    @property
    def band(self) -> str:
        c = self.coverage
        if c is None:
            return "UNKNOWN -- no backend seconds from the child process"
        for lo, label in COVERAGE_BANDS:
            if c >= lo:
                return label
        return COVERAGE_BANDS[-1][1]                                         # pragma: no cover

    @property
    def verdict(self) -> str:
        """One line. The parse check is asked FIRST, because if the parse is broken the coverage
        number is a fiction and reporting it as 'the time is elsewhere' is the exact inversion this
        package has already shipped once."""
        if self.broken:
            lost, f = self.lines_lost, self.fidelity
            bits = []
            if not self.split_ok:
                bits.append(f"the leaf/pipeline split did not close -- leaves "
                            f"{self.get('leaf_seconds', 0.0):.6g} s + aggregates "
                            f"{self.get('aggregate_seconds', 0.0):.6g} s vs parsed "
                            f"{self.get('parsed_seconds', 0.0):.6g} s, with "
                            f"{self.get('unmatched_pipelines', 0)} pipeline(s) left open across "
                            f"{self.get('log_threads', 1)} log thread(s)")
            if lost:
                bits.append(f"XLA counted {self['xla_pass_count']} pass invocations and scopex "
                            f"parsed {self['parsed_passes']} -- {lost} LOST")
            if self.biggest_pass_lost:
                bits.append(f"XLA reports a single pass of {self['xla_max_pass_s']:.4g} s and the "
                            f"largest scopex parsed is {self.get('parsed_max_s', 0.0):.4g} s")
            if f is not None and f < FIDELITY_FLOOR:
                bits.append(f"scopex totals {self['parsed_seconds']:.4g} s where XLA's own "
                            f"cumulative says {self['xla_cumulative_s']:.4g} s (fidelity {f:.4f})")
            return ("PARSE BROKEN -- do not read the ranking. " + "; ".join(bits) +
                    ". Fix scopex/_parse.py, then run scopex.selftest().")
        if self.get("backend_s") is None:
            why = self.get("why_no_backend") or "the child emitted no jax.monitoring numbers"
            return (f"PARSE OK (fidelity {self.fidelity:.3f}) but COVERAGE UNKNOWN -- {why}. The "
                    f"ranking is trustworthy as a ranking; what fraction of the compile it "
                    f"explains is not known."
                    if self.fidelity is not None else
                    f"COVERAGE UNKNOWN -- {why}, and XLA printed no totals to check the parse "
                    f"against either. Nothing here is cross-checked.")
        over = ("; leaf seconds EXCEED the backend interval -- the autotuner's candidate compiles "
                f"run concurrently across {self.get('log_threads', 1)} log threads, so this is a "
                "sum of CPU-seconds against a wall-clock denominator and is not a fraction"
                if self.over_unity else "")
        return f"{self.band} (coverage {self.coverage:.2%}, fidelity {self.fidelity:.3f}){over}" \
            if self.fidelity is not None else \
            (f"{self.band} (coverage {self.coverage:.2%}, parse UNCHECKED -- XLA printed no "
             f"(cumulative:, max:, #called:) totals, so nothing here was verified against the "
             f"compiler's own arithmetic; branch on Coverage.checked){over}")

    def __str__(self) -> str:
        f, c = self.fidelity, self.coverage
        rows = ["CHECK    parsed vs XLA's own accounting over the same lines"]
        if not self.checked:
            rows.append("  XLA printed no (cumulative: ..., #called: ...) fields -- UNCHECKED")
            rows.append("  (Coverage.checked is False: no self-check ran. `broken` cannot fire.)")
        else:
            rows.append(f"  pass lines   scopex {self['parsed_passes']:>8d}   "
                        f"XLA {self['xla_pass_count']:>8d}"
                        f"{'   <-- LOST ' + str(self.lines_lost) if self.lines_lost else '   ok'}")
            rows.append(f"  seconds      scopex {self['parsed_seconds']:>8.4g}   "
                        f"XLA {self['xla_cumulative_s']:>8.4g}   fidelity {f:.4f}")
            flag = "   <-- THE BIG ONE WAS DROPPED" if self.biggest_pass_lost else "   ok"
            rows.append(f"  slowest pass scopex {self.get('parsed_max_s', 0.0):>8.4g}   "
                        f"XLA {self['xla_max_pass_s']:>8.4g}{flag}")
        rows.append(f"  leaf+pipeline {self.get('leaf_seconds', 0.0):>8.4g} + "
                    f"{self.get('aggregate_seconds', 0.0):.4g} = "
                    f"{self.get('leaf_seconds', 0.0) + self.get('aggregate_seconds', 0.0):.4g}"
                    f"{'   ok' if self.split_ok else '   <-- SPLIT DOES NOT ADD UP'}")
        rows.append("")
        rows.append("MEASURE  HLO passes as a fraction of the compile they ran inside")
        if c is None:
            rows.append(f"  backend seconds unavailable -- {self.get('why_no_backend', '?')}")
        else:
            rows.append(f"  leaf passes   {self.get('leaf_seconds', 0.0):9.4f}   "
                        f"({self.get('n_leaves', 0)} of {self['parsed_passes']} lines; the "
                        f"other {self.get('n_aggregates', 0)} are pipeline totals, which "
                        f"would double-count)")
            rows.append(f"  backend       {self['backend_s']:9.4f}   "
                        f"({self.get('n_backend_compiles', 0)} compile(s) in the child)")
            rows.append(f"  coverage      {c:9.2%}"
                        + ("   <-- OVER 1.0: concurrent candidate compiles, see .over_unity"
                           if self.over_unity else ""))
            if not self.over_unity:
                rows.append(f"  NOT in passes {1 - c:9.2%}   "
                            f"emitter / LLVM / link / autotune-in-thunk")
            if self.get("trace_s") is not None:
                rows.append(f"  (child also: trace {self['trace_s']:.4f} s, lower "
                            f"{self['lower_s']:.4f} s, wall "
                            f"{self.get('child_wall_s', 0):.2f} s)")
        rows.append("")
        rows.append("VERDICT  " + self.verdict)
        return "\n".join(rows)
