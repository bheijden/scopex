"""Reading what a compile left behind: per-pass growth, a timeline, and codegen size.

These are the hand-rolled routines that actually localised pathologies during a 30-case
investigation, promoted to API. Each earned its place by being the only instrument that worked on
some case:

``pass_growth``     Counts instructions in every per-pass snapshot. On a size-cliff case both arms
                    tracked within 1.21x right up to layout assignment, and then ONE pass took the
                    slow arm 2,458 -> 176,189 instructions while the fast arm went 2,030 -> 5,446.
                    Same pass sequence in both. No other instrument reported it.
``pass_timeline``   Times passes from dump-file MTIMES. XLA writes each snapshot as the pass
                    completes, so the timestamps reconstruct where the seconds went -- including in
                    the gaps AFTER the last HLO pass, which is where LLVM lives and where the HLO
                    pass timer cannot see. This localised a 522x case on which every count-based
                    view was null.
``codegen_size``    LLVM IR lines and object bytes. Several pathologies have near-identical HLO and
                    differ only below it; one measured 1540 identical BufferAllocations with
                    optimised IR differing 5.3x.
``custom_calls``    A census of ``custom_call_target``. XLA records pass DECISIONS here -- whether
                    the sort rewriter chose a CUB kernel or fell back to a generic bitonic
                    lowering, say -- and an opcode census cannot see the difference because both
                    are opcode ``custom-call``.

and two that exist because the four above are all DERIVED NUMBERS, which is the shape of every bug
this package has shipped -- a plausible dict, no error, and the wrong pass named:

``pass_conservation``  Checks a per-pass curve against things it was not built from: a SECOND
                    instruction counter over the same bytes, and the two optimization-boundary
                    files. Returns the evidence and two separate flags -- whether the counting is
                    right, and whether it spans the compile. It found a live defect on the first
                    dump it read: XLA writes copy-insertion's three intra-pass stages with no
                    ``.before_`` field, ``dump_snapshot_name`` returned None for all three, and
                    every CPU curve in this package had been quietly skipping three consecutive
                    pass boundaries.
``boundary_diff``   WHAT differs between two arms at one boundary, where ``diverge`` says only
                    which pass. The three views the investigations kept rebuilding by hand -- the
                    opcode census, the computation sizes, the operand arity -- plus the check that
                    all three (and an independent line count) agree on how many instructions are
                    there. Aggregate only: XLA renames instructions freely and records no lineage,
                    so per-instruction correspondence is NOT derivable and nothing here implies it.

END-TO-END VALIDATION, ``bisect_m94`` vs its m=96 control, CPU, jax 0.10.2, serial, 2026-07-29.
969 MB / 941 files and 24 MB / 271 files under ``dump(passes='.*')``; 87.1 s and 0.95 s to compile.

    conservation      head_gap 0, tail_gap 0, coverage 1.0000 on BOTH arms; 0 counter
                      disagreements over 45 and 38 snapshots; no index gaps. The last snapshot's
                      count equals ``after_optimizations`` equals ``walk_hlo`` on the live
                      executable: 330,402, three routes to the same number.
    biggest_step      ``fusion``, 3,568 -> 374,369 = 104.9x in one pass, against the same pass
                      taking the control 2,050 -> 5,466 = 2.67x.
    boundary_diff     at that boundary: ``slice`` 45,311 vs 570, ``dynamic-slice`` VANISHED
                      (0 vs 294), 296 computations vs 187 of which 189 vs 3 hold 257+
                      instructions, max operand count 286 vs 55.

Those control-side numbers -- 570, 294, 202 computations, 55 operands -- are the ones the original
investigation derived BY HAND and to the instruction. The case-side numbers are m=94's analogues of
the m=95 figures it recorded (46,268 / 294 / 306 / 289). Note that the numbers docstringed elsewhere
in this package as m=94's (2,458 -> 176,189) are m=64's; m=94 is 3,568 -> 374,369.

All of these read a directory produced by :func:`scopex.dump`. None of them recompiles.
"""

from __future__ import annotations

import collections
import os
import pathlib
import tempfile
import warnings
from typing import NamedTuple

from . import _parse

__all__ = ["PassStep", "pass_growth", "pass_timeline", "codegen_size", "custom_calls",
           "modules_in", "diverge", "opcode_census", "opcode_delta", "hlo_at", "boundaries_in",
           "pass_conservation", "boundary_diff", "resolve_boundary"]

# The dump-FILENAME grammar and the fallback instruction-line pattern both live in scopex._parse,
# next to a real `ls` of a dump directory and a guard that raises when a parse comes back emptier
# than its input. `dump_snapshot_name` returns None for the many files in a dump that are not
# per-pass snapshots (.ll, .o, debug_options, the before/after-optimization modules), so the
# population -- not the call -- is what gets guarded, in `pass_growth` below.
from ._parse import dump_snapshot_name, is_hlo_instruction_line  # noqa: E402


def _regex_count(text: str) -> int:
    """Instructions per the LINE PATTERN. The second, independent counter -- see :func:`_count`."""
    return sum(1 for ln in text.splitlines() if is_hlo_instruction_line(ln))


def _count(text: str) -> tuple[int, int, str, int]:
    """``(instructions, computations, how, instructions_by_regex)`` for one snapshot.

    Prefers ``scopex.levels.hlo_module``, i.e. XLA's own text parser, which gives an exact count
    from the object graph instead of a per-line guess. It parsed 2,811 of 2,811 real per-pass
    snapshots. ``how`` records which route ran, so a silent slide back onto the regex is visible in
    the returned data rather than invisible.

    THE FOURTH RETURN IS THE CROSS-CHECK, and it is why this function does the work twice. The
    native count and the line count are two independent implementations reading the same bytes: one
    walks XLA's object graph in C++, the other matches a python regex per line. A curve built from
    either alone is unfalsifiable. Built from both, a counting bug has to fool two unrelated
    readers to stay hidden -- and the bug this package actually shipped (an ``\\S+`` shape group
    that could not match a TUPLE shape, because a tuple shape contains a space, undercounting every
    ``while`` / ``call`` / ``custom-call`` and every control-flow parameter by 31.8%) fooled exactly
    one of them. :func:`pass_conservation` reports the disagreement.
    """
    try:
        from .levels import hlo_module
        m = hlo_module(text)
        comps = m.computations()
        return (sum(len(c.instructions()) for c in comps), len(comps), "native",
                _regex_count(text))
    except Exception:
        n = _regex_count(text)
        return n, text.count(" {\n") + text.count("{\n"), "regex", n


class PassStep(NamedTuple):
    """One per-pass snapshot. ``instrs`` is the module as it stood BEFORE ``before_pass`` ran.

    ``how`` is ``"native"`` when the count came from XLA's parser and ``"regex"`` when it fell back
    to the line pattern; a mixed set of steps is not comparable and :func:`pass_growth` says so.
    """
    index: int
    pipeline: str
    after_pass: str
    before_pass: str
    instrs: int
    computations: int
    path: str
    mtime: float
    how: str = "native"
    #: The same snapshot counted by the LINE PATTERN instead of XLA's parser. Appended last so the
    #: tuple stays backward-compatible. Equal to ``instrs`` when ``how == "regex"`` (there is only
    #: one counter in that case, and :func:`pass_conservation` says so rather than claiming
    #: agreement). Its disagreement with ``instrs`` is the cross-check on the whole curve.
    instrs_regex: int = -1

    @property
    def name(self) -> str:
        # `before_pass` is empty for the intra-pass STAGE snapshots (copy-insertion writes three),
        # which have an `after_` field and no `before_` one. Naming those by the stage that just ran
        # keeps every name unique and keeps the string readable; it does not change what the other
        # snapshots are called.
        return f"{self.pipeline}/{self.before_pass or 'after_' + self.after_pass}"


def modules_in(dump_dir: str | os.PathLike) -> list[str]:
    """Module stems present, largest first by snapshot count.

    A dump directory holds JAX's warm-up modules (``jit_convert_element_type`` and friends)
    alongside the one you care about. Picking the wrong stem is the most common way to read a dump
    and conclude nothing happened."""
    c: collections.Counter = collections.Counter()
    names = os.listdir(dump_dir)
    for f in names:
        m = dump_snapshot_name(f)
        if m:
            c[m["module"]] += 1
    if not c and any(".before_" in f for f in names):
        # There are snapshot-shaped filenames here and none of them parsed. That is the filename
        # grammar moving, not a dump without snapshots, and the two must not look alike.
        raise _parse.ParseError(
            f"scopex parser 'dump_snapshot_name' matched none of the {len(names)} files in "
            f"{dump_dir}, yet some of them contain '.before_'.\n"
            f"  built by   : xla/service/dump.cc\n"
            f"  example    : {next(f for f in names if '.before_' in f)}\n"
            f"  Fix _SNAPSHOT in scopex/_parse.py; do not let an empty module list read as "
            f"'this compile ran no passes'.")
    return [k for k, _ in c.most_common()]


def _pick(dump_dir, module: str | None) -> str:
    mods = modules_in(dump_dir)
    if not mods:
        raise FileNotFoundError(
            f"no per-pass snapshots in {dump_dir}. dump() needs passes='.*' (or a regex) -- "
            f"without it XLA writes only the before/after-optimisation modules.")
    if module is None:
        return mods[0]
    hit = [m for m in mods if module in m]
    if not hit:
        raise KeyError(f"no module matching {module!r}; present: {mods}")
    return hit[0]


def pass_growth(dump_dir: str | os.PathLike, *, module: str | None = None) -> list[PassStep]:
    """Instruction count at every pass boundary, in pipeline order.

    ``module`` selects a stem by substring; the default is the one with the most snapshots, which
    is the program you compiled rather than a JAX warm-up module.
    """
    stem = _pick(dump_dir, module)
    out: list[PassStep] = []
    for f in os.listdir(dump_dir):
        m = dump_snapshot_name(f)
        if not m or m["module"] != stem:
            continue
        p = pathlib.Path(dump_dir) / f
        text = p.read_text(errors="replace")
        n, ncomp, how, nrx = _count(text)
        out.append(PassStep(
            index=m["index"], pipeline=m["pipeline"],
            after_pass=m["after_pass"], before_pass=m["before_pass"],
            instrs=n, computations=ncomp,
            path=str(p), mtime=p.stat().st_mtime, how=how, instrs_regex=nrx))
    out.sort(key=lambda s: s.index)
    fell_back = [s.name for s in out if s.how == "regex"]
    if fell_back and len(fell_back) != len(out):
        # A curve counted partly one way and partly the other is not a curve. The regex undercounts
        # tuple-shaped instructions, so a mixed curve shows a fake step exactly where the route
        # changed -- and pass_growth exists to find real steps.
        warnings.warn(
            f"{len(fell_back)} of {len(out)} snapshots fell back to the line-based counter while "
            f"the rest were counted natively, so this curve mixes two scales and its steps are not "
            f"trustworthy. First: {fell_back[:3]}", RuntimeWarning, stacklevel=2)
    return out


# ══════════════════════════════════════════════════════════════════════════════════════════════
# THE CONSERVATION CHECK
#
# A per-pass curve is a chain of counts, and a chain of counts is exactly the shape of instrument
# this package keeps getting wrong: it returns a plausible list of plausible numbers whether or not
# the counting worked. Three real bugs, all of which produced a curve that LOOKED fine:
#
#   * a shape group of `\S+` that could not match a TUPLE shape (a tuple shape contains a space),
#     so every `while` / `call` / `custom-call` and every control-flow parameter went uncounted --
#     31.8% on the measured module, and biased toward exactly the instructions a control-flow
#     pathology is made of;
#   * a filename grammar that silently drops snapshots it cannot parse, so the curve skips pass
#     boundaries and credits several passes' work to one of them;
#   * `_pick` choosing a JAX warm-up module instead of the program, giving a short flat curve for a
#     compile that did a great deal.
#
# So the curve is checked against things it did not use to build itself:
#
#   1. TWO COUNTERS.  XLA's C++ HLO parser and a python line regex read the same bytes. They must
#      agree. A uniform bias in either is invisible to every other check here, and is what bug one
#      was.
#   2. TWO INDEPENDENT ENDPOINTS.  `*.before_optimizations.txt` and `*_after_optimizations.txt` are
#      separate files that the curve is not built from. The first snapshot must equal the former.
#      The distance from the last snapshot to the latter is the part of the compile the curve DOES
#      NOT SEE, and it is reported as a fraction rather than assumed away -- the same shape as the
#      coverage ratio that would have caught the min-units bug in `pass_timings`.
#   3. THE INDEX SEQUENCE.  XLA numbers its snapshots 0000, 0001, ... contiguously. A hole means a
#      file exists that the grammar could not read.
#
# The telescoping identity `sum(deltas) == last - first` is reported too, but on its own it is a
# TAUTOLOGY -- it holds for any list of numbers, including a list of wrong ones. It is here to fail
# loudly if the chain is ever built from something other than consecutive counts, not because it
# validates anything by itself. The endpoints and the second counter are what make the check real.
# ══════════════════════════════════════════════════════════════════════════════════════════════

def pass_conservation(dump_dir: str | os.PathLike, *, module: str | None = None,
                      steps: list[PassStep] | None = None) -> dict:
    """Does this dump's per-pass curve conserve? Returns the evidence, not a boolean.

    Pass ``steps`` if you already called :func:`pass_growth` -- parsing a large dump twice is
    minutes. Everything else is read from filenames and from the two optimization-boundary files.

    Returned keys, and what each one can catch::

        counters      {"disagreements": n, "worst": (pass, native, regex, shortfall), ...}
                      XLA's parser vs the line pattern on the SAME bytes. Nonzero means one of the
                      two counters is wrong and the curve inherits whichever is.
        anchors       {"before_optimizations": n, "first_snapshot": n, "head_gap": d, ...}
                      `head_gap != 0` means the curve does not start where the compile did: a wrong
                      module stem, a missing leading snapshot, or two counters that disagree.
        coverage      {"net_change": n, "seen_by_curve": n, "fraction": f, "tail_gap": d}
                      The fraction of the module's net instruction-count change that happens
                      BETWEEN the first and last snapshot. XLA writes no snapshot after the final
                      pass, so `1 - fraction` is real work the curve cannot see, and a curve that
                      explains 10% of the change is not a localisation however clean it looks.
        chain         {"sum_of_deltas": n, "last_minus_first": n, "residual": 0}
                      The telescoping identity. Tautological; see the block comment above.
        indices       {"n": n, "expected_span": n, "missing": [...], "unreadable_files": [...]}
        counting_consistent   the numbers are the numbers: the two counters agree, the index
                      sequence is contiguous, the chain telescopes.
        covers_whole_compile  the numbers span the compile: the curve starts at
                      before_optimizations and reaches after_optimizations. SEPARATE on purpose --
                      a dump taken with a narrow `passes=` regex is a perfectly good dump whose
                      curve legitimately starts late and stops early, and a flag that failed on
                      that would fail on ordinary use and get ignored.
        complaints    every sentence from both, so neither kind gets buried. Empty when clean.

    Measured on ``bisect_m94`` and its m=96 control, both under ``passes='.*'`` on CPU: head_gap 0,
    tail_gap 0, coverage 1.0000, 0 counter disagreements over 45 and 38 snapshots, and the final
    snapshot count equal to ``walk_hlo`` on the live executable (330,402). Injected faults it was
    made to fail on: the tuple-shape regex (38/38 snapshots disagree, worst -12.5%), one snapshot
    made unreadable (``missing == [20]``), and a curve truncated before ``fusion`` runs (every
    internal check still passes; coverage 0.8%).
    """
    stem = _pick(dump_dir, module)
    steps = pass_growth(dump_dir, module=module) if steps is None else steps
    steps = [s for s in steps if s.path and pathlib.Path(s.path).name.startswith(stem + ".")]
    out: dict = {"module": stem, "n_snapshots": len(steps)}
    bad: list[str] = []          # the counting is wrong
    gap: list[str] = []          # the counting is right and does not reach the whole compile
    if not steps:
        return out | {"counting_consistent": False, "covers_whole_compile": False,
                      "complaints": [f"no snapshots for module {stem!r} in {dump_dir}"]}

    # ── 1. two counters, same bytes ─────────────────────────────────────────────────────────────
    dis = [(s.name, s.instrs, s.instrs_regex,
            round((s.instrs - s.instrs_regex) / max(1, s.instrs), 4))
           for s in steps
           if s.instrs_regex >= 0 and s.instrs_regex != s.instrs and s.how != "regex"]
    dis.sort(key=lambda r: -abs(r[3]))
    n_native = sum(1 for s in steps if s.how == "native")
    out["counters"] = {
        "native": n_native, "regex_fallback": len(steps) - n_native,
        "cross_checked": sum(1 for s in steps if s.how == "native" and s.instrs_regex >= 0),
        "disagreements": len(dis), "worst": dis[0] if dis else None,
        "worst_shortfall": dis[0][3] if dis else 0.0}
    if dis:
        bad.append(
            f"the two instruction counters disagree on {len(dis)} of {len(steps)} snapshots; worst "
            f"{dis[0][0]}: XLA's parser {dis[0][1]}, the line pattern {dis[0][2]} "
            f"({dis[0][3]:+.1%}). One of them is wrong and this curve is built from XLA's. This is "
            f"the signature of the tuple-shape bug: a line pattern that cannot match "
            f"'(s32[], f32[8]{{0}})' drops every while/call/custom-call.")
    if n_native and n_native != len(steps):
        bad.append(f"{len(steps) - n_native} of {len(steps)} snapshots fell back to the line "
                   f"counter while the rest were counted natively -- the curve mixes two scales.")

    # ── 2. two endpoints the curve was not built from ───────────────────────────────────────────
    ends: dict = {}
    for key, at in (("before_optimizations", "before"), ("after_optimizations", "after")):
        try:
            ends[key] = _count(hlo_at(dump_dir, at, module=stem))[0]
        except Exception as e:                                               # pragma: no cover
            ends[key] = None
            bad.append(f"no {key} module to anchor against ({type(e).__name__}); the curve has "
                       f"nothing independent to check against; its endpoints are unverified")
    first, last = steps[0].instrs, steps[-1].instrs
    head = None if ends["before_optimizations"] is None else first - ends["before_optimizations"]
    tail = None if ends["after_optimizations"] is None else ends["after_optimizations"] - last
    out["anchors"] = {"before_optimizations": ends["before_optimizations"],
                      "first_snapshot": first, "head_gap": head,
                      "last_snapshot": last, "after_optimizations": ends["after_optimizations"],
                      "tail_gap": tail}
    if head:
        # A COVERAGE complaint and not a counting one, for the same reason the tail is: under
        # `dump(passes='fusion')` XLA writes no snapshot until fusion runs, so the first snapshot
        # legitimately sits well past the start. Counting it as a counting failure would mark every
        # narrow dump broken.
        gap.append(
            f"the first snapshot has {first} instructions but {stem}.before_optimizations.txt has "
            f"{ends['before_optimizations']} (gap {head:+d}). The curve does not start where the "
            f"compile did. EXPECTED under a narrow dump(passes=...); otherwise a dropped leading "
            f"snapshot -- and everything before the first snapshot is change this curve cannot "
            f"attribute to any pass.")

    # ── the coverage ratio: how much of the change the curve actually witnessed ──────────────────
    if ends["before_optimizations"] is not None and ends["after_optimizations"] is not None:
        net = ends["after_optimizations"] - ends["before_optimizations"]
        seen = last - first
        out["coverage"] = {
            "net_change": net, "seen_by_curve": seen, "tail_gap": tail, "head_gap": head,
            "fraction": round(seen / net, 4) if net else None,
            "gross_seen": sum(abs(steps[i].instrs - steps[i - 1].instrs)
                              for i in range(1, len(steps)))}
        if net and not (0.9 <= seen / net <= 1.1):
            gap.append(
                f"the per-pass curve accounts for {seen:+d} of the module's {net:+d} net "
                f"instruction change ({seen / net:.1%}); {tail:+d} of it happens after the last "
                f"snapshot. So any pass this curve names is a localisation within {seen / net:.0%} "
                f"of the change, not within all of it. This is EXPECTED when dump(passes=...) was "
                f"narrower than '.*' -- you asked for a slice and got one -- and is "
                f"a missing-snapshot bug otherwise. The counting itself is unaffected either way.")

    # ── 3. the telescoping identity, and the index sequence ─────────────────────────────────────
    ssum = sum(steps[i].instrs - steps[i - 1].instrs for i in range(1, len(steps)))
    out["chain"] = {"sum_of_deltas": ssum, "last_minus_first": last - first,
                    "residual": ssum - (last - first)}
    if out["chain"]["residual"]:                                             # pragma: no cover
        bad.append(f"sum of per-pass deltas {ssum} != last - first {last - first}. The chain is "
                   f"built from consecutive counts.")

    idx = [s.index for s in steps]
    span = set(range(min(idx), max(idx) + 1))
    missing = sorted(span - set(idx))
    unreadable = sorted(f for f in os.listdir(dump_dir)
                        if f.startswith(stem + ".") and not dump_snapshot_name(f)
                        and _parse.SNAPSHOT_INDEX.match(f[len(stem) + 1:]))
    out["indices"] = {"n": len(idx), "first": min(idx), "last": max(idx),
                      "expected_span": len(span), "missing": missing,
                      "unreadable_files": unreadable[:8], "n_unreadable": len(unreadable)}
    if missing:
        bad.append(
            f"{len(missing)} snapshot indices are missing from the curve ({missing[:6]}), so "
            f"{len(missing)} pass boundaries went uncounted and their work is credited to the "
            f"neighbouring step. {len(unreadable)} file(s) here carry a snapshot index "
            f"and do not parse: {unreadable[:3]}")

    # TWO FLAGS AND NOT ONE. `counting_consistent` is about whether the numbers are the numbers;
    # `covers_whole_compile` is about whether they span the compile. Folding the second into the
    # first would fail every dump taken with a narrow `passes=` regex -- a completely ordinary way
    # to use `dump()` -- and a flag that fails on ordinary use gets ignored, taking the counting
    # complaints down with it. Both sets of sentences land in `complaints`, so neither is buried.
    out["counting_consistent"] = not bad
    out["covers_whole_compile"] = not gap
    out["complaints"] = bad + gap
    return out


# ── pass_timeline LIVES IN scopex/timeline.py NOW, AND THE MOVE IS THE POINT ──────────────────
# What used to be here was eleven lines: subtract consecutive snapshot mtimes, then scan for a
# `.ll` and a `.o` and call the gaps LLVM. Every number it returned was plausible and none of them
# was ever checked against anything. It is the only instrument in this package that can see BELOW
# the HLO pass pipeline, which is exactly why shipping it unvalidated was the worst trade here.
#
# `scopex.timeline` is the same measurement with a falsifiable alignment test attached: every glog
# line carries a CLOCK_REALTIME microsecond timestamp, `st_mtime` is on that same clock, and XLA
# writes the snapshot INSIDE the pass's scoped timer -- so `mtime(after_P)` must fall between that
# pass's START and END log lines. Measured 683/683 inside across 12 compiles. The tail then carries
# an error bound taken from the worst observed offset rather than an adjective, and a `.verdict`
# that says UNVALIDATED when no log was supplied instead of quietly reading the same as a checked
# one. The old three-way `<llvm ir emission>` / `<llvm optimisation>` / `<object codegen>` split is
# still computed, but only when it is DEFINED -- one kernel module, not interleaved -- because with
# 223 kernels compiling concurrently those boundaries order nothing.
from .timeline import pass_timeline  # noqa: E402,F401  (re-exported: one implementation, not two)


def codegen_size(dump_dir: str | os.PathLike) -> dict:
    """Emitted-code size, per backend artifact kind.

    ``ir_no_opt_lines`` / ``ir_with_opt_lines``  LLVM IR, both backends
    ``obj_bytes``   ``.o``   host objects            -- XLA:CPU
    ``ptx_bytes``   ``.ptx`` device assembly         -- XLA:GPU
    ``code_bytes``  whichever of the two this backend actually emitted
    ``kinds``       the artifact extensions found, so a zero is attributable

    WHY ``kinds`` EXISTS. This counted only ``.ll`` and ``.o``, and the CUDA backend writes
    NEITHER object form -- it writes ``.ptx``. So every GPU dump reported ``obj_bytes = 0``: an
    absence of instrumentation presented as a measured zero, which is the failure shape this
    package exists to prevent. Measured on a real CUDA dump at the time: 8 ptx files totalling
    27,488 bytes, reported as nothing. ``kinds`` now distinguishes "this backend emitted no code
    artifact" from "scopex was not looking for the one it emitted", and a caller can tell which.

    A genuinely empty result is still meaningful: a program that constant-folds to a literal emits
    no LLVM module at all, and so does one XLA hands to a library kernel.
    """
    out: dict = {"ir_no_opt_lines": 0, "ir_with_opt_lines": 0,
                 "obj_bytes": 0, "ptx_bytes": 0, "code_bytes": 0,
                 "kinds": {}, "files": {}}
    for f in os.listdir(dump_dir):
        p = pathlib.Path(dump_dir) / f
        ext = "".join(pathlib.Path(f).suffixes[-1:]) or pathlib.Path(f).suffix
        if f.endswith(".ll"):
            n = sum(1 for _ in p.open(errors="replace"))
            out["ir_with_opt_lines" if "with-opt" in f else "ir_no_opt_lines"] += n
            out["files"][f] = n
        elif f.endswith(".o"):
            out["obj_bytes"] += p.stat().st_size
            out["files"][f] = p.stat().st_size
        elif f.endswith(".ptx"):
            out["ptx_bytes"] += p.stat().st_size
            out["files"][f] = p.stat().st_size
        else:
            continue
        out["kinds"][ext] = out["kinds"].get(ext, 0) + 1
    out["code_bytes"] = out["obj_bytes"] + out["ptx_bytes"]
    return out


def custom_calls(source) -> collections.Counter:
    """Census of ``custom_call_target``, from a ``Compiled``, an ``HloModule``, or HLO text.

    Needed because an opcode census cannot distinguish two different pass DECISIONS: a CUB sort and
    a generic bitonic lowering are both opcode ``custom-call``, and which one XLA picked was the
    entire answer on two cases.

    ``custom_call_target`` is not on ``HloInstruction`` (its whole surface is ``async_wrapped_root,
    name, opcode, operands, to_string, users``), so the target itself still comes out of the printed
    form. But the INSTRUCTIONS are enumerated natively and filtered on the opcode enum, so the
    pattern only ever runs on a string already known to be a custom-call. Scanning the whole module
    text instead also counts the literal ``custom_call_target=`` that appears inside a
    ``backend_config`` blob or inside an embedded pre-fusion module.
    """
    try:
        from .levels import hlo_module
        m = hlo_module(source)
    except Exception:
        if not isinstance(source, str):                                      # pragma: no cover
            from .flags import hlo_text
            source = hlo_text(source)
        return collections.Counter(_parse.custom_call_targets(source))
    out: collections.Counter = collections.Counter()
    for comp in m.computations():
        for i in comp.instructions():
            if i.opcode.name == "kCustomCall":
                out.update(_parse.custom_call_targets(i.to_string()))
    return out


def diverge(case_dir, control_dir, *, module: str | None = None, factor: float = 1.5) -> dict:
    """Where two arms separate. THE routine this module exists for.

    Walks both per-pass curves in lockstep and returns the first pass at which the case's
    instruction count exceeds the control's by more than ``factor``, together with both curves.

    ``pass_sequence_identical`` is reported and matters: if the two arms ran DIFFERENT passes, the
    divergence point is a pass-selection difference rather than one pass behaving badly, and those
    want opposite fixes.

    ``biggest_step`` is the answer to hand onward, and it is not ``diverges_at``. Two reasons.
    ``diverges_at`` is a THRESHOLD crossing, so on a case that is already 1.65x the control at the
    first snapshot it fires at snapshot 0 and names a pass that did nothing -- measured, on
    ``bisect_m94``. And its ``pass`` field is the snapshot's own name, built from ``before_pass``,
    i.e. the pass that has not run yet. ``biggest_step`` gives the largest single-pass ratio, the
    pass that CAUSED it (``caused_by``), the same pass's behaviour in the control, and a
    ``boundary`` string to pass straight to :func:`boundary_diff`. On ``bisect_m94``: ``caused_by
    'fusion'``, 3,568 -> 374,369 (104.9x), the control's same pass 2,050 -> 5,466 (2.67x).

    ``conservation`` carries :func:`pass_conservation` for BOTH arms, and it is part of the answer
    rather than a separate call you might not make. Every field above is derived from two chains of
    counts; if the counting is broken, ``diverges_at`` still comes back naming a pass, with no
    outward sign that it is naming the wrong one. Read ``conservation[arm]["complaints"]`` before
    reading ``diverges_at``. ``conservation[arm]["coverage"]["fraction"]`` in particular says how
    much of the module's net change these curves witnessed at all -- XLA writes no snapshot after
    the last pass, so a divergence point is a localisation within that fraction and not within the
    whole compile.
    """
    a = pass_growth(case_dir, module=module)
    b = pass_growth(control_dir, module=module)
    an = [s.name for s in a]
    bn = [s.name for s in b]
    at = {s.name: s for s in a}
    bt = {s.name: s for s in b}
    first = None
    for name in an:
        if name not in bt:
            continue
        r = at[name].instrs / max(1, bt[name].instrs)
        if r > factor:
            first = {"pass": name, "case_instrs": at[name].instrs,
                     "control_instrs": bt[name].instrs, "ratio": round(r, 2)}
            break
    # ── THE PASS THAT DID IT, WHICH IS NOT THE PASS THE CURVE IS LABELLED WITH ──────────────────
    # A snapshot file is `after_A.before_B` and holds the module BETWEEN them, so `PassStep.name`
    # -- built from `before_pass` -- names the pass that has NOT RUN YET. A jump into snapshot i was
    # therefore caused by snapshot i's `after_pass`, and reporting the curve label instead names an
    # innocent pass downstream. Measured on bisect_m94: the 104.9x jump sits in the snapshot named
    # `HLO_passes_after_layout_assignment/fusion-wrapper` and was done by `fusion`. `diverges_at`
    # keeps the old label because callers read it; `biggest_step` is the corrected form, and
    # `boundary` is the string to hand to `boundary_diff`.
    #
    # Nor can the culprit be had by looking one snapshot back: `after_pass` stays at the last pass
    # that actually CHANGED the module, so consecutive snapshots repeat it (7 in a row reading
    # `after=pipeline-start` on a two-line program). Hence the control is matched on the FIRST
    # snapshot carrying that `after_pass`, not on position.
    steps = [(a[i].instrs / max(1, a[i - 1].instrs), i) for i in range(1, len(a))]
    biggest = None
    if steps:
        ratio, i = max(steps)
        same = next((k for k in range(1, len(b)) if b[k].after_pass == a[i].after_pass), None)
        biggest = {
            "caused_by": a[i].after_pass,
            "snapshot_named": a[i].name,
            "boundary": a[i].name,        # hand THIS to boundary_diff, not diverges_at["pass"]
            "index": a[i].index,
            "before": a[i - 1].instrs, "after": a[i].instrs, "ratio": round(ratio, 2),
            "control_same_pass": None if same is None else {
                "before": b[same - 1].instrs, "after": b[same].instrs,
                "ratio": round(b[same].instrs / max(1, b[same - 1].instrs), 2)}}

    cons = {"case": pass_conservation(case_dir, module=module, steps=a),
            "control": pass_conservation(control_dir, module=module, steps=b)}
    return {
        "diverges_at": first,
        "biggest_step": biggest,
        "pass_sequence_identical": an == bn,
        "case_only_passes": [n for n in an if n not in bt],
        "control_only_passes": [n for n in bn if n not in at],
        "case_final": a[-1].instrs if a else 0,
        "control_final": b[-1].instrs if b else 0,
        "case_curve": [(s.name, s.instrs) for s in a],
        "control_curve": [(s.name, s.instrs) for s in b],
        "conservation": cons,
        "conserves": all(c.get("counting_consistent") for c in cons.values()),
        "covers_whole_compile": all(c.get("covers_whole_compile") for c in cons.values()),
        "complaints": [f"{k}: {m}" for k, c in cons.items() for m in c.get("complaints", ())],
    }


# ══════════════════════════════════════════════════════════════════════════════════════════════
# OPCODE CENSUSES, AND SUBTRACTING TWO OF THEM
#
# Several findings in the 30-case investigation were opcode censuses compared BY HAND: slice
# 46,268 vs 570 with dynamic-slice 0 vs 294 on one pair, select 1,404 vs 300 on another. Doing it
# by hand is slow and, worse, it is done at whatever boundary happened to be open -- and the answer
# depends entirely on the boundary. The same pair of arms compared before optimization and after it
# can differ in opposite directions, because the whole point of the pass pipeline is to change the
# opcode mix. `opcode_delta` therefore refuses to default the boundary silently: `at` is part of
# the returned dict, and `boundaries_in` lists what a directory can actually answer for.
# ══════════════════════════════════════════════════════════════════════════════════════════════

_BEFORE = ".before_optimizations.txt"
_AFTER = "_after_optimizations.txt"


class _Row(NamedTuple):
    """One snapshot, read from its FILENAME ONLY -- no file is opened."""
    index: int
    pipeline: str
    after_pass: str
    before_pass: str
    path: str

    @property
    def name(self) -> str:
        return f"{self.pipeline}/{self.before_pass or 'after_' + self.after_pass}"


def _rows(dump_dir, module: str | None) -> list[_Row]:
    """Every snapshot's identity, in pipeline order, WITHOUT reading any of them.

    :func:`pass_growth` parses ~1 GB of HLO text to produce the same names; a caller that wants to
    NAME a boundary rather than count it must not pay for that. Measured on ``bisect_m94``: the
    dump is 941 files and 969 MB, and locating one boundary in it costs a directory listing.
    """
    stem = _pick(dump_dir, module)
    out = []
    for f in os.listdir(dump_dir):
        m = dump_snapshot_name(f)
        if m and m["module"] == stem:
            out.append(_Row(m["index"], m["pipeline"], m["after_pass"], m["before_pass"],
                            str(pathlib.Path(dump_dir) / f)))
    out.sort(key=lambda r: r.index)
    return out


def boundaries_in(dump_dir: str | os.PathLike, *, module: str | None = None) -> list[str]:
    """Every boundary in ``dump_dir`` that :func:`hlo_at` can read, in pipeline order.

    ``"before"`` and ``"after"`` are always there; the rest are ``pipeline/pass`` names and exist
    only if the dump was taken with ``passes=`` set. Reads filenames only.
    """
    names = os.listdir(dump_dir)
    stem = module
    if stem is None:
        mods = modules_in(dump_dir) if any(dump_snapshot_name(f) for f in names) else []
        stem = mods[0] if mods else None
    out = []
    if any(f.endswith(_BEFORE) and (stem is None or f.startswith(stem)) for f in names):
        out.append("before")
    have = any(map(dump_snapshot_name, names))
    out += [r.name for r in (_rows(dump_dir, module) if have else [])]
    if any(f.endswith(_AFTER) and (stem is None or f.startswith(stem)) for f in names):
        out.append("after")
    return out


def resolve_boundary(dump_dir: str | os.PathLike, at: str = "after", *,
                     module: str | None = None) -> dict:
    """WHICH file ``at`` names in this dump, and how it was matched. Opens nothing.

    Returned so that a comparison across two dumps can PROVE it read both arms at the same place.
    ``at`` is matched, in order: exactly against ``pipeline/before_pass``; as a substring of it; as
    a substring of ``after_pass``. Those three routes select genuinely different boundaries -- on a
    CPU dump ``"fusion"`` is a substring of the snapshot NAMED ``.../fusion-wrapper``, which is the
    module AFTER the pass called ``fusion`` -- so ``how`` and ``after_pass`` come back with the hit
    rather than a bare filename.
    """
    names = sorted(os.listdir(dump_dir))
    if at in ("before", "after"):
        suffix = _BEFORE if at == "before" else _AFTER
        hits = [f for f in names if f.endswith(suffix)]
        if module:
            hits = [f for f in hits if module in f]
        elif hits:
            # Same rule as `modules_in`: prefer the module with the most per-pass snapshots, i.e.
            # the program you compiled rather than a JAX warm-up module.
            mods = modules_in(dump_dir) if any(map(dump_snapshot_name, names)) else []
            for m in mods:
                if any(f.startswith(m + ".") for f in hits):
                    hits = [f for f in hits if f.startswith(m + ".")]
                    break
        if not hits:
            raise FileNotFoundError(
                f"no *{suffix} in {dump_dir}"
                + (f" for module {module!r}" if module else "")
                + f". Present boundaries: {boundaries_in(dump_dir, module=module)}")
        return {"asked": at, "how": "optimization-boundary", "index": None, "name": at,
                "pipeline": "", "after_pass": at, "before_pass": "",
                "path": str(pathlib.Path(dump_dir) / hits[0])}
    # Filenames only. This used to call `pass_growth`, i.e. parse every snapshot in the directory,
    # to recover names it can read off the directory listing -- 969 MB and minutes on `bisect_m94`
    # to answer "where is the file called X".
    rows = _rows(dump_dir, module)
    for how, hits in (("exact-name", [r for r in rows if r.name == at]),
                      ("substring-of-name", [r for r in rows if at in r.name]),
                      ("substring-of-after_pass", [r for r in rows if at in r.after_pass])):
        if hits:
            r = hits[0]
            return {"asked": at, "how": how, "index": r.index, "name": r.name,
                    "pipeline": r.pipeline, "after_pass": r.after_pass,
                    "before_pass": r.before_pass, "path": r.path, "n_matched": len(hits)}
    raise KeyError(f"no pass boundary matching {at!r} in {dump_dir}. Present: "
                   f"{boundaries_in(dump_dir, module=module)}")


def hlo_at(dump_dir: str | os.PathLike, at: str = "after", *, module: str | None = None) -> str:
    """The HLO text at one boundary of a dump. ``at`` is ``"before"``, ``"after"``, or a pass name.

    A pass name may be given as ``pipeline/before_pass`` (what :class:`PassStep` calls itself) or as
    a plain substring of it; the FIRST matching snapshot in pipeline order wins, and an ambiguous
    or absent name raises with the list of what is there rather than falling back to a boundary the
    caller did not ask for. :func:`resolve_boundary` returns the same choice without reading it.
    """
    return pathlib.Path(
        resolve_boundary(dump_dir, at, module=module)["path"]).read_text(errors="replace")


def opcode_census(source, *, per_computation: bool = False) -> collections.Counter:
    """``Counter`` of XLA opcodes, walked NATIVELY from the object model.

    ``source`` may be a ``Compiled``, an ``HloModule``, or HLO text -- including one per-pass dump
    snapshot. Opcodes come from the ``HloOpcode`` enum via :func:`scopex.levels.opcode_of`, not
    from a line pattern, so a tuple-shaped instruction (``while``, ``call``, ``custom-call``) is
    counted like any other; the line-based counter this replaces dropped every one of them.

    ``per_computation`` prefixes each key with its computation, which separates "the fusion body
    grew" from "there are more fusions".
    """
    from .levels import hlo_module, opcode_of
    m = hlo_module(source)
    c: collections.Counter = collections.Counter()
    for comp in m.computations():
        for i in comp.instructions():
            c[f"{comp.name}/{opcode_of(i)}" if per_computation else opcode_of(i)] += 1
    return c


def opcode_delta(case_dump, control_dump, *, at: str = "after", module: str | None = None,
                 top: int = 12) -> dict:
    """The opcodes that differ most between two dumps, at one chosen boundary. One call.

    ``at`` picks the boundary in BOTH dumps -- ``"before"``, ``"after"``, or a pass name
    (see :func:`boundaries_in`). It is echoed in the result because the answer is meaningless
    without it: the pass pipeline exists to change the opcode mix, so the same two arms compared
    before and after optimization can differ in opposite directions.

    Returns ``{"at", "delta", "case_only", "control_only", "case_total", "control_total",
    "case_census", "control_census"}``. ``delta`` is ``[(opcode, case_n, control_n, case_n -
    control_n), ...]`` sorted by absolute difference, longest first.

    ``case_only``/``control_only`` are the opcodes present in one arm and ABSENT in the other, and
    they are listed separately on purpose: an opcode that went 294 -> 0 is a lowering decision,
    while one that went 46,268 -> 570 is the same decision taken at a different scale, and those
    want different fixes.
    """
    a = opcode_census(hlo_at(case_dump, at, module=module))
    b = opcode_census(hlo_at(control_dump, at, module=module))
    rows = [(k, a.get(k, 0), b.get(k, 0), a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b)]
    rows.sort(key=lambda r: (-abs(r[3]), r[0]))
    return {
        "at": at,
        "delta": rows[:top] if top else rows,
        "case_only": sorted(k for k in a if k not in b),
        "control_only": sorted(k for k in b if k not in a),
        "case_total": sum(a.values()),
        "control_total": sum(b.values()),
        "case_census": dict(a.most_common()),
        "control_census": dict(b.most_common()),
    }


# ══════════════════════════════════════════════════════════════════════════════════════════════
# WHAT CHANGED, AT ONE BOUNDARY
#
# `diverge` names the pass. That is where the instrument used to stop, and it is one sentence short
# of a bug report: "fusion took the case 2,458 -> 176,189" is a fact about a number, not about a
# program. Several of the thirty investigations closed the gap BY HAND, and always with the same
# three views -- the opcode census (static `slice` 46,268 vs 570 while `dynamic-slice` went 294 ->
# 0), the computation sizes (306 computations vs 202), and the operand arity (max 289 vs 55).
# Those three are what `boundary_diff` returns.
#
# WHAT IT DOES NOT RETURN, AND WILL NOT.  Instruction-level LINEAGE. XLA renames instructions at
# will -- a fused instruction is a new `HloInstruction` with a generated name, `fusion.4711`, and
# nothing anywhere records that it came from `%multiply.203` and `%add.204`. There is no lineage
# field, no ancestry table, no naming rule that survives a pass. So "this instruction became that
# one" IS NOT DERIVABLE from a dump, and a report that implied it would be inventing evidence.
# Every number here is an AGGREGATE over a population, which is sound, and the one thing that
# looks like lineage -- a computation or instruction NAME appearing in one arm and not the other --
# is qualified in the returned data by `name_overlap`: measure how many names the two arms share
# before reading anything into which names they do not.
#
# THE ONE RESTRICTED CASE WHERE NAMES DO CARRY INFORMATION, scoped tightly: two arms of the SAME
# program at the SAME boundary derive their names from the same lowering, so a name present in both
# is very likely the same instruction. That licenses set arithmetic over names when the overlap is
# high, and nothing more. It NEVER licenses matching a name in one arm to a DIFFERENT name in the
# other, it says nothing about instructions XLA created (`fusion.N`, `copy.N`, `bitcast.N` are
# numbered by a per-module counter that shifts when anything upstream shifts), and it collapses
# entirely when the overlap is low -- which `boundary_diff` reports rather than assumes.
# ══════════════════════════════════════════════════════════════════════════════════════════════

_ARITY_BUCKETS = ((0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64),
                  (65, 128), (129, 256), (257, 1 << 30))

# A computation name XLA MADE UP rather than one that came down from the program lives in
# `scopex._parse` with the names it was measured against, like every other pattern over compiler
# output. Measured on bisect_m94 vs its control at the fusion boundary: computation-name overlap
# 96.8%, and yet 115 names appear in the case alone -- every one of them a fusion the control's
# counter never reached. Hence the name lists in `boundary_diff` carry the generated fraction.
_generated = _parse.generated_computation_name


def _bucket(n: int) -> str:
    for lo, hi in _ARITY_BUCKETS:
        if lo <= n <= hi:
            return f"{lo}" if lo == hi else (f"{lo}+" if hi == 1 << 30 else f"{lo}-{hi}")
    return "?"                                                               # pragma: no cover


def _profile(text: str, *, top: int = 12) -> dict:
    """One module, walked ONCE, into the three populations :func:`boundary_diff` compares.

    Every instruction is counted into all three -- an opcode, a computation, an arity bucket -- so
    the three totals must be equal and must equal the module's instruction count. They are three
    partitions of one population, and `boundary_diff` checks them against each other and against
    the independent line-pattern count of the same text. A view that quietly drops instructions
    (an opcode filter that misses an enum member, a computation the walk cannot enter) shows up as
    a total that no longer matches.
    """
    import heapq

    from .levels import hlo_module, opcode_of
    m = hlo_module(text)
    opcodes: collections.Counter = collections.Counter()
    arity: collections.Counter = collections.Counter()
    comps: list[tuple[str, int, str]] = []
    names: set[str] = set()
    widest: list[tuple[int, str, str, str]] = []
    total = arity_sum = 0
    for comp in m.computations():
        ins = comp.instructions()
        local: collections.Counter = collections.Counter()
        for i in ins:
            op = opcode_of(i)
            opcodes[op] += 1
            local[op] += 1
            # `operands` is a METHOD on HloInstruction, not a property -- `len(i.operands)` raises
            # `object of type 'nanobind.nb_bound_method' has no len()`, which at least fails loudly.
            # `i.operands` in a truth test would not have.
            k = len(i.operands())
            arity[k] += 1
            arity_sum += k
            names.add(i.name)
            item = (k, comp.name, i.name, op)
            if len(widest) < top:
                heapq.heappush(widest, item)
            elif item > widest[0]:
                heapq.heapreplace(widest, item)
        total += len(ins)
        comps.append((comp.name, len(ins), local.most_common(1)[0][0] if local else ""))
    comps.sort(key=lambda c: (-c[1], c[0]))
    def pct(q: float):
        """Percentile straight off the histogram -- never materialise one int per instruction."""
        if not total:
            return None
        want, seen = q * total, 0
        for k in sorted(arity):
            seen += arity[k]
            if seen >= want:
                return k
        return max(arity)                                                    # pragma: no cover

    return {
        "instrs": total, "opcodes": opcodes, "arity": arity, "arity_sum": arity_sum,
        "computations": comps, "names": names,
        "widest": [tuple(w) for w in sorted(widest, reverse=True)],
        "arity_max": max(arity) if arity else 0,
        "arity_mean": round(arity_sum / total, 3) if total else 0.0,
        "arity_p50": pct(0.5), "arity_p99": pct(0.99),
        "regex_instrs": _regex_count(text),
    }


def boundary_diff(case_dump, control_dump, *, at: str, module: str | None = None,
                  top: int = 12) -> dict:
    """WHAT is different between two arms at one boundary -- not just how much.

    ``at`` is ``"before"``, ``"after"``, or a pass name (see :func:`boundaries_in`); it selects the
    same boundary in both dumps and ``resolved`` proves which file that was in each. Use the pass
    :func:`diverge` names, translated through ``after_pass`` -- a snapshot is
    ``after_A.before_B`` and holds the module BETWEEN them, so the pass that CAUSED a jump into a
    snapshot is its ``after_pass``, never the ``before_pass`` its name is built from.

    Returns, all of it aggregate and none of it lineage (see the block comment above)::

        instrs        case / control / ratio at this boundary
        opcodes       delta (biggest absolute difference first), `appeared` (case has it, control
                      has none), `vanished` (control has it, case has none), both censuses
        computations  counts, the largest in each arm with its dominant opcode, the size histogram,
                      and `name_overlap` -- read `only_in_case` only if that is high
        arity         operand-count max / mean / p50 / p99 per arm, a bucketed histogram with the
                      per-bucket delta, and the widest instructions in each arm
        self_check    the four totals that must agree, and whether they do
        caveats       the sentences that qualify what is above, when they apply

    ``self_check`` is not decoration. The opcode census, the computation sizes and the arity
    histogram are three partitions of the SAME instructions, so their totals must be equal; and the
    line-pattern count of the same text is a fourth, independent reading. A number here that
    disagrees means one of the four views is dropping instructions, and the whole report inherits
    it -- which is the failure mode this package has shipped three times.
    """
    res = {k: resolve_boundary(d, at, module=module)
           for k, d in (("case", case_dump), ("control", control_dump))}
    a = _profile(pathlib.Path(res["case"]["path"]).read_text(errors="replace"), top=top)
    b = _profile(pathlib.Path(res["control"]["path"]).read_text(errors="replace"), top=top)

    caveats: list[str] = []
    if res["case"]["after_pass"] != res["control"]["after_pass"]:
        caveats.append(
            f"THE TWO ARMS WERE READ AT DIFFERENT PASSES: {at!r} resolved to after_"
            f"{res['case']['after_pass']} in the case and after_{res['control']['after_pass']} in "
            f"the control. Every number below compares two different points in the pipeline. Name "
            f"the boundary exactly, or compare at 'before'/'after'.")
    if res["case"]["how"] not in ("exact-name", "optimization-boundary"):
        caveats.append(
            f"{at!r} was matched by {res['case']['how']}, not exactly: the case's boundary is the "
            f"snapshot named {res['case']['name']!r}, which holds the module AFTER the pass "
            f"{res['case']['after_pass']!r} and BEFORE {res['case']['before_pass']!r}. Those are "
            f"different passes and only one of them did whatever you are looking at.")

    # ── the three populations ───────────────────────────────────────────────────────────────────
    oa, ob = a["opcodes"], b["opcodes"]
    rows = [(k, oa.get(k, 0), ob.get(k, 0), oa.get(k, 0) - ob.get(k, 0)) for k in set(oa) | set(ob)]
    rows.sort(key=lambda r: (-abs(r[3]), r[0]))

    ca, cb = dict((n, s) for n, s, _ in a["computations"]), dict(
        (n, s) for n, s, _ in b["computations"])
    shared = set(ca) & set(cb)
    only_case = sorted(((n, ca[n]) for n in ca if n not in cb), key=lambda t: -t[1])
    only_ctrl = sorted(((n, cb[n]) for n in cb if n not in ca), key=lambda t: -t[1])
    def _order(s: str) -> int:
        return int(s.rstrip("+").split("-")[0])

    csize = {}
    for label, comps in (("case", a["computations"]), ("control", b["computations"])):
        h: collections.Counter = collections.Counter()
        for _, s, _op in comps:
            h[_bucket(s)] += 1
        csize[label] = dict(sorted(h.items(), key=lambda kv: _order(kv[0])))

    buckets = sorted({_bucket(k) for k in list(a["arity"]) + list(b["arity"])}, key=_order)
    ah: collections.Counter = collections.Counter()
    bh: collections.Counter = collections.Counter()
    for k, v in a["arity"].items():
        ah[_bucket(k)] += v
    for k, v in b["arity"].items():
        bh[_bucket(k)] += v

    # ── the check on the report itself ──────────────────────────────────────────────────────────
    sc = {}
    for label, p in (("case", a), ("control", b)):
        totals = {"instructions": p["instrs"], "opcode_census": sum(p["opcodes"].values()),
                  "computation_sizes": sum(s for _, s, _o in p["computations"]),
                  "arity_histogram": sum(p["arity"].values()), "line_pattern": p["regex_instrs"]}
        sc[label] = totals | {"agree": len(set(totals.values())) == 1}
        if not sc[label]["agree"]:
            caveats.append(
                f"{label}: the views of this boundary do not agree on how many instructions it has "
                f"({totals}). Three of those are partitions of one population and the fourth is an "
                f"independent reading of the same bytes; a mismatch means at least one view here "
                f"is dropping instructions and every number derived from it is short.")

    n_overlap = len(a["names"] & b["names"])
    comp_overlap = round(len(shared) / max(1, min(len(ca), len(cb))), 4)
    instr_overlap = round(n_overlap / max(1, min(len(a["names"]), len(b["names"]))), 4)
    gen_case = sum(1 for n, _ in only_case if _generated(n))
    gen_ctrl = sum(1 for n, _ in only_ctrl if _generated(n))
    gen_frac = round((gen_case + gen_ctrl) / max(1, len(only_case) + len(only_ctrl)), 4)
    if gen_frac > 0.5 and (only_case or only_ctrl):
        caveats.append(
            f"{gen_frac:.0%} of the {len(only_case) + len(only_ctrl)} computations named in one "
            f"arm "
            f"and not the other carry XLA-GENERATED names (fused_computation.N, *.clone, "
            f"*_fusion). Those are numbered by a per-module counter, so a name missing from one "
            f"arm "
            f"means the counter stopped earlier -- NOT that the computation is absent. Read "
            f"`size_buckets` and `case_largest`, which need no name matching at all; on bisect_m94 "
            f"the sound form of this finding is '189 computations of 257+ instructions against the "
            f"control's 3', and it does not depend on a single name.")
    if comp_overlap < 0.5:
        caveats.append(
            f"the two arms share only {comp_overlap:.0%} of their computation names, so name-based "
            f"set arithmetic is not informative here at all -- the two modules were named "
            f"independently. Read the SIZES and the opcode mix.")
    caveats.append(
        "no field here maps an instruction in one arm to an instruction in the other. XLA renames "
        "freely and records no lineage, so that mapping is not derivable from a dump at all.")

    return {
        "at": at, "resolved": res,
        "instrs": {"case": a["instrs"], "control": b["instrs"],
                   "ratio": round(a["instrs"] / max(1, b["instrs"]), 2)},
        "opcodes": {
            "delta": rows[:top] if top else rows,
            "appeared": sorted(((k, oa[k]) for k in oa if k not in ob), key=lambda t: -t[1]),
            "vanished": sorted(((k, ob[k]) for k in ob if k not in oa), key=lambda t: -t[1]),
            "case_census": dict(oa.most_common()), "control_census": dict(ob.most_common()),
            "case_total": sum(oa.values()), "control_total": sum(ob.values())},
        "computations": {
            "case_n": len(ca), "control_n": len(cb),
            "case_largest": a["computations"][:top], "control_largest": b["computations"][:top],
            "size_buckets": csize,
            "only_in_case": only_case[:top], "only_in_control": only_ctrl[:top],
            "n_only_in_case": len(only_case), "n_only_in_control": len(only_ctrl),
            "name_overlap": comp_overlap, "only_generated_fraction": gen_frac},
        "arity": {
            "case_max": a["arity_max"], "control_max": b["arity_max"],
            "case_mean": a["arity_mean"], "control_mean": b["arity_mean"],
            "case_p50": a["arity_p50"], "control_p50": b["arity_p50"],
            "case_p99": a["arity_p99"], "control_p99": b["arity_p99"],
            "operand_edges": {"case": a["arity_sum"], "control": b["arity_sum"]},
            "histogram": [(k, ah.get(k, 0), bh.get(k, 0), ah.get(k, 0) - bh.get(k, 0))
                          for k in buckets],
            "case_widest": a["widest"], "control_widest": b["widest"]},
        "names": {"instruction_overlap": instr_overlap,
                  "shared_instruction_names": n_overlap,
                  "case_instruction_names": len(a["names"]),
                  "control_instruction_names": len(b["names"])},
        "self_check": sc,
        "lineage": None,
        "caveats": caveats,
    }


_PROBE_SRC = '''"""scopex selftest probe -- a marked program with real user frames."""
import jax
import jax.numpy as jnp


def leaf(x):
    return jnp.tanh(x)


def body(x):
    with jax.named_scope("scopex:user.Probe.body"):
        return leaf(x) * 2.0


def program(x):
    with jax.named_scope("scopex:lib.selftest"):
        return jnp.sum(body(x) @ x)
'''


def _probe_program():
    """``(program, path)`` for a marked probe living in a real file outside this package.

    Nested calls on purpose: ``program -> body -> leaf`` gives the optimized module a stack-frame
    chain several frames deep, so :func:`scopex.levels.hlo_sites` is exercised on a chain and not
    just on a leaf. A one-frame probe passes even when the parent links are read wrongly.
    """
    import importlib.util
    d = pathlib.Path(tempfile.mkdtemp(prefix="scopex-selftest-"))
    p = d / "scopex_selftest_probe.py"
    p.write_text(_PROBE_SRC)
    spec = importlib.util.spec_from_file_location("scopex_selftest_probe", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.program, str(p)


def selftest(verbose: bool = True, *, strict: bool = True) -> dict:
    """Run every parser in scopex against BOTH its frozen sample and a freshly compiled program.

    Two halves, and both are needed:

    * :func:`scopex._parse.conformance` replays each parser over a verbatim capture of the text it
      was written for. That half needs no jax, so it separates "somebody edited the parser" from
      "the compiler moved".
    * this function compiles a small MARKED program, dumps it, and checks that every level and every
      artifact view comes back NON-EMPTY, that the levels still agree with each other, and that the
      HLO stack-frame tables still resolve to source lines the jaxpr also reports.

    ``strict=True`` (the default) RAISES on any failure. That is the point: this package has three
    times shipped a parser that returned a plausible empty answer, and each time the answer was
    believed. Run it after any jax upgrade -- and before the first compile in the process, since XLA
    reads its dump flags when the backend is first initialised.
    """
    import jax
    import jax.numpy as jnp

    from .flags import backend_initialized, dump, hlo_text, stablehlo_text
    from .levels import frame_tables, hlo_instructions, walk_hlo, walk_stablehlo
    from .walk import NO_FRAME, walk

    if backend_initialized():
        raise RuntimeError("run selftest before the first compile in the process")

    r: dict = {}
    bad: list[str] = []

    # ── half one: the frozen samples, no jax involved ───────────────────────────────────────────
    try:
        r["conformance"] = _parse.conformance()["ok"]
    except Exception as e:
        r["conformance"] = False
        bad.append(f"embedded-sample conformance: {str(e).splitlines()[0]}")

    # ── half two: a real compile ────────────────────────────────────────────────────────────────
    # The probe is written to a FILE OUTSIDE this package and imported, rather than defined here.
    # That is load-bearing: both site resolvers filter out frames inside jax and inside scopex, so a
    # probe defined in this module has no user frame at all and the cross-level site join -- the one
    # check that can catch a frame table resolving to the WRONG line rather than to none -- would
    # compare two empty sets and pass.
    program, probe_file = _probe_program()

    with dump(passes=".*", fusion=False, keep=True) as d:
        low = jax.jit(program).lower(jnp.ones((32, 32)))
        c = low.compile()
    r["dump_dir"] = d

    def check(name, fn, *, count=True):
        try:
            v = fn()
        except Exception as e:
            bad.append(f"{name}: raised {type(e).__name__}: {str(e).splitlines()[0]}")
            return None
        r[name] = len(v) if count and hasattr(v, "__len__") else v
        if not v:
            bad.append(f"{name}: EMPTY, from a program that demonstrably has some")
        return v

    check("modules", lambda: modules_in(d))
    check("pass_steps", lambda: pass_growth(d))
    check("timeline_entries", lambda: pass_timeline(d))
    r["codegen"] = codegen_size(d)
    r["custom_calls"] = dict(custom_calls(c))          # legitimately empty on this program

    text = hlo_text(c)
    eqns = list(walk(jax.make_jaxpr(program)(jnp.ones((32, 32)))))
    sh = check("stablehlo_units", lambda: list(walk_stablehlo(low)))
    hl = check("hlo_units", lambda: list(walk_hlo(c)))
    check("hlo_instructions", lambda: list(hlo_instructions(c)))
    check("stablehlo_chars", lambda: len(stablehlo_text(low)), count=False)
    tab = check("frame_tables", lambda: frame_tables(text)["frames"])
    r["parent_offset"] = frame_tables(text)["parent_offset"]

    # The checks a count cannot make. Each one is a way a parser here has actually failed:
    if hl:
        if not any(i.path for i in hl):
            bad.append("hlo_units: not one instruction carries an op_name. The metadata parse is "
                       "returning empty dicts -- which is how the optimized module came to be "
                       "written up as carrying no provenance at all")
        if tab and not any(i.site not in (NO_FRAME, "?", "") for i in hl):
            bad.append("hlo_units: the module HAS stack-frame tables and not one instruction "
                       "resolved to a file:line. stack_frame_id is being dropped (bug #2)")
        if not any("scopex:user.Probe.body" in i.path for i in hl):
            bad.append("hlo_units: the marked scope reached zero optimized instructions -- either "
                       "the name stack is not surviving lowering or op_name is not being read")
        # THE ONE CHECK THAT CATCHES A PARSER RESOLVING TO THE WRONG ANSWER RATHER THAN TO NONE.
        # Both levels resolve a site independently -- the jaxpr from python tracebacks, the
        # optimized HLO by walking XLA's frame tables -- so they must land on the same lines of the
        # probe file. If the parent links are read with the wrong convention the HLO side still
        # produces plausible file:line pairs; they are just somebody else's.
        sites = {i.site for i in hl if i.site not in (NO_FRAME, "?", "")}
        jaxpr_sites = {e.site for e in eqns if e.site != NO_FRAME}
        probe = {s for s in sites if s.startswith(probe_file)}
        r["site_join"] = round(len(sites & jaxpr_sites) / max(1, len(sites)), 4)
        if not jaxpr_sites or not sites:
            bad.append(f"site join is untestable: {len(sites)} HLO sites, {len(jaxpr_sites)} jaxpr "
                       f"sites. One of the two site resolvers returned nothing at all")
        elif not probe:
            bad.append(f"hlo_units: not one instruction resolved into the probe file {probe_file}; "
                       f"got {sorted(sites)[:3]}. The frame tables resolve to the WRONG frames")
        elif not (sites & jaxpr_sites):
            bad.append(
                f"hlo_units: no optimized-HLO site matches any jaxpr site "
                f"({sorted(sites)[:2]} vs {sorted(jaxpr_sites)[:2]}). The frame tables are "
                f"resolving to the WRONG frames, which no emptiness check can see -- re-derive the "
                f"parent_frame_id convention in scopex/_parse.py:_parent_offset.")
    if sh:
        r["stablehlo_named"] = sum(1 for i in sh if i.path)
        if not r["stablehlo_named"]:
            bad.append("stablehlo_units: walked operations but not one carries a name stack. This "
                       "is bug #1 -- the level looks empty rather than broken")
        if len(sh) < 4:
            bad.append(f"stablehlo_units: {len(sh)} units from a module containing a tanh, a dot "
                       f"and a reduce. That is the shape of bug #1")

    r["ok"] = not bad
    r["broken"] = bad
    if verbose:
        print(f"scopex.selftest: {'OK' if not bad else 'FAILED'}")
        for k, v in r.items():
            if k != "broken":
                print(f"  {k:18s} {v}")
        for b in bad:
            print(f"  BROKEN: {b}")
    if bad:
        msg = ("scopex.selftest FAILED -- these parsers no longer read what jax/XLA emit:\n  - "
               + "\n  - ".join(bad)
               + f"\nDump kept at {d} for inspection. Every parser lives in scopex/_parse.py next "
                 "to the sample it was written against; fix it there and run selftest() again.")
        if strict:
            raise _parse.ParseError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return r
