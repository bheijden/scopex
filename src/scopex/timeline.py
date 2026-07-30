"""The mtime timeline, and the overlap agreement that decides whether to believe its tail.

``scopex.pass_timeline`` reconstructs where a compile's seconds went from the mtimes of the files
XLA writes under ``--xla_dump_to``. It is the only instrument in this package that can see BELOW
the HLO pass pipeline -- the interval after the last HLO snapshot, which is emitter + LLVM +
codegen and which is where several corpus pathologies actually live. It is also, until this module,
plausible-looking and unvalidated, which is the dangerous state this package keeps paying for.

THE DESIGN THAT EARNS THE TAIL
------------------------------
The mtime clock and XLA's own VLOG pass timer BOTH see the HLO passes. Make them agree there, and
only then trust the mtime clock where VLOG cannot reach. So every number here ships with an
agreement measure computed on the overlap, and the tail carries an ERROR BOUND derived from it
rather than an adjective.

WHAT A SNAPSHOT MTIME ACTUALLY MEANS, from xla/hlo/pass/hlo_pass_pipeline.cc:138-225::

    for (i, pass) in passes:
        XLA_SCOPED_LOGGING_TIMER("HLO pass: " + pass_name);   # logs at scope EXIT     (line 176)
        VLOG(1) << "  HLO pass " << pass_name;                # logs at pass START     (line 181)
        RunHelper(pass, hlo)
        if (!dump_regex.empty() && (pass_changed || dump_regex != ".*"))
            MaybeDumpHloAndSaveFilenames(hlo, after=pass_name, before=passes[i+1]->name())
        RecordPassEndMetadata(...)
        if (pass_changed) RunInvariantCheckers(...)           # logs at lines 86/89

Four consequences, each of which the obvious reading gets wrong:

1. The snapshot is written AFTER its pass runs and is named ``after_<that pass>``. So
   ``mtime(after_P)`` is an instant strictly INSIDE pass P's timer, near its end -- not a pass
   boundary, and not the start of the next pass.
2. Both log lines carry a glog wall-clock timestamp at microsecond resolution, taken from
   CLOCK_REALTIME -- the SAME clock ``st_mtime`` comes from. The two clocks are therefore
   comparable instant-against-instant, which yields a falsifiable per-snapshot test:
   ``mtime(after_P)`` must lie inside ``[start_P, end_P]``. That test validates the ALIGNMENT
   itself, so nothing here rests on trusting that a filename and a log line refer to the same pass.
   Measured: 100.0% containment on 12 compiles across 5 programs, 0 violations out of 683
   matched snapshots, including a 4.7x-overloaded machine.
3. The scoped timer covers the WHOLE loop iteration, so a pass's reported time INCLUDES the write
   of its own snapshot and its invariant checkers. The dump I/O sits inside BOTH clocks, which is
   why they are comparable at all -- and why a dense dump inflates the passes it is timing.
4. With ``xla_dump_hlo_pass_re`` EXACTLY ``".*"`` the dump guard reduces to ``pass_changed``, so
   only passes that CHANGED the module are snapshotted. With any OTHER regex the guard is
   ``true``. ``".+"`` therefore dumps STRICTLY MORE than ``".*"`` -- same matched passes, no
   `changed` filter. Measured on one control program: ``.*`` -> 26 snapshots, ``.+`` -> 160.
   This is not a documented XLA feature and reads backwards; it is in the source above.

TWO ARTIFACT CLASSES SHARE THE DIRECTORY AND MUST NOT BE CONFUSED
-----------------------------------------------------------------
* BETWEEN-pass  ``NNNN.<pipeline>.after_<P>.before_<Q>.txt``  DumpHloModuleBetweenPassesIfEnabled
* DURING-pass   ``NNNN.<pass>.<step>.txt``                    DumpHloModuleDuringPassIfEnabled

The second has NO ``before_`` component and its second field is a PASS name, not a pipeline name.
``copy-insertion`` writes three of them per compile on CPU. They are real timestamped boundaries
INSIDE a single pass, so treating them as pass boundaries invents three passes that never ran.

NESTED PIPELINES DOUBLE COUNT
-----------------------------
A pass can itself be an ``HloPassPipeline``, and it gets its own ``HLO pass:`` line whose time
CONTAINS every pass inside it. Summing all ``HLO pass:`` lines -- which is what a by-name aggregate
does -- therefore counts those intervals twice. Measured on the corpus's gather case: 143 pass
lines, of which 3 are pipelines, and the naive sum reads 9.95 ms against a true leaf sum of
7.09 ms, a 40% overcount. Leaf-ness here is derived from the log's own structure -- a pass is an
aggregate iff a pipeline header opens under it -- not from a hand-maintained list of which names
happen to be pipelines, and NOT from 'did it have children': an EMPTY pipeline has none and is
still an aggregate.
"""

from __future__ import annotations

import datetime as _dt
import os
import pathlib
import statistics
import subprocess
import sys

from . import _parse
from ._parse import (
    dump_snapshot_name,
    emitter_dump_name,
    glog_prefix,
    pass_leaf_split,
    pass_timing_lines,
)

__all__ = ["pass_timeline", "timeline_agreement", "PassTimeline"]

PIPELINE_START = "pipeline-start"

# Every pattern that reads compiler output lives in scopex._parse, next to a verbatim sample and
# the component that printed it. This module owns none: `glog_prefix` gives the timestamp, thread
# and source line; `_ANNOUNCE` / `_PIPELINE_LINE` / `pass_timing_lines` give the three kinds of
# pass event. Line 176 is the scoped timer firing at pass END and line 181 is the announcement at
# pass START -- their text differs only by a colon, so this module keys on the SOURCE LINE NUMBER
# rather than on a text pattern of its own.
_LINE_END, _LINE_START = 176, 181

# Terminal codegen artifacts, newest-last. `.ptx` is XLA:GPU's object form -- omitting it was
# the same blind spot that made codegen_size report 0 bytes on CUDA. Measured on a CUDA dump:
# the last .ptx lands 21 ms BEFORE the last .ir-with-opt.ll, so the tail was not actually
# truncated there; it is listed because the next backend need not be so forgiving.
_LLVM = ((".ir-no-opt.ll", "ir_no_opt"), (".ir-with-opt.ll", "ir_with_opt"),
         (".o", "obj"), (".ptx", "ptx"))


def _norm(s):
    """Comparison key bridging the dump filename's spelling and the log's.

    ``dump.cc`` sanitises spaces to underscores, so the log's ``HLO passes through layout
    assignment`` is ``HLO_passes_through_layout_assignment`` on disk, while
    ``post_scatter_expansion_simplification`` has real underscores in both. The map is many-to-one
    and is used ONLY as a comparison key -- never to recover a name, and never as the label shown
    to a caller, because two distinct pipelines could in principle collapse onto one key.
    """
    return (s or "").replace("_", " ").strip()


# ══════════════════════════════════════════════════════════════════════════════════════════════
# THE MTIME SIDE
# ══════════════════════════════════════════════════════════════════════════════════════════════

def snapshots(dump_dir, module=None):
    """Per-pass snapshots with mtimes, in pipeline order, tagged by which dump call wrote them."""
    out = []
    for f in os.listdir(dump_dir):
        m = dump_snapshot_name(f)
        if not m or (module and m["module"] != module):
            continue
        st = (pathlib.Path(dump_dir) / f).stat()
        out.append(dict(index=m["index"], pipeline=m["pipeline"], after=m["after_pass"],
                        before=m["before_pass"] or None, mtime=st.st_mtime,
                        mtime_ns=st.st_mtime_ns, bytes=st.st_size, file=f, module=m["module"],
                        kind=("pipeline-start" if m["after_pass"] == PIPELINE_START
                              else "during-pass" if not m["before_pass"] else "between-pass")))
    out.sort(key=lambda s: s["index"])
    return out


def measured_resolution(snaps):
    """The mtime clock's ACHIEVED resolution on this dump, measured, never assumed.

    Two very different regimes exist on the same machine and the difference is not academic:

    * A synthetic writer creating files in a tight loop with no intervening ``stat`` gets the
      COARSE timestamp clock. Measured on ext4 / linux 6.17: 3000 files -> 84 distinct mtimes,
      quantum exactly 1,000,004 ns. 97.3% of consecutive deltas were ZERO.
    * XLA's real dump path gets fine-grained timestamps. Measured on the same filesystem minutes
      later: 26/26 and 160/160 snapshot mtimes distinct, minimum non-zero delta 110-157 us,
      residues mod 1 ms uniformly spread.

    Which regime you are in depends on the kernel, the filesystem and XLA's write path, so this
    reports what the data shows: ties are the symptom that matters, and an NFS or a 1-second
    granularity filesystem would make every number in the timeline meaningless while still
    returning a plausible dict.
    """
    ns = sorted(s["mtime_ns"] for s in snaps)
    d = [b - a for a, b in zip(ns, ns[1:], strict=False)]
    nz = [x for x in d if x > 0]
    return {
        "n_snapshots": len(ns),
        "n_distinct_mtimes": len(set(ns)),
        "n_ties": sum(1 for x in d if x == 0),
        "min_nonzero_delta_s": (min(nz) / 1e9) if nz else None,
        "median_nonzero_delta_s": (statistics.median(nz) / 1e9) if nz else None,
        # An mtime clock that cannot separate two consecutive snapshots cannot time the pass
        # between them; ties are how that shows up in the data rather than in the seconds.
        "resolution_s": (min(nz) / 1e9) if nz else None,
    }


def kernel_modules(dump_dir):
    """LLVM kernel modules and whether their phases interleave.

    XLA:CPU emits, optimises and object-codegens each kernel module as a unit and does several
    CONCURRENTLY, so a GLOBAL phase boundary (max mtime over every ``.ir-no-opt.ll``, then over
    every ``.ir-with-opt.ll``) describes a real ordering only when there is exactly one kernel.
    Measured: the corpus gather case has 1 and the split is sound; ndtri jacrev d4 has 223 with
    every consecutive pair overlapping, and there the split is NOT DEFINED.
    """
    kern = {}
    for f in os.listdir(dump_dir):
        e = emitter_dump_name(f)
        if e and e["kind"] in ("ir-no-opt", "ir-with-opt", "obj"):
            kern.setdefault((e["module"], e["kernel"]), {})[e["kind"]] = \
                (pathlib.Path(dump_dir) / f).stat().st_mtime
    no = [v["ir-no-opt"] for v in kern.values() if "ir-no-opt" in v]
    wo = [v["ir-with-opt"] for v in kern.values() if "ir-with-opt" in v]
    ob = [v["obj"] for v in kern.values() if "obj" in v]
    inter = bool((wo and no and min(wo) < max(no)) or (ob and wo and min(ob) < max(wo)))
    return {"n_kernel_modules": len(kern), "interleaved": inter, "_kern": kern}


# ══════════════════════════════════════════════════════════════════════════════════════════════
# THE VLOG SIDE
# ══════════════════════════════════════════════════════════════════════════════════════════════

def parse_log(log, year):
    """Ordered pass events with absolute CLOCK_REALTIME timestamps, TAGGED BY THREAD.

    The year is supplied by the caller because glog prints ``MMDD`` and no year. A compile
    spanning midnight on 31 December would be misdated by a year rather than by a second; that is
    the one input this function cannot check for itself, so it is stated rather than guarded.
    """
    ev = []
    for line in log.splitlines():
        g = glog_prefix(line)
        if not g:
            continue
        try:
            base = _dt.datetime(year, g["month"], g["day"]).timestamp()
        except ValueError:                                                     # pragma: no cover
            continue
        t = base + g["secs_of_day"]
        rest, ln, tid = g["rest"], g["line"], g["tid"]
        mh = _parse._PIPELINE_LINE.search(rest)
        if mh:
            ev.append(dict(kind="header", t=t, tid=tid, module=mh.group("module"),
                           pipeline=mh.group("pipeline")))
            continue
        if ln == _LINE_END:
            pt = pass_timing_lines(rest)
            if pt:
                ev.append(dict(kind="end", t=t, tid=tid, name=pt[0].name, secs=pt[0].seconds))
            continue
        if ln == _LINE_START:
            # _ANNOUNCE is anchored on the `] ` that ends the glog prefix, which glog_prefix has
            # already stripped; re-supplying it keeps the pattern in the quarantine unchanged.
            ma = _parse._ANNOUNCE.search("] " + rest)
            if ma:
                ev.append(dict(kind="start", t=t, tid=tid, name=ma.group("name")))
    return ev


def structure(ev):
    """One record per pass INVOCATION, with true start/end and leaf-ness from nesting depth.

    Starts and ends nest perfectly -- a nested pipeline's own start precedes its header and its own
    end follows all of its children -- so a LIFO over start events recovers the tree without any
    knowledge of which pass names happen to be pipelines. A pass is a LEAF iff no other pass ended
    inside it, and only leaves may be summed.

    THE ORDER IS ONLY AN ORDER WITHIN ONE THREAD, so the stack is per-thread. glog interleaves
    every thread into one stderr, and XLA's GPU autotuner compiles its candidate modules in
    parallel -- measured elsewhere in this package at 21 threads on one conv arm, where a
    top-to-bottom stack machine matches an announcement from one thread against a timing from
    another and mislabels leaves as aggregates. The CPU compiles used to validate this module
    already interleave two threads (313 lines and 3), so this is not a GPU-only concern.
    """
    stacks: dict = {}
    out, pipes = [], []
    for e in ev:
        tid = e.get("tid", "")
        if e["kind"] == "header":
            pipes.append((e["t"], e["module"], e["pipeline"], tid))
            st = stacks.get(tid) or []
            # The pass that is about to become this pipeline is the one already announced and
            # still open. Marking it here -- rather than inferring "it had children" -- is what
            # makes an EMPTY pipeline an aggregate too.
            if st and _norm(st[-1]["name"]) == _norm(e["pipeline"]):
                st[-1]["is_pipeline"] = True
        elif e["kind"] == "start":
            stacks.setdefault(tid, []).append(
                {"name": e["name"], "t_start": e["t"], "kids": 0, "is_pipeline": False})
        elif e["kind"] == "end":
            st = stacks.setdefault(tid, [])
            f = st.pop() if (st and st[-1]["name"] == e["name"]) \
                else {"name": e["name"], "t_start": None, "kids": 0, "is_pipeline": False}
            if st:
                st[-1]["kids"] += 1
            mod = pipe = None
            for t, m, p, ptid in pipes:
                if ptid == tid and (f["t_start"] is None or t <= f["t_start"]):
                    mod, pipe = m, p
            # LEAF-NESS IS "IS NOT ITSELF A PIPELINE", NOT "HAD NO CHILDREN". The two differ on a
            # pipeline that ran ZERO passes, which is not hypothetical: `after layout assignment`
            # is empty on the corpus gather case and its snapshot is
            # `.after_pipeline-start.before_pipeline-end`. A depth-based rule calls it a leaf and
            # adds its 145 us to the leaf sum, which is a 2.9% overcount on that compile and
            # breaks the `leaves + aggregates == every line` identity that makes the cross-check
            # against _parse.pass_leaf_split meaningful.
            out.append(dict(name=e["name"], t_start=f["t_start"], t_end=e["t"], secs=e["secs"],
                            leaf=not f["is_pipeline"], module=mod, pipeline=pipe, tid=tid))
    return out, [(t, m, p) for t, m, p, _ in pipes]


# ══════════════════════════════════════════════════════════════════════════════════════════════
# THE OVERLAP
# ══════════════════════════════════════════════════════════════════════════════════════════════

def _align(snaps, passes, pipes):
    """Match each snapshot to the event that wrote it, then VERIFY by timestamp containment.

    Matching is by name with a forward cursor; the containment test is what makes the match
    trustworthy. ``matched`` and ``inside`` are reported separately on purpose -- a run where they
    diverge is a run whose ALIGNMENT failed, which is a different and more serious thing than a run
    whose clocks merely disagree, and collapsing the two would hide it.
    """
    rows, cur, pcur = [], 0, 0
    for s in snaps:
        r = dict(s)
        if s["kind"] == "pipeline-start":
            hit = next((j for j in range(pcur, len(pipes))
                        if _norm(pipes[j][2]) == _norm(s["pipeline"])), None)
            if hit is None:
                r.update(matched=False, why="no pipeline header with this name")
            else:
                t = pipes[hit][0]
                r.update(matched=True, t_start=t, t_end=None, vlog_secs=None,
                         offset=s["mtime"] - t, inside=(s["mtime"] >= t))
                pcur = hit + 1
        else:
            # A during-pass dump names its enclosing PASS in the field a between-pass dump uses
            # for the pipeline, so that is where its name comes from -- and its cursor must not
            # advance, because several of them share one pass.
            want = _norm(s["pipeline"] if s["kind"] == "during-pass" else s["after"])
            hit = next((j for j in range(cur, len(passes))
                        if _norm(passes[j]["name"]) == want), None)
            if hit is None:
                r.update(matched=False, why="no VLOG pass with this name after the cursor")
            else:
                p = passes[hit]
                r.update(matched=True, vlog_name=p["name"], vlog_secs=p["secs"],
                         t_start=p["t_start"], t_end=p["t_end"], leaf=p["leaf"],
                         offset=s["mtime"] - p["t_end"],
                         inside=(p["t_start"] is not None
                                 and p["t_start"] <= s["mtime"] <= p["t_end"]))
                if s["kind"] == "between-pass":
                    cur = hit + 1
        rows.append(r)
    return rows


def _q(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo = int(k)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (k - lo)


def _agreement(rows, leaves, res):
    """Correlation, median ratio, worst disagreement, and what sits below the mtime resolution."""
    m = [r for r in rows if r.get("matched")]
    ins = [r for r in m if r.get("inside")]
    # BETWEEN-PASS ONLY, and the distinction is not pedantic. A during-pass dump is written from
    # the middle of a long pass by construction, so its distance to that pass's end timestamp is a
    # correct measurement, not a clock error -- on ndtri jacrev d4 it is 438 ms while the
    # between-pass offsets on the same compile top out at 17 ms. Folding the two together inflates
    # the error bound 25x and made the instrument report TOO SMALL TO TRUST on a tail it can
    # resolve to 1%.
    offs = [abs(r["offset"]) for r in m if r["kind"] == "between-pass"]
    offs_during = [abs(r["offset"]) for r in m if r["kind"] == "during-pass"]
    anchored = [r for r in rows if r.get("matched") and r.get("t_end")]
    gap = []
    for a, b in zip(anchored, anchored[1:], strict=False):
        sel = [p for p in leaves if a["t_end"] < p["t_end"] <= b["t_end"]]
        gap.append(dict(after=b["after"], dt=b["mtime"] - a["mtime"],
                        vlog=sum(p["secs"] for p in sel), n=len(sel)))
    ratios = [g["dt"] / g["vlog"] for g in gap if g["vlog"] > 0]
    corr = None
    if len(gap) > 2:
        dt = [g["dt"] for g in gap]
        vl = [g["vlog"] for g in gap]
        n = len(gap)
        mdt, mvl = sum(dt) / n, sum(vl) / n
        sdt = (sum((x - mdt) ** 2 for x in dt) / n) ** .5
        svl = (sum((x - mvl) ** 2 for x in vl) / n) ** .5
        if sdt > 0 and svl > 0:
            cov = sum((x - mdt) * (y - mvl) for x, y in zip(dt, vl, strict=False)) / n
            corr = cov / (sdt * svl)
    worst = max(gap, key=lambda g: abs(g["dt"] - g["vlog"])) if gap else None
    tot = sum(p["secs"] for p in leaves) or 1e-12
    r = res or 0.0
    return {
        "n_snapshots": len(rows), "n_matched": len(m), "n_unmatched": len(rows) - len(m),
        # ── does the alignment hold at all ──
        "frac_inside_pass_timer": (len(ins) / len(m)) if m else None,
        "n_outside_pass_timer": len(m) - len(ins),
        # ── the four the caller asked for ──
        "corr": corr,
        "median_ratio_mtime_over_vlog": statistics.median(ratios) if ratios else None,
        "worst_gap": worst,
        "worst_abs_disagreement_s": (abs(worst["dt"] - worst["vlog"]) if worst else None),
        "n_leaf_passes_below_resolution": sum(1 for p in leaves if p["secs"] < r),
        "frac_leaf_passes_below_resolution": (sum(1 for p in leaves if p["secs"] < r)
                                              / max(1, len(leaves))),
        "frac_leaf_TIME_below_resolution": sum(p["secs"] for p in leaves if p["secs"] < r) / tot,
        # ── span-level, and the per-boundary error the tail inherits ──
        "sum_mtime_gaps_s": sum(g["dt"] for g in gap),
        "sum_vlog_leaf_s": sum(g["vlog"] for g in gap),
        "span_ratio": (sum(g["dt"] for g in gap) / sum(g["vlog"] for g in gap)
                       if sum(g["vlog"] for g in gap) else None),
        "boundary_offset_p50_s": _q(offs, .5), "boundary_offset_max_s": max(offs) if offs else None,
        "n_boundaries_measured": len(offs),
        "during_pass_offset_max_s": max(offs_during) if offs_during else None,
        "n_leaf_passes": len(leaves), "leaf_sum_s": sum(p["secs"] for p in leaves),
        "naive_sum_all_pass_lines_s": None,   # filled by caller; needs the non-leaf set
    }


# ══════════════════════════════════════════════════════════════════════════════════════════════
# THE RESULT
# ══════════════════════════════════════════════════════════════════════════════════════════════

class PassTimeline(list):
    """``[(label, seconds), ...]`` -- and everything you need in order to distrust it.

    Subclasses ``list`` so the historical ``for label, secs in pass_timeline(d)`` keeps working.
    Read ``.verdict`` before reading any seconds.
    """

    resolution: dict = {}
    tail: dict = {}
    agreement: dict | None = None
    kernels: dict = {}
    warnings: list = []

    @property
    def tail_total(self) -> float:
        return self.tail.get("total_s", 0.0)

    @property
    def tail_usable(self) -> bool:
        return bool(self.tail.get("usable"))

    @property
    def verdict(self) -> str:
        return self.tail.get("verdict", "unvalidated: no VLOG supplied, agreement not computed")

    def __str__(self):
        out = [f"{'interval':52s} {'seconds':>9s}"]
        out.append("-" * len(out[0]))
        for lab, s in self:
            out.append(f"{lab[:52]:52s} {s:9.4f}")
        r, a, t = self.resolution, self.agreement, self.tail
        out.append("")
        out.append(f"mtime resolution   : {r.get('resolution_s')} s "
                   f"({r.get('n_ties')} ties in {r.get('n_snapshots')} snapshots)")
        if a:
            out.append(f"alignment          : {a['n_matched']}/{a['n_snapshots']} matched, "
                       f"{a['frac_inside_pass_timer']:.1%} inside their pass timer "
                       f"({a['n_outside_pass_timer']} violations)")
            out.append(f"agreement          : corr {a['corr']!s:.5s}  median ratio "
                       f"{a['median_ratio_mtime_over_vlog']:.3f}  span ratio {a['span_ratio']:.3f}")
            out.append(f"worst disagreement : {a['worst_abs_disagreement_s']*1e3:.3f} ms  "
                       f"({a['worst_gap']['after'] if a['worst_gap'] else '-'})")
            out.append(f"below resolution   : {a['frac_leaf_passes_below_resolution']:.1%} of "
                       f"passes, {a['frac_leaf_TIME_below_resolution']:.1%} of pass time")
        out.append(f"TAIL               : {t.get('total_s', 0.0):.4f} s   "
                   f"error bound +/- {t.get('error_bound_s', float('nan'))*1e3:.3f} ms   "
                   f"SNR {t.get('snr', 0):.1f}x")
        out.append(f"VERDICT            : {self.verdict}")
        for w in self.warnings:
            out.append(f"  ! {w}")
        return "\n".join(out)


def _tail(dump_dir, snaps, kern, agreement, resolution):
    """The interval the VLOG cannot reach, with an error bound taken from the overlap.

    THE TOTAL AND THE SPLIT ARE DIFFERENT CLAIMS. The total is one difference of two observed
    mtimes and is well defined however many kernel modules there are. The split into
    emitter / llvm_opt / codegen requires that the phases do not overlap, which needs exactly one
    kernel module -- with 223 of them compiling concurrently the per-phase boundaries are not
    orderings of anything.
    """
    if not snaps:
        return {"usable": False, "verdict": "no HLO snapshots: nothing to measure a tail from"}
    last = max(s["mtime"] for s in snaps)
    best = {}
    for f in os.listdir(dump_dir):
        for suf, lab in _LLVM:
            if f.endswith(suf):
                best[lab] = max(best.get(lab, 0.0), (pathlib.Path(dump_dir) / f).stat().st_mtime)
    t = {"from_mtime": last, "n_llvm_artifacts": len(best)}
    if not best:
        # NO .ll AND NO .o. Not "there was nothing below HLO" -- there is nothing to measure TO.
        return {**t, "total_s": 0.0, "usable": False,
                "verdict": ("no LLVM artifacts: everything after the last HLO pass is UNMEASURABLE "
                            "by mtime here, and 0.0 is an absence of evidence, not a measured "
                            "zero. Usual cause on CPU is a library kernel (kCustom fusion) rather "
                            "than an emitted LLVM module.")}
    prev = last
    single = kern["n_kernel_modules"] == 1 and not kern["interleaved"]
    for lab in ("ir_no_opt", "ir_with_opt", "obj"):
        if lab in best:
            if single:
                t[f"{lab}_s"] = best[lab] - prev
            prev = best[lab]
    t["total_s"] = prev - last
    t["split_defined"] = single
    t["n_kernel_modules"] = kern["n_kernel_modules"]

    # The tail is ONE difference of two mtimes, so it inherits the PER-BOUNDARY error of the mtime
    # clock -- NOT the span-level ratio, which is about what the VLOG attributes rather than about
    # when files were written. That distinction is the whole reason the tail survives a span_ratio
    # of 1.35 while remaining accurate to a fraction of a millisecond.
    #
    # The bound is the worst observed distance between a between-pass snapshot's mtime and the
    # independent microsecond timestamp of the same event. It is empirical and all-in: any clock
    # quantisation, filesystem coarseness or scheduling delay is already inside it, so `resolution`
    # is NOT added on top (doing so double counts, and on ndtri that meant adding a 7 ms
    # "resolution" that was really just the closest two passes happened to fall).
    err = None
    if agreement and agreement.get("boundary_offset_max_s") is not None:
        err = agreement["boundary_offset_max_s"]
    elif resolution:
        err = resolution
    t["error_bound_s"] = err = err or 0.0
    t["snr"] = (t["total_s"] / err) if err > 0 else float("inf")

    if agreement is None:
        t["usable"] = False
        t["verdict"] = ("UNVALIDATED -- no VLOG was supplied, so the mtime clock was not checked "
                        "against anything. Use scopex.timeline_agreement() to get a validated one.")
    elif agreement["frac_inside_pass_timer"] is None or agreement["frac_inside_pass_timer"] < 0.95:
        t["usable"] = False
        t["verdict"] = (f"ALIGNMENT FAILED: only "
                        f"{(agreement['frac_inside_pass_timer'] or 0):.1%} of snapshot mtimes fall "
                        f"inside the pass timer the VLOG reports for that pass. The two clocks are "
                        f"not describing the same events, so no interval here means anything.")
    elif t["snr"] < 10:
        t["usable"] = False
        t["verdict"] = (f"TOO SMALL TO TRUST: tail {t['total_s']*1e3:.1f} ms against a "
                        f"per-boundary "
                        f"error bound of {err*1e3:.1f} ms (SNR {t['snr']:.1f}x). The mtime "
                        f"clock "
                        f"cannot separate this from zero.")
    else:
        t["usable"] = True
        t["verdict"] = (f"USABLE: {t['total_s']:.4f} s +/- {err*1e3:.1f} ms (SNR {t['snr']:.0f}x). "
                        f"Every snapshot mtime landed inside its pass's timer "
                        f"({agreement['frac_inside_pass_timer']:.1%}), so the two clocks agree on "
                        f"WHEN things happened; the tail is one difference of two such instants."
                        + ("" if single else
                           f" The tail TOTAL is valid, but its split into emitter/llvm/codegen is "
                           f"NOT -- {kern['n_kernel_modules']} kernel modules compile concurrently "
                           f"here, so per-phase boundaries order nothing."))
    return t


def pass_timeline(dump_dir, *, module=None, log=None) -> PassTimeline:
    """``[(label, seconds), ...]`` from snapshot mtimes, plus the tail and an agreement measure.

    ``log`` is the stderr of the compile that produced ``dump_dir``, captured with
    ``TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE=hlo_pass_pipeline=1``. It MUST come from the same
    compile: comparing two different compiles confounds clock disagreement with run-to-run
    variance, which on this corpus is 26% on a loaded machine. Without it the intervals are still
    returned but ``.agreement`` is None and ``.verdict`` says UNVALIDATED, because a derived number
    that has not been checked against anything is exactly the shape of the bugs this package keeps
    paying for. :func:`timeline_agreement` runs one subprocess with both clocks on.
    """
    snaps = snapshots(dump_dir, module)
    if not snaps and module is None:
        counts = {}
        for s in snapshots(dump_dir):
            counts[s["module"]] = counts.get(s["module"], 0) + 1
        module = max(counts, key=counts.get) if counts else None
        snaps = snapshots(dump_dir, module)
    if module is None:
        counts = {}
        for s in snapshots(dump_dir):
            counts[s["module"]] = counts.get(s["module"], 0) + 1
        if counts:
            module = max(counts, key=counts.get)
            snaps = snapshots(dump_dir, module)

    res = measured_resolution(snaps)
    kern = kernel_modules(dump_dir)
    warns = []

    out = PassTimeline((f"{s['pipeline']}/{s['after']}", s["mtime"] - snaps[i - 1]["mtime"])
                       for i, s in enumerate(snaps) if i)
    agree = None
    if log is not None:
        text = pathlib.Path(log).read_text(errors="replace") if (
            isinstance(log, (str, os.PathLike)) and os.path.exists(str(log))
            and len(str(log)) < 4096) else str(log)
        year = _dt.datetime.fromtimestamp(pathlib.Path(dump_dir).stat().st_mtime).year
        passes, pipes = structure(parse_log(text, year))
        vmod = module.split(".", 1)[1] if module and "." in module else module
        mine = [p for p in passes if p["module"] == vmod]
        mypipes = [p for p in pipes if p[1] == vmod]
        rows = _align(snaps, mine, mypipes)
        leaves = [p for p in mine if p["leaf"] and p["secs"] is not None]
        nonleaf = [p for p in mine if not p["leaf"] and p["secs"] is not None]
        agree = _agreement(rows, leaves, res["resolution_s"])
        agree["naive_sum_all_pass_lines_s"] = agree["leaf_sum_s"] + sum(p["secs"] for p in nonleaf)
        agree["double_counted_by_naive_sum_s"] = sum(p["secs"] for p in nonleaf)
        agree["nested_pipeline_passes"] = sorted({p["name"] for p in nonleaf})

        # CROSS-CHECK THE LEAF SPLIT AGAINST AN INDEPENDENT IMPLEMENTATION.
        # `_parse.pass_leaf_split` reaches the same leaf/aggregate partition from the log alone,
        # by a different route: it matches announcements to pipeline headers textually, where this
        # module recovers nesting from start/end DEPTH and never looks at which names are
        # pipelines. Two implementations that disagree mean one of them is wrong, and the leaf sum
        # is the denominator of every coverage number below it -- exactly the kind of derived
        # quantity that has no business going unchecked.
        try:
            ps = pass_leaf_split(text)
            theirs = sum(t.seconds for t in ps.leaves)
            agree["leaf_sum_crosscheck_s"] = theirs
            agree["leaf_sum_crosscheck_threads"] = ps.threads
            agree["leaf_sum_crosscheck_unmatched"] = ps.unmatched_closes
            # theirs spans EVERY module in the log; ours is one module, so ours must not exceed it.
            agree["leaf_sum_crosscheck_ok"] = agree["leaf_sum_s"] <= theirs * 1.001
            if not agree["leaf_sum_crosscheck_ok"]:
                warns.append(
                    f"leaf-sum cross-check FAILED: this module's leaf sum "
                    f"{agree['leaf_sum_s']:.4f} s "
                    f"exceeds _parse.pass_leaf_split's whole-log leaf sum {theirs:.4f} s. Two "
                    f"independent readings of the same log disagree, so the nesting was misread "
                    f"and every share derived from it is suspect.")
        except Exception as e:                                                 # pragma: no cover
            agree["leaf_sum_crosscheck_s"] = None
            warns.append(f"leaf-sum cross-check could not run: {type(e).__name__}: {e}")
        if agree["n_unmatched"]:
            warns.append(f"{agree['n_unmatched']} snapshots did not match any VLOG pass.")
        if agree["n_outside_pass_timer"]:
            warns.append(f"{agree['n_outside_pass_timer']} snapshot mtimes fell OUTSIDE the pass "
                         f"timer they were matched to -- the alignment is suspect.")
    if res["n_ties"]:
        warns.append(f"{res['n_ties']} pairs of snapshots share an mtime: the filesystem clock "
                     f"cannot separate them and those intervals read 0.0 s spuriously.")
    tail = _tail(dump_dir, snaps, kern, agree, res["resolution_s"])
    if tail.get("total_s"):
        out.append(("<below HLO: emitter + LLVM + codegen>", tail["total_s"]))
    out.resolution, out.agreement, out.tail = res, agree, tail
    out.kernels, out.warnings = kern, warns
    return out


# ══════════════════════════════════════════════════════════════════════════════════════════════
# BOTH CLOCKS, ONE COMPILE
# ══════════════════════════════════════════════════════════════════════════════════════════════

def timeline_agreement(module_src: str, *, passes: str = ".+", python: str | None = None,
                       timeout: int = 1800, dump_dir: str | None = None,
                       module: str | None = None) -> PassTimeline:
    """Compile ``module_src`` ONCE in a subprocess with BOTH clocks on; return a validated timeline.

    A subprocess is not laziness. ``TF_CPP_VMODULE`` is read by the C++ logging layer when the
    shared library loads, i.e. during ``import jax``; setting it afterwards produces exactly zero
    log lines. ``--xla_dump_to`` can be set in-process but only before the backend comes up. One
    subprocess is the only place both can be true of the same compile, and they MUST describe the
    same compile or the agreement measures run-to-run variance instead of clock disagreement.

    ``passes`` defaults to ``".+"`` and not ``".*"`` deliberately: XLA's dump guard is
    ``pass_changed || dump_regex != ".*"``, so ``".+"`` snapshots EVERY matched pass while ``".*"``
    snapshots only the ones that changed the module. Measured on one control: 160 snapshots vs 26,
    which takes the median per-gap agreement from 1.39 to 1.04 because each gap then holds one
    pass instead of a dozen.

    IT COSTS WHAT IT MEASURES. The snapshot write happens INSIDE the pass timer, so a dense dump
    inflates the pass region it is timing. Measured on the corpus control, median of 8 interleaved
    reps: backend 0.0629 s undumped, 0.0639 s with VLOG only (+1.6%), 0.0716 s dumping without
    per-pass snapshots (+13.9%), 0.0735 s at ``.*`` (+17.0%), 0.0863 s at ``.+`` (+37.4%). At
    ``.+`` the per-pass write is ~110 us and the leaf-pass sum goes 5.5 ms -> 21.9 ms, so the
    measurement is 4x the thing measured. The TAIL is untouched by this -- it contains no
    snapshots -- which is the asymmetry that makes the dense mode worth using.
    """
    import tempfile
    d = dump_dir or tempfile.mkdtemp(prefix="scopex-timeline-")
    os.makedirs(d, exist_ok=True)
    pre = ("import os\n"
           f"os.environ['XLA_FLAGS'] = ' '.join([r'--xla_dump_to={d}',"
           f" r'--xla_dump_hlo_pass_re={passes}'] + "
           "([os.environ['XLA_FLAGS']] if os.environ.get('XLA_FLAGS') else []))\n")
    env = dict(os.environ)
    env.update(_parse_vmodule())
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    env.pop("XLA_FLAGS", None)
    p = subprocess.run([python or sys.executable, "-c", pre + module_src],
                       capture_output=True, text=True, timeout=timeout, env=env)
    log = p.stderr + p.stdout
    if not any(s for s in os.listdir(d) if dump_snapshot_name(s)):
        raise RuntimeError(
            f"no per-pass snapshots in {d}. This is what a dump that silently did not happen looks "
            f"like, and an empty timeline here would read as 'the backend did nothing'.\n"
            f"stderr tail:\n{p.stderr[-2000:]}")
    tl = pass_timeline(d, module=module, log=log)
    tl.stderr_tail = p.stderr[-2000:]
    return tl


def _parse_vmodule():
    """Both variables are required: importing jax sets MIN_LOG_LEVEL=1, which suppresses every
    VLOG, so TF_CPP_VMODULE alone is a silent no-op that yields an empty log and a timeline with
    no agreement measure at all."""
    return {"TF_CPP_MIN_LOG_LEVEL": "0", "TF_CPP_VMODULE": "hlo_pass_pipeline=1"}
