"""RECIPE -- why there is no per-instruction lineage, and the measurement that settles it.

THIS RECIPE SHIPS A NEGATIVE RESULT. It exports no mapping. It exports the measurement that says a
mapping should not be built, so that the next person to want one can re-run it in thirty seconds
instead of re-deriving the conclusion over an afternoon. It lived in `src/scopex/lineage.py` for
part of a day and was moved here on the ship call: a package module whose headline number is 96.7%
and whose conclusion is "do not use this" is a trap with a docstring in front of it. A recipe is
allowed to be an argument.

This module deliberately exports no mapping. It exports the measurement that says a mapping should
not be built, so that the next person to want one can re-run it in thirty seconds instead of
re-deriving the conclusion over an afternoon.

Per-instruction provenance across a pass would be the most useful thing in this package. "The
fusion pass is 41% of your compile" becomes "these 176,189 instructions are 412 copies of this one
subgraph, and here is the instruction each copy came from". Nothing else in scopex reaches that.

═══ WHAT XLA ACTUALLY RECORDS ═══════════════════════════════════════════════════════════════════

Three candidate mechanisms exist. All three were checked against the XLA source jaxlib 0.10.2 was
built from (commit 5a9e73cb, sha256 verified against jax's own pin) AND against a real GPU dump of
268 files.

**A. ``original_value`` -- a real lineage field, switched off.** ``xla/hlo/ir/hlo_original_value.h``
defines ``OriginalArray`` as "the name of the instruction IN THE UNOPTIMIZED HLO MODULE that
produces this array". That is exactly the wanted relation, first-class, per instruction, with a
proto, a recovery table on the module (``HloModule::original_value_recovery_table_``), and helpers
that passes call to carry it across a rewrite (``xla/hlo/ir/hlo_original_value_util.cc``). It is
printed by default: ``HloPrintOptions`` initialises ``print_original_value_(true)``.

It is populated by exactly one pass, ``AddOriginalValue`` ("add-original-value",
``xla/hlo/transforms/add_original_value.cc``), and that pass is registered in exactly one place::

    xla/hlo/tools/hlo_opt/opt_lib.cc:233:  RegisterPass<AddOriginalValue>();

``hlo_opt`` is a standalone developer tool. Neither ``cpu_compiler.cc`` nor ``gpu_compiler.cc``
adds it. Two independent confirmations that it therefore never runs under jax:

* ``add-original-value`` appears in 0 of the 23,533 ``HLO pass:`` lines this project logged across
  both backends (``tools/pass_names.json`` has 213 names; that is not one of them);
* ``origin={`` -- how ``OriginalValue`` prints -- occurs 0 times in all 268 files of a GPU dump
  taken with ``passes=".*"``.

So the honest statement is not "XLA records no lineage". It is: **XLA has a lineage mechanism, it
is good, and JAX's compiler pipelines do not turn it on.** That is a much more actionable finding,
and it is the recommendation below.

**B. the tracking suffix -- real, narrow, GPU-only.** ``AddTrackingSuffixToInstructionNames``
(pass name ``rename-instructions``, ``xla/backends/gpu/transforms/``) appends ``.0`` to instruction
names so that priority-fusion's duplicates can be traced back; its own header says "One can match
instructions before and after by their original name". It runs unconditionally in the GPU
``pre-fusion`` pipeline (``gpu_compiler.cc:1254``) and it demonstrably fires -- measured across
that boundary in a real dump, 0 instructions carried a ``.N.M`` double suffix before it and 10 of
26 did after.

Ten of twenty-six. The pass skips parameters, custom-calls, existing fusions and anything
``!IsFusible()``, so the convention covers the instructions it covers and is silent about the
rest -- and its own suffix is indistinguishable from the ordinary uniquifier suffix that XLA
appends to every duplicated name.

**C. ``metadata=`` -- survives, and is not an identity.** ``op_name`` and ``stack_frame_id`` do
propagate; that is what the rest of scopex is built on. But they point at SOURCE, and they are
many-to-one by construction: in the measured module, 8 distinct ``op_name`` values covered 40
optimized instructions, one of them covering 10. A source pointer is not a predecessor.

═══ THE MEASUREMENT THAT SETTLES IT ═════════════════════════════════════════════════════════════

The only remaining route is matching instruction NAMES across consecutive per-pass snapshots. Run
:func:`name_survival` on any dump taken with ``passes=".*"``. Measured on a GPU dump of a
``sum(tanh(a*2+1) @ a) + sum(exp(a)*a)`` compile, 64 snapshots, 63 boundaries:

    ==========================================  =========  ========  ===================
    boundary                                    n before   n after   name-identical
    ==========================================  =========  ========  ===================
    (51 boundaries that changed nothing)              --        --   100%
    computation-deduplicator                          21        18   100%
    dce                                               41        37   100%
    algsimp                                           30        26   100%
    reduction-dimension-grouper                       24        26    92%
    fusion-wrapper                                    37        40    90%
    rename_fusions                                    37        37    89%
    tree-reduction-rewriter                           26        30    87%
    flatten-call-graph                                18        21    86%
    sanitize-constant-names                           40        40    85%
    rename-instructions                               26        26    62%
    triton-gemm-rewriter                              21        24    54%
    priority-fusion                                   26        41    49%
    ==========================================  =========  ========  ===================

    all 63 boundaries pooled: 1,914 / 1,979 = 96.7% name-identical

**96.7% is the number that would get this shipped, and it is the reason not to ship it.** The
aggregate is carried by the fifty-one boundaries at which nothing happened, where a lineage
mapping is worth nothing because the identity map is already correct. At the boundaries where a
pass actually restructured the module -- which is the only place anyone would ever ask -- it falls
to 49-62%, and it falls furthest at ``priority-fusion``, which is the pass this project's
investigations most often had to explain. Half the answers wrong, precisely where the tool is
used, behind a 96.7% headline.

That is the exact shape of the failure this package's governing rule was written about: a
plausible dict naming the wrong thing. Worse, unlike the regex bug, no coverage ratio can catch
it -- a name-matcher does not know that the instruction it matched is not the instruction it came
from, and there is no second, independent source of truth to check it against. That is what makes
it unshippable rather than merely unfinished.

═══ RECOMMENDATION ══════════════════════════════════════════════════════════════════════════════

1. **Do not build a name-based lineage mapping.** Not behind a flag, not marked experimental. It
   is right where it is useless and wrong where it is wanted, and it cannot be cross-checked.
2. **The tractable version of this feature is upstream, and it is small.** ``original_value``
   already exists, already prints, already survives passes. What is missing is a debug option that
   adds ``AddOriginalValue`` at the head of the CPU/GPU pipelines -- today it is reachable only
   from ``hlo_opt``. That is a plausible XLA feature request from this project, and it would make
   per-instruction provenance EXACT rather than heuristic. :func:`original_value_present` is the
   one-line check that will start returning True if it ever lands.
3. **Until then, the honest instrument is the one that already exists**: ``scopex.pass_growth``
   and ``diverge`` say WHICH PASS multiplied the module, and ``metadata``/``op_name`` says which
   SOURCE the survivors came from. Neither claims a per-instruction predecessor, and neither is
   wrong.
"""

from __future__ import annotations

import collections
import os
import pathlib

from scopex._parse import dump_snapshot_name, has_tracking_suffix, hlo_instruction_names

__all__ = ["name_survival", "original_value_present", "tracking_suffix_present"]


def _ordered_snapshots(dump_dir, module=None):
    found = []
    for f in os.listdir(dump_dir):
        m = dump_snapshot_name(f)
        if m:
            found.append((m["index"], f, m))
    if not found:
        return []
    stem = module or collections.Counter(m["module"] for _, _, m in found).most_common(1)[0][0]
    return sorted((i, f, m) for i, f, m in found if m["module"] == stem)


def name_survival(dump_dir: str | os.PathLike, *, module: str | None = None) -> dict:
    """How often an instruction name survives a pass, boundary by boundary. A NEGATIVE result.

    Returns ``{"boundaries": [{"pass", "n_before", "n_after", "identical", "share"}, ...],
    "pooled_share": float, "changed_boundaries": int}``.

    Read ``boundaries`` and not ``pooled_share``. The pooled figure is high (96.7% measured)
    because most boundaries change nothing; the per-boundary figures at the passes that restructure
    the module are 49-62%, and those are the only ones a lineage question is ever asked about. The
    module docstring has the measured table and the recommendation that follows from it.
    """
    snaps = _ordered_snapshots(dump_dir, module)
    d = pathlib.Path(dump_dir)
    rows = []
    for k in range(1, len(snaps)):
        f0, m1 = snaps[k - 1][1], snaps[k][2]
        n0 = hlo_instruction_names((d / f0).read_text(errors="replace"))
        n1 = hlo_instruction_names((d / snaps[k][1]).read_text(errors="replace"))
        s0 = set(n0)
        same = sum(1 for x in n1 if x in s0)
        rows.append({"pass": m1["after_pass"], "n_before": len(n0), "n_after": len(n1),
                     "identical": same, "share": same / len(n1) if n1 else 1.0})
    tot_after = sum(r["n_after"] for r in rows)
    tot_same = sum(r["identical"] for r in rows)
    return {"boundaries": rows,
            "pooled_share": tot_same / tot_after if tot_after else 1.0,
            "changed_boundaries": sum(1 for r in rows
                                      if r["n_before"] != r["n_after"]
                                      or r["identical"] != r["n_after"])}


def original_value_present(dump_dir: str | os.PathLike) -> dict:
    """Does this dump carry XLA's real lineage field? ``{"files": n, "occurrences": n}``.

    Measured 0 and 0 on jaxlib 0.10.2, because ``AddOriginalValue`` is registered only in
    ``hlo_opt``. This is the check that will start returning non-zero if a future jaxlib runs that
    pass -- at which point per-instruction lineage becomes EXACT and worth building on, and the
    recommendation in this module's docstring should be revisited rather than re-derived.
    """
    d = pathlib.Path(dump_dir)
    files = occ = 0
    for f in os.listdir(dump_dir):
        try:
            n = (d / f).read_text(errors="replace").count("origin={")
        except Exception:                                                    # pragma: no cover
            continue
        if n:
            files += 1
            occ += n
    return {"files": files, "occurrences": occ}


def tracking_suffix_present(dump_dir: str | os.PathLike, *, module: str | None = None) -> dict:
    """Did ``rename-instructions`` run, and on how much of the module?

    GPU only, and partial by design: the pass skips parameters, custom-calls, fusions and anything
    not fusible. Returns the double-suffix (``.N.M``) counts on either side of the boundary --
    measured 0 before and 10 of 26 after.
    """
    snaps = _ordered_snapshots(dump_dir, module)
    d = pathlib.Path(dump_dir)
    out = {"ran": False, "before": None, "after": None}
    for _i, f, m in snaps:
        names = hlo_instruction_names((d / f).read_text(errors="replace"))
        got = (sum(1 for x in names if has_tracking_suffix(x)), len(names))
        if m["before_pass"] == "rename-instructions":
            out["before"] = got
        if m["after_pass"] == "rename-instructions":
            out["after"] = got
            out["ran"] = True
    return out


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        print("usage: python why_no_instruction_lineage.py <dump_dir taken with passes='.*'>")
        print("\nRe-runs the three measurements the argument above rests on. The one that")
        print("matters is the per-boundary column, NOT the pooled share.")
        raise SystemExit(0)

    d = sys.argv[1]
    r = name_survival(d)
    print(f"{'boundary':44s} {'before':>7s} {'after':>7s} {'same':>7s}  share")
    print("-" * 78)
    for row in r["boundaries"]:
        if row["identical"] == row["n_after"] and row["n_before"] == row["n_after"]:
            continue                       # a boundary that changed nothing proves nothing
        print(f"{row['pass'][:44]:44s} {row['n_before']:7d} {row['n_after']:7d} "
              f"{row['identical']:7d}  {row['share']:6.1%}")
    print(f"\npooled over every boundary: {r['pooled_share']:.1%}  <- DO NOT READ THIS ONE")
    print(f"boundaries that changed anything: {r['changed_boundaries']} of "
          f"{len(r['boundaries'])}")
    print(f"\nXLA's real lineage field in this dump: {original_value_present(d)}")
    print(f"the GPU tracking suffix:               {tracking_suffix_present(d)}")
