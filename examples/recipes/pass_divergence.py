"""RECIPE -- the program grew below the jaxpr. WHICH PASS multiplied it?

`scopex.dump(passes='.*')` makes XLA write the module as it stood before every pass;
`scopex.pass_growth` counts each snapshot; `scopex.diverge` walks the two curves in lockstep and
returns the first pass where the case pulls away from the control. This recipe adds the two
readings that turned those curves into answers on real cases:

  * the ratio at SNAPSHOT 0 (before_optimizations). If the arms already differ there, no XLA pass
    is to blame and the whole dump was the wrong place to look.
  * the per-pass MULTIPLICATION FACTOR of each arm, ranked by how much more the case grows than the
    control at the same pass. `diverge` reports where a ratio THRESHOLD is first crossed, which on
    a case that starts out 3.5x apart is crossed at snapshot 0 and tells you nothing.

FOUND ON:

  ndtri_scan_jacrev_d4/_d16 (jax#2609), CPU, x64 -- A PASS IS TO BLAME
      The arms ENTER the pipeline 3.5x apart (5,763 vs 1,414 instructions pre-optimization) and
      LEAVE the `fusion` pass 6.0x apart (25,482 vs 4,262), because fusion multiplies the case by
      3.93x and the control by only 2.32x. The pass sequence is essentially identical -- 49 case
      snapshots against 48, the case gaining one `after_gather_expander` -- so no pass is uniquely
      SELECTED, one pass simply behaves differently on the bigger input.
      The opcode census on the post-fusion snapshot then names the mechanism in one line:
      select 1,404 vs 300, compare 1,146 vs 278, dynamic-slice 618 vs 0. That is AD keeping
      residuals for BOTH arms of the piecewise Cephes `lax.select`, exactly as the case file
      predicts.

  arity_tree_50/_100 (jax#4667), CPU, x64 -- NO PASS IS TO BLAME, AND THAT IS THE RESULT
      Snapshot 0 is already 9,057 instructions vs 187 = 48.4x, and the ratio then stays between 33x
      and 50x for all 37 snapshots with an identical pass sequence apart from one control-only
      `cpu-parallel-task-assigner`. The program is simply 100x bigger the moment jax hands it over.
      Running this recipe on that arm is how you find out that the dump has nothing for you --
      cheaply, and with a number rather than a hunch.

MEASURED (re-run for this recipe, JAX_PLATFORMS=cpu, x64):

  ndtri_scan_jacrev_d4 vs _control
      enters   5,892 vs 1,428 = 4.13x        leaves  25,914 vs 4,340 = 5.97x
      separating pass  HLO_passes_after_layout_assignment/fusion-wrapper
                       case x3.921, control x2.315  (1.69x more), 6,501 -> 25,492 instructions
      post-fusion opcodes  case {constant 5323, multiply 5137, add 3729, parameter 2898,
                                 select 1404, compare 1146}
                           ctrl {constant 1024, multiply  911, add  617, parameter  437,
                                 select  300, compare  278}
      The published 3.93x/2.32x fusion multipliers and the select 1404 vs 300 / compare 1146 vs 278
      census reproduce EXACTLY. The pass is spelled `fusion-wrapper` inside the
      `HLO_passes_after_layout_assignment` pipeline in this dump.

  arity_tree_50 vs _control
      enters   9,057 vs 187 = 48.43x         leaves   9,451 vs 189 = 50.01x
      34 snapshots vs 35, one control-only `HLO_passes_after_layout_assignment/dce`
      -> NO PASS IS TO BLAME. Published snapshot-0 numbers (9,057 vs 187 = 48.4x) reproduce
      exactly; the one asymmetric snapshot is now a `dce` rather than the
      `cpu-parallel-task-assigner` recorded in 2026-07.

  AND THE LIVE DEMONSTRATION OF WHY `separating_passes` EXISTS: on BOTH arms above,
  `scopex.diverge`'s own `diverges_at` returns the very first pass
  (`async-collective/async-collective-replacer`), because a threshold crossing is crossed at
  snapshot 0 whenever the arms enter apart. It is right and it is useless. Read `entry_ratio` and
  `separating_passes`.

WHEN IT WORKS
    After `level_census.py` says GREW BELOW THE JAXPR. It is the instrument that converts "the
    optimized module is 6x bigger than the ratio at the jaxpr" into the name of a pass.

WHEN IT DOES NOT
    * `passes='.*'` snapshots EVERY pass. That is a lot of files and it slows the compile down;
      the timings from a dumping run are not comparable to a clean one.
    * `scopex.dump` RAISES if XLA's backend is already up, because XLA_FLAGS is read at backend
      initialisation and setting it later is a silent no-op. Two arms therefore need TWO
      PROCESSES, which is why this recipe takes case NAMES or dump DIRECTORIES rather than
      `(fn, args)`.
    * AN INSTRUCTION-COUNT CURVE IS BLIND TO ANY FOLDING OR MATERIALISATION PATHOLOGY, because
      folding SHRINKS the count while materialising megabytes. Measured on dusfold_sum_200
      (jax#12789): the case peaks at 96 instructions and collapses to 1, snapshot count 58 vs 58,
      and the curve says nothing -- while the same curve rebuilt in BYTES (summing sizeof(shape)
      over every instruction) shows the case carrying 128 MB at pass 0, peaking at 320 MB at
      `simplification/scatter-slice-simplifier`, holding 256-320 MB for four more passes, then
      dropping to 0, against a control that starts at 64 MB and is at 0 six passes earlier.
      `pass_growth` returns `instrs` and `computations`, not bytes. If you suspect folding, use
      pass_timings_coverage.py instead -- it names constant_folding directly.
    * It cannot see anything after the last HLO pass. On CPU that is where the IR emitter and LLVM
      live, and on two cases in this corpus that is 78% and 99% of the compile. See
      phase_timeline.py.
    * `pass_growth` warns and its steps become untrustworthy if some snapshots were counted by
      XLA's parser and some by the line regex; a mixed curve shows a fake step exactly where the
      route changed. The `how` field on every PassStep records which ran.
"""

from __future__ import annotations

import collections
import os
import pathlib
import tempfile

import scopex


def _dump_arm(name_or_dir: str, *, platform: str, timeout: int) -> str:
    """A dump directory for one arm. Accepts an existing directory or a corpus case name."""
    if os.path.isdir(name_or_dir):
        return name_or_dir
    import _cases
    d = tempfile.mkdtemp(prefix=f"scopex-recipe-{name_or_dir}-")
    r = _cases.run_in_subprocess(
        f'fn, args = _cases.load({name_or_dir!r})\n'
        f'import jax\n'
        # fusion=False: the priority-fusion decision log is a GPU pass and asking for it on CPU
        # only earns a warning.
        f'with scopex.dump({d!r}, passes=".*", fusion=False, keep=True) as d:\n'
        f'    jax.jit(fn).lower(*args).compile()\n'
        f'emit({{"dir": d, "files": len(__import__("os").listdir(d))}})\n',
        platform=platform, timeout=timeout)
    if r["files"] < 3:
        raise RuntimeError(f"dump of {name_or_dir} produced {r['files']} files -- an empty dump is "
                           f"what you get when XLA_FLAGS was set too late. See scopex.dump.")
    return r["dir"]


def which_pass_multiplied_the_module(case, control, *, platform="cpu", module=None,
                                     factor=1.5, timeout=3600) -> dict:
    """One line: which XLA pass grows the slow arm faster than it grows the control?

    ``case``/``control`` are corpus case names (dumped here, one subprocess each) or existing dump
    directories. Returns `scopex.diverge`'s reading plus the pre-optimization ratio and the ranked
    per-pass multiplication factors.
    """
    a_dir = _dump_arm(case, platform=platform, timeout=timeout)
    b_dir = _dump_arm(control, platform=platform, timeout=timeout)

    d = scopex.diverge(a_dir, b_dir, module=module, factor=factor)
    a = scopex.pass_growth(a_dir, module=module)
    b = scopex.pass_growth(b_dir, module=module)
    bt = {s.name: s for s in b}

    entry_ratio = a[0].instrs / max(1, b[0].instrs)

    # Per-pass multiplication, case against control at the SAME pass. This is the reading that
    # separates "one pass behaves badly" from "the arms were already apart".
    seps = []
    for i in range(1, len(a)):
        name = a[i].name
        if name not in bt or a[i - 1].name not in bt:
            continue
        ca = a[i].instrs / max(1, a[i - 1].instrs)
        cb = bt[name].instrs / max(1, bt[a[i - 1].name].instrs)
        if a[i].instrs < 50:            # ignore churn on modules too small to matter
            continue
        seps.append((name, round(ca, 3), round(cb, 3), round(ca / max(1e-9, cb), 3),
                     a[i - 1].instrs, a[i].instrs))
    seps.sort(key=lambda r: -r[3])

    final_ratio = d["case_final"] / max(1, d["control_final"])
    if entry_ratio >= 0.7 * final_ratio:
        verdict = (f"NO PASS IS TO BLAME. The arms enter the pipeline {entry_ratio:.1f}x apart and "
                   f"leave it {final_ratio:.1f}x apart. jax handed XLA a program that was already "
                   f"too big; read blame_the_line.py, not this dump.")
    elif seps:
        n, ca, cb, mult, before, after = seps[0]
        verdict = (f"{n!r} -- it multiplies the case by {ca}x and the control by {cb}x at the same "
                   f"pass ({mult}x more), taking the case {before} -> {after} instructions. Arms "
                   f"enter {entry_ratio:.1f}x apart and leave {final_ratio:.1f}x apart.")
    else:
        verdict = (f"no pass separates the arms; they enter {entry_ratio:.1f}x apart and leave "
                   f"{final_ratio:.1f}x apart.")

    return {
        "dirs": {"case": a_dir, "control": b_dir},
        "entry_ratio": round(entry_ratio, 3),
        "final_ratio": round(final_ratio, 3),
        "entry": {"case": a[0].instrs, "control": b[0].instrs, "pass": a[0].name},
        "final": {"case": d["case_final"], "control": d["control_final"]},
        "n_snapshots": {"case": len(a), "control": len(b)},
        "pass_sequence_identical": d["pass_sequence_identical"],
        "case_only_passes": d["case_only_passes"],
        "control_only_passes": d["control_only_passes"],
        "diverges_at": d["diverges_at"],
        "separating_passes": seps[:5],
        "counted_how": dict(collections.Counter(s.how for s in a)),
        "verdict": verdict,
    }


def opcode_census(dump_dir, after_pass: str, *, module=None, top=8) -> collections.Counter:
    """Opcode histogram of the snapshot taken AFTER ``after_pass`` -- what the pass actually made.

    This is the second half of the ndtri answer: 'fusion multiplied it 3.93x' is a size, and
    'select 1,404 vs 300' is a mechanism.
    """
    steps = [s for s in scopex.pass_growth(dump_dir, module=module) if after_pass in s.after_pass]
    if not steps:
        raise KeyError(f"no snapshot taken after {after_pass!r}; present: "
                       f"{sorted({s.after_pass for s in scopex.pass_growth(dump_dir)})[:20]}")
    text = pathlib.Path(steps[0].path).read_text(errors="replace")
    m = scopex.hlo_module(text)
    c: collections.Counter = collections.Counter()
    for comp in m.computations():
        for i in comp.instructions():
            c[scopex.levels.opcode_of(i)] += 1
    return collections.Counter(dict(c.most_common(top)))


if __name__ == "__main__":
    for case, control, note in (
            ("ndtri_scan_jacrev_d4", "ndtri_scan_jacrev_d4_control",
             "jax#2609 -- published: enters 3.5x apart, leaves `fusion` 6.0x apart"),
            ("arity_tree_50", "arity_tree_50_control",
             "jax#4667 -- published: snapshot 0 already 48.4x, flat for all 37 snapshots"),
    ):
        r = which_pass_multiplied_the_module(case, control, platform="cpu")
        print(f"\n=== {case} vs {control} (cpu) " + "=" * 20)
        print(f"  {note}")
        print(f"  entry (before_optimizations)  {r['entry']['case']} vs {r['entry']['control']}"
              f"  = {r['entry_ratio']}x")
        print(f"  final                         {r['final']['case']} vs {r['final']['control']}"
              f"  = {r['final_ratio']}x")
        print(f"  snapshots                     {r['n_snapshots']}, "
              f"sequence identical: {r['pass_sequence_identical']}")
        print(f"  case-only passes              {r['case_only_passes'][:3]}")
        print(f"  control-only passes           {r['control_only_passes'][:3]}")
        print(f"  scopex.diverge says           {r['diverges_at']}")
        print("  separating passes (pass, case_mult, control_mult, ratio, before, after):")
        for row in r["separating_passes"][:4]:
            print(f"      {row}")
        print(f"  counted how                   {r['counted_how']}")
        print(f"  VERDICT {r['verdict']}")

        if case.startswith("ndtri"):
            for arm, d in (("case", r["dirs"]["case"]), ("control", r["dirs"]["control"])):
                try:
                    print(f"  post-fusion opcodes ({arm}): "
                          f"{dict(opcode_census(d, 'fusion', top=6))}")
                except KeyError as e:
                    print(f"  post-fusion opcodes ({arm}): {e}")
