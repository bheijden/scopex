"""RECIPE -- the optimized module is enormous but the jaxpr is not. WHICH PASS grew it?

A different question from "which pass took the seconds", and it wants a different instrument. Here
the compile is slow because XLA is compiling a program far larger than the one you wrote, and the
growth happened inside the pipeline: the jaxpr and the StableHLO are the same size as the control's
and the optimized HLO is 60x bigger. The pass that did it is named by walking both arms' per-pass
snapshots in lockstep.

    scopex.dump(d, passes='.*')          one dumped compile per arm
    scopex.walk_hlo(compiled)            the headline: how much bigger is the FINAL module
    scopex.diverge(case_dir, control_dir) the localisation: the first pass at which the two
                                         instruction-count curves separate, plus whether the two
                                         arms ran the SAME PASSES at all

That last field decides which kind of bug you have. Curves that separate on a shared pass mean one
pass behaving badly on your program. Curves that separate because the arms ran DIFFERENT passes mean
a pass-SELECTION difference, and those want opposite fixes (see codegen/custom-call recipes).

FOUND ON: bisect_m94 / bisect_m64 (jax#10621), CPU.

MEASURED (original investigation):
    optimized HLO instrs   330,397 (m=94)  vs  5,498 (m=96 control)   = 60.1x
    backend seconds        250.5 s         vs  3.007 s                = 83x
    jaxpr equations        53              vs  53      IDENTICAL primitives, identical nesting
    StableHLO lines        801             vs  564                    = 1.42x
    So a 1.42x StableHLO becomes a 60x HLO, and the entire amplification is inside XLA.
    THE LOCALISATION: both arms track within 1.21x right up to layout assignment (case 2,458 /
    control 2,030) and then ONE pass, `fusion`, takes the case 2,458 -> 176,189 (71.7x in a single
    pass) while it takes the control 2,030 -> 5,446 (2.68x). The pass sequence is otherwise
    identical. No other instrument reported this.
    Mechanism (from a two-axis sweep, outside scopex): the cliff is not at m=95/96 but at
    `length // unroll == 1`. jax emits a single-trip while loop, XLA inlines it, the whole Sturm
    sequence becomes straight-line code under two vmaps, and fusion then replicates producers into
    every consumer.

RE-MEASURED for this recipe (bisect_m64 -- the cheapest pathological rung -- against the same m=96
control, CPU, serial, both compiles under a full pass dump):
    jaxpr equations       53 vs 53                 IDENTICAL, as reported
    StableHLO units       446 vs 362 = 1.23x       (walk_stablehlo walks the MLIR natively now; the
                                                    original counted text lines only, 801 vs 564)
    walk_hlo units        155,862 vs 5,518 = 28.2x (original at m=64: 155,857 -- five instructions
                                                    apart, four years of nothing having moved)
    compile seconds       52.1 vs 1.9 under the dump
    BIGGEST JUMP          fusion: 2,458 -> 176,189 = 71.7x IN ONE PASS
    the control's fusion  2,050 -> 5,466 = 2.67x
    dump size             462.5 MB (case) / 23.9 MB (control), 111 s wall for the pair
The localisation reproduces to the instruction. Note `pass_sequence_identical` came back FALSE here
(the case runs three extra snapshots -- a dce, a pipeline-end and cpu-parallel-task-assigner), which
is why this recipe reports the biggest jump independently of the lockstep walk.

WHEN IT WORKS
    * Whenever a size count at the optimized-HLO level is large and the same count at the jaxpr
      level is not. That gap IS the statement "XLA did this, not your program".
    * It works even when the pass TIMER is unavailable or wrong. On this very case XLA printed
      `HLO pass: fusion time: 2.06 min` and the old pass-timing parser dropped the line silently,
      reporting 18.9 s of a 301 s compile topped by copy-insertion. The growth curve named `fusion`
      regardless, because it counts instructions and never looks at a clock.

WHEN IT DOES NOT
    * A FLAT CURVE IS COMMON AND IS A REAL ANSWER. On the gather, switch_ident, arity_tree and
      dusfold families the case is simply N times bigger than the control at the FIRST snapshot and
      stays exactly N times bigger: no pass diverges, because the size came in with the program.
      `diverge` reports that honestly (`diverges_at` at index 0, or a constant ratio) and you should
      route back to the jaxpr level, not hunt for a pass.
    * IT SAYS NOTHING ABOUT SECONDS. A pass can grow a module 70x and be cheap, and a pass can be
      98% of a compile without changing the instruction count by one (autotuning). Pair it with
      pass_timings_coverage.py; on this case both agree, which is what makes the answer strong.
    * `dump(passes='.*')` IS EXPENSIVE AND UNWARNED: 671 files and 443 MB for bisect_m64, 944 MB
      across 950 files on m=95. This recipe reports the size. Do not run it on a laptop disk without
      looking.
    * Counting is the whole method, so it inherits the counter. `scopex.pass_growth` counts through
      XLA's own parser and warns if any snapshot falls back to the line-based counter -- a mixed
      curve shows a fake step exactly where the route changed. If you see that warning, the step it
      found is not real.
    * The two arms must run the same pipeline for a lockstep walk to mean anything. Check
      `pass_sequence_identical` before reading `diverges_at`.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import scopex

SENTINEL = "__SCOPEX_GROWTH__"


def _snapshot_passes(dump_dir: str, module=None) -> list[str]:
    """``after_pass`` for every snapshot, in index order, READ OFF THE FILENAMES.

    Free: it never opens a file. `scopex.pass_growth` parses 460 MB of HLO text to count
    instructions, so re-calling it just to recover a pass NAME would double the cost of this recipe.
    """
    from scopex._parse import dump_snapshot_name

    stem = module or (scopex.modules_in(dump_dir) or [""])[0]
    rows = []
    for f in os.listdir(dump_dir):
        m = dump_snapshot_name(f)
        if m and (m["module"] == stem or (module and module in m["module"])):
            rows.append((m["index"], m["after_pass"]))
    rows.sort()
    return [a for _, a in rows]


def _dump_once(src: str, d: str, timeout: int = 5400) -> dict:
    """One dumped compile in a FRESH interpreter. Fresh because XLA reads XLA_FLAGS when its
    backend is first initialised, so a second dump in the same process is a SILENT no-op."""
    env = dict(os.environ)
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    env.update(scopex.dump_flags(d, fusion=False, passes=".*"))
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True,
                       timeout=timeout, env=env)
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError(f"dumped compile failed (rc={p.returncode}):\n{p.stderr[-3000:]}")
    if not os.listdir(d):
        raise RuntimeError(f"{d} is EMPTY after a successful compile -- XLA_FLAGS was ignored.")
    extra = {}
    for line in p.stdout.splitlines():
        if line.startswith(SENTINEL):
            extra = json.loads(line[len(SENTINEL):])
    return {"wall_s": wall} | extra


def which_pass_grew_the_module(case_src, control_src, *, module=None, factor=1.5,
                               keep_dumps=False) -> dict:
    """One line: which XLA pass turns your program into a much larger program?

    ``case_src`` / ``control_src`` are python source that compiles one arm (see ``corpus_src``
    below). Source rather than callables because each arm needs its own interpreter for the dump to
    exist at all.

    Returns the final-size ratio, the divergence point, both curves, and whether the arms even ran
    the same passes.
    """
    dirs, arms = {}, {}
    for label, src in (("case", case_src), ("control", control_src)):
        d = tempfile.mkdtemp(prefix=f"scopex-{label}-")
        dirs[label] = d
        arms[label] = _dump_once(src, d)                 # SERIAL: one compile at a time
        arms[label]["dump_mb"] = round(sum(f.stat().st_size for f in pathlib.Path(d).rglob("*")
                                           if f.is_file()) / 1e6, 1)

    dv = scopex.diverge(dirs["case"], dirs["control"], module=module, factor=factor)
    curve_a, curve_b = dv["case_curve"], dv["control_curve"]
    # The biggest single-pass JUMP in the case, which is the statement "this pass did it" even when
    # the two arms' pass lists are not identical and the lockstep walk cannot align them.
    #
    # AND IT IS NOT THE PASS THE CURVE IS LABELLED WITH. A snapshot's filename is
    # `after_X.before_Y`; its instruction count is the module as it stood AFTER X. `PassStep.name`
    # (what `diverge` puts in the curve) is Y, the pass about to run. So the pass that CAUSED the
    # jump into position i is snapshot i's `after_pass`, and reporting the curve label instead names
    # the innocent pass that ran next -- here, `fusion-wrapper` for work done by `fusion`.
    # Nor can you get it by looking one entry back: `after_pass` stays at the last pass that
    # actually changed the module, so consecutive snapshots often repeat it (measured: 7 snapshots
    # in a row reading `after=pipeline-start` on a two-line program). Read it off the FILENAMES,
    # which costs nothing -- the expensive part of pass_growth is parsing the file CONTENTS.
    caused_a = _snapshot_passes(dirs["case"], module)
    caused_b = _snapshot_passes(dirs["control"], module)
    jumps = [(caused_a[i], curve_a[i][0], curve_a[i - 1][1], curve_a[i][1],
              curve_a[i][1] / max(1, curve_a[i - 1][1])) for i in range(1, len(curve_a))]
    biggest = max(jumps, key=lambda j: j[4]) if jumps else None
    same_pass_in_control = None
    if biggest and biggest[0] in caused_b:
        i = caused_b.index(biggest[0])
        if i:
            same_pass_in_control = (curve_b[i - 1][1], curve_b[i][1],
                                    round(curve_b[i][1] / max(1, curve_b[i - 1][1]), 2))

    out = {
        "final_instrs": {"case": dv["case_final"], "control": dv["control_final"],
                         "ratio": round(dv["case_final"] / max(1, dv["control_final"]), 1)},
        "walk_hlo_units": {"case": arms["case"].get("walk_hlo"),
                           "control": arms["control"].get("walk_hlo")},
        "jaxpr_eqns": {"case": arms["case"].get("jaxpr_eqns"),
                       "control": arms["control"].get("jaxpr_eqns")},
        "stablehlo_units": {"case": arms["case"].get("stablehlo_units"),
                            "control": arms["control"].get("stablehlo_units")},
        "backend_s": {"case": arms["case"].get("backend_s"),
                      "control": arms["control"].get("backend_s")},
        "pass_sequence_identical": dv["pass_sequence_identical"],
        "case_only_passes": dv["case_only_passes"][:5],
        "control_only_passes": dv["control_only_passes"][:5],
        "diverges_at": dv["diverges_at"],
        "biggest_single_pass_jump": {
            "pass": biggest[0], "next_pass_label_in_curve": biggest[1],
            "before": biggest[2], "after": biggest[3], "x": round(biggest[4], 1),
            "same_pass_in_control(before, after, x)": same_pass_in_control} if biggest else None,
        "n_snapshots": {"case": len(curve_a), "control": len(curve_b)},
        "dump_mb": {k: v["dump_mb"] for k, v in arms.items()},
        "wall_s": {k: round(v["wall_s"], 1) for k, v in arms.items()},
        "curve_head": {"case": curve_a[:3], "control": curve_b[:3]},
    }
    b = out["biggest_single_pass_jump"]
    if b and b["x"] > 3:
        out["verdict"] = (f"{b['pass']} grew the module {b['before']} -> {b['after']} "
                          f"({b['x']}x in one pass); the control's same pass: "
                          f"{same_pass_in_control}")
    else:
        out["verdict"] = ("NO pass grows this module -- the curve is flat and the case is already "
                          f"{out['final_instrs']['ratio']}x the control at the first snapshot. The "
                          "size came in with the program; go back up to the jaxpr level.")
    if not keep_dumps:
        import shutil
        for d in dirs.values():
            shutil.rmtree(d, ignore_errors=True)
    else:
        out["dump_dirs"] = dirs
    return out


def corpus_src(name: str) -> str:
    """python source compiling corpus case ``name`` and printing the cross-level census on stdout,
    so one compile answers both 'how much bigger' and 'which pass'."""
    import _cases
    return (
        'import os, json; os.environ.setdefault("JAX_ENABLE_X64", "1")\n'
        'import importlib.util, jax, scopex, time\n'
        'jax.config.update("jax_enable_x64", True)\n'
        f'spec = importlib.util.spec_from_file_location("case", {str(_cases.find(name))!r})\n'
        'm = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n'
        f'fn, args, _ = m.CASES[{name!r}]\n'
        'j = jax.make_jaxpr(fn)(*args)\n'
        'low = jax.jit(fn).lower(*args)\n'
        't0 = time.perf_counter(); comp = low.compile(); t1 = time.perf_counter()\n'
        'print("' + SENTINEL + '" + json.dumps({\n'
        '    "jaxpr_eqns": sum(1 for _ in scopex.walk(j)),\n'
        '    "stablehlo_units": sum(1 for _ in scopex.walk_stablehlo(low)),\n'
        '    "walk_hlo": sum(1 for _ in scopex.walk_hlo(comp)),\n'
        '    "backend_s": round(t1 - t0, 3)}))\n')


if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")

    CASE, CONTROL = "bisect_m64", "bisect_m95_control"
    print(f"dumping {CASE} and {CONTROL} -- this is minutes and hundreds of MB, serially")
    r = which_pass_grew_the_module(corpus_src(CASE), corpus_src(CONTROL))

    print(f"\n=== {CASE} vs {CONTROL} (CPU) " + "=" * 30)
    print(f"  jaxpr equations   {r['jaxpr_eqns']}")
    print(f"  stablehlo units   {r['stablehlo_units']}")
    print(f"  walk_hlo units    {r['walk_hlo_units']}")
    print(f"  final instrs      {r['final_instrs']}")
    print(f"  compile seconds   {r['backend_s']}   (dumped: {r['wall_s']} s wall, "
          f"{r['dump_mb']} MB)")
    print(f"  same pass list?   {r['pass_sequence_identical']}  "
          f"(case-only {r['case_only_passes']}, control-only {r['control_only_passes']})")
    print(f"  diverges at       {r['diverges_at']}")
    print(f"  biggest jump      {r['biggest_single_pass_jump']}")
    print(f"  VERDICT  {r['verdict']}")
