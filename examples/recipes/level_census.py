"""RECIPE -- at which LEVEL does the program get big: was it born big, or did XLA multiply it?

One line per level, both arms, ratios down the column. The SHAPE of the ratio column is the answer:

    flat ratio          the program is already N times too big when jax hands it over. Nothing XLA
                        does is to blame; go and find the python that emitted it (blame_the_line.py).
    ratio that GROWS    something below the jaxpr is multiplying it. Go to pass_divergence.py and
    down the column     find the pass.

FOUND ON: two arms with opposite answers, which is the whole point of running this before anything
expensive.

  arity_tree_100 / _control (jax#4667, 100 pytree leaves at a jit boundary), CPU, x64 -- BORN BIG
      jaxpr equations   18,400 vs 183       = 100x     (and exactly 184 x NLEAVES: 4,600 / 9,200 /
                                                        18,400 / 36,800 at 25 / 50 / 100 / 200
                                                        against a control flat at 183, so the ratio
                                                        IS the pathology parameter)
      StableHLO lines   42,442 vs 460       = 92x
      walk_hlo units    18,900 vs 189       = 100x
      HLO computations  100 vs 1            = NLEAVES vs 1
      record            trace 6.212 / lower 1.207 / backend 8.857 s vs 0.101 / 0.076 / 0.210 s
      The dump then proves the divergence PRE-DATES XLA, which is the useful part: snapshot 0
      (before_optimizations) is already 9,057 instructions vs 187 = 48.4x, and the ratio stays
      between 33x and 50x for all 37 snapshots with an identical pass sequence. No pass amplifies
      or repairs anything.

  ndtri_scan_jacrev_d16 / _control (jax#2609, reverse-mode AD through a scanned Cephes ndtri),
  CPU, x64 -- GREW INSIDE XLA
      jaxpr equations   7,885 vs 3,437      = 2.29x    (2.28x at d4 too: a flat, weak ratio)
      walk_hlo units    108,056 vs 17,689   = 6.11x    (5.98x at d4)
      computations      3,232 vs 429        = 7.5x     (3,227 fusions vs 424)
      record            trace 7.648 / lower 1.734 / backend 102.058 s vs 2.829 / 0.530 / 4.502 s
      backend ratio 22.7x at d16 against 6.5x at d4, i.e. the ratio itself grows with depth.
      Below HLO it keeps growing: 344 LLVM kernel modules vs 39, ir-no-opt.ll 46,353 vs 5,012
      lines, object bytes 459,216 vs 62,696.
      TWO CLAIMS IN THAT CASE FILE ARE REFUTED on jax 0.10.2 and the census is what refutes them:
      (a) "expect the cost in lower, not compile" -- lower is 1.734 s of a 112 s wall and backend
      is 91%; (b) "reverse mode is 5.4x forward" -- the jacfwd arm's backend is 146.294 s against
      jacrev's 102.058 s, so the AD-direction control is INVERTED.

MEASURED (re-run for this recipe on smaller rungs, JAX_PLATFORMS=cpu, x64):

    arity_tree_50           case   control   ratio        ndtri_scan_jacrev_d4  case  control ratio
    jaxpr_eqns              9200      183    50.27        jaxpr_eqns            1981     869   2.28
    stablehlo_ops          21102      424    49.77        stablehlo_ops         3751    1052   3.57
    stablehlo_lines        21248      466    45.60        stablehlo_lines       3966    1242   3.19
    hlo_opt_instrs          9451      189    50.01        hlo_opt_instrs       25914    4340   5.97
    hlo_opt_computations      51        2    25.50        hlo_opt_computations   820     116   7.07
    -> BORN BIG (flat, 0.99x growth)                      -> GREW BELOW THE JAXPR (2.62x growth)

    9,200 = 184 x 50 exactly, as published. The ndtri HLO row reproduces the published d4 numbers
    (25,904 vs 4,330) to within 10 instructions.

WHEN IT WORKS
    Any two arms of the same program family. It is the cheapest way to decide whether to spend the
    next twenty minutes on dumps and pass timings at all, and it is the only instrument that
    distinguishes "jax emitted a huge program" from "XLA grew a small one".

WHEN IT DOES NOT
    * A FLAT RATIO COLUMN IS NOT "NO SIGNAL". It can mean the size is not in the counts at all --
      in the operand list of one node (widest_instruction.py), in a pass DECISION
      (codegen_decision.py), or in an emitter that produces small output slowly
      (phase_timeline.py). gatherchain2d_9 reads 1.43x / 1.26x / 1.04x down this column against a
      522x compile.
    * `walk_hlo` needs a Compiled object, so this recipe pays a full cold compile per arm. On an
      arm that compiles for minutes, run the jaxpr and StableHLO rows first (`compile=False`) --
      they need no backend at all and on arity_tree they already carry the whole 100x.
    * The HLO row is the OPTIMIZED module. Anything a pass erased is not in it: on stackcond, CSE
      collapses 3,021 instructions to 22 and the optimized row comes back INVERTED. When the jaxpr
      and StableHLO rows disagree with the HLO row, believe them and go read the dump.
    * `computations` counts HLO computations, which is the right axis for multiplicity cases
      (switch_ident: 513 computations at n=512) and says nothing about their size.
"""

from __future__ import annotations

import jax

import scopex


def _levels(fn, args, *, compile: bool = True) -> dict:
    jaxpr = jax.make_jaxpr(fn)(*args)
    units = list(scopex.walk(jaxpr))
    lowered = jax.jit(fn).lower(*args)
    row = {
        "jaxpr_eqns": len(units),
        "jaxpr_primitives": len(scopex.attribute(units, "kind")),
        "stablehlo_ops": len(list(scopex.walk_stablehlo(lowered))),
        "stablehlo_lines": len(scopex.stablehlo_text(lowered).splitlines()),
    }
    if compile:
        c = lowered.compile()
        row["hlo_opt_instrs"] = len(list(scopex.walk_hlo(c)))
        row["hlo_opt_computations"] = len(scopex.hlo_module(c).computations())
    row["platform"] = jax.devices()[0].platform
    return row


def where_does_the_program_get_big(fn, args, control_fn, control_args, *,
                                   compile: bool = True) -> dict:
    """One line: is the program already huge at the jaxpr, or does something below it multiply it?

    Returns one row per level for both arms, the ratio column, and a verdict that reads the SHAPE
    of that column -- flat means born big, rising means something below the jaxpr multiplied it.
    """
    case = _levels(fn, args, compile=compile)
    ctrl = _levels(control_fn, control_args, compile=compile)
    order = [k for k in ("jaxpr_eqns", "stablehlo_ops", "stablehlo_lines",
                         "hlo_opt_instrs", "hlo_opt_computations") if k in case]
    # The device is an axis, not a footnote: which backend you compile for decides which passes
    # run at all, so it decides whether a pathology exists.
    ratio = {k: round(case[k] / max(1, ctrl[k]), 2) for k in order}

    # The ratio SHAPE is read over the SIZE rows only. `hlo_opt_computations` is multiplicity, a
    # different quantity on a different scale: on arity_tree it reads 25.5x while every size row
    # reads 50x, and including it in the trend turns a flat column into a fake collapse.
    size_rows = [k for k in order if k != "hlo_opt_computations"]
    first, last = ratio[size_rows[0]], ratio[size_rows[-1]]
    growth = last / max(1e-9, first)
    if max(ratio[k] for k in size_rows) < 1.3:
        verdict = ("FLAT AND SMALL -- the counts do not see this at all. Do NOT conclude 'no "
                   "signal': try widest_instruction.py (size in one operand list), "
                   "codegen_decision.py (a pass DECISION), phase_timeline.py (an emitter that is "
                   "slow while its output is small).")
    elif growth > 1.8:
        verdict = (f"GREW BELOW THE JAXPR -- the ratio rises {first}x -> {last}x down the column. "
                   f"Something between the jaxpr and the optimized module multiplied it. Next: "
                   f"pass_divergence.py, which names the pass.")
    elif growth < 0.6:
        verdict = (f"SHRANK BELOW THE JAXPR -- {first}x -> {last}x. A pass erased the evidence "
                   f"(CSE and constant folding both do this). The optimized module is the wrong "
                   f"place to look; read the pre-optimization snapshot via pass_divergence.py.")
    else:
        verdict = (f"BORN BIG -- the ratio is flat at ~{first}x from the jaxpr down, so jax handed "
                   f"XLA a program that was already {first}x too large and no pass is to blame. "
                   f"Next: blame_the_line.py, which names the python that emitted the equations.")

    return {"case": case, "control": ctrl, "ratio": ratio, "levels": order,
            "size_rows": size_rows,
            "ratio_growth_across_levels": round(growth, 2), "verdict": verdict}


if __name__ == "__main__":
    import _cases

    # Smaller rungs than the published ones so this finishes in a couple of minutes; the ratio
    # SHAPE, which is what the recipe reads, is the same at every rung on both cases.
    for case, control, headline in (
            ("arity_tree_50", "arity_tree_50_control",
             "jax#4667 -- published at nleaves=100: 18,400 vs 183 eqns, flat to the HLO"),
            ("ndtri_scan_jacrev_d4", "ndtri_scan_jacrev_d4_control",
             "jax#2609 -- published at d16: jaxpr 2.29x but walk_hlo 6.11x"),
    ):
        fn, args = _cases.load(case)
        cfn, cargs = _cases.load(control)
        r = where_does_the_program_get_big(fn, args, cfn, cargs, compile=True)
        print(f"\n=== {case} vs {control} "
              f"[platform={r['case']['platform']}] " + "=" * 14)
        print(f"  {headline}")
        print(f"  {'level':22s} {'case':>10s} {'control':>10s} {'ratio':>8s}")
        print("  " + "-" * 52)
        for k in r["levels"]:
            print(f"  {k:22s} {r['case'][k]:10d} {r['control'][k]:10d} {r['ratio'][k]:8.2f}")
        print(f"  ratio growth across levels: {r['ratio_growth_across_levels']}x")
        print(f"  VERDICT {r['verdict']}")
