"""RECIPE -- backend-bound: is it ONE HLO pass, and do the passes account for the compile at all?

The instrument is `scopex.pass_timings`. The number this recipe insists on is COVERAGE:

    top_share = pass_timings(src)["passes"][top] / record(fn, *args)["backend"]

`pass_timings` ALWAYS returns a plausible dict. Coverage is what tells you whether that dict is the
answer or a distraction, and on this corpus it runs from ~1.0 (the pass IS the compile) down to
0.0002 (the pass timer is looking in the wrong place entirely). Read it before the top entry.

THIS RECIPE'S CENTRAL NUMBER IS NOW IN THE PACKAGE, AND BETTER MEASURED THAN HERE. `pass_timings`
returns `result["coverage"]`, a `scopex.Coverage`. Three things it fixes about the version below,
all of them visible in this file's own MEASURED blocks:

  * ONE COMPILE, NOT TWO. `top_share` here divides a number from `pass_timings`' subprocess by a
    number from `record`'s, so it straddles 1 on a drifting machine -- measured 0.88, 0.95, 1.04,
    1.11, 1.52 for a quantity that cannot exceed 1 by that mechanism. `Coverage` takes both from
    the same child, via a `jax.monitoring` listener armed by an import hook.
  * THE DOUBLE-COUNT IS REMOVED. `sum_share` below reaches 2.23 and 3.07 because XLA prints a
    nested pipeline's aggregate alongside its members. `Coverage.coverage` divides by the LEAF sum
    (`dusfold_sum_200`: 186% naive -> 92.8%); the naive figure is kept as `naive_coverage`.
  * A SEPARATE NUMBER SAYS "THE PARSE IS BROKEN". `Coverage.fidelity` compares scopex's sum to
    XLA's own `cumulative:` and is ~1.000 on every healthy compile from 0.01% to 99.7% coverage.
    Low coverage and a broken parse are different failures and this recipe could not tell them
    apart. See `scopex/coverage.py` and `tests/test_coverage_guard.py`.

What is kept here and is NOT in the package: the CASE/CONTROL pairing, which is what turns "this
pass is 41% of the compile" into "this pass is 6,745x the control's".

FOUND ON -- three arms, and the third is why coverage is in the return value:

  switch_ident_512 (jax#4453), GPU (RTX 4090 Laptop sm_8.9), jax 0.10.2, x64, contended device
      copy-insertion  41.00 s  vs  0.00032 s  = 128,000x
      record backend  31.85 s  vs  0.166 s    = 192x
      The pass is larger than the separately-measured backend, i.e. top_share >= 1: THE PASS IS THE
      COMPILE. Rung below: 5.91 s at n=256, so 6.94x for 2x branches (~n^2.8) against a backend
      growth of 7.28x. Every scopex structural count is exactly LINEAR across the sweep
      64/128/256/512/1024 (jaxpr eqns 66/130/258/514/1026, optimized instrs 394/778/1546/3082/6154,
      computations 65/129/257/513/1025) while backend goes 0.98/1.04/4.38/31.85/256.46 s -- so the
      counts confirm the multiplicity and cannot explain the superlinearity, and only the pass
      timer can. This also REFUTES the issue's own 2020 diagnosis, which profiled it into LLVM:
      the dump gives byte-identical codegen at 2x the branches (3 ptx files, 58 ptx lines, 57
      ir-with-opt.ll lines at both n=128 and n=256).

  dusfold_sum_350 (jax#12789), GPU, x64
      constant_folding  6.0703 s  vs  0.0009 s  = 6,745x, on an identical 188-pass sequence in both
      arms, and nothing else in either profile exceeds 15 ms.
      The optimized HLO is a PERFECT TRAP: both arms end at exactly 2 instructions (constant +
      copy), shape f64[], identical opcode histogram, identical hlo_opt_lines=28. Anybody comparing
      final HLO concludes the two programs are identical.

  gatherchain2d_9 (jax#32704), CPU -- THE NEGATIVE
      passes sum to 0.021 s against a backend of 99.67 s. COVERAGE 0.02%. The top entry is real and
      irrelevant, and the pathological arm's total pass time is LOWER than its control's (0.0207 vs
      0.0223 s). 98.9 s of that compile is inside the CPU loop-fusion IR emitter, which XLA neither
      times as a pass nor snapshots into the dump. Route to phase_timeline.py.
      (argsort_f32_1e6, GPU, is the same shape at 7.2% coverage with a misleading 'autotuner' on
      top; route there to codegen_decision.py.)

MEASURED (re-run for this recipe -- switch_ident on CUDA, x64, nvidia-smi showing 0 other compute
processes, i.e. UNCONTENDED where the original reading was not). Swap the two case names in
``__main__`` to reproduce:
    n=512   backend 15.696 s vs 0.096 s = 163x
            copy-insertion 13.8321 s vs 0.00018 s = 76,845x     top_share 0.88, sum_share 0.91
    n=256   backend  2.979 s vs 0.090 s
            copy-insertion  1.5841 s vs 0.000168 s = 9,429x     top_share 0.53
    runners-up at n=512: computation-deduplicator 0.1132 s, float_normalization 0.0396 s,
            remat-pipeline 0.0238 s -- three orders of magnitude below the top entry
    copy-insertion grows 8.73x for 2x branches (published: 6.94x); the original 41.00 s / 128,000x
    reading was taken with ten foreign GPU processes at 100% utilisation, so it is an upper bound
    and this run is 2.6x faster in absolute terms with an identical shape.
    ALSO: `modules` here is ['fused_clamp', 'jit__flat_switch', 'wrapped_multiply_computation'] for
    the case and ['jit__one_switch', 'wrapped_multiply_computation'] for the control -- three
    modules summed into one dict, and differently named per arm. This is the caveat below made
    concrete.

MEASURED (re-run for this recipe -- dusfold_sum_300 on CUDA, x64, nvidia-smi showing 0 other
compute processes at the time, so these are not contended):
    backend         2.077 s  vs 0.094 s   = 22x
    top pass        simplification    2.1568 s vs 0.00169 s = 1274x   top_share 1.04
    nested in it    constant_folding  2.1552 s vs 0.00058 s = 3717x
    runners-up      simplify-while-loops 0.0003 s, scatter_expander 0.0002 s -- nothing else is
                    above a millisecond, exactly as published
    modules         ['jit__fold_sum'] vs ['jit__nofold_sum'] -- the two arms compile DIFFERENTLY
                    NAMED modules, so `module=` cannot be a single string across a pair
    The published n=350 reading was constant_folding 6.0703 s vs 0.0009 s = 6,745x on a 4.534 s
    backend. Same mechanism, one rung down.

MEASURED (re-run for this recipe -- dusfold_sum_200 on CPU, JAX_PLATFORMS=cpu, x64, two runs):
    top pass        simplification    0.286 / 0.450 s  vs 0.0006 / 0.0010 s  = 454x / 450x
    nested in it    constant_folding  0.285 / 0.449 s  vs 0.00015 / 0.00023 s
    backend         0.257 / 0.476 s   vs 0.017 / 0.034 s
    top_share 1.11 / 0.95 / 1.52 across three runs ; sum_share 2.23 / 1.90 / 3.07
    modules ['jit__fold_sum']
    The mechanism reproduces on CPU at a third of the GPU size. Absolute seconds move ~2.5x between
    back-to-back runs on a shared box, which is exactly why the ratio and the share are the answer
    and the seconds are not. NOTE ALSO that `top_share` is a ratio of TWO DIFFERENT COMPILES -- the
    numerator comes from `pass_timings`' vmodule subprocess, the denominator from `record`'s -- so
    it straddles 1 and should be read as "order one" rather than as a percentage. It also exposes a
    gotcha the GPU numbers hid, below: on CPU the top entry is the enclosing `simplification`
    pipeline-registered-as-a-pass, with `constant_folding` nested inside it at 99.8% of it.

WHEN IT WORKS
    When `stage_split.py` says backend-bound AND `top_share` comes back high. A pass that is 95%+
    of a compile is not subtly slow: its ratio against the control is typically 10^3-10^5.

WHEN IT DOES NOT
    * `top_share` below ~0.5 with a low `sum_share` means the seconds are NOT in HLO passes at all.
      Do not report the top pass. That is a real result, not a failed measurement.
    * SUM/BACKEND CAN EXCEED 1 AND USUALLY MEANS NESTING, NOT AN ERROR. XLA registers some
      pipelines AS passes, so `HLO pass: simplification` is printed alongside the
      `HLO pass: constant_folding` that ran inside it, and summing the dict double-counts. Measured
      above: 1.90. This recipe reports `nested_in_top` -- the entries whose time fits inside the
      top entry -- so the containment is visible rather than mysterious. (On the switch_ident GPU
      arm the >1 had a different cause: two separately-timed compiles on a contended box.)
    * `pass_timings` costs a FULL COLD COMPILE per arm in a subprocess, and this recipe spends a
      second one on `record` for the denominator: four compiles for a case/control pair. On a
      four-minute arm that is sixteen minutes. Pass `case_backend_s=`/`control_backend_s=` when you
      already have them from `stage_split.py`.
    * `passes` sums over EVERY module XLA compiled, including jax's warm-up
      `jit_convert_element_type`. A total is not "your program" unless `modules` has one entry --
      which is why this recipe returns that list. Pass `module="jit_your_fn"` to scope it.
    * It sees nothing below HLO and nothing XLA does not log as a pass.
    * Historical warning, because it is what this instrument is famous for: XLA switches the
      printed unit to `min` at large magnitudes, and the slowest pass is by construction the one
      most likely to be printed that way. The version of this parser that knew only {us, ms, s}
      dropped an autotuner line worth 98.8% of a 72.5 s compile and returned a plausible dict
      topped by `remat-pipeline: 0.1196`. Unknown units now raise. Never make that parse lenient.
"""

from __future__ import annotations

import scopex


def which_pass_owns_the_compile(case_src, control_src, *, module=None,
                                case_backend_s=None, control_backend_s=None,
                                timeout=1800) -> dict:
    """One line: which XLA pass is the compile -- and do the passes account for it at all?

    ``case_src``/``control_src`` are python SOURCE STRINGS that each cold-compile one arm (build
    them with ``_cases.src``). Deliberately not ``(fn, args)``: ``TF_CPP_VMODULE`` is read by the
    C++ logging layer at ``import jax``, so the compile must happen in a process that does not
    exist yet, and a live closure cannot be sent there.

    ``*_backend_s`` are ``scopex.record(...)["backend"]`` for the two arms. Without them coverage
    is ``None`` and the top pass is unverified -- which this function says out loud rather than
    quietly returning a ranking.
    """
    a = scopex.pass_timings(case_src, module=module, timeout=timeout)
    b = scopex.pass_timings(control_src, module=module, timeout=timeout)
    if not a["passes"]:
        raise RuntimeError(
            "pass_timings parsed no pass lines for the case arm. That is a broken measurement, not "
            "a compile without passes.\nstderr tail:\n" + a["stderr_tail"])

    tot_a, tot_b = sum(a["passes"].values()), sum(b["passes"].values())

    # Rank by ABSOLUTE excess over the control, not by ratio: a pass that goes 1 us -> 30 us is 30x
    # and irrelevant, and every real profile has dozens of those.
    movers = sorted(((n, s, b["passes"].get(n, 0.0)) for n, s in a["passes"].items()),
                    key=lambda r: -(r[1] - r[2]))
    top_name, top_case, top_ctrl = movers[0]

    top_share = None if not case_backend_s else top_case / max(1e-9, case_backend_s)
    sum_share = None if not case_backend_s else tot_a / max(1e-9, case_backend_s)

    # Entries small enough to have run INSIDE the top one. XLA registers some pipelines as passes,
    # so the dict is not a partition and its sum is not a total.
    nested = [(n, round(s, 4)) for n, s in a["passes"].items()
              if n != top_name and s <= top_case and s > 0.05 * top_case]

    if top_share is None:
        verdict = "UNKNOWN -- no backend seconds supplied, so the top pass is unverified"
    elif top_share >= 0.5:
        verdict = (f"TRUSTED -- {top_name!r} is {top_share:.0%} of the backend. That pass IS the "
                   f"compile.")
    elif sum_share is not None and sum_share < 0.5:
        verdict = (f"DO NOT REPORT THE TOP PASS -- every HLO pass together accounts for "
                   f"{sum_share:.2%} of the backend. {1 - sum_share:.0%} of the seconds are "
                   f"somewhere the pass timer cannot see (emitter, LLVM, autotuning). Read "
                   f"phase_timeline.py, then codegen_decision.py.")
    else:
        verdict = (f"SPREAD -- no single pass exceeds half the backend (top {top_share:.0%}) but "
                   f"the passes together cover {sum_share:.0%}. The cost is the pipeline, not a "
                   f"pass; compare the two profiles entry by entry.")

    return {
        "top_pass": top_name,
        "case_seconds": round(top_case, 4),
        "control_seconds": round(top_ctrl, 6),
        "ratio": round(top_case / max(1e-9, top_ctrl), 1),
        "excess_seconds": round(top_case - top_ctrl, 4),
        "top_share": None if top_share is None else round(top_share, 4),
        "sum_share": None if sum_share is None else round(sum_share, 4),
        "passes_total": {"case": round(tot_a, 4), "control": round(tot_b, 4)},
        "backend_seconds": {"case": case_backend_s, "control": control_backend_s},
        "nested_in_top": nested,
        "verdict": verdict,
        "modules": {"case": a["modules"], "control": b["modules"], "filter": module},
        "next_five": [(n, round(s, 4), round(c, 6)) for n, s, c in movers[1:6]],
        "case_log_lines": a["n_lines"],
    }


if __name__ == "__main__":
    import _cases

    # dusfold_sum_200 rather than _350, and CPU rather than GPU: the mechanism is the same shape
    # (constant folding is the only thing that moves) at a third of the compile. The published
    # 6.0703 s vs 0.0009 s is the n=350 CUDA number.
    PLATFORM, CASE, CONTROL = "cpu", "dusfold_sum_200", "dusfold_sum_200_control"

    print(f"measuring backend seconds for the coverage denominator ({PLATFORM}) ...")
    ba = _cases.backend_seconds(CASE, platform=PLATFORM)
    bb = _cases.backend_seconds(CONTROL, platform=PLATFORM)
    print(f"  {CASE:26s} {ba}")
    print(f"  {CONTROL:26s} {bb}")

    r = which_pass_owns_the_compile(
        _cases.src(CASE), _cases.src(CONTROL),
        case_backend_s=ba["backend"], control_backend_s=bb["backend"])

    print(f"\n=== {CASE} vs control, platform={PLATFORM} " + "=" * 20)
    print(f"  top pass       {r['top_pass']}")
    print(f"  case/control   {r['case_seconds']} s vs {r['control_seconds']} s = {r['ratio']}x")
    print(f"  top_share      {r['top_share']}   (top pass / backend seconds)")
    print(f"  sum_share      {r['sum_share']}   (>1 means pipelines are counted as passes too)")
    print(f"  nested in top  {r['nested_in_top']}")
    print(f"  passes total   {r['passes_total']}")
    print(f"  backend        {r['backend_seconds']}")
    print(f"  modules        {r['modules']['case']}")
    print(f"  runners-up     {r['next_five'][:3]}")
    print(f"  VERDICT        {r['verdict']}")

    # ── the arm as it was FOUND: dusfold_sum_300 on CUDA ────────────────────────────────────────
    # Skipped when the device is busy. Per-pass SECONDS are a timing, and a second CUDA process on
    # the box makes both arms' absolute numbers upper bounds -- the ratio survives, the seconds do
    # not, and this recipe's whole point is that the share is what you read.
    busy, desc = _cases.gpu_busy()
    print(f"\n=== dusfold_sum_300 vs control, platform=cuda " + "=" * 14)
    print(f"  nvidia-smi: {desc}")
    if busy:
        print("  SKIPPED -- another process holds the device. Re-run on a quiet GPU; the published\n"
              "  reading is constant_folding 6.0703 s vs 0.0009 s = 6,745x at n=350.")
    else:
        ga = _cases.backend_seconds("dusfold_sum_300", platform="cuda")
        gb = _cases.backend_seconds("dusfold_sum_300_control", platform="cuda")
        g = which_pass_owns_the_compile(
            _cases.src("dusfold_sum_300"), _cases.src("dusfold_sum_300_control"),
            case_backend_s=ga["backend"], control_backend_s=gb["backend"])
        print(f"  backend        {g['backend_seconds']}")
        print(f"  top pass       {g['top_pass']}  {g['case_seconds']} s vs "
              f"{g['control_seconds']} s = {g['ratio']}x   top_share {g['top_share']}")
        print(f"  nested in top  {g['nested_in_top']}")
        print(f"  modules        {g['modules']['case']} vs {g['modules']['control']}")
        print(f"  VERDICT        {g['verdict']}")
