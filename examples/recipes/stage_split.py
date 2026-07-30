"""RECIPE -- which STAGE owns the wall: python tracing, lowering, or the backend compiler?

Run this before anything else. It costs one compile per arm, needs no flags, no dump and no
subprocess, and it routes every other question in this directory:

    trace-bound    -> the compiler is innocent. Nothing below the jaxpr can explain it, and
                      dump/pass-timing/opcode censuses will all come back null. Profile python.
    lower-bound    -> jax's own MLIR emission. Look at program SIZE: see level_census.py.
    backend-bound  -> XLA. Go to pass_timings_coverage.py next, which tells you whether the
                      seconds are in HLO passes at all.

FOUND ON: einsum_optimal_n10 (jax#2583), CPU -- and the finding is platform-independent by
construction, because the cost is pure python before any backend is consulted.

MEASURED (original investigation, jax 0.10.2, x64, JAX_PLATFORMS=cpu):
    trace    4.467 s (case) vs 0.018 s (control) = 248x       <- the whole story
    backend  0.239 s        vs 0.225 s           = 1.06x      <- and the compiler is clean
    regime   'trace-bound'  vs 'backend-bound'
    nops ladder: trace 0.317 / 4.467 / 49.013 s at nops 8 / 10 / 11 against a control flat at
    0.015-0.018 s. Super-exponential, as the issue claims.
    The case ships a `_pathlit` twin (the optimal contraction path pasted in as a literal, so the
    solver never runs). It produces BYTE-IDENTICAL optimized HLO to the case -- md5
    01de14767cbfb41c1f69aaa56ee2212d for both -- with identical LLVM IR (16 .ll files, 2,443 lines,
    127,482 bytes) and identical object bytes (8,576), and costs 42x less. There is provably
    nothing downstream to find.

MEASURED (re-run for this recipe, same box, JAX_PLATFORMS=cpu, x64):
    trace 4.778 s vs 0.011 s = 427x; backend 0.143 s vs 0.133 s = 1.07x; labels unchanged.

WHEN IT WORKS
    Always, and it is the only instrument in scopex that can see a trace-stage pathology at all.
    Every level below the jaxpr is structurally blind to this case: 10 jaxpr equations vs 10,
    StableHLO 31 lines vs 31, walk_hlo 61 units vs 60, and a per-pass instruction curve that stays
    within 0.86x-1.33x across all 33 snapshots.

WHEN IT DOES NOT
    * `scopex.regime` is a HEURISTIC over shares of the three-stage total, and it UNDERSTATES a
      stage that is huge in absolute terms but shares the wall with an equally-inflated backend.
      Measured on arity_tree_100 (jax#4667): trace 6.212 s against a control's 0.101 s -- 61x --
      and regime still returns 'mixed', because backend is 8.857 s in the same run. That is why
      this recipe reports the per-stage RATIO against the control and calls the ratio, not the
      label, the answer. A 61x regression labelled 'mixed' is a missed call.
    * It cannot look inside `backend`, which is one number covering HLO passes, autotuning and
      codegen -- three things that want opposite responses.
    * It measures a COLD compile only because `record` calls `jax.clear_caches()`. A hand-rolled
      version without that reads zero: timing `jax.make_jaxpr` in a process that has already
      lowered the same function measured 0.002 s for a 49 s path search, because opt_einsum caches
      contraction paths independently of jax. Re-checked here: `record` DOES defeat that trap on
      jax 0.10.2 -- a second `record` of the same einsum arm in the same process still read
      4.75 s -- so use `record`, never a stopwatch.
    * `Timings.matched` is worth an assert in any script you keep. If jax renames a monitoring
      metric every stage silently reads 0.0, which is indistinguishable from an instant compile.
"""

from __future__ import annotations

import scopex


def which_stage_owns_the_wall(fn, args, control_fn, control_args) -> dict:
    """One line: is this compile even a compile, and if so which stage is it?

    Returns per-stage seconds for both arms, the per-stage case/control RATIO (the actual answer),
    both regime labels, and a routing verdict naming the recipe to read next.
    """
    t = scopex.record(fn, *args)
    c = scopex.record(control_fn, *control_args)
    if not t.matched or not c.matched:
        raise RuntimeError(
            "scopex.record matched no jax.monitoring metrics, so every stage reads 0.0 and this "
            f"answer would be a fiction. jax emitted: {t.get('seen_names')}")

    stages = ("trace", "lower", "backend")
    ratio = {k: t.get(k, 0.0) / max(1e-9, c.get(k, 0.0)) for k in stages}
    worst = max(stages, key=lambda k: ratio[k])
    # The ratio is the answer; the label is a heuristic that shares a denominator with the other
    # stages and therefore hides a big absolute regression behind a big absolute sibling.
    excess = {k: t.get(k, 0.0) - c.get(k, 0.0) for k in stages}
    dominant = max(stages, key=lambda k: excess[k])

    route = {
        "trace": "the compiler is innocent -- profile python (cProfile over the first trace). "
                 "dump/pass_timings/opcode censuses will all be null here.",
        "lower": "jax's own MLIR emission -- read level_census.py, the program is probably "
                 "already huge at the jaxpr.",
        "backend": "XLA -- read pass_timings_coverage.py next, and believe the COVERAGE number "
                   "before you believe the top pass.",
    }[dominant]

    return {
        "case": {k: round(t.get(k, 0.0), 4) for k in stages} | {"wall": round(t["wall"], 4)},
        "control": {k: round(c.get(k, 0.0), 4) for k in stages} | {"wall": round(c["wall"], 4)},
        "ratio": {k: round(ratio[k], 2) for k in stages},
        "excess_seconds": {k: round(excess[k], 4) for k in stages},
        "regime": {"case": scopex.regime(t), "control": scopex.regime(c)},
        "worst_ratio_stage": worst,
        "dominant_excess_stage": dominant,
        "verdict": f"{dominant}-bound vs control ({ratio[dominant]:.0f}x, "
                   f"+{excess[dominant]:.3f} s)",
        "next": route,
        "regime_agrees": scopex.regime(t) == f"{dominant}-bound",
    }


if __name__ == "__main__":
    import _cases

    for n in (8, 10):
        fn, args = _cases.load(f"einsum_optimal_n{n}")
        cfn, cargs = _cases.load(f"einsum_optimal_n{n}_control")
        r = which_stage_owns_the_wall(fn, args, cfn, cargs)
        print(f"\n=== einsum_optimal_n{n} vs _control " + "=" * 34)
        print(f"  case     {r['case']}")
        print(f"  control  {r['control']}")
        print(f"  ratio    {r['ratio']}")
        print(f"  regime   {r['regime']}   (label agrees with ratio: {r['regime_agrees']})")
        print(f"  VERDICT  {r['verdict']}")
        print(f"  next     {r['next']}")

    print("\n=== the counter-example the label gets wrong " + "=" * 25)
    print("  arity_tree_100: trace 6.212 s vs 0.101 s = 61x, and regime says 'mixed' because")
    print("  backend is 8.857 s in the same run. Read `ratio`, not `regime`.")
