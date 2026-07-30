"""RECIPE -- the HLO passes account for none of the compile and every size count is flat. Where
did the seconds actually go?

This is the recipe for the case that defeats every other instrument in the package. You arrive here
from ``pass_timings_coverage.py`` with a coverage number near zero: XLA's own pass timer says the
pipeline is idle, the jaxpr is the same size as the control's, the optimized HLO is the same size,
and the compile is still 85x slower. There is exactly one place left to look, and XLA already wrote
it to disk -- the MTIMES of its own dump files.

    scopex.dump(d, passes='.*')     XLA writes each snapshot AS ITS PASS COMPLETES
    scopex.pass_timeline(d)         consecutive mtimes -> per-pass seconds, PLUS the tail after the
                                    last HLO snapshot -- emitter + LLVM + codegen. That tail is
                                    where this family lives and no other scopex instrument reaches
                                    it.

    scopex.timeline_agreement(src)  THE SAME THING, CHECKED, in one subprocess. Prefer it.

READ `.verdict` BEFORE ANY SECONDS HERE. As of the hardening round `pass_timeline` returns a
`PassTimeline`, which is still a list of `(label, seconds)` -- so everything below still works --
but it now carries the evidence that the mtime clock was or was not checked against XLA's own
microsecond timestamps:

    tl.verdict      USABLE / UNVALIDATED / ALIGNMENT FAILED / TOO SMALL TO TRUST
    tl.tail         total_s, error_bound_s, snr, split_defined
    tl.agreement    frac_inside_pass_timer (683/683 = 100.0% on the validation set), corr, ...

WITHOUT A LOG IT SAYS UNVALIDATED, and that is the honest reading of what this recipe's own numbers
were before: `dump()` gives no VLOG, so `scopex.pass_timeline(d)` alone has nothing to check the
mtimes against. The tail below is a difference of two file timestamps 30 seconds apart against a
per-boundary error of 0.18 ms, so it survives easily -- but that is an argument, and
`timeline_agreement` is a measurement.

THE THREE-WAY SPLIT IS NOW CONDITIONAL. `<llvm ir emission>` / `<llvm optimisation>` /
`<object codegen>` appear in `tl.tail` only when there is exactly ONE kernel module and its phases
do not interleave. With 223 kernel modules compiling concurrently the boundaries order nothing, so
the instrument reports the tail TOTAL and suppresses the split rather than inventing one. The
top-level entry is always `<below HLO: emitter + LLVM + codegen>`.

FOUND ON: gatherchain2d (jax#32704), CPU (JAX_PLATFORMS=cpu, jax 0.10.2, x64). GPU IS FLAT -- the
same code, same shapes, ncycles 4..9, does not grow at all on CUDA, so a "does not reproduce"
verdict here without a named backend is worthless.

MEASURED (original investigation, ncycles ladder 5/6/7/8):
    <llvm ir emission>   0.231 / 1.262 / 3.736 / 29.957 s   case
                         0.070 / 0.072 / 0.075 s            control -- DEAD FLAT
                         = 3.3x / 17.5x / 49.8x / ~400x
    At ncycles=8 that one interval is 29.957 s of a 30.203 s compile = 99.2%.
    llvm-opt (0.03-0.12 s) and codegen (0.02-0.04 s) are flat and identical in BOTH arms, so the
    tail is not "LLVM is slow", it is specifically XLA:CPU's fused ElementalIrEmitter.
    Everything a count could see is null: jaxpr equations 1.43x, optimized HLO instructions 1.04x,
    the emitted ir-no-opt.ll is +6 LINES, and the OBJECT FILE IS SMALLER IN THE PATHOLOGICAL ARM
    (1504/1544/1584 B case vs 1512/1552/1592 B control). pass_timings totals 0.021 s in BOTH arms
    to three decimals.
    Mechanism, from a rank ladder outside scopex: cost is exponential jointly in start-index rank R
    and chain depth. At depth 5, R=1/2/3/4 -> 0.18 / 0.95 / 4.16 / 36.4 s. R=1 is exactly the
    flattened control, which is why the control is flat.

RE-MEASURED for this recipe (gatherchain2d_8 vs its control, CPU, serial, uncontended box):
    <llvm ir emission>   5.606 s  vs  0.0258 s  = 217x, and 99.5% of the case's dumped compile
    <llvm optimisation>  0.0152 s vs  0.0135 s  = 1.13x   flat, as reported
    <object codegen>     0.0081 s vs  0.0061 s  = 1.33x   flat, as reported
    every HLO pass       <= 0.0013 s in BOTH arms
    optimized instrs     73 vs 70      ir-no-opt.ll 173 vs 158 lines      obj 1784 vs 1632 B
    A second run of the identical script, with other work on the box, read 10.784 s vs 0.0508 s =
    212.3x and 99.5% -- the same ratio and the same share, at twice the absolute seconds. That is
    what a wall-clock instrument does under load, and it is why the SHARE is the number to quote.
    The absolute seconds are 5.3x lower than the original 29.957 s -- a different box, and the
    original ran under a dump that also had fusion=True. The SHAPE is identical: one tail phase at
    ~99% of the compile, every pass at milliseconds, and a size ratio of 1.04-1.09x that no
    count-based instrument could act on. (The original also saw the object file SMALLER in the
    pathological arm; here it is 1.09x larger. Either way it is a null -- the point is that object
    size and compile time are unrelated on this family, not the sign of the difference.)

WHEN IT WORKS
    * Any compile whose seconds are AFTER the last HLO pass: LLVM IR emission, LLVM optimisation,
      object/PTX codegen, the ORC JIT. Measured elsewhere in this corpus: stackcond spends 16.0 s of
      a 17.8 s compile in the ORC JIT, likewise invisible to the pass timer.
    * It is the only instrument in scopex that gives a phase a NAME and a number when the phase is
      not a pass. It costs one dumped compile per arm and no extra machinery: the timestamps are
      already on disk.

WHEN IT DOES NOT
    * IT IS WALL TIME, not CPU time, and it includes whatever else the machine was doing. Run it
      SERIALLY. Two compiles at once and every interval here is noise.
    * mtime resolution and filesystem buffering put a floor of a few milliseconds on each interval;
      do not read a 3 ms pass as a measurement.
    * The interval attributed to ``<llvm ir emission>`` is really "everything between the last HLO
      snapshot and the first .ll": buffer assignment and scheduling are inside it too. It localises
      to a PHASE, not to a function. Naming the emitter on the gather case took an XLA kill-switch
      sweep (``--xla_cpu_use_fusion_emitters=false`` took gather2d_8 from 22.60 s to 0.320 s) and
      ``--xla_dump_emitter_re``, which ``scopex.dump_flags`` does not emit.
    * NO .ll AND NO .o AT ALL is a result, not a failure: a program that constant-folds to a literal
      emits no LLVM module, and ``codegen_size`` returns zeros. That is the fingerprint of
      compile-time constant folding (see pass_timings_coverage.py's adconst arm).
    * ``dump(passes='.*')`` IS EXPENSIVE AND UNWARNED. Hundreds of files, and on one corpus arm
      944 MB across 950 files. This recipe reports the directory size so you can see what it cost.
    * On GPU the tail suffixes differ (``.ptx``, ``.cubin``); ``pass_timeline`` looks for .ll/.o, so
      the GPU tail is reported as one gap rather than split. Read ``codegen_size`` and the PTX file
      census alongside it there.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile
import time

import scopex


def _dump_once(src: str, d: str, timeout: int = 3600) -> dict:
    """Compile ``src`` in a FRESH interpreter with XLA dumping on, into ``d``.

    A subprocess and not ``with scopex.dump()`` because two arms need two dumps, and the second one
    in a process where XLA's backend is already up would be SILENTLY EMPTY -- XLA reads XLA_FLAGS
    when the backend is first initialised. ``scopex.dump()`` raises in that situation; here we avoid
    the situation entirely.
    """
    env = dict(os.environ)
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    env.update(scopex.dump_flags(d, fusion=False, passes=".*"))
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True,
                       timeout=timeout, env=env)
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError(f"dumped compile failed (rc={p.returncode}):\n{p.stderr[-2000:]}")
    if not os.listdir(d):
        raise RuntimeError(
            f"{d} is EMPTY after a successful compile. XLA_FLAGS was ignored -- that is the silent "
            f"no-op this recipe exists to avoid, and an empty dump must never read as 'nothing "
            f"happened'.")
    return {"wall_s": wall, "dir": d}


def where_did_the_backend_seconds_go(case_src, control_src=None, *, module=None,
                                     keep_dumps=False, top=6) -> dict:
    """One line: which PHASE of the backend owns the compile, including the phases that are not
    passes.

    ``case_src`` / ``control_src`` are python SOURCE that compiles one arm -- not callables, for the
    same reason ``pass_timings`` takes source: the flags that make this measurable are read before
    ``import jax``, and a closure cannot in general be shipped into a fresh interpreter. See
    ``corpus_src`` at the bottom of this file for the three-line builder.

    Returns both arms' phase timelines, the phase with the largest case-minus-control gap, the
    codegen sizes (which on this family point the WRONG WAY, deliberately reported), and a verdict.
    """
    out = {}
    tmp = []
    for label, src in (("case", case_src), ("control", control_src)):
        if src is None:
            continue
        d = tempfile.mkdtemp(prefix=f"scopex-{label}-")
        tmp.append(d)
        meta = _dump_once(src, d)                       # SERIAL: one compile at a time, always
        tl = scopex.pass_timeline(d, module=module)
        steps = scopex.pass_growth(d, module=module)
        span = sum(s for _, s in tl)
        out[label] = {
            "wall_s": round(meta["wall_s"], 3),
            "timeline_span_s": round(span, 3),
            "phases": [(n, round(s, 4)) for n, s in
                       sorted(tl, key=lambda kv: -kv[1])[:top]],
            "tail_phases": {n: round(s, 4) for n, s in tl if n.startswith("<")},
            # The instrument's own account of whether its clock was checked. `dump()` writes no
            # VLOG, so this reads UNVALIDATED here by construction -- reported rather than hidden,
            # because the alternative is a tail that looks exactly like a validated one.
            "timeline_verdict": tl.verdict,
            "tail_error_bound_s": tl.tail.get("error_bound_s"),
            "tail_snr": tl.tail.get("snr"),
            "tail_split_defined": tl.tail.get("split_defined"),
            "n_snapshots": len(steps),
            "instrs_first_last": (steps[0].instrs, steps[-1].instrs) if steps else (0, 0),
            "codegen": {k: v for k, v in scopex.codegen_size(d).items() if k != "files"},
            "dump_dir": d,
            "dump_mb": round(sum(f.stat().st_size for f in pathlib.Path(d).rglob("*")
                                 if f.is_file()) / 1e6, 1),
        }

    a = out["case"]
    if "control" not in out:
        worst = max(a["phases"], key=lambda kv: kv[1]) if a["phases"] else (None, 0.0)
        out["verdict"] = (f"{worst[0]} = {worst[1]:.3f} s of a {a['timeline_span_s']:.3f} s dumped "
                          f"compile ({worst[1] / max(1e-9, a['timeline_span_s']):.1%}); no control "
                          f"given, so this is a share, not a regression")
        return out

    b = out["control"]
    ta, tb = dict(a["phases"]) | a["tail_phases"], dict(b["phases"]) | b["tail_phases"]
    # The answer is the phase with the biggest ABSOLUTE gap. A ratio alone promotes a 3 ms pass
    # that happened to be 20x; seconds are what the user lost.
    gaps = {k: ta[k] - tb.get(k, 0.0) for k in ta}
    worst = max(gaps, key=lambda k: gaps[k])
    out["worst_phase"] = {
        "phase": worst,
        "case_s": round(ta[worst], 4),
        "control_s": round(tb.get(worst, 0.0), 4),
        "gap_s": round(gaps[worst], 4),
        "ratio": round(ta[worst] / max(1e-9, tb.get(worst, 0.0)), 1),
        "share_of_case_compile": round(ta[worst] / max(1e-9, a["timeline_span_s"]), 4),
    }
    out["size_is_a_null"] = {
        "final_instrs": (a["instrs_first_last"][1], b["instrs_first_last"][1]),
        "ir_no_opt_lines": (a["codegen"]["ir_no_opt_lines"], b["codegen"]["ir_no_opt_lines"]),
        "ir_with_opt_lines": (a["codegen"]["ir_with_opt_lines"], b["codegen"]["ir_with_opt_lines"]),
        "obj_bytes": (a["codegen"]["obj_bytes"], b["codegen"]["obj_bytes"]),
        "note": "if these are ~1x (or INVERTED) while the phase gap is large, no count-based "
                "instrument can find this pathology -- which is the finding",
    }
    w = out["worst_phase"]
    out["verdict"] = (f"{w['phase']}: {w['case_s']:.3f} s vs {w['control_s']:.4f} s "
                      f"({w['ratio']}x), = {w['share_of_case_compile']:.1%} of the case's compile")
    out["next"] = ("a phase, not a function. If it is a <...> tail phase, the HLO levels cannot "
                   "refine it -- bisect with XLA kill switches "
                   "(--xla_cpu_use_fusion_emitters=false, --xla_backend_optimization_level=0) and "
                   "with --xla_dump_emitter_re, which scopex.dump_flags does not emit.")
    if not keep_dumps:
        import shutil
        for d in tmp:
            shutil.rmtree(d, ignore_errors=True)
        out["dumps_deleted"] = True
    return out


def corpus_src(name: str) -> str:
    """python source that compiles corpus case ``name`` -- what ``*_src`` above wants."""
    import _cases
    return (
        'import os; os.environ.setdefault("JAX_ENABLE_X64", "1")\n'
        'import importlib.util, jax\n'
        'jax.config.update("jax_enable_x64", True)\n'
        f'spec = importlib.util.spec_from_file_location("case", {str(_cases.find(name))!r})\n'
        'm = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n'
        f'fn, args, _ = m.CASES[{name!r}]\n'
        'jax.jit(fn).lower(*args).compile()\n')


if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")     # jax#32704 is a CPU pathology; GPU is FLAT
    os.environ.setdefault("JAX_ENABLE_X64", "1")

    NAME = "gatherchain2d_8"                          # ncycles=8, the top rung measured with a dump
    r = where_did_the_backend_seconds_go(corpus_src(NAME), corpus_src(NAME + "_control"))

    print(f"=== {NAME} vs {NAME}_control (CPU) " + "=" * 30)
    for arm in ("case", "control"):
        d = r[arm]
        print(f"  {arm:7s} wall {d['wall_s']:>8.3f} s   span {d['timeline_span_s']:>8.3f} s"
              f"   {d['n_snapshots']} snaps   dump {d['dump_mb']} MB")
        print(f"          top phases {d['phases'][:4]}")
        print(f"          tail       {d['tail_phases']}")
        print(f"          clock      {d['timeline_verdict'][:100]}")
        print(f"          codegen    {d['codegen']}")
    print(f"\n  WORST PHASE  {r['worst_phase']}")
    print(f"  size null    {r['size_is_a_null']}")
    print(f"  VERDICT      {r['verdict']}")
    print(f"  next         {r['next']}")
