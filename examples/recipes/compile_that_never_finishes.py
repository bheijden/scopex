"""RECIPE -- the compile never returns. It eats host memory until the OOM killer takes the process.
WHICH PASS never returned?

Every instrument in scopex that needs a ``Compiled`` -- ``walk_hlo``, ``hlo_text``,
``custom_calls``, ``memory_analysis``, ``codegen_size`` -- is unavailable here BY CONSTRUCTION,
because there is no executable and there never will be. ``scopex.record`` is worse than
unavailable: it has no memory cap and no timeout, so calling it on a runaway compile grows YOUR
process without bound until the kernel kills it, and ``Timings`` has no memory field to report the
resource that is actually being consumed.

Exactly one thing survives, and only by luck: ``scopex.dump`` works because XLA writes each
snapshot to disk AS ITS PASS COMPLETES. So you run the compile in a child you can kill, watch its
RSS from outside, and then read how far it got off the filesystem.

    child:   XLA_FLAGS=--xla_dump_to=D --xla_dump_hlo_pass_re=.*   +   TF_CPP_VMODULE
    parent:  sample /proc/<pid> RSS at 2 Hz, kill at a cap you choose
    after:   highest-numbered snapshot in D   AND   the last completed 'HLO pass:' log line
             then diff against a control's pass ORDER -- the next pass is the one that hung

FOUND ON: topk_pow20p1024_k128 (jax#19653), GPU (CUDA, sm_8.9, jax 0.10.2, x64).

MEASURED (original investigation):
    RSS grows PERFECTLY LINEARLY at 0.168 GB/s from 0.57 GB with no plateau; 9 GB at 51 s.
    Deterministic: 55.8 s and 59.5 s to the cap on two arms.
    The dump stops after exactly 13 files, the last being
        module_0000.jit_top_k.0007.optimization.after_pipeline-start.before_ragged_dot_rewriter.txt
    The vmodule log stops after exactly 19 'HLO pass:' lines; the control logs 303. The 19th is
    windowed-einsum-handler, and the CONTROL's pass #20 is topk-splitter. That names it.
    The three healthy arms compile in 0.309-0.375 s at 0.78 GB peak, with 11 HLO instructions and
    custom_call_target='xla.gpu.ext.cub_sort_pairs'.
    Mechanism, sharpened into a predicate and tested 10/10: the gate is
        n % 1024 == 0  AND  n / 1024 is NOT a power of two
    2**19, 2**20, 2**21 (q = 512, 1024, 2048) are all FINE; 2**20+1 and 2**20+512 (non-integer q)
    are FINE; 2**20+1024 (q=1025), +2048, +3072, +7168 and 1536*1024 (q=1536) ALL blow up. That is
    why 2**20 itself -- the length the issue was filed about -- is fast on this build, which
    alignment alone cannot explain: TopKSplitter fires on an exact 1024 batch split and the emitter
    behind it needs a power-of-two row count.

RE-MEASURED for this recipe (topk_pow20p1024_k128 against topk_pow20_k128, same GPU but IDLE):
    killed at the 9 GB cap after 32.45 s, peak 9.1 GB, RSS +0.2849 GB/s, LINEAR, no plateau
    passes completed      19 (case)  vs  289 (control)
    last completed pass   windowed-einsum-handler        <- exactly as reported
    last dump snapshot    index 7, optimization pipeline, after_pipeline-start.before_
                          ragged_dot_rewriter, 13 XLA files    <- exactly as reported
    pass prefix identical  True
    NAMED PASS            topk-splitter
    control               3.51 s, 0.65 GB peak, 87 dump files
The original saw 0.168 GB/s and 51 s to 9 GB on a contended box; here it is 0.285 GB/s and 32 s.
Everything discrete -- 19, windowed-einsum-handler, snapshot 7, ragged_dot_rewriter, topk-splitter
-- is identical. The counts are the reproducible part; the slope is not.

WHEN IT WORKS
    * Any compile that hangs or grows without bound, on either backend. It is the only route to a
      localisation when there is no artifact to inspect, and it costs one killed compile.
    * The RSS trajectory itself is diagnostic. LINEAR with no plateau says "a loop that does not
      terminate or an object that grows per iteration", not "a big allocation": a big allocation is
      a step. The slope in GB/s is a number you can quote.

WHEN IT DOES NOT
    * The pass it names is an INFERENCE from two lower bounds, not a direct observation. The dump is
      the weaker one -- XLA only writes a snapshot when a pass CHANGED the module, so 13 files
      against 19 completed passes is normal and the last filename UNDERSTATES progress. The log is
      the tighter bound, because a pass logs its time when it finishes. The named pass is
      "control's pass number (last completed + 1)", which is only valid if both arms run the same
      pass list up to that point -- this recipe checks the prefix and says so.
    * It cannot tell you WHY the pass does not terminate. That took a size sweep and a predicate.
    * ``pass_timings`` on a dying compile is honest but useless: its subprocess is killed too and it
      returns ``{}`` with a ``stderr_tail``. Do not read the empty dict as "no passes ran".
    * The cap is yours to choose and it is a real risk to the machine. Pick a cap well under free
      RAM, and remember the child may also hold device memory.
    * Linux only: RSS is read from ``/proc``.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile
import time

import scopex


def _rss_gb(pid: int) -> float:
    try:
        for line in pathlib.Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024 / 1024
    except (FileNotFoundError, ProcessLookupError, ValueError):
        pass
    return 0.0


def _run_watched(src: str, dump_dir: str, *, rss_cap_gb: float, timeout_s: float,
                 sample_hz: float) -> dict:
    """Compile in a child with dumping and vmodule on; kill it at ``rss_cap_gb``.

    Both env settings must be made HERE, before the child imports jax: XLA reads XLA_FLAGS when its
    backend is first initialised, and the C++ logging layer reads TF_CPP_VMODULE when the shared
    library loads. Setting either in-process afterwards is a silent no-op. TF_CPP_MIN_LOG_LEVEL=0
    is required alongside VMODULE -- importing jax sets it to 1, which suppresses every VLOG.
    """
    env = dict(os.environ)
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    env.update(scopex.dump_flags(dump_dir, fusion=False, passes=".*"))
    env.update(scopex.vmodule_env("hlo_pass_pipeline=1"))
    log = pathlib.Path(dump_dir) / "_child.log"
    traj: list[tuple[float, float]] = []
    t0 = time.perf_counter()
    with log.open("w") as fh:
        p = subprocess.Popen([sys.executable, "-c", src], stdout=fh, stderr=subprocess.STDOUT,
                             env=env)
        killed = reason = None
        while p.poll() is None:
            el = time.perf_counter() - t0
            r = _rss_gb(p.pid)
            if r:
                traj.append((round(el, 2), round(r, 3)))
            if r > rss_cap_gb:
                killed, reason = True, f"RSS cap {rss_cap_gb} GB"
            elif el > timeout_s:
                killed, reason = True, f"timeout {timeout_s} s"
            if killed:
                p.kill()
                p.wait(timeout=30)
                break
            time.sleep(1.0 / sample_hz)
    wall = time.perf_counter() - t0
    peak = max((r for _, r in traj), default=0.0)
    slope = None
    if len(traj) > 4:
        # A straight line through the samples above the starting plateau. Linear-with-no-plateau is
        # the signature; a step would give a poor fit and you should look at the trajectory itself.
        xs = [t for t, _ in traj[2:]]
        ys = [r for _, r in traj[2:]]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
        slope = (cov / den) if den else None
    return {"killed": bool(killed), "kill_reason": reason, "returncode": p.returncode,
            "wall_s": round(wall, 2), "peak_rss_gb": round(peak, 2),
            "start_rss_gb": round(traj[0][1], 2) if traj else 0.0,
            "rss_slope_gb_per_s": round(slope, 4) if slope else None,
            "rss_trajectory": traj[::max(1, len(traj) // 12)],
            "log": log.read_text(errors="replace")}


def _passes_completed(log: str, module: str | None) -> list[str]:
    """Passes XLA logged as FINISHED, in order, for one module.

    Segmented on the pipeline headers exactly as ``scopex.pass_timings`` does, because the log
    interleaves jax's warm-up modules with the program you asked about and an unsegmented list
    attributes someone else's passes to you.
    """
    from scopex._parse import pass_pipeline_headers, pass_timing_lines

    if module:
        keep, on = [], False
        for line in log.splitlines():
            hdr = pass_pipeline_headers(line)
            if hdr:
                on = module in hdr[0][0]
            if on:
                keep.append(line)
        log = "\n".join(keep)
    return [t.name for t in pass_timing_lines(log)]


def _last_snapshot(dump_dir: str) -> dict | None:
    from scopex._parse import dump_snapshot_name

    rows = [m for m in (dump_snapshot_name(f) for f in os.listdir(dump_dir)) if m]
    if not rows:
        return None
    m = max(rows, key=lambda r: r["index"])
    return {"index": m["index"], "module": m["module"], "pipeline": m["pipeline"],
            "after_pass": m["after_pass"], "before_pass": m["before_pass"],
            "n_snapshots": len(rows)}


def which_pass_never_returned(case_src, control_src, *, module=None, rss_cap_gb=9.0,
                              timeout_s=300.0, sample_hz=2.0, keep_dumps=False) -> dict:
    """One line: how far did a compile that never finishes get, and what was it about to run?

    ``case_src`` is python source that compiles the runaway arm; ``control_src`` compiles an arm
    that COMPLETES, and is not optional -- its pass ORDER is what turns "stopped after 19 passes"
    into a pass name. Source rather than callables because the child needs the flags set before it
    imports jax, and because the whole point is that this must not run in your process.
    """
    dirs = {k: tempfile.mkdtemp(prefix=f"scopex-{k}-") for k in ("case", "control")}
    # SERIAL, and the case first: if it takes the machine down, the control's numbers were never
    # going to be comparable anyway.
    a = _run_watched(case_src, dirs["case"], rss_cap_gb=rss_cap_gb, timeout_s=timeout_s,
                     sample_hz=sample_hz)
    b = _run_watched(control_src, dirs["control"], rss_cap_gb=rss_cap_gb, timeout_s=timeout_s,
                     sample_hz=sample_hz)

    pa = _passes_completed(a["log"], module)
    pb = _passes_completed(b["log"], module)
    sa, sb = _last_snapshot(dirs["case"]), _last_snapshot(dirs["control"])

    prefix_ok = pa == pb[:len(pa)]
    culprit = pb[len(pa)] if prefix_ok and len(pb) > len(pa) else None

    out = {
        "case": {k: v for k, v in a.items() if k != "log"},
        "control": {k: v for k, v in b.items() if k != "log"},
        "passes_completed": {"case": len(pa), "control": len(pb)},
        "last_completed_pass": {"case": pa[-1] if pa else None,
                                "control": pb[-1] if pb else None},
        "last_snapshot": {"case": sa, "control": sb},
        "pass_prefix_identical": prefix_ok,
        "hung_in": culprit,
        # minus one: the child's own log lives in the same directory and is not an XLA artifact
        "dump_files": {k: len(os.listdir(v)) - 1 for k, v in dirs.items()},
    }
    if a["killed"] and culprit:
        out["verdict"] = (
            f"killed by {a['kill_reason']} after {a['wall_s']} s at {a['peak_rss_gb']} GB "
            f"(RSS +{a['rss_slope_gb_per_s']} GB/s, linear, no plateau). It completed {len(pa)} "
            f"passes; the control completes {len(pb)}. The pass it never returned from is "
            f"'{culprit}'.")
    elif a["killed"]:
        out["verdict"] = (
            f"killed by {a['kill_reason']} after {a['wall_s']} s at {a['peak_rss_gb']} GB, but the "
            f"pass lists do not share a prefix (case {len(pa)} / control {len(pb)} passes), so the "
            f"next-pass inference is NOT valid here. Read last_snapshot and the two pass lists.")
    else:
        out["verdict"] = (f"the case COMPLETED in {a['wall_s']} s at {a['peak_rss_gb']} GB peak. "
                          f"Nothing hung -- this recipe has nothing to say; use stage_split.py.")
    if not keep_dumps:
        import shutil
        for d in dirs.values():
            shutil.rmtree(d, ignore_errors=True)
    else:
        out["dump_dirs"] = dirs
    return out


def corpus_src(name: str) -> str:
    """python source that compiles corpus case ``name``."""
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
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    import _cases

    busy, desc = _cases.gpu_busy()
    print(f"GPU: {desc}")
    if busy:
        raise SystemExit("another process is on the GPU; this case is GPU-only. Re-run when free.")

    CASE, CONTROL = "topk_pow20p1024_k128", "topk_pow20_k128"
    print(f"running {CASE} to DEATH under a 9 GB cap, then {CONTROL} for its pass order.")
    print("this deliberately drives a process into an out-of-memory kill. Serial, one at a time.")
    r = which_pass_never_returned(corpus_src(CASE), corpus_src(CONTROL), module="jit_top_k")

    print(f"\n=== {CASE} vs {CONTROL} (GPU) " + "=" * 28)
    for arm in ("case", "control"):
        d = r[arm]
        print(f"  {arm:7s} killed={d['killed']!s:5s} {d['kill_reason']}  wall {d['wall_s']} s  "
              f"peak {d['peak_rss_gb']} GB  slope {d['rss_slope_gb_per_s']} GB/s")
        print(f"          rss {d['rss_trajectory']}")
    print(f"  passes done   {r['passes_completed']}")
    print(f"  last pass     {r['last_completed_pass']}")
    print(f"  last snapshot {r['last_snapshot']['case']}")
    print(f"  dump files    {r['dump_files']}")
    print(f"  prefix same?  {r['pass_prefix_identical']}")
    print(f"  VERDICT  {r['verdict']}")
