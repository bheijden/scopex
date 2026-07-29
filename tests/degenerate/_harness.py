"""Measure every degenerate case identically, serially, in a fresh process each time.

Three properties this harness has that ad-hoc timing loops do not, each because getting it wrong
has produced a wrong published number before:

**Fresh subprocess per measurement.** ``jax.monitoring`` listeners are process-global and cannot be
removed in jax 0.10.2, so repeated in-process measurement accumulates them. Compilation caches and
XLA's own autotune cache also persist. One process per measurement, always.

**Order rotation.** A machine drifts. Measuring arm A then arm B repeatedly credits the drift to B.
Arms are rotated (never reversed -- a reversal puts the same arm first twice). A 16% "win" once
survived four rounds of un-rotated measurement and vanished under rotation.

**Paired per-round ratios.** The statistic is the median of per-round ``case/control`` ratios, not
the ratio of medians. Pairing across rounds instead of within once manufactured a 2.41x multiplier
from 1.89x data.

**Device is an axis, not a footnote.** Which backend you compile for decides which passes run at
all, so it decides whether a pathology exists. Measured on jax#32704 (chained 2D fancy indexing),
ncycles=9, same code, same sizes:

    CPU   218.666 s   against a 0.882 s control   -- 248x, and XLA prints its own "Very slow
                                                     compile?" warning
    GPU     ~1 s      against a ~1 s control      -- no growth at all across ncycles 4..9

Had only the GPU been run, that case would have been filed "does not reproduce" and discarded. So
every result here is labelled with its platform, a verdict is never rendered without one, and
`--platform` takes a comma-separated list. An absence on one backend is a result ABOUT that
backend, never about the case.

A case is REPRODUCED when compile >= MIN_COMPILE_S and compile/runtime >= MIN_RATIO and, where a
control exists, compile >= MIN_VS_CONTROL x the control's compile -- on a NAMED platform.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
MIN_COMPILE_S = 3.0
MIN_RATIO = 1000.0
MIN_VS_CONTROL = 10.0

_CHILD = r'''
import json, os, resource, sys, time
os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax
jax.config.update("jax_enable_x64", True)
assert not jax.config.jax_compilation_cache_dir, "persistent cache on: timings would be garbage"
import importlib.util
path, name = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("case_mod", path)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
fn, args, note = mod.CASES[name]
t0 = time.perf_counter(); low = jax.jit(fn).lower(*args); t1 = time.perf_counter()
comp = low.compile(); t2 = time.perf_counter()
out = jax.block_until_ready(comp(*args)); t3 = time.perf_counter()
for _ in range(3):
    out = jax.block_until_ready(comp(*args))
t4 = time.perf_counter()
try:
    n_instr = len(comp.runtime_executable().hlo_modules()[0].to_string().splitlines())
except Exception:
    n_instr = -1
# COMPILE-TIME MEMORY, not just seconds. Several pathologies spend their cost in host RSS or in
# device temporaries rather than in wall clock -- constant folding materialising a literal, an
# interval-packing search, a rematerialization threshold. Without these the harness scores every
# such case "no (below floor)" and its real artifact has to live in prose.
peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
try:
    ma = comp.memory_analysis()
    mem = {"temp_bytes": getattr(ma, "temp_size_in_bytes", None),
           "output_bytes": getattr(ma, "output_size_in_bytes", None),
           "arg_bytes": getattr(ma, "argument_size_in_bytes", None),
           "alias_bytes": getattr(ma, "alias_size_in_bytes", None)}
except Exception:
    mem = {}
print("__RESULT__" + json.dumps({
    "name": name, "note": note,
    "platform": jax.devices()[0].platform,
    "device": str(jax.devices()[0]),
    "lower_s": t1 - t0, "compile_s": t2 - t1, "first_run_s": t3 - t2,
    "runtime_s": (t4 - t3) / 3, "hlo_lines": n_instr,
    "peak_rss_mb": peak_rss_mb, "memory": mem}))
'''


def _run_one(path: pathlib.Path, name: str, timeout: int = 900, platform: str = "") -> dict:
    env = dict(os.environ)
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    if platform:
        env["JAX_PLATFORMS"] = platform
    # A timeout is DATA, not a crash. `subprocess.run` raises `TimeoutExpired`, and letting that
    # propagate aborts the whole sweep and discards every row already measured -- which is exactly
    # what happened on `stackcond_n30000`, a case whose compile legitimately runs past 300 s. One
    # unbounded case must not cost the other hundred their results.
    try:
        p = subprocess.run([sys.executable, "-c", _CHILD, str(path), name],
                           capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = (e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        return {"name": name, "timeout_s": timeout,
                "error": f"TIMEOUT after {timeout} s (still compiling)\n" + (err or out)[-600:],
                "requested_platform": platform or "<default>"}
    for line in p.stdout.splitlines():
        if line.startswith("__RESULT__"):
            r = json.loads(line[len("__RESULT__"):])
            # XLA prints this itself when a single module is slow. It is a free second opinion on
            # the verdict, emitted by the compiler rather than inferred by us.
            r["xla_slow_warning"] = "Very slow compile?" in (p.stdout + p.stderr)
            return r
    return {"name": name, "error": (p.stderr or p.stdout)[-800:],
            "requested_platform": platform or "<default>"}


def gpu_contention() -> tuple[int, int, list[str]]:
    """(used_MiB, total_MiB, other_pids). Empty pid list when nothing else holds the device."""
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        used, total = (int(x) for x in q.stdout.strip().splitlines()[0].split(","))
        a = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        pids = [x.strip() for x in a.stdout.splitlines() if x.strip()]
        return used, total, pids
    except Exception:
        return -1, -1, []


def preflight(strict: bool = True) -> bool:
    """Refuse to publish absolute seconds measured on a contended device.

    This is not hypothetical. During one mining pass another workload on this box held 11-14 GB of
    16 GB across up to 17 processes; it caused CUDA_ERROR_OUT_OF_MEMORY inside the conv autotuner
    and blocked CUDA init outright for two probes. Every absolute compile time measured then is an
    UPPER BOUND. Control RATIOS survive contention when the two arms run back-to-back under the
    same load -- which is the deeper reason every case here ships a control.
    """
    used, total, pids = gpu_contention()
    if used < 0:
        print("  preflight: no nvidia-smi; assuming CPU-only run")
        return True
    busy = len(pids) > 0 or (total > 0 and used / total > 0.15)
    print(f"  preflight: GPU {used}/{total} MiB, {len(pids)} other compute process(es)"
          + ("   <== CONTENDED" if busy else "   ok"))
    if busy:
        print("  Absolute seconds from this run are UPPER BOUNDS. Control ratios remain usable\n"
              "  because each arm meets the same load. Re-baseline on a quiet device before\n"
              "  quoting any absolute number.")
    return not busy or not strict


def discover(root: pathlib.Path = HERE) -> dict[str, tuple[pathlib.Path, str]]:
    """``{case_name: (file, key)}`` for every ``case_*.py`` exposing ``CASES``."""
    found: dict[str, tuple[pathlib.Path, str]] = {}
    for f in sorted(root.glob("case_*.py")):
        spec = importlib.util.spec_from_file_location(f.stem, f)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            print(f"  !! {f.name} failed to import: {type(e).__name__}: {e}")
            continue
        for k in getattr(mod, "CASES", {}):
            if k in found:
                print(f"  !! duplicate case name {k!r} in {f.name}")
            found[k] = (f, k)
    return found


def classify(r: dict, control: dict | None) -> str:
    """Verdict for one case on one platform.

    THE CONTROL OUTRANKS EVERYTHING. When a near-identical fast twin exists, the difference between
    the two arms IS the pathology and nothing else has to be inferred. ``compile/runtime`` is only
    a stand-in for "is this compile-bound at all", needed when no control exists -- and it is a bad
    stand-in whenever the case does real work at runtime. Applying it above the control comparison
    scored jax#32704 at ncycles=6 as "no" while it was compiling 9.7x slower than its own control,
    purely because a 200k-element gather takes 55 ms to run.
    """
    if "timeout_s" in r:
        # Distinguished from ERROR on purpose: a case that never finishes is the strongest possible
        # reproduction, not a broken file. `size_cliff_bisection_unroll` exists for this outcome.
        return f"TIMEOUT (>{r['timeout_s']}s, still compiling)"
    if "error" in r:
        return "ERROR"
    if r["compile_s"] < MIN_COMPILE_S:
        return "no (below floor)"
    if control and not control.get("error"):
        f = r["compile_s"] / max(1e-9, control["compile_s"])
        return f"YES ({f:.0f}x control)" if f >= MIN_VS_CONTROL else f"no ({f:.1f}x control)"
    if r["compile_s"] / max(1e-9, r["runtime_s"]) < MIN_RATIO:
        return "no (compile/runtime low, no control)"
    return "YES (no control)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cases", nargs="*", help="case names; default all discovered")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--platform", default="",
                    help="comma-separated backends, e.g. 'cpu,cuda'. Default: whatever jax picks. "
                         "A pathology can exist on one backend and not another -- see the module "
                         "docstring -- so a sweep is usually what you want.")
    ap.add_argument("--out", default=str(HERE / "results.json"))
    a = ap.parse_args()

    found = discover()
    names = a.cases or sorted(found)
    missing = [n for n in names if n not in found]
    if missing:
        print(f"unknown cases: {missing}\nknown: {sorted(found)}")
        return 2
    plats = [x.strip() for x in a.platform.split(",") if x.strip()] or [""]
    quiet = preflight(strict=False)
    if not quiet:
        print("  -> results will be tagged contended=true\n")
    print(f"{len(names)} case(s) x {len(plats)} platform(s) x {a.rounds} round(s), "
          f"one subprocess each\n")

    out: dict = {}
    for plat in plats:
        runs: dict[str, list[dict]] = {n: [] for n in names}
        for rd in range(a.rounds):
            seq = names[rd % len(names):] + names[:rd % len(names)]   # rotate, never reverse
            for n in seq:
                f, key = found[n]
                t0 = time.perf_counter()
                r = _run_one(f, key, a.timeout, plat)
                r["wall_s"] = time.perf_counter() - t0
                runs[n].append(r)
                tag = ("TIMEOUT" if "timeout_s" in r else
                       "ERR" if "error" in r else f"{r['compile_s']:8.2f}s compile")
                warn = "  [XLA: very slow compile]" if r.get("xla_slow_warning") else ""
                print(f"  {r.get('platform', plat or '?'):5s} round {rd + 1}  "
                      f"{n:36s} {tag}{warn}")

        agg: dict = {}
        for n in names:
            ok = [r for r in runs[n] if "error" not in r]
            if not ok:
                first = runs[n][0]
                row = {"platform": plat or "default", "error": first.get("error", "")[:800]}
                # A timeout is a verdict, not a missing measurement: carry it into the table.
                if "timeout_s" in first:
                    row["timeout_s"] = first["timeout_s"]
                    row["reproduced"] = classify(first, None)
                out[f"{plat or 'default'}/{n}"] = row
                continue
            agg[n] = {k: statistics.median(r[k] for r in ok)
                      for k in ("compile_s", "runtime_s", "lower_s")}
            agg[n]["peak_rss_mb"] = statistics.median(
                r.get("peak_rss_mb", 0.0) for r in ok)
            agg[n]["memory"] = ok[0].get("memory", {})
            agg[n]["platform"] = ok[0].get("platform", plat or "?")
            agg[n]["xla_slow_warning"] = any(r.get("xla_slow_warning") for r in ok)

        actual = next((v["platform"] for v in agg.values()), plat or "?")
        print(f"\n=== PLATFORM: {actual} " + "=" * 50)
        hdr = f"{'case':36s} {'compile':>9s} {'runtime':>10s} {'ratio':>9s}  reproduced"
        print(hdr)
        print("-" * len(hdr))
        for n in names:
            if n not in agg:
                row = out.get(f"{plat or 'default'}/{n}", {})
                print(f"{n:36s}     " + (row.get("reproduced") or "ERROR"))
                continue
            r = agg[n]
            ctrl = agg.get(f"{n}_control")
            verdict = classify(r, ctrl)
            if r["xla_slow_warning"]:
                verdict += "  [XLA agrees]"
            print(f"{n:36s} {r['compile_s']:9.2f} {r['runtime_s']:10.6f} "
                  f"{r['compile_s'] / max(1e-9, r['runtime_s']):9.0f}  {verdict}")
            out[f"{actual}/{n}"] = {**r, "reproduced": verdict, "gpu_contended": not quiet}

    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.out}")
    if len(plats) > 1:
        print("NOTE: a 'no' on one platform is a statement about THAT platform, not about the case.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
