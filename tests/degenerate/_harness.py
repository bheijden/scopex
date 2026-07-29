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

A case is REPRODUCED when compile >= MIN_COMPILE_S and compile/runtime >= MIN_RATIO and, where a
control exists, compile >= MIN_VS_CONTROL x the control's compile.
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
import json, os, sys, time
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
print("__RESULT__" + json.dumps({
    "name": name, "note": note,
    "lower_s": t1 - t0, "compile_s": t2 - t1, "first_run_s": t3 - t2,
    "runtime_s": (t4 - t3) / 3, "hlo_lines": n_instr}))
'''


def _run_one(path: pathlib.Path, name: str, timeout: int = 900) -> dict:
    env = dict(os.environ)
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    p = subprocess.run([sys.executable, "-c", _CHILD, str(path), name],
                       capture_output=True, text=True, timeout=timeout, env=env)
    for line in p.stdout.splitlines():
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    return {"name": name, "error": (p.stderr or p.stdout)[-800:]}


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
    if "error" in r:
        return "ERROR"
    if r["compile_s"] < MIN_COMPILE_S:
        return "no (below floor)"
    if r["compile_s"] / max(1e-9, r["runtime_s"]) < MIN_RATIO:
        return "no (compile/runtime low)"
    if control and not control.get("error"):
        f = r["compile_s"] / max(1e-9, control["compile_s"])
        return f"YES ({f:.0f}x control)" if f >= MIN_VS_CONTROL else f"no ({f:.1f}x control)"
    return "YES (no control)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cases", nargs="*", help="case names; default all discovered")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", default=str(HERE / "results.json"))
    a = ap.parse_args()

    found = discover()
    names = a.cases or sorted(found)
    missing = [n for n in names if n not in found]
    if missing:
        print(f"unknown cases: {missing}\nknown: {sorted(found)}")
        return 2
    print(f"{len(names)} case(s), {a.rounds} round(s), one subprocess each\n")

    runs: dict[str, list[dict]] = {n: [] for n in names}
    for rd in range(a.rounds):
        seq = names[rd % len(names):] + names[:rd % len(names)]      # rotate, never reverse
        for n in seq:
            f, key = found[n]
            t0 = time.perf_counter()
            r = _run_one(f, key, a.timeout)
            r["wall_s"] = time.perf_counter() - t0
            runs[n].append(r)
            tag = "ERR" if "error" in r else f"{r['compile_s']:8.2f}s compile"
            print(f"  round {rd + 1}  {n:38s} {tag}")

    print()
    hdr = f"{'case':38s} {'compile':>9s} {'runtime':>10s} {'ratio':>9s}  reproduced"
    print(hdr); print("-" * len(hdr))
    agg, out = {}, {}
    for n in names:
        ok = [r for r in runs[n] if "error" not in r]
        if not ok:
            print(f"{n:38s}     ERROR   {runs[n][0].get('error','')[:40]}")
            out[n] = {"error": runs[n][0].get("error", "")[:800]}
            continue
        agg[n] = {k: statistics.median(r[k] for r in ok)
                  for k in ("compile_s", "runtime_s", "lower_s")}
    for n in names:
        if n not in agg:
            continue
        r = agg[n]
        ctrl = agg.get(f"{n}_control") or (agg.get(n[:-8]) if n.endswith("_control") else None)
        verdict = classify(r, ctrl)
        print(f"{n:38s} {r['compile_s']:9.2f} {r['runtime_s']:10.6f} "
              f"{r['compile_s'] / max(1e-9, r['runtime_s']):9.0f}  {verdict}")
        out[n] = {**r, "reproduced": verdict}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
