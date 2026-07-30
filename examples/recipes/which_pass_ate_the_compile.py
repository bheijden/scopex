"""Which single XLA pass is the compile, and is the profile you are reading complete?

RECIPE: ``which_pass_ate_the_compile(case, control=None)``.

This is the recipe with the widest reach in the corpus -- six of the arms below are answered by one
call and one number -- and it is also the one that shipped a CONFIDENTLY WRONG answer twice. Both
failures are now guarded here rather than merely fixed upstream, because a per-pass profile is a
dict of plausible numbers whether or not it is complete, and nothing about the dict says which.

FOUND ON six arms, and the numbers are the point:

    arm                    platform   the pass            share of compile     what scopex said
    convT64_dilate16       gpu        autotuner           71.65 s / 72.5 s     "remat-pipeline 0.12"
    switch_ident_1024      gpu        copy-insertion      177.2 s / 188.5 s    a millisecond pass
    bisect_m95             cpu        fusion              172 s / 395 s        "copy-insertion 13.0"
    xtile_issue            gpu        autotuner           7.140 s vs 0.0048 s  correct (1487x)
    dusfold_sum_300        gpu        constant_folding    top vs ~1 ms         correct
    jitfib_t22/t24         cpu        call-inliner        6.330 s vs 0.00373 s correct (1697x)

RE-MEASURED BY THIS RECIPE, 2026-07-29, jax 0.10.2, RTX 4090 laptop / this CPU:

    jitfib_t22        cpu   call-inliner 2.91 s = 57.5% of a 5.06 s compile, 1163x the control.
                            Naive coverage 1.044 -- i.e. >100% of the compile, the double count.
    dusfold_sum_300   gpu   constant_folding 2.23 s = 95.6% of 2.33 s, 3273x the control.
                            Naive coverage 1.92: the pipeline entries nearly DOUBLE the total.
    xtile_issue       gpu   autotuner 1.71 s = 74.3% of 2.30 s, 2132x the control's 0.0008 s.
    convT64_dilate16  gpu   autotuner 53.99 s = 99.5% of a 54.26 s compile, 64.1x the control.
    switch_ident_1024 gpu   copy-insertion 110.45 s = 95.3% of a 115.90 s compile, and
                            ``big_unit_lines == ['copy-insertion']`` -- THE `min` LINE FIRED LIVE
                            AND WAS COUNTED. Ratio against the control: 665,387x. This is the exact
                            arm on which scopex previously returned a plausible profile topped by a
                            millisecond-scale pass.

    THAT LAST LINE IS A FINDING CHANGING ITS VERDICT. convT was filed `no-signal`: every structural
    count at every level was exactly 1.00x, and the one instrument holding the answer returned an
    INVERTED profile that put more pass time in the fast arm than the slow one. With the unit table
    fixed it is now the cleanest hit in the corpus -- one call, one pass name, 99.5% of the wall
    clock. Nothing about the case changed; the parser did.

    Caveat recorded honestly: on this machine convT's autotuner took 53.99 s, which is UNDER the
    ~60 s point where XLA switches to ``min``, so convT's own ``big_unit_lines`` came back empty and
    that run did not exercise the guard the case originally broke. A faster machine HIDES bug #3
    rather than fixing it -- which is the argument for never letting the parse be lenient.
    ``switch_ident_1024`` did cross the threshold on the same day and the guard fired there, and the
    verbatim convT line still parses correctly in isolation, both ways::

        HLO pass: autotuner time: 1.19 min (71651421 us) (cumulative: 1.2 min, ...)
        -> 71.651 s from the parenthesised microseconds (exact=True)
        -> 71.4   s from the `min` headline when the parenthesis is absent
        -> ParseError, not a silent skip, when the unit is one scopex does not know

THE TWO GUARDS, both earned:

1. UNITS. XLA switches to ``min`` above ~60 s, so the pass most likely to be dropped by a us/ms/s
   parser is BY CONSTRUCTION the slowest one. ``convT64_dilate16``: exactly 1 of 640 pass lines used
   ``min``, it was the autotuner at 98.8% of the compile, and dropping it left a plausible dict
   topped by ``remat-pipeline: 0.1196``. Fixed in ``scopex._parse.UNITS`` (an unknown unit now
   raises), and this recipe still reports ``big_unit_lines`` so a re-regression is visible in the
   data rather than in a changelog.

2. COVERAGE. ``sum(passes)`` divided by the wall compile is the one number that would have screamed.
   On ``convT`` it is 0.006. Coverage is not in ``scopex.pass_timings``' return, so this
   recipe makes
   the child print its own wall time -- ``pass_timings`` merges child stdout into the log it parses,
   so the line rides along and costs no second compile.

AND A THIRD THING THE ONE-CALL FORM CANNOT DO: XLA logs a PIPELINE and the passes inside it with the
same ``HLO pass: NAME time:`` line, so a naive total double-counts. Measured on a two-line CPU
program: 31% of the total is pipeline entries. On ``jitfib_t22``: 11.581 s of "pass time" against a
10.856 s backend, i.e. 107% -- impossible, and it looked fine; re-measured here at coverage 1.044,
same story. On ``dusfold_sum_300`` the pipeline entries nearly DOUBLE the total (naive coverage
1.92). The pipeline names are in the log (``Running HLO pass pipeline on module M: PIPELINE``) and
``pass_timings`` reads them for its ``modules`` field and then throws the pipeline half away. So
this recipe keeps the log. Separating the two is NOT a name lookup -- see the block comment above
``_split``, which is where the first version of this recipe deleted the GPU autotuner.

WHEN IT WORKS
    The cost is inside XLA's HLO pass pipeline -- which includes autotuning, since the GPU autotuner
    is a pass. Works equally on CPU and GPU. Needs no dump, no flags in the parent, one compile.

WHEN IT DOES NOT
    * Cost BELOW the pass pipeline is invisible here. ``stackcond_n3000``: HLO passes are 0.872 s of
      a 17.8 s compile (5%) and 16.0 s sits in the LLVM ORC JIT, after the last pass. The tell is
      low coverage with no big-unit line -- then go to ``scopex.pass_timeline`` (mtime gaps) and
      ``scopex.codegen_size``. Low coverage is a ROUTING signal, not a failure.
    * Cost ABOVE the backend is invisible here: ``condrec_grad_512`` spends 116 s in trace. Run
      ``scopex.record`` first; if ``regime`` is not backend-bound this recipe is the wrong tool.
    * The profile sums over EVERY module XLA compiled, including JAX's warm-up
      ``jit_convert_element_type``. Pass ``module="jit_your_fn"`` when the program is small enough
      for that to matter; ``modules`` is in the result so you can see whether it does.
"""

from __future__ import annotations

import os
import subprocess
import sys

import scopex
from scopex import _parse                    # see `pipelines` below: the log-level fields
                                             # `scopex.pass_timings` does not return

import _cases

__all__ = ["which_pass_ate_the_compile", "source_for"]


# JAX_PLATFORMS DOES NOT KNOW THE WORD "gpu". Its vocabulary is {'cpu', 'cuda', 'rocm', 'tpu'}, and
# JAX_PLATFORMS=gpu fails with "Backend 'rocm' is not in the list of known backends" -- an error
# that names a backend nobody asked for and mentions neither the variable nor the word you passed.
# Every recipe here takes platform="gpu" because that is what the corpus and the findings call it.
_KNOWN = {"gpu": "cuda", "nvidia": "cuda", "cuda": "cuda", "cpu": "cpu",
          "rocm": "rocm", "tpu": "tpu"}


def _plat(p: str) -> str:
    try:
        return _KNOWN[p.lower()]
    except KeyError:
        raise ValueError(f"platform={p!r}; JAX_PLATFORMS accepts {sorted(set(_KNOWN.values()))} "
                         f"(pass 'gpu' or 'cuda' for an NVIDIA device)") from None


def source_for(name: str, *, platform: str = "cpu") -> str:
    """Python SOURCE that compiles corpus arm ``name`` -- what ``scopex.pass_timings`` wants.

    ``pass_timings`` takes source and not a function on purpose: ``TF_CPP_VMODULE`` is read by the
    C++ logging layer while ``import jax`` runs, so setting it in-process afterwards yields exactly
    zero log lines. A per-pass profile therefore needs a fresh interpreter, and a fresh interpreter
    needs source.

    The extra ``SCOPEX_WALL_COMPILE`` print is what makes coverage computable.
    """
    return (
        "import importlib.util, os, time\n"
        f"os.environ.setdefault('JAX_PLATFORMS', {platform!r})\n"
        "os.environ.setdefault('JAX_ENABLE_X64', '1')\n"
        "import jax\n"
        "jax.config.update('jax_enable_x64', True)\n"
        f"spec = importlib.util.spec_from_file_location('case_mod', {str(_cases.find(name))!r})\n"
        "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
        f"fn, args, _ = mod.CASES[{name!r}]\n"
        "low = jax.jit(fn).lower(*args)\n"
        "t0 = time.perf_counter(); low.compile(); t1 = time.perf_counter()\n"
        "print('SCOPEX_WALL_COMPILE %.6f' % (t1 - t0), flush=True)\n"
    )


def _profile(src: str, *, module: str | None = None, timeout: int = 3600) -> dict:
    """``scopex.pass_timings``, with the log kept.

    The one-call form is::

        scopex.pass_timings(src, module="jit_f")   # -> {"passes", "n_lines", "modules", ...}

    and it is the right call when you only want the ranking. It returns neither the PIPELINE names
    nor any wall time, so it can neither de-duplicate a pipeline against the passes inside it nor
    tell you that its own total covers 0.6% of the compile. Both of those have produced a published
    wrong answer, so this recipe runs the same subprocess with the same environment
    (:func:`scopex.vmodule_env`) and reads the log with the same parsers ``pass_timings`` uses --
    ``scopex._parse.pass_timing_lines`` and ``scopex._parse.pass_pipeline_headers``.

    THIS IS THE API GAP, stated once: ``pass_timings`` should return
    ``{"passes":..., "pipelines":..., "wall_compile_s":..., "coverage":...}``.
    """
    env = dict(os.environ)
    env.update(scopex.vmodule_env("hlo_pass_pipeline=1"))
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    p = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True,
                       timeout=timeout, env=env)
    log = p.stderr + p.stdout
    wall = None
    for line in p.stdout.splitlines():
        if line.startswith("SCOPEX_WALL_COMPILE"):
            wall = float(line.split()[1])
    if "HLO pass:" not in log:
        raise RuntimeError(
            "no pass lines in the child's log. Either the compile failed or vmodule is off.\n"
            f"rc={p.returncode}\nstderr tail:\n{p.stderr[-1500:]}")
    return {"log": log, "wall_compile_s": wall, "returncode": p.returncode, "module": module}


# ══════════════════════════════════════════════════════════════════════════════════════════════
# TELLING A PIPELINE'S AGGREGATE APART FROM A LEAF PASS, AND WHY THE NAME CANNOT DO IT
#
# XLA prints three kinds of line, and only reading all three in ORDER separates them:
#
#   hlo_pass_pipeline.cc:181]   HLO pass NAME                     <- about to run a pass
#   hlo_pass_pipeline.cc:303] Running HLO pass pipeline on module M: NAME
#   hlo_pass_pipeline.cc:176] HLO pass: NAME time: 460 us (...)   <- how long it took
#
# A NESTED PIPELINE announces itself as a pass and THEN opens a pipeline of the same name, so its
# `time:` line is the SUM of its children and adding it to them double-counts:
#
#   :181]   HLO pass simplification
#   :303] Running HLO pass pipeline on module jit_f: simplification
#   :181]   HLO pass algsimp        :176] HLO pass: algsimp time: 33 us
#   :181]   HLO pass dce            :176] HLO pass: dce time: 12 us
#   :176] HLO pass: simplification time: 460 us      <- the aggregate
#
# THE FIRST VERSION OF THIS RECIPE DE-DUPLICATED BY NAME -- drop any timing whose name also appears
# as a pipeline -- and that rule is WRONG in the one place it matters most. On GPU the order is
# INVERTED for the autotuner: a top-level pipeline named `autotuner` containing a leaf pass also
# named `autotuner`.
#
#   :303] Running HLO pass pipeline on module jit__issue: autotuner   <- pipeline opens FIRST
#   :181]   HLO pass autotuner                                         <- a real leaf pass
#   :176] HLO pass: autotuner time: 1.51 s (1511020 us) (#called: 3064)
#
# So the name rule silently deleted `autotuner` -- 1.511 s of a 2.06 s compile on xtile_issue, and
# 98.8% of the compile on convT64_dilate16. That is bug #3 rebuilt out of different parts: a blind
# spot that lands exactly on the pass being looked for. Caught by running the recipe on a GPU arm
# whose answer was already known.
#
# THE ORDER IS THE DISCRIMINATOR. An occurrence is a nested pipeline only when its `:181` pass
# announcement is IMMEDIATELY followed by a `:303` header for the same name. Tracked on a stack,
# because the aggregate `time:` line arrives when the nested pipeline closes, LIFO.
# ══════════════════════════════════════════════════════════════════════════════════════════════

# The `:181` announcement is the only one of the three that scopex._parse does not read. `HLO pass:`
# (with the colon) is the timing line, so the space is what distinguishes them.
_ANNOUNCE = __import__("re").compile(r"\]\s+HLO pass (?P<name>[^\s:][^\n]*?)\s*$")


def _split(log: str, *, module: str | None = None) -> dict:
    """Pass entries, pipeline entries, and the guards -- all from one log, read IN ORDER."""
    passes: dict[str, float] = {}
    pipelines: dict[str, float] = {}
    headers: list[tuple[str, str]] = []
    times = []
    stack: list[str] = []
    pending: str | None = None
    on = module is None

    for line in log.splitlines():
        hdr = _parse.pass_pipeline_headers(line)
        if hdr:
            mod, pipe = hdr[0]
            headers.append(hdr[0])
            if module:
                on = module in mod
            if pending == pipe:              # announced as a pass, then opened as a pipeline
                stack.append(pipe)
            pending = None
            continue
        m = _ANNOUNCE.search(line)
        if m:
            pending = m.group("name")
            continue
        got = _parse.pass_timing_lines(line)
        if not got:
            continue
        pending = None
        t = got[0]
        if stack and stack[-1] == t.name:
            stack.pop()
            if on:
                pipelines[t.name] = pipelines.get(t.name, 0.0) + t.seconds
        elif on:
            times.append(t)
            passes[t.name] = passes.get(t.name, 0.0) + t.seconds

    return {
        "passes": dict(sorted(passes.items(), key=lambda kv: -kv[1])),
        "pipelines": dict(sorted(pipelines.items(), key=lambda kv: -kv[1])),
        "modules": sorted({m for m, _ in headers}),
        "n_pass_lines": len(times),
        # A line XLA printed in min/h is the one a us/ms/s parser drops, and it is the slow one.
        "big_unit_lines": sorted({t.name for t in times if t.unit in ("min", "m", "h", "hr")}),
        "exact_us_lines": sum(1 for t in times if t.exact),
    }


def which_pass_ate_the_compile(case: str, control: str | None = None, *,
                               platform: str = "cpu", module: str | None = None,
                               timeout: int = 3600) -> dict:
    """Rank XLA's passes for one arm (and optionally a control), with the completeness guards.

    FOUND ON: convT64_dilate16 / switch_ident_1024 / xtile_issue / dusfold_sum_300 (gpu),
    bisect_m95 / jitfib_t22 (cpu).
    MEASURED: convT autotuner 71.65 s of a 72.505 s compile (98.8%), 1 of 640 lines in ``min``;
    switch_ident_1024 copy-insertion 177.15 s of 188.46 s (94.0%); jitfib_t24 call-inliner 6.330 s
    vs a control's 0.00373 s = 1697x; xtile autotuner 7.140 s vs 0.0048 s = 1487x.

    Returns, per arm: ``passes`` (pipeline entries removed), ``pipelines``, ``top``,
    ``wall_compile_s``, ``coverage`` = sum(passes)/wall, ``big_unit_lines``, ``modules``.
    With a control it adds ``ratio`` per pass name and ``verdict``.

    READ IT IN THIS ORDER.
      1. ``coverage``. Below ~0.3 the passes are not where the time is -- route to
         ``scopex.pass_timeline`` / ``scopex.codegen_size``, do not believe ``top``.
      2. ``big_unit_lines``. Non-empty means at least one pass was slow enough that XLA changed
         units; that is the pass, and it is the one three parsers have dropped.
      3. ``top`` and, if you have a control, ``ratio`` -- a pass that is merely BIG in both arms is
         not the finding; a pass that is 1697x is.
    """
    out: dict = {"case": case, "control": control, "platform": platform}
    for label, name in (("case", case), ("control", control)):
        if name is None:
            continue
        raw = _profile(source_for(name, platform=_plat(platform)), module=module, timeout=timeout)
        r = _split(raw["log"], module=module)
        r["wall_compile_s"] = raw["wall_compile_s"]
        tot = sum(r["passes"].values())
        r["pass_total_s"] = round(tot, 6)
        r["coverage"] = (round(tot / raw["wall_compile_s"], 4)
                         if raw["wall_compile_s"] else None)
        # The double count, quantified rather than asserted.
        r["naive_total_s"] = round(tot + sum(r["pipelines"].values()), 6)
        r["naive_coverage"] = (round(r["naive_total_s"] / raw["wall_compile_s"], 4)
                               if raw["wall_compile_s"] else None)
        top = next(iter(r["passes"].items()), (None, 0.0))
        r["top"] = {"pass": top[0], "seconds": round(top[1], 6),
                    "share_of_wall": (round(top[1] / raw["wall_compile_s"], 4)
                                      if raw["wall_compile_s"] else None)}
        out[label] = r

    if control is not None:
        a, b = out["case"]["passes"], out["control"]["passes"]
        # RANK BY THE CASE'S OWN SECONDS, NOT BY THE RATIO. A pass that took 2 us in the control and
        # 4 ms in the case is a 2000x ratio and is not the finding; on jitfib_t22 an unfiltered
        # ratio ranking puts `qr_expander` (2183x, 0.4 ms) above `call-inliner` (1490x, 3.9 s).
        floor = 0.01 * max(sum(a.values()), 1e-12)
        out["ratio"] = {
            k: (round(a[k] / b[k], 1) if b.get(k, 0) > 0 else float("inf"))
            for k, _ in sorted(a.items(), key=lambda kv: -kv[1]) if a[k] >= floor}
        out["case_only_passes"] = sorted(set(a) - set(b))
        out["control_only_passes"] = sorted(set(b) - set(a))
    cov = out["case"].get("coverage")
    out["verdict"] = (
        "coverage unknown -- no wall time" if cov is None else
        f"PASS-BOUND: {out['case']['top']['pass']} is "
        f"{100 * out['case']['top']['share_of_wall']:.1f}% of the compile" if cov >= 0.3 else
        f"NOT PASS-BOUND: passes are only {100 * cov:.1f}% of the compile -- the time is below the "
        f"pass pipeline (LLVM/codegen) or above it (trace/lower). Use scopex.pass_timeline and "
        f"scopex.codegen_size, and do NOT report {out['case']['top']['pass']!r} as the culprit.")
    return out


if __name__ == "__main__":
    import json

    CASE, CONTROL = "jitfib_t22", "jitfib_t22_control"
    print(f"{CASE}  --  {_cases.note(CASE)}\n")
    r = which_pass_ate_the_compile(CASE, CONTROL, platform="cpu")

    for arm, name in (("case", CASE), ("control", CONTROL)):
        a = r[arm]
        print(f"── {arm}: {name} ──────────────────────────────")
        print(f"   wall compile      {a['wall_compile_s']:.3f} s")
        print(f"   pass total        {a['pass_total_s']:.4f} s   coverage {a['coverage']}")
        print(f"   naive total       {a['naive_total_s']:.4f} s   coverage {a['naive_coverage']}"
              f"   <- pipelines counted twice")
        print(f"   modules           {a['modules']}")
        print(f"   big-unit lines    {a['big_unit_lines'] or 'none (no pass exceeded ~60 s)'}")
        for k, v in list(a["passes"].items())[:5]:
            print(f"      {k:<45s} {v:9.4f} s")
    print("\ncase/control ratio, for passes worth >=1% of the case's pass time:")
    for k, v in list(r["ratio"].items())[:6]:
        print(f"   {k:<45s} {v}x")
    print("\npasses that ran ONLY in the case:", r["case_only_passes"] or "none")
    print("\nVERDICT:", r["verdict"])
    print("\n" + json.dumps({"top": r["case"]["top"], "coverage": r["case"]["coverage"],
                             "naive_coverage": r["case"]["naive_coverage"],
                             "big_unit_lines": r["case"]["big_unit_lines"]}, indent=1))
