"""RECIPE -- WHICH PASSES DID XLA ACTUALLY RUN, and is the guilty pass even in the list?

Two arms can differ not in how long a pass took but in WHICH PASSES RAN AT ALL. That is a
pass-SELECTION difference, and it wants the opposite fix from "one pass is slow": you are looking
for the predicate that routed the two arms down different pipelines. It is also the only honest way
to answer "does this pathology exist on the backend I am compiling for?" -- if the guilty pass is
not in this backend's pass list, the case cannot reproduce here and saying "does not reproduce"
without naming the backend is a false negative.

    scopex.dump(d, passes='.*')             one snapshot per pass, per arm
    [s.name for s in scopex.pass_growth(d, module="jit_yours")]     the route XLA took
    then diff the two lists and report the FIRST INDEX where they part

FOUND ON:

  topk_pow20p2048_k128 / topk_pow20p1024_k128 (jax#19653 -- `lax.top_k` at particular input
  LENGTHS), GPU (CUDA, RTX 4090 Laptop sm_8.9), jax 0.10.2, x64
      The case reaches 19 passes and the control 303; the routes diverge at INDEX 19, and the pass
      the case dies entering is `topk-splitter`. Nothing else separates the arms: single-equation
      jaxpr, same primitive, same k, same dtype, 1024 elements apart out of 1.05 M. The pass list
      IS the signal.
      RSS on the way down: 0.642 -> 9.016 GB, linear at 0.153 GB/s, killed at 55.5 s (the +1024 arm
      is 0.168 GB/s -- indistinguishable).
      Predicate, 10/10 across the sweep: `n % 1024 == 0 and n // 1024 is not a power of two`.
      For the runaway-compile half of this -- watchdog, RSS trajectory, reading a dump left by a
      process that was killed -- see `compile_that_never_finishes.py`, which is the recipe for that
      question. This one is about the pass LIST.

  argsort_f32_1e6 / _control (xla#35587 -- `jnp.argsort` on float32 vs int32), GPU, x64
      `estimate-cub-sort-scratch-size` appears ONLY in the control's pass list. One dtype token
      changes which passes run. backend 5.534 s vs 0.469 s = 11.8x, optimized HLO 44 vs 12
      instructions, jaxpr identical.

MEASURED (re-run for this recipe, JAX_PLATFORMS=cpu, x64) -- and it is a NEGATIVE RESULT, which is
the point of the recipe:
    topk_pow20p2048_k128 vs topk_pow20_k128_control, on CPU:
        29 passes vs 29, IDENTICAL sequences, routes never diverge
        both compile in 2.76 s, peak RSS 0.566 / 0.563 GB
        `topk-splitter` does not appear in the CPU pass list at all
    So on CPU this case cannot reproduce, and the reason is visible rather than inferred: the pass
    that blows up is not part of this backend's pipeline. The corpus headline number for the same
    principle is jax#32704 at 248x on CPU and DEAD FLAT on GPU.
    (The GPU arm was NOT re-measured in this session: nvidia-smi showed another compute process at
    100% utilisation and racing it would have made both sets of numbers noise.)

WHEN IT WORKS
    Whenever two arms are structurally indistinguishable. It is cheap -- one dumped compile per arm
    -- and it answers a question no count and no timing can: did XLA take a different route?

WHEN IT DOES NOT
    * `pass_growth` picks the module with the MOST snapshots by default, and on a dump of a compile
      that died early that is often one of jax's warm-up modules rather than yours. Pass `module=`.
      This recipe requires it and refuses to guess.
    * The list comes from DUMP FILENAMES, so it contains only passes XLA snapshotted. A pass that
      the dump regex did not match is absent from the list without being absent from the compile.
      `passes='.*'` is what makes the list complete; anything narrower makes it a sample.
    * A pass present in both lists can still be the difference -- that is the other case, and
      `pass_divergence.py` (sizes) or `pass_timings_coverage.py` (seconds) handles it. An identical
      pass list is not "no difference", it is "not a routing difference".
    * It names a pass, not a cause. Why `topk-splitter` fires on one length and not another is in
      XLA's emitter source, not in any artifact on disk.
"""

from __future__ import annotations

import os
import tempfile

import scopex


def _dump(name: str, *, platform: str, timeout: int) -> str:
    import _cases
    d = tempfile.mkdtemp(prefix=f"scopex-passes-{name}-")
    _cases.run_in_subprocess(
        f'fn, args = _cases.load({name!r})\n'
        f'import jax\n'
        f'with scopex.dump({d!r}, passes=".*", fusion=False, keep=True) as dd:\n'
        f'    jax.jit(fn).lower(*args).compile()\n'
        f'emit({{"dir": dd}})\n', platform=platform, timeout=timeout)
    if len(os.listdir(d)) < 3:
        raise RuntimeError(f"empty dump for {name}: XLA_FLAGS was set too late. See scopex.dump.")
    return d


def which_passes_ran(case, control, *, module, platform="cpu", timeout=3600,
                     dump_dirs: tuple | None = None) -> dict:
    """One line: did the two arms take the same route through XLA, and where did they part?

    ``case``/``control`` are corpus case names (dumped here, one subprocess each) unless you pass
    ``dump_dirs=(case_dir, control_dir)``. ``module`` is REQUIRED: it is a substring of your
    module's dump stem, e.g. ``"jit_top_k"``. Defaulting it is how you end up reading jax's warm-up
    module's pass list and believing it.
    """
    a_dir, b_dir = dump_dirs or (_dump(case, platform=platform, timeout=timeout),
                                 _dump(control, platform=platform, timeout=timeout))
    a = [s.name for s in scopex.pass_growth(a_dir, module=module)]
    b = [s.name for s in scopex.pass_growth(b_dir, module=module)]
    if not a or not b:
        raise RuntimeError(f"no snapshots for module={module!r}. Stems present -- case: "
                           f"{scopex.modules_in(a_dir)}, control: {scopex.modules_in(b_dir)}")

    n = min(len(a), len(b))
    first = next((i for i in range(n) if a[i] != b[i]), None)
    identical = first is None and len(a) == len(b)
    only_a = [p for p in a if p not in set(b)]
    only_b = [p for p in b if p not in set(a)]

    if identical:
        verdict = (f"IDENTICAL ROUTES -- {len(a)} passes, same names, same order. This is not a "
                   f"pass-selection difference. If the arms still differ in cost, the difference "
                   f"is inside a pass they both ran (pass_divergence.py, "
                   f"pass_timings_coverage.py) or below HLO (phase_timeline.py).")
    elif first is not None:
        verdict = (f"ROUTES PART AT INDEX {first}: the case runs {a[first]!r} where the control "
                   f"runs {b[first]!r}. That is a pass-SELECTION difference -- look for the "
                   f"predicate that routed them, not for a slow pass.")
    else:
        verdict = (f"SAME PREFIX, DIFFERENT LENGTH -- case {len(a)} passes, control {len(b)}. The "
                   f"shorter arm stopped early (a killed compile leaves exactly this shape; see "
                   f"compile_that_never_finishes.py) or ran a shorter pipeline.")

    return {
        "dirs": {"case": a_dir, "control": b_dir},
        "n_passes": {"case": len(a), "control": len(b)},
        "identical": identical,
        "first_divergence": first,
        "around_divergence": {
            "case": a[max(0, (first or n) - 2):(first or n) + 3],
            "control": b[max(0, (first or n) - 2):(first or n) + 3]},
        "passes_only_in_case": only_a,
        "passes_only_in_control": only_b,
        "verdict": verdict,
    }


def pass_runs_here(dump_dir, needle: str, *, module) -> dict:
    """Is a named pass in this backend's pass list at all? The device-axis check in one call."""
    names = [s.name for s in scopex.pass_growth(dump_dir, module=module)]
    hits = [n for n in names if needle in n]
    return {"needle": needle, "found": bool(hits), "matches": hits, "n_passes": len(names)}


if __name__ == "__main__":
    import _cases

    busy, desc = _cases.gpu_busy()
    print(f"nvidia-smi: {desc}")
    print("This recipe's cases were FOUND on GPU. The run below is on CPU -- deliberately, "
          "because\nthe CPU answer is itself the finding, and because racing another process for "
          "the device\nwould make every number noise.\n")

    CASE, CONTROL, MODULE = "topk_pow20p2048_k128", "topk_pow20_k128_control", "jit_top_k"
    r = which_passes_ran(CASE, CONTROL, module=MODULE, platform="cpu")

    print(f"=== {CASE} vs {CONTROL} (cpu) " + "=" * 20)
    print(f"  passes                 {r['n_passes']}")
    print(f"  identical routes       {r['identical']}")
    print(f"  first divergence       {r['first_divergence']}")
    print(f"  only in case           {r['passes_only_in_case'][:5]}")
    print(f"  only in control        {r['passes_only_in_control'][:5]}")
    for needle in ("topk-splitter", "cub-sort", "sort"):
        print(f"  does {needle!r} run here? "
              f"{pass_runs_here(r['dirs']['case'], needle, module=MODULE)}")
    print(f"\n  VERDICT {r['verdict']}")
    print("\n  PUBLISHED (GPU, NOT re-measured here): 19 passes vs 303, routes part at index 19,\n"
          "  the case dies entering `topk-splitter`, RSS 0.642 -> 9.016 GB at 0.153 GB/s.\n"
          "  On CPU that pass does not exist, so the case cannot reproduce here -- which is a\n"
          "  statement about the BACKEND, never about the case.")
