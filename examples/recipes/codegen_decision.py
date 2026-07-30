"""RECIPE -- the instruction counts match. Did XLA take a different DECISION, and what did the
backend actually EMIT?

Three censuses that survive when every magnitude view is null, in the order they cost you:

    scopex.custom_calls(compiled)                  which library kernel XLA chose -- free
    scopex.attribute(scopex.walk_hlo(c), "kind")   the opcode histogram -- free
    scopex.codegen_size(dump_dir)                  LLVM IR lines and object bytes -- one dump

`custom_calls` exists because an opcode census CANNOT distinguish two pass decisions: a CUB sort
and a generic bitonic lowering are both opcode `custom-call`, and which one XLA picked was the
entire answer on this case.

FOUND ON:

  argsort_f32_1e6 / _control (xla#35587 -- `jnp.argsort` on float32 vs int32), GPU (RTX 4090
  Laptop sm_8.9), jax 0.10.2, x64. ONE DTYPE TOKEN apart.
      jaxpr             identical: 5 equations, same primitives                     NULL
      optimized HLO     44 vs 12 instructions = 3.67x    real, far too small for 11.8x
      backend           5.534 s vs 0.469 s = 11.8x
      pass_timings      coverage 0.397 / 5.534 = 7.2%, top entry 'autotuner' 0.304 s -- NOT the
                        story; the pass timer is 14x short.
      custom_calls      'xla.gpu.ext.cub_sort_pairs' PRESENT in the control, ABSENT in the case
                        <- THE ANSWER, one line
      pass list         'estimate-cub-sort-scratch-size' runs only in the control
      opcodes           comparator ops (compare 10, select 7, xor 2) only in the case
      codegen           57 PTX kernels vs 3, 34,466 PTX lines vs 110 (313x), ir-with-opt.ll
                        44,890 vs 119 lines, 56 kernel thunks vs 3
      n=1e6 -> 1e7 doubles the kernel count (57 -> 107) while backend moves only 1.28x, which is
      the honest shape of an O(log^2 n) stage count under a fixed per-kernel LLVM cost.
      This is the cleanest case in the corpus for the proposition that a pathology can be
      simultaneously invisible in COUNTS, invisible to the PASS TIMER, and screaming in codegen.

  gather2d_9 / _control (jax#32704 -- chained 2-D fancy indexing), CPU, x64.
      backend           99.67 s vs 0.191 s = 522x
      jaxpr             80 vs 56 = 1.43x, and the ratio does NOT grow with ncycles
      StableHLO         183 vs 145 lines = 1.26x
      optimized HLO     80 vs 77 instructions = 1.04x
      LLVM IR (dump)    358 vs 319 lines = 1.12x, object 1,784 vs 1,632 B = 1.09x, and both arms
                        add exactly +18 IR lines per added link -- IDENTICAL SLOPE
      opcode census     concatenate 9 vs 0, all inside ONE kind=kLoop fusion, and the count equals
                        ncycles in the case and is 0 at every ncycles in the control  <- POSITIVE
      also              gather start_index_map={0,1}, index_vector_dim=1 vs {0}; max operands 5 vs
                        3; buffer allocations 6 vs 4
      Every magnitude knob is null against a 522x cost. The opcode census is the only structural
      view that says anything, and what it says -- a concatenate feeding gather.start_indices, one
      per link -- is the mechanism.

MEASURED (re-run for this recipe, argsort_f32_1e6 vs _control, CUDA, x64) -- EXACT reproduction of
the decision census, on a device carrying one other 246 MiB compute process, so the SECONDS are
upper bounds and the census is not a timing:
    case     custom_calls {}                                <- cub_sort_pairs ABSENT
             46 optimized instructions
             opcodes {parameter 11, compare 10, select 7, constant 5, xor 2, bitcast 2, iota 2,
                      fusion 2}   <- the inlined IEEE-754 comparator, published as
                                     "compare 10, select 7, xor 2"
             compile 4.061 s
    control  custom_calls {'xla.gpu.ext.cub_sort_pairs': 1}
             13 optimized instructions
             opcodes {parameter 4, fusion 2, add 1, reduce 1, slice 1, constant 1, iota 1,
                      get-tuple-element 1}
             compile 0.197 s
    One dtype token; 20.6x compile; one custom-call target present in one arm and not the other.
    The published counts were 44 vs 12 instructions at 5.534 s vs 0.469 s.

MEASURED (re-run for this recipe, gather2d_6 vs _control, JAX_PLATFORMS=cpu, x64):
    hlo_opt_instrs   64 vs 62  = 1.03x        computations 2 vs 2, fusion-flagged 1 vs 1
    custom_calls     {} vs {}  -- empty on both, which is normal on CPU and discriminates nothing
    opcode delta     concatenate (6, 0)  <- EXCLUSIVE to the case, and 6 == ncycles exactly
                     convert (7, 6), select (7, 6), bitcast (13, 12), compare (7, 6),
                     broadcast (2, 3), multiply (0, 1)
    codegen_size     ir_no_opt 154 vs 147 lines, ir_with_opt 152 vs 150, obj 1,888 vs 1,784 bytes
                     -- 1.05x, a NULL, exactly as at ncycles 8 and 9
    The published ncycles=9 reading (concatenate 9 vs 0) is the same law one rung up: the
    concatenate count equals ncycles in the case and is 0 at every ncycles in the control.

WHEN IT WORKS
    When `level_census.py` comes back flat and `pass_timings_coverage.py` comes back with low
    coverage. Those two nulls together mean the difference is a DECISION, not a size, and this is
    where decisions are visible.

WHEN IT DOES NOT
    * `custom_calls` is empty on most CPU programs, and an empty census is not evidence of
      anything. It discriminates when one arm calls a vendor library and the other does not.
    * A DECISION CENSUS TELLS YOU WHICH BRANCH WAS TAKEN, NOT WHY IT WAS TAKEN. The reason the
      generic bitonic lowering ran at all -- SortRewriter rejecting an IEEE-754 comparator -- is
      in XLA's source, not in any artifact. On gather2d the corresponding veto,
      'Tiled emitter failed due to tiling failure: UNIMPLEMENTED ... falling back to loop emitter',
      exists ONLY in a vmodule stderr stream and never in the dump directory.
    * `scopex.codegen_size` counts `.ll` and `.o` ONLY. On GPU the interesting artifacts are the
      `.ptx` files and `thunk_sequence.txt`, and it does not look at them; the PTX numbers above
      were counted by hand and this recipe carries a small local counter for them. Treat that as a
      known gap, not as "the GPU emitted nothing".
    * On gather2d the codegen size is a NULL (1.12x) against a 522x compile, because the emitter is
      exponential while its OUTPUT is linear. A small artifact does not mean a cheap compile. That
      is what phase_timeline.py is for.
    * THIS FUNCTION COMPILES IN PROCESS, so it censuses whatever backend jax picked -- and the
      census is backend-specific. Caught live while writing this recipe: the identical
      `gather2d_6 vs _control` call returns 64 vs 62 instructions with `concatenate (6, 0)` under
      `JAX_PLATFORMS=cpu`, and a completely different opcode delta (`parameter (21, 6)`,
      `fusion (6, 1)`) when the same script is run with the GPU visible. Both are correct; they
      are different programs. The returned dict carries `platform` for that reason -- never quote
      an opcode census without it.
"""

from __future__ import annotations

import collections
import os
import pathlib
import tempfile

import jax

import scopex


def ptx_census(dump_dir) -> dict:
    """`.ptx` file and line counts. Not in `scopex.codegen_size`, which reads `.ll` and `.o` only.

    Also counts kernel thunks, which is what actually gets launched.
    """
    files = [pathlib.Path(dump_dir) / f for f in os.listdir(dump_dir) if f.endswith(".ptx")]
    lines = sum(sum(1 for _ in p.open(errors="replace")) for p in files)
    thunks = 0
    for f in os.listdir(dump_dir):
        if "thunk_sequence" in f:
            t = (pathlib.Path(dump_dir) / f).read_text(errors="replace")
            thunks += t.count("kKernel") or t.lower().count("kernel")
    return {"ptx_files": len(files), "ptx_lines": lines, "kernel_thunks": thunks}


def what_did_the_backend_decide(fn, args, control_fn, control_args, *,
                                dump_dirs: tuple | None = None) -> dict:
    """One line: same-sized modules -- did XLA choose a different kernel or a different lowering?

    Returns the custom-call census, the opcode census and the opcode DELTA for both arms. Pass
    ``dump_dirs=(case_dir, control_dir)`` from a previous `scopex.dump` to add codegen sizes; this
    function will not dump for you, because `scopex.dump` must precede the first compile in the
    process and this one compiles.
    """
    out: dict = {}
    for label, f, a in (("case", fn, args), ("control", control_fn, control_args)):
        c = jax.jit(f).lower(*a).compile()
        hlo = list(scopex.walk_hlo(c))
        out[label] = {
            "custom_calls": dict(scopex.custom_calls(c)),
            "opcodes": dict(scopex.attribute(hlo, "kind").most_common()),
            "hlo_opt_instrs": len(hlo),
            "computations": len(scopex.hlo_module(c).computations()),
            "fusion_flagged": sum(1 for i in hlo if i.fusion),
            "platform": jax.devices()[0].platform,
        }

    ca, cb = out["case"]["custom_calls"], out["control"]["custom_calls"]
    only_case = {k: v for k, v in ca.items() if k not in cb}
    only_ctrl = {k: v for k, v in cb.items() if k not in ca}

    oa, ob = out["case"]["opcodes"], out["control"]["opcodes"]
    delta = {k: (oa.get(k, 0), ob.get(k, 0)) for k in set(oa) | set(ob)
             if oa.get(k, 0) != ob.get(k, 0)}
    delta = dict(sorted(delta.items(), key=lambda kv: -(kv[1][0] - kv[1][1])))
    exclusive = {k: v for k, v in delta.items() if v[1] == 0 or v[0] == 0}

    if dump_dirs:
        for label, d in zip(("case", "control"), dump_dirs):
            out[label]["codegen_size"] = {k: v for k, v in scopex.codegen_size(d).items()
                                          if k != "files"}
            out[label]["ptx"] = ptx_census(d)

    if only_case or only_ctrl:
        verdict = (f"DIFFERENT KERNEL CHOSEN. custom_call targets only in the case: "
                   f"{only_case or '{}'}; only in the control: {only_ctrl or '{}'}. That is a pass "
                   f"DECISION and no opcode census can see it -- both spellings are opcode "
                   f"'custom-call'.")
    elif exclusive:
        k, (na, nb) = next(iter(exclusive.items()))
        verdict = (f"DIFFERENT LOWERING. Opcode {k!r} appears {na} times in the case and {nb} in "
                   f"the control. Opcodes present in exactly one arm: "
                   f"{ {k2: v for k2, v in list(exclusive.items())[:5]} }")
    else:
        verdict = ("No decision visible here: the two arms use the same custom-call targets and "
                   "the same opcodes. The difference is elsewhere -- phase_timeline.py next.")

    return {**out, "custom_calls_only_in_case": only_case,
            "custom_calls_only_in_control": only_ctrl,
            "opcode_delta": delta, "opcodes_exclusive_to_one_arm": exclusive,
            "verdict": verdict}


if __name__ == "__main__":
    import _cases

    # ncycles=6 rather than 9: backend is 1.4 s instead of 99.7 s and the concatenate count still
    # equals ncycles exactly, which is the whole signal. The published numbers above are ncycles=9.
    CASE, CONTROL = "gather2d_6", "gather2d_6_control"
    fn, args = _cases.load(CASE)
    cfn, cargs = _cases.load(CONTROL)

    # Dumps first, in their own processes: scopex.dump raises once the backend is up.
    dirs = []
    for name in (CASE, CONTROL):
        d = tempfile.mkdtemp(prefix=f"scopex-recipe-{name}-")
        _cases.run_in_subprocess(
            f'fn, args = _cases.load({name!r})\n'
            f'import jax\n'
            f'with scopex.dump({d!r}, passes=None, fusion=False, keep=True) as dd:\n'
            f'    jax.jit(fn).lower(*args).compile()\n'
            f'emit({{"dir": dd}})\n', platform="cpu")
        dirs.append(d)

    r = what_did_the_backend_decide(fn, args, cfn, cargs, dump_dirs=tuple(dirs))

    print(f"=== {CASE} vs {CONTROL} "
          f"[platform={r['case']['platform']}] " + "=" * 20)
    if r["case"]["platform"] != "cpu":
        print("  NOTE: not running on cpu. The published gather2d numbers are CPU numbers and the\n"
              "  opcode census differs by backend -- set JAX_PLATFORMS=cpu to compare with them.")
    for label in ("case", "control"):
        row = r[label]
        print(f"\n  {label}")
        print(f"    hlo_opt_instrs   {row['hlo_opt_instrs']}   computations "
              f"{row['computations']}   fusion-flagged {row['fusion_flagged']}")
        print(f"    custom_calls     {row['custom_calls'] or '{} (none -- normal on CPU)'}")
        print(f"    opcodes          {dict(list(row['opcodes'].items())[:8])}")
        print(f"    codegen_size     {row.get('codegen_size')}")
        print(f"    ptx              {row.get('ptx')}")
    print(f"\n  opcode delta (case, control)  "
          f"{ {k: v for k, v in list(r['opcode_delta'].items())[:6]} }")
    print(f"  exclusive to one arm          {r['opcodes_exclusive_to_one_arm']}")
    print(f"\n  VERDICT {r['verdict']}")

    # ── the GPU arm, where `custom_calls` is the whole answer ───────────────────────────────────
    # Run as a structural census only. Which custom-call target XLA chose is not a timing, so a
    # busy device does not change it -- but the seconds it prints alongside are upper bounds and
    # must not be quoted as measurements.
    busy, desc = _cases.gpu_busy()
    print(f"\n=== argsort_f32_1e6 vs _control (cuda) " + "=" * 22)
    print(f"  nvidia-smi: {desc}"
          + ("  -- CONTENDED: seconds below are UPPER BOUNDS" if busy else ""))
    try:
        g = _cases.run_in_subprocess(
            'import jax, time\n'
            'out = {}\n'
            'for name in ("argsort_f32_1e6", "argsort_f32_1e6_control"):\n'
            '    fn, args = _cases.load(name)\n'
            '    t0 = time.perf_counter(); c = jax.jit(fn).lower(*args).compile()\n'
            '    t1 = time.perf_counter()\n'
            '    hlo = list(scopex.walk_hlo(c))\n'
            '    out[name] = {"custom_calls": dict(scopex.custom_calls(c)),\n'
            '                 "instrs": len(hlo),\n'
            '                 "opcodes": dict(scopex.attribute(hlo, "kind").most_common(8)),\n'
            '                 "compile_s": round(t1 - t0, 3),\n'
            '                 "platform": jax.devices()[0].platform}\n'
            'emit(out)\n', platform="cuda", timeout=600)
        for name, row in g.items():
            print(f"  {name}")
            print(f"      custom_calls {row['custom_calls'] or '{}  <- ABSENT'}")
            print(f"      instrs {row['instrs']}  compile {row['compile_s']} s "
                  f"({row['platform']})")
            print(f"      opcodes {row['opcodes']}")
    except Exception as e:
        print(f"  skipped: {type(e).__name__}: {str(e).splitlines()[0]}")
