"""RECIPE -- the slow arm has FEWER instructions than the fast one. Is the cost the NUMBER OF
DISTINCT SHAPES you are asking XLA to compile a kernel for?

Every count-based instrument in scopex gets this family's sign WRONG. The pathological arm has
fewer HLO instructions, fewer fusions, an identical jaxpr and identical StableHLO -- and compiles
17x slower. The surviving variable is not how much program there is, it is how many DIFFERENT
shapes it contains, because the GPU autotuner memoises on a shape-keyed cache and a program with N
distinct dot shapes pays N searches for one program's worth of instructions.

    recs = list(scopex.hlo_instructions(scopex.hlo_text(compiled)))
    len({r["shape"] for r in recs if r["opcode"] in ("dot", "fusion")})

NOT ``walk_hlo``. ``scopex.Ins`` has no ``shape`` field -- ``levels._to_ins`` discards the shape
that ``hlo_instructions`` already parsed -- so ``attribute``/``crosstab`` cannot express this view
at all and you must drop to the dict-level accessor. For an autotuning family whose cost is
literally the cardinality of a shape-keyed memo table, that is the one field that matters.

FOUND ON: gemm_shapes_k64 / _k8 (xla#35955), GPU (CUDA, sm_8.9, jax 0.10.2, x64). There is no GEMM
autotuner in the XLA:CPU backend, so THIS CASE DOES NOT EXIST ON CPU -- the device is the axis.

MEASURED (original investigation, at K=8):
    distinct shapes, all opcodes    17  vs  4
    distinct FUSION shapes           9  vs  2         <- the winning number
    optimized HLO instructions     124  vs  128       <- INVERTED: the slow arm is smaller
    autotuner candidate sub-modules in the dump   96 vs 21 = 4.57x, ptx files 18 vs 4
    K ladder (backend seconds)  21.83 / 40.34 / 176.26 (case) vs 5.58 / 6.48 / 10.30 (control)
                                = 3.9x / 6.2x / 17.1x, linear in K at ~2.7 s per distinct shape
    At every rung the counts are held fixed or inverted: jaxpr equations identical (24/48/192),
    StableHLO lines identical (65/105/345), optimized instrs 127 vs 152, 246 vs 256, 1048 vs 1216,
    and the slow arm has FEWER fusions (151 vs 193 at K=64).
    The chain is fully measurable end to end: distinct fusion shapes (9 vs 2) -> autotuner
    candidate sub-modules (96 vs 21) -> autotuner pass seconds (46.20 vs 6.53 at K=16) -> compile.

RE-MEASURED for this recipe (gemm_shapes_k8 vs its control, same GPU, but on an IDLE device --
the original batch ran with ten foreign processes at 100% utilisation):
    distinct dot+fusion shapes    9 vs 2   = 4.5x     <- reproduces exactly
    distinct shapes, all opcodes 24 vs 5   = 4.8x
    optimized HLO instructions  131 vs 136 = 0.96x   <- still INVERTED
    backend seconds           10.074 vs 0.101         = 99.7x
    per-opcode: the case's 9 distinct shapes are {fusion: 9, dot: 1}; the control's 2 are
    {fusion: 2} and it has no dot in the optimized module at all.
The time ratio is 26x larger than the original's 3.9x purely because the device is idle here. The
STRUCTURAL numbers -- which are what this recipe returns -- are unchanged, which is the argument for
counting shapes instead of seconds on a shared machine.

AND ONE THING BOTH EARLIER MEASUREMENTS MISSED, found by running this recipe twice. On GPU the
OPTIMIZED MODULE IS NOT STABLE ACROSS RUNS, because autotuning decides the fusion structure. Two
runs of this identical script, same box, same idle device:
    run 1   case 131 instrs / 9 distinct dot+fusion shapes | control 136 / 2   ratio 4.5x
    run 2   case 131 instrs / 9 distinct dot+fusion shapes | control 152 / 3   ratio 3.0x
The CASE is stable at 9 -- it is set by the program's K distinct shapes, by construction. The
CONTROL moved (its fusion count went 17 -> 25 and it grew 8 `dot` instructions), and with it the
ratio. So quote the case's cardinality as the measurement and the ratio as approximate, and never
compare two GPU optimized modules captured in different processes without re-checking both.

WHEN IT WORKS
    * GPU compiles that are backend-bound with a flat or inverted instruction count, especially when
      ``pass_timings`` puts ``autotuner`` on top. Shape cardinality is the independent variable for
      the whole GEMM/conv autotuning family, and it is measurable BEFORE you spend the compile: the
      distinct-shape count of the pre-optimization module already predicts it.
    * It needs one compile per arm and no dump, no flags, no subprocess.

WHEN IT DOES NOT
    * IT IS NOT A TIMING. It counts a thing that correlates with autotuning cost on this family. It
      cannot tell you the autotuner ran at all -- confirm with ``pass_timings`` (the autotuner IS a
      pass, so it shows up there) or with the sub-module count in a dump.
    * ON CPU IT IS USUALLY A NULL, and a null here means "no autotuner", not "no problem".
    * The shape string is XLA's printed form INCLUDING LAYOUT (``f32[64,64]{1,0}``). Two logically
      identical shapes with different layouts count as two -- which is correct for a kernel cache
      key and wrong if you wanted a logical-shape census. Strip the ``{...}`` for the latter,
      and say which one you did.
    * ``hlo_instructions`` reads the shape out of the instruction's printed form. TUPLE-shaped
      instructions (``sort``, ``while``, ``conditional``, multi-output fusions, custom-calls with a
      scratch output) print a shape containing spaces and used to be dropped entirely; if your
      program is built from those, check the instruction count against the module's assignment lines
      before trusting a cardinality.
    * At K=64 the corroborating pass timer used to DIE on this exact case: XLA printed
      ``HLO pass: autotuner time: 2.1 min`` -- the only ``min`` line among 31,624 pass lines -- and
      a us/ms/s-only parser dropped it, returning a profile whose largest entry was milliseconds.
      Fixed in ``scopex._parse.UNITS``, but it is why the cardinality count, which needs no clock,
      is the knob of record for this family.
"""

from __future__ import annotations

import collections

import scopex

INTERESTING = ("dot", "fusion", "convolution", "custom-call")


def shape_cardinality(recs, opcodes=None) -> dict:
    """``{opcode: number of DISTINCT shapes}`` plus the overall distinct-shape count.

    ``recs`` is ``list(scopex.hlo_instructions(...))``. Kept separate from the comparison below
    because it is the one line most people actually want.
    """
    per: dict[str, set] = collections.defaultdict(set)
    for r in recs:
        if opcodes is None or r["opcode"] in opcodes:
            per[r["opcode"]].add(r["shape"])
    return {"per_opcode": {k: len(v) for k, v in sorted(per.items(), key=lambda kv: -len(kv[1]))},
            "distinct_overall": len({s for v in per.values() for s in v}),
            "n_instructions": sum(1 for r in recs
                                  if opcodes is None or r["opcode"] in opcodes)}


def is_the_cost_shape_cardinality(fn, args, control_fn, control_args, *,
                                  opcodes=("dot", "fusion")) -> dict:
    """One line: does DISTINCT-SHAPE COUNT explain the compile when instruction count does not?

    Compiles both arms (serially, in this process -- no dump and no vmodule are involved, so no
    subprocess is needed) and returns the two censuses side by side together with the ratio that
    matters: cardinality ratio against instruction-count ratio. When the second is <= 1 and the
    first is large, you have this family.
    """
    import jax

    out = {}
    for label, f, a in (("case", fn, args), ("control", control_fn, control_args)):
        # Two compile CALLS per arm: `record` reports the stage split and throws the executable
        # away, and the census needs the executable. Measured cost of the second call: near zero --
        # jax's in-process cache serves it (12.8 s of wall for a pair whose recorded backends were
        # 10.07 s and 0.10 s). It is not free if you have disabled that cache.
        t = scopex.record(f, *a)
        comp = jax.jit(f).lower(*a).compile()
        recs = list(scopex.hlo_instructions(scopex.hlo_text(comp)))
        out[label] = {
            "backend_s": round(t.get("backend", 0.0), 3),
            "regime": scopex.regime(t),
            "all_instructions": len(recs),
            "kinds": dict(collections.Counter(r["opcode"] for r in recs).most_common(6)),
            "selected": shape_cardinality(recs, opcodes),
            "all_shapes": shape_cardinality(recs)["distinct_overall"],
            "custom_calls": dict(scopex.custom_calls(comp)),
        }

    a, b = out["case"], out["control"]
    card_ratio = a["selected"]["distinct_overall"] / max(1, b["selected"]["distinct_overall"])
    instr_ratio = a["all_instructions"] / max(1, b["all_instructions"])
    time_ratio = a["backend_s"] / max(1e-9, b["backend_s"])
    out["ratios"] = {
        "backend": round(time_ratio, 2),
        "instructions": round(instr_ratio, 2),
        f"distinct_{'+'.join(opcodes)}_shapes": round(card_ratio, 2),
        "all_distinct_shapes": round(a["all_shapes"] / max(1, b["all_shapes"]), 2),
    }
    inverted = instr_ratio <= 1.0
    out["instruction_count_is_inverted"] = inverted
    if card_ratio > 1.5 and instr_ratio < card_ratio:
        out["verdict"] = (
            f"cardinality, not size: {a['selected']['distinct_overall']} distinct "
            f"{'/'.join(opcodes)} shapes vs {b['selected']['distinct_overall']} "
            f"({card_ratio:.1f}x) "
            f"while instructions are {instr_ratio:.2f}x"
            + (" -- INVERTED, the slow arm is the smaller program" if inverted else "")
            + f", and backend is {time_ratio:.1f}x")
        out["next"] = ("confirm the autotuner is the consumer: scopex.pass_timings should put "
                       "`autotuner` on top, and a dump should show one XLA sub-module per "
                       "candidate compile. Fix by reducing shape diversity (pad/bucket), not by "
                       "reducing op count.")
    else:
        out["verdict"] = (f"shape cardinality does NOT explain this: {card_ratio:.1f}x distinct "
                          f"shapes against {instr_ratio:.2f}x instructions and {time_ratio:.1f}x "
                          f"backend. Look elsewhere.")
        out["next"] = "pass_timings_coverage.py, then phase_timeline.py"
    return out


if __name__ == "__main__":
    import os

    os.environ.setdefault("JAX_ENABLE_X64", "1")
    import _cases

    busy, desc = _cases.gpu_busy()
    print(f"GPU: {desc}")
    if busy:
        raise SystemExit(
            "another process is on the GPU. This case has no CPU counterpart (there is no GEMM "
            "autotuner in XLA:CPU), and absolute seconds measured against a contended device are "
            "not numbers. Re-run when it is free.")

    NAME = "gemm_shapes_k8"          # the cheapest rung; K=64 is ~3 minutes per arm
    fn, args = _cases.load(NAME)
    cfn, cargs = _cases.load(NAME + "_control")
    r = is_the_cost_shape_cardinality(fn, args, cfn, cargs)

    print(f"\n=== {NAME} vs {NAME}_control (GPU) " + "=" * 28)
    for arm in ("case", "control"):
        d = r[arm]
        print(f"  {arm:7s} backend {d['backend_s']:>7.3f} s ({d['regime']})   "
              f"{d['all_instructions']} instructions   {d['all_shapes']} distinct shapes")
        print(f"          dot+fusion: {d['selected']['n_instructions']} instrs, "
              f"{d['selected']['distinct_overall']} distinct shapes {d['selected']['per_opcode']}")
        print(f"          opcodes {d['kinds']}")
    print(f"\n  RATIOS   {r['ratios']}")
    print(f"  inverted instrs: {r['instruction_count_is_inverted']}")
    print(f"  VERDICT  {r['verdict']}")
    print(f"  next     {r['next']}")
