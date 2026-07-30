"""At which LEVEL does the blowup first appear -- jaxpr, StableHLO, or optimized HLO?

RECIPE: ``which_level_shows_it(case, control)``.

scopex exposes four levels and they disagree, routinely, by orders of magnitude and sometimes in
SIGN. Reading one of them and rendering a verdict is the single most common way to get a compile
pathology wrong. This recipe reads all of them for both arms in one call and reports the SHAPE of
the ratio curve, because the shape is the diagnosis:

    widening downward   the program is born too big and XLA then multiplies the excess
    flat / inverted     the cost is not module size -- stop counting, go to pass timings
    huge then null      an early pass or CSE erases the pathology before the level you looked at

FOUND ON four arms:

    stackcond_n3000   cpu   jaxpr 6 vs 6 (FLAT) -> StableHLO 3,245 vs 45 text lines (72x) ->
                            optimized HLO 30 vs 31 instructions (0.97x, INVERTED). backend 9.200 s
                            vs 0.114 s = 81x. The pathology is one concatenate with 3,000 operands;
                            CSE folds the 3,000 identical reshapes into one bitcast reused 3,000
                            times, so the optimized module -- the level ``walk_hlo`` reads -- is the
                            ONE level where it is invisible. Max operand arity: 3,000 vs 3.
    ndtri_..._d16     cpu   2.29x -> 3.58x -> 6.11x, monotone: jaxpr 7,885 vs 3,437; StableHLO
                            14,472 vs 4,044 lines; optimized HLO 108,056 vs 17,689 in 3,227 vs 424
                            computations; backend 253.1 s vs 8.40 s = 30.1x.
    jitfib_t24        cpu   jaxpr 139,102 vs 45 = 3,091x, optimized HLO 2 vs 2 = PERFECT NULL, and
                            ``len(str(jaxpr))`` is 3.7x SMALLER in the pathological arm. Any
                            byte-count instrument gives the opposite answer; the printer emits a
                            shared sub-jaxpr once, ``walk`` revisits it (correctly).
    convT64_dilate16  gpu   EVERY count at EVERY level exactly 1.00x against a 33.9x compile. An
                            all-flat cascade is a real result: it says the cost is not in
                            transforming IR (there it was cuDNN autotuning), and it ROUTES you to
                            ``which_pass_ate_the_compile.py``.

RE-MEASURED BY THIS RECIPE, 2026-07-29, jax 0.10.2, cpu, ``stackcond_n3000`` vs control:

    record backend        3.297 s vs  0.056 s   58.8x    (the original run measured 9.200 / 0.114)
    jaxpr_eqns                6 vs      6        1.00x   FLAT, exactly as the case file predicts
    jaxpr_chars            7698 vs    498       15.5x
    stablehlo_ops          3215 vs     15      214.3x    <- walk_stablehlo, now that it works
    stablehlo_lines        3248 vs     48       67.7x
    hlo_opt_instrs           30 vs     31        0.97x   <- INVERTED. the original measured 30/31.
    hlo_opt_computations      7 vs      8        0.88x
    max_operands           3000 vs      3     1000.0x    <- a `concatenate` with 3,000 operands
    loc(" 25 vs loc(# 3214                               <- the exact ratio bug #1 tripped on

Three numbers in that table are byte-identical to the original investigation (30/31, 3000/3, 3214)
and one is new: ``stablehlo_ops`` used to be **1**, on this exact module. That level went from
returning a length-1 iterator to returning 3,215 operations, and nothing about the program changed.

AND IT CARRIES THE SELF-CHECK FOR THE BUG THAT MADE THIS LEVEL SET UNUSABLE.
``walk_stablehlo`` used to return 1 unit for any jax 0.10.2 program -- 1 unit on modules of 3,214
and 21,000 operations, on 16 of 21 real arms. jax emits ALIAS-form locations (``#loc11 = loc(...)``
declared once, ``loc(#loc11)`` on each op) and the parser matched only the inline ``loc("name")``
form: on ``stackcond_n3000``, 23 lines use the inline form and 3,214 use the alias form. The level
did not look broken, it looked EMPTY, and a length-1 iterator passes every ``if not units`` guard.
It is fixed (``walk_stablehlo`` walks ``jaxlib.mlir.ir`` now, not text), and this recipe still
witnesses the operation count against the module text every time it runs, because the next such bug
will also return a plausible number.

WHEN IT WORKS
    Any pathology whose signature is SIZE at some level, and any where the useful information is
    that size is NOT the signature. Costs one lower + one compile per arm; no dump, no flags.

WHEN IT DOES NOT
    * Ratios systematically UNDER-report when compile cost is superlinear in absolute size. ndtri
      from d4 to d16: the structural ratios are essentially flat (2.28 -> 2.29, 3.20 -> 3.58,
      5.98 -> 6.11) while the time ratio goes 4.8x -> 30.1x. Read the absolute counts, not only the
      ratios -- 108k instructions in 3,227 computations is the number that predicts the time.
    * It says nothing about WHERE IN THE PIPELINE, and nothing about time. Pair it with
      ``which_pass_ate_the_compile.py`` and ``where_do_the_arms_diverge.py``.
    * Counting cannot represent an ARITY pathology; ``max_operands`` is included here for exactly
      that reason (``scopex.Ins`` still has no operand-count field, so it comes off
      ``scopex.hlo_instructions``).
"""

from __future__ import annotations

import scopex

import _cases

__all__ = ["which_level_shows_it", "levels_of"]


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


_CHILD = '''
import re
name = {name!r}
fn, args = _cases.load(name)

t = scopex.record(fn, *args)                      # cold compile: trace / lower / backend
low = jax.jit(fn).lower(*args)
comp = low.compile()
jaxpr = jax.make_jaxpr(fn)(*args)

eqns = list(scopex.walk(jaxpr))
distinct = len({{id(j) for e in eqns for _, j in scopex.subjaxprs(e.eqn)}})

sh_text = scopex.stablehlo_text(low)
sh_units = list(scopex.walk_stablehlo(low))

hlo_units = list(scopex.walk_hlo(comp))
recs = list(scopex.hlo_instructions(comp))
max_operands = max((len(r["operands"]) for r in recs), default=0)
arity_holder = max(recs, key=lambda r: len(r["operands"]), default=None)

try:
    temp_bytes = comp.memory_analysis().temp_size_in_bytes
except Exception:
    temp_bytes = None

emit({{
  "record": {{k: t.get(k, 0.0) for k in ("trace", "lower", "backend", "wall")}},
  "regime": scopex.regime(t),
  "jaxpr_eqns": len(eqns),
  "jaxpr_chars": len(str(jaxpr)),
  "jaxpr_distinct_subjaxprs": distinct,
  "stablehlo_ops": len(sh_units),
  "stablehlo_lines": len(sh_text.splitlines()),
  "stablehlo_named": sum(1 for u in sh_units if u.path),
  # The witness for bug #1. `= stablehlo.` counts SSA assignments in the printed module; the IR walk
  # must find at least that many operations (it finds more -- region bodies and func.func have no
  # assignment line). One unit against thousands of assignments is the shape of the bug.
  "stablehlo_assign_lines": len(re.findall(r"= stablehlo\\\\.", sh_text)),
  "loc_inline": sh_text.count('loc("'),
  "loc_alias": sh_text.count("loc(#"),
  "hlo_opt_instrs": len(hlo_units),
  "hlo_opt_computations": len({{u.container for u in hlo_units}}),
  "hlo_opt_fusions": sum(1 for u in hlo_units if u.fusion),
  "max_operands": max_operands,
  "max_operand_opcode": arity_holder["opcode"] if arity_holder else None,
  "temp_bytes": temp_bytes,
}})
'''

# The order matters: this is the descent, and the whole point is whether the ratio widens along it.
LEVELS = ("jaxpr_eqns", "stablehlo_ops", "stablehlo_lines", "hlo_opt_instrs",
          "hlo_opt_computations")


def levels_of(name: str, *, platform: str = "cpu", timeout: int = 3600) -> dict:
    """Every level's unit count for one arm, from one lower + one compile in a fresh process."""
    return _cases.run_in_subprocess(_CHILD.format(name=name), platform=_plat(platform),
                                    timeout=timeout)


def which_level_shows_it(case: str, control: str, *, platform: str = "cpu",
                         timeout: int = 3600) -> dict:
    """Count units at every level for both arms and report the shape of the ratio curve.

    FOUND ON: stackcond_n3000 (cpu), ndtri_scan_jacrev_d16 (cpu), jitfib_t24 (cpu),
    convT64_dilate16 (gpu).
    MEASURED: stackcond jaxpr 6 vs 6, StableHLO text 3,245 vs 45 lines (72x), optimized HLO 30 vs 31
    (0.97x -- inverted), max operand arity 3,000 vs 3, backend 9.200 s vs 0.114 s;
    ndtri d16 2.29x -> 3.58x -> 6.11x monotone; jitfib_t24 jaxpr 3,091x and optimized HLO 1.00x.

    Returns ``arms`` (both raw dicts), ``ratio`` (case/control per level), ``shape``
    (``widening`` / ``flat`` / ``inverted`` / ``collapsing``), ``stablehlo_selfcheck`` and a
    ``verdict`` naming the level to trust and the recipe to go to next.
    """
    arms = {"case": levels_of(case, platform=_plat(platform), timeout=timeout),
            "control": levels_of(control, platform=_plat(platform), timeout=timeout)}

    def r(k):
        a, b = arms["case"].get(k) or 0, arms["control"].get(k) or 0
        return round(a / b, 3) if b else None

    ratio = {k: r(k) for k in LEVELS}
    ratio["jaxpr_chars"] = r("jaxpr_chars")          # the ANTI-SIGNAL; kept so it can be seen lying
    ratio["max_operands"] = r("max_operands")
    ratio["temp_bytes"] = r("temp_bytes")
    ratio["backend_s"] = round(arms["case"]["record"]["backend"]
                               / max(1e-9, arms["control"]["record"]["backend"]), 2)

    seq = [ratio[k] for k in LEVELS if ratio[k] is not None]
    lo, hi = (min(seq), max(seq)) if seq else (1.0, 1.0)
    first, last = (seq[0], seq[-1]) if seq else (1.0, 1.0)
    if hi < 1.15 and lo > 0.85:
        shape = "flat"
    elif last > first * 1.3:
        shape = "widening"
    elif last < 1.0 < hi:
        shape = "collapsing"
    elif last < first * 0.7:
        shape = "narrowing"
    else:
        shape = "mixed"
    if ratio["hlo_opt_instrs"] is not None and ratio["hlo_opt_instrs"] < 1.0 < hi:
        shape = "collapsing"

    check = []
    for label, a in arms.items():
        if a["stablehlo_assign_lines"] and a["stablehlo_ops"] < a["stablehlo_assign_lines"]:
            check.append(f"{label}: walk_stablehlo yielded {a['stablehlo_ops']} operations from a "
                         f"module with {a['stablehlo_assign_lines']} '= stablehlo.' assignment "
                         f"lines. That is the shape of bug #1 -- an under-count that reads as a "
                         f"small program. Run scopex.selftest().")
        if a["stablehlo_ops"] and not a["stablehlo_named"]:
            check.append(f"{label}: {a['stablehlo_ops']} operations, none carrying a name stack. "
                         f"The module was built without debug info.")

    top = max(LEVELS, key=lambda k: (ratio[k] or 0))
    verdict = {
        "flat": (f"FLAT cascade: every level is ~1.0x against a {ratio['backend_s']}x compile. "
                 f"Module size is not the pathology. Go to which_pass_ate_the_compile.py -- on "
                 f"convT that turned a no-signal arm into 'autotuner, 98.8% of the compile'."),
        "widening": (f"WIDENING: {ratio[LEVELS[0]]}x at the jaxpr becomes {ratio[LEVELS[-1]]}x at "
                     f"the optimized HLO. The program is born too big AND XLA multiplies the "
                     f"excess. Use where_do_the_arms_diverge.py to name the multiplying pass."),
        "collapsing": (f"COLLAPSING: the ratio peaks at {top} ({ratio[top]}x) and the "
                       f"optimized HLO "
                       f"is {ratio['hlo_opt_instrs']}x -- a pass erased the pathology before the "
                       f"level walk_hlo reads. Do NOT read the optimized module here. "
                       f"max_operands is {ratio['max_operands']}x: check arity, not counts."),
        "narrowing": (f"NARROWING: the signal is upstream ({top} at {ratio[top]}x) and "
                      f"shrinks below."),
        "mixed": f"MIXED: strongest at {top} ({ratio[top]}x).",
    }[shape]

    return {"case": case, "control": control, "platform": platform,
            "arms": arms, "ratio": ratio, "shape": shape,
            "stablehlo_selfcheck": check or ["ok"], "verdict": verdict}


if __name__ == "__main__":
    CASE, CONTROL = "stackcond_n3000", "stackcond_n3000_control"
    print(f"{CASE}  --  {_cases.note(CASE)}\n")
    r = which_level_shows_it(CASE, CONTROL, platform="cpu")

    rows = [("record backend s", "record"), *((k, k) for k in
            ("jaxpr_eqns", "jaxpr_chars", "jaxpr_distinct_subjaxprs", "stablehlo_ops",
             "stablehlo_lines", "hlo_opt_instrs", "hlo_opt_computations", "hlo_opt_fusions",
             "max_operands", "temp_bytes"))]
    print(f"{'level':<28}{'case':>12}{'control':>12}{'ratio':>10}")
    print("-" * 62)
    for label, k in rows:
        if k == "record":
            a = r["arms"]["case"]["record"]["backend"]
            b = r["arms"]["control"]["record"]["backend"]
            print(f"{label:<28}{a:>12.3f}{b:>12.3f}{r['ratio']['backend_s']:>9}x")
            continue
        a, b = r["arms"]["case"].get(k), r["arms"]["control"].get(k)
        rr = r["ratio"].get(k)
        print(f"{k:<28}{str(a):>12}{str(b):>12}{('' if rr is None else str(rr) + 'x'):>10}")

    print(f"\nmax-operand instruction is a {r['arms']['case']['max_operand_opcode']!r} "
          f"with {r['arms']['case']['max_operands']} operands "
          f"(control: {r['arms']['control']['max_operands']})")
    print(f"regime: case {r['arms']['case']['regime']}, control {r['arms']['control']['regime']}")
    print(f"\nlocation forms in the case's StableHLO text: "
          f"inline loc(\"  {r['arms']['case']['loc_inline']}   "
          f"alias loc(#  {r['arms']['case']['loc_alias']}"
          f"   <- bug #1 matched only the inline form")
    print("stablehlo self-check:", r["stablehlo_selfcheck"])
    print(f"\nshape: {r['shape']}")
    print("VERDICT:", r["verdict"])
