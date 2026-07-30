"""RECIPE -- before you believe ANY level: is that level actually reading your program, or is it
returning a plausible empty answer?

Run this first, in a fresh process, before the compile you care about. It is the cheapest thing in
this directory and it is the one that would have prevented three published wrong conclusions.

    scopex.conformance()   replays every parser over a VERBATIM capture of the text it was written
                           for. Needs no jax, so it separates "somebody edited the parser" from
                           "the compiler moved".
    scopex.selftest()      compiles a small MARKED probe, dumps it, and asserts every level and
                           every artifact view comes back NON-EMPTY, that the levels still agree
                           with each other, and that HLO stack frames resolve to the same source
                           lines the jaxpr reports. RAISES by default. Must run BEFORE the first
                           compile in the process, because it dumps.
    then, on YOUR program:  units at each level against an independent count of the same text.

WHY THIS IS A RECIPE AND NOT A TEST. An empty level is indistinguishable from a level with nothing
to report. Every failure in this package's history has been silent and plausible, and every one was
found by USING the tool, never by reading it.

FOUND ON: a cross-cutting defect measured across 21 arms of the corpus (six case families), CPU --
but the cause was jax's MLIR printer, so it was platform-independent.

MEASURED (the defect, before the fix):
    scopex.walk_stablehlo returned EXACTLY 1 UNIT on every one of the 21 arms, including modules of
    3,245 / 10,713 / 21,242 / 31,507 / 42,442 StableHLO lines. 16 of the 21 were over 40 lines, so
    it was not a small-module artefact. `attribute(walk_stablehlo(low), "kind")` read `{'func': 1}`
    -- the entire StableHLO tier of the observation stack was dead, on every case, silently.
    Cause: the parser required an INLINE quoted location on the instruction line, and jax 0.10.2
    emits location ALIASES. On one module: 18 occurrences of `loc("` (the #locN DEFINITIONS plus
    func.func argument annotations) against 83 of `loc(#`, and every operation line ends in
    `loc(#loc10)`. Only the func.func signature line matched. Cost: for the case whose 3,000
    reshape operations are individually present at NO OTHER LEVEL, the tier that could see them was
    the tier returning 1.
    Two sibling failures in the same family: HLO metadata was parsed with a QUOTED-values-only
    pattern, so `stack_frame_id=5` was dropped and the optimized module was written up as carrying
    no source location at all (it does); and the pass-timing parser knew {us, ms, s} but not `min`,
    so the slowest pass in a compile was the one guaranteed to be dropped.

RE-MEASURED for this recipe (CPU, jax 0.10.2):
    conformance()   ok, 17 parsers replayed over frozen captures, parent_offset=1
    selftest()      ok: 11 StableHLO units, 19 HLO units, 19 hlo_instructions, 6 frame tables,
                    19 pass steps, 21 timeline entries, and site_join = 1.0 -- every optimized-HLO
                    site the probe resolves is also a jaxpr site, which is the check that catches a
                    frame table resolving to the WRONG line rather than to none.
    ratio guard on a scan+cond+sort program (16x16):
        jaxpr      13 units   vs 13 printed lines    1.00x   ok
        stablehlo  55 units   vs 37 printed lines    1.49x   ok
        hlo_opt    83 units   vs 83 printed lines    1.00x   ok, 3 sites resolved, join 1.0
    AND THE OLD PARSER, RUN ON THAT SAME MODULE, RETURNS 1 UNIT. 55 against 1: the defect is
    reproducible on demand, which is why the legacy pattern ships with this recipe instead of being
    deleted. All three historical bugs are fixed:
``walk_stablehlo`` walks ``jaxlib.mlir.ir`` natively instead of matching lines, so it now also sees
every region-bearing operation (``while``, ``case``, ``sort``, ``reduce``, ``scatter``) that a
line parser structurally cannot. This recipe ships the OLD pattern next to the new walk and prints
both numbers, so you can see the size of the hole rather than take it on trust.

WHEN IT WORKS
    * After any jax/jaxlib upgrade, and before trusting a level you have not used recently. XLA and
      MLIR change their printed forms without notice and every parser here reads printed forms.
    * The ratio guard at the bottom works on YOUR program, which matters: `selftest` proves the
      parsers work on a small marked probe, and a probe cannot exercise the operation mix that
      breaks them.

WHEN IT DOES NOT
    * ``selftest`` DUMPS, so it must run before the first compile in the process, and it raises if
      the backend is already up. Put it at the top of the script, not next to the failure.
    * A ratio check catches "reads much less than it should". It cannot catch a parser that
      resolves to the WRONG answer at the right cardinality -- that is what ``selftest``'s
      cross-level site join is for, and it is the only check here that can catch it.
    * The independent counts below are text heuristics and are meant to be LOWER BOUNDS. The native
      walks legitimately exceed them: ``walk_stablehlo`` sees operations inside regions that are
      never printed on a line of their own, and ``walk_hlo`` sees tuple-shaped instructions that a
      line pattern drops. Native >= text is healthy; native << text is the alarm.
    * ``len(str(jaxpr))`` is NOT a size instrument and is not used here. The printer emits a shared
      sub-jaxpr once while ``walk`` revisits every call site (correctly), so on one corpus arm the
      pathological program's jaxpr TEXT is 3.7x SMALLER than its control's while its equation count
      is 3,091x larger.
"""

from __future__ import annotations

import re
import warnings

import scopex

# The pattern walk_stablehlo used to use, kept verbatim so the failure is reproducible rather than
# historical. It requires an INLINE quoted location on the instruction's own line.
_LEGACY_STABLEHLO = re.compile(
    r'^\s*(?:%\S+\s*=\s*)?"?(?P<op>[\w.]+)"?.*?loc\("(?P<loc>[^"]*)"')


def legacy_stablehlo_units(text: str) -> int:
    """What the old line-based StableHLO parser would return on this module. Ships with the recipe
    because "the parser is fixed" is a claim, and this is the measurement."""
    return sum(1 for line in text.splitlines() if _LEGACY_STABLEHLO.match(line))


def is_this_level_reading_my_program(fn, args, *, floor: float = 0.5) -> dict:
    """One line: does every level of the stack return a count consistent with the text it read?

    Compiles ``fn(*args)`` once and, for each level, compares the number of units the walker
    yielded against an INDEPENDENT count taken from the printed form. Returns a per-level verdict
    and a list of alarms. Cheap: no dump, no flags, no subprocess.
    """
    import jax

    j = jax.make_jaxpr(fn)(*args)
    low = jax.jit(fn).lower(*args)
    comp = low.compile()

    sh_text = scopex.stablehlo_text(low)
    hlo = scopex.hlo_text(comp)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        jx = list(scopex.walk(j))
        sh = list(scopex.walk_stablehlo(low))
        hl = list(scopex.walk_hlo(comp))
        ins = list(scopex.hlo_instructions(hlo))

    # Independent counts, deliberately naive and deliberately LOWER BOUNDS -- see WHEN IT DOES NOT.
    sh_lines = sum(1 for ln in sh_text.splitlines() if " = stablehlo." in ln or " = func." in ln
                   or " = mhlo." in ln or " = chlo." in ln)
    hlo_lines = sum(1 for ln in hlo.splitlines()
                    if re.match(r"\s*(ROOT )?%?[\w.\-]+ = ", ln))
    jaxpr_lines = sum(1 for ln in str(j).splitlines() if " = " in ln)

    levels = {
        "jaxpr": {"units": len(jx), "text_floor": jaxpr_lines,
                  "note": "walk revisits shared sub-jaxprs; units >> printed lines is CORRECT"},
        "stablehlo": {"units": len(sh), "text_floor": sh_lines,
                      "legacy_line_parser_would_say": legacy_stablehlo_units(sh_text),
                      "note": "native walk also sees region-internal ops that are never printed "
                              "on their own line"},
        "hlo_opt": {"units": len(hl), "text_floor": hlo_lines,
                    "hlo_instructions": len(ins),
                    "note": "native walk also sees tuple-shaped instructions a line pattern drops"},
    }
    alarms = []
    for name, d in levels.items():
        floor_n = max(1, d["text_floor"])
        d["ratio_to_text"] = round(d["units"] / floor_n, 3)
        if d["units"] <= 1 and floor_n > 4:
            d["verdict"] = "BROKEN"
            alarms.append(f"{name}: {d['units']} unit(s) from text containing {floor_n} "
                          f"instruction lines. This is the shape of the walk_stablehlo bug -- an "
                          f"empty level reads as a level with nothing to report.")
        elif d["ratio_to_text"] < floor:
            d["verdict"] = "SUSPECT"
            alarms.append(f"{name}: {d['units']} units against {floor_n} instruction lines "
                          f"({d['ratio_to_text']}x). Below {floor}x the walker is dropping a class "
                          f"of operation; find out which class before using this level.")
        else:
            d["verdict"] = "ok"

    sites = {u.site for u in hl if u.site not in ("<no-frame>", "?", "")}
    jsites = {e.site for e in jx if e.site != "<no-frame>"}
    levels["hlo_opt"]["sites_resolved"] = len(sites)
    levels["hlo_opt"]["site_join_with_jaxpr"] = round(len(sites & jsites) / max(1, len(sites)), 3)
    if hl and not sites:
        alarms.append("hlo_opt: not one instruction resolved to a file:line. stack_frame_id is "
                      "being dropped -- that is the second historical bug, and it is what made "
                      "the optimized module look as though it carried no provenance.")

    return {
        "levels": levels,
        "warnings_raised": [str(w.message).splitlines()[0] for w in caught],
        "alarms": alarms,
        "ok": not alarms,
        "verdict": ("every level returns a count consistent with the text it read"
                    if not alarms else f"{len(alarms)} level(s) look wrong -- do not read them"),
    }


if __name__ == "__main__":
    import os

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")

    # 1. the parsers against their frozen samples -- no jax involved, so a failure here is an edit,
    #    not an upgrade.
    conf = scopex.conformance()
    print(f"conformance      ok={conf['ok']}   {len(conf['counts'])} parsers replayed over frozen "
          f"captures, parent_offset={conf['parent_offset']}")
    for name, n in sorted(conf["counts"].items()):
        print(f"    {name:34s} {n}")
    for f in conf["failures"]:
        print(f"    FAILED: {f}")

    # 2. the whole stack against a real compile. BEFORE any other compile in this process: it dumps.
    st = scopex.selftest(verbose=False, strict=False)
    print(f"\nselftest         ok={st['ok']}")
    for k in ("modules", "pass_steps", "timeline_entries", "stablehlo_units", "hlo_units",
              "hlo_instructions", "frame_tables", "site_join", "codegen"):
        if k in st:
            print(f"    {k:20s} {st[k]}")
    for b in st.get("broken", []):
        print(f"    BROKEN: {b}")

    # 3. the ratio guard on a real program with control flow -- the operation mix a small probe
    #    cannot exercise, and exactly what the old parser could not see.
    import jax.numpy as jnp
    from jax import lax

    def prog(x):
        def body(c, _):
            return lax.cond(jnp.sum(c) > 0, lambda a: jnp.tanh(a), lambda a: a * 2.0, c), None
        y = lax.scan(body, x, xs=None, length=4)[0]
        return jnp.sum(jnp.sort(y, axis=-1) * jnp.arange(y.shape[-1], dtype=y.dtype))

    r = is_this_level_reading_my_program(prog, (jnp.ones((16, 16)),))
    print("\nratio guard on a scan+cond+sort program")
    for name, d in r["levels"].items():
        print(f"    {name:10s} {d['verdict']:8s} units={d['units']:<6d} "
              f"text_floor={d['text_floor']:<6d} ratio={d['ratio_to_text']}")
        if "legacy_line_parser_would_say" in d:
            print(f"               the OLD line-based parser on this same module: "
                  f"{d['legacy_line_parser_would_say']} unit(s)")
    print(f"    hlo sites resolved {r['levels']['hlo_opt']['sites_resolved']}, "
          f"join with jaxpr {r['levels']['hlo_opt']['site_join_with_jaxpr']}")
    for a in r["alarms"]:
        print(f"    ALARM: {a}")
    print(f"    VERDICT  {r['verdict']}")
