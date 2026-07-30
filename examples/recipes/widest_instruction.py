"""RECIPE -- the two programs have the SAME number of instructions and one compiles 500x slower.
Where is the size?

In the operand list of a single node. Every count-based view in scopex is blind to it -- two of
them are worse than blind, they are INVERTED and report the pathological arm as the smaller
program. The one view that sees it is a callable you write yourself, which `scopex.attribute`
accepts precisely so you are not limited to the views someone thought of:

    scopex.attribute(list(scopex.walk(jaxpr)), lambda u: len(u.eqn.invars))

The max key of that histogram is the operand arity of the widest instruction in your program.

FOUND ON: stackcond_n3000 / _n10000 / _n30000 (diffrax#606, `jnp.stack` of N arrays inside a
`lax.cond`), CPU, jax 0.10.2, x64.

MEASURED (N=3000 unless stated):
    arity histogram   {2: 2, 1: 3, 3000: 1}  vs  {2: 2, 1: 4}      MAX ARITY 3000 vs 2
    jaxpr equations   6 vs 6                 PERFECT NULL (1.00x at every N)
    walk_hlo units    30 vs 31               NULL *and INVERTED* -- the slow arm is smaller
                      (at N=10000: 19 vs 19 units with byte-identical kind counts, identical
                       fusion-flagged=2 and outlined=19)
    optimized HLO     77 vs 81 lines         INVERTED again
    StableHLO lines   3,245 vs 45  = 72x     (N=10000: 10,713 vs 45 = 238x; linear in N)
    pre-optimization  3,016 vs 16  = 188x    (from the dump; erased by CSE at snapshot 8)
    backend seconds   8.850 vs 0.147 = 60x   (N=10000: 79.96 vs 0.157 = 509x)
    max operand count stays pinned at N in EVERY per-pass snapshot including
    cpu_after_optimizations, versus 3 in the control: the N-operand concatenate survives the whole
    pipeline even though the instruction COUNT collapses to ~22 after CSE.

MEASURED (re-run for this recipe, N=3000, JAX_PLATFORMS=cpu, x64) -- and ONE NUMBER HAS CHANGED:
    arity histogram   {1: 3, 2: 2, 3000: 1}  vs  {1: 4, 2: 2}     MAX ARITY 3000 vs 2   unchanged
    widest node       `stack` with 3000 operands, at case_operand_arity_stack_in_cond.py:82
    jaxpr equations   6 vs 6                                       unchanged null
    walk_hlo units    30 vs 31   (0.968x)                          unchanged, still INVERTED
    optimized lines   80 vs 84   (0.952x)                          unchanged, still INVERTED
    StableHLO lines   3,248 vs 48 = 68x                            unchanged
    walk_stablehlo    3,215 vs 15 units = 214x                     *** WAS 1 vs 1 ***
    The original investigation recorded `scopex.walk_stablehlo` returning ONE unit on both arms and
    concluded that raw StableHLO TEXT LENGTH was "the only scopex-reachable discriminator". That
    was bug #1 -- the location regex matched only the inline `loc("name")` form and missed jax's
    `loc(#loc17)` indirection -- and it has since been fixed. On the fixed instrument the StableHLO
    level is now the loudest structured signal on this case (214x, larger than the 68x from text
    length), so prefer `len(list(scopex.walk_stablehlo(lowered)))` to counting lines. The arity
    knob is unaffected and remains the one that names the mechanism rather than just its size.

WHY THE OTHER VIEWS FAIL, since that is the transferable part
    The single jaxpr `stack` equation lowers to N reshapes plus one N-operand concatenate. CSE then
    collapses 3,021 instructions to 22 because all N reshapes are identical, so from the eighth
    pass onward the two arms sit between 0.81x and 1.05x of each other and every count-based
    instrument agrees they are the same program. The cost is below HLO: XLA:CPU emits one
    straight-line LLVM function of 21,205 lines at N=3000 (strictly 7.1 LLVM instructions per
    operand, exactly linear) and LLVM's opt+ISel+regalloc over a single enormous basic block is
    superlinear -- codegen grows 6.58x for 3x N. The control has no such kernel at all; its largest
    .ll is 78 lines.

WHEN IT WORKS
    Any time two arms have matching equation counts. It costs one `make_jaxpr`, no compile, no
    dump, and it runs before you have spent a minute on a compile you do not need. Reach for it the
    moment a count-based comparison comes back ~1.0x against a large time ratio.

WHEN IT DOES NOT
    * ONLY AT THE JAXPR LEVEL. `scopex.Ins` -- the record for a StableHLO operation or an HLO
      instruction -- has slots for level/kind/path/unit/container/site/loc/function/depth/fusion/
      outlined and NONE for operands or shape, and `scopex.hlo_instructions` yields dicts of
      name/opcode/shape/computation/op_name/source_file/source_line/stack_frame_id with no operand
      list either. So there is no `BY['arity']` and no way to ask this question below the jaxpr
      through the API. The HLO half of the number in MEASURED above came from a hand-written parse
      of `scopex.hlo_text`.
    * It finds WIDTH, not depth or multiplicity. A program that is slow because it has 100 separate
      computations (switch_ident) or 18,400 equations (arity_tree) has an unremarkable arity
      histogram; use level_census.py for those.
    * `len(u.eqn.invars)` reaches through `Eqn.eqn` into the raw jax equation. That is deliberate
      here and it is the abstraction leak the finding is about, not a trick: the number is not
      exposed anywhere else.
    * The StableHLO LINE COUNT is a confirmation, not a primary instrument: it is text length and
      it moves with formatting. Use `walk_stablehlo` now that it works. The original write-up
      called line length "the only scopex-reachable discriminator" on the N=10000 arm; that was
      true of a broken parser, not of the level.
    * Nothing here explains WHY a wide instruction is expensive, only that it is. The cost of this
      one is in LLVM, three levels below the number this recipe returns; phase_timeline.py goes
      there.
"""

from __future__ import annotations

import collections

import jax

import scopex


def how_wide_is_the_widest_instruction(fn, args, control_fn, control_args, *,
                                       include_hlo: bool = True) -> dict:
    """One line: when the instruction counts match, which single node has the enormous fan-in?

    Returns the operand-arity histogram of both arms at the jaxpr level, the max arity, and -- so
    the null is visible next to the signal rather than having to be taken on trust -- the
    count-based views that fail on this case.
    """
    out: dict = {}
    for label, f, a in (("case", fn, args), ("control", control_fn, control_args)):
        jaxpr = jax.make_jaxpr(f)(*a)
        units = list(scopex.walk(jaxpr))

        # THE KNOB. `attribute` takes any callable, which is the whole reason this is answerable.
        arity = scopex.attribute(units, lambda u: len(u.eqn.invars))
        widest = max(units, key=lambda u: len(u.eqn.invars))

        lowered = jax.jit(f).lower(*a)
        row = {
            "eqns": len(units),
            "arity_histogram": dict(sorted(arity.items())),
            "max_arity": max(arity),
            "widest_primitive": str(widest.eqn.primitive),
            "widest_site": widest.site,
            "primitives": dict(collections.Counter(u.kind for u in units).most_common(5)),
            "stablehlo_lines": len(scopex.stablehlo_text(lowered).splitlines()),
            "stablehlo_units": len(list(scopex.walk_stablehlo(lowered))),
            "platform": jax.devices()[0].platform,
        }
        if include_hlo:
            compiled = lowered.compile()
            hlo = list(scopex.walk_hlo(compiled))
            row["hlo_opt_units"] = len(hlo)
            row["hlo_opt_lines"] = len(scopex.hlo_text(compiled).splitlines())
            row["hlo_opt_kinds"] = dict(scopex.attribute(hlo, "kind").most_common(6))
        out[label] = row

    c, k = out["case"], out["control"]

    def ratio(key):
        return round(c[key] / max(1, k[key]), 3)

    out["ratios"] = {key: ratio(key) for key in
                     ("eqns", "max_arity", "stablehlo_lines", "stablehlo_units")
                     + (("hlo_opt_units", "hlo_opt_lines") if include_hlo else ())}
    inverted = [key for key, r in out["ratios"].items() if r < 1.0]
    out["null_views"] = [key for key, r in out["ratios"].items() if 0.9 <= r <= 1.1]
    out["inverted_views"] = inverted
    out["verdict"] = (
        f"widest instruction: {c['widest_primitive']} with {c['max_arity']} operands "
        f"(control: {k['max_arity']}) at {c['widest_site']}, in a jaxpr of {c['eqns']} equations "
        f"against the control's {k['eqns']}."
        + (f"  NOTE these views are INVERTED and report the slow arm as smaller: {inverted}."
           if inverted else ""))
    return out


if __name__ == "__main__":
    import _cases

    N = 3000        # backend ~9 s; N=10000 is 80 s and N=30000 runs past the 900 s harness timeout
    fn, args = _cases.load(f"stackcond_n{N}")
    cfn, cargs = _cases.load(f"stackcond_n{N}_control")

    r = how_wide_is_the_widest_instruction(fn, args, cfn, cargs, include_hlo=True)
    print(f"=== stackcond_n{N} vs _control "
          f"[platform={r['case']['platform']}] " + "=" * 22)
    for label in ("case", "control"):
        row = r[label]
        print(f"\n  {label}")
        for k2 in ("eqns", "max_arity", "widest_primitive", "widest_site", "arity_histogram",
                   "stablehlo_lines", "stablehlo_units", "hlo_opt_units", "hlo_opt_lines"):
            if k2 in row:
                print(f"    {k2:20s} {row[k2]}")
    print(f"\n  ratios (case/control) {r['ratios']}")
    print(f"  null views            {r['null_views']}")
    print(f"  INVERTED views        {r['inverted_views']}")
    print(f"\n  VERDICT {r['verdict']}")
