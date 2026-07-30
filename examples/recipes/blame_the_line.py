"""RECIPE -- the jaxpr is N times too big. WHICH LINE OF PYTHON emitted the equations?

    scopex.attribute(list(scopex.walk(jaxpr)), "site")

and print it with `scopex.table`. `site` is `file:line` resolved from the python traceback jax
attached to each equation, with jax's and scopex's own frames filtered out, and it is NEVER
silently inherited from a caller -- an equation jax could not place lands in an honest
`<no-frame>` bucket instead of being credited to whoever called it.

FOUND ON: arity_tree_25/_50/_100 (jax#4667 -- a pytree with N leaves crossing a jit boundary), CPU,
jax 0.10.2, x64.

MEASURED (NLEAVES=25):
    jaxpr equations   4,600 vs 183
    attribute(units, "site")
        case_lowering_arity_pytree.py:124   4,450 of 4,600  = 96.7%   <- the jax.tree.map line
        control's top site is :137 with 178 of 183.
    The equation count is 184 x NLEAVES exactly (4,600 / 9,200 / 18,400 / 36,800 at 25 / 50 / 100 /
    200) against a control flat at 183, so once the line is named the multiplier is obvious.

MEASURED (re-run for this recipe, arity_tree_25, JAX_PLATFORMS=cpu, x64) -- EXACT reproduction,
plus one thing the original investigation could not do:
    jaxpr        4,600 vs 183 equations
    site (case)  case_lowering_arity_pytree.py:124   4,450   96.7%
                 case_lowering_arity_pytree.py:125     150    3.3%   <no-frame>: 0
    site (ctrl)  case_lowering_arity_pytree.py:137     178   97.3%
    line 124 is `state = jax.tree.map(lambda a, b: 0.999 * a + 1e-3 * b, state, params)`;
    line 137 is the control's `state = 0.999 * state + 1e-3 * params` on one stacked array.
    contract views placed 0 of 4,600 units (author/library/package/role/split all empty).
    OPTIMIZED HLO, which the original write-up said carried no provenance at all:
                 4,726 instructions, 3,775 of them resolve to the SAME line :124
                 (control: 189 instructions, 151 resolve to :137)

WHEN IT WORKS
    After `level_census.py` says BORN BIG. It needs no compile, no dump and no flags -- one
    `make_jaxpr` -- and on a case like this one line of the table is the whole answer.
    It also works on the OPTIMIZED module: `scopex.attribute(scopex.walk_hlo(compiled), "site")`
    resolves XLA's stack-frame tables back to the same python lines. That route was written up as
    impossible -- "the optimized module carries no source location at all" -- which was bug #2: the
    metadata parser accepted only QUOTED values, and `stack_frame_id=5` is an unquoted int. It has
    been fixed, and this recipe reports both levels so the claim can be checked rather than
    believed.

WHEN IT DOES NOT
    * ONLY `site` WORKS ON UNMARKED CODE. On this case `author`, `innermost_author`, `library`,
      `package`, `role` and `split` return `<none>`/`<unmarked>` for all 18,400 equations, because
      nothing in the corpus calls `jax.named_scope` with a contract-shaped string. The attribution
      views in `scopex.BY` are a contract a framework opts into (see `scopex.mark`); they are not a
      fallback. This recipe reports how many units each view could place, so an empty view is
      visible as empty rather than read as "one bucket".
    * A TRACE-BOUND PATHOLOGY HAS NOTHING TO ATTRIBUTE. On einsum_optimal_n10, `site` is
      `<no-frame>` for all ten equations and every other view is a constant across both arms --
      structurally, because `opt_einsum.contract_path` emits no equations at all, it only chooses
      an order. 49 seconds, zero equations to charge it to. Check `stage_split.py` first.
    * The line it names is where the equations were EMITTED, which is not always where the fix is.
      Here `jax.tree.map` is the honest answer and the fix is at the jit boundary.
    * `site` is per-equation. A case whose cost is one equation with 3,000 operands attributes
      100% to one line and still tells you nothing; see widest_instruction.py.
"""

from __future__ import annotations

import collections

import jax

import scopex

CONTRACT_VIEWS = ("author", "innermost_author", "library", "package", "role", "split")


def which_line_wrote_the_program(fn, args, control_fn, control_args, *,
                                 compile: bool = False, top: int = 6) -> dict:
    """One line: which python line emitted the equations that make this jaxpr huge?

    Returns the site census for both arms, the case's excess per site (case count minus the
    control's count at the same site, which is what isolates the pathological line), and a
    coverage report for every contract view so an empty attribution is visible as empty.
    """
    out: dict = {}
    for label, f, a in (("case", fn, args), ("control", control_fn, control_args)):
        units = list(scopex.walk(jax.make_jaxpr(f)(*a)))
        sites = scopex.attribute(units, "site")
        row = {
            "eqns": len(units),
            "sites": dict(sites.most_common(top)),
            "no_frame": sites.get("<no-frame>", 0),
            # How many units each contract view could actually place. A view that placed nothing
            # is not "one bucket", it is a view that does not apply to this code.
            "view_coverage": {
                v: len(units) - scopex.attribute(units, v).get("<none>", 0)
                     - scopex.attribute(units, v).get("<unmarked>", 0)
                for v in CONTRACT_VIEWS},
        }
        if compile:
            c = jax.jit(f).lower(*a).compile()
            hlo = list(scopex.walk_hlo(c))
            hsites = scopex.attribute(hlo, "site")
            row["hlo_opt_instrs"] = len(hlo)
            row["hlo_opt_sites"] = dict(hsites.most_common(top))
            row["hlo_opt_resolved"] = sum(n for s, n in hsites.items()
                                          if s not in ("<no-frame>", "?", ""))
        out[label] = row

    excess = {s: n - out["control"]["sites"].get(s, 0)
              for s, n in out["case"]["sites"].items()}
    excess = dict(sorted(excess.items(), key=lambda kv: -kv[1]))
    top_site, top_n = next(iter(excess.items()))
    share = top_n / max(1, out["case"]["eqns"])

    usable = [v for v, n in out["case"]["view_coverage"].items() if n]
    out["excess_by_site"] = excess
    out["verdict"] = (
        f"{top_site} accounts for {top_n} of the case's {out['case']['eqns']} equations "
        f"({share:.1%}) that the control does not have."
        + (f"  Contract views that placed anything: {usable}."
           if usable else
           "  NO contract view placed a single unit -- this code calls no jax.named_scope, so "
           "`site` is the only attribution available. That is a property of the code, not a "
           "failure of the tool."))
    return out


if __name__ == "__main__":
    import _cases

    CASE, CONTROL = "arity_tree_25", "arity_tree_25_control"
    fn, args = _cases.load(CASE)
    cfn, cargs = _cases.load(CONTROL)
    r = which_line_wrote_the_program(fn, args, cfn, cargs, compile=True)

    for label in ("case", "control"):
        row = r[label]
        print(f"\n=== {label}: {CASE if label == 'case' else CONTROL} "
              f"({row['eqns']} jaxpr equations) " + "=" * 12)
        print(scopex.table(collections.Counter(row["sites"]),
                           total=row["eqns"], label="site"))
        print(f"  <no-frame>: {row['no_frame']}")
        print(f"  contract-view coverage (units placed): {row['view_coverage']}")
        if "hlo_opt_sites" in row:
            print(f"  optimized HLO: {row['hlo_opt_instrs']} instructions, "
                  f"{row['hlo_opt_resolved']} resolved to a source line")
            for s, n in list(row["hlo_opt_sites"].items())[:3]:
                print(f"      {s}  {n}")

    print(f"\n  excess by site (case - control): "
          f"{list(r['excess_by_site'].items())[:3]}")
    print(f"\n  VERDICT {r['verdict']}")
