"""Is the compile N COPIES of one computation rather than one big computation?

RECIPE: ``is_the_cost_multiplicity(case, control)``.

Instruction counts answer "how big"; they do not answer "how many of the same thing". A program that
XLA compiles as 1,025 separate computations and a program of the same instruction count compiled as
one computation cost wildly different amounts, because per-computation work (codegen, copy
insertion, kernel emission) is repeated. When the pathology is multiplicity, NAMING THE
MULTIPLICITY IS THE ANSWER -- there is no single guilty instruction to point at.

The knob is one line and it is not the count of units::

    len({i.container for i in scopex.walk_hlo(compiled)})

``Ins.container`` is the enclosing HLO computation, so a set of containers is the computation count.
``scopex.walk_hlo`` walks natively through XLA's object model, which matters here more than
elsewhere: the line-based parser it replaced could not see TUPLE-shaped instructions, i.e. every
``call``, every ``while``, and every ``parameter`` of a control-flow body -- exactly the
instructions that hold a multi-computation program together (measured: 23 of 30 instructions on a
``while_loop`` program, 46 of 53 on a ``scan``).

FOUND ON
    switch_ident_1024   gpu   1,025 computations vs 1, 6,154 optimized instructions vs 4,
                              record backend 256.46 s vs 0.165 s = 1551x. Counts stay exactly
                              LINEAR in the branch count (2.00x per doubling) while time goes
                              4.2x-8.1x, so the count is honest but under-reports.
    bisect_m95          cpu   306 computations vs 202, and the largest fused computation is 3,164
                              instructions vs 627 -- multiplicity AND size at once.

VERIFIED ON CPU AT n=256 BY THIS RECIPE: 514 computations vs 2, 1,548 instructions vs 4, compile
3.1 s vs 0.08 s. Note the CPU shape differs from the GPU shape (CPU emits 2n+2 computations where
GPU emitted n+1); the MULTIPLICITY is the reproducible fact, the exact constant is per-backend.

THE INSTRUMENT THAT LIES HERE, and it is the reason this recipe exists as its own file: at n=1024
``scopex.pass_timings`` returned a non-empty, entirely plausible profile whose largest entry was a
millisecond-scale pass, because the real culprit was
``HLO pass: copy-insertion time: 2.95 min (177154325 us)`` -- 94.0% of the compile, the ONLY one of
299 pass lines in that log using ``min`` units, and dropped silently by a us/ms/s parser. n=512
(copy-insertion at 41.0 s, under the 60 s threshold where XLA switches units) was a clean hit and
n=1024 was a confident wrong answer: the tool appeared to work on the small rung of the ladder and
failed on the big one. The unit table is fixed and ``which_pass_ate_the_compile.py`` guards it, but
the structural count above never needed the fix -- it was honest at every rung.

WHEN IT WORKS
    Any program built from ``lax.switch``, many ``jit`` sub-calls, many pallas kernels, or an
    unrolled control-flow construct. One compile per arm; no dump, no flags, no subprocess tricks.

WHEN IT DOES NOT
    * It cannot tell you WHY the copies exist. Pair it with ``which_transform_wrote_these_eqns.py``
      (jaxpr side) or ``where_do_the_arms_diverge.py`` (which pass created them: on ``jitfib``,
      ``flatten-call-graph`` clones one computation per call site and goes 21 -> 13,530
      computations in one pass).
    * The counts under-report superlinear cost. On switch_ident they are exactly 2.00x per doubling
      against an n^2.8 time curve. Report the ABSOLUTE computation count, not only the ratio.
    * A program that constant-folds away has no computations left to count: ``jitfib`` ends at 2
      instructions in 1 computation in BOTH arms even though ``flatten-call-graph`` built 35,422
      computations mid-pipeline. If the optimized module is tiny and the compile was long, the
      multiplicity was TRANSIENT -- go and count it in the per-pass snapshots instead.
"""

from __future__ import annotations

import collections

import scopex

import _cases

__all__ = ["is_the_cost_multiplicity", "multiplicity_of"]


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
import collections, time
name = {name!r}
fn, args = _cases.load(name)

t = scopex.record(fn, *args)
low = jax.jit(fn).lower(*args)
t0 = time.perf_counter(); comp = low.compile(); compile_s = time.perf_counter() - t0

units = list(scopex.walk_hlo(comp))
per = collections.Counter(u.container for u in units)

# ARE THE COPIES THE SAME COMPUTATION? A signature of the sorted opcode multiset is crude and
# sufficient: N computations collapsing to ONE signature means XLA is not deduplicating identical
# bodies, which is the mechanism jax#4453 is about. (The corpus tests it directly:
# `switch_distinct_512` has genuinely different branch bodies and costs the same as the identical
# ones.) Cheap: no text, no shapes, straight off the units walk_hlo already produced.
sig = collections.defaultdict(collections.Counter)
for u in units:
    sig[u.container][u.kind] += 1
sigs = collections.Counter(tuple(sorted(c.items())) for c in sig.values())

biggest = per.most_common(1)[0] if per else ("<none>", 0)
emit({{
  "record": {{k: t.get(k, 0.0) for k in ("trace", "lower", "backend", "wall")}},
  "regime": scopex.regime(t),
  "compile_s": compile_s,
  "n_instrs": len(units),
  "n_computations": len(per),
  "n_fusions": sum(1 for u in units if u.fusion),
  "biggest_computation": [biggest[0], biggest[1]],
  "median_computation": sorted(per.values())[len(per) // 2] if per else 0,
  "distinct_signatures": len(sigs),
  "most_repeated_signature_count": sigs.most_common(1)[0][1] if sigs else 0,
  "jaxpr_eqns": sum(1 for _ in scopex.walk(jax.make_jaxpr(fn)(*args))),
}})
'''


def multiplicity_of(name: str, *, platform: str = "cpu", timeout: int = 3600) -> dict:
    """Computation count, size distribution and body-signature census for one arm."""
    return _cases.run_in_subprocess(_CHILD.format(name=name), platform=_plat(platform),
                                    timeout=timeout)


def is_the_cost_multiplicity(case: str, control: str, *, platform: str = "cpu",
                             timeout: int = 3600) -> dict:
    """Decide between "one big computation" and "N copies of a small one", with a control.

    FOUND ON: switch_ident_1024 (gpu), bisect_m95 (cpu).
    MEASURED: switch_ident_1024 -> 1,025 computations vs 1 and 6,154 instructions vs 4, against a
    backend of 256.46 s vs 0.165 s; bisect_m95 -> 306 computations vs 202 with the largest fused
    computation 3,164 instructions vs 627. Verified here on CPU at n=256: 514 vs 2 computations.

    Returns ``arms``, ``ratio``, and a ``verdict`` that distinguishes:
      MULTIPLICITY   computation count grew much faster than mean computation size
      SIZE           computation size grew, count did not
      BOTH           bisect's shape
      NEITHER        both flat -- multiplicity is not the story; go to
                     ``which_pass_ate_the_compile.py``
    """
    arms = {"case": multiplicity_of(case, platform=_plat(platform), timeout=timeout),
            "control": multiplicity_of(control, platform=_plat(platform), timeout=timeout)}
    a, b = arms["case"], arms["control"]

    def rr(k, default=1):
        return round((a.get(k) or 0) / max(default, b.get(k) or 0), 2)

    mean_a = a["n_instrs"] / max(1, a["n_computations"])
    mean_b = b["n_instrs"] / max(1, b["n_computations"])
    ratio = {
        "n_computations": rr("n_computations"),
        "n_instrs": rr("n_instrs"),
        "mean_computation_size": round(mean_a / max(1e-9, mean_b), 2),
        "biggest_computation": round(a["biggest_computation"][1]
                                     / max(1, b["biggest_computation"][1]), 2),
        "jaxpr_eqns": rr("jaxpr_eqns"),
        "backend_s": round(a["record"]["backend"] / max(1e-9, b["record"]["backend"]), 2),
    }
    many = ratio["n_computations"] >= 2.0
    # A RATIO ALONE CANNOT SAY "BIG". On switch_ident_256 the largest computation is 8 instructions
    # against the control's 2 -- a 4.0x ratio on a computation nobody would call large, which made
    # an earlier version of this classifier report "BOTH" for a pure multiplicity case. The absolute
    # floor is what separates bisect_m95's 3,164-instruction fused computation from that.
    big = (a["biggest_computation"][1] >= 64
           and (ratio["mean_computation_size"] >= 2.0 or ratio["biggest_computation"] >= 2.0))
    kind = ("BOTH" if many and big else "MULTIPLICITY" if many
            else "SIZE" if big else "NEITHER")

    verdict = {
        "MULTIPLICITY": (
            f"MULTIPLICITY: {a['n_computations']} HLO computations vs {b['n_computations']} "
            f"({ratio['n_computations']}x) at a mean size of {mean_a:.0f} vs {mean_b:.0f} "
            f"instructions ({ratio['mean_computation_size']}x). "
            f"{a['most_repeated_signature_count']} of them share one body signature "
            f"({a['distinct_signatures']} distinct signatures in all) -- XLA is not deduplicating "
            f"identical bodies. Naming the multiplicity IS the answer; there is no guilty "
            f"instruction."),
        "SIZE": (f"SIZE: {ratio['n_computations']}x computations but the biggest one is "
                 f"{ratio['biggest_computation']}x ({a['biggest_computation'][1]} vs "
                 f"{b['biggest_computation'][1]} instructions). Not a multiplicity story."),
        "BOTH": (f"BOTH: {ratio['n_computations']}x computations AND "
                 f"{ratio['biggest_computation']}x "
                 f"in the largest one -- bisect_m95's shape (306 vs 202 computations, largest "
                 f"3,164 vs 627)."),
        "NEITHER": (f"NEITHER: computations {ratio['n_computations']}x, largest "
                    f"{ratio['biggest_computation']}x, against a {ratio['backend_s']}x backend. "
                    f"Multiplicity is not the story -- go to which_pass_ate_the_compile.py. "
                    f"Beware the transient case: jitfib ends at 2 instructions in both arms and "
                    f"still built 35,422 computations mid-pipeline."),
    }[kind]

    return {"case": case, "control": control, "platform": platform, "arms": arms,
            "ratio": ratio, "kind": kind, "verdict": verdict}


if __name__ == "__main__":
    CASE, CONTROL = "switch_ident_256", "switch_ident_256_control"
    print(f"{CASE}  --  {_cases.note(CASE)}\n")
    r = is_the_cost_multiplicity(CASE, CONTROL, platform="cpu")
    a, b = r["arms"]["case"], r["arms"]["control"]

    print(f"{'metric':<32}{'case':>12}{'control':>12}{'ratio':>10}")
    print("-" * 66)
    for k in ("jaxpr_eqns", "n_instrs", "n_computations"):
        print(f"{k:<32}{a[k]:>12}{b[k]:>12}{str(r['ratio'][k]) + 'x':>10}")
    print(f"{'mean computation size':<32}"
          f"{a['n_instrs'] / max(1, a['n_computations']):>12.1f}"
          f"{b['n_instrs'] / max(1, b['n_computations']):>12.1f}"
          f"{str(r['ratio']['mean_computation_size']) + 'x':>10}")
    print(f"{'biggest computation':<32}{a['biggest_computation'][1]:>12}"
          f"{b['biggest_computation'][1]:>12}"
          f"{str(r['ratio']['biggest_computation']) + 'x':>10}")
    print(f"{'distinct body signatures':<32}{a['distinct_signatures']:>12}"
          f"{b['distinct_signatures']:>12}")
    print(f"{'computations sharing one body':<32}{a['most_repeated_signature_count']:>12}"
          f"{b['most_repeated_signature_count']:>12}")
    print(f"{'record backend s':<32}{a['record']['backend']:>12.3f}"
          f"{b['record']['backend']:>12.3f}{str(r['ratio']['backend_s']) + 'x':>10}")
    print(f"\nregime: case {a['regime']}, control {b['regime']}")
    print(f"biggest computation is {a['biggest_computation'][0]!r}")
    print(f"\nkind: {r['kind']}")
    print("VERDICT:", r["verdict"])
