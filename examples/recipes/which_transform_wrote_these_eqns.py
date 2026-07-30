"""My TRACE stage is slow. Which transform generated all these equations, and at which source line?

RECIPE: ``which_transform_wrote_these_eqns(case, control)``.

Everything else in this directory looks at or below the jaxpr. This one is for the case where the
backend is innocent and the cost is jax's own -- ``scopex.record`` says ``trace-bound`` and every
per-pass and per-level instrument is therefore aimed at the wrong stage. The jaxpr is the only level
that exists yet, and it is also the level where provenance is richest: every equation carries the
TRANSFORM stack that built it and the source line that wrote it.

FOUND ON: ``condrec_grad_512`` / ``_128`` / ``_32``, cpu (jax#8239, grad through inlined lax.cond).

MEASURED
    equations           4,350 vs 478 (9.1x) at max_steps=32
                       21,502 vs 1,918 (11.2x) at 128
                      102,398 vs 7,678 (13.3x) at 512      <- the ratio GROWS with the parameter
    make_jaxpr           116.6 s vs 3.2 s at 512 (36.2x)
    record at 128        trace 15.866 s vs 0.669 s = 23.7x; lower 1.648 vs 0.457; backend 1.802 vs
                         0.662. regime 'trace-bound' vs 'mixed'.
    transform census     {jvp: 96,766, transpose/jvp: 5,632} vs {<primal>: 7,678} at 512
    site census          66,049 of 102,398 equations at case_ad_transpose_cond_recursion.py:139,
                         which is `return lax.cond(cond_fun(val), go, lambda v: v, val)` --
                         the recursion construct, not the MLP payload the control's top sites are.

RE-MEASURED BY THIS RECIPE, 2026-07-29, jax 0.10.2, cpu, at max_steps=128:

    equations       21,502 vs 1,918   11.21x   (identical to the original 21,502 / 1,918)
    make_jaxpr        7.51 s vs 0.40 s 18.7x
    record trace     12.098 s vs 0.558 s 21.7x ; lower 4.17x ; backend 2.96x
    regime          trace-bound vs mixed       (identical to the original)
    transform       {jvp: 20,094, transpose/jvp: 1,408}  vs  {<primal>: 1,918}
                    jvp/transpose = 14.3x -- the same asymmetry the 512 rung shows at 17.2x
    top site        case_ad_transpose_cond_recursion.py:139, 12,417 eqns (58%)  <- the lax.cond
    control sites   lines 160 and 161 at 384 each -- the two `jnp.tanh(x @ w + b)` layers
    <no-frame>      19% of equations, honestly unattributable

THE TRANSFORM CENSUS REFUTED THE ISSUE TITLE, which is why it is the winning knob and not the
equation count. jax#8239 is filed as an AD-TRANSPOSE problem. The equations are 96,766 ``jvp``
against 5,632 ``transpose/jvp``: the LINEARISATION side is 17x the transpose side. What explodes is
the forward JVP of 2*max_steps-1 inlined ``lax.cond`` levels, not the cotangent pass. One
``attribute`` call separates those, and nothing else in the toolkit does.

XLA IS NOT THE AMPLIFIER HERE AND IN FACT UNDERSTATES THE PROBLEM: the HLO ratio is 2.11x at
snapshot 0 and stays between 1.92x and 2.08x for all 34 pass snapshots (final 638 vs 313). 2x at HLO
for an 11x jaxpr and a 24x trace. So a compile-time tool that starts at HLO scores this case "no".

WHEN IT WORKS
    ``record`` says trace-bound or lower-bound, or the jaxpr equation count is the ratio that
    tracks the pathology parameter. Costs one ``make_jaxpr`` per arm and no compile at all if you
    pass ``with_record=False`` -- which matters, because at max_steps=512 the trace alone is 116 s.

WHEN IT DOES NOT
    * The CONTRACT views (``author``, ``library``, ``package``, ``role``, ``split``) return
      ``<none>`` / ``<unmarked>`` for every equation of an unmarked program, and every corpus case
      is unmarked. Only ``transform`` and ``site`` carry information here. That is not a defect --
      it is the contract in ``scopex.mark`` doing exactly what it says: a framework that wants its
      users' code distinguishable has to call ``jax.named_scope`` with the documented strings. This
      recipe reports the marked share so the emptiness is never read as "nothing to attribute".
    * ``site`` has an honest unattributable bucket: 16,380 of 102,398 equations (16%) resolve to
      ``<no-frame>`` on condrec_grad_512. jax's own ``source_locations`` keeps the same bucket. A
      sixth of the blowup genuinely has no python frame; do not paper over it.
    * ``trace_time_via_einsum``-style cases are invisible to this: ``opt_einsum.contract_path``
      burns 49 s choosing a contraction order and emits NO equations, so a jaxpr-unit attributor has
      nothing to attribute. If ``record`` says trace-bound and the equation count is FLAT, the cost
      is in python before a jaxpr exists -- reach for ``cProfile``, not for scopex.
"""

from __future__ import annotations

import scopex

import _cases

__all__ = ["which_transform_wrote_these_eqns", "census_of", "eqn_ratio_sweep"]


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
name, with_record = {name!r}, {with_record!r}
fn, args = _cases.load(name)

import time
t0 = time.perf_counter()
jaxpr = jax.make_jaxpr(fn)(*args)
make_jaxpr_s = time.perf_counter() - t0

units = list(scopex.walk(jaxpr))
ours, theirs, equal = scopex.verify_parity(jaxpr)   # walk() vs jaxpr_util.all_eqns

def top(by, n=8):
    return scopex.attribute(units, by).most_common(n)

rec, reg = None, None
if with_record:
    t = scopex.record(fn, *args)
    rec = {{k: t.get(k, 0.0) for k in ("trace", "lower", "backend", "wall")}}
    reg = scopex.regime(t)

emit({{
  "n_eqns": len(units),
  "make_jaxpr_s": make_jaxpr_s,
  "parity": [ours, theirs, equal],
  "transform": top("transform"),
  "site": top("site"),
  "kind": top("kind"),
  "split": top("split"),          # <unmarked> unless the program calls jax.named_scope
  "no_frame": scopex.attribute(units, "site").get("<no-frame>", 0),
  "crosstab": {{k: dict(v.most_common(3))
                for k, v in scopex.crosstab(units, rows="transform", cols="site").items()}},
  "record": rec, "regime": reg,
}})
'''


def census_of(name: str, *, platform: str = "cpu", with_record: bool = True,
              timeout: int = 3600) -> dict:
    """Trace one arm in a fresh process and census its equations by transform, site and kind."""
    return _cases.run_in_subprocess(_CHILD.format(name=name, with_record=with_record),
                                    platform=_plat(platform), timeout=timeout)


def which_transform_wrote_these_eqns(case: str, control: str, *, platform: str = "cpu",
                                     with_record: bool = True, timeout: int = 3600) -> dict:
    """Census both arms' equations by transform and by source line; report the AD asymmetry.

    FOUND ON: condrec_grad_32 / _128 / _512, cpu.
    MEASURED: at 512, 102,398 vs 7,678 equations (13.3x) with
    ``{'jvp': 96766, 'transpose/jvp': 5632}`` against ``{'<primal>': 7678}`` -- a 17.2x
    linearisation/transpose asymmetry that reverses the reading the issue title invites -- and
    66,049 equations on the single ``lax.cond`` line that builds the recursion.

    Returns ``arms``, ``ratio`` (equations, make_jaxpr seconds, and the record stages),
    ``ad_asymmetry`` (jvp equations over transpose equations), ``attribution_health``
    (unattributable and unmarked shares) and a ``verdict``.
    """
    arms = {"case": census_of(case, platform=_plat(platform), with_record=with_record,
                              timeout=timeout),
            "control": census_of(control, platform=_plat(platform), with_record=with_record,
                                 timeout=timeout)}
    a, b = arms["case"], arms["control"]

    def tsum(arm, pred):
        return sum(n for k, n in arm["transform"] if pred(k))

    jvp = tsum(a, lambda k: k == "jvp")
    trn = tsum(a, lambda k: "transpose" in k)
    ratio = {
        "eqns": round(a["n_eqns"] / max(1, b["n_eqns"]), 2),
        "make_jaxpr_s": round(a["make_jaxpr_s"] / max(1e-9, b["make_jaxpr_s"]), 2),
    }
    if a["record"] and b["record"]:
        ratio.update({f"{k}_s": round(a["record"][k] / max(1e-9, b["record"][k]), 2)
                      for k in ("trace", "lower", "backend")})

    health = {
        "no_frame_share": round(a["no_frame"] / max(1, a["n_eqns"]), 4),
        "unmarked_share": round(dict(a["split"]).get("<unmarked>", 0) / max(1, a["n_eqns"]), 4),
        "walk_parity_ok": a["parity"][2],
    }

    top_site = a["site"][0] if a["site"] else (None, 0)
    verdict = (
        f"{a['n_eqns']} equations vs {b['n_eqns']} ({ratio['eqns']}x). "
        + (f"AD asymmetry jvp/transpose = {round(jvp / trn, 1)}x ({jvp} vs {trn}): the cost is the "
           f"FORWARD linearisation, not the cotangent pass. "
           if trn else
           f"No transpose equations at all -- {jvp} jvp. " if jvp else
           "No AD transforms present; the equations are primal. ")
        + f"Heaviest source line: {top_site[0]} with {top_site[1]} equations "
          f"({100 * top_site[1] / max(1, a['n_eqns']):.0f}%).")
    if health["unmarked_share"] > 0.99:
        verdict += (" NOTE: the program is unmarked, so author/library/package/split are empty by "
                    "construction -- only `transform` and `site` carry information.")

    return {"case": case, "control": control, "platform": platform, "arms": arms,
            "ratio": ratio, "ad_asymmetry": (jvp, trn, round(jvp / trn, 2) if trn else None),
            "attribution_health": health, "verdict": verdict}


def eqn_ratio_sweep(pairs, *, platform: str = "cpu", timeout: int = 3600) -> list:
    """``[(case, control, n_case, n_control, ratio), ...]`` -- trace only, no compile.

    One point cannot tell "exponential in the parameter" from "big program". On condrec the ratio
    goes 9.1x -> 11.2x -> 13.3x across max_steps 32 / 128 / 512, which is what makes the equation
    count a signal about the PATHOLOGY rather than about the size.
    """
    out = []
    for c, k in pairs:
        a = census_of(c, platform=_plat(platform), with_record=False, timeout=timeout)
        b = census_of(k, platform=_plat(platform), with_record=False, timeout=timeout)
        out.append((c, k, a["n_eqns"], b["n_eqns"], round(a["n_eqns"] / max(1, b["n_eqns"]), 2)))
    return out


if __name__ == "__main__":
    CASE, CONTROL = "condrec_grad_128", "condrec_grad_128_control"
    print(f"{CASE}  --  {_cases.note(CASE)}\n")
    r = which_transform_wrote_these_eqns(CASE, CONTROL, platform="cpu")
    a, b = r["arms"]["case"], r["arms"]["control"]

    print(f"equations      {a['n_eqns']:>8} vs {b['n_eqns']:>6}   {r['ratio']['eqns']}x")
    print(f"make_jaxpr s   {a['make_jaxpr_s']:>8.2f} vs {b['make_jaxpr_s']:>6.2f}   "
          f"{r['ratio']['make_jaxpr_s']}x")
    if a["record"]:
        for k in ("trace", "lower", "backend"):
            print(f"record {k:<8}{a['record'][k]:>8.3f} vs {b['record'][k]:>6.3f}   "
                  f"{r['ratio'][k + '_s']}x")
        print(f"regime         {a['regime']} vs {b['regime']}")

    print("\nattribute(walk(jaxpr), 'transform')   <- THE WINNING KNOB")
    print(f"   case    : {dict(a['transform'])}")
    print(f"   control : {dict(b['transform'])}")
    print(f"   jvp / transpose = {r['ad_asymmetry'][2]}x  "
          f"({r['ad_asymmetry'][0]} jvp vs {r['ad_asymmetry'][1]} transpose)")

    print("\nattribute(walk(jaxpr), 'site'), top 4")
    for label, arm in (("case", a), ("control", b)):
        print(f"   {label}:")
        for s, n in arm["site"][:4]:
            print(f"      {n:>8}  {s}")

    print("\ncrosstab(transform x site), case:")
    for t, sites in list(a["crosstab"].items())[:4]:
        print(f"   {t:<16} {sites}")

    print("\nattribution health:", r["attribution_health"])
    print("\nVERDICT:", r["verdict"])
