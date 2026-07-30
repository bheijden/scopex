"""Which pass blew the module up -- and did the two arms even run the same passes?

RECIPE: ``where_do_the_arms_diverge(case, control)``.

``which_pass_ate_the_compile.py`` answers "where did the SECONDS go". This answers "where did the
INSTRUCTIONS come from", and the two disagree often enough that both are needed: on ``jitfib`` the
expensive pass (``call-inliner``, 3.8 s) is not the pass that created the problem
(``flatten-call-graph``, 0.8 s, which cloned one computation per call site and handed the inliner
27,058 instructions to chew).

FOUND ON three arms:

    jitfib_t20      cpu   snapshot 0003 ``[sharding-removal] after_flatten-call-graph`` goes
                          57 -> 27,058 instructions and 21 -> 13,530 computations IN ONE PASS,
                          AND THAT SNAPSHOT DOES NOT EXIST IN THE CONTROL. (t=16: 45 -> 3,946;
                          t=22: 63 -> 70,842 -- x2.618 per +2 in t, i.e. phi^2.)
    bisect_m95      cpu   both arms track within 1.76x to snapshot 31 (after layout assignment,
                          3,605 vs 2,050). Snapshot 32, after ``fusion``: 382,246 vs 5,507 = 69x.
                          One pass multiplied the m=95 module by 106x and the m=96 module by 2.7x.
    ndtri_..._d16   cpu   born 4.39x too big at snapshot 0000 (22,368 vs 5,100), and ``fusion``
                          then widens 3.69x -> 6.08x (25,509 -> 106,480 vs 6,909 -> 17,508).

RE-MEASURED BY THIS RECIPE, 2026-07-29, jax 0.10.2, cpu, on ``jitfib_t16`` vs its control -- the
whole mechanism in four rows of the returned curve, and the last three are the same passes
``which_pass_ate_the_compile.py`` ranks by seconds:

    flatten-call-graph      45 ->  3,946   x87.7      snapshot ABSENT from the control
    call-inliner         3,946 ->  1,973   x0.5       inlines the clones back in
    constant_folding     1,973 ->      1   x0.0       evaluates the argument-free program
    final module             2 vs      2              a PERFECT NULL at the optimized level

``case_only_caused_by == ['flatten-call-graph']`` -- one pass, named, from a one-element list.

THE OFF-BY-ONE THIS RECIPE EXISTS TO CORRECT, found by running it: ``scopex.diverge`` reports
``diverges_at['pass'] = 'SubbytePacker_pipeline/sub-byte-size-setter'`` on that same run. A snapshot
file is ``after_A.before_B`` and ``PassStep.name`` is built from ``before_pass`` -- the pass that
has NOT RUN YET -- so a jump belongs to the LATER snapshot's ``after_pass``. ``diverge`` also
skips snapshots the control lacks, which walks the answer two more passes downstream, onto a pass
that changed nothing. Every jump this recipe reports is translated through ``after_pass``.

THE THING TO INTERNALISE, because it is not obvious and it is the cleanest localisation available:
XLA WRITES A PER-PASS SNAPSHOT ONLY WHEN THE PASS CHANGED THE MODULE. So a pass name PRESENT in one
arm's snapshot list and ABSENT in the other's is itself the finding -- no counting required. That is
``diverge()["case_only_passes"]``, and on jitfib it is a one-element list.

WHEN IT WORKS
    A control exists, and the pathology shows up as module SIZE at some pass boundary. Three of the
    six pathologies in this slice are exactly that.

WHEN IT DOES NOT
    * ``dusfold_sum_300``: the guilty pass (``constant_folding``) SHRINKS the instruction count
      while materialising a 216 MB literal. The instruction curve points the wrong way. That is why
      this recipe also returns a BYTES curve off the snapshot file sizes -- ``pass_growth`` has no
      bytes metric and the count metric is actively misleading here.
    * ``switch_ident``, ``arity_tree``, ``convT``, gather: the curve is FLAT and identical in both
      arms. Flat is a real answer (the cost is not module growth) but it is not a localisation.
    * ``stackcond_n3000``: the curve COLLAPSES -- 3,016 instructions through pass 7, then 22 after
      ``cse``, because CSE folds 3,000 identical reshapes into one bitcast reused 3,000 times. The
      pathology (one instruction with 3,000 operands) survives; the count does not represent it.
    * The cost may be after the last pass entirely. ``pass_timeline`` is returned for that reason:
      it times the gap between the last HLO snapshot and the emitted ``.ll``/``.o``, which is LLVM,
      and on ``stackcond`` that gap was 16.0 s of a 17.8 s compile.

COST WARNING, MEASURED: ``dump(passes='.*')`` on ``bisect_m95`` wrote 944 MB across 950 files and
took 358 s -- from a 24-equation jaxpr. ``dump()`` warns about none of this. This recipe reports
``dump_bytes`` and ``n_files`` per arm and refuses to proceed past ``max_dump_gb`` without consent.
"""

from __future__ import annotations

import os
import pathlib

import scopex

import _cases

__all__ = ["where_do_the_arms_diverge", "dump_arm"]


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
name = {name!r}
path = {path!r}
fn, args = _cases.load(name)
# scopex.dump RAISES if XLA's backend is already up -- XLA_FLAGS is read at backend init and setting
# it later is a SILENT no-op, which yields an empty directory that reads as "nothing happened".
# Hence one fresh process per arm; there is no in-process route.
with scopex.dump(path=path, passes={passes!r}, fusion={fusion!r}, keep=True) as d:
    import jax, time
    t0 = time.perf_counter()
    jax.jit(fn).lower(*args).compile()
    t1 = time.perf_counter()
import os
n = b = 0
for f in os.listdir(d):
    n += 1
    b += os.path.getsize(os.path.join(d, f))
emit({{"dir": d, "compile_s": t1 - t0, "n_files": n, "dump_bytes": b}})
'''


def dump_arm(name: str, path: str, *, platform: str = "cpu", passes: str = ".*",
             fusion: bool = False, timeout: int = 3600) -> dict:
    """Compile one arm in a fresh process with XLA dumping on. Returns dir, files, bytes, seconds.

    ``fusion=False`` by default: the priority-fusion decision log is a GPU pass, and asking for
    it on CPU makes ``scopex.dump`` warn about an absence that is not a problem. Turn it on for
    GPU work --
    it is free, it is written during a pass that already runs, and it is the only place XLA states
    its fusion vetoes in words (``xtile_issue``: 85 x "not fusing because there are only bitcast
    users" against the control's 1).
    """
    return _cases.run_in_subprocess(
        _CHILD.format(name=name, path=path, passes=passes, fusion=fusion),
        platform=_plat(platform), timeout=timeout)


def where_do_the_arms_diverge(case: str, control: str, *, platform: str = "cpu",
                              workdir: str | None = None, factor: float = 1.5,
                              max_dump_gb: float = 4.0, timeout: int = 3600) -> dict:
    """Walk both per-pass curves in lockstep and name the pass where they separate.

    FOUND ON: jitfib_t16/t20 (cpu), bisect_m95 (cpu), ndtri_scan_jacrev_d16 (cpu).
    MEASURED: jitfib_t20 57 -> 27,058 instructions across ``flatten-call-graph``, a snapshot the
    control does not have at all; bisect_m95 3,605 -> 382,246 across ``fusion`` (106x) against the
    control's 2,050 -> 5,466 (2.7x); ndtri d16 born 4.39x too big at snapshot 0000.

    Returns ``diverge`` (``scopex.diverge``: first pass exceeding ``factor``, whether the pass
    SEQUENCES match, and both curves), ``growth_head`` (the biggest single-step jumps),
    ``bytes_jump`` (same, on snapshot file size -- the metric that works when the count shrinks),
    ``timeline_head`` (``scopex.pass_timeline``, incl. the post-pass LLVM gap), and the dump cost.
    """
    root = pathlib.Path(workdir or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), f"scopex-diverge-{case}"))
    root.mkdir(parents=True, exist_ok=True)
    dirs, meta = {}, {}
    for label, name in (("case", case), ("control", control)):
        d = str(root / label)
        meta[label] = dump_arm(name, d, platform=_plat(platform), timeout=timeout)
        dirs[label] = meta[label]["dir"]
        gb = meta[label]["dump_bytes"] / 2**30
        if gb > max_dump_gb:
            raise RuntimeError(
                f"{label} dump is {gb:.1f} GB in {meta[label]['n_files']} files, over max_dump_gb="
                f"{max_dump_gb}. bisect_m95 produced 944 MB / 950 files / 358 s from a 24-equation "
                f"jaxpr; dump(passes='.*') is not free. Raise max_dump_gb or narrow `passes`.")

    d = scopex.diverge(dirs["case"], dirs["control"], factor=factor)

    steps = {k: scopex.pass_growth(v) for k, v in dirs.items()}
    by_name = {s.name: s for s in steps["case"]}

    # ── THE OFF-BY-ONE THAT NAMES THE WRONG PASS ────────────────────────────────────────────────
    # A snapshot file is `...after_A.before_B.txt` and holds the module BETWEEN A and B, and
    # `PassStep.name` is built from the pipeline and `before_pass` -- i.e. from the pass that has
    # NOT RUN YET. So a jump between consecutive snapshots was caused by the LATER snapshot's
    # `after_pass`, never by its `name`. On jitfib_t16 the jump 45 -> 3,946 sits in the snapshot
    # named `sharding-removal/sharding-remover`, and the pass that did it is `flatten-call-graph`.
    # `scopex.diverge` reports the snapshot NAME in `diverges_at`, and it additionally skips
    # snapshots the control does not have -- so on jitfib it names
    # `SubbytePacker_pipeline/sub-byte-size-setter`, two passes downstream of the culprit and a pass
    # that changed nothing. Always translate through `after_pass`.
    def jumps_for(arm, size):
        out = []
        for i in range(1, len(steps[arm])):
            p, q = steps[arm][i - 1], steps[arm][i]
            a, b = size(p), size(q)
            out.append({"caused_by": q.after_pass, "snapshot_named": q.name,
                        "before": a, "after": b, "ratio": round(b / max(1, a), 2)})
        out.sort(key=lambda j: -j["ratio"])
        # Steps of exactly 1.0 are snapshots where a pass changed the module without changing its
        # size. They are not jumps; keeping them pads the head of the list with nothing.
        return [j for j in out if j["ratio"] != 1.0] or out[:1]

    jumps = jumps_for("case", lambda s: s.instrs)
    # THE BYTES CURVE. pass_growth has no bytes metric, and on dusfold the instruction count SHRINKS
    # across the guilty pass while the module text goes to hundreds of MB. Snapshot file size is a
    # free proxy: the file is the module XLA printed at that boundary.
    bjumps = jumps_for("case", lambda s: os.path.getsize(s.path))

    dv = {k: d[k] for k in ("diverges_at", "pass_sequence_identical", "case_only_passes",
                            "control_only_passes", "case_final", "control_final")}
    if dv["diverges_at"]:
        dv["diverges_at"]["caused_by"] = by_name[dv["diverges_at"]["pass"]].after_pass
    # Same translation for the snapshots that exist in only one arm: the informative name is the
    # pass that WROTE the snapshot, because XLA writes one only when a pass changed the module.
    dv["case_only_caused_by"] = sorted({by_name[n].after_pass for n in dv["case_only_passes"]})

    return {
        "case": case, "control": control, "platform": platform,
        "dirs": dirs,
        "dump_cost": {k: {"n_files": v["n_files"],
                          "dump_mb": round(v["dump_bytes"] / 2**20, 1),
                          "compile_s": round(v["compile_s"], 3)} for k, v in meta.items()},
        "n_snapshots": {k: len(v) for k, v in steps.items()},
        "diverge": dv,
        "growth_head": jumps[:5],
        "bytes_jump": bjumps[:5],
        "timeline_head": sorted(scopex.pass_timeline(dirs["case"]), key=lambda t: -t[1])[:6],
        "case_curve": d["case_curve"],
        "control_curve": d["control_curve"],
    }


if __name__ == "__main__":
    CASE, CONTROL = "jitfib_t16", "jitfib_t16_control"
    print(f"{CASE}  --  {_cases.note(CASE)}\n")
    r = where_do_the_arms_diverge(CASE, CONTROL, platform="cpu")

    print("dump cost:", r["dump_cost"])
    print("snapshots:", r["n_snapshots"], "  (XLA writes one only when a pass CHANGED the module)")
    print("\nscopex.diverge:")
    for k, v in r["diverge"].items():
        print(f"   {k:26s} {v}")
    print("   NOTE: diverges_at['pass'] is the snapshot's NAME, i.e. the pass about to run --"
          "\n         two passes downstream of the culprit here, and one that changed nothing."
          "\n         `caused_by` and `case_only_caused_by` are the translated, correct answers.")
    print("\nbiggest single-pass instruction jumps in the case "
          "(CAUSED BY, not the snapshot's own name):")
    for j in r["growth_head"]:
        print(f"   {j['caused_by']:<34s} {j['before']:7d} -> {j['after']:7d}  x{j['ratio']:<8}"
              f" [snapshot named {j['snapshot_named']}]")
    print("\nbiggest single-pass BYTE jumps (the metric that works when the count shrinks):")
    for j in r["bytes_jump"]:
        print(f"   {j['caused_by']:<34s} {j['before']:9d} -> {j['after']:9d}  x{j['ratio']}")
    print("\nslowest intervals from snapshot mtimes (scopex.pass_timeline):")
    for name, s in r["timeline_head"]:
        print(f"   {name:<52s} {s:8.3f} s")
    print("\ncase curve  :", r["case_curve"][:6], "...")
    print("control curve:", r["control_curve"][:6], "...")
