"""Reading what a compile left behind: per-pass growth, a timeline, and codegen size.

These are the hand-rolled routines that actually localised pathologies during a 30-case
investigation, promoted to API. Each earned its place by being the only instrument that worked on
some case:

``pass_growth``     Counts instructions in every per-pass snapshot. On a size-cliff case both arms
                    tracked within 1.21x right up to layout assignment, and then ONE pass took the
                    slow arm 2,458 -> 176,189 instructions while the fast arm went 2,030 -> 5,446.
                    Same pass sequence in both. No other instrument reported it.
``pass_timeline``   Times passes from dump-file MTIMES. XLA writes each snapshot as the pass
                    completes, so the timestamps reconstruct where the seconds went -- including in
                    the gaps AFTER the last HLO pass, which is where LLVM lives and where the HLO
                    pass timer cannot see. This localised a 522x case on which every count-based
                    view was null.
``codegen_size``    LLVM IR lines and object bytes. Several pathologies have near-identical HLO and
                    differ only below it; one measured 1540 identical BufferAllocations with
                    optimised IR differing 5.3x.
``custom_calls``    A census of ``custom_call_target``. XLA records pass DECISIONS here -- whether
                    the sort rewriter chose a CUB kernel or fell back to a generic bitonic
                    lowering, say -- and an opcode census cannot see the difference because both
                    are opcode ``custom-call``.

All of these read a directory produced by :func:`scopex.dump`. None of them recompiles.
"""

from __future__ import annotations

import collections
import os
import pathlib
import tempfile
import warnings
from typing import NamedTuple

from . import _parse

__all__ = ["PassStep", "pass_growth", "pass_timeline", "codegen_size", "custom_calls",
           "modules_in", "diverge", "opcode_census", "opcode_delta", "hlo_at", "boundaries_in"]

# The dump-FILENAME grammar and the fallback instruction-line pattern both live in scopex._parse,
# next to a real `ls` of a dump directory and a guard that raises when a parse comes back emptier
# than its input. `dump_snapshot_name` returns None for the many files in a dump that are not
# per-pass snapshots (.ll, .o, debug_options, the before/after-optimization modules), so the
# population -- not the call -- is what gets guarded, in `pass_growth` below.
from ._parse import dump_snapshot_name, is_hlo_instruction_line  # noqa: E402


def _count(text: str) -> tuple[int, int, str]:
    """``(instructions, computations, how)`` for one snapshot.

    Prefers ``scopex.levels.hlo_module``, i.e. XLA's own text parser, which gives an exact count
    from the object graph instead of a per-line guess. It parsed 2,811 of 2,811 real per-pass
    snapshots. ``how`` records which route ran, so a silent slide back onto the regex is visible in
    the returned data rather than invisible.
    """
    try:
        from .levels import hlo_module
        m = hlo_module(text)
        comps = m.computations()
        return sum(len(c.instructions()) for c in comps), len(comps), "native"
    except Exception:
        return (sum(1 for ln in text.splitlines() if is_hlo_instruction_line(ln)),
                text.count(" {\n") + text.count("{\n"), "regex")


class PassStep(NamedTuple):
    """One per-pass snapshot. ``instrs`` is the module as it stood BEFORE ``before_pass`` ran.

    ``how`` is ``"native"`` when the count came from XLA's parser and ``"regex"`` when it fell back
    to the line pattern; a mixed set of steps is not comparable and :func:`pass_growth` says so.
    """
    index: int
    pipeline: str
    after_pass: str
    before_pass: str
    instrs: int
    computations: int
    path: str
    mtime: float
    how: str = "native"

    @property
    def name(self) -> str:
        return f"{self.pipeline}/{self.before_pass}"


def modules_in(dump_dir: str | os.PathLike) -> list[str]:
    """Module stems present, largest first by snapshot count.

    A dump directory holds JAX's warm-up modules (``jit_convert_element_type`` and friends)
    alongside the one you care about. Picking the wrong stem is the most common way to read a dump
    and conclude nothing happened."""
    c: collections.Counter = collections.Counter()
    names = os.listdir(dump_dir)
    for f in names:
        m = dump_snapshot_name(f)
        if m:
            c[m["module"]] += 1
    if not c and any(".before_" in f for f in names):
        # There are snapshot-shaped filenames here and none of them parsed. That is the filename
        # grammar moving, not a dump without snapshots, and the two must not look alike.
        raise _parse.ParseError(
            f"scopex parser 'dump_snapshot_name' matched none of the {len(names)} files in "
            f"{dump_dir}, yet some of them contain '.before_'.\n"
            f"  built by   : xla/service/dump.cc\n"
            f"  example    : {next(f for f in names if '.before_' in f)}\n"
            f"  Fix _SNAPSHOT in scopex/_parse.py; do not let an empty module list read as "
            f"'this compile ran no passes'.")
    return [k for k, _ in c.most_common()]


def _pick(dump_dir, module: str | None) -> str:
    mods = modules_in(dump_dir)
    if not mods:
        raise FileNotFoundError(
            f"no per-pass snapshots in {dump_dir}. dump() needs passes='.*' (or a regex) -- "
            f"without it XLA writes only the before/after-optimisation modules.")
    if module is None:
        return mods[0]
    hit = [m for m in mods if module in m]
    if not hit:
        raise KeyError(f"no module matching {module!r}; present: {mods}")
    return hit[0]


def pass_growth(dump_dir: str | os.PathLike, *, module: str | None = None) -> list[PassStep]:
    """Instruction count at every pass boundary, in pipeline order.

    ``module`` selects a stem by substring; the default is the one with the most snapshots, which
    is the program you compiled rather than a JAX warm-up module.
    """
    stem = _pick(dump_dir, module)
    out: list[PassStep] = []
    for f in os.listdir(dump_dir):
        m = dump_snapshot_name(f)
        if not m or m["module"] != stem:
            continue
        p = pathlib.Path(dump_dir) / f
        text = p.read_text(errors="replace")
        n, ncomp, how = _count(text)
        out.append(PassStep(
            index=m["index"], pipeline=m["pipeline"],
            after_pass=m["after_pass"], before_pass=m["before_pass"],
            instrs=n, computations=ncomp,
            path=str(p), mtime=p.stat().st_mtime, how=how))
    out.sort(key=lambda s: s.index)
    fell_back = [s.name for s in out if s.how == "regex"]
    if fell_back and len(fell_back) != len(out):
        # A curve counted partly one way and partly the other is not a curve. The regex undercounts
        # tuple-shaped instructions, so a mixed curve shows a fake step exactly where the route
        # changed -- and pass_growth exists to find real steps.
        warnings.warn(
            f"{len(fell_back)} of {len(out)} snapshots fell back to the line-based counter while "
            f"the rest were counted natively, so this curve mixes two scales and its steps are not "
            f"trustworthy. First: {fell_back[:3]}", RuntimeWarning, stacklevel=2)
    return out


def pass_timeline(dump_dir: str | os.PathLike, *, module: str | None = None) -> list[tuple]:
    """``[(label, seconds), ...]`` from snapshot mtimes, plus the tail after the last HLO pass.

    XLA writes each snapshot as its pass completes, so consecutive mtimes bound how long the pass
    between them took. This is a coarse instrument -- it measures wall time including whatever else
    the machine was doing -- but it reaches somewhere ``pass_timings`` cannot: the interval between
    the LAST HLO snapshot and the emitted ``.ll``/``.o`` is LLVM, and on one case that interval was
    29.957 s of a compile whose HLO passes summed to a fraction of a second.
    """
    steps = pass_growth(dump_dir, module=module)
    if not steps:
        return []
    out = [(steps[i].name, steps[i].mtime - steps[i - 1].mtime) for i in range(1, len(steps))]
    last = steps[-1].mtime
    for suffix, label in ((".ir-no-opt.ll", "<llvm ir emission>"),
                          (".ir-with-opt.ll", "<llvm optimisation>"),
                          (".o", "<object codegen>")):
        cands = [pathlib.Path(dump_dir) / f for f in os.listdir(dump_dir) if f.endswith(suffix)]
        if cands:
            t = max(c.stat().st_mtime for c in cands)
            if t > last:
                out.append((label, t - last))
                last = t
    return out


def codegen_size(dump_dir: str | os.PathLike) -> dict:
    """LLVM IR line counts and object bytes. Empty dict when the backend emitted none.

    An empty result is meaningful: a program that constant-folds to a literal emits no LLVM module
    at all, which is itself the diagnosis.
    """
    out: dict = {"ir_no_opt_lines": 0, "ir_with_opt_lines": 0, "obj_bytes": 0, "files": {}}
    for f in os.listdir(dump_dir):
        p = pathlib.Path(dump_dir) / f
        if f.endswith(".ll"):
            n = sum(1 for _ in p.open(errors="replace"))
            key = "ir_with_opt_lines" if "with-opt" in f else "ir_no_opt_lines"
            out[key] += n
            out["files"][f] = n
        elif f.endswith(".o"):
            out["obj_bytes"] += p.stat().st_size
            out["files"][f] = p.stat().st_size
    return out


def custom_calls(source) -> collections.Counter:
    """Census of ``custom_call_target``, from a ``Compiled``, an ``HloModule``, or HLO text.

    Needed because an opcode census cannot distinguish two different pass DECISIONS: a CUB sort and
    a generic bitonic lowering are both opcode ``custom-call``, and which one XLA picked was the
    entire answer on two cases.

    ``custom_call_target`` is not on ``HloInstruction`` (its whole surface is ``async_wrapped_root,
    name, opcode, operands, to_string, users``), so the target itself still comes out of the printed
    form. But the INSTRUCTIONS are enumerated natively and filtered on the opcode enum, so the
    pattern only ever runs on a string already known to be a custom-call. Scanning the whole module
    text instead also counts the literal ``custom_call_target=`` that appears inside a
    ``backend_config`` blob or inside an embedded pre-fusion module.
    """
    try:
        from .levels import hlo_module
        m = hlo_module(source)
    except Exception:
        if not isinstance(source, str):                                      # pragma: no cover
            from .flags import hlo_text
            source = hlo_text(source)
        return collections.Counter(_parse.custom_call_targets(source))
    out: collections.Counter = collections.Counter()
    for comp in m.computations():
        for i in comp.instructions():
            if i.opcode.name == "kCustomCall":
                out.update(_parse.custom_call_targets(i.to_string()))
    return out


def diverge(case_dir, control_dir, *, module: str | None = None, factor: float = 1.5) -> dict:
    """Where two arms separate. THE routine this module exists for.

    Walks both per-pass curves in lockstep and returns the first pass at which the case's
    instruction count exceeds the control's by more than ``factor``, together with both curves.

    ``pass_sequence_identical`` is reported and matters: if the two arms ran DIFFERENT passes, the
    divergence point is a pass-selection difference rather than one pass behaving badly, and those
    want opposite fixes.
    """
    a = pass_growth(case_dir, module=module)
    b = pass_growth(control_dir, module=module)
    an = [s.name for s in a]
    bn = [s.name for s in b]
    at = {s.name: s for s in a}
    bt = {s.name: s for s in b}
    first = None
    for name in an:
        if name not in bt:
            continue
        r = at[name].instrs / max(1, bt[name].instrs)
        if r > factor:
            first = {"pass": name, "case_instrs": at[name].instrs,
                     "control_instrs": bt[name].instrs, "ratio": round(r, 2)}
            break
    return {
        "diverges_at": first,
        "pass_sequence_identical": an == bn,
        "case_only_passes": [n for n in an if n not in bt],
        "control_only_passes": [n for n in bn if n not in at],
        "case_final": a[-1].instrs if a else 0,
        "control_final": b[-1].instrs if b else 0,
        "case_curve": [(s.name, s.instrs) for s in a],
        "control_curve": [(s.name, s.instrs) for s in b],
    }


# ══════════════════════════════════════════════════════════════════════════════════════════════
# OPCODE CENSUSES, AND SUBTRACTING TWO OF THEM
#
# Several findings in the 30-case investigation were opcode censuses compared BY HAND: slice
# 46,268 vs 570 with dynamic-slice 0 vs 294 on one pair, select 1,404 vs 300 on another. Doing it
# by hand is slow and, worse, it is done at whatever boundary happened to be open -- and the answer
# depends entirely on the boundary. The same pair of arms compared before optimization and after it
# can differ in opposite directions, because the whole point of the pass pipeline is to change the
# opcode mix. `opcode_delta` therefore refuses to default the boundary silently: `at` is part of
# the returned dict, and `boundaries_in` lists what a directory can actually answer for.
# ══════════════════════════════════════════════════════════════════════════════════════════════

_BEFORE = ".before_optimizations.txt"
_AFTER = "_after_optimizations.txt"


def boundaries_in(dump_dir: str | os.PathLike, *, module: str | None = None) -> list[str]:
    """Every boundary in ``dump_dir`` that :func:`hlo_at` can read, in pipeline order.

    ``"before"`` and ``"after"`` are always there; the rest are ``pipeline/pass`` names and exist
    only if the dump was taken with ``passes=`` set.
    """
    names = os.listdir(dump_dir)
    stem = module
    if stem is None:
        mods = modules_in(dump_dir) if any(dump_snapshot_name(f) for f in names) else []
        stem = mods[0] if mods else None
    out = []
    if any(f.endswith(_BEFORE) and (stem is None or f.startswith(stem)) for f in names):
        out.append("before")
    for s in (pass_growth(dump_dir, module=module) if any(map(dump_snapshot_name, names)) else []):
        out.append(s.name)
    if any(f.endswith(_AFTER) and (stem is None or f.startswith(stem)) for f in names):
        out.append("after")
    return out


def hlo_at(dump_dir: str | os.PathLike, at: str = "after", *, module: str | None = None) -> str:
    """The HLO text at one boundary of a dump. ``at`` is ``"before"``, ``"after"``, or a pass name.

    A pass name may be given as ``pipeline/before_pass`` (what :class:`PassStep` calls itself) or as
    a plain substring of it; the FIRST matching snapshot in pipeline order wins, and an ambiguous
    or absent name raises with the list of what is there rather than falling back to a boundary the
    caller did not ask for.
    """
    names = sorted(os.listdir(dump_dir))
    if at in ("before", "after"):
        suffix = _BEFORE if at == "before" else _AFTER
        hits = [f for f in names if f.endswith(suffix)]
        if module:
            hits = [f for f in hits if module in f]
        elif hits:
            # Same rule as `modules_in`: prefer the module with the most per-pass snapshots, i.e.
            # the program you compiled rather than a JAX warm-up module.
            mods = modules_in(dump_dir) if any(map(dump_snapshot_name, names)) else []
            for m in mods:
                if any(f.startswith(m + ".") for f in hits):
                    hits = [f for f in hits if f.startswith(m + ".")]
                    break
        if not hits:
            raise FileNotFoundError(
                f"no *{suffix} in {dump_dir}"
                + (f" for module {module!r}" if module else "")
                + f". Present boundaries: {boundaries_in(dump_dir, module=module)}")
        return (pathlib.Path(dump_dir) / hits[0]).read_text(errors="replace")
    steps = pass_growth(dump_dir, module=module)
    hit = [s for s in steps if s.name == at] or [s for s in steps if at in s.name]
    if not hit:
        raise KeyError(f"no pass boundary matching {at!r} in {dump_dir}. Present: "
                       f"{boundaries_in(dump_dir, module=module)}")
    return pathlib.Path(hit[0].path).read_text(errors="replace")


def opcode_census(source, *, per_computation: bool = False) -> collections.Counter:
    """``Counter`` of XLA opcodes, walked NATIVELY from the object model.

    ``source`` may be a ``Compiled``, an ``HloModule``, or HLO text -- including one per-pass dump
    snapshot. Opcodes come from the ``HloOpcode`` enum via :func:`scopex.levels.opcode_of`, not
    from a line pattern, so a tuple-shaped instruction (``while``, ``call``, ``custom-call``) is
    counted like any other; the line-based counter this replaces dropped every one of them.

    ``per_computation`` prefixes each key with its computation, which separates "the fusion body
    grew" from "there are more fusions".
    """
    from .levels import hlo_module, opcode_of
    m = hlo_module(source)
    c: collections.Counter = collections.Counter()
    for comp in m.computations():
        for i in comp.instructions():
            c[f"{comp.name}/{opcode_of(i)}" if per_computation else opcode_of(i)] += 1
    return c


def opcode_delta(case_dump, control_dump, *, at: str = "after", module: str | None = None,
                 top: int = 12) -> dict:
    """The opcodes that differ most between two dumps, at one chosen boundary. One call.

    ``at`` picks the boundary in BOTH dumps -- ``"before"``, ``"after"``, or a pass name
    (see :func:`boundaries_in`). It is echoed in the result because the answer is meaningless
    without it: the pass pipeline exists to change the opcode mix, so the same two arms compared
    before and after optimization can differ in opposite directions.

    Returns ``{"at", "delta", "case_only", "control_only", "case_total", "control_total",
    "case_census", "control_census"}``. ``delta`` is ``[(opcode, case_n, control_n, case_n -
    control_n), ...]`` sorted by absolute difference, longest first.

    ``case_only``/``control_only`` are the opcodes present in one arm and ABSENT in the other, and
    they are listed separately on purpose: an opcode that went 294 -> 0 is a lowering decision,
    while one that went 46,268 -> 570 is the same decision taken at a different scale, and those
    want different fixes.
    """
    a = opcode_census(hlo_at(case_dump, at, module=module))
    b = opcode_census(hlo_at(control_dump, at, module=module))
    rows = [(k, a.get(k, 0), b.get(k, 0), a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b)]
    rows.sort(key=lambda r: (-abs(r[3]), r[0]))
    return {
        "at": at,
        "delta": rows[:top] if top else rows,
        "case_only": sorted(k for k in a if k not in b),
        "control_only": sorted(k for k in b if k not in a),
        "case_total": sum(a.values()),
        "control_total": sum(b.values()),
        "case_census": dict(a.most_common()),
        "control_census": dict(b.most_common()),
    }


_PROBE_SRC = '''"""scopex selftest probe -- a marked program with real user frames."""
import jax
import jax.numpy as jnp


def leaf(x):
    return jnp.tanh(x)


def body(x):
    with jax.named_scope("scopex:user.Probe.body"):
        return leaf(x) * 2.0


def program(x):
    with jax.named_scope("scopex:lib.selftest"):
        return jnp.sum(body(x) @ x)
'''


def _probe_program():
    """``(program, path)`` for a marked probe living in a real file outside this package.

    Nested calls on purpose: ``program -> body -> leaf`` gives the optimized module a stack-frame
    chain several frames deep, so :func:`scopex.levels.hlo_sites` is exercised on a chain and not
    just on a leaf. A one-frame probe passes even when the parent links are read wrongly.
    """
    import importlib.util
    d = pathlib.Path(tempfile.mkdtemp(prefix="scopex-selftest-"))
    p = d / "scopex_selftest_probe.py"
    p.write_text(_PROBE_SRC)
    spec = importlib.util.spec_from_file_location("scopex_selftest_probe", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.program, str(p)


def selftest(verbose: bool = True, *, strict: bool = True) -> dict:
    """Run every parser in scopex against BOTH its frozen sample and a freshly compiled program.

    Two halves, and both are needed:

    * :func:`scopex._parse.conformance` replays each parser over a verbatim capture of the text it
      was written for. That half needs no jax, so it separates "somebody edited the parser" from
      "the compiler moved".
    * this function compiles a small MARKED program, dumps it, and checks that every level and every
      artifact view comes back NON-EMPTY, that the levels still agree with each other, and that the
      HLO stack-frame tables still resolve to source lines the jaxpr also reports.

    ``strict=True`` (the default) RAISES on any failure. That is the point: this package has three
    times shipped a parser that returned a plausible empty answer, and each time the answer was
    believed. Run it after any jax upgrade -- and before the first compile in the process, since XLA
    reads its dump flags when the backend is first initialised.
    """
    import jax
    import jax.numpy as jnp

    from .flags import backend_initialized, dump, hlo_text, stablehlo_text
    from .levels import frame_tables, hlo_instructions, walk_hlo, walk_stablehlo
    from .walk import NO_FRAME, walk

    if backend_initialized():
        raise RuntimeError("run selftest before the first compile in the process")

    r: dict = {}
    bad: list[str] = []

    # ── half one: the frozen samples, no jax involved ───────────────────────────────────────────
    try:
        r["conformance"] = _parse.conformance()["ok"]
    except Exception as e:
        r["conformance"] = False
        bad.append(f"embedded-sample conformance: {str(e).splitlines()[0]}")

    # ── half two: a real compile ────────────────────────────────────────────────────────────────
    # The probe is written to a FILE OUTSIDE this package and imported, rather than defined here.
    # That is load-bearing: both site resolvers filter out frames inside jax and inside scopex, so a
    # probe defined in this module has no user frame at all and the cross-level site join -- the one
    # check that can catch a frame table resolving to the WRONG line rather than to none -- would
    # compare two empty sets and pass.
    program, probe_file = _probe_program()

    with dump(passes=".*", fusion=False, keep=True) as d:
        low = jax.jit(program).lower(jnp.ones((32, 32)))
        c = low.compile()
    r["dump_dir"] = d

    def check(name, fn, *, count=True):
        try:
            v = fn()
        except Exception as e:
            bad.append(f"{name}: raised {type(e).__name__}: {str(e).splitlines()[0]}")
            return None
        r[name] = len(v) if count and hasattr(v, "__len__") else v
        if not v:
            bad.append(f"{name}: EMPTY, from a program that demonstrably has some")
        return v

    check("modules", lambda: modules_in(d))
    check("pass_steps", lambda: pass_growth(d))
    check("timeline_entries", lambda: pass_timeline(d))
    r["codegen"] = codegen_size(d)
    r["custom_calls"] = dict(custom_calls(c))          # legitimately empty on this program

    text = hlo_text(c)
    eqns = list(walk(jax.make_jaxpr(program)(jnp.ones((32, 32)))))
    sh = check("stablehlo_units", lambda: list(walk_stablehlo(low)))
    hl = check("hlo_units", lambda: list(walk_hlo(c)))
    check("hlo_instructions", lambda: list(hlo_instructions(c)))
    check("stablehlo_chars", lambda: len(stablehlo_text(low)), count=False)
    tab = check("frame_tables", lambda: frame_tables(text)["frames"])
    r["parent_offset"] = frame_tables(text)["parent_offset"]

    # The checks a count cannot make. Each one is a way a parser here has actually failed:
    if hl:
        if not any(i.path for i in hl):
            bad.append("hlo_units: not one instruction carries an op_name. The metadata parse is "
                       "returning empty dicts -- which is how the optimized module came to be "
                       "written up as carrying no provenance at all")
        if tab and not any(i.site not in (NO_FRAME, "?", "") for i in hl):
            bad.append("hlo_units: the module HAS stack-frame tables and not one instruction "
                       "resolved to a file:line. stack_frame_id is being dropped (bug #2)")
        if not any("scopex:user.Probe.body" in i.path for i in hl):
            bad.append("hlo_units: the marked scope reached zero optimized instructions -- either "
                       "the name stack is not surviving lowering or op_name is not being read")
        # THE ONE CHECK THAT CATCHES A PARSER RESOLVING TO THE WRONG ANSWER RATHER THAN TO NONE.
        # Both levels resolve a site independently -- the jaxpr from python tracebacks, the
        # optimized HLO by walking XLA's frame tables -- so they must land on the same lines of the
        # probe file. If the parent links are read with the wrong convention the HLO side still
        # produces plausible file:line pairs; they are just somebody else's.
        sites = {i.site for i in hl if i.site not in (NO_FRAME, "?", "")}
        jaxpr_sites = {e.site for e in eqns if e.site != NO_FRAME}
        probe = {s for s in sites if s.startswith(probe_file)}
        r["site_join"] = round(len(sites & jaxpr_sites) / max(1, len(sites)), 4)
        if not jaxpr_sites or not sites:
            bad.append(f"site join is untestable: {len(sites)} HLO sites, {len(jaxpr_sites)} jaxpr "
                       f"sites. One of the two site resolvers returned nothing at all")
        elif not probe:
            bad.append(f"hlo_units: not one instruction resolved into the probe file {probe_file}; "
                       f"got {sorted(sites)[:3]}. The frame tables resolve to the WRONG frames")
        elif not (sites & jaxpr_sites):
            bad.append(
                f"hlo_units: no optimized-HLO site matches any jaxpr site "
                f"({sorted(sites)[:2]} vs {sorted(jaxpr_sites)[:2]}). The frame tables are "
                f"resolving to the WRONG frames, which no emptiness check can see -- re-derive the "
                f"parent_frame_id convention in scopex/_parse.py:_parent_offset.")
    if sh:
        r["stablehlo_named"] = sum(1 for i in sh if i.path)
        if not r["stablehlo_named"]:
            bad.append("stablehlo_units: walked operations but not one carries a name stack. This "
                       "is bug #1 -- the level looks empty rather than broken")
        if len(sh) < 4:
            bad.append(f"stablehlo_units: {len(sh)} units from a module containing a tanh, a dot "
                       f"and a reduce. That is the shape of bug #1")

    r["ok"] = not bad
    r["broken"] = bad
    if verbose:
        print(f"scopex.selftest: {'OK' if not bad else 'FAILED'}")
        for k, v in r.items():
            if k != "broken":
                print(f"  {k:18s} {v}")
        for b in bad:
            print(f"  BROKEN: {b}")
    if bad:
        msg = ("scopex.selftest FAILED -- these parsers no longer read what jax/XLA emit:\n  - "
               + "\n  - ".join(bad)
               + f"\nDump kept at {d} for inspection. Every parser lives in scopex/_parse.py next "
                 "to the sample it was written against; fix it there and run selftest() again.")
        if strict:
            raise _parse.ParseError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return r
