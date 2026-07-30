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
import re
import warnings
from typing import NamedTuple

__all__ = ["PassStep", "pass_growth", "pass_timeline", "codegen_size", "custom_calls",
           "modules_in", "diverge"]

# module_0656.jit_solve_fn.0000.fusion.after_sort-iota-fusion.before_priority-fusion.txt
_SNAP = re.compile(
    r"^module_(?P<mod>\d+)\.(?P<fn>.+?)\.(?P<idx>\d{4})\.(?P<pipeline>[^.]+)"
    r"\.after_(?P<after>[^.]+)\.before_(?P<before>[^.]+)\.txt$")
# an HLO instruction line, same shape scopex.levels uses
_INSTR = re.compile(r"^\s*(?:ROOT\s+)?%?[\w.\-]+\s*=\s*\S+\s+[a-z][\w-]*\(")


class PassStep(NamedTuple):
    """One per-pass snapshot. ``instrs`` is the module as it stood BEFORE ``before_pass`` ran."""
    index: int
    pipeline: str
    after_pass: str
    before_pass: str
    instrs: int
    computations: int
    path: str
    mtime: float

    @property
    def name(self) -> str:
        return f"{self.pipeline}/{self.before_pass}"


def modules_in(dump_dir: str | os.PathLike) -> list[str]:
    """Module stems present, largest first by snapshot count.

    A dump directory holds JAX's warm-up modules (``jit_convert_element_type`` and friends)
    alongside the one you care about. Picking the wrong stem is the most common way to read a dump
    and conclude nothing happened."""
    c: collections.Counter = collections.Counter()
    for f in os.listdir(dump_dir):
        m = _SNAP.match(f)
        if m:
            c[f"module_{m.group('mod')}.{m.group('fn')}"] += 1
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
        m = _SNAP.match(f)
        if not m or f"module_{m.group('mod')}.{m.group('fn')}" != stem:
            continue
        p = pathlib.Path(dump_dir) / f
        text = p.read_text(errors="replace")
        out.append(PassStep(
            index=int(m.group("idx")), pipeline=m.group("pipeline"),
            after_pass=m.group("after"), before_pass=m.group("before"),
            instrs=sum(1 for ln in text.splitlines() if _INSTR.match(ln)),
            computations=text.count(" {\n") + text.count("{\n"),
            path=str(p), mtime=p.stat().st_mtime))
    return sorted(out, key=lambda s: s.index)


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


_CC = re.compile(r'custom_call_target="([^"]*)"')


def custom_calls(text_or_compiled) -> collections.Counter:
    """Census of ``custom_call_target``, from optimized-HLO text or a ``Compiled``.

    Needed because an opcode census cannot distinguish two different pass DECISIONS: a CUB sort and
    a generic bitonic lowering are both opcode ``custom-call``, and which one XLA picked was the
    entire answer on two cases.
    """
    if not isinstance(text_or_compiled, str):
        from .flags import hlo_text
        text_or_compiled = hlo_text(text_or_compiled)
    return collections.Counter(_CC.findall(text_or_compiled))


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


def selftest(verbose: bool = True) -> dict:
    """Compile a tiny program, dump it, and check every parser here returns something.

    These are all TEXT parsers over formats XLA does not promise to keep stable, and this project
    has twice shipped one that returned an empty result which read as 'nothing to see' rather than
    'I am broken'. Run this after any jax upgrade.
    """
    import jax
    import jax.numpy as jnp
    from .flags import backend_initialized, dump

    if backend_initialized():
        raise RuntimeError("run selftest before the first compile in the process")
    with dump(passes=".*", fusion=False, keep=True) as d:
        c = jax.jit(lambda x: jnp.sum(jnp.tanh(x) @ x)).lower(jnp.ones((32, 32))).compile()
    r = {"modules": len(modules_in(d)), "pass_steps": len(pass_growth(d)),
         "timeline_entries": len(pass_timeline(d)), "codegen": codegen_size(d),
         "custom_calls": dict(custom_calls(c))}
    bad = [k for k in ("modules", "pass_steps", "timeline_entries") if not r[k]]
    r["ok"] = not bad
    r["broken"] = bad
    if bad:
        warnings.warn(f"scopex.artifacts parsers returning nothing: {bad}. The dump format has "
                      f"probably moved; do not trust these views until fixed.", RuntimeWarning)
    if verbose:
        print(r)
    return r
