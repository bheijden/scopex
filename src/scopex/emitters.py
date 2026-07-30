"""The EMITTER level: XLA's own MLIR pipeline, which runs BELOW every HLO pass.

``pass_growth`` stops where the HLO passes stop. Underneath it each backend hands every fusion to
an emitter that builds an MLIR module and runs a second, entirely separate pass pipeline on it --
~65 passes on jaxlib 0.10.2 -- before LLVM ever sees a line of IR. ``pass_timeline`` could only see
that region as one undifferentiated gap between the last HLO snapshot and the ``.ll`` file. On one
case that gap was 29.957 s of the compile.

``--xla_dump_emitter_re`` opens it. :func:`scopex.dump` passes it when ``emitter=True``.

    with scopex.dump(passes=".*", emitter=True) as d:
        jax.jit(fn).lower(x).compile()
    for k in scopex.emitter_growth(d):
        print(k.kernel, k.n_passes, k.ops_first, "->", k.ops_last, k.obj_bytes)

WHAT THE FLAG ACTUALLY WRITES, MEASURED ON 0.10.2 -- AND THE TWO WAYS IT LOOKS EMPTY
-----------------------------------------------------------------------------------

1. **The flag's argument is not a kernel filter.** It is partial-matched against the fixed string
   ``"mlir-fusion"``, so ``--xla_dump_emitter_re=my_kernel`` writes NOTHING on a compile that has
   ``my_kernel``. See the block comment in :mod:`scopex._parse` for the bisection that pinned it.
   :func:`scopex.dump` never forwards a user string here.

2. **A kernel that reaches this level is the exception, not the rule.** The emitters own only the
   fusions their backend routes to them. On CPU/0.10.2 a program of ``tanh``, ``dot`` and ``reduce``
   compiled to ONE ``__ynn_fusion`` custom fusion and produced no emitter files at all, while the
   ``jit_broadcast_in_dim`` module JAX compiles behind your back produced four. So an empty
   ``emitter_growth`` means "these fusions did not go through the MLIR emitters", which is a
   finding, and NOT "the emitter did no work" -- and the first module in the directory is very
   likely to be someone else's. That is why every record carries its ``module``.

The backends also differ in what they leave behind, and the difference decides what can be joined:

===============  ====================================  =====================================
                 CPU                                   GPU
===============  ====================================  =====================================
mlir-passes.log  per kernel, by NAME                    per kernel, by NAME
stage ``.mlir``  pre-optimization / post-lowering /     ABSENT -- measured, 0 of them
                 post-optimization
``.ll``          per kernel, by NAME                    per LLVM module, by INDEX
object           ``.o``, per kernel, by NAME            ``.ptx``, per LLVM module, by INDEX
===============  ====================================  =====================================

So on CPU the MLIR curve, the LLVM IR and the object bytes are one row per kernel. On GPU the
codegen products are keyed by an index that is not the kernel name, and this module reports
``codegen_joined=False`` rather than pairing them by position and hoping.
"""

from __future__ import annotations

import collections
import os
import pathlib
from typing import NamedTuple

from . import _parse
from ._parse import emitter_dump_name, mlir_log_damage, mlir_op_lines, mlir_pass_dumps

__all__ = ["EmitterKernel", "emitter_growth", "emitter_files", "emitter_summary"]

_STAGES = ("pre-optimization", "post-lowering", "post-optimization")


class EmitterKernel(NamedTuple):
    """One kernel's trip through the emitter's MLIR pipeline, plus what it emitted.

    ``steps`` is the per-MLIR-pass curve, in pipeline order. A pass that runs at ``func.func``
    scope is printed once per function, so ``steps`` is longer than the number of distinct passes
    and each step names the ``symbol`` it ran on.

    ``codegen_joined`` says whether ``ir_*_lines``/``obj_bytes`` really belong to THIS kernel. On
    GPU they cannot be joined by name (see the module docstring) and are reported as 0 with the
    flag False, rather than filled in from a positional guess.
    """
    module: str
    kernel: str
    steps: tuple[_parse.MlirPassDump, ...]
    stages: dict          # {"pre-optimization": {"lines": n, "ops": n}, ...} -- CPU only
    ir_no_opt_lines: int
    ir_with_opt_lines: int
    obj_bytes: int
    ptx_bytes: int
    codegen_joined: bool
    log_bytes: int
    log_path: str
    damage: dict = {}     # {"headers", "complete", "torn", "nul_bytes"} -- see mlir_log_damage

    @property
    def n_passes(self) -> int:
        """Distinct MLIR pass NAMES, as opposed to ``len(steps)`` pass RUNS."""
        return len({s.pass_name for s in self.steps})

    @property
    def ops_first(self) -> int:
        return self.steps[0].ops if self.steps else 0

    @property
    def ops_last(self) -> int:
        return self.steps[-1].ops if self.steps else 0

    @property
    def peak(self):
        """``(pass_name, ops)`` for the largest IR any pass ever saw -- where the level costs.

        Falls back to the ``pre-optimization`` stage when this kernel has no pass log, which is
        the ordinary case for a kernel XLA served from its cache: the module was still built, and
        its size is still the thing you are looking for.
        """
        if not self.steps:
            return ("<pre-optimization>", max((v["ops"] for v in self.stages.values()), default=0))
        s = max(self.steps, key=lambda s: s.ops)
        return (s.pass_name, s.ops)

    def growth(self) -> list[tuple[str, int]]:
        """``[(pass_name, ops), ...]``: the emitter-level analogue of :func:`scopex.pass_growth`."""
        return [(s.pass_name, s.ops) for s in self.steps]

    def jumps(self, factor: float = 1.5) -> list[tuple[str, int, int, float]]:
        """Consecutive steps where the operation count grew by more than ``factor``.

        The emitter level's whole reason to exist: a pipeline of ~65 passes over one kernel, where
        one of them is the unroller.
        """
        out = []
        for a, b in zip(self.steps, self.steps[1:], strict=False):
            if a.ops and b.ops / a.ops > factor:
                out.append((b.pass_name, a.ops, b.ops, round(b.ops / a.ops, 2)))
        return out


def emitter_files(dump_dir: str | os.PathLike) -> dict:
    """Raw inventory: ``{(module, kernel): {kind: path}}``, plus the counts by kind.

    Separated from :func:`emitter_growth` so that "the flag wrote nothing" and "the flag wrote
    files this reader could not parse" are distinguishable without reading any IR.
    """
    names = os.listdir(dump_dir)
    by: dict = collections.defaultdict(dict)
    kinds: collections.Counter = collections.Counter()
    for f in names:
        d = emitter_dump_name(f)
        if not d:
            continue
        by[(d["module"], d["kernel"])][d["kind"]] = str(pathlib.Path(dump_dir) / f)
        kinds[d["kind"]] += 1
    # The emitter filename grammar has to guess where the module stem ends and the kernel begins,
    # because both are dotted (`module_0000.jit__f_scan.__compute_module_add_bitcast_fusion.265`).
    # A wrong guess produces a plausible stem and a wrong kernel, so the stems are checked against
    # the ones the REST of the dump names -- which are read by an unrelated pattern.
    known = {f[:f.index(".before_optimizations.txt")] for f in names
             if ".before_optimizations.txt" in f}
    known |= {d["module"] for d in map(_parse.dump_snapshot_name, names) if d}
    unknown = sorted({m for m, _ in by} - known) if known else []
    if unknown:
        raise _parse.ParseError(
            f"scopex parser 'emitter_dump_name' produced module stem(s) {unknown} that no other "
            f"file in {dump_dir} mentions (the dump names {sorted(known)}).\n"
            f"  built by   : xla/service/dump.cc via the backend emitters\n"
            f"  The stem/kernel split in scopex/_parse.py:_EMIT_* has landed in the wrong place, "
            f"which means every kernel in this report is named wrongly -- plausibly, not emptily.")
    unparsed = [f for f in names
                if (f.endswith(".mlir") or f.endswith(".mlir-passes.log"))
                and not emitter_dump_name(f)]
    if unparsed:
        # Emitter-shaped filenames that this grammar did not accept. That is dump.cc moving, not a
        # compile without an emitter level, and the two must not look alike.
        raise _parse.ParseError(
            f"scopex parser 'emitter_dump_name' rejected {len(unparsed)} of the emitter-shaped "
            f"files in {dump_dir}.\n"
            f"  built by   : xla/service/dump.cc via the backend emitters\n"
            f"  example    : {unparsed[0]}\n"
            f"  Fix the _EMIT_* patterns in scopex/_parse.py; do not let a rejected filename read "
            f"as 'this compile has no emitter level'.")
    return {"kernels": dict(by), "kinds": dict(kinds), "n_files": sum(kinds.values())}


def emitter_growth(dump_dir: str | os.PathLike, *, module: str | None = None,
                   steps: bool = True) -> list[EmitterKernel]:
    """Per-kernel MLIR/LLVM-IR/object sizes for a dump taken with ``emitter=True``.

    Largest first, by peak MLIR operation count. ``module`` selects a module stem by substring; the
    default is every module in the directory, JAX's warm-up modules included -- deliberately, since
    on CPU it is entirely possible for the only kernels at this level to belong to one of them, and
    silently hiding that would misreport the level as empty.

    ``steps=False`` skips reading the ``.mlir-passes.log`` bodies (they run to hundreds of KB per
    kernel) and returns sizes only.
    """
    inv = emitter_files(dump_dir)
    out: list[EmitterKernel] = []
    for (mod, kernel), paths in inv["kernels"].items():
        if module and module not in mod:
            continue
        log = paths.get("passes-log")
        dumps: tuple = ()
        damage: dict = {}
        if log and steps:
            t = pathlib.Path(log).read_text(errors="replace")
            dumps = tuple(mlir_pass_dumps(t))
            damage = mlir_log_damage(t)
        stages = {}
        for st in _STAGES:
            p = paths.get(st)
            if p:
                t = pathlib.Path(p).read_text(errors="replace")
                stages[st] = {"lines": len(t.splitlines()),
                              "ops": len(mlir_op_lines(t, allow_empty=True))}
        # The codegen join is by NAME and only valid when the .ll/.o filenames actually carry the
        # kernel name -- i.e. when this kernel has an mlir-passes.log under the same key. On GPU the
        # key is an LLVM module index, so those rows have no log and are dropped below.
        joined = bool(log) and any(k in paths for k in ("ir-no-opt", "ir-with-opt", "obj"))

        def _lines(k, _p=paths):
            f = _p.get(k)
            return sum(1 for _ in open(f, errors="replace")) if f else 0

        def _bytes(k, _p=paths):
            f = _p.get(k)
            return os.path.getsize(f) if f else 0

        if not log and not stages:
            # Codegen products only, no MLIR at all -- on GPU that is an LLVM-module INDEX row
            # (`module_0002.jit_elem.1.ptx`), not a kernel. Do not invent one.
            continue
        # A kernel with stage .mlir files but NO passes-log is REAL and common: measured 12 such
        # kernels against 5 with logs in one CPU dump. XLA writes the log only when it actually
        # runs the pipeline, so a log-less kernel is one whose emitted module was served from the
        # kernel cache. Dropping them (the first version did) undercounted the level by 70%.
        out.append(EmitterKernel(
            module=mod, kernel=kernel, steps=dumps, stages=stages,
            ir_no_opt_lines=_lines("ir-no-opt") if joined else 0,
            ir_with_opt_lines=_lines("ir-with-opt") if joined else 0,
            obj_bytes=_bytes("obj") if joined else 0,
            ptx_bytes=_bytes("ptx") if joined else 0,
            codegen_joined=joined, log_bytes=_bytes("passes-log"), log_path=log or "",
            damage=damage))
    out.sort(key=lambda k: -k.peak[1])
    return out


def emitter_summary(dump_dir: str | os.PathLike, *, module: str | None = None) -> dict:
    """One dict for a whole dump: kernels, total MLIR pass runs, the biggest jump anywhere.

    ``worst_jump`` is the point of the whole level -- ``(module, kernel, pass, before, after,
    ratio)`` for the single MLIR pass that grew some kernel's IR the most.
    """
    ks = emitter_growth(dump_dir, module=module)
    worst = None
    for k in ks:
        for name, a, b, r in k.jumps(1.0001):
            if worst is None or r > worst[-1]:
                worst = (k.module, k.kernel, name, a, b, r)
    return {
        "kernels": len(ks),
        "modules": sorted({k.module for k in ks}),
        "pass_runs": sum(len(k.steps) for k in ks),
        "distinct_passes": len({s.pass_name for k in ks for s in k.steps}),
        "peak_ops": max((k.peak[1] for k in ks), default=0),
        "worst_jump": worst,
        "obj_bytes": sum(k.obj_bytes for k in ks),
        # Non-zero means XLA wrote at least one pass log torn, so at least one curve has a hole
        # in it that is XLA's and not the program's. See scopex._parse.mlir_log_damage.
        "torn_snapshots": sum(k.damage.get("torn", 0) for k in ks),
        "kernels_without_pass_log": sum(1 for k in ks if not k.steps),
        "codegen_joined": all(k.codegen_joined for k in ks) if ks else None,
        "by_kernel": [(k.module, k.kernel, len(k.steps), k.ops_first, k.peak[1], k.ops_last)
                      for k in ks],
    }
