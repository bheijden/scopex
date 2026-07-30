"""Getting the text out, and the ways of asking that silently return nothing.

Every accessor here exists because the obvious call returns a plausible-looking answer that is
wrong. The numbers are from jax 0.10.2, probing a program with two nested named scopes around a
``tanh * sin`` reduction and counting occurrences of the inner scope name.

===========================================  ==========  =========================================
call                                         found       verdict
===========================================  ==========  =========================================
jaxpr ``source_info.name_stack``                  4      correct
``Lowered.as_text()``                             0      TRAP -- defaults to debug_info=False
``Lowered.as_text(debug_info=True)``              4      correct
``str(Lowered.compiler_ir('stablehlo'))``         0      TRAP -- the PRINTER's default, not the IR
``compiler_ir('hlo').as_hlo_text()``              0      TRAP -- the PRINTER again, see below
``compiler_ir('hlo').get_hlo_module()``           4      correct -- scopex.pre_optimization_hlo
``Compiled.as_text()``                            9      correct
``executable.get_hlo_text()``                     9      correct
``executable.hlo_modules()[0].to_string()``       9      correct
===========================================  ==========  =========================================

The ``compiler_ir`` rows are the dangerous ones: it is the accessor that *looks* structured and
principled, so a reader trusts its empty answer and concludes the IR carries no provenance.

AND ONE OF THEM WAS OVER-READ, WHICH IS ITS OWN LESSON. ``compiler_ir('stablehlo')`` returns an
``ir.Module`` -- the very object jax lowered into, ``is``-identical across calls. It does not drop
anything. Only ``__str__`` does, because ``Operation.__str__`` defaults to
``enable_debug_info=False``. Ask the same object to print with debug info and you get text
BYTE-IDENTICAL to ``as_text(debug_info=True)`` (48,770 chars, 478 ``loc(``, measured on
``examples/marked_framework.py``)::

    m = lowered.compiler_ir('stablehlo')
    str(m)                                    # 0 loc(  -- the trap
    m.operation.print(enable_debug_info=True) # 478 loc( -- same as as_text(debug_info=True)

So "this accessor prints nothing useful" was true and "this route cannot see provenance" was not.
:func:`scopex.walk_stablehlo` walks that module directly.

AND THEN THE ``hlo`` ROW TURNED OUT TO BE THE SAME STORY, WHICH IS WHY THAT SENTENCE IS WORTH
KEEPING. This file used to end by warning that the ``hlo`` row had not been re-examined and must
not be assumed to behave like the ``stablehlo`` one. It does. ``compiler_ir('hlo')`` returns an
``XlaComputation`` with two printers, and only one of them is lossy::

    c = lowered.compiler_ir('hlo')
    c.as_hlo_text()                    #   692 chars, 0 op_name, 0 stack_frame_id  -- the trap
    c.get_hlo_module().to_string()     # 2,092 chars, 9 op_name, 6 stack_frame_id  -- and the
                                       #   StackFrameIndex tables, so sites resolve

Same call, same object, 3x the text. So the pre-optimization module carries FULL provenance and
always did; :func:`scopex.pre_optimization_hlo` is that route, and it needs no dump directory.
Twice now the lossy thing has been a printer default and the loud conclusion has been about the
IR. When an accessor here looks empty, print the object another way before believing it.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings

from . import _parse
from .coverage import Coverage

__all__ = ["stablehlo_text", "hlo_text", "check_env", "dump_flags", "vmodule_env",
           "backend_initialized", "dump", "pass_timings", "Coverage", "TRAPS"]

# The pass-log pattern and its unit table now live in scopex._parse, next to a verbatim capture of
# the lines XLA printed -- including the `min` line that broke them. Kept reachable under their old
# names so existing callers and tests still find them; there is exactly one definition.
_PASS_LINE = _parse._PASS_LINE
_UNIT = _parse.UNITS

TRAPS = {
    "lowered_as_text_default": (
        "Lowered.as_text() defaults to debug_info=False and prints no locations. "
        "Use scopex.stablehlo_text(lowered), or pass debug_info=True."),
    "compiler_ir": (
        "str(Lowered.compiler_ir('stablehlo')) prints no locations -- Operation.__str__ defaults "
        "to enable_debug_info=False. The OBJECT keeps them: print it with enable_debug_info=True "
        "and you get text byte-identical to as_text(debug_info=True), or walk it with "
        "scopex.walk_stablehlo. compiler_ir('hlo') is the SAME STORY, one accessor further down: "
        "XlaComputation.as_hlo_text() strips every metadata= block (692 chars, 0 op_name) while "
        "the same object's .get_hlo_module().to_string() keeps all of it (2,092 chars, 9 op_name, "
        "6 stack_frame_id) -- use scopex.pre_optimization_hlo."),
    "vmodule_silent": (
        "TF_CPP_VMODULE alone is a silent no-op: importing jax sets TF_CPP_MIN_LOG_LEVEL=1, which "
        "suppresses all VLOG output. Set TF_CPP_MIN_LOG_LEVEL=0 as well."),
    "barrier_erased": (
        "optimization_barrier is erased by OptimizationBarrierExpander before the optimized HLO "
        "exists. Counting it in the optimized module always returns 0 and is not a survival check "
        "-- count it in the pre-optimization module."),
    "persistent_cache": (
        "jax_compilation_cache_dir is set. Cold-compile measurements will be contaminated by cache "
        "hits. Unset it for any timing run."),
}


def stablehlo_text(lowered) -> str:
    """StableHLO WITH locations. The plain ``as_text()`` omits them."""
    return lowered.as_text(debug_info=True)


def hlo_text(compiled) -> str:
    """Optimized HLO with metadata. All three spellings were measured equivalent on jax 0.10.2;
    this one goes through the executable so it does not depend on ``Compiled.as_text``'s defaults
    staying put."""
    try:
        return compiled.runtime_executable().hlo_modules()[0].to_string()
    except Exception:                                   # pragma: no cover -- backend without it
        return compiled.as_text()


def vmodule_env(spec: str = "hlo_pass_pipeline=1") -> dict[str, str]:
    """Environment for XLA per-pass VLOG. Both keys are required; the second alone does nothing.

    Returns a dict to merge into a SUBPROCESS environment -- setting it in-process after jax is
    imported is too late, because the log level is read at import."""
    return {"TF_CPP_MIN_LOG_LEVEL": "0", "TF_CPP_VMODULE": spec}


def dump_flags(path: str, *, fusion: bool = True, passes: str | None = None,
               emitter: bool = False) -> dict[str, str]:
    """XLA_FLAGS for dumping compiler artifacts.

    ``fusion`` adds the priority-fusion decision dump (free: it is written during the pass that
    already runs). ``passes`` is a regex of pass names to snapshot HLO around; ``".*"`` snapshots
    every pass and is large.

    ``emitter`` opens the level BELOW the HLO passes: each backend's emitter runs its own MLIR
    pipeline (~65 passes on 0.10.2) before LLVM, and ``--xla_dump_emitter_re`` writes a per-kernel
    snapshot of every one. Verified present on jaxlib 0.10.2 on CPU and GPU; read the result with
    :func:`scopex.emitter_growth`. It is a BOOL and not a regex on purpose -- the flag's argument
    looks like ``--xla_dump_hlo_pass_re``'s but is matched against the fixed dump-kind tag
    ``"mlir-fusion"``, so naming the kernel you want returns an empty level and no error. Measured
    by bisection; see the block comment in ``scopex/_parse.py``. Not free: ~400 KB of log per
    kernel.

    NOT included: ``--xla_dump_fusion_visualization``. Measured 2.08x compile time, 14.3 MB, and no
    timestamps -- the structured fusion dump supersedes it.

    ``--xla_dump_hlo_pass_re`` IS EMITTED AT MOST ONCE, AS AN ALTERNATION, AND SOMETIMES NOT AT
    ALL. Both of those are bug fixes for a silent empty dump, measured on jax 0.10.2, CPU::

        --xla_dump_to=D                                            33 files
        --xla_dump_to=D --xla_dump_hlo_pass_re=priority-fusion       0 files   <- the old default
        --xla_dump_to=D --xla_dump_hlo_pass_re=.*                   92 files
        --xla_dump_to=D --xla_dump_hlo_pass_re=priority-fusion \\
                        --xla_dump_hlo_pass_re=.*                    0 files   <- fusion + passes

    Two separate traps, and the first one fired on ``scopex.dump()`` WITH NO ARGUMENTS. A regex
    that matches no pass on this backend does not merely skip the per-pass snapshots: it
    suppresses the whole dump, including ``before_optimizations`` and the buffer assignment, which
    have nothing to do with passes. ``priority-fusion`` is a GPU pass, so on CPU the default
    produced an empty directory -- and every reader downstream (``pass_growth``, ``codegen_size``,
    ``modules_in``) then reports nothing, which reads as "this compile did nothing" rather than as
    "the flag was wrong". Giving the flag twice is the same failure by a different route.

    So: the two requests are joined into one alternation, and ``fusion=True`` contributes nothing
    on a backend that has no ``priority-fusion`` pass, where it could only ever subtract.
    """
    parts = [f"--xla_dump_to={path}"]
    res = []
    if fusion and _has_priority_fusion():
        res.append("priority-fusion")
    if passes:
        res.append(passes)
    if res:
        parts.append("--xla_dump_hlo_pass_re=" + ("|".join(f"({r})" for r in res)
                                                  if len(res) > 1 else res[0]))
    if emitter:
        parts.append(f"--xla_dump_emitter_re={_parse.EMITTER_DUMP_KIND}")
    return {"XLA_FLAGS": " ".join(parts)}


def _has_priority_fusion() -> bool:
    """Is ``priority-fusion`` a pass on the backend we are about to compile for?

    It is a GPU pass. Asked before any backend exists (the normal case, since dumping must be
    switched on first), this has to answer without initialising one -- so it reads the platform
    jax has been TOLD to use rather than the platform it has built, and errs toward True, because
    a spurious extra alternation branch costs nothing and a missing one costs the fusion log.
    """
    plat = os.environ.get("JAX_PLATFORMS", "").split(",")[0].strip().lower()
    if plat in ("cpu", "tpu"):
        return False
    if backend_initialized():
        try:
            import jax
            return jax.devices()[0].platform == "gpu"
        except Exception:                                                    # pragma: no cover
            return True
    return True


def check_env(*, warn: bool = True) -> list[str]:
    """Report environment settings that would make a measurement lie. Returns the list of problems;
    also warns unless ``warn=False``."""
    bad = []
    if os.environ.get("TF_CPP_VMODULE") and os.environ.get("TF_CPP_MIN_LOG_LEVEL", "1") != "0":
        bad.append(TRAPS["vmodule_silent"])
    try:
        import jax
        if getattr(jax.config, "jax_compilation_cache_dir", None):
            bad.append(TRAPS["persistent_cache"])
    except Exception:                                                        # pragma: no cover
        pass
    if warn:
        for b in bad:
            warnings.warn(b, RuntimeWarning, stacklevel=2)
    return bad


# ══════════════════════════════════════════════════════════════════════════════════════════════
# TURNING THE HEAVY INSTRUMENTS ON
#
# The two settings that unlock XLA's own internals behave DIFFERENTLY, and both fail silently.
# Measured on jax 0.10.2 by counting dump files and pass-log lines:
#
#   XLA_FLAGS=--xla_dump_to=DIR      set before `import jax`          -> 30 dump files
#                                    set AFTER import, before compile -> 30 dump files   OK
#                                    set after the first compile      ->  0 dump files   SILENT
#
#   TF_CPP_VMODULE=hlo_pass_pipeline=1
#                                    set before `import jax`          -> 829 log lines
#                                    set AFTER import                 ->   0 log lines   SILENT
#
# So dumping CAN be switched on from inside a running python process, as long as no compile has
# happened yet -- the flags are read when the XLA backend is first initialised. vmodule CANNOT be:
# the C++ logging layer reads it when the shared library loads, which is during `import jax`.
# That asymmetry is why `dump()` is a context manager and `pass_timings()` is a subprocess.
# ══════════════════════════════════════════════════════════════════════════════════════════════


def backend_initialized() -> bool:
    """Has XLA's backend been created yet? Once it has, XLA_FLAGS changes do nothing."""
    try:
        from jax._src import xla_bridge as xb
        return bool(getattr(xb, "_backends", {}))
    except Exception:                                                        # pragma: no cover
        return True                                                          # assume the worst


@contextlib.contextmanager
def dump(path: str | None = None, *, passes: str | None = None, fusion: bool = True,
         emitter: bool = False, keep: bool = True):
    """Compile inside this block with XLA dumping enabled, and yield the directory.

        with scopex.dump() as d:
            jax.jit(fn).lower(x).compile()
        # d now holds module_*.txt, the priority-fusion decision log, per-pass snapshots

    RAISES if the backend is already up, because setting XLA_FLAGS then is a silent no-op and a
    silent no-op is worse than an error -- you get an empty directory and conclude there was
    nothing to see. Call this before your first compile, or use a fresh process.

    AND THE EXIT IS NOT AN UNDO. The env var is restored on the way out, but the XLA backend was
    already CONSTRUCTED from it, and nothing rebuilds the backend. So every later compile in this
    process still runs under the dump flags. Measured in this project's own test suite: a module
    that opened a dump made a `call` instruction vanish from an unrelated later compile, failing a
    test that passed in isolation. If you need a compile that is not under dump flags, it has to be
    a different process -- which is why the corpus harness runs one subprocess per measurement.
    """
    if backend_initialized():
        raise RuntimeError(
            "XLA's backend is already initialised, so XLA_FLAGS would be ignored SILENTLY and this "
            "dump would be empty. Enable dumping before the first compile in the process, or run "
            "the compile in a subprocess (see scopex.pass_timings for that pattern)."
        )
    if fusion:
        try:
            # NOT `jax.devices()`. THIS IS THE BUG THIS BLOCK USED TO BE.
            #
            # `jax.devices()` CONSTRUCTS THE BACKEND. This function's first statement raises if the
            # backend is already up, because XLA reads its dump flags exactly once, when the
            # backend is built -- and three lines later the warning check was building one, before
            # XLA_FLAGS had been set. So `scopex.dump()` WITH DEFAULT ARGUMENTS returned an empty
            # directory, on every platform, and every reader downstream (`pass_growth`,
            # `codegen_size`, `modules_in`) reported nothing from it. Measured on jax 0.10.2:
            #
            #     dump(fusion=True)                 CPU  0 files      GPU  0 files
            #     dump(fusion=True, passes=".*")    CPU  0 files      GPU  0 files
            #     dump(fusion=False, passes=".*")   CPU 92 files
            #     dump(fusion=False)                CPU 33 files
            #
            # An empty dump is the single failure mode this module exists to prevent, it was the
            # DEFAULT, and it was caused by the check that warns about a lesser version of itself.
            # The platform is therefore read from the environment, which costs nothing and builds
            # nothing.
            plat = os.environ.get("JAX_PLATFORMS", "").split(",")[0].strip().lower()
            if plat and plat != "cuda" and plat != "gpu":
                warnings.warn(
                    "fusion=True requests the priority-fusion decision log, which is a GPU pass. "
                    f"On {plat} it is simply absent -- measured 77 dump files on GPU incl. "
                    "priority_fusion_dump.txt. Not an error, and scopex now leaves "
                    "--xla_dump_hlo_pass_re off entirely rather than passing a regex that matches "
                    "no pass (which suppresses the WHOLE dump), but do not read its absence as "
                    "'no fusion happened'.",
                    RuntimeWarning, stacklevel=3)
        except Exception:                                                    # pragma: no cover
            pass
    d = path or tempfile.mkdtemp(prefix="scopex-dump-")
    os.makedirs(d, exist_ok=True)
    prev = os.environ.get("XLA_FLAGS")
    os.environ["XLA_FLAGS"] = " ".join(
        ([prev] if prev else [])
        + list(dump_flags(d, fusion=fusion, passes=passes, emitter=emitter).values()))
    try:
        yield d
    finally:
        if prev is None:
            os.environ.pop("XLA_FLAGS", None)
        else:
            os.environ["XLA_FLAGS"] = prev
        if not keep:
            shutil.rmtree(d, ignore_errors=True)


# The log is TEXT and there is no other route to it, so the pattern that reads it lives in
# scopex._parse together with a verbatim sample of the lines XLA printed -- including the one that
# broke it. Recorded here because it is the reason this module does not own a regex any more:
#
#     HLO pass: async-collective-replacer time: 34 us (34 us) (cumulative: 34 us, max: 34 us, ...)
#     HLO pass: autotuner time: 1.19 min (71651421 us) (cumulative: 1.2 min, ...)
#
# XLA SWITCHES UNITS ON MAGNITUDE, which made the first version of this parser dangerous rather than
# merely incomplete: it knew us/ms/s, so a pass reported in `min` was silently dropped -- and the
# pass reported in `min` is BY CONSTRUCTION the slowest one. Measured on a GPU conv autotuning case:
# exactly 1 of 640 pass lines used `min`, it was the autotuner at 98.8% of a 72.5 s compile, and
# dropping it left `pass_timings` returning a plausible dict topped by `remat-pipeline: 0.1196`. The
# tool did not fail to answer; it reported the OPPOSITE of the truth with no warning. An unknown
# unit is now a ParseError, and a parse that reads fewer lines than the log visibly contains is too.
# An even earlier attempt matched "<word> ... <number> s" and reported the glog timestamp prefix
# `I0729` as the most expensive pass in the program.


# ══════════════════════════════════════════════════════════════════════════════════════════════
# THE DENOMINATOR, FROM THE SAME COMPILE
#
# `pass_timings` sums seconds. The only thing that makes that sum readable is the compile it is a
# fraction OF, and the only source for that is `jax.monitoring`, which lives inside the child
# process -- the one that has vmodule set. Taking the denominator from a SECOND compile in the
# parent (which is what examples/recipes/pass_timings_coverage.py had to do) makes the ratio a
# comparison of two different runs on a machine that drifts: measured 0.88, 1.04, 1.11, 0.95 and
# 1.52 for quantities that cannot exceed 1 by that mechanism. One compile, both numbers.
#
# THE CHILD MUST NOT IMPORT JAX BEFORE THE USER'S SOURCE DOES. A preamble that says `import jax` to
# get at `jax.monitoring` would freeze `JAX_PLATFORMS`, `JAX_ENABLE_X64` and every other env var
# jax reads into its config AT IMPORT -- so any `module_src` that sets one before importing jax
# (the documented way to do it) would silently stop working, and the failure would look like a
# platform mismatch rather than like scopex. So the listener is registered by a hook on
# `builtins.__import__` that fires the moment `jax` finishes initialising, whenever that is, and
# never earlier. If it never fires the child reports `registered: false` and coverage comes back
# None with a reason attached -- not a zero, and not a guess.
_SENTINEL = "__SCOPEX_COVERAGE__"

_CHILD = r'''
import atexit as _a, builtins as _b, json as _j, sys as _s, time as _t
_KEYS = {}
for _stem, _lab in %(stages)r.items():
    _KEYS["/jax/core/compile/" + _stem] = _lab
    _KEYS["/jax/core/compile/" + _stem + "_secs"] = _lab
_ACC = {"trace": 0.0, "lower": 0.0, "backend": 0.0}
_N = {"trace": 0, "lower": 0, "backend": 0}
_SEEN = set()
_ST = {"registered": False, "t0": _t.perf_counter(), "err": ""}
_REAL = _b.__import__

def _cb(_name, _value, **_kw):
    _SEEN.add(_name)
    _lab = _KEYS.get(_name)
    if _lab is not None:
        _ACC[_lab] += float(_value)
        _N[_lab] += 1

def _arm():
    if _ST["registered"]:
        return
    _jx = _s.modules.get("jax")
    if _jx is None or getattr(getattr(_jx, "__spec__", None), "_initializing", False):
        return
    _mon = getattr(_jx, "monitoring", None)
    if _mon is None:
        return
    try:
        _mon.register_event_duration_secs_listener(_cb)
    except Exception as _e:
        _ST["err"] = repr(_e)
        return
    _ST["registered"] = True
    _b.__import__ = _REAL

def _hook(_name, *_a, **_kw):
    _m = _REAL(_name, *_a, **_kw)
    _arm()
    return _m

_b.__import__ = _hook

def _emit():
    _s.stderr.write("\n" + %(sentinel)r + _j.dumps({
        "trace": _ACC["trace"], "lower": _ACC["lower"], "backend": _ACC["backend"],
        "n": _N, "wall": _t.perf_counter() - _ST["t0"],
        "registered": _ST["registered"], "err": _ST["err"], "seen": sorted(_SEEN)}) + "\n")
    _s.stderr.flush()

_a.register(_emit)
with open(%(srcpath)r) as _f:
    _code = _f.read()
_s.argv = ["-c"]
exec(compile(_code, %(srcpath)r, "exec"),
     {"__name__": "__main__", "__file__": %(srcpath)r, "__builtins__": _b, "__doc__": None})
'''


def _child_source(src_path: str) -> str:
    # One table for the metric names, imported late so this module stays jax-free at import.
    from .monitor import _STAGES
    return _CHILD % {"stages": dict(_STAGES), "sentinel": _SENTINEL, "srcpath": src_path}


def pass_timings(module_src: str, *, python: str | None = None, timeout: int = 1800,
                 vmodule: str = "hlo_pass_pipeline=1", module: str | None = None,
                 coverage: bool = True, log_dir: str | None = None) -> dict:
    """Per-XLA-pass timings, by running ``module_src`` in a FRESH SUBPROCESS with vmodule set.

    A subprocess is not laziness. ``TF_CPP_VMODULE`` is read by the C++ logging layer when the
    shared library loads, i.e. during ``import jax`` -- setting it afterwards produces exactly zero
    log lines, measured. There is no in-process route.

    ``module_src`` is python source that compiles something. It runs with a clean environment plus
    ``TF_CPP_MIN_LOG_LEVEL=0`` and ``TF_CPP_VMODULE``. Both are required: importing jax sets
    MIN_LOG_LEVEL=1, which suppresses every VLOG, so VMODULE alone is a silent no-op.

    Returns ``{"passes": {name: seconds}, "coverage": Coverage, "n_lines": int, "modules": [...],
    "stderr_tail": str}``.

    READ ``coverage`` BEFORE ``passes``. It is a :class:`scopex.Coverage`, it prints, and it carries
    the two ratios that decide whether the ranking above it means anything -- ``fidelity`` (scopex's
    arithmetic against XLA's own, which must be ~1.0) and ``coverage`` (the passes as a fraction of
    ``jax.monitoring``'s backend seconds for the SAME compile, which can be anything). A pass
    ranking read without them is exactly the artifact this instrument shipped for the arm where it
    reported the opposite of the truth. ``coverage=False`` restores the older, unchecked behaviour
    for callers that must run ``module_src`` verbatim under ``python -c``.

    ``modules`` IS PART OF THE ANSWER, not decoration. One compile logs several modules -- JAX's own
    warm-up ``jit_convert_element_type`` and friends run through the same pipelines as the program
    you asked about -- and ``passes`` sums over all of them, so a total is not "your program" unless
    that list has one entry. Measured on a two-line CPU program: 832 log lines, 384 pass lines, 3
    modules. Pass ``module="jit_your_fn"`` to keep only the pipelines XLA ran for that module.

    A ``module=`` FILTER DOES NOT NARROW THE DENOMINATOR. ``jax.monitoring`` reports one
    ``backend_compile_duration`` per compile with no module name attached, so the backend seconds
    are the child's TOTAL. With a filter set, ``coverage.coverage`` is still computed on the
    unfiltered sum -- so it keeps meaning "how much of this process's backend time was HLO passes"
    -- and the filtered figure is reported separately as ``returned_seconds``.
    """
    env = dict(os.environ)
    env.update(vmodule_env(vmodule))
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    tmp = None
    try:
        if coverage:
            fd, tmp = tempfile.mkstemp(prefix="scopex-src-", suffix=".py")
            with os.fdopen(fd, "w") as f:
                f.write(module_src)
            argv = [python or sys.executable, "-c", _child_source(tmp)]
        else:
            argv = [python or sys.executable, "-c", module_src]
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
    finally:
        if tmp:
            os.unlink(tmp)

    log = p.stderr + p.stdout
    child, why = _read_sentinel(log, p)
    # The sentinel is scopex's own line and must not be counted as, or parsed as, compiler output.
    log = "\n".join(ln for ln in log.splitlines() if not ln.startswith(_SENTINEL))

    if module:
        # Keep only the stretches of log between a header naming this module and the next header.
        keep, on = [], False
        for line in log.splitlines():
            hdr = _parse.pass_pipeline_headers(line)
            if hdr:
                on = module in hdr[0][0]
            if on:
                keep.append(line)
        log_used = "\n".join(keep)
    else:
        log_used = log

    every = _parse.pass_timing_lines(log)        # all modules -- the cross-check's numerator
    out: dict[str, float] = {}
    for t in _parse.pass_timing_lines(log_used) if module else every:
        out[t.name] = out.get(t.name, 0.0) + t.seconds

    # Leaves vs pipeline aggregates. The naive sum double-counts by up to 1.87x (measured), so the
    # fraction-of-the-compile number uses the leaves -- and the split is only trustworthy because
    # the two halves must add back up to XLA's own cumulative, which is asserted below.
    split = _parse.pass_leaf_split(log)
    leaf_s = sum(t.seconds for t in split.leaves)
    agg_s = sum(t.seconds for t in split.aggregates)

    tot = _parse.pass_log_totals(log)
    cov = Coverage(
        parsed_passes=len(every),
        parsed_seconds=sum(t.seconds for t in every),
        parsed_max_s=max((t.seconds for t in every), default=0.0),
        leaf_seconds=leaf_s,
        aggregate_seconds=agg_s,
        n_leaves=len(split.leaves),
        n_aggregates=len(split.aggregates),
        unmatched_pipelines=split.unmatched_closes,
        log_threads=split.threads,
        xla_pass_count=tot["n_called"],
        xla_cumulative_s=tot["cumulative_s"],
        xla_max_pass_s=tot["max_pass_s"],
        tolerance=tot["tolerance"],
        counter_monotone=tot["monotone"],
        backend_s=None if child is None else child["backend"],
        trace_s=None if child is None else child["trace"],
        lower_s=None if child is None else child["lower"],
        child_wall_s=None if child is None else child["wall"],
        n_backend_compiles=0 if child is None else child["n"]["backend"],
        metrics_seen=[] if child is None else child["seen"],
        why_no_backend=why,
        returned_seconds=sum(out.values()),
        module_filter=module,
    )
    # THE LOG ITSELF, not a 1500-character tail of it. Every number above is derived from this text
    # and nothing else, so a reader who doubts the ranking should be able to grep rather than
    # rerun a compile -- see scopex/raw.py for why this is a path and not the string.
    from .raw import raw_of
    d = log_dir or tempfile.mkdtemp(prefix="scopex-vlog-")
    os.makedirs(d, exist_ok=True)
    log_path = os.path.join(d, "hlo_pass_pipeline.vlog")
    with open(log_path, "w") as f:
        f.write(log)
    raw = raw_of(log_path, "vlog", produced_by="xla/hlo/pass/hlo_pass_pipeline.cc:176 (VLOG(1))",
                 witness=r"HLO pass:\s", parsed_count=len(every), text=log)

    return {"passes": dict(sorted(out.items(), key=lambda kv: -kv[1])),
            "coverage": cov,
            "raw": raw,
            "n_lines": len(log.splitlines()),
            "modules": sorted({m for m, _ in _parse.pass_pipeline_headers(log)}),
            "module_filter": module,      # None => `passes` sums over every module in `modules`
            "unknown_units": [],          # unconvertible units now raise; kept for shape stability
            "stderr_tail": p.stderr[-1500:] if not out else ""}


def _read_sentinel(log: str, p) -> tuple[dict | None, str]:
    """The child's jax.monitoring numbers, or ``None`` and the reason there are none.

    A reason and not a zero. Every way this can fail -- the child died, it never imported jax, jax
    renamed the metrics -- produces a DIFFERENT sentence, because the responses differ and because
    a coverage of 0.0 that means "we could not measure" is the same shape of lie as a pass ranking
    that means "we could not parse".
    """
    hits = [ln for ln in log.splitlines() if ln.startswith(_SENTINEL)]
    if not hits:
        if p.returncode != 0:
            return None, (f"the child exited {p.returncode} before scopex could read its metrics "
                          f"(stderr tail: {p.stderr[-300:].strip()!r})")
        return None, ("the child printed no scopex sentinel -- it may have called os._exit(), been "
                      "killed, or replaced sys.stderr")
    try:
        d = json.loads(hits[-1][len(_SENTINEL):])
    except Exception as e:                                                   # pragma: no cover
        return None, f"the child's sentinel line did not parse: {e!r}"
    if not d.get("registered"):
        died = (f", and it exited {p.returncode}: {p.stderr[-300:].strip()!r}"
                if p.returncode != 0 else "")
        return None, ("the child never finished importing jax, so no jax.monitoring listener was "
                      "ever armed" + (f" ({d['err']})" if d.get("err") else "") + died)
    if d["backend"] <= 0.0:
        return None, (f"jax.monitoring emitted no backend_compile_duration -- either nothing was "
                      f"compiled, or the metric names moved. The child saw: {d['seen']}")
    return d, ""
