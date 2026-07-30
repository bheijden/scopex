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
import os
import shutil
import subprocess
import sys
import tempfile
import warnings

from . import _parse

__all__ = ["stablehlo_text", "hlo_text", "check_env", "dump_flags", "vmodule_env",
           "backend_initialized", "dump", "pass_timings", "TRAPS"]

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
    """
    parts = [f"--xla_dump_to={path}"]
    if fusion:
        parts.append("--xla_dump_hlo_pass_re=priority-fusion")
    if passes:
        parts.append(f"--xla_dump_hlo_pass_re={passes}")
    if emitter:
        parts.append(f"--xla_dump_emitter_re={_parse.EMITTER_DUMP_KIND}")
    return {"XLA_FLAGS": " ".join(parts)}


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
            import jax
            if jax.devices()[0].platform != "gpu":
                warnings.warn(
                    "fusion=True requests the priority-fusion decision log, which is a GPU pass. "
                    f"On {jax.devices()[0].platform} it is simply absent -- measured 77 dump files "
                    "on GPU incl. priority_fusion_dump.txt, 27 on CPU without it. Not an error, "
                    "but do not read its absence as 'no fusion happened'.",
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


def pass_timings(module_src: str, *, python: str | None = None, timeout: int = 1800,
                 vmodule: str = "hlo_pass_pipeline=1", module: str | None = None) -> dict:
    """Per-XLA-pass timings, by running ``module_src`` in a FRESH SUBPROCESS with vmodule set.

    A subprocess is not laziness. ``TF_CPP_VMODULE`` is read by the C++ logging layer when the
    shared library loads, i.e. during ``import jax`` -- setting it afterwards produces exactly zero
    log lines, measured. There is no in-process route.

    ``module_src`` is python source that compiles something. It runs with a clean environment plus
    ``TF_CPP_MIN_LOG_LEVEL=0`` and ``TF_CPP_VMODULE``. Both are required: importing jax sets
    MIN_LOG_LEVEL=1, which suppresses every VLOG, so VMODULE alone is a silent no-op.

    Returns ``{"passes": {name: seconds}, "n_lines": int, "modules": [...], "stderr_tail": str}``.

    ``modules`` IS PART OF THE ANSWER, not decoration. One compile logs several modules -- JAX's own
    warm-up ``jit_convert_element_type`` and friends run through the same pipelines as the program
    you asked about -- and ``passes`` sums over all of them, so a total is not "your program" unless
    that list has one entry. Measured on a two-line CPU program: 832 log lines, 807 pass lines, 3
    modules. Pass ``module="jit_your_fn"`` to keep only the pipelines XLA ran for that module.
    """
    env = dict(os.environ)
    env.update(vmodule_env(vmodule))
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    p = subprocess.run([python or sys.executable, "-c", module_src],
                       capture_output=True, text=True, timeout=timeout, env=env)
    log = p.stderr + p.stdout
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
    out: dict[str, float] = {}
    for t in _parse.pass_timing_lines(log_used):
        out[t.name] = out.get(t.name, 0.0) + t.seconds
    return {"passes": dict(sorted(out.items(), key=lambda kv: -kv[1])),
            "n_lines": len(log.splitlines()),
            "modules": sorted({m for m, _ in _parse.pass_pipeline_headers(log)}),
            "module_filter": module,      # None => `passes` sums over every module in `modules`
            "unknown_units": [],          # unconvertible units now raise; kept for shape stability
            "stderr_tail": p.stderr[-1500:] if not out else ""}
