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
``Lowered.compiler_ir('stablehlo')``              0      TRAP -- drops location info
``Lowered.compiler_ir('hlo')``                    0      TRAP -- drops metadata
``Compiled.as_text()``                            9      correct
``executable.get_hlo_text()``                     9      correct
``executable.hlo_modules()[0].to_string()``       9      correct
===========================================  ==========  =========================================

The ``compiler_ir`` rows are the dangerous ones: it is the accessor that *looks* structured and
principled, so a reader trusts its empty answer and concludes the IR carries no provenance.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings

__all__ = ["stablehlo_text", "hlo_text", "check_env", "dump_flags", "vmodule_env",
           "backend_initialized", "dump", "pass_timings", "TRAPS"]

TRAPS = {
    "lowered_as_text_default": (
        "Lowered.as_text() defaults to debug_info=False and prints no locations. "
        "Use scopex.stablehlo_text(lowered), or pass debug_info=True."),
    "compiler_ir": (
        "Lowered.compiler_ir('stablehlo'|'hlo') drops location/metadata entirely. "
        "It is not a structured alternative to as_text(debug_info=True); it is a lossy one."),
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


def dump_flags(path: str, *, fusion: bool = True, passes: str | None = None) -> dict[str, str]:
    """XLA_FLAGS for dumping compiler artifacts.

    ``fusion`` adds the priority-fusion decision dump (free: it is written during the pass that
    already runs). ``passes`` is a regex of pass names to snapshot HLO around; ``".*"`` snapshots
    every pass and is large.

    NOT included: ``--xla_dump_fusion_visualization``. Measured 2.08x compile time, 14.3 MB, and no
    timestamps -- the structured fusion dump supersedes it.
    """
    parts = [f"--xla_dump_to={path}"]
    if fusion:
        parts.append("--xla_dump_hlo_pass_re=priority-fusion")
    if passes:
        parts.append(f"--xla_dump_hlo_pass_re={passes}")
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
         keep: bool = True):
    """Compile inside this block with XLA dumping enabled, and yield the directory.

        with scopex.dump() as d:
            jax.jit(fn).lower(x).compile()
        # d now holds module_*.txt, the priority-fusion decision log, per-pass snapshots

    RAISES if the backend is already up, because setting XLA_FLAGS then is a silent no-op and a
    silent no-op is worse than an error -- you get an empty directory and conclude there was
    nothing to see. Call this before your first compile, or use a fresh process.
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
        ([prev] if prev else []) + list(dump_flags(d, fusion=fusion, passes=passes).values()))
    try:
        yield d
    finally:
        if prev is None:
            os.environ.pop("XLA_FLAGS", None)
        else:
            os.environ["XLA_FLAGS"] = prev
        if not keep:
            shutil.rmtree(d, ignore_errors=True)


# The line XLA actually prints (hlo_pass_pipeline.cc:176), verified by reading the log rather than
# guessing at it:
#     HLO pass: async-collective-replacer time: 24 us (24 us) (cumulative: 24 us, max: 24 us, ...)
# A first attempt matched "<word> ... <number> s" and dutifully reported the glog timestamp prefix
# `I0729` as the most expensive pass in the program. This format is NOT a stable interface --
# `n_lines` and `stderr_tail` exist so a parse failure is visible instead of returning {}.
_PASS_LINE = re.compile(
    r"HLO pass:\s+(?P<name>\S+)\s+time:\s+(?P<val>[\d.]+)\s*(?P<unit>us|ms|s)\b")
_UNIT = {"us": 1e-6, "ms": 1e-3, "s": 1.0}


def pass_timings(module_src: str, *, python: str | None = None, timeout: int = 1800,
                 vmodule: str = "hlo_pass_pipeline=1") -> dict:
    """Per-XLA-pass timings, by running ``module_src`` in a FRESH SUBPROCESS with vmodule set.

    A subprocess is not laziness. ``TF_CPP_VMODULE`` is read by the C++ logging layer when the
    shared library loads, i.e. during ``import jax`` -- setting it afterwards produces exactly zero
    log lines, measured. There is no in-process route.

    ``module_src`` is python source that compiles something. It runs with a clean environment plus
    ``TF_CPP_MIN_LOG_LEVEL=0`` and ``TF_CPP_VMODULE``. Both are required: importing jax sets
    MIN_LOG_LEVEL=1, which suppresses every VLOG, so VMODULE alone is a silent no-op.

    Returns ``{"passes": {name: seconds}, "n_lines": int, "stderr_tail": str}``. An empty ``passes``
    with a non-empty ``stderr_tail`` usually means the vmodule spec did not match this XLA build --
    the log format is not a stable interface and this parser is best-effort by nature.
    """
    env = dict(os.environ)
    env.update(vmodule_env(vmodule))
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    p = subprocess.run([python or sys.executable, "-c", module_src],
                       capture_output=True, text=True, timeout=timeout, env=env)
    log = p.stderr + p.stdout
    out: dict[str, float] = {}
    for m in _PASS_LINE.finditer(log):
        out[m.group("name")] = out.get(m.group("name"), 0.0) + \
            float(m.group("val")) * _UNIT[m.group("unit")]
    return {"passes": dict(sorted(out.items(), key=lambda kv: -kv[1])),
            "n_lines": len(log.splitlines()),
            "stderr_tail": p.stderr[-1500:] if not out else ""}
