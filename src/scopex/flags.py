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

import os
import warnings

__all__ = ["stablehlo_text", "hlo_text", "check_env", "dump_flags", "vmodule_env", "TRAPS"]

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
