"""scopex -- attribute a jitted JAX program's compilation artifacts back to the code that wrote it.

Three things, in the order you normally want them::

    import scopex

    print(scopex.record(fn, x))              # 1. which STAGE is slow: trace / lower / backend
    units = list(scopex.walk(jaxpr))         # 2. every equation, with complete provenance
    print(scopex.table(scopex.attribute(units, "site")))   # 3. collapsed however you like

scopex reads what JAX and XLA already emit. It has one dependency, ``jax``, and it asks nothing of
the libraries whose programs you are profiling: a framework that wants its users' code
distinguishable from its own calls stock ``jax.named_scope`` with strings in a documented shape and
never imports scopex at all. See :mod:`scopex.mark`.
"""

from __future__ import annotations

from ._parse import ParseError, conformance
from .artifacts import (
                        codegen_size,
                        custom_calls,
                        diverge,
                        modules_in,
                        pass_conservation,
                        pass_growth,
                        selftest,
)
from .coverage import Coverage
from .flags import (
                        backend_initialized,
                        check_env,
                        dump,
                        dump_flags,
                        hlo_text,
                        pass_timings,
                        stablehlo_text,
                        vmodule_env,
)
from .levels import (
                        frame_tables,
                        hlo_instructions,
                        hlo_module,
                        metadata,
                        stablehlo_module,
                        walk_hlo,
                        walk_stablehlo,
)
from .mark import USER, mark_callable, mark_framework, named_scope, parse, scope
from .monitor import Timings, record, regime
from .raw import Raw, raw_step
from .records import LEVELS, Eqn, Ins
from .timeline import pass_timeline, timeline_agreement
from .views import BY, attribute, crosstab, table
from .walk import at, subjaxprs, verify_parity, walk

# ── NOT EXPORTED, DELIBERATELY ───────────────────────────────────────────────────────────────────
# `scopex.phases` (backend_split), `scopex.tracing` (trace_profile), `scopex.sharing`
# (jaxpr_sharing), `scopex.emitters` (emitter_growth), `scopex.fusion` (fusion_steps,
# fusion_consistency), `scopex.autotune` (autotune_cost), `scopex.passmap` (pass_source,
# pass_sources, pipelines_in, cross_check, verify_pass_map), `artifacts.boundary_diff` /
# `opcode_delta` / `resolve_boundary` / `boundaries_in`, and `levels.pre_optimization_hlo` are
# reachable as
#
#     from scopex.phases import backend_split
#     from scopex.passmap import pass_source
#     from scopex.artifacts import boundary_diff
#
# and are used by files under examples/recipes/. They are not top-level API, and the distinction is
# the point rather than an oversight.
#
# THE BAR, and it did not move for this round of work. Each of the above was asked for by one or two
# of the thirty investigations that produced this package. Only two things cleared three -- per-pass
# growth (8) and codegen size (10). Something new enters `__all__` only if it is a SAFETY MECHANISM
# for a name already there, or the VALIDATED VERSION of one. Five did, and each one is checking
# something already exported rather than answering a new question:
#
#   Coverage            checks `pass_timings`' ranking against XLA's own arithmetic
#   pass_conservation   checks `pass_growth` / `diverge`'s curves against the endpoint files
#   Raw, raw_step       hand back the bytes any of the above was parsed from, hashed
#   timeline_agreement  the only way to get a `pass_timeline` that has been checked at all
#
# and `pass_timeline` itself was REPLACED in place by the validated implementation in
# `scopex.timeline` rather than added beside it.
#
# THREE THINGS WERE BUILT, VALIDATED, AND STILL NOT PROMOTED, which is the bar working:
#
# * `scopex.passmap` -- a pass name -> the XLA file that implements it, 213 rows, generated from
#   the exact tarball jaxlib 0.10.2 was built from and checkable against a checkout
#   (`verify_pass_map`) and against the log's own nesting (`cross_check`). Fifteen of the thirty
#   investigations needed it, which is a strong case for a name -- but a top-level name is a promise
#   to keep working across jax releases and this table is pinned to ONE XLA commit. It ships checked
#   and unpromoted, with `XLA_COMMIT` / `BUILT_FOR` in the data so a caller can tell.
# * `artifacts.boundary_diff` -- the "what changed at this boundary" report. It answers a NEW
#   question rather than checking an old answer, and it sits beside `opcode_delta`, which was held
#   at exactly this level for exactly this reason.
# * `scopex.autotune` -- GPU-only, and it answers a question `pass_timings` cannot: `autotuner` is
#   ~98% of the compile on BOTH `convT64_dilate16` and `gemm_shapes_k16`, and on one the seconds are
#   cuDNN kernels on real buffers while on the other they are Triton candidates being compiled. It
#   rests on `--xla_gpu_dump_autotune_*` and one guessed proto field number. Checked, not promoted.
#
# A recipe is the better home for those anyway. It shows the whole procedure, which is what someone
# profiling their own program needs; an API hides it behind a name. If one of them starts appearing
# in three unrelated recipes, promote it then -- with the recipes as the evidence.

__version__ = "0.1.0"

__all__ = [
    # stage split -- start here
    "record", "Timings", "regime",
    # traversal and records
    "walk", "at", "subjaxprs", "verify_parity", "Eqn", "Ins", "LEVELS",
    "walk_hlo", "walk_stablehlo", "hlo_instructions", "metadata", "frame_tables",
    # views
    "attribute", "crosstab", "table", "BY",
    # the marking contract
    "scope", "named_scope", "mark_framework", "mark_callable", "parse", "USER",
    # getting text/IR out without hitting a silent trap
    "stablehlo_text", "hlo_text", "stablehlo_module", "hlo_module",
    "dump", "dump_flags", "vmodule_env", "pass_timings", "check_env", "backend_initialized",
    # what fraction of its parent a timing view explains -- read this BEFORE the ranking above it
    "Coverage",
    # reading what a compile left behind
    "pass_growth", "pass_timeline", "diverge", "codegen_size", "custom_calls", "modules_in",
    # ... and the checks that say whether those are describing your compile
    "pass_conservation", "timeline_agreement",
    # show your working: the text every number above was parsed from
    "Raw", "raw_step",
    # self-checks -- run after any jax upgrade
    "selftest", "conformance", "ParseError",
    "__version__",
]
