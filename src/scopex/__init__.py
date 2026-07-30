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
from .artifacts import (codegen_size, custom_calls, diverge, modules_in, pass_growth,
                        pass_timeline, selftest)
from .flags import (backend_initialized, check_env, dump, dump_flags, hlo_text, pass_timings,
                    stablehlo_text, vmodule_env)
from .levels import (frame_tables, hlo_instructions, hlo_module, metadata, stablehlo_module,
                     walk_hlo, walk_stablehlo)
from .mark import USER, mark_callable, mark_framework, named_scope, parse, scope
from .monitor import Timings, record, regime
from .records import LEVELS, Eqn, Ins
from .views import BY, attribute, crosstab, table
from .walk import at, subjaxprs, verify_parity, walk

# ── NOT EXPORTED, DELIBERATELY ───────────────────────────────────────────────────────────────────
# `scopex.phases` (backend_split), `scopex.tracing` (trace_profile), `scopex.sharing`
# (jaxpr_sharing), `scopex.emitters` (emitter_growth), `scopex.fusion` (fusion_steps), and
# `levels.opcode_delta` / `pre_optimization_hlo` are reachable as
#
#     from scopex.phases import backend_split
#
# and are used by files under examples/recipes/. They are not top-level API, and the distinction is
# the point rather than an oversight.
#
# Each was asked for by one or two of the thirty investigations that produced this package. Only
# two things cleared three -- per-pass growth (8) and codegen size (10) -- and those are in
# `artifacts`. Everything in `__init__` is a promise: documented, tested, and kept working across
# jax releases. That is worth paying for a view you reach for constantly and not for one that
# answered a single question well.
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
    # reading what a compile left behind
    "pass_growth", "pass_timeline", "diverge", "codegen_size", "custom_calls", "modules_in",
    # self-checks -- run after any jax upgrade
    "selftest", "conformance", "ParseError",
    "__version__",
]
