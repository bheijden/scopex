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

from .levels import hlo_instructions, walk_hlo, walk_stablehlo
from .artifacts import (codegen_size, custom_calls, diverge, modules_in, pass_growth,
                        pass_timeline, selftest)
from .flags import (backend_initialized, check_env, dump, dump_flags, hlo_text,
                    pass_timings, stablehlo_text, vmodule_env)
from .mark import LIB, USER, mark_callable, mark_framework, mark_methods, named_scope, parse, scope
from .monitor import Timings, record, regime
from .records import LEVELS, Eqn, Ins
from .views import BY, attribute, crosstab, table
from .walk import at, subjaxprs, verify_parity, walk

__version__ = "0.1.0"

__all__ = [
    # stage split -- start here
    "record", "Timings", "regime",
    # traversal and records
    "walk", "at", "subjaxprs", "verify_parity", "Eqn", "Ins", "LEVELS",
    "walk_hlo", "walk_stablehlo", "hlo_instructions",
    # views
    "attribute", "crosstab", "table", "BY",
    # the marking contract
    "scope", "named_scope", "mark_methods", "mark_framework", "mark_callable", "parse", "LIB", "USER",
    # getting text out without hitting a silent trap
    "stablehlo_text", "hlo_text", "dump_flags", "vmodule_env", "check_env",
    "dump", "pass_timings", "backend_initialized",
    # reading what a compile left behind
    "pass_growth", "pass_timeline", "diverge", "codegen_size", "custom_calls",
    "modules_in", "selftest",
    "__version__",
]
