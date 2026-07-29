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

from .flags import check_env, dump_flags, hlo_text, stablehlo_text, vmodule_env
from .mark import LIB, USER, mark_methods, mark_subclasses, named_scope, parse, scope
from .monitor import Timings, record, regime
from .record import LEVELS, Eqn, Ins
from .views import BY, attribute, crosstab, table
from .walk import at, subjaxprs, verify_parity, walk

__version__ = "0.1.0"

__all__ = [
    # stage split -- start here
    "record", "Timings", "regime",
    # traversal and records
    "walk", "at", "subjaxprs", "verify_parity", "Eqn", "Ins", "LEVELS",
    # views
    "attribute", "crosstab", "table", "BY",
    # the marking contract
    "scope", "named_scope", "mark_methods", "mark_subclasses", "parse", "LIB", "USER",
    # getting text out without hitting a silent trap
    "stablehlo_text", "hlo_text", "dump_flags", "vmodule_env", "check_env",
    "__version__",
]
