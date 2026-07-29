"""The annotation contract: how a framework marks the library/user split.

The contract is a NAMING CONVENTION, not an API. A framework that wants its users' code
distinguishable from its own calls stock ``jax.named_scope`` with strings in the shape below and
depends on nothing in this package. ``scopex`` recognises the shape when it walks the program.

That direction matters. If the contract were an API, every framework that adopted it would take a
runtime dependency on a profiler, and the two would have to stay version-locked forever. A
convention costs the framework one import it already has.

THE SHAPE
---------
::

    <pkg>:<role>
    <pkg>:<role>.<detail>

``pkg``    the framework's distribution name, e.g. ``dflux``. Namespaces the mark so two
           frameworks marking in the same program cannot collide.
``role``   ``lib`` for the framework's own code, ``user`` for code its users wrote.
``detail`` free-form, dotted. Conventionally ``Class.method`` for user code and a subsystem path
           for library code.

Examples::

    dflux:lib.solve
    dflux:user.MyColumn.residual
    flax:user.MyModule.__call__

THE ONE HARD RULE: NO ``/`` INSIDE A NAME
-----------------------------------------
JAX renders a name stack by joining entries with ``/``. Measured on jax 0.10.2: two nested scopes
``"a"`` then ``"b"`` and a single scope ``"a/b"`` BOTH render to ``'a/b'``. At the jaxpr level the
structured ``name_stack.stack`` still distinguishes them, but below the jaxpr only the rendered
string survives, so a ``/`` inside a name is permanently unrecoverable there.

Everything else measured safe and verbatim through to optimized HLO: ``:`` ``.`` ``-`` ``_``
spaces, brackets, parens, mixed case, digits, and non-ASCII.

COST
----
Measured on jax 0.10.2, 60-leaf trace, 21 rounds, order-rotated and paired within round: one scope
per leaf is 1.029x plain, two nested scopes per leaf 1.007x. The per-round range spans 0.485-1.157,
so the effect is not resolvable above noise at this scale. Treat marking as free at trace time and
measure again before marking something entered millions of times.
"""

from __future__ import annotations

import contextlib
import functools
from typing import Iterable

import jax

__all__ = [
    "LIB", "USER", "scope", "named_scope", "parse", "validate",
    "mark_methods", "mark_subclasses", "InvalidScopeName",
]

SEP = ":"
DOT = "."
NEST = "/"

LIB = "lib"
USER = "user"


class InvalidScopeName(ValueError):
    """Raised for a scope name that could not be parsed back out of a rendered name stack."""


def validate(name: str) -> str:
    """Reject names that cannot survive the round trip. Fail loudly at mark time rather than
    silently produce an unparseable program."""
    if not name:
        raise InvalidScopeName("scope name is empty")
    if NEST in name:
        raise InvalidScopeName(
            f"{name!r} contains {NEST!r}, which JAX uses to join name-stack entries. Below the "
            f"jaxpr a name containing {NEST!r} is indistinguishable from two nested scopes. Use "
            f"{DOT!r} to separate parts of a name."
        )
    return name


def scope(pkg: str, role: str, detail: str = "") -> str:
    """Build a contract-shaped scope string. ``scope('dflux', USER, 'MyColumn.residual')``."""
    if role not in (LIB, USER):
        raise InvalidScopeName(f"role must be {LIB!r} or {USER!r}, got {role!r}")
    for part in (pkg, role, detail):
        if part:
            validate(part)
    return f"{pkg}{SEP}{role}" + (f"{DOT}{detail}" if detail else "")


@contextlib.contextmanager
def named_scope(pkg: str, role: str, detail: str = ""):
    """Sugar over ``jax.named_scope``. Entirely optional -- a framework that would rather not
    import scopex writes ``jax.named_scope("dflux:user.MyColumn.residual")`` and gets the identical
    result. This exists so the validation runs."""
    with jax.named_scope(scope(pkg, role, detail)):
        yield


def parse(name: str) -> tuple[str, str, str] | None:
    """``'dflux:user.MyColumn.residual'`` -> ``('dflux', 'user', 'MyColumn.residual')``.

    Returns None for anything not contract-shaped, which is most scopes in most programs -- an
    unmarked scope is not an error, it just carries no authorship."""
    if SEP not in name:
        return None
    pkg, _, rest = name.partition(SEP)
    role, _, detail = rest.partition(DOT)
    if role not in (LIB, USER) or not pkg:
        return None
    return pkg, role, detail


# ── applying the mark ────────────────────────────────────────────────────────────────────────────
# Two helpers. Both are optional sugar: a framework can hand-write the `with jax.named_scope(...)`
# blocks and scopex will read them identically.

def mark_methods(cls, pkg: str, role: str, methods: Iterable[str]):
    """Wrap named methods of `cls` in a contract-shaped scope, in place. Idempotent."""
    for m in methods:
        fn = cls.__dict__.get(m)
        if fn is None or not callable(fn) or getattr(fn, "_scopex_marked", False):
            continue
        tag = scope(pkg, role, f"{cls.__qualname__}{DOT}{m}")

        def wrap(fn=fn, tag=tag):
            @functools.wraps(fn)
            def marked(*a, **kw):
                with jax.named_scope(tag):
                    return fn(*a, **kw)
            marked._scopex_marked = True
            return marked

        setattr(cls, m, wrap())
    return cls


def mark_subclasses(pkg: str, methods: Iterable[str]):
    """Return an ``__init_subclass__`` that marks FOREIGN subclasses as user code.

    A subclass is foreign when its top-level module package differs from ``pkg``. That is a
    package-identity test, deliberately not a file-path test: file layout changes with refactors,
    package identity does not.

    Usage in a framework's base class::

        class Block:
            __init_subclass__ = classmethod(
                scopex.mark_subclasses("dflux", ("residual", "cell", "operator")))

    The equivalent with no scopex dependency is ~15 lines of ``jax.named_scope`` in the framework's
    own ``__init_subclass__``; see the README.

    LIMIT: this only reaches frameworks whose extension point is subclassing. A library of plain
    functions (optax-style) has no class to hook and must mark its own entry points directly, or
    rely on the unmarked fallback (traceback package attribution).
    """
    def __init_subclass__(cls, **kw):
        super(cls, cls).__init_subclass__(**kw)
        if (getattr(cls, "__module__", "") or "").split(DOT)[0] != pkg:
            mark_methods(cls, pkg, USER, methods)
    return __init_subclass__
