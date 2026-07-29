"""The lossless unit record.

One rule governs this module: **every accessor returns the FULL ordered sequence, never a pick.**

That is not fastidiousness. An earlier version returned the first match for authorship, so an
equation traced through a user's ``cell`` called from inside their ``residual`` was credited to
``residual`` alone -- and 4,567 equations were reported as 1,168. The interesting question about a
compiled program is almost always an INTERACTION (one line of user code reached through several
library phases through several autodiff transforms), and a record that keeps only the first of each
has destroyed the answer before any view runs.

Collapsing is the caller's job. See :mod:`scopex.views`.
"""

from __future__ import annotations

from typing import NamedTuple

from .mark import parse as _parse_mark

__all__ = ["Eqn", "Ins", "LEVELS", "TRANSFORMS"]

LEVELS = ("jaxpr", "stablehlo", "hlo", "hlo_opt")

# jax builds NameStack `Transform` entries with exactly these three names (jax 0.10.2; verified by
# grepping `transform_name_stack` across the jax tree). Anything else in a name stack is a Scope.
TRANSFORMS = ("vmap", "jvp", "transpose")


def _split_nesting(s: str) -> list[str]:
    """Split a rendered name stack on '/' at parenthesis depth 0.

    Depth matters: jax renders a transform as ``vmap(inner/path)``, so a naive ``s.split('/')``
    shatters one transform entry into pieces."""
    out, buf, d = [], [], 0
    for ch in s:
        if ch == "(":
            d += 1
        elif ch == ")":
            d -= 1
        if ch == "/" and d == 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    out.append("".join(buf))
    return [x for x in out if x]


def _peel(seg: str) -> tuple[tuple[str, ...], str]:
    """``'vmap(jvp(X))'`` -> ``(('vmap','jvp'), 'X')``.

    Only the three real transform names peel. ``'jit(solve_fn)'`` stays whole, because an inlined
    callee's name legitimately looks like a call."""
    xf: list[str] = []
    while True:
        i = seg.find("(")
        if i > 0 and seg.endswith(")") and seg[:i] in TRANSFORMS:
            xf.append(seg[:i])
            seg = seg[i + 1:-1]
        else:
            return tuple(xf), seg


class _Unit:
    """Shared accessors. Subclasses supply ``path`` (the rendered name stack) and ``site``."""

    __slots__ = ()

    # ── the three orthogonal readings of the name stack ──────────────────────────────────────────
    @property
    def entries(self) -> tuple[str, ...]:
        """Nesting entries, outermost first, transforms still attached."""
        return tuple(_split_nesting(self.path or ""))

    @property
    def scopes(self) -> tuple[str, ...]:
        """Every scope name, outermost first, with transform wrappers peeled off."""
        return tuple(_peel(e)[1] for e in self.entries)

    @property
    def transforms(self) -> tuple[str, ...]:
        """Every autodiff/batching transform on the path, outermost first, WITH repeats.

        ``vmap`` twice is not the same as ``vmap`` once, so this is a sequence and not a set."""
        out: list[str] = []
        for e in self.entries:
            out.extend(_peel(e)[0])
        return tuple(out)

    # ── the contract (see scopex.mark) ───────────────────────────────────────────────────────────
    @property
    def marks(self) -> tuple[tuple[str, str, str], ...]:
        """Every contract-shaped scope on the path, outermost first, as ``(pkg, role, detail)``.

        Full sequence on purpose: a unit inside ``mylib:user.Col.cell`` inside
        ``mylib:lib.solve`` inside ``mylib:user.Col.residual`` has three, and which one you want
        depends on the question."""
        out = []
        for s in self.scopes:
            m = _parse_mark(s)
            if m is not None:
                out.append(m)
        return tuple(out)

    @property
    def packages(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(p for p, _, _ in self.marks))

    @property
    def authored(self) -> tuple[str, ...]:
        """The ``user``-role details, outermost first. Empty for library or unmarked code."""
        return tuple(d or f"{p}:user" for p, r, d in self.marks if r == "user")

    @property
    def library(self) -> tuple[str, ...]:
        return tuple(d or f"{p}:lib" for p, r, d in self.marks if r == "lib")

    @property
    def role(self) -> str:
        """The INNERMOST role, or ``'<unmarked>'``. The one deliberately lossy accessor, because
        'who most directly wrote this' is a genuinely common question -- but ``marks`` is the
        truthful answer and every aggregate in :mod:`scopex.views` uses that."""
        return self.marks[-1][1] if self.marks else "<unmarked>"


class Eqn(NamedTuple):
    """One jaxpr equation with its complete provenance.

    A NamedTuple cannot take a mixin base, so the shared accessors are attached below -- see the
    loop at the bottom of the module, which gives `Eqn` and `Ins` literally the same property
    objects rather than two implementations that have to be kept in step."""

    eqn: object                # the jax equation
    path: str                  # rendered name stack: caller's, concatenated with its own
    depth: int                 # nesting depth; 0 = top-level jaxpr
    addr: tuple                # e.g. (3, 'jaxpr', 17): `at(jaxpr, addr) is eqn`, exactly
    site: str                  # 'file:line', or a '<bucket>' -- never silently inherited
    frame: object = None       # resolved user frame; None <=> jax had no frame for this equation

    level = "jaxpr"

    @property
    def kind(self) -> str:
        """Level-neutral name for what this unit IS: primitive here, mlir op or HLO opcode below."""
        return str(self.eqn.primitive)

    primitive = kind


class Ins:
    """One unit BELOW the jaxpr: a StableHLO operation or an HLO instruction.

    Same accessor names as :class:`Eqn` on purpose, so a view written against one level runs
    unchanged on the others. What differs is what each level can honestly answer, and that is
    declared in :mod:`scopex.views`, not silently degraded here.
    """

    __slots__ = ("level", "kind", "path", "unit", "container", "site", "loc",
                 "function", "depth", "fusion", "outlined")

    def __init__(self, level, kind, path, *, unit="", container="", site="?", loc=None,
                 function="?", depth=-1, fusion=False, outlined=False):
        self.level, self.kind, self.path = level, kind, path
        self.unit, self.container = unit, container
        self.site, self.loc = site, (site if loc is None else loc)
        self.function, self.depth = function, depth
        self.fusion, self.outlined = fusion, outlined

    def __repr__(self):
        return f"Ins({self.level} {self.kind} {self.unit or ''} {self.site} {self.path!r})"


# Both records get the SAME property objects, so the two levels cannot drift apart.
for _n in ("entries", "scopes", "transforms", "marks", "packages", "authored", "library", "role"):
    _p = getattr(_Unit, _n)
    setattr(Ins, _n, _p)
    setattr(Eqn, _n, _p)
