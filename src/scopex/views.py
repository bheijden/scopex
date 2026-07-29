"""Views: every way of collapsing the lossless record.

No view is privileged. :mod:`scopex.record` keeps the whole truth about each unit; a view is a
function from a unit to a hashable key, and ``attribute`` counts them. ``by=`` takes a name from
:data:`BY` or ANY callable you write, because the view you need is usually not one we thought of.

The distinction that matters most in practice is between the ``*_path`` views and the others.
``by='author'`` keys on the FULL nesting (``MyModel.residual/MyModel.forward``), so a unit reached through two
user methods is its own bucket. ``by='innermost_author'`` keys on the last one. The first answers
"which combination", the second "who most directly". Reporting the second as though it were the
first is how 4,567 equations once became 1,168.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable

__all__ = ["BY", "attribute", "crosstab", "table"]


def _key(name_or_fn):
    if callable(name_or_fn):
        return name_or_fn
    try:
        return BY[name_or_fn]
    except KeyError:
        raise KeyError(f"unknown view {name_or_fn!r}; known: {sorted(BY)}") from None


BY: dict[str, Callable] = {
    # ── the contract ────────────────────────────────────────────────────────────────────────────
    "author":            lambda u: "/".join(u.authored) or "<none>",
    "innermost_author":  lambda u: (u.authored[-1] if u.authored else "<none>"),
    "library":           lambda u: "/".join(u.library) or "<none>",
    "role":              lambda u: u.role,
    "package":           lambda u: "/".join(u.packages) or "<unmarked>",
    # split is the headline number: whose code is this, in one word
    "split":             lambda u: ("user" if u.authored else
                                    "library" if u.library else "<unmarked>"),
    # ── structure ───────────────────────────────────────────────────────────────────────────────
    "scope":             lambda u: (u.scopes[-1] if u.scopes else "<root>"),
    "scope_path":        lambda u: "/".join(u.scopes) or "<root>",
    "transform":         lambda u: "/".join(u.transforms) or "<primal>",
    "depth":             lambda u: getattr(u, "depth", -1),
    # ── source ──────────────────────────────────────────────────────────────────────────────────
    "site":              lambda u: u.site,
    "file":              lambda u: u.site.rsplit(":", 1)[0],
    # ── what it is ──────────────────────────────────────────────────────────────────────────────
    "kind":              lambda u: u.kind,
    "level":             lambda u: u.level,
    # ── raw ─────────────────────────────────────────────────────────────────────────────────────
    # `path` joins across levels almost not at all: the rendered stack at the jaxpr level and at the
    # optimized-HLO level share very few keys, because XLA rewrites names. Use it to inspect, not to
    # join. `site` and the contract views are what join.
    "path":              lambda u: u.path,
}


def attribute(units: Iterable, by="split", *, top: int | None = None) -> Counter:
    """Count units by a view. ``attribute(walk(j), 'site')``."""
    k = _key(by)
    c = Counter(k(u) for u in units)
    return Counter(dict(c.most_common(top))) if top else c


def crosstab(units: Iterable, rows="split", cols="transform") -> dict:
    """Two views at once. The interactions live here -- e.g. which user methods cost most under
    ``transpose``, which is the reverse-mode adjoint and a common source of surprise."""
    kr, kc = _key(rows), _key(cols)
    out: dict = {}
    for u in units:
        out.setdefault(kr(u), Counter())[kc(u)] += 1
    return out


def table(counter: Counter, *, top: int = 15, total: int | None = None, label: str = "key") -> str:
    """A counter as an aligned text table with shares. Printing is a first-class use of this
    library, so it lives here rather than in every caller."""
    tot = total if total is not None else sum(counter.values())
    w = max([len(label)] + [len(str(k)) for k, _ in counter.most_common(top)] or [8])
    out = [f"{label:<{w}}  {'count':>8}  {'share':>7}"]
    out.append("-" * len(out[0]))
    for k, n in counter.most_common(top):
        out.append(f"{str(k):<{w}}  {n:8d}  {100 * n / max(1, tot):6.1f}%")
    rest = tot - sum(n for _, n in counter.most_common(top))
    if rest > 0:
        out.append(f"{'(' + str(len(counter) - top) + ' more)':<{w}}  {rest:8d}  "
                   f"{100 * rest / max(1, tot):6.1f}%")
    return "\n".join(out)
