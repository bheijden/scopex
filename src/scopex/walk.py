"""Traversal: every equation of a jaxpr, including the ones inside other equations.

Two things here are easy to get wrong and both were, before being measured against JAX's own
traversal.

**1. Sub-jaxprs come in two spellings.** ``scan``/``cond``/``pjit`` carry a ``ClosedJaxpr``, which
answers ``.jaxpr``. ``remat2`` carries a BARE ``Jaxpr``, which does not -- and so does
``scatter-add``, in its ``update_jaxpr`` parameter. Detecting sub-programs by ``getattr(x,'jaxpr')``
finds only the first kind: on a ``jax.checkpoint``-wrapped payload that reported **1 equation where
``jaxpr_util.all_eqns`` reported 82**. Duck-typing on structure finds both, and needs no list of
primitives to keep in sync as JAX adds them.

**2. ``named_scope`` does not propagate into sub-jaxprs.** The scope sits on the CALLING equation;
the body sees nothing. A walker that does not concatenate the caller's path onto the body's reports
roughly a third of a real program as unattributed. So paths ARE inherited -- that is required for
correctness.

Source SITES are a different matter and are NOT inherited by default. Silently giving an equation
its caller's ``file:line`` once turned a 16.2% honestly-unattributable bucket into a reported 0%,
and inflated one file's share from 48.6% to 60.0%. JAX itself reports that bucket; so do we. Pass
``inherit_site=True`` if you want the estimate, and know that you asked for it.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import jax

from .record import Eqn

__all__ = ["walk", "subjaxprs", "verify_parity", "at"]

# Frames inside these directories are never the answer to "where did this come from".
#
# Resolved to REAL package directories, not matched as substrings. A substring rule like
# `"/scopex/" in path` also excludes a user's own project when it happens to live under a directory
# of that name -- which is exactly what happens when you check this repository out, and it silently
# turns every attribution into the unknown bucket.
def _pkg_dir(mod_name: str) -> str:
    try:
        import importlib.util
        spec = importlib.util.find_spec(mod_name)
        if spec and spec.origin:
            return os.path.dirname(os.path.abspath(spec.origin)) + os.sep
    except Exception:                                                     # pragma: no cover
        pass
    return "\0"                                                           # matches nothing


_SKIP = tuple(d for d in (_pkg_dir("jax"), _pkg_dir("jaxlib"),
                          os.path.dirname(os.path.abspath(__file__)) + os.sep) if d != "\0")

NO_FRAME = "<no-frame>"


def subjaxprs(eqn) -> Iterator[tuple[object, object]]:
    """``(param_key, jaxpr)`` for every sub-program an equation carries.

    Generic by construction: no primitive is named, so a higher-order primitive added to JAX
    tomorrow is covered today."""
    for k, v in eqn.params.items():
        seq = v if isinstance(v, (list, tuple)) else [v]
        for i, x in enumerate(seq):
            j = getattr(x, "jaxpr", x)      # ClosedJaxpr -> its Jaxpr; bare Jaxpr -> itself
            if hasattr(j, "eqns") and hasattr(j, "invars"):
                yield (k if not isinstance(v, (list, tuple)) else (k, i)), j


def _site(eqn) -> tuple[str, object]:
    """``(file:line, frame)`` for the first frame that is neither jax nor scopex.

    Returns ``(NO_FRAME, None)`` when there is no such frame -- which happens, and is information.
    JAX's own ``jaxpr_util.source_locations`` keeps the same honest empty bucket."""
    si = getattr(eqn, "source_info", None)
    tb = getattr(si, "traceback", None)
    if tb is None:
        return NO_FRAME, None
    try:
        for fr in tb.frames:
            fn = getattr(fr, "file_name", "") or ""
            if fn and not any(fn.startswith(m) for m in _SKIP):
                return f"{fn}:{_line_of(fr)}", fr
    except Exception:                                                     # pragma: no cover
        pass
    return NO_FRAME, None


# jaxlib's Frame exposes `line_num` (jax 0.10.2). Earlier and later spellings are tried in turn so a
# rename degrades to a missing line number rather than to a wrong one.
_LINE_ATTRS = ("line_num", "start_line", "lineno", "line")


def _line_of(fr) -> str:
    for a in _LINE_ATTRS:
        v = getattr(fr, a, None)
        if isinstance(v, int):
            return str(v)
    return "?"


def _jaxpr_of(x):
    j = getattr(x, "jaxpr", x)
    if not (hasattr(j, "eqns") and hasattr(j, "invars")):
        raise TypeError(f"not a jaxpr: {type(x).__name__}")
    return j


def walk(x, *, inherit_site: bool = False) -> Iterator[Eqn]:
    """Every equation of ``x``, innermost included, each with its complete provenance.

    ``x`` may be a ``Jaxpr``, a ``ClosedJaxpr``, or anything answering ``.jaxpr``.

    Equations are yielded in traversal order, parents before their children. Verified equal in
    COUNT to ``jax._src.jaxpr_util.all_eqns(revisit_inner_jaxprs=True)`` -- see
    :func:`verify_parity`, which the test suite runs on every corpus case.
    """
    root = _jaxpr_of(x)

    def rec(jaxpr, path: str, depth: int, addr: tuple, csite: str):
        for i, e in enumerate(jaxpr.eqns):
            own = str(getattr(getattr(e, "source_info", None), "name_stack", "") or "")
            # The body of a sub-jaxpr renders its OWN stack only; the caller's scopes live on the
            # calling equation. Concatenate, or the body looks scope-less.
            full = f"{path}/{own}" if (path and own) else (path or own)
            s, fr = _site(e)
            if s == NO_FRAME and inherit_site and csite != NO_FRAME:
                s, fr = csite, None
            yield Eqn(eqn=e, path=full, depth=depth, addr=addr + (i,), site=s, frame=fr)
            for key, sub in subjaxprs(e):
                yield from rec(sub, full, depth + 1, addr + (i, key), s)

    yield from rec(root, "", 0, (), NO_FRAME)


def at(x, addr: tuple):
    """The equation at ``addr``. ``at(j, e.addr) is e.eqn`` for every ``e`` in ``walk(j)``."""
    jaxpr, cur = _jaxpr_of(x), None
    it = iter(addr)
    for i in it:
        cur = jaxpr.eqns[i]
        key = next(it, None)
        if key is None:
            return cur
        jaxpr = dict(subjaxprs(cur))[key]
    return cur


def verify_parity(x) -> tuple[int, int, bool]:
    """``(ours, jax's, equal)`` against ``jaxpr_util.all_eqns``.

    The traversal here cannot simply BE ``all_eqns``, because we need each equation's address and
    its inherited scope path and that function yields neither. So it is an independent
    implementation, and this is the check that it stayed honest."""
    from jax._src import jaxpr_util as ju
    ours = sum(1 for _ in walk(x))
    theirs = sum(1 for _ in ju.all_eqns(_jaxpr_of(x), revisit_inner_jaxprs=True))
    return ours, theirs, ours == theirs


def make_jaxpr(fn, *args, **kwargs):
    """Convenience: trace and return the ClosedJaxpr."""
    return jax.make_jaxpr(fn, **kwargs)(*args)
