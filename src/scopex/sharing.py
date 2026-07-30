"""Duplicate subexpressions in a jaxpr -- why a small program becomes a large module.

JAX does no CSE at the jaxpr level, so a program can contain N syntactically identical
subexpressions and emit all N of them. :func:`scopex.walk` counts equations and
:func:`scopex.attribute` says where they came from; neither says that 511 of them are the same
equation written 512 times. That is a different question with a different fix, and on the
``switch_ident`` family it is the whole answer.

Two independent notions of "identical", reported separately BECAUSE THEY HAVE DIFFERENT FIXES:

VALUE duplicates
    Same primitive, same params, same OPERAND IDENTITIES. These compute the literally same value.
    A CSE pass collapses them, so the cost is usually compile-time only, and the fix is in your code
    (hoist it) or nowhere (XLA will CSE it later).

SHAPE duplicates
    Alpha-equivalent SUB-JAXPRS -- ``switch`` branches, ``pjit`` bodies, ``scan`` bodies -- whose
    operands differ but whose structure does not. XLA emits one computation per branch regardless,
    so these survive CSE and the fix is to not build N of them (``jnp.where`` over a stacked result,
    or one branch parameterised by an operand).

MEASURED on the corpus's codegen-multiplicity family::

    switch_ident_128    130 equations, 128 sub-jaxprs, 127 redundant equations in ONE value group
                        (128 x integer_pow), and 128 sub-jaxprs in ONE alpha-equivalence class
    switch_ident_512    511 redundant in one class -- i.e. the program is one branch, 512 times

Alpha-equivalence is a structural hash (:func:`struct_hash`) memoised on ``id()``; jaxprs are
immutable, so the memo is safe and the walk stays linear in equations rather than quadratic in
branches.
"""

from __future__ import annotations

from collections import defaultdict

from .walk import subjaxprs

__all__ = ["jaxpr_sharing", "Sharing", "struct_hash"]


def _jaxpr_of(x):
    j = getattr(x, "jaxpr", x)
    if not (hasattr(j, "eqns") and hasattr(j, "invars")):
        raise TypeError(
            f"not a jaxpr: {type(x).__name__}. Pass a Jaxpr, a ClosedJaxpr, or anything with a "
            f".jaxpr attribute (e.g. jax.make_jaxpr(f)(x)).")
    return j


def _atom(a, env):
    """A canonical id for a jaxpr atom, stable under variable renaming."""
    v = getattr(a, "val", None)
    if v is not None or type(a).__name__ == "Literal":
        try:
            return ("lit", str(a.aval.str_short()), repr(v))
        except Exception:                                                    # pragma: no cover
            return ("lit", repr(v))
    return env.get(id(a), ("free", str(getattr(a, "aval", "?"))))


def _param_key(k, v, memo):
    """Hashable key for one eqn param.

    Sub-jaxprs become their STRUCTURAL fingerprint, so two switch branches differing only in
    variable names hash the same. Unhashable params fall back to ``repr`` rather than being dropped
    -- a dropped param would merge two equations that differ, which is the wrong direction to fail.
    """
    j = getattr(v, "jaxpr", v)
    if hasattr(j, "eqns") and hasattr(j, "invars"):
        return (k, "jaxpr", struct_hash(j, memo))
    if isinstance(v, (list, tuple)):
        return (k, tuple(_param_key(i, x, memo)[1:] for i, x in enumerate(v)))
    try:
        hash(v)
        return (k, v)
    except TypeError:
        return (k, repr(v))


def struct_hash(jaxpr, memo=None) -> int:
    """Alpha-equivalence fingerprint: two jaxprs hash equal iff identical up to variable names.

    Cached on ``id()`` -- jaxprs are immutable, so the memo cannot go stale within a call.
    """
    memo = {} if memo is None else memo
    if id(jaxpr) in memo:
        return memo[id(jaxpr)]
    j = _jaxpr_of(jaxpr)
    env = {}
    for i, v in enumerate(list(j.invars) + list(j.constvars)):
        env[id(v)] = ("in", i)
    parts = []
    for n, e in enumerate(j.eqns):
        parts.append((str(e.primitive),
                      tuple(_atom(a, env) for a in e.invars),
                      tuple(sorted(_param_key(k, v, memo) for k, v in e.params.items()))))
        for oi, o in enumerate(e.outvars):
            env[id(o)] = ("eq", n, oi)
    parts.append(("out", tuple(_atom(a, env) for a in j.outvars)))
    h = hash(tuple(parts))
    memo[id(jaxpr)] = h
    return h


def _value_key(eqn, env, memo):
    return (str(eqn.primitive),
            tuple(_atom(a, env) for a in eqn.invars),
            tuple(sorted(_param_key(k, v, memo) for k, v in eqn.params.items())))


class Sharing(dict):
    """Duplicate census. ``value_dup_eqns`` and ``subjaxpr_dup`` count REDUNDANT copies (n-1 per
    group), so 0 means everything is distinct."""

    @property
    def redundant_fraction(self) -> float:
        """Share of equations that are value-duplicates of another equation."""
        return self["value_dup_eqns"] / max(1, self["n_eqns"])

    def __str__(self):
        v, s = self["value_groups"], self["subjaxpr_groups"]
        out = [f"jaxpr_sharing: {self['n_eqns']} equations, {self['n_subjaxprs']} sub-jaxprs",
               f"  VALUE duplicates    {self['value_dup_eqns']} eqns in {len(v)} groups "
               f"({100 * self.redundant_fraction:.1f}% of equations redundant)",
               f"  SHAPE duplicates    {self['subjaxpr_dup']} sub-jaxprs in {len(s)} "
               f"alpha-equivalence classes"]
        if v:
            out.append("  top value groups (count x primitive):")
            for cnt, prim, sites in v[:8]:
                out.append(f"    {cnt:6d} x {prim:<24s} {sites[0]}")
        if s:
            out.append("  top alpha-classes (count x n_eqns_each):")
            for cnt, ne, where in s[:8]:
                out.append(f"    {cnt:6d} x {ne:4d} eqns   first seen under {where}")
        return "\n".join(out)


def jaxpr_sharing(x, *, top: int = 20) -> Sharing:
    """Duplicate-subexpression census for ``x`` (a Jaxpr, ClosedJaxpr, or anything with ``.jaxpr``).

    Recurses through every sub-jaxpr via :func:`scopex.subjaxprs`, so no primitive is named and a
    higher-order primitive added to JAX tomorrow is covered today.
    """
    root = _jaxpr_of(x)
    memo: dict = {}
    n_eqns = 0
    n_sub = 0
    val: dict = defaultdict(list)
    sub: dict = defaultdict(list)

    def rec(j, path):
        nonlocal n_eqns, n_sub
        env = {}
        for i, v in enumerate(list(j.invars) + list(j.constvars)):
            env[id(v)] = ("in", path, i)
        for n, e in enumerate(j.eqns):
            n_eqns += 1
            val[_value_key(e, env, memo)].append((path, n, e))
            for oi, o in enumerate(e.outvars):
                env[id(o)] = ("eq", path, n, oi)
            for k, sj in subjaxprs(e):
                n_sub += 1
                sub[struct_hash(sj, memo)].append((f"{path}/{n}:{e.primitive}[{k}]", sj))
                rec(sj, f"{path}/{n}")

    rec(root, "")

    vg = sorted(((len(g), str(g[0][2].primitive), [f"{p}/{n}" for p, n, _ in g[:3]])
                 for g in val.values() if len(g) > 1), key=lambda r: -r[0])
    sg = sorted(((len(g), len(_jaxpr_of(g[0][1]).eqns), g[0][0])
                 for g in sub.values() if len(g) > 1), key=lambda r: -r[0])
    return Sharing(
        n_eqns=n_eqns, n_subjaxprs=n_sub,
        value_dup_eqns=sum(c - 1 for c, _, _ in vg),
        subjaxpr_dup=sum(c - 1 for c, _, _ in sg),
        value_groups=vg[:top], subjaxpr_groups=sg[:top],
    )
