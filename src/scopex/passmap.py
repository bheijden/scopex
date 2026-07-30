"""Given a pass name XLA printed, point at the file that implements it. A pointer, not an analysis.

Fifteen of this project's thirty investigations finished only after somebody opened XLA's source.
That last step is judgement and should not be automated. THE LOOKUP is not judgement, and it is
where the friction is: XLA prints ``triton-gemm-rewriter`` and the file is ``gemm_fusion.h``; it
prints ``rename-instructions`` and the file is ``add_tracking_suffix_to_instruction_names.h``; it
prints ``cse_barrier_expander`` and the file is ``optimization_barrier_expander.h``. Measured on
the 213 pass names this project's corpus actually produces, **30 have a file name that does not
echo the pass name**, and seven of those are unguessable. Guessing the path is a coin flip on one
name in seven; the table is not.

WHAT THIS IS NOT. It does not say what the pass does, why it was slow, or whether it is your
problem. It says: this string was returned by ``name()`` at this file and this line, in the XLA
tree your jaxlib was built from. Everything after that is reading.

WHERE THE ROWS COME FROM, AND WHY THEY ARE CHECKABLE. jax pins its XLA revision and publishes the
hash of the tarball. The table was built by grepping that exact tarball after checking its sha256
against jax's own ``XLA_SHA256``; ``tools/build_passmap.py`` rebuilds it and prints its working.
The names were ENUMERATED from real logs (``tools/probe_passes.py``: ~45 small programs per
backend under ``TF_CPP_VMODULE=hlo_pass_pipeline=1``), never listed from memory.

THREE THINGS THE TABLE KNOWS THAT A GUESS CANNOT

* **Some names are pipelines, not passes.** ``simplification`` is fifteen passes on CPU; its
  seconds are their seconds, and they are timed separately in the same log. A pointer that sent
  you to a ``simplification.cc`` would be pointing at a file that does not exist. Twelve of the
  213 names are like this, and :attr:`PassSource.kind` says so.
* **Some names mean different files on different backends.** ``fusion-wrapper`` is
  ``xla/backends/gpu/transforms/fusion_wrapper.h`` on GPU and ``xla/service/cpu/fusion_wrapper.h``
  on CPU. ``simplification`` is built in ``cpu_compiler.cc`` and in ``gpu_compiler.cc``. Ask
  without a platform and you get both, never one of them silently.
* **Four names are a pass wrapped in a pipeline OF THE SAME NAME.** XLA writes
  ``AddPass<HloPassPipeline>("sanitize-constant-names")`` and puts ``SanitizeConstantNames``
  inside it, so the log carries two nested timings that differ only in depth. Measured: 426 such
  pairs in one GPU probe log. Anything that sums by name adds the pass to its own wrapper.
  :attr:`PassSource.wrapper_pipeline` is the pointer to the wrapper.

FOUR NAMES WERE OMITTED AT ONE POINT AND ARE NOT ANY MORE, AND THAT IS WORTH SAYING OUT LOUD. An
earlier build of this table mapped ``float_normalization`` to ``third_party/tsl/.../numbers.h``
because a loose prefix rule matched ``"float"`` next to a ``StrCat``, and mapped ``simplification``
to the GPU file alone, which is wrong for every CPU user. Both were caught by auditing the rows
whose file name does not echo the pass name -- the same 30 rows that make the table worth having.
The generator's rules were tightened until it could no longer produce either; see
``tools/build_passmap.py``.
"""

from __future__ import annotations

import os
from typing import NamedTuple

from ._passmap import (
    BUILT_FOR,
    OMITTED,
    PASSES,
    SOURCE_URL,
    XLA_COMMIT,
    XLA_SHA256,
)

__all__ = ["PassSource", "pass_source", "pass_sources", "pipelines_in", "cross_check",
           "unmapped", "verify_pass_map", "XLA_COMMIT", "XLA_SHA256", "BUILT_FOR"]

_PLATFORMS = ("cpu", "gpu")


class PassSource(NamedTuple):
    """Where one pass name is defined in XLA. A pointer, with the evidence attached."""

    name: str
    file: str                       # the header (or .cc) the name literal was found in
    line: int
    impl: str | None                # the .cc beside it, when the tree has one -- what to read
    kind: str                       # "pass" | "pipeline"
    platforms: tuple[str, ...]      # backends this name was OBSERVED on
    source_line: str                # the C++ the pointer resolves to
    wrapper_pipeline: tuple[str, int] | None = None
    ambiguous_platforms: dict | None = None   # set when asked without a platform and it matters

    @property
    def url(self) -> str:
        """A permalink into the exact revision, so the line number is not a moving target."""
        return SOURCE_URL.format(commit=XLA_COMMIT, file=self.file, line=self.line)

    @property
    def read(self) -> str:
        """The file a human should open: the implementation if there is one, else the header."""
        return self.impl or self.file

    def __str__(self) -> str:
        head = f"{self.name} -> {self.read}"
        if self.kind == "pipeline":
            head += (f"\n  NOT A PASS: this name is an HloPassPipeline built at {self.file}:"
                     f"{self.line}. Its seconds are other passes' seconds, and those are timed "
                     f"separately in the same log.")
        if self.wrapper_pipeline:
            head += (f"\n  XLA also wraps this pass in a pipeline of the SAME name at "
                     f"{self.wrapper_pipeline[0]}:{self.wrapper_pipeline[1]}, so the log holds two "
                     f"nested timings under one name; summing by name counts it twice.")
        if self.ambiguous_platforms:
            head += ("\n  BACKEND-SPECIFIC: "
                     + ", ".join(f"{p} -> {f}" for p, (f, _) in self.ambiguous_platforms.items())
                     + ". Pass platform= to pick one.")
        return head + f"\n  {self.file}:{self.line}  {self.source_line}\n  {self.url}"


def _row(name: str):
    return PASSES.get(name)


def pass_source(name: str, *, platform: str | None = None) -> PassSource | None:
    """The XLA file that implements ``name``, or ``None`` if this table cannot say.

    ``name`` is a key of ``scopex.pass_timings(...)["passes"]`` -- exactly the string XLA printed.

    ``None`` is a real answer and is returned rather than a guess: the name may be from a jaxlib
    built off a different XLA revision, or it may be one of the names the generator refused to
    resolve (see :func:`unmapped`). A wrong pointer costs an afternoon in the wrong file, so this
    function has no fallback and no fuzzy match.

    ``platform`` is required in spirit and optional in signature. Two of the 213 names resolve to
    DIFFERENT FILES per backend; asked without a platform, the result carries both in
    ``ambiguous_platforms`` and ``file`` is the one for the backend the name is most associated
    with, never a silent pick of one.
    """
    row = _row(name)
    if row is None:
        return None
    where, impl, kind, plats, wrap, ev = row
    amb = None
    if isinstance(where, dict):
        # Keyed by BACKEND, and so is `impl`. Indexing one with a key from the other is how a
        # per-platform row quietly loses its implementation file; the tests pin both.
        if platform not in where:
            amb = dict(where)
        key = platform if platform in where else sorted(where)[0]
        f, ln = where[key]
        impl = impl.get(key) if isinstance(impl, dict) else impl
    else:
        f, ln = where
    # The four same-name wrappers are all built in gpu_compiler.cc, so the warning is a GPU fact.
    # Attaching it to a CPU answer would be a true sentence about the wrong backend.
    if wrap and platform and platform != "gpu":
        wrap = None
    return PassSource(name=name, file=f, line=ln, impl=impl, kind=kind,
                      platforms=tuple(plats), source_line=ev,
                      wrapper_pipeline=wrap, ambiguous_platforms=amb)


def pass_sources(result, *, platform: str | None = None, top: int | None = None) -> list:
    """Annotate a whole ``pass_timings`` result. ``[(name, seconds, PassSource | None), ...]``.

    Accepts the dict :func:`scopex.pass_timings` returns, its ``"passes"`` sub-dict, or any
    iterable of names. Order is preserved, so a ranking stays a ranking.

    Names this table cannot resolve come back with ``None`` beside them and are NOT dropped: a
    silently shorter list would hide exactly the pass that a jaxlib upgrade has renamed.
    """
    if isinstance(result, dict):
        passes = result.get("passes", result)
        items = list(passes.items()) if isinstance(passes, dict) else [(n, None) for n in passes]
    else:
        items = [(n, None) if isinstance(n, str) else n for n in result]
    if top:
        items = items[:top]
    return [(n, s, pass_source(n, platform=platform)) for n, s in items]


def pipelines_in(result, *, platform: str | None = None) -> dict:
    """Which rows of a ``pass_timings`` result are PIPELINES, and which wrap a same-named pass.

    Both are reasons a per-name total is not a per-pass total. This function only reports what the
    table knows about the names; it does not re-time anything and it does not correct the numbers
    -- ``scopex.pass_timings``' own ``coverage`` does that from the log's nesting.

    Returns ``{"pipelines": {name: (file, line)}, "self_wrapped": {name: (file, line)},
    "unmapped": [name, ...]}``.
    """
    pipes, wrapped, miss = {}, {}, []
    for name, _sec, src in pass_sources(result, platform=platform):
        if src is None:
            miss.append(name)
            continue
        if src.kind == "pipeline":
            pipes[name] = (src.file, src.line)
        if src.wrapper_pipeline:
            wrapped[name] = src.wrapper_pipeline
    return {"pipelines": pipes, "self_wrapped": wrapped, "unmapped": miss}


def cross_check(result) -> dict:
    """THE CROSS-CHECK FOR THE ``kind`` COLUMN, against something the table was not built from.

    The table's ``pass`` / ``pipeline`` column comes from grepping C++. Whether a name is a
    pipeline is also observable at RUN TIME and independently: ``hlo_pass_pipeline.cc`` announces
    a nested pipeline with a header line before timing it, so ``scopex._parse.pass_leaf_split``
    can tell an aggregate from a leaf without ever seeing the source. Two routes, no shared step.

    Needs the raw log, which is why this lives next to :class:`scopex.Raw`: pass a
    :func:`scopex.pass_timings` result and it reads ``result["raw"]``.

    Returns ``{"ok", "checked", "agree", "disagree": [...], "unmapped": [...]}``. A disagreement
    is a real finding in either direction -- either the table is stale, or XLA changed a pass into
    a pipeline (which changes what its seconds MEAN, and is the kind of thing that silently
    invalidates a published ranking).

    Measured when written, on a CPU compile of ``sum(tanh(a) @ a)``: 76 names checked, 76 agree.
    """
    raw = result.get("raw") if isinstance(result, dict) else None
    if raw is None or not raw.exists():
        return {"ok": False, "checked": 0, "agree": 0, "disagree": [], "unmapped": [],
                "why": "no raw log on this result -- pass_timings(...) returns one under 'raw'; "
                       "without the text there is nothing independent to check against."}
    from . import _parse
    split = _parse.pass_leaf_split(raw.text())
    ran_as_pipeline = {t.name for t in split.aggregates}
    ran_as_leaf = {t.name for t in split.leaves}
    agree, disagree, miss = 0, [], []
    for name in sorted(ran_as_pipeline | ran_as_leaf):
        src = pass_source(name)
        if src is None:
            miss.append(name)
            continue
        # A name that is BOTH (the same-name wrapper case) is consistent with either column.
        both = name in ran_as_pipeline and name in ran_as_leaf
        want_pipeline = name in ran_as_pipeline
        if both or (src.kind == "pipeline") == want_pipeline or src.wrapper_pipeline:
            agree += 1
        else:
            disagree.append({
                "pass": name,
                "table_says": src.kind,
                "log_says": "pipeline (announced a nested run before it was timed)" if want_pipeline
                            else "leaf pass (timed with nothing nested inside it)",
                "file": src.read})
    return {"ok": not disagree, "checked": agree + len(disagree), "agree": agree,
            "disagree": disagree, "unmapped": miss}


def unmapped() -> dict:
    """Pass names that were observed but deliberately NOT given a pointer, with the reason.

    Empty on the shipped table, and the function still exists: the generator can and does refuse,
    and a reader needs to be able to tell "no row" from "no such pass". Rebuilding the table
    against a different XLA revision will populate this.
    """
    return dict(OMITTED)



def verify_pass_map(xla_src: str | os.PathLike, *, limit: int | None = None) -> dict:
    """THE CROSS-CHECK. Re-read every row against a real XLA checkout and report what moved.

    ``xla_src`` is the root of an XLA source tree (the directory containing ``xla/``). For a
    verdict that means anything it should be the revision your jaxlib was built from::

        curl -s https://raw.githubusercontent.com/jax-ml/jax/jax-v0.10.2/third_party/xla/revision.bzl
        curl -sL https://github.com/openxla/xla/archive/$XLA_COMMIT.tar.gz | tar xz

    Every row is checked two ways, and the second is the one that matters:

    * the file exists and has that many lines -- catches a moved or deleted file;
    * the pass name literal occurs within two lines of the recorded line number -- catches the
      row having drifted onto somebody else's code, which is the failure that still LOOKS like a
      working pointer.

    Returns ``{"ok", "checked", "confirmed", "problems": [...], "commit_expected"}``. It does not
    raise: a table built for jaxlib 0.10.2 checked against a newer XLA is expected to have drifted,
    and that is information rather than an error.
    """
    root = os.fspath(xla_src)
    problems, confirmed, checked = [], 0, 0
    for name, (where, _impl, _kind, _plats, _wrap, _ev) in list(PASSES.items())[:limit]:
        places = list(where.values()) if isinstance(where, dict) else [where]
        for f, ln in places:
            checked += 1
            p = os.path.join(root, f)
            if not os.path.isfile(p):
                problems.append(f"{name}: {f} is not in this tree")
                continue
            try:
                with open(p, errors="replace") as fh:
                    lines = fh.read().splitlines()
            except Exception as e:                                           # pragma: no cover
                problems.append(f"{name}: {f} unreadable ({e!r})")
                continue
            if not (1 <= ln <= len(lines)):
                problems.append(f"{name}: {f}:{ln} is past the end ({len(lines)} lines)")
                continue
            window = " ".join(lines[max(0, ln - 3):ln + 2])
            if f'"{name}"' in window or name[:12] in window:
                confirmed += 1
            else:
                problems.append(f"{name}: {f}:{ln} no longer contains the name; the line is now "
                                f"{lines[ln - 1].strip()[:70]!r}")
    return {"ok": not problems, "checked": checked, "confirmed": confirmed,
            "problems": problems, "commit_expected": XLA_COMMIT, "built_for": BUILT_FOR}
