"""The other three levels: StableHLO, HLO, and optimized HLO.

The jaxpr is where provenance is richest, and it is also the level furthest from what actually
takes the compile time. Everything here exists so a question can be asked at the level that can
answer it, and so the answers can be COMPARED.

STABLEHLO IS READ AS AN IR, NOT AS TEXT. jax lowers through ``jaxlib.mlir.ir``, so the module is
already an object graph with typed locations on every operation, and :func:`walk_stablehlo` walks
that. See its docstring for what the text parser it replaced could not see -- the short version is
that a line-based parser cannot see any operation that carries a REGION, because MLIR prints the
``loc(...)`` after the closing brace, and ``while``/``case``/``sort``/``reduce`` are exactly the
operations one most wants to find.

WHAT SURVIVES, MEASURED. XLA rewrites instruction names freely, so joining levels on instruction
name barely works at all. What does survive is the ``op_name`` metadata, which carries the rendered
JAX name stack verbatim::

    op_name="jit(f)/mylib:lib.solve/mylib:user.MyModel.residual/tanh"

That string is present on optimized-HLO instructions, and it is the whole basis of cross-level
attribution: measured across eight real programs the op_name join ran 0.9976-1.0000.

THE SOURCE LINE IS THERE TOO, BUT INDIRECTED. It is tempting to conclude from a metadata dict that
reads only ``{op_name="..."}`` that the optimized module carries no source location. It does --
jaxlib 0.10.2 emits ``stack_frame_id=N`` indexing four MODULE-LEVEL tables (``FileNames``,
``FunctionNames``, ``FileLocations``, ``StackFrames``), and every frame carries a
``parent_frame_id``, so the whole python stack is recoverable. Inline ``source_file=``/
``source_line=`` appear only under ``--xla_hlo_print_inline_stack_frames=true``.
:func:`hlo_sites` follows the indirection and applies the same jax-tree filter that
``source_info_util.user_frames`` applies at the jaxpr level -- which is precisely why the two
levels join on ``site``.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterator

from . import _parse
from .records import Ins

__all__ = ["walk_hlo", "walk_stablehlo", "stablehlo_module", "hlo_instructions", "hlo_sites",
           "frame_tables", "metadata", "hlo_frame_stack", "hlo_module", "opcode_of", "OPCODE_TEXT",
           "pre_optimization_hlo", "pre_optimization_text"]


# ══════════════════════════════════════════════════════════════════════════════════════════════
# HLO IS READ THROUGH THE OBJECT MODEL TOO, AND HERE IS EXACTLY HOW FAR IT REACHES
#
# `jaxlib.xla_client.hlo` (jaxlib/_hlo.so) hands back a real HloModule. Measured on jaxlib 0.10.2,
# these are the COMPLETE member lists, recorded so the next person need not re-probe to find out
# what is missing:
#
#   HloModule       as_serialized_hlo_module_proto, computations, create_empty_schedule,
#                   from_serialized_hlo_module_proto, make_nonfusion_computations, name, schedule,
#                   set_schedule, spmd_output_sharding, spmd_parameters_shardings, to_string
#   HloComputation  create_unary_instruction, instructions, name, render_html, replace_instruction
#   HloInstruction  async_wrapped_root, name, opcode, operands, to_string, users
#
# So STRUCTURE is native and exact: computations, instruction lists, names, opcode (a real
# HloOpcode enum), operands, users. There is NO `.metadata` on HloInstruction, NO `.shape`, and no
# accessor for the module's stack-frame tables. Metadata is reachable only through
# `HloInstruction.to_string()` -- which does print it, by default, despite the missing attribute.
#
# That is still the whole win, because the parse is now ANCHORED: one instruction per string,
# obtained from an object we already know exists, instead of a line-classifier over a whole module.
# The old parser had to decide per line "is this an instruction, and which computation am I in",
# and it got that wrong: `_INSTR` captured the shape as `\S+`, which cannot match a TUPLE shape
# like `(s32[], f32[8]{0})` because of the space after the comma. Measured against this native walk
# on 2,811 real per-pass dump snapshots -- the regex UNDERCOUNTED on 895 of them (31.8%), never
# overcounted, and missed 1,208 instructions. The dropped ones are not a random sample. They are
# exactly the tuple-shaped instructions: `while`, `call`, `tuple`, `custom-call` with a scratch
# output, `svd`, and every `parameter` of a control-flow body. On a `while_loop` program the regex
# saw 23 of 30 instructions; on a `scan`, 46 of 53. Same failure family as the three in the README:
# the blind spot sat on the control-flow and library-call instructions that carry the attribution.
# ══════════════════════════════════════════════════════════════════════════════════════════════

# THE PROTOBUF ROUTE, AND WHY IT IS A DEAD END HERE RATHER THAN AN UNEXPLORED ONE.
# `HloModule.as_serialized_hlo_module_proto()` exists and works -- it returns a real serialized
# `HloModuleProto` (2,087 bytes for a 16-instruction module), and `from_serialized_hlo_module_proto`
# round-trips it back to an equivalent module. A proto IS a more stable interface than printed text,
# so this looked like the answer. It is not, for one reason:
#
#   THERE IS NO SCHEMA. jaxlib 0.10.2 ships no `hlo_pb2`, no `*_pb2.py` of any kind (0 files under
#   site-packages), and no `.proto` source -- `hlo.proto` and `xla_data.proto` are absent from the
#   whole filesystem, `protoc` is not installed, and `google.protobuf` is not a jax dependency and
#   so is not importable either.
#
# Without a schema the binary wire format carries field NUMBERS and nothing else. Decoding it means
# hardcoding "computations are field 3, stack_frame_index is field 17" -- which is checkable against
# nothing, and whose failure mode when a guess is wrong is an empty list. That is strictly worse
# than the printed text, which at least names its fields. (Verified reachable by a generic wire
# walk, for the record: field 1 = name, 3 = computations, 17 = stack_frame_index. Correct today,
# unverifiable tomorrow.)
#
# The text-proto DUMPS are a different story and do get parsed as protos -- see `scopex.fusion`.
# Text format is self-describing, so it needs no schema, and the whole objection above evaporates.

# XLA's PRINTED opcode spelling differs from the HloOpcode enum name for a handful of opcodes.
# Harvested rather than guessed: every (enum, printed) pair observed across 3,017 real dump
# snapshots plus a targeted transcendental/comparison sweep. `hlo_instructions` warns on any
# divergence NOT listed here, so an unlisted one is loud rather than silently renamed.
OPCODE_TEXT = {
    "cos": "cosine",
    "exp": "exponential",
    "expm1": "exponential-minus-one",
    "log1p": "log-plus-one",
    # Found by `scopex.opcode_census(pre_optimization_text(f, x))["opt-barrier"]` returning 0 on a
    # module that visibly contains one: the enum is `kOptimizationBarrier` and XLA prints
    # `opt-barrier`. It matters more than the rest of this table, because `TRAPS["barrier_erased"]`
    # tells you to count exactly this opcode in exactly this module as a survival check -- and the
    # unlisted spelling made that check answer 0 for the same reason the barrier being erased
    # would. Right answer, wrong reason, no way to tell them apart.
    "optimization-barrier": "opt-barrier",
    "sin": "sine",
    "tan": "tangent",
    "top-k": "topk",
}
_CAMEL = re.compile(r"(?<!^)(?=[A-Z])")
_WARNED: set = set()


def opcode_of(instr) -> str:
    """XLA's printed opcode for a native ``HloInstruction``, derived from its ``HloOpcode`` enum.

    The enum is the native truth and cannot fail to parse; :data:`OPCODE_TEXT` carries the few enum
    names whose printed spelling differs (``kExp`` prints as ``exponential``).
    """
    k = instr.opcode.name
    k = _CAMEL.sub("-", k[1:] if k.startswith("k") else k).lower()
    return OPCODE_TEXT.get(k, k)


def hlo_module(obj):
    """The native ``jaxlib._hlo.HloModule`` for a ``Compiled``, an HLO text, or a module itself.

    Text goes through ``xla_client.hlo.hlo_module_from_text``, i.e. XLA's own parser rather than
    ours. It parsed 2,811 of 2,811 per-pass dump snapshots, and it RAISES on input that is not an
    HLO module (``thunk_metadata.txt``, ``buffer-assignment.txt``) instead of returning an empty
    one -- which is the property this project keeps having to buy the hard way.
    """
    from jaxlib.xla_client import hlo as _h
    if isinstance(obj, _h.HloModule):
        return obj
    if isinstance(obj, str):
        return _h.hlo_module_from_text(obj)
    for get in (lambda o: o.runtime_executable().hlo_modules()[0],
                lambda o: o.hlo_modules()[0]):
        try:
            return get(obj)
        except Exception:                                                    # pragma: no cover
            continue
    raise TypeError(f"cannot obtain an HloModule from {type(obj).__name__}")


# ══════════════════════════════════════════════════════════════════════════════════════════════
# THE BEFORE-OPTIMIZATION MODULE, WITHOUT A DUMP DIRECTORY -- AND A CORRECTION
#
# `scopex.flags.TRAPS["compiler_ir"]` said "compiler_ir('hlo') does drop metadata". Measured on
# jax 0.10.2, that is the SAME OVER-READ the same file records for the stablehlo row, one accessor
# further down. The object keeps everything; one of its two printers throws it away:
#
#     low.compiler_ir("hlo")                      -> jaxlib._hlo.XlaComputation
#       .as_hlo_text()                     692 chars   0 op_name    0 stack_frame_id   TRAP
#       .get_hlo_module().to_string()    2,092 chars   9 op_name    6 stack_frame_id   correct
#
# Same computation, same call, 3x the text. `as_hlo_text()` is the natural-looking accessor and it
# is the lossy one -- so "the pre-optimization module has no provenance" was a statement about a
# printer's defaults, not about the IR, exactly as with `str(compiler_ir('stablehlo'))`.
#
# WHAT THIS BUYS THAT THE OPTIMIZED MODULE CANNOT. The pre-optimization module is the only place
# some things exist at all. `TRAPS["barrier_erased"]` names one: OptimizationBarrierExpander runs
# before the optimized module is printed, so counting `opt-barrier` there always returns 0 and is
# not a survival check. Measured on one program through this route: 1 in pre-optimization, 0 in
# optimized. Until now that check needed `scopex.dump()`, i.e. a fresh process, because XLA reads
# its dump flags when the backend is first initialised. It no longer does.
# ══════════════════════════════════════════════════════════════════════════════════════════════

def pre_optimization_hlo(fn, *args, **kwargs):
    """The HLO as XLA received it -- BEFORE any pass -- as a native ``HloModule``. No dump needed.

    ``fn`` may be a ``jax.stages.Lowered``, a jitted function, or a plain callable; anything but a
    ``Lowered`` is lowered with ``*args``/``**kwargs`` first. Nothing is compiled and nothing is
    cached-over: lowering is the stage before the backend runs.

    This is the same module a dump writes as ``module_NNNN.<name>.before_optimizations.txt``, and
    it carries the full ``metadata={op_name=... stack_frame_id=...}`` plus the stack-frame tables,
    so :func:`walk_hlo`, :func:`hlo_sites` and :func:`scopex.attribute` all work on it::

        m = scopex.pre_optimization_hlo(f, x)
        units = list(scopex.walk_hlo(m.to_string(), level="hlo_pre"))

    Read the block comment above before assuming otherwise: ``XlaComputation.as_hlo_text()``, the
    obvious accessor, prints the same module with every ``metadata=`` block stripped.
    """
    low = fn if hasattr(fn, "compiler_ir") else _lower(fn, *args, **kwargs)
    comp = low.compiler_ir("hlo")
    m = comp.get_hlo_module()
    # The guard that would have caught the claim this function corrects. `as_hlo_text()` returns a
    # module with zero metadata; if `to_string()` ever starts doing the same, this level goes
    # provenance-free and every attribution through it silently empties.
    t = m.to_string()
    if "metadata={" not in t and "op_name" not in t:
        raise _parse.ParseError(
            "scopex.pre_optimization_hlo: HloModule.to_string() returned a module with NO "
            "metadata at all. That is what XlaComputation.as_hlo_text() does and what "
            "to_string() did not, measured on jaxlib 0.10.2 (2,092 chars / 9 op_name vs 692 / 0). "
            "If jaxlib has swapped the printers, this level now carries no provenance and every "
            "attribution through it is empty rather than wrong -- fix it here, do not let it "
            "return quietly.")
    return m


def pre_optimization_text(fn, *args, **kwargs) -> str:
    """:func:`pre_optimization_hlo` as text, with metadata and stack-frame tables. See its notes."""
    return pre_optimization_hlo(fn, *args, **kwargs).to_string()


def _lower(fn, *args, **kwargs):
    import jax
    try:
        return fn.lower(*args, **kwargs)
    except AttributeError:
        return jax.jit(fn).lower(*args, **kwargs)


# The two things the object model does not expose -- shape and metadata -- are read from ONE
# instruction's own to_string(), never from a guessed line of a whole module. Both patterns live in
# scopex._parse with a verbatim sample and a guard; see `hlo_shape` there for why the shape group is
# `.+?` and not `\S+`, and `hlo_metadata` for why it must read unquoted values too.
_shape_and_opcode = _parse.hlo_shape_and_opcode


def hlo_instructions(source) -> Iterator[dict]:
    """One dict per instruction of an HLO module, walked NATIVELY.

    ``source`` may be a ``jax.stages.Compiled``, HLO module text, or an ``HloModule``.

    Structure -- computation, instruction identity, opcode, operands -- comes from the object model
    and involves no pattern matching at all. Only ``shape`` and the ``metadata={...}`` block are
    read out of the instruction's own ``to_string()``, because ``HloInstruction`` exposes neither.
    """
    m = hlo_module(source)
    for comp in m.computations():
        for i in comp.instructions():
            s = i.to_string()
            # `metadata()` and not a local regex: metadata mixes QUOTED values (op_name="...") with
            # UNQUOTED ints (stack_frame_id=5). A quoted-only pattern drops the int silently, which
            # is how source resolution appeared impossible at this level.
            meta = metadata(s)
            shape, printed = _shape_and_opcode(s)
            op = opcode_of(i)
            if printed and printed != op and op not in _WARNED:
                _WARNED.add(op)
                warnings.warn(
                    f"HloOpcode {i.opcode.name!r} maps to {op!r} but XLA printed "
                    f"{printed!r}. Add it to scopex.levels.OPCODE_TEXT.",
                    RuntimeWarning, stacklevel=2)
            yield {
                "name": i.name,
                "opcode": op,
                "shape": shape,
                "computation": comp.name,
                "operands": [o.name for o in i.operands()],
                "op_name": meta.get("op_name", ""),
                "source_file": meta.get("source_file", ""),
                "source_line": meta.get("source_line", ""),
                "stack_frame_id": meta.get("stack_frame_id", ""),
            }


# ── metadata and the stack-frame tables: the part that is IRREDUCIBLY text ───────────────────────
# `HloInstruction` has no `.metadata`, and `HloModule` has no accessor for the four frame tables.
# Both are printed and nothing else exposes them, so these parsers stay -- but they no longer live
# HERE. Every pattern in the package sits in scopex._parse next to a verbatim sample of the text it
# reads and a guard that raises when it comes back emptier than its input, and scopex.selftest()
# runs the lot against both that sample and a fresh compile. What changed on this side is the blast
# radius: `metadata()` runs on one instruction's own to_string() rather than on a guessed line, and
# `frame_tables()` reads four small fixed-format tables rather than classifying a module.
#
# THE PARENT LINK IS NOT WHAT IT LOOKS LIKE. `hlo_sites` used to read `parent_frame_id` literally
# and guard the resulting self-loops as if jaxlib merely wrote an odd root frame. Measured on three
# programs: the ids are printed in a DIFFERENT INDEX SPACE from the row ids, so the literal reading
# makes EVERY leaf frame its own parent and silently truncates every stack to one frame -- the
# innermost line is still right, which is exactly why nobody noticed. `_parse` derives the offset
# per module and refuses to guess when neither convention gives a tree.

metadata = _parse.hlo_metadata                # xla::OpMetadata, one instruction at a time
frame_tables = _parse.hlo_frame_tables        # the four StackFrameIndexProto tables
hlo_sites = _parse.hlo_site                   # (file:line, function), jax-filtered
hlo_frame_stack = _parse.hlo_frame_stack      # the WHOLE python stack, innermost first


def _to_ins(rec: dict, level: str, tab=None) -> Ins:
    src = rec["source_file"]
    if src:                                            # only with inline stack frames enabled
        site, fn = f"{src}:{rec['source_line']}", rec.get("function", "?")
    elif tab and rec.get("stack_frame_id"):
        site, fn = hlo_sites(int(rec["stack_frame_id"]), tab)
    else:
        from .walk import NO_FRAME
        site, fn = NO_FRAME, "?"
    return Ins(level, rec["opcode"], rec["op_name"],
               unit=rec["name"], container=rec["computation"], site=site, function=fn,
               fusion=rec["opcode"] == "fusion",
               outlined=rec["computation"] not in ("<module>", "main"))


def walk_hlo(compiled, *, level: str = "hlo_opt") -> Iterator[Ins]:
    """Every instruction of the OPTIMIZED HLO, with its scope path AND its source line.

    ``compiled`` is a ``jax.stages.Compiled`` (an ``HloModule`` or HLO text also work). The walk is
    native -- see the block comment above for what the line-based parser it replaced could not see,
    the short version being every tuple-shaped instruction, i.e. ``while``/``call``/``custom-call``.

    Source lines are resolved through the module's stack-frame tables: metadata carries
    ``stack_frame_id``, not an inline path. Those tables are printed and have no accessor, so the
    module TEXT is still fetched when it is available -- but if it is not, the walk still yields
    every instruction, with ``site`` unresolved rather than nothing at all.

    WHEN THE TEXT IS AVAILABLE IT IS ALSO USED AS A CHECK. ``metadata()`` reads one instruction at a
    time, and at that scale a dropped key is indistinguishable from a key the instruction never had
    -- which is exactly how bug #2 hid: a quoted-only pattern still returned ``op_name``, so every
    instruction looked fine while every ``stack_frame_id`` silently vanished and this level was
    written up as carrying no source location at all. ``check_metadata_coverage`` counts what the
    module visibly contains against what came out, once, and raises on a shortfall.

    Parses eagerly and returns an iterator over the result, so that check runs when you CALL this --
    not when, or whether, somebody finishes iterating.
    """
    text = ""
    tab: dict = {}
    try:
        from .flags import hlo_text
        text = compiled if isinstance(compiled, str) else hlo_text(compiled)
        tab = frame_tables(text)
    except Exception:                                                        # pragma: no cover
        pass
    recs = list(hlo_instructions(compiled))
    if text:
        _parse.check_metadata_coverage(recs, text)
    return iter([_to_ins(rec, level, tab) for rec in recs])


# ══════════════════════════════════════════════════════════════════════════════════════════════
# STABLEHLO, READ THROUGH THE SAME BINDINGS JAX LOWERS THROUGH
#
# THE OBJECT IS NOT LOSSY; ITS DEFAULT PRINTER IS. `Lowered.compiler_ir('stablehlo')` was written
# up as a trap that "drops location info", because `str(...)` of what it returns contains ZERO
# `loc(`. That reading was one level too shallow. Measured on the marked_framework example:
#
#     str(compiler_ir('stablehlo'))                                     0 loc(   34,724 chars
#     compiler_ir('stablehlo').operation.print(enable_debug_info=True)  478 loc(  48,770 chars
#     Lowered.as_text(debug_info=True)                                  478 loc(  48,770 chars
#
# The last two are BYTE-IDENTICAL. So `compiler_ir` hands back the module jax built, locations and
# all -- `a is b` across two calls, i.e. nothing is re-lowered and nothing is copied -- and the only
# thing that ever dropped anything was `__str__`'s default `enable_debug_info=False`. The accessor
# trap is real; the conclusion drawn from it ("this route cannot see provenance") was not.
#
# HOW JAX BUILDS A LOCATION (jax/_src/interpreters/mlir.py::source_info_to_location, 0.10.2):
#
#     loc = <frame location>                       # a CallSiteLoc chain, or one FileLineColLoc
#     loc = NameLoc(f"{name_stack}/{primitive}", childLoc=loc)
#     loc = NameLoc(f"{primitive}:",             childLoc=loc)   # ONLY when lowered inline
#
# and a frame location is `CallSiteLoc(NameLoc(fn, FileLineColLoc) at <caller chain>)`. Both facts
# matter, and both are why `_name_of` peels exactly two levels and not "as many as there are":
#
#   * peel too FEW and an inline-lowered op reads as `"scatter:"` -- a name stack with no scopes in
#     it, so every contract accessor comes back empty for it. 3 of 311 ops here.
#   * peel too MANY and an op whose traceback is a SINGLE frame reads as that frame's function
#     name: `NameLoc("jit(program)", NameLoc("<module>", file:83))` collapses to `"<module>"`. That
#     is 26 of 311 ops on the same module -- and it is a plausible-looking string, which is how this
#     one nearly shipped. The marker test (`outer == f"{prim}:"` and `inner`'s last '/' segment is
#     that same `prim`) separates the two cases exactly.
# ══════════════════════════════════════════════════════════════════════════════════════════════


def _mlir():
    """``jaxlib.mlir.ir``, or None. jax lowers through it, so on any jax that can lower, it imports.
    Kept optional anyway so importing scopex never depends on a private module path."""
    try:
        from jaxlib.mlir import ir
        return ir
    except Exception:                                                        # pragma: no cover
        return None


def stablehlo_module(obj):
    """The ``ir.Module`` for ``obj``, WITHOUT re-lowering it, or None.

    Accepts a ``jax.stages.Lowered`` (uses its own stored module -- verified ``is``-identical
    across calls, so this costs nothing and cannot disagree with what jax compiled), an
    ``ir.Module``/``ir.Operation`` as-is, or StableHLO text, which is re-parsed.

    Text has to be parsed in a context that has the dialects registered: a bare ``ir.Context()``
    fails with ``Dialect 'func' not found for custom op 'func.func'`` even with
    ``allow_unregistered_dialects``, because ``func.func`` is custom *syntax*, not an unknown op.
    ``jax._src.interpreters.mlir.make_ir_context()`` is the registered one. Round-trip measured
    lossless: 54 operations in, 54 out, 51 of them still carrying a ``NameLoc``.
    """
    ir = _mlir()
    if ir is None:                                                           # pragma: no cover
        return None
    if isinstance(obj, ir.Module):
        return obj
    if isinstance(obj, ir.Operation) or isinstance(obj, ir.OpView):
        return obj
    if not isinstance(obj, str):
        try:
            m = obj.compiler_ir("stablehlo")
        except Exception:
            m = None
        if isinstance(m, ir.Module):
            return m
        try:
            from .flags import stablehlo_text
            obj = stablehlo_text(obj)
        except Exception:                                                    # pragma: no cover
            return None
    try:
        from jax._src.interpreters import mlir as _jmlir
        ctx = _jmlir.make_ir_context()
    except Exception:                                                        # pragma: no cover
        return None
    try:
        with ctx:
            return ir.Module.parse(obj)      # the Module keeps its Context alive; do not close it
    except Exception:                                                        # pragma: no cover
        return None


def _name_of(loc, ir, _depth: int = 0) -> str:
    """The rendered name stack on a location. See the block comment above for why this peels
    exactly the two wrappers jax puts on, and stops."""
    if isinstance(loc, ir.FusedLoc) and _depth < 4:
        for sub in loc.locations:                       # a pass may fuse two ops' locations
            n = _name_of(sub, ir, _depth + 1)
            if n:
                return n
        return ""
    if not isinstance(loc, ir.NameLoc):
        return ""
    name, child = loc.name_str, loc.child_loc
    if (name.endswith(":") and isinstance(child, ir.NameLoc)
            and child.name_str.rsplit("/", 1)[-1] == name[:-1]):
        return child.name_str                           # NameLoc("tanh:", NameLoc(".../tanh", ..))
    return name


def _rest_of(loc, ir) -> object:
    """What is left of a location once :func:`_name_of` has taken the name off it: the frames."""
    if isinstance(loc, ir.FusedLoc):
        for sub in loc.locations:
            if _name_of(sub, ir):
                return _rest_of(sub, ir)
        return loc
    if not isinstance(loc, ir.NameLoc):
        return loc
    name, child = loc.name_str, loc.child_loc
    if (name.endswith(":") and isinstance(child, ir.NameLoc)
            and child.name_str.rsplit("/", 1)[-1] == name[:-1]):
        return child.child_loc
    return child


def _frames(loc, ir) -> list[tuple[str, int, str]]:
    """``(file, line, function)`` for every python frame on a location, INNERMOST FIRST.

    The nesting is ``CallSiteLoc(callee at caller)`` with the callee innermost, each callee being
    ``NameLoc(function, FileLineColLoc)``. The caller chain is the deep one -- one level per python
    frame -- so it is iterated, not recursed."""
    out: list[tuple[str, int, str]] = []

    def rec(cur, d):
        if d > 64:                                                           # pragma: no cover
            return
        while isinstance(cur, ir.CallSiteLoc):
            rec(cur.callee, d + 1)
            cur = cur.caller
        if isinstance(cur, ir.NameLoc):
            ch = cur.child_loc
            if isinstance(ch, ir.FileLineColLoc):
                out.append((ch.filename, ch.start_line, cur.name_str))
            else:
                rec(ch, d + 1)
        elif isinstance(cur, ir.FileLineColLoc):
            out.append((cur.filename, cur.start_line, "?"))
        elif isinstance(cur, ir.FusedLoc):
            for sub in cur.locations:
                rec(sub, d + 1)

    rec(loc, 0)
    return out


def _site_of(loc, ir) -> tuple[str, str]:
    """``(file:line, function)`` -- the innermost frame that is neither jax nor scopex.

    Deliberately the SAME rule as :func:`scopex.walk._site` applies at the jaxpr level, using the
    same ``_SKIP`` directories, so the two levels join on ``site``. Measured on the
    marked_framework example: 285 of 311 StableHLO operations land on a ``file:line`` that also
    exists at the jaxpr level, and the honest ``<no-frame>`` bucket is kept rather than papered
    over with a caller's line."""
    from .walk import NO_FRAME, _SKIP
    for f, line, fn in _frames(loc, ir):
        if f and not any(f.startswith(m) for m in _SKIP):
            return f"{f}:{line}", fn
    return NO_FRAME, "?"


def _sym_name(op, ir) -> str:
    try:
        return ir.StringAttr(op.attributes["sym_name"]).value
    except Exception:                                                        # pragma: no cover
        return "?"


def _ops(op, ir, container: str, depth: int):
    """Every operation under ``op``, parents before children, carrying its enclosing ``func.func``
    and its region-nesting depth. Recursion over ``regions -> blocks -> operations`` rather than
    ``Operation.walk``, because it is lazy, it cannot lose a python exception inside a C++ callback,
    and it is where ``container``/``depth`` come from for free."""
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                o = child.operation
                yield o, container, depth
                if o.name == "func.func":
                    yield from _ops(o, ir, _sym_name(o, ir), 0)
                else:
                    yield from _ops(o, ir, container, depth + 1)


def walk_stablehlo(lowered) -> Iterator[Ins]:
    """Every StableHLO operation, with the name stack and the source line its location carries.

    Walks the ``jaxlib.mlir.ir`` module jax lowered into -- the real IR, not its printout. Accepts
    a ``Lowered``, an ``ir.Module``, or StableHLO text (see :func:`stablehlo_module`).

    WHAT THE TEXT PARSER THIS REPLACED COULD NOT SEE. It matched one operation per LINE, and MLIR
    prints an operation that owns a region across many lines with its ``loc(...)`` after the closing
    brace. So every region-bearing operation was invisible -- and those are ``while``, ``case``,
    ``sort``, ``reduce``, ``scatter``, i.e. precisely the expensive ones. Nor could it see an
    operation that is never printed at all: ``stablehlo.reduce`` in its short ``applies
    stablehlo.add across dimensions`` form has a real ``add`` and ``return`` inside its region that
    have no line to match. Measured, regex units vs native units on five programs::

        marked_framework   296 -> 311      control flow (while/cond/scan/sort)   90 -> 107
        nested cond+fori    69 ->  91      grad                                   19 ->  21
        vmap of scan        24 ->  29

    Native is a strict superset ON EVERY OPCODE in all five: no kind lost a single unit. What it
    gains are 2 ``while``, 1 ``case``, 1 ``sort``, the reduce-body ``add``/``return`` pairs, and the
    outlined ``func.func``s and their ``func.call`` sites.

    IT ALSO ANSWERS A QUESTION THE TEXT LEVEL DECLINED. ``site`` used to be the literal string
    ``"<see-jaxpr-level>"`` for every unit. The callsite chain is on the location, so it is now a
    real ``file:line``, filtered by the same rule the jaxpr level uses, and joins with it.
    """
    ir = _mlir()
    module = stablehlo_module(lowered) if ir is not None else None
    if module is None:
        warnings.warn(
            "walk_stablehlo could not obtain an MLIR module (jaxlib.mlir.ir missing, or this "
            "object exposes neither compiler_ir('stablehlo') nor parseable text). Falling back to "
            "the LINE-BASED parser, which cannot see any operation that owns a region -- while, "
            "case, sort, reduce and scatter will be missing from this answer.",
            RuntimeWarning, stacklevel=2)
        yield from _walk_stablehlo_text(lowered)
        return

    top = module.operation if hasattr(module, "operation") else module
    # ONE AsmState for the whole module, and it is not a micro-optimisation.
    # `Value.get_name()` with no state builds an SSA numbering for the enclosing module on EVERY
    # call, so reading the name of n values is O(n^2). Measured on a 1,806-operation module:
    # 10.24 s without the state, 0.012 s with it -- 858x, byte-identical names. The old text parser
    # did the same job in 0.04 s, so this is the difference between the IR walk being a strict
    # improvement and being unusable on exactly the big modules it was written for.
    try:
        state = ir.AsmState(top)
    except Exception:                                                        # pragma: no cover
        state = None
    n = named = 0
    for o, container, depth in _ops(top, ir, "<module>", 0):
        loc = o.location
        name = _name_of(loc, ir)
        site, fn = _site_of(_rest_of(loc, ir), ir)
        n += 1
        named += bool(name)
        if o.name == "func.func":
            unit = "@" + _sym_name(o, ir)
        elif state is None or not len(o.results):                            # pragma: no cover
            unit = ""
        else:
            try:
                unit = o.results[0].get_name(state)
            except Exception:                                                # pragma: no cover
                unit = ""
        yield Ins("stablehlo", o.name.split(".")[-1], name,
                  unit=unit, container=container, site=site, function=fn, depth=depth,
                  outlined=container not in ("<module>", "main"))
    if n and not named:
        # An all-empty level is the failure mode this module exists to make impossible. It means
        # the module was built without debug info, not that the program has no scopes.
        warnings.warn(
            f"walk_stablehlo walked {n} operations and not one carried a name -- every location is "
            "unknown. The module was almost certainly built without debug info; lower it normally "
            "(jax.jit(f).lower(x)) rather than reading a stripped dump.",
            RuntimeWarning, stacklevel=2)


# ── the line-based parser, kept ONLY as the no-bindings fallback ─────────────────────────────────
# It is retained because a wrong answer is worse than a fragile one, and this one is at least
# measured: it is a strict subset of what the IR walk returns. It is never reached silently -- the
# only caller warns first.

# The patterns are in scopex._parse -- with the sample they were measured against, and with two
# guards this version never had: the operation count is witnessed against SSA assignment lines (so
# "1 unit from a 3,214-operation module" raises instead of reading as an empty level), and every
# `loc(#locNN)` an operation cites must be DEFINED in the module (so a change to the alias-definition
# syntax cannot silently blank every name while the operation count still looks perfect).
_loc_aliases = _parse.stablehlo_loc_aliases


def _walk_stablehlo_text(lowered) -> Iterator[Ins]:
    """One operation per line of ``stablehlo_text``. Subset of :func:`walk_stablehlo`; see there.

    Two traps are already handled and both are still real: ``Lowered.as_text()`` defaults to
    ``debug_info=False`` and prints no locations at all, and operations carry ``loc(#loc17)`` rather
    than ``loc("name")``. Matching only the inline form yielded **1 unit on 16 of 21 real
    programs**, including modules of 3,214 and 21,000 operations.

    One thing this fallback does that the IR walk need not: when jax lowers a primitive inline it
    emits ``#loc73 = loc("scatter:"(#loc43))``, so the name is the bare marker ``"scatter:"`` and
    not a name stack. The IR walk peels that; here it comes through as-is.
    """
    from .flags import stablehlo_text
    text = lowered if isinstance(lowered, str) else stablehlo_text(lowered)
    for op, name in _parse.stablehlo_op_lines(text):
        yield Ins("stablehlo", op.split(".")[-1], name, site="<see-jaxpr-level>")
