"""The other three levels: StableHLO, HLO, and optimized HLO.

The jaxpr is where provenance is richest, and it is also the level furthest from what actually
takes the compile time. Everything here exists so a question can be asked at the level that can
answer it, and so the answers can be COMPARED.

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
from collections.abc import Iterator

from .records import Ins

__all__ = ["walk_hlo", "walk_stablehlo", "hlo_instructions", "hlo_sites",
           "frame_tables", "metadata"]

# `%add.3 = f32[16,16]{1,0} add(%a, %b), metadata={op_name="jit(f)/.../add"}`
# The leading `%` is absent on some lines and ROOT may prefix the name.
_INSTR = re.compile(
    r"^\s*(?:ROOT\s+)?%?(?P<name>[\w.\-]+)\s*=\s*(?P<shape>\S+)\s+(?P<opcode>[a-z][\w-]*)\(")
_COMP = re.compile(r"^\s*(?:%|ENTRY\s+)?(?P<comp>[\w.\-]+)\s*(?:\([^)]*\))?\s*(?:->.*)?\{\s*$")


def hlo_instructions(text: str) -> Iterator[dict]:
    """Parse an HLO module's text into one dict per instruction.

    Deliberately a text parser. The python HloModule object does not expose per-instruction
    metadata in a stable way across jaxlib versions, and the printed form is the same thing XLA's
    own tooling consumes.
    """
    comp = "<module>"
    for line in text.splitlines():
        m = _COMP.match(line)
        if m and "=" not in line:
            comp = m.group("comp")
            continue
        if line.strip() in ("}", ""):
            continue
        mi = _INSTR.match(line)
        if not mi:
            continue
        # `metadata()` and not a local regex: metadata mixes QUOTED values (op_name="...") with
        # UNQUOTED ints (stack_frame_id=5). A quoted-only pattern drops the int silently, which is
        # how source resolution appeared impossible at this level.
        meta = metadata(line)
        yield {
            "name": mi.group("name"),
            "opcode": mi.group("opcode"),
            "shape": mi.group("shape"),
            "computation": comp,
            "op_name": meta.get("op_name", ""),
            "source_file": meta.get("source_file", ""),
            "source_line": meta.get("source_line", ""),
            "stack_frame_id": meta.get("stack_frame_id", ""),
        }


def _to_ins(rec: dict, level: str, tab=None) -> Ins:
    src = rec["source_file"]
    if src:                                            # only with inline stack frames enabled
        site, fn = f"{src}:{rec['source_line']}", rec.get("function", "?")
    elif tab and rec.get("stack_frame_id"):
        site, fn = hlo_sites(int(rec["stack_frame_id"]), tab)
    else:
        site, fn = "<no-frame>", "?"
    return Ins(level, rec["opcode"], rec["op_name"],
               unit=rec["name"], container=rec["computation"], site=site, function=fn,
               fusion=rec["opcode"] == "fusion",
               outlined=rec["computation"] != "<module>")


def walk_hlo(compiled, *, level: str = "hlo_opt") -> Iterator[Ins]:
    """Every instruction of the OPTIMIZED HLO, with its scope path AND its source line.

    ``compiled`` is a ``jax.stages.Compiled``. Uses the executable's own printer rather than
    ``Compiled.as_text()`` so it does not depend on that method's default staying put.

    Source lines are resolved through the module's stack-frame tables (see the module docstring):
    metadata carries ``stack_frame_id``, not an inline path.
    """
    from .flags import hlo_text
    text = hlo_text(compiled)
    tab = frame_tables(text)
    for rec in hlo_instructions(text):
        yield _to_ins(rec, level, tab)


def _loc_aliases(text: str) -> dict[str, str]:
    """``{"#loc17": "jit(f)/mylib:lib.solve/tanh"}`` from a module's alias definitions.

    MLIR does not put the name on the operation. It emits ``loc(#loc17)`` on the op and defines
    ``#loc17 = loc("jit(f)/.../tanh"(#loc12))`` separately, often chaining through a callsite. The
    first quoted string in the definition is the name stack; anything else is the file/line of an
    inner frame.
    """
    out: dict[str, str] = {}
    for m in re.finditer(r'^(#loc\w*)\s*=\s*loc\((.*)\)\s*$', text, re.M):
        alias, body = m.group(1), m.group(2)
        q = re.search(r'"((?:[^"\\]|\\.)*)"', body)
        if q and not body.startswith("callsite"):
            out[alias] = q.group(1)
    return out


def walk_stablehlo(lowered) -> Iterator[Ins]:
    """Every StableHLO operation, with the name stack its location points at.

    TWO REASONS THIS IS NOT A ONE-LINE REGEX, both measured on jax 0.10.2:

    1. The accessor. ``Lowered.as_text()`` defaults to ``debug_info=False`` and prints no locations
       at all; ``compiler_ir('stablehlo')`` drops them too. Both return zero occurrences of a scope
       that is demonstrably present. :func:`scopex.stablehlo_text` passes ``debug_info=True``.
    2. The indirection. Operations carry ``loc(#loc17)``, NOT ``loc("name")``. An earlier version of
       this function matched only the inline form and therefore yielded **1 unit on 16 of 21 real
       programs**, including modules of 3,214 and 21,000 operations -- the level looked empty rather
       than broken, which is the worst way for an instrument to fail.
    """
    from .flags import stablehlo_text
    text = stablehlo_text(lowered)
    alias = _loc_aliases(text)
    op = re.compile(
        r'^\s*(?:%[\w#]+(?:\s*,\s*%[\w#]+)*\s*=\s*)?'          # optional result list
        r'(?P<op>[a-z_][\w.]*\.[\w.]+|return|func\.func)'         # dialect.op
        r'.*?\bloc\((?P<loc>#\w+|"(?:[^"\\]|\\.)*")\)')        # its location
    for line in text.splitlines():
        m = op.match(line)
        if not m:
            continue
        raw = m.group("loc")
        name = alias.get(raw, "") if raw.startswith("#") else raw.strip('"')
        yield Ins("stablehlo", m.group("op").split(".")[-1], name,
                  site="<see-jaxpr-level>")
