"""The other three levels: StableHLO, HLO, and optimized HLO.

The jaxpr is where provenance is richest, and it is also the level furthest from what actually
takes the compile time. Everything here exists so a question can be asked at the level that can
answer it, and so the answers can be COMPARED.

WHAT SURVIVES, MEASURED. XLA rewrites instruction names freely, so joining levels on instruction
name barely works at all. What does survive is the ``op_name`` metadata, which carries the rendered
JAX name stack verbatim::

    op_name="jit(f)/mylib:lib.solve/mylib:user.Col.residual/tanh"

That string is present on optimized-HLO instructions, and it is the whole basis of cross-level
attribution: measured across eight real programs the op_name join ran 0.9976-1.0000. Note what it
does NOT contain -- on the backends checked there is no ``source_file``/``source_line`` in the
optimized module, so below the jaxpr the SCOPE is the carrier and the source line is not. That is
why a framework's marks earn their keep here and a traceback cannot reach.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from .records import Ins

__all__ = ["walk_hlo", "walk_stablehlo", "hlo_instructions"]

# `%add.3 = f32[16,16]{1,0} add(%a, %b), metadata={op_name="jit(f)/.../add"}`
# The leading `%` is absent on some lines and ROOT may prefix the name.
_INSTR = re.compile(
    r"^\s*(?:ROOT\s+)?%?(?P<name>[\w.\-]+)\s*=\s*(?P<shape>\S+)\s+(?P<opcode>[a-z][\w-]*)\(")
_META = re.compile(r'metadata=\{(?P<body>[^}]*)\}')
_FIELD = re.compile(r'(\w+)="([^"]*)"')
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
        meta = {}
        mm = _META.search(line)
        if mm:
            meta = dict(_FIELD.findall(mm.group("body")))
        yield {
            "name": mi.group("name"),
            "opcode": mi.group("opcode"),
            "shape": mi.group("shape"),
            "computation": comp,
            "op_name": meta.get("op_name", ""),
            "source_file": meta.get("source_file", ""),
            "source_line": meta.get("source_line", ""),
        }


def _to_ins(rec: dict, level: str) -> Ins:
    src = rec["source_file"]
    site = f"{src}:{rec['source_line']}" if src else "<no-source-metadata>"
    return Ins(level, rec["opcode"], rec["op_name"],
               unit=rec["name"], container=rec["computation"], site=site,
               fusion=rec["opcode"] == "fusion",
               outlined=rec["computation"] != "<module>")


def walk_hlo(compiled, *, level: str = "hlo_opt") -> Iterator[Ins]:
    """Every instruction of the OPTIMIZED HLO, with its scope path.

    ``compiled`` is a ``jax.stages.Compiled``. Uses the executable's own printer rather than
    ``Compiled.as_text()`` so it does not depend on that method's default staying put.
    """
    from .flags import hlo_text
    for rec in hlo_instructions(hlo_text(compiled)):
        yield _to_ins(rec, level)


def walk_stablehlo(lowered) -> Iterator[Ins]:
    """Every StableHLO operation carrying a location.

    NOTE the accessor: ``Lowered.as_text()`` defaults to ``debug_info=False`` and prints no
    locations at all, and ``compiler_ir('stablehlo')`` drops them too. Both were measured returning
    zero occurrences of a scope that is demonstrably present.
    """
    from .flags import stablehlo_text
    txt = stablehlo_text(lowered)
    # StableHLO carries provenance as `loc("name"(...))` suffixes rather than HLO metadata.
    pat = re.compile(r'^\s*(?:%\S+\s*=\s*)?"?(?P<op>[\w.]+)"?.*?loc\("(?P<loc>[^"]*)"')
    for line in txt.splitlines():
        m = pat.match(line)
        if not m:
            continue
        yield Ins("stablehlo", m.group("op").split(".")[-1], m.group("loc"),
                  site="<no-source-metadata>")
