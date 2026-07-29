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


# ── SOURCE LOCATION AT THE HLO LEVEL ────────────────────────────────────────────────────────────
_MD_STR = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')
_MD_INT = re.compile(r'(\w+)=(-?\d+)(?![\w."])')


def metadata(line):
    i = line.find("metadata={")
    if i < 0:
        return {}
    d, j = 0, i + len("metadata=")
    for k in range(j, len(line)):
        if line[k] == "{":
            d += 1
        elif line[k] == "}":
            d -= 1
            if d == 0:
                body = line[j + 1:k]
                break
    else:
        body = line[j + 1:]
    out = dict(_MD_STR.findall(body))
    for a, b in _MD_INT.findall(body):
        out.setdefault(a, int(b))
    return out


def frame_tables(text):
    """The module-level FileNames / FunctionNames / FileLocations / StackFrames tables.

    jaxlib 0.10.2 HLO metadata has NO inline `source_file=`/`source_line=` unless you compile with
    `--xla_hlo_print_inline_stack_frames=true`; by default it carries `stack_frame_id=N` indexing
    these tables, and every frame has a `parent_frame_id`, i.e. the FULL python stack is there."""
    files, funcs, locs, frames, sect = {}, {}, {}, {}, None
    for ln in text.splitlines():
        t = ln.strip()
        if t in ("FileNames", "FunctionNames", "FileLocations", "StackFrames"):
            sect = t; continue
        if not t or sect is None:
            continue
        if sect in ("FileNames", "FunctionNames"):
            m = re.match(r'^(\d+)\s+"(.*)"$', t)
            if m:
                (files if sect == "FileNames" else funcs)[int(m.group(1))] = m.group(2)
            else:
                sect = None
        else:
            m = re.match(r"^(\d+)\s+\{(.*)\}$", t)
            if m:
                d = {k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", m.group(2))}
                (locs if sect == "FileLocations" else frames)[int(m.group(1))] = d
            else:
                sect = None
    return files, funcs, locs, frames


_JAXTREE = ("/site-packages/jax/", "/jax/_src/")


def hlo_sites(fid, tab, maxdepth=64):
    """stack_frame_id -> ('file:line', function). Innermost frame outside the jax tree -- the same
    filter `source_info_util.user_frames` applies at the jaxpr level, so the two levels join."""
    files, funcs, locs, frames = tab
    seen = set()
    while fid and fid in frames and fid not in seen and len(seen) < maxdepth:
        seen.add(fid)
        fr = frames[fid]
        lo = locs.get(fr.get("file_location_id", 0), {})
        f = files.get(lo.get("file_name_id", 0), "")
        if f and not any(s in f for s in _JAXTREE):
            return (f"{f.rsplit('/', 1)[-1]}:{lo.get('line', -1)}",
                    funcs.get(lo.get("function_name_id", 0), "?"))
        nxt = fr.get("parent_frame_id", 0)
        if nxt == fid:
            break
        fid = nxt
    return "?", "?"

