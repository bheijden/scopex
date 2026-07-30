"""XLA's own fusion decision log, read as the protobuf it is.

``<dump>/module_NNNN.<fn>.priority_fusion_dump.txt`` is a ``FusionProcessDumpProto`` written in
TEXT-PROTO form by the GPU priority-fusion pass. It records, in order, every decision the pass
made: which producer it refused to fuse and why, which producer/consumer pair it actually fused and
what it named the result, and every re-priced fusion candidate with its modelled runtime.

WHY THIS IS A PROTO PARSER AND NOT A REGEX. The obvious implementation greps for
``producer_name: "(.*)"`` and ``reason: "(.*)"``. That implementation knows the three step kinds
that exist today -- ``producer_ineligible``, ``fusion``, ``update_priority`` -- and a fourth one
added upstream would not appear in its output at all. It would return a shorter, entirely
plausible list. That is the failure this package keeps paying for.

Text-proto does not require that bargain, because unlike the binary wire format it is
SELF-DESCRIBING: the field names are in the file. :func:`parse_textproto` therefore needs no schema
at all, and an unknown step kind arrives as an unknown key rather than as silence. Nothing is
dropped because nothing is matched against a list of things worth keeping.

WHAT IS NOT AVAILABLE, AND WHY THAT IS FINE HERE. There is no ``hlo_pb2`` in jaxlib 0.10.2, no
``.proto`` file ships with it, and ``google.protobuf`` is not a jax dependency -- so the *schema*
is out of reach and there is no way to validate field names or types against it. For the BINARY
route (``HloModule.as_serialized_hlo_module_proto()``) that is disqualifying: raw wire format is
just field NUMBERS, so a schema-less reader has to hardcode them and guesses wrong silently. For
text-proto it costs only the type of a scalar, which :func:`_scalar` infers.

THE DUMP IS GPU-ONLY. priority-fusion is a GPU pass. On CPU the file is simply absent, which is not
an error and must not be read as "no fusion happened".
"""

from __future__ import annotations

import os
import re
from typing import Any, NamedTuple

__all__ = ["parse_textproto", "FusionStep", "fusion_dump", "fusion_steps", "fusion_summary",
           "fusion_consistency"]


# ── a schema-free text-proto reader ──────────────────────────────────────────────────────────────
# Grammar (proto3 text format, the subset XLA emits plus the alternatives it is allowed to emit):
#   field    := NAME ':' scalar | NAME '{' message '}' | NAME '<' message '>' | NAME ':' '[' ... ']'
#   scalar   := STRING+ | NUMBER | BAREWORD          (adjacent STRINGs concatenate)
# Repeated fields are repetitions of the same NAME and become a list.

_TOKEN = re.compile(r"""
      (?P<ws>\s+)
    | (?P<comment>\#[^\n]*)
    | (?P<string>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
    | (?P<punct>[:{}<>\[\],])
    | (?P<bare>[^\s:{}<>\[\],\#]+)
""", re.X)

_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "'": "'",
            "a": "\a", "b": "\b", "f": "\f", "v": "\v", "?": "?", "0": "\0"}


class TextProtoError(ValueError):
    """Raised when a text-proto will not parse. Deliberately loud: the alternative is an empty dict
    that reads as 'this compile made no fusion decisions'."""


def _unquote(tok: str) -> str:
    body, out, i = tok[1:-1], [], 0
    while i < len(body):
        c = body[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        i += 1
        if i >= len(body):
            break
        e = body[i]
        # OCTAL IS TESTED FIRST, AND THE ORDER IS THE WHOLE POINT. Proto text format escapes every
        # non-printable byte as a THREE-DIGIT octal escape, so `\021` is byte 17. `_ESCAPES` has a
        # "0" key for the `\0` spelling of NUL; when that lookup ran first, `\021` matched it,
        # emitted NUL and left "21" as two literal characters -- so every byte in `\000`-`\077`
        # decoded as three characters instead of one, silently, shifting every offset after it.
        # Bytes `\100`-`\377` were unaffected, which is why it survived: invisible on ASCII-only
        # dumps, appearing only in embedded binary (a serialised `Any`). Octal-first is also
        # correct for the `\0` spelling, since `int("0", 8) == 0`.
        if e.isdigit():                                  # \NNN octal (3 digits, in practice)
            m = re.match(r"[0-7]{1,3}", body[i:])
            out.append(chr(int(m.group(0), 8)))
            i += len(m.group(0))
        elif e in _ESCAPES:
            out.append(_ESCAPES[e])
            i += 1
        elif e == "x":                                   # \xHH
            m = re.match(r"[0-9a-fA-F]{1,2}", body[i + 1:])
            out.append(chr(int(m.group(0), 16)) if m else "x")
            i += 1 + (len(m.group(0)) if m else 0)
        else:
            out.append(e)
            i += 1
    return "".join(out)


def _scalar(tok: str) -> Any:
    """A bare token's python value. Enums and other barewords stay strings, which is correct --
    without the schema there is nothing better to turn them into, and a string round-trips."""
    if tok in ("true", "false"):
        return tok == "true"
    try:
        return int(tok, 0)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        return tok


def _lex(text: str) -> list[tuple[str, str]]:
    out, i, n = [], 0, len(text)
    while i < n:
        m = _TOKEN.match(text, i)
        if not m or m.end() == i:
            raise TextProtoError(f"cannot tokenise at offset {i}: {text[i:i + 40]!r}")
        i = m.end()
        kind = m.lastgroup
        if kind in ("ws", "comment"):
            continue
        out.append((kind, m.group()))
    return out


def parse_textproto(text: str) -> dict:
    """A text-format protobuf as nested dicts, WITHOUT a schema.

    Repeated fields become lists; a field that appears once is scalar, so callers should use
    :func:`as_list` rather than assuming. Field names and nesting come from the file itself, so a
    message this code has never heard of round-trips into the result instead of vanishing.

    Raises :class:`TextProtoError` rather than returning ``{}`` on malformed input.
    """
    toks = _lex(text)
    pos = 0

    def add(msg: dict, key: str, val: Any) -> None:
        if key in msg:
            if not isinstance(msg[key], list):
                msg[key] = [msg[key]]
            msg[key].append(val)
        else:
            msg[key] = val

    def message(depth: int, closing: str | None) -> dict:
        nonlocal pos
        if depth > 100:
            raise TextProtoError("text-proto nested past 100 levels; refusing to recurse")
        msg: dict = {}
        while pos < len(toks):
            kind, tok = toks[pos]
            if closing and kind == "punct" and tok == closing:
                pos += 1
                return msg
            if kind != "bare" and kind != "string":
                raise TextProtoError(f"expected a field name, got {tok!r}")
            key = tok
            pos += 1
            if pos >= len(toks):
                raise TextProtoError(f"field {key!r} has no value")
            k2, t2 = toks[pos]
            if k2 == "punct" and t2 in "{<":
                pos += 1
                add(msg, key, message(depth + 1, "}" if t2 == "{" else ">"))
                continue
            if k2 == "punct" and t2 == ":":
                pos += 1
                if pos >= len(toks):
                    raise TextProtoError(f"field {key!r} has no value")
                k3, t3 = toks[pos]
                if k3 == "punct" and t3 == "[":                    # short repeated
                    pos += 1
                    while pos < len(toks) and toks[pos][1] != "]":
                        if toks[pos][1] == ",":
                            pos += 1
                            continue
                        kx, tx = toks[pos]
                        add(msg, key, _unquote(tx) if kx == "string" else _scalar(tx))
                        pos += 1
                    pos += 1
                    continue
                if k3 == "punct" and t3 in "{<":                   # `field: { ... }` is legal
                    pos += 1
                    add(msg, key, message(depth + 1, "}" if t3 == "{" else ">"))
                    continue
                if k3 == "string":                                 # adjacent strings concatenate
                    parts = []
                    while pos < len(toks) and toks[pos][0] == "string":
                        parts.append(_unquote(toks[pos][1]))
                        pos += 1
                    add(msg, key, "".join(parts))
                    continue
                add(msg, key, _scalar(t3))
                pos += 1
                continue
            raise TextProtoError(f"expected ':' or '{{' after field {key!r}, got {t2!r}")
        if closing:
            raise TextProtoError(f"unclosed message, expected {closing!r}")
        return msg

    return message(0, None)


def as_list(v) -> list:
    """A repeated field's value as a list whether it repeated once, many times, or not at all.
    Text-proto gives no way to tell 'repeated with one element' from 'singular', so every caller
    needs this and forgetting it is a silent single-element bug."""
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


# ── the typed view of a fusion dump ──────────────────────────────────────────────────────────────

class FusionStep(NamedTuple):
    """One decision, in the order the pass made it.

    ``kind`` is the text-proto field name of the step's oneof arm -- ``"fusion"``,
    ``"producer_ineligible"``, ``"update_priority"``, or whatever XLA adds next, because it is read
    off the file rather than matched against a list. ``fields`` is that arm's whole message, so
    nothing is lost by this projection.
    """
    index: int
    kind: str
    producer: str
    consumer: str
    reason: str
    fields: dict

    @property
    def fused(self) -> bool:
        return self.kind == "fusion"


def fusion_dump(path: str | os.PathLike) -> dict:
    """The whole ``priority_fusion_dump.txt`` as nested dicts. Raises if it will not parse."""
    with open(path, errors="replace") as fh:
        return parse_textproto(fh.read())


def fusion_steps(source) -> list[FusionStep]:
    """Every decision in a fusion dump, in pass order.

    ``source`` is a path to a ``priority_fusion_dump.txt``, or an already-parsed dict.
    """
    d = source if isinstance(source, dict) else fusion_dump(source)
    out: list[FusionStep] = []
    for n, step in enumerate(as_list(d.get("fusion_steps"))):
        # The step is a oneof: exactly one field is set, and its NAME is the decision kind. Reading
        # the name off the message is what makes an unrecognised kind visible instead of dropped.
        for kind, body in step.items():
            b = body if isinstance(body, dict) else {}
            out.append(FusionStep(
                index=n, kind=kind,
                producer=str(b.get("producer_name", "")),
                consumer=str(b.get("consumer_name", b.get("consumer_names", ""))),
                reason=str(b.get("reason", "")),
                fields=b))
    return out


def fusion_summary(source) -> dict:
    """What the priority-fusion pass did, and what it declined to do.

    ``kinds`` counts every step kind PRESENT IN THE FILE, so a kind this module has never seen
    still shows up in the count -- the census cannot go quietly out of date.
    """
    import collections
    d = source if isinstance(source, dict) else fusion_dump(source)
    steps = fusion_steps(d)
    refused: collections.Counter = collections.Counter()
    fused = []
    for s in steps:
        if s.kind == "producer_ineligible":
            refused[s.reason] += 1
        elif s.fused:
            fused.append((s.producer, s.consumer, str(s.fields.get("fusion_name", ""))))
    dev = d.get("gpu_device_info", {})
    return {
        "steps": len(steps),
        "kinds": dict(collections.Counter(s.kind for s in steps)),
        "fusions": fused,
        "refusals": dict(refused.most_common()),
        "device": dev.get("name", "") if isinstance(dev, dict) else "",
        "has_module_before": bool(d.get("hlo_module_before_fusion")),
    }


# ── is the decision log internally consistent, and is it about the module you think? ──────────────

def fusion_consistency(source, before_snapshot: str | None = None) -> dict:
    """Check a priority-fusion decision log against evidence, and return the evidence.

    THE CHECK THAT DOES NOT WORK, stated first because it is the obvious one. "Every fusion the log
    says it made is in the module afterwards" is NOT an invariant. Priority fusion is iterative: a
    producer it refuses at step 3 can be fused at step 90 once its bitcast users are gone, and a
    fusion it makes early can be absorbed into a later one. Measured on ``xtile_issue`` with a
    correct parse: 2 of 22 claimed fusions and 19 of 101 refused producers are absent from the
    after-pass snapshot. A check that fires there is a false-alarm generator, and this package has
    already paid for one instrument that cried wolf.

    THE TWO THAT DO.

    1. CAUSAL CLOSURE. Every instruction a step names must either be in the module the pass STARTED
       from, or have been created by an EARLIER step in this same log. The proto carries that
       starting module itself, as HLO text, in ``hlo_module_before_fusion``. This is sensitive to
       ORDER, not just to contents, so it is the check that notices if the reader loses a step or
       shuffles a repeated field -- which a schema-free text-proto reader could silently do.
       Measured on ``xtile_issue``: 129 steps, 20 fusions created, 0 forward references.

    2. THE MODULE AGREEMENT (only when ``before_snapshot`` is given). The embedded module's
       instruction names must equal those of the last per-pass snapshot the PIPELINE wrote before
       this pass. Two writers, one module, no shared code -- and it is what says the log you are
       reading belongs to the module you think it does, which no amount of internal consistency can.
       Measured on ``xtile_issue``: 184 names, 0 either way.

    Returns a dict; ``consistent`` is the boolean and everything else is the working.
    """
    # HLO text is parsed by `_parse` and nowhere else -- see tests/test_parse_quarantine.py. This
    # module owns the text-proto grammar; it does not own the HLO grammar.
    from ._parse import hlo_instruction_names

    d = source if isinstance(source, dict) else fusion_dump(source)
    steps = fusion_steps(d)
    before = d.get("hlo_module_before_fusion")

    def names(text: str) -> set:
        return set(hlo_instruction_names(text))

    start = names(before) if isinstance(before, str) and before else set()
    created: set = set()
    forward: list = []
    for s in steps:
        for n in (s.producer, s.consumer):
            if n and start and n not in start and n not in created:
                forward.append((s.index, s.kind, n))
        if s.fused:
            fn = str(s.fields.get("fusion_name", ""))
            if fn:
                created.add(fn)

    agree = None
    if before_snapshot and start:
        with open(before_snapshot, errors="replace") as fh:
            disk = names(fh.read())
        agree = {"snapshot": os.path.basename(before_snapshot),
                 "embedded_names": len(start), "snapshot_names": len(disk),
                 "embedded_only": sorted(start - disk)[:8],
                 "snapshot_only": sorted(disk - start)[:8],
                 "match": start == disk}

    return {
        "steps": len(steps),
        "has_start_module": bool(start),
        "start_instructions": len(start),
        "fusions_created": len(created),
        "forward_references": forward,
        "closed": bool(start) and not forward,
        "module_agreement": agree,
        # `None` for "not checked", never False -- a check that could not run must not read as a
        # pass, and must not read as a failure either.
        "consistent": (not forward and (agree is None or agree["match"])) if start else None,
    }
