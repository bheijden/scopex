"""Hand back the text a number was parsed from, so a sceptic can check instead of believing.

Every derived number in this package comes out of text that XLA printed and scopex then threw
away. ``pass_timings`` returned a dict and dropped the glog stream; ``pass_growth`` read 78 MB of
per-pass snapshots, counted their instructions and dropped every byte. A reader who doubts the
ranking has, today, exactly one option: rerun the whole compile and hope it lands the same way.

That is the wrong default for a package whose stated rule is that a wrong method is worse than a
missing one. Four of this project's investigations ended in a confidently wrong pass name, and in
every one of them the RAW LOG contained the refutation in plain sight -- a line with ``min`` in it,
a pass name with a space in it. Handing the log back turns a twenty-minute rerun into a ``grep``.

WHY A PATH AND A LAZY ACCESSOR, NOT THE STRING. Measured on this machine, jax 0.10.2:

    ================================================  ==========  =============================
    artifact                                          size        what holding it would cost
    ================================================  ==========  =============================
    pass_timings log, 45 trivial programs, CPU          1.86 MB    per call, for the process
    pass_timings log, same programs, CUDA               4.54 MB    per call, for the process
    dump(passes=".*"), grad of a 128-step scan         78.3 MB    490 files, 64 snapshots
    largest single snapshot in that dump                4.94 MB    x64 if pass_growth kept them
    ================================================  ==========  =============================

The 78.3 MB is from a compile that takes 8.8 s; the corpus has arms that take 200 s and reach
176,189 instructions in one module. ``pass_growth`` already reads every one of those files, so
keeping the text would turn a bounded-memory instrument into one that holds the whole dump. A
:class:`Raw` is 200 bytes and opens the file when, and only when, somebody asks.

WHAT ``Raw`` PROMISES, AND WHAT IT CANNOT. It records the sha256 of the bytes AS PARSED, and
:meth:`Raw.verify` re-reads the file, re-hashes it and re-counts the witness. So it catches: a
file truncated or rotated under you, a temp directory reaped, the wrong artifact handed back, a
number that came from a different run than the text beside it. It does NOT catch a parser that is
wrong the same way twice -- re-running the same regex over the same bytes agrees with itself by
construction. For that the artifact itself is the instrument: read it.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import re
from typing import NamedTuple

__all__ = ["Raw", "raw_of", "raw_step"]

_HEAD = 8192


class Raw(NamedTuple):
    """A pointer to the bytes a scopex result was parsed from, plus what was true of them.

    Cheap to hold and cheap to pass around: nothing here reads the file until you call one of the
    accessors. ``sha256`` and the counts were taken at parse time, from the bytes the parser saw.
    """

    path: str
    kind: str                 # "vlog" | "hlo-snapshot" | "mlir-pass-log" | ...
    sha256: str               # of the bytes the parser was given
    size_bytes: int
    n_lines: int
    produced_by: str = ""     # the XLA function that printed it
    witness: str = ""         # the crude pattern whose count bounds a correct parse
    witness_count: int = 0    # how often it occurred, at parse time
    parsed_count: int = 0     # how many results the parser returned from it

    # ── reading it ──────────────────────────────────────────────────────────────────────────────
    def text(self, *, max_bytes: int | None = None) -> str:
        """The whole artifact. ``max_bytes`` truncates -- these files reach 5 MB each."""
        with open(self.path, errors="replace") as f:
            return f.read() if max_bytes is None else f.read(max_bytes)

    def lines(self):
        """Iterate lines lazily. The only accessor safe on an arbitrarily large artifact."""
        with open(self.path, errors="replace") as f:
            yield from f

    def head(self, n: int = 20) -> str:
        out = []
        for i, ln in enumerate(self.lines()):
            if i >= n:
                break
            out.append(ln.rstrip("\n"))
        return "\n".join(out)

    def tail(self, n: int = 20) -> str:
        keep: list[str] = []
        for ln in self.lines():
            keep.append(ln.rstrip("\n"))
            if len(keep) > n:
                keep.pop(0)
        return "\n".join(keep)

    def grep(self, pattern: str, *, limit: int = 200) -> list[tuple[int, str]]:
        """``[(1-based line number, line), ...]``. The point of the whole module.

        The bug that made ``pass_timings`` name the wrong pass was one line in 640 containing
        ``min``; ``raw.grep(" min ")`` finds it without a rerun.
        """
        pat = re.compile(pattern)
        out = []
        for i, ln in enumerate(self.lines(), 1):
            if pat.search(ln):
                out.append((i, ln.rstrip("\n")))
                if len(out) >= limit:
                    break
        return out

    # ── checking it ─────────────────────────────────────────────────────────────────────────────
    def exists(self) -> bool:
        return os.path.isfile(self.path)

    def verify(self) -> dict:
        """Re-read the artifact and check it is still the one these numbers came from.

        Returns ``{"ok": bool, "problems": [...], ...}``. ``ok`` is False when the file is gone,
        when its bytes no longer hash to what the parser saw, or when the crude witness now occurs
        more often than the parser returned results -- which is the exact shape of every parser bug
        this package has shipped.
        """
        r: dict = {"path": self.path, "ok": False, "problems": []}
        if not self.exists():
            r["problems"].append(
                f"the artifact is gone: {self.path}. scopex writes vlog captures under a temp "
                f"directory; if /tmp was reaped, rerun the measurement -- do not trust numbers "
                f"whose evidence cannot be produced.")
            return r
        h = hashlib.sha256()
        n_lines = 0
        n_wit = 0
        pat = re.compile(self.witness) if self.witness else None
        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        with open(self.path, errors="replace") as f:
            for ln in f:
                n_lines += 1
                if pat and pat.search(ln):
                    n_wit += 1
        r.update(sha256=h.hexdigest(), size_bytes=os.path.getsize(self.path), n_lines=n_lines,
                 witness_count=n_wit)
        if h.hexdigest() != self.sha256:
            r["problems"].append(
                f"sha256 changed: parsed {self.sha256[:16]}..., file is now {h.hexdigest()[:16]}"
                f"... The numbers beside this handle were computed from DIFFERENT BYTES.")
        if n_lines != self.n_lines:
            r["problems"].append(f"line count changed: parsed {self.n_lines}, file has {n_lines}")
        if pat and self.parsed_count and n_wit > self.parsed_count:
            r["problems"].append(
                f"the witness {self.witness!r} occurs {n_wit} times but the parser returned "
                f"{self.parsed_count} results. A parse that reads fewer lines than its input "
                f"visibly contains is the failure mode scopex._parse.expect exists to raise on.")
        r["ok"] = not r["problems"]
        return r

    def __repr__(self) -> str:
        w = f", {self.witness_count}x{self.witness!r}" if self.witness else ""
        return (f"Raw({self.kind} {self.size_bytes / 1e6:.2f} MB, {self.n_lines} lines{w}, "
                f"{self.path})")


def raw_step(step) -> Raw:
    """The snapshot a :class:`scopex.PassStep` was counted from.

    ``pass_growth`` already carries ``.path`` for every step, so the handover here costs one
    ``stat`` and one hash of a file that is on disk anyway -- and NOT the 78 MB of text that
    holding the snapshots would. The witness is the instruction-line pattern, so
    :meth:`Raw.verify` re-counts the module and disagrees loudly if the count beside it came from
    a different file.

    Nothing is cached: hashing a 4.94 MB snapshot is ~10 ms, and a stale digest is worse than a
    slow one.
    """
    return raw_of(step.path, "hlo-snapshot",
                  produced_by="xla/service/dump.cc (--xla_dump_hlo_pass_re)",
                  witness=r"^\s+%?[\w.-]+ = ", parsed_count=step.instrs)


def raw_of(path: str | os.PathLike, kind: str, *, produced_by: str = "", witness: str = "",
           parsed_count: int = 0, text: str | None = None) -> Raw:
    """Build a :class:`Raw` for a file, hashing it once.

    ``text`` lets a caller that already holds the bytes avoid a second read; it must be the exact
    content written to ``path``, because the digest is what makes the handle checkable.
    """
    p = pathlib.Path(path)
    if text is None:
        data = p.read_bytes()
        body = data.decode(errors="replace")
    else:
        body = text
        data = text.encode()
    n_wit = len(re.findall(witness, body)) if witness else 0
    return Raw(path=str(p), kind=kind, sha256=hashlib.sha256(data).hexdigest(),
               size_bytes=len(data), n_lines=body.count("\n") + (0 if body.endswith("\n") else 1),
               produced_by=produced_by, witness=witness, witness_count=n_wit,
               parsed_count=parsed_count)
