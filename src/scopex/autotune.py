"""What XLA:GPU's autotuner spent its seconds on -- and three ways to check the answer.

WHY THIS EXISTS. On every other arm of this corpus a slow compile is slow because XLA transformed
IR: more instructions, more passes, superlinear pass cost. On GPU there is a second mechanism that
looks nothing like that. ``GpuConvAlgorithmPicker`` and the fusion autotuners BENCHMARK candidate
kernels on real buffers at compile time, so the compile is slow because the *machine ran code*, not
because the compiler thought hard. The instruction count, the pass count, the opcode histogram and
the PTX are all identical between a fast and a slow arm -- verified to that depth in
``docs/INVESTIGATIONS.md`` D1 -- and none of them move.

WHAT ``pass_timings`` ALONE CANNOT TELL YOU, AND WHY THAT MATTERS. The autotuner is registered as an
HLO pass, so its seconds land inside a pass timer and :class:`scopex.Coverage` reads ~100%.
Measured: ``autotuner`` 52.05 s of a 52.22 s backend, coverage 99.75%. That is arithmetically
correct and it routes the reader wrong, because ``Coverage``'s band then says "PASS-BOUND -- the HLO
passes ARE the compile. Rank them; the top entry is the answer." The top entry IS the answer, but a
reader who follows a pass name to :func:`scopex.passmap.pass_source` lands in an XLA source
file that does
not contain the seconds. ``coverage`` answers "were the seconds inside a pass timer"; it was never
able to answer "were the seconds spent transforming IR". On CPU those coincide. On GPU they do not,
and this module is the difference.

THE ARTIFACT. ``--xla_gpu_dump_autotune_logs_to=FILE`` writes an ``xla.AutotuningLogs`` text-proto:
for every autotuned instruction, EVERY candidate that was tried and the ``run_time`` XLA measured
for it. That is the compiler's own GPU-clock accounting of the interval its host-side pass timer
was also measuring, produced by a different subsystem, and this package had never read it. It plays
exactly the role ``(cumulative:, max:, #called:)`` plays for :mod:`scopex.coverage`.

THE CHECKS, none of which shares code with the reading it checks. Each is followed by the negative
control that was measured to make it fire; all five live in ``tests/test_autotune_guard.py``.

1. KERNEL SHARE -- ``sum(candidate run_time)`` against the ``autotuner`` pass seconds from the VLOG.
   Two XLA subsystems, one interval. This is the measurement AND the check at once, and it has no
   single correct value: 49.7% on ``convT64_dilate16`` and 0.11% on ``gemm_shapes_k16``, and that
   difference is the finding, not an error. What IS an error is exceeding 1.0 -- and that check has
   fired in anger, on a dump directory XLA had APPENDED a second compile into (1.51).
2. WINNERS -- every algorithm XLA recorded as chosen in ``--xla_gpu_dump_autotune_results_to`` must
   appear among the candidates parsed out of the logs file. Pure containment, so it fires when a
   candidate is DROPPED however the seconds read.
3. ARGMIN -- and it must be within :attr:`Autotune.SELECTION_TOLERANCE` of the fastest candidate
   parsed for some instruction. A margin, not equality, because XLA's GEMM autotuner really does
   pass over a 3.8%-faster cuBLAS candidate to keep Triton; see :func:`winner_slowdown`.
4. IDENTITIES -- no instruction may hold two candidates with the same identity and different
   measured times. This is the one check that survives a key function which fails to recognise a
   backend, because such a collapse also collapses the recorded winner and check 3 would pass.
5. NAMES -- each instruction name decoded from the logs file must appear as a module name in the
   VLOG, because the autotuner compiles every candidate as its own HLO module named after the
   instruction. This is what makes the one unavoidable guess in this module checkable; see
   :func:`instr_name`.

READ ``kernel_share`` BEFORE THE RANKING, for the same reason ``Coverage`` must be read before
``passes``, and the corpus makes the case better than an argument can. ``convT64_dilate16`` and
``gemm_shapes_k16`` produce the SAME headline from ``pass_timings``: ``autotuner`` is 98% of the
compile, coverage says PASS-BOUND, fidelity ~1.0. They have opposite causes. On the first, 49.7% of
a 51.9 s pass is measured kernel execution -- 18 cuDNN algorithms benchmarked on real 134 MB buffers
for 25.8 s, the slowest single candidate taking 6.1 s and the winner 0.44 s. On the second, 0.11% is
kernel execution: 456 Triton candidates that run for 20 ms in total, so the 17.8 s is spent
COMPILING them. Same pass name, same band, a 450x difference in what to do about it, and nothing in
``pass_timings`` distinguishes them.

THE RESIDUAL IS PART OF THE ANSWER. ``pass_s - candidate_s`` is 26.1 s on the conv arm and this
module does not pretend to attribute it; the artifact simply does not carry it. What can be said is
that it is not more kernel time, because the VLOG independently shows exactly 18 candidate
sub-module compiles for that instruction, matching the 18 candidates recorded.

THIS IS GPU-ONLY. On CPU the files are absent and that is not an error.
"""

from __future__ import annotations

import os
from typing import Any, NamedTuple

from .fusion import as_list, parse_textproto

__all__ = [
    "Candidate", "Autotuned", "Autotune",
    "autotune_logs", "autotune_results", "autotune_report", "autotune_cost",
    "instr_name", "merge_by_instruction", "winner_matching", "winner_slowdown", "key_collisions",
]


# ── the one guess in this module, and the evidence for it ────────────────────────────────────────
# `AutotuningLogs.logs[].instr` is a `google.protobuf.Any` wrapping a serialised
# `xla.HloInstructionProto`. jaxlib 0.10.2 ships no `.proto` and no `hlo_pb2`, so the schema is out
# of reach and the binary wire format carries field NUMBERS only. Reading it therefore requires
# guessing that field 1 is `name` and field 2 is `opcode`.
#
# `scopex.fusion`'s docstring calls exactly that bargain disqualifying, and it is right to -- for an
# UNCHECKABLE guess. This one is checkable, which is the only reason it is here: the autotuner
# compiles each candidate as its own HLO module named after the instruction, so every name decoded
# this way must also appear as a module header in the VLOG, a stream that shares nothing with the
# proto. `Autotune.names_confirmed` reports that fraction and `names_ok` is the check. If a future
# XLA renumbers the fields, the decoded strings stop matching module names and the check fires
# instead of the module silently mislabelling every instruction.
#
# The wire WALK below needs no schema at all -- a proto tag encodes its own field number and wire
# type -- so an unknown field round-trips into `fields` rather than derailing the parse.

def _varint(b: bytes, i: int) -> tuple[int, int]:
    v = s = 0
    while i < len(b):
        c = b[i]
        v |= (c & 0x7F) << s
        i += 1
        if not c & 0x80:
            return v, i
        s += 7
        if s > 70:
            raise ValueError("varint too long")
    raise ValueError("truncated varint")


def proto_fields(b: bytes) -> dict[int, list]:
    """``{field_number: [values]}`` for one serialised proto message, WITHOUT a schema.

    Length-delimited fields come back as ``bytes``, varints as ``int``. Nothing is matched against a
    list of known fields, so a field this code has never seen is present in the result.
    """
    out: dict[int, list] = {}
    i = 0
    while i < len(b):
        tag, i = _varint(b, i)
        fn, wt = tag >> 3, tag & 7
        if fn == 0:
            raise ValueError(f"field number 0 at offset {i}")
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 2:
            n, i = _varint(b, i)
            if i + n > len(b):
                raise ValueError("length-delimited field runs past the end")
            v, i = b[i:i + n], i + n
        elif wt == 5:
            v, i = b[i:i + 4], i + 4
        elif wt == 1:
            v, i = b[i:i + 8], i + 8
        else:
            raise ValueError(f"wire type {wt} is not readable without a schema")
        out.setdefault(fn, []).append(v)
    return out


def _payload(any_msg: Any) -> bytes:
    """The bytes inside a text-proto ``Any``.

    ``parse_textproto`` returns the ``value`` as a str of code points 0-255 (it decodes octal
    escapes to characters), so latin-1 is the round trip, not utf-8.
    """
    if not isinstance(any_msg, dict):
        return b""
    v = any_msg.get("value", "")
    if isinstance(v, bytes):
        return v
    return v.encode("latin-1", "replace")


def instr_name(entry: dict) -> tuple[str, str]:
    """``(name, opcode)`` for one ``logs`` entry, or ``("", "")`` if the payload will not walk.

    Guessed field numbers -- see the note above. Never raises: a failure here must not take down a
    timing report, and :attr:`Autotune.names_ok` is where it becomes visible.
    """
    try:
        f = proto_fields(_payload(entry.get("instr")))
    except Exception:
        return ("", "")

    def s(n):
        v = f.get(n, [b""])[0]
        return v.decode("utf-8", "replace") if isinstance(v, bytes) else ""
    return (s(1), s(2))


# ── candidates ───────────────────────────────────────────────────────────────────────────────────

def _duration_s(d: Any) -> float:
    """A ``google.protobuf.Duration`` in seconds.

    BOTH FIELDS, DELIBERATELY. Duration splits a time into ``seconds`` and ``nanos``, and proto3
    omits whichever is zero, so a sub-second value has no ``seconds`` key and a reader that only
    ever saw small candidates will happily ship ``nanos``-only. Measured cost of that omission on
    ``convT64_dilate16``: 51.05 s read as 18.05 s, silently, because the slow candidates are the
    ones with whole seconds. That is the ``min``-units bug in a different costume, so it gets the
    same treatment -- read every magnitude the format has.
    """
    if not isinstance(d, dict):
        return 0.0
    return float(d.get("seconds", 0) or 0) + float(d.get("nanos", 0) or 0) / 1e9


# Everything on a candidate message that describes the MEASUREMENT rather than the CANDIDATE.
# Identity is "all the other fields", so a backend kind this module has never heard of becomes a
# new key instead of colliding with every other unknown.
_MEASUREMENT_FIELDS = frozenset({"run_time", "scratch_bytes", "failure"})


def _canon(v):
    if isinstance(v, dict):
        return tuple((k, _canon(x)) for k, x in sorted(v.items()))
    if isinstance(v, list):
        return tuple(_canon(x) for x in v)
    return v


def _key(msg: dict) -> tuple:
    """A candidate's IDENTITY: every field that is not a measurement, read OFF THE MESSAGE.

    NOT AN ENUMERATION OF BACKENDS, and the first draft of this function was one -- it knew
    ``algorithm`` (cuDNN) and ``other`` (the emitters) and returned a single ``("none",)`` for
    everything else. On the GEMM arms that silently collapsed all 25 Triton configs of an
    instruction into one key, because Triton candidates identify themselves through
    ``triton {block_m, block_n, block_k, num_stages, num_warps, num_ctas}`` and cuBLAS through
    ``gemm {algorithm, autotune_workspace_size}``. The argmin check caught it; the fix is to stop
    naming the kinds. :mod:`scopex.fusion`'s docstring makes this exact argument about step kinds
    and it applies unchanged here.

    Used only for EQUALITY, never for timing, which is what keeps the argmin check unit-free.
    """
    if not isinstance(msg, dict):
        return ("none",)
    out = tuple((k, _canon(v)) for k, v in sorted(msg.items())
                if k not in _MEASUREMENT_FIELDS)
    return out or ("none",)


class Candidate(NamedTuple):
    """One candidate kernel the autotuner tried, and how long XLA measured it running."""
    seconds: float
    key: tuple
    label: str
    failed: bool
    scratch_bytes: int


class Autotuned(NamedTuple):
    """Every candidate tried for one instruction, in the order XLA tried them."""
    index: int
    name: str
    opcode: str
    candidates: list[Candidate]

    @property
    def seconds(self) -> float:
        return sum(c.seconds for c in self.candidates)

    @property
    def contenders(self) -> list[Candidate]:
        """The candidates that could actually have won.

        A DISQUALIFIED candidate carries ``failure {kind: ..., msg: ...}`` and an EMPTY
        ``run_time``, so it reads as 0 seconds -- i.e. as the fastest thing in the list. XLA treats
        it as infinity. Measured on ``gemm_shapes_k16``: 56 of 456 Triton configs disqualified for
        shared-memory overflow or register spilling, every one of them at 0 s. Leaving them in made
        the winner check fail on 27 of 46 instructions, and the instrument would have been "broken"
        because of a semantic it got wrong, not because XLA did anything unusual.
        """
        return [c for c in self.candidates if not c.failed]

    @property
    def best(self) -> Candidate | None:
        return min(self.contenders, key=lambda c: c.seconds, default=None)

    @property
    def argmin_keys(self) -> set:
        """Every contending candidate key tied for fastest. A SET because ties are real and XLA
        breaks them by order -- two candidates at exactly 5120 ns is the measured case."""
        cs = self.contenders
        if not cs:
            return set()
        b = min(c.seconds for c in cs)
        return {c.key for c in cs if c.seconds <= b + 1e-12}

    def __repr__(self) -> str:                                          # pragma: no cover
        return (f"Autotuned({self.name!r} {self.opcode} "
                f"{len(self.candidates)} candidates {self.seconds:.4f}s)")


def _label(msg: dict) -> str:
    """A short human name for a candidate. Falls back to the FIELD NAME for kinds not spelled out
    here, so an unrecognised backend prints as itself rather than as ``?``."""
    if not isinstance(msg, dict):
        return "?"
    fields = [k for k in msg if k not in _MEASUREMENT_FIELDS]
    if not fields:
        return "?"
    k = fields[0]
    v = msg[k] if isinstance(msg.get(k), dict) else {}
    if k == "algorithm":
        return f"cudnn algo {v.get('algo_id')}"
    if k == "other":
        return str(v.get("name", "other"))
    if k == "triton":
        return (f"triton {v.get('block_m')}x{v.get('block_n')}x{v.get('block_k')}"
                f"/{v.get('num_stages')}st{v.get('num_warps')}w")
    if k == "gemm":
        return f"cublas algo {v.get('algorithm')}"
    return k


def autotune_logs(source) -> list[Autotuned]:
    """Every candidate the autotuner benchmarked, from ``--xla_gpu_dump_autotune_logs_to``.

    ``source`` is a path, the file's text, or an already-parsed dict.
    """
    d = _load(source)
    out = []
    for i, e in enumerate(as_list(d.get("logs"))):
        if not isinstance(e, dict):
            continue
        name, opcode = instr_name(e)
        cands = [Candidate(seconds=_duration_s(c.get("run_time")), key=_key(c), label=_label(c),
                           failed="failure" in c, scratch_bytes=int(c.get("scratch_bytes", 0) or 0))
                 for c in as_list(e.get("results")) if isinstance(c, dict)]
        out.append(Autotuned(index=i, name=name, opcode=opcode, candidates=cands))
    return out


def autotune_results(source) -> list[dict]:
    """What XLA CHOSE, from ``--xla_gpu_dump_autotune_results_to``. One entry per instruction."""
    d = _load(source)
    out = []
    for r in as_list(d.get("results")):
        if not isinstance(r, dict):
            continue
        res = r.get("result", {})
        hlo = str(r.get("hlo", ""))
        out.append({"key": _key(res), "label": _label(res), "hlo": hlo,
                    # A fusion's `hlo` is a computation body in braces; a custom-call's is one
                    # instruction. That is the only instruction-identity this file carries, and it
                    # is what the results are paired to the logs by.
                    "is_computation": hlo.lstrip().startswith("{"),
                    "device": str(r.get("device", ""))})
    return out


def _load(source):
    if isinstance(source, dict):
        return source
    s = str(source)
    if "\n" not in s and len(s) < 4096 and os.path.exists(s):
        with open(s, errors="replace") as fh:
            s = fh.read()
    return parse_textproto(s)


# ── the report ───────────────────────────────────────────────────────────────────────────────────

class Autotune(dict):
    """How much of a GPU compile was spent RUNNING candidate kernels, and whether to believe it.

    Every field is a plain dict key so the whole thing is JSON-able; the properties below are the
    derived numbers and the checks.
    """

    # ── the measurement ──────────────────────────────────────────────────────────────────────
    @property
    def kernel_share(self) -> float | None:
        """Candidate run_time / ``autotuner`` pass seconds.

        THE MEASUREMENT AND THE CROSS-CHECK AT ONCE, and it has no correct value. High means the
        pass really was running kernels; low means the pass spent its time on its own overhead
        (compiling candidates, allocating and redzone-checking buffers). Only >1 is impossible.
        """
        p = self.get("pass_s")
        if not p:
            return None
        return self.get("candidate_s", 0.0) / p

    @property
    def compile_share(self) -> float | None:
        """Candidate run_time as a fraction of the whole backend compile -- the headline number."""
        b = self.get("backend_s")
        if not b:
            return None
        return self.get("candidate_s", 0.0) / b

    @property
    def redundant_s(self) -> float:
        """Seconds spent benchmarking an instruction that had ALREADY been benchmarked.

        Not a defect in this reader -- a property of the compile. XLA's conv picker asks cuDNN for
        candidates in more than one heuristics mode and the lists overlap, so the same algorithm is
        timed again. On ``convT64_dilate16`` this is 25.86 s of a 52.05 s pass.
        """
        seen: dict[str, bool] = {}
        tot = 0.0
        for a in self.get("instructions", []):
            if a.name and a.name in seen:
                tot += a.seconds
            seen[a.name] = True
        return tot

    # ── the checks ───────────────────────────────────────────────────────────────────────────
    #: How much slower than the parsed minimum a recorded winner may be before the PARSE is
    #: suspected rather than XLA's selection policy. The measured populations are 1.000-1.038
    #: (policy) and 6.44 (a Duration read with its ``seconds`` field missing), and the interval
    #: between them is empty on this corpus -- so the exact value is not load-bearing, but the
    #: margin is only 5x wide and a wider corpus could narrow it. It is not 1.0 because the GEMM
    #: autotuner really does pass over a faster cuBLAS candidate to keep Triton.
    SELECTION_TOLERANCE = 1.25

    @property
    def argmin_ok(self) -> bool | None:
        """Is every recorded winner within :attr:`SELECTION_TOLERANCE` of a parsed minimum?

        ``None`` when there are no results to check against. See :func:`winner_slowdown` for why
        this is a margin and not equality.
        """
        if not self.get("n_paired", 0):
            return None
        w = self.get("winner_slowdown")
        return True if w is None else w <= self.SELECTION_TOLERANCE

    @property
    def winner_slowdown(self) -> float | None:
        """The worst recorded winner's time / the fastest candidate parsed. See the free function
        of the same name; ~1.0 means XLA took the minimum, and a large value means the parse did
        not read the same numbers XLA did."""
        return self.get("winner_slowdown")

    @property
    def winners_ok(self) -> bool | None:
        """Does every algorithm XLA recorded as a winner appear among the candidates scopex parsed?

        Containment, weaker than the matching but with a different blind spot: it fires when a
        candidate is DROPPED, whatever the seconds say.
        """
        n = len(self.get("results", []))
        return None if not n else self.get("winners_found", 0) == n

    @property
    def keys_ok(self) -> bool:
        """No instruction has two differently-timed candidates sharing one identity."""
        return self.get("key_collisions", 0) == 0

    @property
    def names_ok(self) -> bool | None:
        """Did the instruction names decoded from the binary ``Any`` show up in the VLOG?

        The check on this module's one guessed field number. ``None`` without a VLOG to check
        against.
        """
        f = self.get("names_confirmed")
        return None if f is None else f >= 0.999

    @property
    def share_ok(self) -> bool:
        """Candidate seconds cannot exceed the pass that ran them."""
        k = self.kernel_share
        return k is None or k <= 1.02          # 2% for the two clocks' skew

    @property
    def broken(self) -> bool:
        """True only when a CHECK failed -- never because a MEASUREMENT came out low.

        ``kernel_share`` of 0.1% is a RESULT (the autotuner compiled candidates rather than running
        them); only the four checks below can say the reading itself is wrong.
        """
        return (self.argmin_ok is False or self.winners_ok is False
                or self.names_ok is False or not self.keys_ok or not self.share_ok)

    @property
    def verdict(self) -> str:
        if self.get("pass_s") is None:
            return ("NO AUTOTUNER PASS in this compile -- either the backend is not GPU, or "
                    "--xla_gpu_autotune_level=0. Nothing here applies.")
        bad = []
        if self.argmin_ok is False:
            bad.append(f"a recorded winner measured {self.get('winner_slowdown'):.4g}x the "
                       f"fastest candidate scopex parsed -- too far to be XLA's selection policy")
        if self.winners_ok is False:
            bad.append(f"{len(self.get('results', [])) - self.get('winners_found', 0)} recorded "
                       f"winner(s) are not among the candidates scopex parsed at all")
        if not self.keys_ok:
            bad.append(f"on {self.get('key_collisions')} instruction(s) two candidates with "
                       f"different run_times share one identity -- a backend's config fields are "
                       f"not being read, so its candidates are indistinguishable")
        if self.names_ok is False:
            bad.append(f"only {self.get('names_confirmed', 0):.0%} of decoded instruction names "
                       f"appear in the VLOG -- HloInstructionProto's field numbers may have moved")
        if not self.share_ok:
            bad.append(f"candidates sum to {self.kernel_share:.2f}x the pass that ran them")
        if bad:
            return "AUTOTUNE READING BROKEN -- " + "; ".join(bad)
        k, c = self.kernel_share, self.compile_share
        head = (f"autotuner {self.get('pass_s'):.4g}s, of which {k:.1%} is measured kernel "
                f"execution ({self.get('n_candidates')} candidates over "
                f"{self.get('n_instructions')} instruction(s))")
        if c is not None and c >= 0.5:
            head += (f" -- {c:.1%} of the WHOLE COMPILE is the machine running candidate kernels, "
                     f"not the compiler transforming IR")
        elif k is not None and k < 0.5:
            head += (" -- the pass is NOT dominated by kernel execution; the rest is autotuner "
                     "overhead (candidate compilation, buffer allocation, redzone checks)")
        r = self.redundant_s
        if r > 0.05 * (self.get("pass_s") or 1):
            head += f". {r:.4g}s of that re-benchmarked an instruction already benchmarked"
        return head

    def __str__(self) -> str:
        L = []
        L.append("CHECK    scopex's parse against XLA's own record of the same autotuning")
        n, m = self.get("n_paired", 0), self.get("argmin_matched", 0)
        L.append(f"  winners      {self.get('winners_found', 0)}/{n} recorded winners are among "
                 f"the parsed candidates   {'ok' if self.winners_ok else 'MISSING'}")
        w = self.get("winner_slowdown")
        L.append(f"  argmin       {m}/{n} are exactly a parsed minimum; worst winner is "
                 f"{f'{w:.4g}x' if w else 'n/a'} the fastest candidate"
                 f"   {'ok' if self.argmin_ok else 'MISMATCH' if n else 'unpaired'}")
        L.append(f"  identities   {self.get('key_collisions', 0)} candidate-key collisions"
                 f"   {'ok' if self.keys_ok else 'COLLAPSED'}")
        f = self.get("names_confirmed")
        L.append(f"  instr names  {f:.0%} confirmed against VLOG module names   "
                 f"{'ok' if self.names_ok else 'FAILED'}" if f is not None
                 else "  instr names  UNCHECKED (no VLOG supplied)")
        k = self.kernel_share
        if k is not None:
            L.append(f"  candidates <= pass    {k:.4f}   {'ok' if self.share_ok else 'IMPOSSIBLE'}")
        L.append("")
        L.append("MEASURE  where the autotuner's seconds went")
        L.append(f"  autotuner pass  {self.get('pass_s'):>10.4f}   (VLOG, host-side pass timer)")
        L.append(f"  candidates      {self.get('candidate_s'):>10.4f}   "
                 f"({self.get('n_candidates')} kernels, XLA's own GPU timers)")
        if k is not None:
            L.append(f"  kernel share    {k:>10.2%}   of the pass was running code")
        if self.get("backend_s"):
            L.append(f"  backend         {self.get('backend_s'):>10.4f}   "
                     f"(jax.monitoring, same child)")
            L.append(f"  compile share   {self.compile_share:>10.2%}   "
                     f"of the WHOLE compile was running candidate kernels")
        r = self.redundant_s
        if r:
            L.append(f"  re-benchmarked  {r:>10.4f}   already-tried instructions, timed again")
        top = sorted(self.get("instructions", []), key=lambda a: -a.seconds)[:4]
        if top:
            L.append("")
            L.append("  slowest instructions")
            for a in top:
                b = a.best
                L.append(f"    {a.name:<26} {a.opcode:<13} {len(a.candidates):>3} cand "
                         f"{a.seconds:>9.4f}s   winner {b.label if b else '-'} "
                         f"{b.seconds if b else 0:.6f}s")
        L.append("")
        L.append("VERDICT  " + self.verdict)
        return "\n".join(L)


def merge_by_instruction(logs: list[Autotuned]) -> list[Autotuned]:
    """One entry per instruction, merging repeat benchmarks and keeping the fastest time per key.

    XLA may benchmark the same instruction more than once -- the conv picker asks cuDNN for
    candidates in several heuristics modes and the lists overlap -- so the logs file can hold ten
    entries for five instructions. Keeping the minimum per candidate key is what XLA itself
    compares against.
    """
    merged: dict[str, Autotuned] = {}
    order: list[str] = []
    for a in logs:
        k = a.name or f"<{a.index}>"
        if k not in merged:
            merged[k] = a
            order.append(k)
        else:
            prev = merged[k]
            best: dict[tuple, Candidate] = {}
            for c in list(prev.candidates) + list(a.candidates):
                if c.key not in best or c.seconds < best[c.key].seconds:
                    best[c.key] = c
            merged[k] = prev._replace(candidates=list(best.values()))
    return [merged[k] for k in order]


def _max_matching(adj: list[set[int]], n_right: int) -> int:
    """Size of a maximum bipartite matching. Plain augmenting-path DFS; the graphs here are ~50."""
    match = [-1] * n_right
    total = 0
    for u in range(len(adj)):
        seen = set()

        # `seen` is bound as a default rather than captured: `aug` recurses into itself and must
        # share ONE visited-set per outer iteration. Capturing gives the same behaviour here only
        # because `aug` never outlives the iteration; binding makes that true by construction.
        def aug(u, seen=seen):
            for v in adj[u]:
                if v in seen:
                    continue
                seen.add(v)
                if match[v] == -1 or aug(match[v]):
                    match[v] = u
                    return True
            return False
        total += aug(u)
    return total


def winner_matching(logs: list[Autotuned], results: list[dict]) -> tuple[int, int]:
    """``(matched, n_results)`` -- can every recorded winner be explained as a parsed argmin?

    ORDER-FREE, AND IT HAS TO BE. The two dumps are written by different code paths in different
    orders: the logs file comes out in autotuning-COMPLETION order across 21 threads, the results
    file in a canonical order. Measured on ``gemm_shapes_k16``, where both hold 46 entries, all of
    them fusions, so there is no class or count to disambiguate by -- pairing them by position gave
    a confident-looking "29 of 46 mismatched" that was entirely an artefact of the pairing. An
    earlier index-pairing did the same thing to ``convT64_dilate16`` (10 log entries, 5 results).

    So no pairing is asserted at all. Instead: build the bipartite graph "this recorded winner is in
    the argmin SET of that instruction's parsed candidates" and ask for a maximum matching. A
    perfect matching means the winners XLA recorded are consistently explainable as the minima
    scopex parsed, without anyone having to know which instruction is which. A parse that mangles
    the seconds moves the argmins and the matching drops below perfect.
    """
    groups = merge_by_instruction(logs)
    adj = []
    for r in results:
        adj.append({j for j, g in enumerate(groups) if r["key"] in g.argmin_keys})
    return _max_matching(adj, len(groups)), len(results)


def winner_slowdown(logs: list[Autotuned], results: list[dict]) -> float | None:
    """How much slower than the fastest candidate XLA's recorded winner was, at worst.

    WHY A RATIO AND NOT A BOOLEAN. XLA's selection rule is not always argmin. The conv picker takes
    the minimum, and so does the emitter choice, but the GEMM autotuner does not: measured on
    ``gemm_shapes_k16_control``, XLA recorded ``triton 64x64x32/3st4w`` at 34016 ns as the winner
    for ``gemm_fusion_dot_general.16`` while ``cublas algo 2`` in the same list measured 32768 ns.
    cuBLAS
    is a fallback there, not a default, so a 3.8% loss does not unseat Triton. A strict argmin test
    calls that a broken parse, which it is not.

    A ratio separates the two failure modes cleanly, because they live orders of magnitude apart. A
    SELECTION POLICY costs a few percent. A PARSE ERROR of the kind this package keeps paying for --
    dropping a magnitude from a duration, mis-scaling a unit -- reorders the list wholesale and
    shows up as a factor of many. Measured negative control: making ``_duration_s`` ignore the
    ``seconds`` field of the Duration, exactly the ``min``-units bug in a new costume, takes this
    from 1.04 to 154.

    CHARITABLE BY CONSTRUCTION. Winners cannot be soundly paired to instructions (see
    :func:`winner_matching`), so for each recorded winner this takes the BEST ratio available over
    every instruction that lists that candidate. A check that must not cry wolf gets the reading
    most favourable to the parse.
    """
    groups = merge_by_instruction(logs)
    worst = None
    for r in results:
        best_ratio = None
        for g in groups:
            cs = g.contenders
            if not cs:
                continue
            here = [c.seconds for c in cs if c.key == r["key"]]
            if not here:
                continue
            lo = min(c.seconds for c in cs)
            ratio = 1.0 if min(here) <= lo + 1e-12 else (
                float("inf") if lo <= 0 else min(here) / lo)
            if best_ratio is None or ratio < best_ratio:
                best_ratio = ratio
        if best_ratio is not None:
            worst = best_ratio if worst is None else max(worst, best_ratio)
    return worst


def key_collisions(logs: list[Autotuned]) -> int:
    """Instructions where two candidates share an identity but were timed DIFFERENTLY.

    The check the matching cannot do. If :func:`_key` fails to see how some backend identifies its
    candidates, every one of them collapses to the same key -- and a matching still succeeds,
    because the recorded winner also collapses to that key. The signature of the collapse is
    exactly this: one identity, two different measured times. Zero on every arm measured; 16 when
    the enumerating version of ``_key`` is restored.
    """
    n = 0
    for a in merge_by_instruction(logs):
        byk: dict[tuple, float] = {}
        for c in a.candidates:
            if c.key in byk and abs(byk[c.key] - c.seconds) > 1e-12:
                n += 1
                break
            byk[c.key] = c.seconds
    return n


def autotune_report(logs_file, results_file=None, *, vlog: str | None = None,
                    pass_s: float | None = None, backend_s: float | None = None) -> Autotune:
    """Read an autotune dump pair and check it. No compile; this reads files.

    ``vlog`` is the ``TF_CPP_VMODULE=hlo_pass_pipeline=1`` stderr from the SAME compile -- supply it
    and the instruction names get checked and ``pass_s`` is read off it.
    """
    logs = autotune_logs(logs_file)
    results = autotune_results(results_file) if results_file else []

    if vlog and pass_s is None:
        from . import _parse
        pass_s = sum(t.seconds for t in _parse.pass_timing_lines(vlog)
                     if t.name == "autotuner")

    names_confirmed = None
    if vlog:
        mods = {m for m, _ in _parse_headers(vlog)}
        named = [a for a in logs if a.name]
        # Only instructions the autotuner actually compiled candidates for get a module of their
        # own; one with a single trivial candidate need not appear, so it cannot count against.
        checkable = [a for a in named if len(a.candidates) > 1]
        if checkable:
            names_confirmed = sum(a.name in mods for a in checkable) / len(checkable)

    matched, n_res = winner_matching(logs, results)
    allkeys = {c.key for a in logs for c in a.candidates}

    return Autotune(
        instructions=logs,
        results=results,
        n_instructions=len({a.name or a.index for a in logs}),
        n_log_entries=len(logs),
        n_candidates=sum(len(a.candidates) for a in logs),
        n_failed=sum(1 for a in logs for c in a.candidates if c.failed),
        candidate_s=sum(a.seconds for a in logs),
        pass_s=pass_s,
        backend_s=backend_s,
        n_paired=n_res,
        argmin_matched=matched,
        winner_slowdown=winner_slowdown(logs, results),
        winners_found=sum(1 for r in results if r["key"] in allkeys),
        key_collisions=key_collisions(logs),
        names_confirmed=names_confirmed,
        device=results[0]["device"] if results else "",
    )


def _parse_headers(vlog: str):
    from . import _parse
    return _parse.pass_pipeline_headers(vlog)


def autotune_cost(module_src: str, *, python: str | None = None, timeout: int = 1800,
                  keep: str | None = None) -> Autotune:
    """Compile ``module_src`` ONCE on GPU with every clock on, and return the checked reading.

    One subprocess, not three. The pass timer, the candidate run_times and the backend seconds all
    have to come from the SAME compile or the comparison measures run-to-run variance instead of
    agreement -- and on these arms that variance is large, because autotuning is the thing being
    measured. Reuses :func:`scopex.pass_timings`, which already forks a child with
    ``TF_CPP_VMODULE`` set, and adds the two autotune dump flags to its environment.

    Costs one compile. Writing the dumps is cheap (two files at the end of the pass); measured on
    ``convT64_dilate16`` the whole compile moved 50.5 s -> 52.2 s, +3.3%, which is within the
    run-to-run spread of that arm and is confounded with machine load, so treat it as an upper
    bound rather than a figure.
    """
    import tempfile

    from .flags import pass_timings

    d = keep or tempfile.mkdtemp(prefix="scopex-autotune-")
    os.makedirs(d, exist_ok=True)
    logs, results = os.path.join(d, "logs.textproto"), os.path.join(d, "results.textproto")

    # XLA APPENDS TO THIS FILE. It does not truncate, so pointing two compiles at one directory
    # silently unions their candidates, and the reader has no way to tell whose seconds are whose.
    # Measured: a second compile of `convT64_dilate16` into a used directory took the file from 10
    # entries / 51.05 s to 15 / 76.14 s, against an `autotuner` pass of 50.47 s -- i.e.
    # `kernel_share` 1.51, which `share_ok` caught, but only because the contamination happened to
    # push the sum PAST the pass. A smaller stale file would have inflated the number quietly.
    # So: refuse to reuse, rather than rely on the check.
    for p in (logs, results):
        if os.path.exists(p):
            if keep:
                raise FileExistsError(
                    f"{p} already exists, and XLA APPENDS to it rather than overwriting -- the "
                    f"result would mix this compile's candidates with an earlier one's. Delete it "
                    f"or pass a fresh `keep=` directory.")
            os.unlink(p)
    extra = (f"--xla_gpu_dump_autotune_results_to={results} "
             f"--xla_gpu_dump_autotune_logs_to={logs}")
    old = os.environ.get("XLA_FLAGS")
    os.environ["XLA_FLAGS"] = (old + " " + extra) if old else extra
    try:
        r = pass_timings(module_src, python=python, timeout=timeout, log_dir=d)
    finally:
        if old is None:
            os.environ.pop("XLA_FLAGS", None)
        else:
            os.environ["XLA_FLAGS"] = old

    cov = r["coverage"]
    if not os.path.exists(logs):
        # No autotuning happened, or the backend was not GPU. Say which; do not return zeros.
        return Autotune(instructions=[], results=[], n_instructions=0, n_log_entries=0,
                        n_candidates=0, n_failed=0, candidate_s=0.0,
                        pass_s=r["passes"].get("autotuner"), backend_s=cov["backend_s"],
                        n_paired=0, argmin_matched=0, winner_slowdown=None, winners_found=0,
                        key_collisions=0, names_confirmed=None, device="",
                        why_no_dump=f"XLA wrote no {os.path.basename(logs)} -- the compile did no "
                                    f"autotuning (CPU backend, or --xla_gpu_autotune_level=0)")
    out = autotune_report(logs, results if os.path.exists(results) else None,
                          # `Raw.text` is a METHOD (the artifact stays on disk; see
                          # scopex/raw.py), so this reads the file rather than the accessor.
                          vlog=r["raw"].text() if r["raw"].exists else None,
                          pass_s=r["passes"].get("autotuner"), backend_s=cov["backend_s"])
    out["coverage"] = cov
    out["passes"] = r["passes"]
    out["dump_dir"] = d
    return out
