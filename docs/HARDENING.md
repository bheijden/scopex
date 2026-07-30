# HARDENING

Which parts of scopex read **text a compiler printed**, what produces that text, what happens when
the producer changes it, and how the two self-checks cover each one.

This document exists because the failure mode of a text parser is not a crash. Three parsers shipped
broken in this package and all three returned a *plausible* answer — an empty level, a ranking topped
by the wrong pass, a module of 21,000 operations reported as one. Every one was found by *using* the
tool, never by reading it. So the rule here is not "avoid regexes"; it is:

> **A parser that cannot fail loudly must not be trusted to fail quietly.**
> Every remaining pattern is paired with a *witness* — an independent, deliberately cruder count of
> what the input visibly contains. Fewer results than the witness is a `ParseError`, not a result.

---

## 0. The three that shipped broken

Kept at the top because they are the shape of the next one.

| # | parser | what it did | why nobody noticed |
|---|---|---|---|
| 1 | `walk_stablehlo` | returned **1 unit on 16 of 21 real programs**, including modules of 3,214 and 21,000 operations | jax emits `loc(#loc17)` on the operation and defines `#loc17 = loc("…")` separately; the pattern matched only the inline form. A length-1 iterator passes `if not units`, so the level looked **empty**, not **broken**. |
| 2 | `walk_hlo` metadata | dropped every **unquoted** metadata value | `(\w+)="([^"]*)"` matches `op_name="…"` but not `stack_frame_id=5`. Every instruction still returned a non-empty dict, so the loss was invisible at instruction scale. It produced the published conclusion that the optimized module carries no source location. It does. |
| 3 | `pass_timings` | dropped the **slowest pass in the program** | XLA switches to `min` above 60 s. Measured: 1 of 640 lines used `min`, it was the autotuner at **98.8% of a 72.5 s compile**, and dropping it left a plausible dict topped by `remat-pipeline: 0.1196`. The tool reported the opposite of the truth. |

The pattern is the point, and it is a design constraint, not an anecdote:

> **A parser whose blind spot correlates with the quantity being measured is worse than no parser.**

Bug 3 is the purest form — the *slowest* pass is precisely the one most likely to be printed in a
unit the parser did not know. Bug 1 is the same shape: region-bearing ops (`while`, `case`, `sort`)
were 100% invisible, and those are exactly the expensive ones.

---

## 1. What is no longer parsed at all

The largest hardening win was deleting parsers, not guarding them. These now read an object graph:

| level | was | now | evidence |
|---|---|---|---|
| StableHLO ops | line regex over printed text | recursive walk of the real MLIR module (`jaxlib.mlir.ir`: `Operation.regions/blocks/operations`) | strict per-opcode **superset** on all 10 test programs; `marked_framework` 296 → **311** units, `control_flow` 90 → 107, `switch_5way` 26 → 37 |
| StableHLO names | regex over `#locNN = loc(...)` | `ir.NameLoc.name_str` / `.child_loc`, two-level peel matched to jax's `source_info_to_location` | 294/311 named; no path may end in `:` and none may be a bare frame name (asserted) |
| StableHLO sites | constant `'<see-jaxpr-level>'` | `ir.CallSiteLoc` + `ir.FileLineColLoc` | real `file:line` for **226/311**; **91.6%** land on a file:line the jaxpr level also reports, so the two levels now *join* |
| HLO instructions | `_INSTR` line classifier | `xla_client.hlo.HloModule.computations() → .instructions()` | over **2,811** real dump snapshots the old regex undercounted on **895 (31.8%)**, never overcounted, missed **1,208** instructions |
| HLO opcodes | text | `HloInstruction.opcode` enum | alias table **harvested**, not guessed: exactly 6 divergences over 3,017 snapshots |
| `pass_growth` counts | `_INSTR` line count | `hlo_module_from_text()` — XLA's own parser | parsed **2,811/2,811**, and *raises* on non-HLO dump files rather than returning an empty module |
| fusion decisions | grep | full text-proto parse (`scopex.fusion`) | **455/455** real dumps parse; an unknown step kind arrives as a **visible key** rather than being dropped |

> **The lesson that generalises.** Bug 1's real cause was not the loc-alias indirection. It was
> **regions**: MLIR prints an op owning a region across many lines with its `loc(...)` after the
> closing brace, so *no single line* carries both opcode and location. A line-oriented reader cannot
> be patched into correctness there — it has the wrong shape. It also cannot see ops that are never
> printed at all: `stablehlo.reduce`'s short `applies stablehlo.add across dimensions` form has a
> real `add` + `return` inside its region with no line to match.

One correction to a documented claim, because it is what sent the original implementation to the
text in the first place: `flags.py` listed `compiler_ir('stablehlo')` as a trap that "drops location
info", because `str()` of it shows 0 `loc(`. That is `Operation.__str__`'s `enable_debug_info=False`
default, **not the IR**. The same object printed with `enable_debug_info=True` is byte-identical to
`as_text(debug_info=True)`: 48,770 chars, 478 `loc(`. The trap is real; the conclusion drawn from it
was not.

---

## 2. The invariant: `expect(...)` with a witness

Everything still parsed from text goes through one function in `scopex/_parse.py`:

```python
expect(parser_name, matches, text, witness=...)
```

`witness` is a **cruder** pattern than the parser it guards, counting what the input *visibly*
contains. Returning fewer results than the witness raises `ParseError`.

**It is a count, not a boolean.** `n > 0` would not have caught bug 1 — one unit from a
3,214-operation module is non-zero. Verified by reintroducing each historical bug against live
compiles:

| reintroduced bug | what the guard said |
|---|---|
| stablehlo inline-loc-only (bug 1) | `returned 1 operations from input that visibly contains at least 7` |
| quoted-only metadata (bug 2) | `returned 0 stack_frame_id values` — on all 6 test programs |
| unit table without `min` (bug 3) | `cannot convert time units` |

Empty input still legitimately returns 0. `allow_empty=True` covers the parsers whose absence is real
information (a program need not contain a custom call).

**The guard found a fourth bug on the first live log it ran against.** XLA pass names contain
**spaces**, and `(?P<name>\S+)` silently dropped them: 384 lines of a CPU compile said `HLO pass: `
and only 378 parsed. The six missing were `HLO pass: simplification after layout assignment` and
`HLO pass: after layout assignment`. Same family as the other three — a silent undercount leaving a
plausible ranking. Nobody was looking; the witness check simply refused to return.

---

## 3. Every remaining text parser

All of them live in `scopex/_parse.py` and nowhere else. This is **enforced**:
`tests/test_parse_quarantine.py::test_no_other_module_parses_compiler_output_with_a_regex` scans
every module in the package for `re.compile|search|match|finditer|findall|sub|split` and fails on any
that has escaped, with a two-entry allowlist that names why each is exempt. *(This test caught a
duplicate filename parser being introduced into `phases.py` during this very round; the fix was to
delete it and call `_parse.emitter_dump_name`, which was already hardened against a bug the duplicate
would have reintroduced.)*

Each parser carries a docstring naming the exact printing component and a **verbatim measured
sample** frozen as a module constant.

### 3.1 Optimized-HLO instruction text

| | |
|---|---|
| **parsers** | `hlo_metadata`, `hlo_shape`, `hlo_shape_and_opcode`, `is_hlo_instruction_line`, `custom_call_targets` |
| **produced by** | `xla::HloInstruction::ToString`, `xla::OpMetadata`, `xla::HloComputation::ToString` |
| **why not native** | `HloInstruction`'s complete non-dunder surface on jaxlib 0.10.2 is `async_wrapped_root, name, opcode, operands, to_string, users`. **No `.metadata`, no `.shape`.** `HloPrintOptions` has `print_metadata`, but `HloInstruction.to_string()` takes no arguments — only `HloModule.to_string(options)` does. |
| **blast radius** | reduced from "classify every line of a module" to "one instruction's own string". Instructions are enumerated **natively** and only then handed to the pattern. |
| **if the format changes** | per-instruction: a `metadata={` present but unreadable raises. Per-**module**: `check_metadata_coverage` counts how many instructions resolved and raises if the ratio collapses. The module-level check is the one that matters — with a quoted-only pattern every instruction still returns a non-empty dict, so bug 2 is invisible at instruction scale. Before this guard, reintroducing bug 2 returned 19 units with no error; now all 6 test programs raise. |
| **conformance** | `hlo_metadata` (≥2, both a quoted and an unquoted value), `hlo_metadata[module]` (≥12), `hlo_metadata[stack_frame_id]` (≥9 — bug 2 resolved zero), `check_metadata_coverage`, `hlo_shape` (≥20), `hlo_shape[tuple]` (≥2), `is_hlo_instruction_line` (≥20), `custom_call_targets` (≥1) |
| **known trap** | tuple shapes contain a **space** — `(s32[], f32[8]{0})` — and `\S+` dropped every `while`/`call`/`tuple`. Shape group is `.+?`. Stressed on 8 real programs (while, scan, cond, eigh, sort, grad, vmap, topk): 203 instructions, 0 unparsed shapes, 0 false positives. |
| **and one I reintroduced** | writing the fix, `re.findall` reports a non-participating group as `""`, not `None`, so `q if q is not None else u` blanked every unquoted value and all source resolution silently returned `<no-frame>`. Fixed to `finditer`; `test_metadata_keeps_unquoted_values` guards both the original bug and this variant. |

### 3.2 HLO stack-frame index tables

| | |
|---|---|
| **parsers** | `hlo_frame_tables`, `hlo_frame_stack`, `hlo_site`, `_parent_offset` |
| **produced by** | `xla::HloModule::Print` (StackFrameIndexProto: FileNames / FunctionNames / FileLocations / StackFrames) |
| **why not native** | `HloModule` exposes no accessor; they are only printed. `as_serialized_hlo_module_proto()` exists but **no schema ships** — see INVESTIGATIONS §A4. |
| **importance** | this is the **only** route to a source line at the optimized-HLO level. |
| **if the format changes** | empty tables stay a legal state (per-pass snapshots carry `stack_frame_id` but not the tables); a `StackFrames` section that is *present and parses to nothing* raises. |
| **conformance** | `hlo_frame_tables` (≥8 frames), `hlo_frame_stack` (≥5 — `tanh` is 5 frames deep: leaf/residual/solve/top/`<module>`) |
| **two traps, both load-bearing** | **(a)** the walk must be cycle-guarded and depth-capped (512): jaxlib 0.10.2 writes a root frame whose `parent_frame_id` equals its own id. **(b)** far worse, an earlier version *papered over* those self-loops with the cycle guard and described them as "jaxlib writes a root frame whose parent is itself". In fact **jaxlib prints frame ids and parent ids in different index spaces**, and reading `parent_frame_id` literally truncated every python stack to **one frame** — and the innermost line is still right, which is exactly why it looked fine. `_parent_offset` now *derives* the convention per module: whichever of {0,1} yields an acyclic forest with every parent in range; literal wins if both work; `ParseError` if neither. Live: derived offset 1 gives `[leaf:2, mid:3, top:4, <module>:12]`; the literal reading gives `[leaf:2]` only. |
| **semantic check** | frames 6 and 8 are the same source line reached via `residual()` and via `solve()` and **must not share a caller** — asserted, because a flat-table misreading gives them the same one. |

### 3.3 StableHLO text — **fallback only**

| | |
|---|---|
| **parsers** | `stablehlo_op_lines`, `stablehlo_loc_aliases`, `stablehlo_file_aliases` |
| **produced by** | MLIR `AsmPrinter` |
| **status** | **dead code on this environment.** The primary route is the native `jaxlib.mlir.ir` walk (§1). jax lowers *through* `jaxlib.mlir.ir`, so if jax can lower, the bindings import. |
| **if reached** | emits a `RuntimeWarning` naming what will be missing (region-bearing ops). It is never reached silently. |
| **extra guards it never had** | operation count witnessed against SSA-assignment lines; and **dangling aliases** — every `loc(#locNN)` an operation cites must be defined in the module, or the name stack was silently lost. |
| **conformance** | `stablehlo_op_lines` (≥10), `stablehlo_op_lines[named]` (≥10 — bug 1 returned **1** here), `stablehlo_loc_aliases` (≥15), `stablehlo_file_aliases` (≥10) |
| **semantic checks** | a returned name must not be a **file path** (bug 1's pattern took the first quoted string, which for an inline-lowered primitive is the file); and the name stack behind `loc(#locNN)` must survive. |

### 3.4 XLA pass-timing log

| | |
|---|---|
| **parsers** | `pass_timing_lines`, `pass_pipeline_headers` |
| **produced by** | `hlo_pass_pipeline.cc:176` (timings) and `:303` (pipeline headers) |
| **why not native** | VLOG output has **no object model at all**, and no in-process route: `TF_CPP_VMODULE` is read by the C++ logging layer when the shared library loads, i.e. during `import jax`. Setting it afterwards produces exactly zero lines. Hence the subprocess in `pass_timings`. |
| **if the format changes** | an **unknown unit now raises** rather than warn-and-drop. That is bug 3's fix and it is deliberately loud: the slowest pass is the one most likely to use an unexpected unit. |
| **conformance** | `pass_timing_lines` (≥4, including the `min` line bug 3 dropped **and** the space-containing pass name from bug 4), `pass_pipeline_headers` (≥2) |
| **semantic checks** | the `min` line must be present, correctly **scaled**, and must still rank first; a pass whose name contains spaces must survive; headers must retain module attribution. |
| **still open, and documented as such** | XLA registers some **pipelines as passes**, so `HLO pass: simplification` is printed alongside the `constant_folding` that ran inside it and `sum(passes.values())` double-counts (measured coverage 186%). Only the **ranking** is safe; the total is an upper bound. `pass_pipeline_headers` at least lets you filter by module: measured 0.0082 s over all modules vs 0.0047 s for the program alone — 43% of the "total" was JAX warm-up modules. |

### 3.5 Dump filename grammars

| | |
|---|---|
| **parsers** | `dump_snapshot_name`, `emitter_dump_name` |
| **produced by** | `xla/service/dump.cc`, the latter via the backend emitters |
| **if the format changes** | `modules_in` **raises** if the directory contains `.before_` filenames and none parsed — an empty module list must never read as "this compile ran no passes". |
| **conformance** | `dump_snapshot_name` (≥2), `emitter_dump_name` (≥7 of 8 sample names — the 8th is a per-pass HLO snapshot, which is *not* an emitter file and must be rejected) |
| **the trap** | **the kernel field can contain a dot and the module stem cannot.** XLA appends `.<n>` when kernel names would collide. With a dot-free `kernel` group — the obvious reading, and the first version — 22 files parsed happily with `kernel="265"`, `kernel="48"`, `kernel="6"`. Not empty, not an error: the disambiguator where the kernel name should be. |
| **overlap** | the two grammars share their first two fields, so a semantic check asserts `emitter_dump_name` rejects a per-pass HLO snapshot. Without it a *pass* is reported as a *kernel*. |

### 3.6 Emitter MLIR pass log

| | |
|---|---|
| **parsers** | `mlir_pass_dumps`, `mlir_op_lines`, `mlir_log_damage` |
| **produced by** | `mlir/lib/Pass/IRPrinting.cpp` (`PrintIRPass`), which XLA switches on for the emitter pipeline |
| **conformance** | `mlir_pass_dumps` (≥3), `mlir_op_lines` (≥12), `mlir_log_damage` (≥1) |
| **traps** | a pass **name** can contain `::` and a pass **spec** can contain nested braces, so neither is `\w+` and neither is `[^{]*`. An attribute alias must not be counted as an operation. |
| **damaged input is a category** | XLA writes this log **torn** when emitters run concurrently. `mlir_log_damage` reports that explicitly, so a truncated header costs one dump rather than being silently absorbed — *damaged input and a parser blind spot must not look the same*. |

### 3.7 Fusion decision log — text proto, not regex

| | |
|---|---|
| **parser** | `scopex.fusion.parse_textproto` (a complete proto3 text-format tokenizer, allowlisted out of the quarantine) |
| **produced by** | `FusionProcessDumpProto`, GPU only |
| **why it is safe** | text-proto is **self-describing** — the field names are in the file — so it needs no schema, which is what makes it viable where the binary route is not. |
| **evidence** | 455/455 real dumps parse, 1,280 steps. On the richest dump (112,665 bytes) the parsed `reason` and `producer_name` multisets are **identical** to the raw file (257 occurrences each). |
| **the decisive property** | `fusion_steps` reads the oneof arm's field **name off the file**, so an unrecognised step kind arrives as a **visible key**. `test_an_unknown_step_kind_survives_instead_of_vanishing` proves a hypothetical 4th kind is counted rather than dropped — precisely the failure a 3-kind grep would ship. |
| **if the format changes** | malformed input raises `TextProtoError`. An empty dict would read as "this compile made no fusion decisions". |
| **platform note** | GPU-only. Its **absence on CPU must not be read as "no fusion happened"**, and `dump(fusion=True)` warns on non-GPU. |

---

## 4. The two self-checks, and the split between them

```python
scopex.conformance()   # 21 parsers vs frozen samples. NO jax, no compile. Belongs in CI.
scopex.selftest()      # a real marked compile, dumped. Run after any jax upgrade.
```

The split is the point:

* **`conformance()`** replays every parser over a **verbatim frozen capture** of the text it was
  written for. Those samples cannot change, so a failure means exactly one thing: *somebody edited
  the parser*. It needs no jax, so it separates "the parser broke" from "the compiler moved".
  It also runs `_semantic_checks()` — the checks a *count* cannot express, every one of which is a
  bug this package actually shipped (the `min` line must rank first; two call paths to one source
  line must not share a caller; a name must not be a file path; an emitter grammar must reject an HLO
  snapshot).
* **`selftest()`** compiles a small **marked** program, dumps it, and checks every level and every
  artifact view comes back non-empty, that the levels still agree, and that the HLO frame tables
  resolve to source lines the jaxpr also reports. A failure here with `conformance` green means the
  **compiler moved**.

`strict=True` (the default) raises. That is deliberate: this package has three times shipped a
parser that returned a plausible empty answer, and each time the answer was believed.

Two constraints on `selftest`, both load-bearing:

* It must run **before the first compile in the process** — XLA reads its dump flags when the backend
  is first initialised, and it raises rather than measure an empty directory.
* Its probe program is written to a **file outside the package** and imported. Both site resolvers
  filter out frames inside jax *and* inside scopex, so a probe defined in `artifacts.py` would have
  no user frame at all, and the cross-level site join — the one check that catches a frame table
  resolving to the **wrong** line rather than to none — would compare two empty sets and pass.

Current output on this environment: `conformance` ok, 21 parsers; `selftest` ok, `site_join 1.0`,
19 HLO units / 19 instructions, 11 StableHLO units, 6 frame tables, `parent_offset 1`.

---

## 5. What is still text, still open, or still uncovered

Honesty section. None of these is guarded into correctness; they are guarded into **noticing**.

* **`pass_timings` double-counts pipelines with their member passes.** Coverage is an upper bound and
  can exceed 1.0 (measured 186%). Only the ranking is safe. This is a real remaining defect, not a
  parser bug — the log genuinely prints both.
* **`custom_call_target` is not on `HloInstruction`**, so the target string still comes from printed
  form. It now runs only on a string already known to be a custom call, instead of matching the
  literal `custom_call_target=` inside a `backend_config` JSON blob or an embedded pre-fusion module.
* **Operand arity, shape and rank are not fields on `Ins`.** Two corpus cases have one of them as the
  control variable; both recipes go around `walk_hlo` to `hlo_instructions` directly and say so.
* **`FusedLoc` is handled defensively but is unexercised.** Location node types observed across
  `marked_framework`: `NameLoc` 1520, `FileLineColLoc` 1223, `CallSiteLoc` 997, `UnknownLoc` 86,
  **`FusedLoc` 0**. A StableHLO pass or a hand-written dump could produce one; the path exists and has
  never run on real output.
* **The StableHLO name-peel depth is pinned to jax's `source_info_to_location` as of 0.10.2.** If jax
  adds a third `NameLoc` wrapper the peel goes stale. This is guarded by *tests* (no path may end in
  `:`; no path may be a bare frame function name), not by the code being version-independent — and
  **both** error directions produce plausible strings, which is why the guard is a pair of assertions
  rather than a single one. Peel too few and an inline-lowered op reads as `scatter:` — a name stack
  with no scopes, so every contract accessor returns empty. Peel too many and an op whose traceback is
  a single frame collapses to `<module>` (26 of 311 ops on `marked_framework`).
* **XLA's own `slow_operation_alarm` output is discarded.** XLA printed *"Constant folding an
  instruction is taking > 1s"* verbatim on stderr during one investigation — a compiler-authored
  diagnosis, thrown away at the moment of capture. Wishlist item 8.
* **`pass_growth`'s regex fallback still exists** (0 of 2,811 snapshots needed it). A fallback that
  silently mixed scales would fake a step in the curve, so `PassStep` records `how`
  (`"native"`/`"regex"`) and `pass_growth` **warns on a mixed curve**.
* **`backend_split` reads mtimes, which are wall clock.** Not a parser, but the same discipline
  applies: it reports `sound` and `coverage`, and the coverage band *revokes* soundness rather than
  annotating it. See INVESTIGATIONS §A2 for why the interleave test is not redundant with coverage.

---

## 6. If you are adding a parser

1. **Try not to.** Check for a native accessor first, and record the negative with evidence if there
   is none — `dir()` output pasted into a block comment, so nobody re-probes. Two such records are in
   `levels.py`.
2. Put it in `_parse.py`. The quarantine test enforces this and will fail your build otherwise.
3. Freeze a **verbatim** sample as a module constant, with a comment naming the component that
   printed it (`xla::OpMetadata`, `hlo_pass_pipeline.cc:176`, `mlir/lib/Pass/IRPrinting.cpp`, …).
4. Give it a **witness** that is cruder than the parser, and add it to `PARSERS` with an `at_least`
   that would have failed on the bug you are worried about — **not** `at_least=1`.
5. Ask the question that catches this package's whole bug family:
   **what input makes this parser return fewer results than it should, and does that input correlate
   with the thing I am measuring?** If it does, the parser is the wrong shape. Change the shape.
6. Add a semantic check if correctness is not expressible as a count. Most of the real bugs here were
   not.
