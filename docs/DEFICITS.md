# DEFICITS

**For someone deciding whether to trust a number scopex just produced.**

Every instrument here is a derived number read out of text a compiler printed. This document says,
per instrument: what it measures, what checks it, **where that check stops working**, what the
instrument structurally cannot see, and what it costs in compiles. It is written to be read *after*
you have a number and *before* you act on it.

The governing rule this package is built on:

> **A wrong method is worse than a missing one.**

It is not a slogan. Four investigations in `INVESTIGATIONS.md` ended in a confidently wrong pass
name because a twelve-character regex dropped any pass XLA printed in minutes, and on one arm the
instrument holding the answer reported the opposite of the truth without complaint. Those arms
would have been better served by no tool. So the question this document answers is never "is it
accurate" — it is **"can you tell when it isn't"**.

---

## The ship call

| instrument | call | one-line reason |
|---|---|---|
| `Coverage` (`pass_timings(...)["coverage"]`) | **SHIP MARKED** | fidelity is flat at [0.9978, 1.0015] over 20 arms and the guard is shown firing on a real log — but the guard goes *silent*, not loud, if XLA stops printing its own totals, so `Coverage.checked` ships in the data for a caller to branch on |
| `pass_timeline` / `timeline_agreement` | **SHIP** (replaces the unvalidated implementation in place) | every snapshot mtime is testable against the microsecond glog timestamp of the same event — 683/683 inside their own pass timer on CPU, 317/317 on the one GPU run — and `.verdict` reads UNVALIDATED when no log was supplied instead of reading like a checked one |
| `pass_conservation` (and `diverge`'s `conserves` / `complaints`) | **SHIP** | checks the per-pass instruction curve against three things it was not built from — a second counter, the two optimization-boundary files, and index contiguity — and returns the evidence rather than a boolean |
| `Raw` / `raw_step` | **SHIP** | it is a hash and a path, not a measurement; it can only fail by saying "these are not the bytes", which is the one thing it exists to say |
| `boundary_diff` | **SHIP, NOT PROMOTED** | validated four ways, but it answers a *new* question rather than checking an old answer — it sits beside `opcode_delta`, held at exactly this level for exactly this reason |
| `scopex.passmap` (`pass_source`, …) | **SHIP MARKED, NOT PROMOTED** | 213 rows checkable against a real XLA checkout and against the log's own nesting, but pinned to **one XLA commit**, which is a promise a top-level name cannot keep; `XLA_COMMIT` / `BUILT_FOR` ride in the data |
| `scopex.autotune` | **SHIP MARKED, NOT PROMOTED** | five checks each shown firing, plus a 169x ablation that shares no code with either parse; GPU-only and resting on `--xla_gpu_dump_autotune_*` plus one guessed proto field number |
| `scopex.fusion` | **SHIP** | a live decoding defect fixed with 8 pinned regressions, plus a causal-closure check against the module the proto carries inside itself |
| per-instruction lineage | **DO NOT SHIP** | 96.7% correct overall, 49–62% at the passes anyone would ever ask about, and **no second source of truth exists to catch it** — the argument and its measurement moved to `examples/recipes/why_no_instruction_lineage.py` |

Exports went **45 → 50**. The five additions are `Coverage`, `pass_conservation`, `Raw`,
`raw_step`, `timeline_agreement`; every one of them checks a name that was already exported, or is
one of those names in validated form. Three validated instruments were deliberately *not* promoted
(rows above), and one module was deleted from the package.

**Everything below was measured on one box**: jax/jaxlib 0.10.2, python 3.12, x64, 20-core CPU,
RTX 4090 Laptop, ext4, linux 6.17. Where a claim depends on that, it says so.

---

## 1. `Coverage` — what fraction of the compile a pass ranking explains

**What it measures.** Two ratios that are deliberately never merged, because merging them is the
failure mode.

* `fidelity` = scopex's pass-second sum ÷ XLA's own `cumulative:` total. **A self-check.** Its
  correct value is ~1.000 for every program, backend and machine. Anything else means the parser
  is broken.
* `coverage` = leaf pass seconds ÷ `jax.monitoring`'s `backend_compile_duration` from the **same**
  child process. **A measurement.** It has no correct value; measured 0.01% → 99.75% across the
  corpus, and every value in that range is a true statement about where the seconds went.

**What validates it.** XLA's own arithmetic, which it prints on every pass line and this package
ignored for its whole life: `(cumulative: 3.49 ms, max: 236 us, #called: 383)`. These are the
compiler's running count, total and maximum over *exactly* the lines scopex parses, giving three
independent comparisons:

| check | catches | why it is not redundant |
|---|---|---|
| count vs count (`lines_lost`) | a dropped line | **unit-free.** The historical bug was a three-entry unit table; a guard sharing that table shares its blind spot. `298 != 299` does not. |
| sum vs sum (`fidelity`) | how much was lost, **in seconds** | `_parse.expect` can say "you dropped 1 of 640". Only this says the line was 98.8% of the compile. |
| max vs max (`biggest_pass_lost`) | *which* line | a ranking whose top entry is smaller than the compiler's own reported maximum is not a ranking of anything |
| leaf + aggregate == every line (`split_ok`) | the leaf/pipeline walk losing its place | an exact identity by construction, so any failure is a bug; it is how the 21-thread interleaving defect was found |

**Shown firing.** The historical parser (`us|ms|s` only) is reconstructed verbatim in
`tests/test_coverage_guard.py` and run against a real captured GPU log. It loses 1 line of 299 and
returns a healthy-looking ranking topped by `computation-deduplicator: 0.2587` — for a 75.71 s
compile. Coverage collapses 96.40% → 1.07%, fidelity reads 0.0111, XLA's `max:` says one pass took
72.6 s while the largest parsed is 0.253 s. All three checks fire. The healthy parser on the same
log puts `copy-insertion` at 72.31 s = 95.5% of backend.

**Where the validation fails.**

1. **The guard can become a no-op, silently.** `fidelity`, `lines_lost` and `biggest_pass_lost` all
   depend on XLA continuing to print `(cumulative:, max:, #called:)`. If a future XLA stops,
   `xla_pass_count` is `None` and **`broken` returns False** — the verdict says "parse UNCHECKED",
   but nothing raises. **This is why the instrument ships MARKED: branch on `Coverage.checked`,
   not on `not Coverage.broken`.** The only surviving guard in that state is `_parse.expect`'s
   witness count, which is weaker and reads the same log.
2. **`coverage` can exceed 1.0, and now has.** `gemm_shapes_k16` read 1.0111 (leaf 18.3103 s
   against backend 18.1094 s) with fidelity 0.9996 and `split_ok` True. The GPU autotuner compiles
   candidate sub-modules concurrently across 21 glog threads — 89 modules in that compile — and
   each runs its own pass pipeline, so concurrent seconds sum past wall clock. `over_unity` reports
   this. It will recur on any arm that autotunes many candidates; the real fix is a thread-aware
   denominator, not a tolerance.
3. **The denominator is load-sensitive; fidelity is not.** Re-running the sweep on a quieter
   machine moved backend seconds by 0.78–1.08x per arm while fidelity moved at most 0.34%. **Read
   `coverage` to a band, never to two decimal places.** The band edges (0.25, 0.75) survive only
   because the measured gaps around them are 34 and 41 percentage points wide — nothing in the
   corpus falls between 16.4% and 50.2%, or between 50.3% and 91.6%. A wider corpus could fill
   those gaps and make the edges arbitrary.
4. **The listener can fail to arm.** The `jax.monitoring` listener is registered by a hook on
   `builtins.__import__`, which is necessary — importing jax in a preamble would freeze
   `JAX_PLATFORMS`/`JAX_ENABLE_X64` and break any `module_src` that sets them first. But a
   `module_src` that imports jax without going through `__import__`, or calls `os._exit`, yields no
   backend number. It degrades to `coverage=None` with a written reason rather than to `0.0`, which
   is the right failure, but it is still a hole.

**What it cannot see.**

* **WHERE the other seconds went.** On `gatherchain2d_9` it correctly reports that 99.99% of a
  24.7 s backend is outside the HLO passes; it cannot say the seconds are in the CPU loop-fusion
  emitter. Routing to `pass_timeline` is prose in the verdict, not measurement.
* **The difference between "inside a pass timer" and "transforming IR".** On CPU those coincide.
  On GPU they come apart: `convT64_dilate16` and `gemm_shapes_k16` are **indistinguishable** here —
  both ~98% `autotuner`, both PASS-BOUND, both fidelity ~1.0 — and 450x apart on what the seconds
  actually are (cuDNN kernels executing vs Triton candidates compiling). The top band's text now
  names this exception; the number cannot. Use `scopex.autotune.autotune_cost`.
* **Per-module coverage.** `jax.monitoring` attaches no module name to `backend_compile_duration`,
  so with `module=` set the numerator is filtered and the denominator is not. The filtered figure
  is reported separately as `returned_seconds` rather than divided — deliberately.

**Cost.** No extra compile: `pass_timings` already forks one and the denominator comes out of that
same child. Parsing measured on a synthetic 31,440-line log: `pass_log_totals` 101 ms,
`pass_leaf_split` 198 ms, against a compile of ~130 s — 0.2%. Under 10 ms on a typical ~600-line
log. One temp file per call, unlinked in a `finally`.

**One defect found here and fixed on the ship call.** `broken` computed its fidelity threshold as
`max(FIDELITY_FLOOR, 1.0 - (tolerance or 0)*2)`. With `tolerance` absent that is `max(0.90, 1.0)` =
**1.0**, so any fidelity below exactly 1.0 read as PARSE BROKEN — and fidelity is never exactly 1.0.
Every healthy compile would have cried wolf on a `Coverage` built without that key. The slack is
now `max(1 - FIDELITY_FLOOR, tolerance*2)` and symmetric, because summing *more* seconds than XLA's
own running total over the same lines is equally a defect and was not being looked at. Pinned in
`tests/test_ship_calls.py`.

---

## 2. `pass_timeline` / `timeline_agreement` — where the seconds went, including below HLO

**What it measures.** Per-snapshot mtime intervals, plus the **tail**: the interval between the
last HLO snapshot and the emitted `.ll`/`.o`. That tail is the only thing in this package that can
see below the HLO pass pipeline, and it is where several corpus pathologies actually live (one arm:
29.957 s of a compile whose HLO passes summed to a fraction of a second).

**What validates it.** Two independent checks, both concrete, both of which fired.

1. **The instant test.** Every VLOG line carries a glog microsecond timestamp from `CLOCK_REALTIME`
   — the same clock `st_mtime` is on. XLA source (`hlo_pass_pipeline.cc:138-225`) shows the
   snapshot is written *inside* the pass's scoped timer, between the line-181 START log and the
   line-176 END log. So each snapshot mtime is tested for containment in `[t_start, t_end]` of the
   pass that wrote it. This validates **the alignment itself** rather than assuming a filename and
   a log line refer to the same pass. Result: **683/683 matched snapshots inside, 100.0%**, across
   12 compiles and 5 programs, including a 4.7x-overloaded machine.
2. **The leaf-sum test.** The module's own leaf/aggregate split is checked against
   `_parse.pass_leaf_split`, an independent implementation by a different route. It initially
   disagreed by 2–3% on all 10 runs and caught a real bug: `after layout assignment` was classified
   as a leaf because it ran zero passes, so "had no children" was the wrong criterion — "is itself
   a pipeline" is right. After the fix the two agree to 5 decimals on all 10 runs.

The tail's error bound is **empirical, not assumed**: the worst observed distance between a
between-pass snapshot mtime and the independent microsecond timestamp of the same event.

**The validation case** (gather2d_8 vs control, CPU, order-rotated):

```
arm   backend    tail    tail%   err_bound   corr   med_ratio  inside
g8a   7.2090   7.1756   99.5%    0.182 ms   0.644    1.509    26/26
c8a   0.0768   0.0523   68.0%    0.109 ms   0.626    1.504    26/26
PAIRED backend ratio 84.7x; PAIRED TAIL ratio 120.7x; tail SNR 39,342x
```

The tail survives a 1.5x *span* disagreement because the median gap ratio is about what the VLOG
**attributes** (inter-pipeline work belongs to no pass), not about when files were written. The
tail is **one difference of two mtimes**, each accurate to 0.18 ms against an independent µs clock
— not an accumulation. That distinction is what the error bound encodes.

**Where the validation fails.**

1. **It requires the VLOG from the same compile.** Without `log=`, `.agreement` is `None` and
   `.verdict` reads UNVALIDATED. That is the old behaviour, now *labelled* rather than silently
   blessed. `timeline_agreement()` runs one subprocess with both clocks on; comparing two compiles
   would measure run-to-run variance (26% under load) instead of clock disagreement.
2. **One machine, one jaxlib, and now exactly one GPU run.** The 12 original validation compiles
   were all CPU, leaving the per-thread stacks untested on the logs that motivated them. **One GPU
   compile was run for this ship call** (idle RTX 4090 Laptop, 57 MiB / 0% before and after, sole
   user), and it holds:

   | | |
   |---|---|
   | snapshots matched | **317 / 317**, 0 unmatched |
   | inside their own pass timer | **100.0000%**, 0 violations |
   | glog threads | **21**, `unmatched_closes` 0, leaf-sum cross-check **ok** |
   | boundary offset | p50 **0.0041 ms**, max **0.0138 ms** — ~2.7x tighter than CPU's 0.011 / 0.66 ms |
   | correlation | **0.98** (vs 0.39–0.68 on CPU, where passes are sub-millisecond) |
   | tail | 4.6704 s, SNR 337,742x, `usable=True` |
   | tail split | **correctly suppressed** — 10 kernel modules, interleaved |

   That is one program on one GPU, not a sweep: read it as "the per-thread stacks and the alignment
   test do not fall over on a 21-thread log", not as a GPU validation set. One thing it exposed and
   nothing else reports: the main module's own leaf-pass sum was **0.0134 s against 4.4572 s for the
   whole log** — on GPU, 99.7% of the logged pass seconds belong to autotuner *candidate
   sub-modules*, not to your module. A per-module timeline on GPU is therefore a slice of something
   much larger, which is the same phenomenon that puts `Coverage` over 1.0.

   Still untested on NFS or any coarse-granularity filesystem, where 1-second mtime granularity
   would make every interval meaningless — the tie counter is the intended alarm and has never
   fired.
3. **The `.+` mode's good agreement is partly self-inflicted.** Median gap ratio 1.04 is achieved
   because both clocks are then dominated by the same ~110 µs snapshot write, which the measurement
   itself created. `.*` is the honest mode for absolute pass times; `.+` is the honest mode for the
   tail.
4. **Correlation is uninformative on fast compiles.** `corr` is 0.39–0.68 wherever passes are
   sub-millisecond, because mtime noise (10–580 µs) is comparable to the pass times themselves. It
   only reaches 0.80 where passes are ~10 ms. Read the containment fraction there instead.
5. **glog prints no year.** MMDD only, so the caller supplies it; a compile spanning midnight on 31
   December would be misdated by a year. Stated, not guarded.

**What it cannot see.**

* **The tail split, on multi-kernel programs.** ndtri jacrev d4 has 223 LLVM kernel modules
  compiling concurrently with interleaved phases, so emitter/llvm_opt/codegen boundaries order
  nothing. The instrument **suppresses the split** and keeps only the total, which remains a valid
  difference of two observed instants. The guard was verified to fire.
* **A pure tail.** It starts at the last HLO snapshot, so buffer assignment and thunk emission are
  inside it and are not separated out.
* **Passes below the achieved mtime resolution.** With `.*`, 93.5% of leaf passes (25.5% of pass
  time) are below it — but those get no snapshot anyway. Note that mtime resolution is *measured
  per dump*, not assumed: a naive probe reports 1 ms (quantum exactly 1,000,004 ns, 97.3% of deltas
  zero) while XLA's real dump path gets 110–157 µs. The 1 ms figure was an artifact of `stat`-ing
  between writes.

**Cost.** `pass_timeline(dump_dir, log=...)` adds **no compile** — it reads a directory and a
string. `timeline_agreement` costs **exactly one** subprocess compile (both clocks share it) rather
than the two a naive dump-then-pass_timings comparison needs. The dump is the real cost, and it is
charged to the pass region rather than the tail (median of 8 interleaved reps, corpus control):

| mode | backend | vs undumped |
|---|---|---|
| undumped | 0.0629 s | — |
| VLOG only | 0.0639 s | +1.6% |
| dump, no per-pass snapshots | 0.0716 s | +13.9% |
| `passes=".*"` | 0.0735 s | +17.0% |
| `passes=".+"` | 0.0863 s | +37.4% |

Disk: 248 KB / 23 files at `.*`, 1.54 MB / 160 files at `.+` on a small control; one corpus arm
reached 944 MB across 950 files. **The measurement is up to 4x the thing measured** at `.+` for
per-pass times — the tail is untouched, which is the asymmetry that makes the dense mode worth
using.

---

## 3. `pass_conservation` — does the per-pass instruction curve conserve?

**What it measures.** Whether the curve `pass_growth` / `diverge` return actually describes your
compile: counter agreement, endpoint anchoring, index contiguity, and what fraction of the module's
net change the curve witnessed at all.

**What validates it.** Three anchors, **none of which the curve is built from**:

1. **Two counters over the same bytes** — XLA's C++ HLO parser vs the python line pattern, run on
   every snapshot. This is the only check that can see a *uniform* bias, which is what the
   tuple-shape bug was: injecting the old `\S+` shape group makes 38/38 snapshots disagree (worst
   −12.5%) while the anchors, the chain and the coverage all stay perfectly satisfied.
2. **Two endpoint files** — `*.before_optimizations.txt` and `*_after_optimizations.txt` give
   head_gap, tail_gap and a coverage fraction.
3. **Index contiguity** — a hole means a file exists that the grammar could not read.

End-to-end on `bisect_m94`: final snapshot count = `after_optimizations` = `walk_hlo` on the live
executable = **330,402**, three routes to one number. Negative controls all fire: tuple-shape regex
restored; a snapshot made unreadable (`missing == [20]`); a curve truncated one pass before
`fusion` (every internal check still passes, coverage 0.8%). The telescoping identity is reported
but **documented as a tautology that validates nothing alone**.

**Where the validation fails.**

1. **A bias shared by both counters is invisible.** They are independent implementations but not
   independent definitions. The check narrows the failure mode; it does not close it.
2. **A wrong module stem gets a clean bill of health.** `_pick` anchors against that stem's own
   before/after files, so reading a JAX warm-up module is fully self-consistent — head_gap 0,
   coverage 1.0, `counting_consistent` True. The only defence is that `module` is echoed in the
   result and `modules_in` orders by snapshot count.
3. **`covers_whole_compile` is uninformative by design on a narrow `passes=` dump.** It will be
   False and that is correct, so on such dumps it carries no signal. Both real CPU dumps measured
   had head_gap 0 and tail_gap 0 under `passes='.*'`; **whether that holds on GPU is unmeasured.**
4. **The coverage fraction is a NET measure.** A curve that grows then shrinks back can show
   fraction 1.0 while hiding a large excursion. `gross_seen` covers the observed span; there is no
   gross measure for the unobserved tail.

**What it cannot see.** Instruction lineage — see §8. And `diverges_at` is still wrong on
`bisect_m94`: it fires at snapshot 0 (the case is already 1.65x at birth) and names
`async-collective-replacer`, a pass that did nothing. **`biggest_step` is the corrected form and is
what should be read**; `diverges_at` is kept only because callers read it.

**Cost.** No extra compile, but the second counter adds a measured **13%** to parse time, and
`pass_growth` on the 969 MB `bisect_m94` dump is **66 s**. `pass_conservation` accepts precomputed
`steps=` so `diverge` does not pay twice — a caller who forgets will.

---

## 4. `Raw` / `raw_step` — the bytes a number was parsed from

**What it measures.** Nothing. It is a path, a sha256 taken *as parsed*, a line count, and a crude
witness count. `Raw.verify()` re-reads, re-hashes and re-counts.

**What it catches.** A file truncated or rotated under you; a temp directory reaped; the wrong
artifact handed back; a number that came from a different run than the text beside it; a parse that
returned fewer results than its input visibly contains.

**What it cannot see.** **A parser that is wrong the same way twice.** Re-running the same regex
over the same bytes agrees with itself by construction. For that, the artifact itself is the
instrument — `raw.grep(" min ")` finds the one line in 640 that broke `pass_timings`, without a
rerun.

**Cost.** One `stat` and one hash of a file already on disk: ~10 ms for a 4.94 MB snapshot. It
holds a path and not the text deliberately — a `pass_timings` log is 1.86 MB (CPU) / 4.54 MB
(CUDA) per call, and a `passes=".*"` dump reached 78.3 MB across 490 files for a compile that takes
8.8 s. Nothing is cached: a stale digest is worse than a slow one.

---

## 5. `boundary_diff` — what changed at a pass boundary *(not promoted)*

**What validates it.** The opcode census, computation sizes and arity histogram are three
partitions of one population and must equal each other and an independent line count — verified
equal on both arms at both boundaries measured. `resolve_boundary` proves which file a boundary
name selected in each arm.

**What it cannot see.** **Instruction lineage, and it cannot be made to.** XLA renames freely and
records no ancestry. The one restricted case where names carry information (two arms of the same
program at the same boundary) is quantified rather than assumed: on `bisect_m94`, computation-name
overlap was 96.8% and yet 115 names appeared in the case alone, **98% of them XLA-generated** —
reported as `only_generated_fraction` with a caveat pointing at `size_buckets` instead. Note that
`generated_computation_name` deliberately **under**-reports: it does not flag a trailing `.<n>`,
because `main.6`, `region_0.1` and `region_3.8` all carry one and all three come down from the
program. The asymmetry is chosen; telling a reader to ignore their own computations is the worse
error.

**Why not promoted.** It answers a new question rather than checking an old answer, and
`opcode_delta` has been held at exactly this level for exactly that reason. Reach it as
`from scopex.artifacts import boundary_diff`.

---

## 6. `scopex.passmap` — a pass name → the XLA file that implements it *(not promoted)*

**What it measures.** Nothing. It is a lookup: this string was returned by `name()` at this file
and this line, in the XLA tree your jaxlib was built from. Worth having because 30 of the 213
observed pass names have a file name that does not echo the pass name and seven are unguessable
(`triton-gemm-rewriter` → `gemm_fusion.h`; `cse_barrier_expander` →
`optimization_barrier_expander.h`).

**What validates it.** `verify_pass_map(xla_src)` re-reads every row against a real checkout and
checks the name literal still occurs within two lines of the recorded number — which catches the
failure that still *looks* like a working pointer. `cross_check(result)` checks the `pass` /
`pipeline` column against the log's own nesting via `_parse.pass_leaf_split`, two routes with no
shared step: 76 names checked, 76 agree.

**Where the validation fails.** The table is pinned to **one XLA commit**. Against any other
jaxlib the rows may have drifted, and the only way to find out is to run `verify_pass_map` against
a checkout of the right revision — which most users will not have. `XLA_COMMIT` / `BUILT_FOR` ride
in the data so a caller can at least tell *which* revision it is speaking for. **That pin is why it
ships unpromoted**: a top-level name is a promise to keep working across jax releases, and this
cannot make it.

**What it cannot see.** What the pass does, why it was slow, or whether it is your problem. And a
`None` is a real answer — there is no fuzzy match, because a wrong pointer costs an afternoon in the
wrong file. An earlier build of the table mapped `float_normalization` into `third_party/tsl` via a
loose prefix rule; both that and a per-platform row that lost its implementation file were caught by
auditing the 30 rows whose file name does not echo the pass name.

---

## 7. `scopex.autotune` — GPU only *(not promoted)*

**What it measures.** `kernel_share` = sum of every candidate kernel's measured `run_time` ÷ the
`autotuner` pass's seconds, read from `--xla_gpu_dump_autotune_logs_to`. It plays exactly the role
`(cumulative:, max:, #called:)` plays for `Coverage`: XLA's own GPU-clock accounting of the interval
its host-side pass timer was also measuring, from a different subsystem.

**The result that justifies it.** `convT64_dilate16` and `gemm_shapes_k16` are **indistinguishable**
to `pass_timings` — both ~98% `autotuner`, both PASS-BOUND, both fidelity ~1.0 — and their causes
are opposite. The first: 49.7% of a 51.86 s pass is measured kernel execution (18 cuDNN algorithms
on real 134 MB buffers, slowest candidate 6.1 s, winner 0.44 s). The second: **0.11%** of a 17.79 s
pass is kernel execution (456 Triton candidates that run for 20 ms *total* — the seconds are spent
compiling them). **450x apart on `kernel_share`, identical on everything `pass_timings` reports.**

**What validates it.** Five checks, each shown firing in `tests/test_autotune_guard.py`: share ≤ 1;
winner containment; argmin margin; identity collisions; instruction names decoded from the binary
`Any` confirmed against VLOG module names (100% on every arm). Plus a **third, independent clock**:
ablation. `--xla_gpu_autotune_level=0` collapses `convT64_dilate16` from 52.22 s to 0.309 s = 169x,
so backend(on) − backend(off) = 51.92 s against an `autotuner` pass of 52.05 s (ratio 0.997) — a
controlled experiment sharing no code with either parse.

**Where the validation fails — read this one.** **XLA *appends* to the autotune dump file rather
than truncating it.** Running `convT64_dilate16` twice into the same directory took the file from 10
entries / 51.05 s to 15 / 76.14 s against a pass of 50.47 s, i.e. `kernel_share` **1.51 —
impossible**, and `share_ok` fired. Reconstructing: an earlier capture already contained more than
one compile's candidates, and 2 × 25.5 s happening to sum to ~51 s against a ~52 s pass made it look
*perfect* (98.1%). The clean reading, reproduced twice from fresh temp directories, is 24 candidates
/ 25.79 s / `kernel_share` 49.7%. `autotune_cost` now unlinks a stale dump and raises
`FileExistsError` if given a used `keep=` directory.

**The blind spot that leaves.** `share_ok` caught the contamination only because it pushed the sum
*past* the pass. **A smaller stale file inflates `kernel_share` quietly**, and because
`kernel_share` has no correct value by design, no check can distinguish an over-recorded dump from a
genuinely kernel-heavy pass. If you point this at a directory scopex did not create, the number is
not defended.

**What it cannot see.** The residual. `pass_s − candidate_s` = 26.07 s on the conv arm is **not
attributed**; all that can be said is that it is not more kernel time, because the VLOG independently
shows exactly 18 candidate sub-module compiles matching the 18 recorded candidates.

**Cost.** `autotune_cost` = one subprocess compile. The ablation is two more. GPU compiles here ran
50–52 s each.

---

## 8. Per-instruction lineage — **DO NOT SHIP**

**The measurement.** Matching instruction names across consecutive per-pass snapshots, on a GPU dump
of 64 snapshots / 63 boundaries: **1,914 / 1,979 = 96.7% name-identical** pooled.

**Why that number is the reason not to ship it.** The aggregate is carried by the **51 boundaries at
which nothing happened**, where a lineage mapping is worth nothing because the identity map is
already correct. At the boundaries where a pass actually restructured the module — the only place
anyone would ever ask — it falls to **49–62%**, and it falls furthest at `priority-fusion`, the pass
this project's investigations most often had to explain. Half the answers wrong, precisely where the
tool is used, behind a 96.7% headline.

**And no cross-check can exist.** Unlike the regex bug, no coverage ratio catches this: a
name-matcher does not know that the instruction it matched is not the instruction it came from, and
there is no second, independent source of truth to check it against. That is what makes it
unshippable rather than merely unfinished.

**What was found instead, and it is actionable.** XLA *has* a real lineage mechanism —
`original_value`, "the name of the instruction in the unoptimized HLO module that produces this
array", with a proto, a recovery table and helpers that carry it across rewrites. It is populated by
exactly one pass, `AddOriginalValue`, registered in exactly one place: `hlo_opt/opt_lib.cc`, a
standalone developer tool. Neither `cpu_compiler.cc` nor `gpu_compiler.cc` adds it. Confirmed twice:
`add-original-value` appears in **0 of 23,533** `HLO pass:` lines logged across both backends, and
`origin={` occurs **0 times** in all 268 files of a `passes=".*"` GPU dump. So the honest statement
is not "XLA records no lineage" — it is **"XLA has a good lineage mechanism and JAX's pipelines do
not turn it on."** That is a small, plausible upstream feature request.

The argument and its three measurements live in
`examples/recipes/why_no_instruction_lineage.py`; `original_value_present()` is the one-line check
that starts returning non-zero if a future jaxlib ever runs that pass.

---

## 9. What is still unvalidated

**Two of the three clocks.** The key asset of this package is that the same interval is measured
three ways:

| clock | source | validated? |
|---|---|---|
| `Timings['backend']` | `jax.monitoring` | **no** — wall clock is its only second opinion |
| per-pass seconds | XLA's own VLOG | **yes** — `Coverage`, §1 |
| per-snapshot intervals | dump file mtimes | **yes** — `timeline_agreement`, §2 |

`record()`'s stage numbers still have only wall clock as a second opinion. The pass log was the easy
one: XLA happened to print its own arithmetic next to the numbers scopex re-derives. **Do not read
this work as evidence that the three-clock comparison is done.**

**And the structural limits that survive everything above.**

* **One box, one jaxlib.** Every number in this document is jax/jaxlib 0.10.2 on one machine. The
  claim that `fidelity` is invariant to program, backend and load is strongly supported *within that
  sample* and untested outside it.
* **The leaf split depends on glog's line format.** `_TID` matches `^\w\d{4} [\d:.]+\s+(\d+) `. If
  glog's prefix changes, every line buckets into one thread and the split regresses to the pre-fix
  behaviour. On a real multi-threaded log `unmatched_pipelines` goes non-zero and `split_ok` catches
  it; **on a single-threaded log the regression is invisible**, and only the hand-written
  interleaved conformance sample would notice.
* **A regression test pinned to the wrong arm is green for the bug's whole lifetime.** The
  min-unit bug could not be reproduced on the arm the study named: re-measured on an *idle* GPU,
  `convT64_dilate16`'s autotuner takes 50.7 s and XLA prints `50.7 s`. XLA only switches to `min`
  above 60 s, so the historical parser reads it correctly and nothing fires. The published 72.5 s
  reading was taken with ten foreign GPU processes at 100% utilisation. **Same program, same
  compiler: catastrophic silent failure on a loaded machine, invisible on an idle one.** The
  min-unit fixture is therefore `switch_ident_1024`, and a test fails loudly if a regenerated
  fixture stops containing a `min` line. `convT64_dilate16` is kept as the positive control.
* **Contention moves absolute seconds and not ratios.** Some measurements above were taken while a
  sibling agent compiled on the same box (load 0.47–5.98 on 20 cores). Absolute seconds are upper
  bounds. Every primary cross-check is computed **within a single compile** (agreement, not
  run-to-run variance), every headline comparison is a paired case/control ratio, and the argmin,
  identity, containment and name checks are unit-free identity comparisons that contention cannot
  move. Evidence the design held: `fidelity` stayed in [0.9987, 1.0015] across all arms while
  `convT64_dilate16`'s backend seconds moved 50.55 / 52.22 / 51.86 / 50.62 s across four compiles.
