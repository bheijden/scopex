# INVESTIGATIONS

An investigation log, not a routing table.

Seventeen case files from `tests/degenerate/`, twenty-eight pathological arms, each paired against
its own control, on CPU and on one CUDA device. For each one: a maintainer is handed a slow
compile, reaches for scopex because it makes the stages, levels and artifacts cheap to look at, and
then has to think. What follows is how that went.

## The headline

**Almost every investigation reached "this stage, this pass, this backend". Almost none reached a
root cause without leaving scopex.** Of the twenty-eight arms, twenty-seven ended with the guilty
stage named and twenty-one with the guilty pass, emitter or Python call site named. But in at least
fifteen of those, the final step — the one that turns "the fusion pass is 41% of your compile" into
"`length // unroll == 1` makes jax emit a single-trip while loop that XLA inlines, and then fusion
replicates the whole Sturm chain into every consumer" — came from an artifact scopex does not hand
over: a raw glog stream, dump file mtimes, a per-pass snapshot diff written by hand, or XLA source.

Two harder truths belong in the first paragraph too. **One arm (`convT64_dilate16`, GPU) had no
scopex-reachable signal at all**, and worse, the one instrument holding the answer reported the
opposite of the truth without complaint. **And on four arms scopex returned a confidently wrong
profile** — a plausible dict naming the wrong pass — because of a twelve-character regex bug that
drops any XLA pass slow enough to be printed in minutes. A tool that fails toward a plausible wrong
culprit is worse on those arms than no tool.

The useful claim this log supports is narrow: *scopex got me from "slow" to "in this stage, in this
pass, on this backend" in three or four commands, reliably, on twenty-seven of twenty-eight arms.*
It does not find the bad line. On this corpus there is no bad line to find; the pathologies are
properties of programs, not of statements.

## How to read the numbers

Every number below carries its platform and its size parameter. They are not comparable across
those axes and several are not comparable across arms of the same case file, because the machine was
under contention for the GPU work (ten foreign GPU processes at 100% utilisation during the CUDA
batch; absolute seconds there are upper bounds, ratios are sound). Environment throughout:
jax/jaxlib 0.10.2, `JAX_ENABLE_X64=1`, CPU arms under `JAX_PLATFORMS=cpu`, GPU arms on an RTX 4090
Laptop (sm_8.9).

**The device is an axis, not a footnote.** Which backend you compile for decides which passes run at
all, so it decides whether a pathology exists. `jax#32704` measures 85x-522x on CPU and is flat on
GPU. `xla#41173` reproduces on GPU only — the CPU arm is already fixed by `kMaxRank=8`. Four of the
GPU findings (`convT`, `gemm_shapes`, `argsort`, `topk`) concern passes that have no CPU counterpart
at all. No "no signal" verdict in this document is stated without its platform.

---

# Part 1 — the investigations

Ordered by where the cost turned out to live, which is roughly the order the compiler visits:
Python tracing, then jaxpr/lowering, then a single HLO pass, then the emitter and LLVM, then
autotuning, then the arm that never finishes compiling. That ordering is itself a finding: the stage
split routes you into exactly one of these five buckets in one command, and the buckets want
completely different tools.

## A. Cost upstream of XLA — in Python

### A1. `einsum_optimal_n10` / `_n11` — `jax#2583`, CPU

**The symptom.** `jnp.einsum(..., optimize='optimal')` over 11 operands: wall 49.8 s against a
control (`optimize='auto'`) at 0.23 s. The harness scores this "no signal" on `compile_s`, and that
verdict is the finding.

**The trail.** `scopex.record` on the first call, before anything else: trace 49.664 s vs 0.011 s
(4,500x at nops=11), lower 0.086 vs 0.054 s, backend 0.219 vs 0.199 s. `scopex.regime` returns
`trace-bound` for the case and `backend-bound` for the control. That single call ends the search for
the stage and rules out the entire compiler: 0.219 s of backend cannot hide 49 s. Everything after
it was confirmation, and it all came back null in exactly the right way — jaxpr 10 equations vs 10,
StableHLO 31 lines vs 31, `walk_hlo` 61 vs 60 units, and a per-pass instruction curve from
`scopex.dump(passes='.*')` that stays between 0.86x and 1.33x across all 33 snapshots with an
identical final module. The case file ships a `_pathlit` arm — the optimal path handed over as a
literal so the solver never runs — which reads trace 0.016 s / backend 0.357 s. That pins the cost
to the *search* and exonerates the resulting contraction.

Then the strongest null available anywhere in this corpus: md5 of
`module_0000.jit_einsum.cpu_after_optimizations.txt` is byte-identical between the case and its
`_pathlit` twin (`01de1476...` for both, `b115cb54...` for the control), with identical LLVM IR
(16 `.ll` files, 2,443 lines, 127,482 bytes), identical object bytes (8,576) and identical buffer
allocations (12). The case and its HLO-identical twin differ 42x in cost. There is provably nothing
downstream to find.

Trace ladder: nops 8 -> 0.317 s, 10 -> 4.467 s, 11 -> 49.013 s against a control flat at
0.015-0.018 s. Super-exponential, as the issue claims.

**Where scopex earned its keep.** The very first knob, one call, no flags, and it is the only
instrument in the stack that can ever see this. Counterfactual: without a stage split I would have
reached for `XLA_FLAGS=--xla_dump_to` and a pass timer, found a 0.2 s compile of a 21-instruction
module, and concluded the report was wrong.

`record` also passes the measurement trap that defeats the obvious hand-rolled version. `record`
calls `jax.clear_caches()`, so two calls in one process read 6.458 s then 7.772 s. Timing
`jax.make_jaxpr` in a process that has already lowered the same function reads **0.002 s for a 49 s
search**, because opt_einsum caches contraction paths independently of jax. My own structure probe
recorded exactly that misleading 0.002 s.

**Where I left scopex.** For the culprit, immediately and irreducibly. Every attribution view
returns a constant for both arms: `author`/`innermost_author`/`library` are `<none>` for all ten
equations, `package`/`role`/`split` are `<unmarked>`, `site` is `<no-frame>`, `transform` is
`<primal>`, and `scope` is the identical einsum spec string in both. That is structurally
unavoidable — `opt_einsum.contract_path` emits no equations at all, it only chooses an order, so a
jaxpr-unit attributor has nothing to attribute 49 s to. This is a genuine feature gap with an
obvious fix (a sampling profiler over the trace stage), not a maintainer's judgement call.

**How far I got.** Root cause named: exhaustive contraction-path DFS in `opt_einsum.paths`, in
Python, before a jaxpr exists. scopex named the stage; `cProfile` would have named the function.

### A2. `jitfib_t24` — `jax#22385`, CPU

**The symptom.** A `@jax.jit`-decorated recursive Fibonacci at t=24: backend 18.876 s vs a control
flat at 0.068 s. Note this re-frames the issue: on 0.10.2 trace is flat at ~0.03 s and lower flat at
~0.10 s in *both* arms, so what was filed as a tracing pathology is now a backend pathology.

**The trail.** `record` says backend, so tracing is innocent — that alone contradicts the issue title
and is worth knowing before touching anything else. Then the jaxpr: `len(list(scopex.walk(jaxpr)))`
= 2,959 / 20,293 / 53,131 / 139,102 at t=16/20/22/24 against a control at 29/37/41/45. Ratio x2.618
per +2 in t, which is phi^2 exactly. That is a signal that scales perfectly with the pathology
parameter and it is available before any compile.

But `len(str(jaxpr))` is 2,324/2,940/3,248/3,556 chars for the case against 5,945/9,229/11,123/13,185
for the control. **The pathological arm's jaxpr text is 3.7x smaller than the control's.** Any
byte-count instrument gives the opposite answer. The reason is the whole mechanism: the call DAG
reuses the same `ClosedJaxpr` object for `fib(t-3)` reached from two parents, so the printer emits it
once while `walk()` (correctly, matching `jaxpr_util.all_eqns(revisit_inner_jaxprs=True)`) revisits.
A sharing ratio built on `walk` plus `subjaxprs` reads 53,131 walked equations over 22 distinct
sub-jaxpr objects = 2,415x at t=22, against 41 over 21 = 2.0x for the control.

`scopex.attribute(walk(jaxpr), 'kind')` -> `{'jit': 35421, 'add': 17710}` at t=22. Then
`pass_timings`: `call-inliner` 6.330 s vs 0.00373 s = 1,697x, `flatten-call-graph` 1.500 s vs ~0.
Then `dump(passes='.*')` closes it: snapshot 0003 `[sharding-removal] after_flatten-call-graph` goes
57 -> 27,058 instructions and 21 -> 13,530 computations in one pass, **and that snapshot does not
exist in the control** — XLA writes a snapshot only when a pass changed the module, so a pass
present in one arm and absent in the other is itself the finding.

And the prettiest identity in the set: `attribute(walk(jaxpr),'kind')['jit']` = 35,421 at t=22, and
`flatten-call-graph` creates 35,422 computations. A jaxpr-level view predicts the HLO blowup to
within one computation, before any compile is run.

The final module is 2 instructions in both arms (`constant`, `copy`) — `constant_folding` evaluates
the whole argument-free program to one literal. So optimized HLO is a perfect null.

**Where scopex earned its keep.** Three independent knobs each sufficient, which is rare here.
`walk`'s revisiting semantics are the thing: the standard jaxpr pretty-printer is actively
misleading on this program and `walk` is not. Counterfactual for the pass story: I would have had to
write the vmodule subprocess, the pass-line parser and a snapshot-filename parser to learn that
`flatten-call-graph` fires in one arm and not the other.

**Where I left scopex.** Dump mtimes for the per-pass wall clock (flatten-call-graph 1.03 s,
call-inliner 7.42 s, constant_folding 0.76 s, everything after under 5 ms at t=22), and an
`id()`-keyed traversal for the DAG-vs-tree ratio. The first is a pure feature gap — the timestamps
are already on disk. The second is a five-line helper nobody will write by hand.

**How far I got.** Root cause named: the program is a DAG that XLA expands into a tree.
`flatten-call-graph` clones one computation per call site, `call-inliner` inlines all 35,421,
`constant_folding` evaluates the result. Verified pass by pass with wall clock.

### A3. `arity_tree_100` / `_200` — `jax#4667`, CPU

**The symptom.** `jax.grad` of a function taking a 200-leaf pytree at the jit boundary: wall
31.497 s vs 0.819 s (38x) at nleaves=200.

**The trail.** `record` is sufficient and every later knob is redundant — which is itself the
finding, because this is the only case whose dominant half no compiler-side instrument can see.
trace 14.183 s vs 0.114 s (124x), lower 2.451 vs 0.101 s (24x), backend 14.256 vs 0.223 s (64x).
trace+lower is 16.6 s of a 31.5 s wall, i.e. the *larger* half, and it is structurally invisible to
`pass_timings`, to `dump()`, and to every HLO-level view.

`walk(jaxpr)` = 4,600/9,200/18,400/36,800 at nleaves 25/50/100/200 — exactly `184 * nleaves` —
against a control flat at 183 for all four. `walk_hlo` 37,800 vs 189 (200x). StableHLO 84,845 lines
vs 463. `attribute(units,'site')` pins 4,450 of 4,600 equations to
`case_lowering_arity_pytree.py:124`, the `jax.tree.map` line, against the control's 178 of 183 at
line 137. That is the one attribution view that carried information on this case.

The dump is a clean null *for localisation* and that is worth stating: the pass sequence is
identical (37 case snapshots vs 38 — the control gains one `cpu-parallel-task-assigner`), and the
case's counts sit at a constant ~48x the control's at *every* snapshot index (pre-opt 9,056 vs 187,
after fusion 9,450 vs 189 at nleaves=50). No pass amplifies or repairs anything; the program is
simply 48x bigger the moment jax hands it over. `pass_timings` is 2.756 s vs 0.300 s = 9.2x —
sublinear in the 48x size ratio, so the HLO pipeline is not where the leverage is.

`before_optimizations.txt` shows `2 * nleaves` parameters (50 at nleaves=25, 100 at 50) against the
control's 2, which is jakevdp's mechanism literally counted. Scaling refutes his prediction on
0.10.2: nleaves 100 -> 200 moves trace 1.93x and backend 1.75x, i.e. linear in leaf count, not
"closer to quadratic".

**Where scopex earned its keep.** The stage split plus `attribute(units,'site')`. One call each, and
between them they say "trace is 124x its control and 97% of your equations come from this
`jax.tree.map` line". Counterfactual: `%prun` around `make_jaxpr` gets you the stage; nothing off
the shelf gets you the site histogram over 36,800 equations.

**Where I left scopex.** Two soft failures. `regime(t)` returns `mixed` on this arm even though
trace is 14.2 s, because the 0.6 threshold is applied to a wall that *includes* the 14.3 s backend —
a `mixed` label on a 124x trace regression is a missed call, and a user who reads it will go tune
XLA flags and lose the larger half. And `author`/`innermost_author`/`library`/`package`/`role`/
`split` are `<none>`/`<unmarked>` for all 36,800 equations, because the corpus cases are unmarked
single-file programs; only `site` worked.

**How far I got.** Root cause named: `2 * nleaves` arrays at the jit boundary, 184 jaxpr equations
per leaf, cost linear in leaf count and split roughly half upstream of XLA. Nothing was hidden at
any level; this is what the stack looks like when it works.

### A4. `condrec_grad_512` — `jax#8239`, CPU

**The symptom.** `jax.grad` through a recursive `lax.cond` at max_steps=512: `make_jaxpr` alone is
116.6 s vs 3.2 s. At max_steps=128, `record` gives trace 15.866 s vs 0.669 s (23.7x), lower 1.648 vs
0.457 s, backend 1.802 vs 0.662 s; `regime` says `trace-bound` vs `mixed`.

**The trail.** `record` says trace and lower, so XLA is exonerated up front — and the dump confirms
it and then some: the HLO ratio is 2.11x at snapshot 0 and stays between 1.92x and 2.08x across all
34 snapshots (final 638 vs 313). XLA not only fails to amplify the problem, it *understates* it: 2x
at HLO for an 11x jaxpr and a 24x trace. Below HLO there is nothing either — the largest kernels are
~1,830-line `.ll` files repeated identically in both arms.

Equation counts discriminate and the ratio grows with the parameter: 4,350 vs 478 (9.1x) at
max_steps=32, 21,502 vs 1,918 (11.2x) at 128, 102,398 vs 7,678 (13.3x) at 512, with `make_jaxpr`
time ratios of 20.8x / 26.2x / 36.2x.

Then the diagnostic payload, and it refutes the reading the issue title invites.
`attribute(units,'transform')` = `{jvp: 96766, 'transpose/jvp': 5632}` on the grad arm against
`{'<primal>': 7678}` on the control. `jax#8239` is filed as an AD-*transpose* problem; the
linearisation side is 17x the transpose side. What explodes is the forward JVP of `2*max_steps - 1`
inlined `lax.cond` levels, not the cotangent pass. `attribute(units,'site')` moves the mass off the
MLP payload and onto the recursion construct: 66,049 of 102,398 equations at
`case_ad_transpose_cond_recursion.py:139`, which is
`return lax.cond(cond_fun(val), go, lambda v: v, val)`, against the control's top sites at lines 160
and 161 (the two `jnp.tanh(x @ w + b)` layers) with 1,536 each.

**Where scopex earned its keep.** `attribute(units,'transform')`. The transform census is the only
view in the stack that separates primal from JVP from transpose equations, and here it overturns the
issue's own framing in one call. Counterfactual: grep `str(jaxpr)` for transform annotations across
102,398 equations and hope the pretty-printer preserved them.

**Where I left scopex.** 16,380 of the 102,398 equations (16%) attribute to `site` `<no-frame>`, so a
sixth of the blowup has no source line at all. And there is no view that says "this equation count
is what your `lower_s` is buying"; the stage that actually hurts at max_steps=512 (trace+lower at
120.6 s) is reachable only via `record`'s two fields.

**How far I got.** Root cause named, and the issue's framing corrected: forward-mode linearisation
of inlined `cond` levels, 17x the transpose side, localised to one source line. The corpus harness
scores this arm "no" because it reads `compile_s` while `lower_s` is 15.2x its control — a reminder
that a one-stage instrument renders a verdict on the wrong stage.

## B. Cost in one HLO pass

### B1. `adconst_idx_2p20` — `pyroki#56` / `jax#12789`, CPU

**The symptom.** Backend 10.037 s vs 0.728 s (13.8x) at M=2^20; 2.647 vs 0.331 (8.0x) at 2^18;
58.711 vs 3.407 (17.2x) at 2^22. The ratio grows with M.

**The trail.** This is the one case in the corpus where `pass_timings` is the right *first* knob
after `record` and simply gets it: `constant_folding` = 17.60 s of an 18.49 s compile (95%) against
0.032 s of 1.32 s in the control, a 550x on one pass. Measured through the scopex API directly at
2^18: `constant_folding` 5.26 s vs 0.0046 s = 1,135x. Done in two calls.

Everything else in this arm is a trap, and the traps are instructive.

The structural signal points the **wrong way**. Optimized HLO is 2 instructions
(`{constant: 1, copy: 1}`) against the control's 24. The entry signature is `() -> f64[1048576]` —
*no parameters at all*. `temp_bytes` 0 vs 1,048,576. Buffer allocations 2 vs 8. The dump contains
**zero `.ll` files and zero object bytes**; nothing is code-generated. Any "rank by IR size"
heuristic scores the pathological arm as the healthy one. The per-pass curve *falls*: 8 instrs -> 4
at `after_constant_folding` -> 1 at `after_dce` -> 2, while the control holds 16-24 throughout. You
have to read the falling curve and the constant's *shape*, not any count.

And XLA diagnoses itself in words. The raw vmodule stream contains
`slow_operation_alarm.cc:73] Constant folding an instruction is taking > 1s:` followed by
`The operation took 17.590259602s`. Neither scopex nor the corpus harness surfaces it — the harness
greps only for the string `Very slow compile?`, so its `xla_slow_warning` field reads False on every
`adconst` arm while XLA is printing a verbatim diagnosis of the pathology on stderr.

**Where scopex earned its keep.** `pass_timings` named the pass and the percentage in one call, and
`dump()` proved the zero-codegen fingerprint. Counterfactual: the vmodule subprocess plus the pass
parser, which is exactly what `pass_timings` is.

**Where I left scopex.** For the alarm text, the `entry_computation_layout` line showing `()` as the
parameter list, and the *absence* of `.ll`/`.o` files in the dump directory. All three are cheap
feature gaps. Also a real defect: `pass_timings`' total is inflated by nested-pipeline lines — XLA
logs `HLO pass: simplification time: 3.52 s` for the *pipeline containing* `constant_folding` in the
same format as its members, and both are summed, so this arm reported 10.54 s of pass time against a
5.4 s compile. An impossible number, returned without complaint.

**How far I got.** Root cause named: AD through a scatter produces an argument-free subgraph that
`constant_folding` materialises as a 2^20-element f64 literal, at 95% of the compile, and a
zero-kernel executable is the fingerprint.

### B2. `dusfold_sum_300` / `_350` — `jax#12789`, GPU

**The symptom.** Backend 1.083 / 3.083 / 4.534 s at n=200/300/350 against a control flat at
0.098/0.107/0.114 s = 11x / 28.8x / 39.8x, superlinear in the literal's element count.

**The trail.** `pass_timings` on both arms: `constant_folding` = 6.070 s vs 0.0009 s = 6,745x on an
identical 188-pass sequence, and nothing else in either profile exceeds 15 ms. Two calls again.

The traps are worse than in B1. **The optimized HLO is a perfect trap:** both arms end at exactly
2 instructions (`constant` + `copy`), shape `f64[]`, identical opcode histogram, identical
`hlo_opt_lines` = 28, identical generated-code bytes. Anyone comparing final HLO concludes the two
programs are the same. The jaxpr differs 7 vs 2 equations but is flat in n, so it cannot be the axis.

**And the per-pass instruction curve is also blind, for an instructive reason: the guilty pass
shrinks the module.** The case peaks at 96 instructions then collapses to 1; snapshot count is 58 in
both arms. So I rebuilt the curve in **bytes** — summing `sizeof(shape)` over every instruction per
snapshot — and it localises cleanly at n=200: the case carries 128 MB at pass 0, peaks at **320 MB**
at `simplification/scatter-slice-simplifier`, holds 256-320 MB for four more passes, then drops to 0
at `simplification/simplify-conditional`; the control starts at 64 MB and is already at 0 by
snapshot 6, six passes earlier, never exceeding 64 MB. Peak module bytes 320 vs 64 MB.

Independent confirmation from RSS sampling: peak excess over control 0.176 / 0.600 / 0.955 GB for
64 / 216 / 343 MB literals — the folder holds roughly 2.8x the literal, and the excess tracks n^3.

**Where scopex earned its keep.** `pass_timings` again, and `dump()` for the raw material of the byte
curve. Counterfactual for the byte curve: it does not exist as an instrument anywhere; I had to
write the shape-sizeof parser over 58 snapshot files per arm.

**Where I left scopex.** The byte metric, the RSS trajectory, and XLA's own alarm. `pass_timings`
populates `stderr_tail` only when the pass parse *fails*, i.e. never when you have a real profile —
so on the one case in the corpus that ships compiler-authored ground truth, scopex discards it at
the moment of success.

**How far I got.** Root cause named: `dynamic_update_slice` over a materialised literal, folded at
compile time, with the cost superlinear in literal bytes. Mechanism localised to one named pass with
a byte curve that shows the materialisation and the release.

### B3. `bisect_m94` / `bisect_m95` — `jax#10621`, CPU

**The symptom.** A bisection eigenvalue solver at m=95 with unroll=48: backend 179.3 s (one arm) /
250.5 s at m=94 (another, under different contention) against the m=96 control at 2.48-3.007 s —
60x-83x. Non-monotone in m: 96 is *fast* and its neighbours 94 and 95 are catastrophic.

**The trail.** `record` says backend and nothing else: 53 jaxpr equations in *both* arms with
identical primitives and identical nesting, lower 0.48 vs 0.24 s, StableHLO 801-806 vs 564 lines =
1.42x. So upstream is nearly blind on purpose, and a 1.4x StableHLO becomes a 60x HLO — the
amplification is entirely inside XLA.

`walk_hlo` says how much: 337,331 units vs 5,498 (61x) at m=95; 330,397 vs 5,498 (60.1x) at m=94.
And the opcode census says it is qualitatively different, not merely bigger: static `slice` 46,268
vs 570 while `dynamic-slice` drops out of the case *entirely* (control 294). 306 computations vs
202. Max operand count 289 vs 55.

Then `dump(passes='.*')` localises the cliff to **one pass**. Snapshot 31
(`after_layout_assignment.pipeline-end`) is 3,605 instructions vs 2,050 — 1.76x, the same program.
Snapshot 32 (`after fusion`) is 382,246 vs 5,507 — 69x. The `fusion` pass alone multiplies the m=95
module by 106x while doing 2.7x to the m=96 module; max operand count jumps 20 -> 289 in the same
step; three extra `simplification_after_layout_assignment` iterations then trim it to 337,333, which
is why the case runs 45 pass snapshots to the control's 38.

XLA's own timer agrees, and this is where scopex fails: the raw `hlo_pass_pipeline=1` log contains
exactly one line above one second besides `cpu-parallel-task-assigner`, and it is
`HLO pass: fusion time: 2.87 min` = 172 s of a 395 s compile (at m=94: `fusion time: 2.06 min
(123480735 us)` = 123.5 s of 301 s). `scopex.flags._PASS_LINE` requires
`[\d.]+\s*(us|ms|s)` and `_UNIT` knows only those three, so the line **fails to match and is dropped
silently**. `pass_timings` reports total 18.69 s and names `copy-insertion: 13.0 s` as the most
expensive pass — wrong by 13x, and pointing at the wrong pass. A unit-tolerant re-parse of the same
log gives 161.12 s. Verified in isolation: `_PASS_LINE.search()` returns `None` on that exact line
and matches fine once `min` is edited to `s`.

The mechanism came from a cheap two-axis sweep. On m at unroll=48: m=64/80/94 (`m//48 == 1`) ->
65.9 s / 155,857 instrs, 136.6 s / 240,881, 250.5 s / 330,397; m=96/112 (`m//48 >= 2`) -> 3.0 s /
5,498 and 5.5 s / 7,978. On unroll at m=95: u16 (5 trips) 3.9 s, u24 (3 trips) 3.7 s, u32 (2 trips)
5.0 s — all fast; u48 (1 trip) is the reported hang. **The cliff is not at m=95/96, it is at
`length // unroll == 1`.** jax emits a single-trip while loop, XLA inlines it, the whole m-step Sturm
sequence becomes straight-line code under two vmaps and the outer unroll=3, and fusion replicates
producers into every consumer. Non-monotone in m only because the trip count is floor-divided.

A lower-only m sweep confirms the periodicity *before any compile*: StableHLO lines are 564 at m=96
(rem 0), 651 at m=64 and m=112 (rem 16), 731 at m=80 and m=128 (rem 32), 801 at m=94, 806 at m=95.
m=96 is a local minimum surrounded by neighbours 1.43x larger — a 1.43x text signal for a 71x
effect, at the cost of a lowering.

**Where scopex earned its keep.** `walk_hlo` plus its opcode census, and `dump(passes='.*')` for the
raw material. The `dynamic-slice` -> `slice` substitution is the mechanism visible as an opcode
census; nothing else names it.

**Where I left scopex.** Everywhere the answer actually was. I wrote the HLO-text instruction
counter, the snapshot-filename parser
(`module_NNNN.<name>.NNNN.<pipeline>.after_X.before_Y.txt`), the sequence-equality check, and the
unit-tolerant pass-line re-parse. The dump itself was 944 MB / 950 files on the m=95 arm and took
358 s — `dump(passes='.*')` on a 24-equation jaxpr, with no warning about the size.

**How far I got.** Root cause named and generalised past the issue's own framing (a `m mod unroll`
story becomes a `m // unroll == 1` story), with the guilty pass timed. This is the one case where
XLA's own instrument would have handed a competent engineer the answer in one call, and a one-token
regex bug converts it into a misleading answer.

### B4. `switch_ident_512` / `_1024` — `jax#4453`, GPU

**The symptom.** `lax.switch` over 512 identical branches: backend 31.85 s vs 0.166 s (192x); at
n=1024, 256.46 s vs 0.165 s (1,551x).

**The trail.** At n=512, `pass_timings` is decisive: `copy-insertion` = 41.00 s against the
control's 0.00032 s = 128,000x, and 41.0 s against a 31.9 s record-backend — the pass *is* the
compile. Rung below: 5.91 s at n=256, i.e. 6.94x for 2x branches, about n^2.8, matching the 7.28x
backend growth.

Every structural count is exactly linear and therefore cannot explain the superlinearity. Sweep
64/128/256/512/1024: backend 0.983/1.038/4.377/31.853/256.458 s while jaxpr equations go
66/130/258/514/1026, optimized HLO instructions 394/778/1546/3082/6154, HLO computations
65/129/257/513/1025, `subjaxprs` = n exactly. Counts confirm the multiplicity and stop there. The
derived per-computation cost does explain it: 15/8/17/62/250 ms per computation, a 30x rise across
the sweep.

**The dump is a null here and that is itself the result.** `d_switch_ident_128` against
`d_switch_ident_256`: 3 PTX files vs 3, 58 PTX lines vs 58, `ir-with-opt.ll` 57 vs 57 lines,
`ir-no-opt.ll` 52 vs 52 — byte-identical codegen output at twice the branches. And the per-pass
instruction curve shows no change at all across the `copy-insertion` boundary; the 41 s pass leaves
the module the same size. This **refutes the issue's own 2020 diagnosis** (hawkinsp profiled it into
LLVM): on jax 0.10.2 / XLA:GPU the cost is 100% an HLO pass, `CopyInsertion`'s interference analysis
over 513 computations, and LLVM has 58 lines of PTX to emit regardless of n.

At n=1024 the localising instrument fails. The raw log reads
`HLO pass: copy-insertion time: 2.95 min (177154325 us) (cumulative: 2.98 min, max: 2.95 min,
#called: 281)` against `WALL_COMPILE 188.46 s` — 94.0% of the compile. That is the **only** one of
299 pass lines in the log using `min` units, so `_PASS_LINE` drops it and `pass_timings` returns a
non-empty, plausible profile whose largest entry is a millisecond-scale pass. n=512 (41.0 s, under
60 s) is a clean hit; n=1024 (177 s, over it) is a confident wrong answer. **The tool appears to
work on the small rung of a ladder and fails silently on the big one** — the most dangerous possible
failure shape for a scaling study. The parenthesised `(177154325 us)` on the same line would have
parsed.

**Where scopex earned its keep.** At n=512, `pass_timings` in one call. Everywhere,
`len({i.container for i in walk_hlo(compiled)})` = 1025 vs 1 names the multiplicity, which the case
file says is the only correct answer.

**Where I left scopex.** The raw vmodule capture, to discover that the n=1024 profile was a lie. A
maintainer who trusted the tool would have gone looking in LLVM, as the issue did in 2020.

**How far I got.** Mechanism localised to one pass with wall clock, the issue's original diagnosis
refuted on this build, and the scaling exponent measured. Root cause is `CopyInsertion` on 513-1025
computations; the interference-analysis complexity itself was not read out of XLA source.

## C. Cost below HLO — the emitter and LLVM

### C1. `jax#32704`, chained 2-D fancy indexing, CPU — investigated three times independently

Two case files (`case_gather_2d_chain.py`, `case_indexing_lowering_gather_chain.py`) and three
independent passes over them. All three converged on the same place from different directions, which
is the strongest evidence in this document about where scopex's floor is.

**The symptom.** `x[i, j]` chained n times, against a control that flattens the index arithmetic to
a single 1-D gather. `gather2d_8`: backend 16.90 s vs 0.199 s (85x). `gatherchain2d_9`: 99.67 s vs
0.191 s (522x). `gatherchain2d_10` at ncycles 5/6/7/8: 0.436/1.385/3.820/17.707 s against a control
flat at 0.173/0.202/0.224/0.212 s. Roughly 4x per added link. **Flat on GPU.**

**The trail — and this is a trail of nulls.** `record`: `backend-bound` in both arms, trace ~0.02 s
and lower ~0.07-0.10 s in both, so the split does not differ in *kind* and is useless for
discrimination. Then, in order:

- `pass_timings`: 93 passes, total **0.0169 s (case) vs 0.0165 s (control)** at ncycles=8 —
  1.02x, identical. So HLO passes are 0.066% of the case's compile against 6.9% of the control's,
  and that fraction collapses monotonically with chain length (6.9% control / 1.4% at n=6 / 0.066%
  at n=8). On the other case file, 0.0207 s vs 0.0223 s — the *pathological* arm's pass time is
  lower. **This null is the clue**: a correct, scaling, quantitative statement that the time is
  outside the HLO pass pipeline.
- `walk(jaxpr)`: 80 vs 56 equations = 1.43x, and the ratio does not grow with ncycles. Null.
- `walk_hlo`: 73 vs 70 = 1.04x. 82 vs 78 on the other file. Structurally near-identical. Null.
- `dump(passes='.*')`: pass sequence byte-identical, 26 snapshots each, same names, same order;
  per-snapshot instruction ratio peaks at 1.34x and settles at 1.02-1.04x from pass 10 on. No pass
  grows the module. Null.
- LLVM artifacts from the dump: `ir-no-opt.ll` 358 vs 319 lines (1.12x), `ir-with-opt.ll` 752 vs 724
  lines, object file 5,224 vs 5,096 bytes (1.03x) — and on the other file the **control's object code
  is larger**. Both arms add exactly +18 LLVM IR lines per added link, i.e. identical slope. Null
  below HLO too.

So: not tracing, not lowering, not the HLO passes, not the HLO structure, not the emitted code size.
The only positive discriminators available anywhere in scopex are qualitative:
`attribute(walk_hlo(c),'kind')['concatenate']` = 4 / 6 / 8 at ncycles 4 / 6 / 8 against 0 / 0 / 0 in
every control — an absolute discriminator that tracks the pathology parameter 1:1 — and the gather's
`start_indices` shape `s32[1000000,2]` vs `s32[1000000,1]` with `start_index_map={0,1}` vs `{0}`,
which is the causal parameter but a constant 2-vs-1 that does not grow.

**Three routes out, all leaving scopex.**

1. **Kill-switch A/B.** Fresh process each, cores pinned, `gather2d_8`: baseline 22.60 s;
   `--xla_disable_all_hlo_passes=true` 3.59 s; `--xla_llvm_disable_expensive_passes=true` 23.76 s;
   `--xla_backend_optimization_level=0` 24.01 s; `--xla_cpu_opt_preset=FAST_COMPILE` 23.98 s;
   **`--xla_cpu_use_fusion_emitters=false` 0.320 s** (control 0.153 s). Neither the pass pipeline's
   runtime nor LLVM's optimiser owns the time — the XLA:CPU MLIR fusion emitter does.
   `--xla_dump_emitter_re='.*'` then produced the localising artifact:
   `gather_bitcast_fusion_kernel_module-pre-optimization.mlir` at ncycles 4 = 5,469 lines vs control
   86 (64x); ncycles 6 = 87,389 vs 116 (753x); ncycles 8 = **1,398,109 vs 146 (9,576x)**. Ratios
   between rungs are 15.98 and 16.00 — exactly 4x per added link, the same 4x/link the issue reports
   for compile time. `mlir-passes.log` shows the case *enters* the MLIR pipeline at 1,398,113 lines,
   is folded to 4,621 by the first `CanonicalizerPass` and to 94 by the following `CSEPass`, after
   which the two arms stay within 10% of each other for 56 more passes. Hence LLVM IR and object
   code are 1.04x while compile is 106x.
2. **Raw glog timestamps.** Under
   `TF_CPP_VMODULE=cpu_compiler=3,fusion_emitter=2,buffer_assignment=1,thunk_emitter=2,jit_compiler=3`,
   the largest wall-clock gap between consecutive glog lines at ncycles=9 is **98.906 s of a
   99.534 s compile**, sitting after
   `fusion_emitter.cc:217] Emitting loop fusion kernel: gather_bitcast_fusion`. The control's same
   line costs 0.055 s -> 1,798x on that one phase. Sweep: 1.277 s (nc6, 90% of compile) / 5.579 s
   (nc7, 97%) / 27.455 s (nc8, 98%) / 98.906 s (nc9, 99.4%). Both arms also log
   `fusion_emitter.cc:302] Tiled emitter failed due to tiling failure: UNIMPLEMENTED: Fusion is not
   supported by the tiled CPU emitter, falling back to loop emitter` — the reason the slow emitter
   runs at all, and that veto line exists only in the vmodule stream, never in the dump directory.
3. **Dump file mtimes.** With `dump(passes='.*', fusion=False)`, the interval between the last
   numbered per-pass snapshot and `<module>.*.ir-no-opt.ll` is XLA:CPU's IR-emission phase:
   **0.231 / 1.262 / 3.736 / 29.957 s** at ncycles 5/6/7/8 against a dead-flat 0.070/0.072/0.075 s
   for the control — 3.3x / 17.5x / 49.8x / ~400x, and at ncycles=8 that is 29.957 s of a 30.203 s
   compile (99.2%). The llvm-opt gap (0.03-0.12 s) and codegen gap (0.02-0.04 s) are flat and
   identical in both arms. **The only quantity that separates the arms is the wall time between two
   dump writes, and it is already on disk after `scopex.dump()`.**

**Mechanism.** An index-rank ablation at constant data size settles it. A chain of R-component fancy
indexes, R=1..4, at depth 4/5/6: R=1 flat in depth (0.468/0.177/0.881 s); R=2 0.507/0.947/3.217;
R=3 0.969/4.155/45.841; R=4 10.250/36.434. At fixed depth 5 the R ladder is 0.18 -> 0.95 -> 4.16 ->
36.4 s, i.e. x5.3, x4.4, x8.8 per added start-index component. R=1 — exactly the flattened control —
is flat. So cost is exponential jointly in (index rank, chain depth). XLA:CPU's `FusedIrEmitter`
memoises generated values on `(HloInstruction*, multidim index)` and invalidates across basic
blocks; a gather calls its start-indices generator once per `index_vector_dim` component, and a
concatenate emits one basic block per operand — so each 2-D link multiplies the descent count by
(2 components x 2 concat operands) = 4. **The emitter's work is exponential while its output is
linear**, which is precisely why no artifact scopex can measure grows: the answer is folded away by
MLIR canonicalisation before any level scopex exposes exists.

**Where scopex earned its keep.** By excluding, quantitatively and scalably. `sum(pass_timings) /
record()['backend']` = 0.066% at ncycles=8, falling monotonically with chain length, is a correct
statement that the time is outside the HLO pass pipeline — and combined with the 1.04x optimized HLO
and the 1.12x LLVM IR it forced the conclusion "the cost is in a phase that transforms nothing",
which is a strong and unusual claim to be able to make. The `concatenate` census gave the shape of
the mechanism. Counterfactual: without `pass_timings` and `dump` I would have spent the time
comparing HLO by hand and finding it identical.

**Where I left scopex.** Completely, for the localisation. `scopex.dump_flags()` emits only
`--xla_dump_to` and `--xla_dump_hlo_pass_re`, so `dump()` physically cannot produce the
`-pre-optimization.mlir` or `mlir-passes.log` files that hold the answer. There is no accessor for
emitter-stage IR, no phase timeline, and no way to ask "how long between these two artifacts". One
ablation was also invalid and worth recording: inserting `lax.optimization_barrier` between links
made the cost 3x *worse*, because XLA erases `optimization_barrier` before fusion — scopex's own
`TRAPS['barrier_erased']` documents exactly this, so the arm was never a valid fusion blocker.

**How far I got.** Root cause named, mechanism explained, base of the exponential measured, and the
`FusedIrEmitter` `value_cache_` keying read out of XLA source. Three independent instruments agree
on 98-99% in one emitter call. None of the three is in scopex, and one of them (mtimes) is free.

### C2. `stackcond_n3000` / `_n10000` / `_n30000` — `diffrax#606`, CPU

**The symptom.** `jnp.stack` of N traced scalars inside a `lax.cond`. Backend 8.85-9.20 s vs 0.114 s
at N=3000 (60-81x); 79.96-82.64 s vs 0.123 s at N=10000 (509-672x). Grows as about N^1.8-2.0 —
quadratic in operand count for a linearly growing program.

**The trail.** `record` says backend in both arms. Then the documented trap fires:
`len(list(walk(jaxpr)))` = **6 vs 6**. A perfect null, 1.00x, at every N. And `walk_hlo` = 30 vs 31
— null *and inverted*, the pathological arm has fewer instructions, with byte-identical kind counts
(parameter 7, broadcast 3, constant 2, fusion 2, add 1, bitcast 1) and identical
`fusion`-flagged and `outlined` flags. HLO text 77 vs 81 lines. **Two of scopex's structured views
report the pathological arm as the smaller program.**

What works, in the order I found it:

- A custom callable to `attribute`, which is the escape hatch:
  `attribute(list(walk(jaxpr)), lambda u: len(u.eqn.invars))` -> `{2:2, 1:3, 3000:1}` vs
  `{2:2, 1:4}`. **Max arity N vs 2**, at an equation count of 6 in both arms. Exact, and linear in N
  by construction. This is the whole pathology and it required breaking the abstraction.
- Raw StableHLO text length: `len(stablehlo_text(lowered).splitlines())` = 3,245 vs 45 at N=3000
  (72x), 10,713 vs 45 at N=10000 (238x). Linear in N, available before any compile, one call. The
  single `stack` equation lowers to N reshapes plus one concatenate.
- The dump's `before_optimizations.txt`: 3,016 vs 16 instructions at N=3000 (188x), 10,016 vs 16 at
  N=10000 (626x). Then the `cse` pass collapses 3,021 -> 22 at snapshot 8 because all N reshapes are
  identical, and the module stays ~22-32 instructions for the remaining 17 passes — **but max
  operand count stays pinned at N in every snapshot including `cpu_after_optimizations`, versus 3 in
  the control.** The N-operand concatenate survives the whole pipeline; the *count* signal exists
  only in the first eight snapshots.
- `pass_timings`: 0.511-0.872 s vs 0.009-0.010 s, but only 3.3-5% of the compile, with
  `layout-assignment` (0.33-0.66 s) as its top entry. Correctly says "not here", and its top entry
  is a red herring.
- The dump below HLO, which is where the cost is:
  `module_0000.jit__slow.stack.6001_elemental_kernel_module.ir-no-opt.ll` at 21,054 lines at N=3000;
  `stack.20001_elemental_kernel_module` at 70,054 lines / 5.82 MB at N=10000 (110,328 total `.ll`
  lines = 281x), against a control whose largest `.ll` is 78 lines and largest object 952 bytes.
  Strictly 7.1 LLVM instructions per operand — 4,011 `getelementptr` + 2,010 `load` per 1,000
  operands. Linear IR, superlinear compile: one huge basic block.
- Dump mtimes give the split at N=3000: HLO passes 0.373 s, IR emission 0.230 s, LLVM opt 1.912 s,
  **LLVM codegen (ir-with-opt.ll -> .o) 8.762 s = 78% of the compile**. Codegen grows 6.58x for 3x N
  (exponent 1.72) on IR that grows 2.78x. A raw vmodule gap confirms it end to end: 16.025 s between
  `cpu_compiler.cc:2028] Collected 3 compiled symbols` and `cpu_compiler.cc:2154] Compilation
  finished`, against the control's 0.121 s — 132x on ORC JIT alone.

**Where scopex earned its keep.** `stablehlo_text` line count: one call, before any compile, 72x-238x
and linear in N. And `attribute`'s acceptance of an arbitrary callable, which is the only reason the
arity histogram was reachable at all.

**Where I left scopex.** For the thing that *is* the pathology. Operand arity is not representable
anywhere in scopex: `hlo_instructions()` yields dicts with exactly
name/opcode/shape/computation/op_name/source_file/source_line/stack_frame_id — no operand list and no
operand count — and `_to_ins()` then drops even `shape`, so `Ins` objects carry neither arity nor
shape. No name in `scopex.BY` exposes it. The one number that is 1,000x-3,333x apart is the one
number the API has no slot for, exactly as the case docstring predicted: *"the size is in the OPERAND
LIST of one node, which most IR summaries never print."* Also outside: the pre-optimization module,
the `.ll` opcode histogram, and the mtimes separating LLVM opt from codegen.

**How far I got.** Root cause named: one concatenate with N operands lowers to a single
fully-unrolled straight-line kernel, and LLVM's ISel/regalloc on one enormous basic block is
superlinear. Split measured to the phase (78% in codegen at N=3000).

### C3. `argsort_f32_1e6` / `_1e7` — `xla#35587`, GPU

**The symptom.** `jnp.argsort` on f32 at n=1e5/1e6/1e7: backend 5.537/5.534/7.105 s against an i32
control flat at 0.412/0.469/0.406 s = 13.4x / 11.8x / 17.5x. Compile is nearly *flat* in n, exactly
as expected for an O(log^2 n)-stage cost.

**The trail.** The jaxpr is byte-identical between arms at every n (5 equations: jit, iota, sort,
slice, reduce_sum) so the jaxpr level is a guaranteed null. Optimized HLO is 44 vs 12 instructions
(3.67x) — present but far too small to explain 11.8x on a nearly-flat curve.

Four independent, mutually confirming signals, all from text or from the dump:

1. **`custom_call_target` census over `hlo_text(compiled)`:** `xla.gpu.ext.cub_sort_pairs` is present
   in the control and **absent** in the case. One boolean that names `SortRewriter`'s decision.
2. **Pass-name set difference:** `estimate-cub-sort-scratch-size` appears only in the control's
   189-pass list.
3. **Opcode census:** the case carries `compare` x10, `select` x7, `xor` x2, `bitcast` x2 — the
   IEEE-754 total-order comparator — where the control has zero of each.
4. **Below HLO:** 107 PTX kernels vs 3 and 48,317 PTX lines vs 107 (451x) at n=1e7; 57 vs 3 and
   34,466 vs 110 (313x) at n=1e6; `ir-with-opt.ll` 63,013 vs 124 lines (508x);
   `thunk_sequence.txt` 56 kernel thunks (55 named `sort`) vs 3. The PTX line distribution shows the
   comparator inlined: 35 kernels of 151 lines, 9 of 1,848, one of 10,897, i.e. ~605 lines per kernel
   against the control's ~37. Both factors of "comparator size x number of stages" are separately
   measurable.

`pass_timings` **under-accounts by 14x** and correctly says nothing is slow: total 0.202 s against a
7.105 s backend = 2.8% coverage at n=1e7 (0.397 s / 5.534 s = 7.2% at n=1e6), largest pass 0.16 s.
The time is in LLVM/PTX emission for 56-107 kernels, below where the pass timer reaches. **A coverage
number of 0.028 is itself the finding** and scopex does not report it.

A discriminating probe closes the loop: `argsort_bitcast_i32_1e6` (float bytes reinterpreted as
integers, so an integer comparator) compiles in 0.370 s and *does* carry `cub_sort_pairs`. The
variable is the comparator, not the data.

**Where scopex earned its keep.** `hlo_text(compiled)` plus a grep. That is a thin claim and it
should be thin: the discriminator was one string in the text, and scopex's contribution was giving
me the text without hitting the `compiler_ir('hlo')` metadata-dropping trap it documents.

**Where I left scopex.** Immediately, and because of a bug. **`hlo_instructions` silently drops every
tuple-shaped instruction.** `_INSTR`'s shape group is `\S+`, and a tuple shape
`(s32[1000000]{0}, s64[1000000]{0}, u8[12342527]{0})` contains spaces. I verified 12 parsed
instructions against 13 assignment lines in the control's optimized HLO, and the one missed line is
the `cub_sort_pairs` custom-call — **the single most diagnostic instruction in the program, and the
entire answer for this case, invisible to `walk_hlo`.** The same bug loses `sort`, `while`,
`conditional` and every multi-output fusion, i.e. exactly the primitives this corpus is built from.
Also outside scopex: every PTX and thunk count.

**How far I got.** Root cause named: `SortRewriter` does not fire for an f32 comparator, so XLA emits
a bitonic sort as 56-107 separate kernels instead of one CUB custom-call, and the cost is LLVM/PTX
emission per kernel. This is the cleanest case in the corpus for the proposition that a pathology can
be simultaneously invisible in counts, invisible to the pass timer, and screaming in the codegen
artifacts.

### C4. `ndtri_scan_jacrev_d16` / `_d32` — `jax#2609`, CPU

**The symptom.** `jacrev` of a `lax.scan` over the Cephes `ndtri`: backend 102.058-253.099 s vs
4.502-8.403 s at depth 16 (22.7x-30.1x), and 9.068 vs 1.394 s at depth 4 (6.5x). The *ratio* grows
with depth.

**The trail.** `record` says backend is 91% of the case's 112 s wall at d16, with lower at 1.734 s.
**That refutes the case file's own prediction** ("expect the cost in `lower_s`, not `compile_s`"), and
only the stage split settles it. A second file claim is also refuted: the `jacfwd` control reads
backend 146.294 s against `jacrev`'s 102.058 s, so forward mode is *slower* on 0.10.2 and the
AD-direction control is inverted.

Then the shape of the answer is a monotone cascade that widens at every level you descend, and the
cascade *is* the diagnosis:

| level | case (d16) | control | ratio |
|---|---|---|---|
| `walk(jaxpr)` equations | 7,885 | 3,437 | 2.29x |
| `stablehlo_text` lines | 14,472 | 4,044 | 3.58x |
| dump `before_optimizations` instrs | 22,368 | 5,100 | 4.39x |
| `walk_hlo` instructions | 108,056 | 17,689 | 6.11x |
| HLO computations | 3,227 | 424 | 7.6x |
| LLVM kernel modules | 344-688 | 39-78 | 8.8x-8.9x |
| total `.ll` lines | 46,353-84,982 | 5,012-9,512 | 8.9x-9.2x |
| object bytes | 459,216 | 62,696 | 7.3x |

The module is born ~4x too big by jax's AD-through-scan and XLA's `fusion` pass multiplies the excess
again: the arms enter the pipeline 3.5x apart (5,763 vs 1,414 pre-opt at d4) and leave `fusion` 6.0x
apart (25,482 vs 4,262), because fusion multiplies the case by 3.93x and the control by 2.32x.

The post-fusion opcode census names the mechanism in one line: `select` 1,404 vs 300, `compare`
1,146 vs 278, `dynamic-slice` 618 vs 0. That is AD keeping residuals for **both** arms of the
piecewise Cephes `lax.select`, exactly as the case file predicts.

There is also a regime *shift* between the arms that no label reports: the control is pass-bound
(HLO passes 6.31 s of an 8.31 s compile = 76%), the case is codegen-bound (HLO passes 37.78 s of
210.7 s = 18%; ORC JIT 110 s = 52%; one fusion emitter 36.6 s; buffer-assignment heap simulation
12.1 s). On XLA:CPU the driver is kernel *count*, not kernel size — 344 separate LLVM modules each
paying its own opt and ISel.

**The scale is misleading and that is worth its own sentence.** From d4 to d16 the structural ratios
are essentially flat (jaxpr 2.28 -> 2.29, StableHLO 3.20 -> 3.58, optimized HLO 5.98 -> 6.11) while
the time ratio grows 4.8x -> 30.1x, because the cost is superlinear in *absolute* module size. A
reader who checks only the ratio at one depth concludes the effect is mild. The absolute counts, not
the ratios, predict the time.

**Where scopex earned its keep.** The four-level census, one line per level. The widening ratio down
the stack is the finding, and getting it required only `walk`, `stablehlo_text` and `walk_hlo`.
Counterfactual: three separate hand-written parsers and no reason to line them up.

**Where I left scopex.** The kernel-module count, the `.ll` line totals, the object-byte totals, the
per-snapshot opcode histograms, and the raw-log gaps at `buffer_assignment.cc:2214` (heap
simulation, 12.1 s) and `cpu_compiler.cc:2028 -> 2154` (ORC JIT, 110 s). `dump()`'s docstring
advertises the GPU `priority_fusion_dump` and mentions CPU `.ll`/`.o` files only in passing, so a
reader never learns that the CPU dump contains a per-kernel codegen bill.

**How far I got.** Root cause named (AD retains residuals for both `select` arms; fusion amplifies;
CPU pays per kernel), two of the case file's own predictions refuted, and the phase split measured.

## D. Cost in autotuning

### D1. `convT64_dilate16` — `xla#5541` / `jax#17464`, GPU. The one arm with no signal.

**The symptom.** A dilated transposed convolution: backend 75.54 s vs 2.227 s (33.9x), both arms
labelled `backend-bound`.

**The trail, which is a complete null.** Exhaustively, because the point of the exercise was to find
out whether *anything* reaches it:

`n_xla_modules` 5 vs 5. PTX files 5 vs 5. Per-pass snapshots 71 vs 71. Pre-optimization instructions
8 vs 8. Optimized instructions 23 vs 23. Thunk kernels 5 vs 5. `priority_fusion_dump` files 1 vs 1.
Fusion candidates 1 vs 1. Computations 5 vs 5. Fusions 4 vs 4. **Identical** opcode histogram
(parameter 8, fusion 4, constant 3, reduce 2, transpose 2, add 1, bitcast 1, pad 1, gte 1). Same
`custom_call_target` `__cudnn$convForward`. Even the same distinct-shape *cardinality*, 9 vs 9.

The only non-unity ratios in the entire instrument stack are PTX lines 864 vs 671 (1.29x),
`ir-no-opt.ll` 1,220 vs 875 lines (1.39x) and `memory_analysis` `temp_bytes` 144.2 MB vs 43.3 MB
(3.33x) — against a 33.9x compile ratio. Only the shape *values* differ:
`f32[1,128,1055,1055]` vs `f32[1,128,95,95]`, i.e. 142.4 M vs 1.16 M elements = 123x. **And that 123x
points the wrong way** — it says "bigger program", while the fast arm does more runtime work (control
3.44 ms, case 450 ms), which the case file explicitly warns about.

**Then the one instrument holding the answer inverts too.** `pass_timings` totals 0.405 s for the
case and 2.445 s for the control: the pass profile says the *control* spent 6x more time in passes,
and `autotuner` is absent from the case's 188-pass list while present in the control's 189 at
2.13 s. The pass-name set difference is exactly `{'autotuner'}`, present only in the fast arm.

The raw log says the opposite:
`HLO pass: autotuner time: 1.19 min (71651421 us) (cumulative: 1.2 min, max: 1.19 min, #called: 617)`
against `WALL_COMPILE 72.505 s` — **the autotuner is 98.8% of the compile.** Exactly one of the 640
pass lines in that log uses `min` units, and it is that one. `_UNIT` knows only us/ms/s, so
`_PASS_LINE` drops it and `pass_timings` returns a plausible dict topped by
`remat-pipeline: 0.1196`.

**Where scopex earned its keep.** Nowhere, on this arm. Its contribution was negative: it reported
the opposite of the truth with no warning. The honest positive is that the exhaustive null is
*correct* — the cost is not in transforming IR, it is in running candidate cuDNN kernels on real
buffers at compile time, so no artifact of the program can differ, and I verified that down to PTX
line counts and thunk counts.

**Where I left scopex.** At the first knob, and I should have gone straight to
`TF_CPP_VMODULE=hlo_pass_pipeline=1` and grepped for `autotuner`.

**How far I got.** Root cause named, but by the raw log, not by scopex. Fix the twelve-character
regex and this becomes the single cleanest hit in the corpus: one call, one pass name, 98.8% of the
wall clock.

### D2. `gemm_shapes_k64` — `xla#35955`, GPU

**The symptom.** K distinct GEMM shapes vs K repetitions of one shape. Backend 21.83 / 40.34 /
176.26 s at K=8/16/64 against a control at 5.58 / 6.48 / 10.30 s = 3.9x / 6.2x / 17.1x — linear in K
at about 2.7 s per distinct shape, while the control stays flat.

**The trail.** Every count-based attributor gets the *sign* wrong, at every rung. Jaxpr equations
identical (24/48/192). StableHLO lines identical (65/105/345). Optimized HLO instructions 127 vs 152,
246 vs 256, 1,048 vs 1,216 — **the slow arm has fewer instructions and fewer fusions** (151 vs 193 at
K=64).

The surviving variable is shape-cache-key cardinality, and it is only reachable by bypassing
`walk_hlo` entirely: `hlo_instructions(hlo_text(compiled))` then
`len({r['shape'] for r in recs if r['opcode'] in ('dot','fusion')})` = **9 vs 2** at K=8 (17 vs 4
distinct shapes overall). The dump corroborates with the autotuner's own candidate-compile count:
`n_xla_modules` 96 vs 21 (4.57x), PTX files 18 vs 4, `priority_fusion_dump` files 72 vs 17.

The chain is fully measurable: distinct fusion shapes (9 vs 2) -> autotuner candidate sub-modules
(96 vs 21) -> autotuner pass time (46.20 s vs 6.53 s at K=16) -> compile time.

**But at the assigned K=64 the pass instrument dies.** Raw vmodule:
`HLO pass: autotuner time: 2.1 min (125924695 us) (cumulative: 2.19 min, max: 2.1 min, #called:
31601)` against `WALL_COMPILE 129.53 s` — 97.2% of the compile, the only `min` line among 31,624
pass lines, dropped. K=16 (46.2 s, under 60 s) survives; K=64 (125.9 s, over it) does not. Again: the
smaller rungs of a deliberately-constructed ladder look fine and the largest silently loses its
answer.

**Where scopex earned its keep.** `hlo_instructions` gave me the shape strings. That is the
lower-level accessor, not the level API.

**Where I left scopex.** `crosstab(rows='kind', cols='shape')` is suggested by the brief as a
reachable idea and it is **not reachable**: `crosstab` works on `Ins`, `Ins` has no shape, and
`levels._to_ins` discards the shape that `hlo_instructions` already parsed. For an autotuning family
whose cost is literally the cardinality of a shape-keyed memo table, that is the one field that
matters.

**How far I got.** Root cause named and the causal chain measured end to end at K=8 and K=16. At
K=64 the localisation rests on a raw log line that scopex drops.

### D3. `xtile_issue` — `xla#41173`, GPU only (the CPU arm is fixed by `kMaxRank=8`, PR #41174)

**The symptom.** Backend 3.579 s vs 0.175 s (20.5x) with **29 vs 30 jaxpr equations** — the control
has *more* operations.

**The trail.** `pass_timings` on both arms: `autotuner` = 7.140 s vs 0.0048 s = 1,487x, on an
identical 189-pass sequence. Note this arm is under the 60 s threshold, so the regex bug does not
fire and the tool works.

Counts under-explain badly (optimized HLO 307 vs 143 = 2.15x for 20.5x time), but the control
variable *is* legible at the HLO level once you look at shapes instead of counts: the case's fusion
shapes include `f32[2,2,2,2,2,2,2,2,2,2,2,2]` — rank 12 — where every control shape is rank <= 2, and
distinct-shape cardinality is 30 vs 4 overall, 13 vs 1 for `dot`, 21 vs 1 for `fusion`.

The dump corroborates and then explains: 104 XLA sub-modules vs 2 (52x), 42 PTX kernels vs 3, 63
`priority_fusion_dump` files vs 1, `ir-no-opt.ll` 6,706 vs 106 lines (63x). **And XLA states the
reason in its own words:** 85 `producer_ineligible` vetoes, all
`not fusing because there are only bitcast users`, against 1 in the control.

A synthetic peak-rank sweep confirms rank is the axis and that counts are anti-correlated with it:
peak rank 8/10/12/14 -> backend 2.876/4.937/6.150/7.515 s (monotone), while optimized HLO
instructions go 139/99/127/155 (the rank-8 control has the second-most) and XLA sub-modules go
32/64/96/132 (monotone with rank). `memory_analysis` `temp_bytes` tracks the control variable
exactly: 1,536/6,144/24,576/98,304 bytes = 4x per +2 rank = 2^rank — the single cleanest
scopex-reachable scalar for this family.

**Where scopex earned its keep.** `pass_timings` named the pass in one call, and
`memory_analysis().temp_size_in_bytes` (reached through the compiled object scopex hands you) tracks
2^rank exactly.

**Where I left scopex.** The 104-vs-2 sub-module count is the loudest number in the whole comparison
and it exists only as a filename-prefix count I derived with a regex over `os.listdir`. The
`priority_fusion_dump` protobuf text — XLA's own explanation, 85 identical veto strings — is handed
to me as a directory and nothing else. And the case module ships its own `max_intermediate_rank()`
helper *because scopex has no rank view*, while at the HLO level rank is sitting in a shape string
that `_to_ins` discards.

**How far I got.** Mechanism localised and the issue's reading refined on this backend: rank-12
transposes and dots acquire bitcast-only users, priority fusion refuses them 85 times, the module
keeps 60 unfused fusions, and the autotuner then compiles 103 candidate sub-modules. The cost lands
in the autotuner, not in symbolic tiling per se.

## E. The arm that never finishes compiling

### E1. `topk_pow20p1024_k128` / `_p2048_k128` — `jax#19653`, GPU

**The symptom.** `jax.lax.top_k(x, 128)` on n = 2^20 + 1024: the compile never returns. RSS grows
perfectly linearly at 0.168 GB/s from 0.57 GB with no plateau, blowing a 9 GB cap at 51 s
(deterministic: 55.8 s and 59.5 s on two runs). The `+2048` arm is indistinguishable: 0.642 ->
9.016 GB at 0.153 GB/s, killed at 55.5 s. Three healthy alignment arms compile in 0.309-0.375 s at
0.78 GB peak with 11 HLO instructions and `custom_call_target = 'xla.gpu.ext.cub_sort_pairs'`.

**The trail.** Half the toolkit is unavailable *by construction*: no `Compiled` object ever exists,
so no `walk_hlo`, no `hlo_text`, no `memory_analysis`, no optimized-HLO census. And `record()` is
worse than unavailable — it has no memory cap and no timeout, so calling it here grows the caller's
RSS without bound until the OOM killer arrives and **takes the calling process down**. `Timings` has
no memory field at all, so even a survivable run reports nothing about the resource actually being
consumed. `pass_timings`' subprocess dies too and returns `{}` with a `stderr_tail`, which is honest
but not a localisation.

Only `dump()` survives, and only because XLA writes its files incrementally as passes complete —
that is luck, not design. Under an external 9 GB RSS watchdog, `dump(passes='.*')` writes exactly 13
files and stops, the last being
`module_0000.jit_top_k.0007.optimization.after_pipeline-start.before_ragged_dot_rewriter.txt`. The
vmodule log stops after exactly 19 `HLO pass:` lines (the control logs 303), the 19th being
`windowed-einsum-handler`. The control's pass order shows **pass #20 is `topk-splitter`.** That names
the pass that never returns.

**Then I sharpened the case file's "alignment" hypothesis into a predicate and tested it.** The gate
is `n % 1024 == 0` **and** `n / 1024` **not** a power of two. n=2^19 (q=512), 2^20 (q=1024), 2^21
(q=2048) are all fine; 2^20+1 and 2^20+512 (non-integer q) are fine; 2^20+1024 (q=1025), +2048
(q=1026), +3072 (q=1027), +7168 (q=1031) and 1536*1024 (q=1536) all blow up. 10/10 consistent. This
explains what alignment alone cannot: **why 2^20 itself, the length the issue was filed about, is
fast on this build.** Reading: `TopKSplitter` fires only on an exact 1024 batch split, and the
emitter behind it needs a power-of-two row count.

**Where scopex earned its keep.** `dump()` plus the last-written snapshot, which localised a
non-terminating compile to a pass. Counterfactual: nothing else in the stack survives a compile that
does not finish.

**Where I left scopex.** For the RSS watchdog (not a scopex API, and the reason the machine survived
at all), for sorting dump filenames by sequence number, and for the pass-order diff. The two arms
differ by 1,024 elements out of 1.05 M and are indistinguishable in every scopex-visible respect —
single-equation jaxpr, same primitive, same k, same dtype — which is exactly why nothing structural
can separate the sick arms from the healthy ones. Only which emitter branch was taken, observable
solely as *the pass the compile dies in*.

**How far I got.** Narrowed sharply but not to a root cause. The pass is named, the trigger predicate
is 10/10 predictive and strictly stronger than the issue's own, and the resource is identified as
unbounded host memory. Why `TopKSplitter`'s downstream emitter allocates without bound was not read
out of XLA source.

---

# Part 2 — across the cases

## What the stage split alone decides

One compile, no flags, no subprocess. `record` matched jax's metrics on every arm — `Timings.matched`
never went False — and named the correct dominant stage on 27 of 28 arms. The exception is `topk`,
where calling it at all was fatal.

That is the good news, and it is genuinely large: **the first command routes you into one of five
buckets, and the buckets want different tools.** Trace-bound sends you to a Python profiler and
nothing else in scopex will help (einsum, condrec, jitfib's original framing). Backend-bound sends
you to `pass_timings` and `dump`. Lower-bound and mixed mean the jaxpr is the artifact to count.

Now the honest half.

**The label discriminated between case and control on only 6 of 28 arms.** For 22 arms `record`
returns `backend-bound` for the case *and* for the control, which is correct and useless — the split
does not differ in kind. On `convT` and on all three gather arms, both labels are `backend-bound`
while the times differ 33x-522x. `regime()`'s own docstring says the split that matters most inside
the backend is pass-bound vs autotune-bound, and declines to make it.

**Four ways the split actively misled:**

- `arity_tree_200`: `regime` returns `mixed` while trace is 124x its control, because the 0.6
  threshold is applied to a wall that includes the 14.3 s backend. A user reading `mixed` with a 14 s
  backend goes and tunes XLA and loses the larger half of the wall.
- `ndtri_scan`: the case file predicted the cost would be in `lower_s`; backend is 91% of a 112 s
  wall. Only the split settles it — so this is a *win* for `record`, but it means a maintainer's
  prior about the stage is worth nothing.
- `jitfib`: filed as a tracing pathology in 2024; on 0.10.2 trace is flat at 0.03 s in both arms and
  the whole 18.9 s is backend. Same lesson, opposite direction.
- `einsum`: trace time is first-trace-only. `record` calls `jax.clear_caches()`, which does *not*
  clear opt_einsum's own path cache, so a repeat measurement of the same function reads 0.002 s for a
  49 s search. Any tool that traces twice, or that measures the second trace, reports zero.

**The derived number nobody computes.** On the arms where the backend dominated,
`sum(pass_timings(src)['passes'].values()) / record(fn,*args)['backend']` was the single most
informative statistic in the stack, and it is a two-call division nobody suggests. Measured coverage:
0.066% (gather2d_8, n=8), 0.02% (gatherchain2d_9, n=9), 2.8% (argsort_f32_1e7), 3.3-5%
(stackcond_n3000), 18% (ndtri_d16 case) vs 76% (its control), 44% (bisect_m94, correctly parsed),
95% (adconst_2p20), 128% (switch_ident_512, i.e. impossible), 0.5% (convT, i.e. a lie). **Low
coverage routes you below HLO; high coverage names a pass; over 100% means the parse is broken.** All
three of those readings were load-bearing in this corpus.

## Which observations recurred

Stated as empirical regularities over 28 arms, not as a procedure.

**1. Counts are flat, null or inverted far more often than they are informative — and inversion is
common enough to be expected, not surprising.** Optimized-HLO instruction count was a null (within
1.1x) or *pointed the wrong way* on 11 arms: `stackcond` (30 vs 31, inverted), `adconst` (2 vs 24,
inverted), `dusfold` (2 vs 2, identical), `jitfib` (2 vs 2), all three gather arms (1.04x),
`gemm_shapes` (fewer instructions in the slow arm at every K), `convT` (23 vs 23, identical),
`xtile` (2.15x for 20.5x), `switch_ident` (exactly linear against n^2.8 time). `len(str(jaxpr))` was
3.7x *smaller* in the pathological arm on `jitfib`. Any tool that ranks programs by IR size scores
the sick arm as the healthy one on at least five of these.

**2. The pathology parameter sweep is what separates a signal from a coincidence, and it is cheap.**
Every finding in this document that survived was checked at two or three sizes, and several
conclusions exist *only* because of the sweep: `bisect`'s cliff is at `m // unroll == 1`, not at
m=95 (two-axis sweep); `topk`'s gate is `n/1024` not a power of two (10 sizes); the gather family's
base of the exponential is index rank (rank ladder at fixed depth); `arity_tree` is linear, not
quadratic, refuting the issue thread. Conversely the sweep is what demotes signals: the gather
family's 1.43x jaxpr ratio does *not* grow with ncycles, which is exactly why it is not the answer.

**3. Ratios systematically under-report when the cost is superlinear in absolute size.** `ndtri`:
structural ratios are flat from d4 to d16 (jaxpr 2.28 -> 2.29, HLO 5.98 -> 6.11) while the time ratio
goes 4.8x -> 30.1x. `stackcond`: every visible signal is linear in N while compile grows as N^1.8.
`switch_ident`: counts exactly 2.00x per doubling, time 4.2x-8.1x. The absolute counts, not the
ratios, predict the time.

**4. A pass name present in one arm and absent in the other is a stronger signal than any timing.**
XLA writes a per-pass dump snapshot only when a pass *changed* the module, and logs a pass line only
when it ran. So: `flatten-call-graph` exists in `jitfib`'s snapshot list and not the control's;
`estimate-cub-sort-scratch-size` exists in `argsort`'s control and not the case; `autotuner` exists
in `convT`'s *control* and not the case; `topk-splitter` is pass #20 in the healthy arm and the pass
the sick arm dies in. Four unrelated mechanisms, one observation, and it is a set difference over
keys that `pass_timings` already returns and nobody diffs.

**5. The per-pass snapshot curve from `dump(passes='.*')` localises when it localises at all, and
the divergence index is the whole answer.** `bisect`: 1.76x at snapshot 31, 69x at snapshot 32,
which is `fusion`. `jitfib`: 57 -> 27,058 instructions in `flatten-call-graph`. `ndtri`: 3.69x ->
6.08x across `fusion`. `switch_ident`, `arity_tree`, `dusfold`, gather: a *flat* curve, which is
equally decisive because it says no pass is responsible. I hand-wrote this parser at least six times.

**6. When counts fail, the metric that works is bytes, shapes or arity.** `dusfold`: instruction
count shrinks while module bytes peak at 320 MB vs 64 MB. `stackcond`: 6 vs 6 equations, max operand
arity 3,000 vs 2. `xtile`: rank 12 vs rank 2, `temp_bytes` = 2^rank. `gemm_shapes`: distinct fusion
shapes 9 vs 2. `convT`: identical everything, shape values 123x apart. `bisect`: max operand count
289 vs 55. **Six unrelated pathologies whose control variable is a shape, a byte count or a fan-in,
and none of the three is a field on `Ins` or a name in `BY`.**

**7. Below HLO, the ratio between artifact size and compile time is the tell.** gather: 278 ms per
dumped LLVM IR line vs 0.86 ms/line = 323x, on IR that is 1.12x. `argsort`: 451x PTX lines against
3.67x HLO instructions. `stackcond`: 85x `.ll` lines, 22x object bytes, 60x-509x time.
`ndtri`: 344 kernel modules vs 39. `adconst`: zero `.ll` files, zero object bytes — a zero-kernel
executable is a one-line fingerprint for compile-time constant folding.

**8. XLA volunteers ground truth in prose and everyone throws it away.** `slow_operation_alarm.cc:73]
Constant folding an instruction is taking > 1s:` / `The operation took 17.590259602s` (adconst,
dusfold). `Very slow compile?` (bisect m=94 only). `fusion_emitter.cc:302] Tiled emitter failed due
to tiling failure: ... falling back to loop emitter` (gather, both arms — the reason the slow emitter
runs at all). `not fusing because there are only bitcast users`, 85 times (xtile). The corpus harness
greps for exactly one of these strings; scopex surfaces none of them, and `pass_timings` populates
`stderr_tail` only when the parse *fails*, i.e. never when you have a real profile.

## The dead ends worth knowing

Named, because they cost time to rediscover.

**`walk_stablehlo` is dead on jax 0.10.2 and fails silently.** It returned exactly **1 unit on 21 of
21 arms measured**, including modules of 3,245, 10,713, 21,242, 31,507, 42,442 and 84,845 StableHLO
lines. `attribute(list(walk_stablehlo(low)),'kind')` returns `{'func': 1}`. Cause: the pattern
`^\s*(?:%\S+\s*=\s*)?"?(?P<op>[\w.]+)"?.*?loc\("(?P<loc>[^"]*)"` requires an *inline quoted*
location, but jax 0.10.2 emits location **aliases** — `#loc11 = loc("...")` declared at the top of
the module and referenced as `loc(#loc11)` on each op. On `stackcond_n3000`, 3,214 of 3,245 lines use
the alias form and 23 use the inline form (argument names only), so only the `func.func` signature
line matches. Reproduced on a four-line program
(`jax.jit(lambda x: jnp.tanh(x).sum()).lower(jnp.ones(8))` -> 1 unit where 3 ops exist), so it is not
a small-module artefact. The failure mode is the one `TRAPS` was written to prevent: it returns a
non-empty length-1 iterator, so it passes an `if not units` guard and any `attribute`/`crosstab`
built on it reports a one-op program with total confidence. Concretely costly twice: StableHLO is the
*only* level where `stackcond`'s 3,000 reshape operations and `arity_tree`'s `2*nleaves` parameters
are individually present without a dump and without a compile, and it is the level that returns 1.
Both cases had to fall back to a raw text line count.

**`pass_timings` on any pass slower than ~60 seconds.** XLA prints durations with
`absl::FormatDuration`, which switches to minutes above 60 s. `_PASS_LINE` accepts only `us|ms|s`, so
the slowest pass in the program is dropped **silently** and the tool returns a plausible profile
naming a millisecond-scale pass. Confirmed on four arms: `bisect_m95` (`fusion time: 2.87 min`
dropped; reported culprit `copy-insertion: 13.0 s`, wrong by 13x), `bisect_m94`
(`fusion time: 2.06 min (123480735 us)`), `switch_ident_1024` (`copy-insertion time: 2.95 min
(177154325 us)`, 94% of compile), `convT64_dilate16` (`autotuner time: 1.19 min (71651421 us)`, 98.8%
of compile), `gemm_shapes_k64` (`autotuner time: 2.1 min (125924695 us)`, 97.2%). In every case
exactly *one* line in the log used minutes and it was the only line that mattered. The parenthesised
always-microseconds value on the same line would have parsed.

**`pass_timings`' total, independently of the above.** XLA logs pipelines and their member passes in
the same format (`HLO pass: simplification time: 3.52 s` is the *pipeline* containing
`constant_folding`), so the total double-counts. Measured impossibilities: 10.54 s of pass time for a
5.4 s compile (`adconst_2p18`), 11.581 s for a 10.856 s backend (`jitfib_t22`), 12.18 s for a 4.53 s
backend (`dusfold`), 41.00 s for a 31.9 s backend (`switch_ident_512`). The `#called:` field and the
pipeline names visible in dump filenames are enough to separate them.

**`hlo_instructions` silently drops tuple-shaped instructions.** `_INSTR`'s shape group is `\S+` and
a tuple shape contains spaces, so `sort`, `while`, `conditional`, every multi-output fusion and every
tuple-returning custom-call vanish. On `argsort` the dropped line *was the answer*
(`xla.gpu.ext.cub_sort_pairs`), and the drop is invisible: 12 parsed instructions against 13
assignment lines.

**Operand arity and shape are unrepresentable.** `Ins.__slots__` is
`(level, kind, path, unit, container, site, loc, function, depth, fusion, outlined)`. No shape, no
rank, no operand list, no operand count — and `levels._to_ins` throws away the `shape` that
`hlo_instructions` already parsed. `BY` has 16 names and none of them is `arity`, `shape`, `rank` or
`nbytes`. The brief suggests "opcode x shape crosstabs at hlo_opt" as a reachable idea; it is not
reachable, and six of the arms above have a shape, a byte count or a fan-in as their control
variable.

**Marking-contract views on this corpus.** `author`, `innermost_author`, `library`, `package`,
`role`, `split` returned `<none>`/`<unmarked>` for every unit on every arm — the cases are unmarked
single-file programs. Only `site` and `transform` ever carried information, and `site` loses 16% of
the units to `<no-frame>` on `condrec_grad_512` (16,380 of 102,398 equations, the single largest
bucket).

**`lax.optimization_barrier` as a fusion blocker.** Made the gather case 3x *worse*, and the dump
shows still one kernel with 93% of the time in emission. XLA erases `optimization_barrier` before
fusion; scopex's own `TRAPS['barrier_erased']` documents exactly this. The arm was never a valid
ablation.

**`memory_analysis().temp_size_in_bytes`.** Tried on eight arms; informative on exactly one
(`xtile`, where it tracks 2^rank perfectly). Elsewhere it is 0 vs 1 MB (`adconst`, i.e. inverted) or
mildly correlated noise.

**Dump size, unwarned.** `dump(passes='.*')` on `bisect_m95` produced **944 MB across 950 files and
took 358 s**, from a 24-equation jaxpr. `ndtri_d16` produced 1,089 files. `dump()` reports nothing
about what it is about to write. Also: `dump_flags()` emits `--xla_dump_hlo_pass_re` twice when
called with `fusion=True` and `passes='.*'`; XLA's last-flag-wins parse makes it harmless today, but
it is a silent flag collision.

## What scopex should add, ordered by how many investigations it would have shortened

Each item names the artifact it would read.

**1. Fix `_PASS_LINE` / `_UNIT`, and return coverage. (6 arms; converts one no-signal to a clear
hit.)** Accept `min` and `h`, or better, parse the always-microseconds parenthesised value on the
same line. Separate pipeline entries from pass entries using the `#called:` field and the pipeline
names already visible in dump filenames. Return
`{'passes':…, 'pipelines':…, 'unparsed_pass_lines':[str], 'wall_compile_s':float, 'coverage':
total/wall}`. Artifact: the `HLO pass: NAME time: 2.06 min (123480735 us) (cumulative: …, #called:
N)` lines in a `TF_CPP_VMODULE=hlo_pass_pipeline=1` stderr stream. Grounding: `convT64_dilate16`
(no-signal -> 98.8% of compile in one call), `switch_ident_1024`, `gemm_shapes_k64`, `bisect_m94`,
`bisect_m95`, plus the double-count on `adconst`, `jitfib`, `dusfold`. A coverage of 0.006 would have
screamed; instead the tool returned a normal-looking dict.

**2. `scopex.pass_growth(dump_dir, metric='instrs'|'bytes'|'computations'|'max_operands')` and
`pass_growth_diff(dir_a, dir_b)`. (7+ arms; hand-written at least six times.)** Return
`[(idx, pipeline, after_pass, before_pass, n_instrs, n_computations, sum_bytes, max_operands)]` plus
a `.diverges_at(other)` helper that aligns on `(pipeline, pass)` and flags sequence divergence. Note
in the docstring that XLA snapshots only passes that *changed* the module, so a name present in one
arm and absent in the other is itself the finding. Artifact:
`module_NNNN.<name>.NNNN.<pipeline>.after_X.before_Y.txt`. Grounding: `bisect` (3,605 -> 382,246 at
`fusion` is the entire result), `jitfib` (57 -> 27,058 at `flatten-call-graph`), `ndtri` (3.69x ->
6.08x across `fusion`), `dusfold` (**the `bytes` metric is not optional** — the guilty pass shrinks
the count while materialising 320 MB), `switch_ident`, `arity_tree`, gather (flat curve, decisive).

**3. Shape, rank and operand arity as first-class fields and views. (6 arms.)** Add `operands` (list)
and `n_operands` to the dicts `hlo_instructions` yields — the parser already has to find the `(` to
match the line, so it is a paren-balance walk over text in hand. Add `shape`, `rank`, `n_operands` to
`Ins.__slots__` and populate them in `_to_ins`. Register `BY['arity']`, `BY['shape']`, `BY['rank']`,
`BY['nbytes']`, and ship `scopex.shape_cardinality(units) -> {opcode: n_distinct_shapes}`. Have
`record`/`table` print `widest instruction: <primitive> with N operands` unprompted. Artifact: the
shape token and operand list on every HLO instruction line, already parsed and then discarded.
Grounding: `stackcond` (3,000 vs 2 is the whole answer and the API has no slot for it),
`gemm_shapes` (9 vs 2 distinct fusion shapes), `xtile` (rank 12), `convT` (only shapes differ),
`bisect` (289 vs 55), gather (concatenate fan-in).

**4. `scopex.codegen_summary(dump_dir)`. (7 arms.)** Return
`{'llvm_modules', 'll_lines_noopt', 'll_lines_opt', 'obj_bytes', 'ptx_kernels', 'ptx_lines',
'thunk_kernels', 'xla_submodules', 'per_kernel': {...}}`. Artifacts: `*.ir-no-opt.ll`,
`*.ir-with-opt.ll`, `*.ptx`, `*.o`, `thunk_sequence.txt`, and the count of distinct `module_NNNN`
filename prefixes. Grounding: `argsort` (451x PTX lines against 3.67x HLO — the whole localisation),
`ndtri` (344 vs 39 kernel modules, the CPU cost driver), `xtile` (104 vs 2 sub-modules, the loudest
number in the comparison), `stackcond` (one 70,054-line kernel), `adconst` (**zero** `.ll` files —
the fingerprint), `switch_ident` (proves the LLVM hypothesis false), gather (proves the IR is
identical while time is 522x).

**5. `scopex.dump_timeline(dump_dir)`. (5 arms; free — the timestamps are already on disk.)** Return
`[(phase, t_offset_s, artifact_path)]` with named phases
`{hlo_passes, buffer_assignment, ir_emission, llvm_opt, codegen}` derived from artifact mtimes, plus
a per-kernel breakdown. Grounding: this **is** the winning knob for `gatherchain2d_10` — IR emission
29.957 s of a 30.203 s compile at ncycles=8, against a control flat at 0.075 s — and it turns
`jax#32704` from "no signal anywhere" into a one-line localisation. Also `stackcond` (codegen 8.762 s
= 78%), `jitfib` (call-inliner 7.42 s), `bisect`, `ndtri`, `arity_tree`.

**6. Fix `walk_stablehlo`'s location-alias parsing. (all 21+ arms; the level is currently dead.)**
Pre-scan `^#(loc\d+) = loc\((.*)\)$` into a table, match operation lines on the
`<dialect>.<op>` token rather than on the presence of an inline `loc`, and resolve `loc(#locN)`
recursively through the table (they nest: `#loc19 = loc(callsite(#loc5 at #loc12))`), falling back to
`<no-loc>` instead of dropping the unit. Add a `Timings.matched`-style guard: warn when the number of
yielded units is far below the count of `= stablehlo.` lines in `stablehlo_text`. A regression test
asserting `>= 3` ops on `jnp.tanh(x).sum()` would have caught it.

**7. `scopex.trace_profile(fn, *args)`. (4 arms.)** A `cProfile` or sampling profiler running only
for the duration of the trace stage, returning `Counter[(file, line, function)]` or
`[(frame, cumulative_s)]`. Grounding: `einsum` (would name `opt_einsum/paths.py` and end the
investigation in one call), `condrec` (23.7x trace), `arity_tree` (124x trace), `jitfib`'s original
framing. This is the biggest hole for trace-bound work: scopex can time the stage and cannot
attribute inside it. `regime()` should also name the dominant third-party package when it says
`trace-bound`, otherwise the label is a dead end. And `record` should warn that trace time is
first-trace-only — `jax.clear_caches()` does not clear opt_einsum's path cache.

**8. `scopex.alarms(fn, *args)` — and always return alarms from `pass_timings`. (3 arms.)** Harvest
every `slow_operation_alarm.cc` line verbatim from the vmodule subprocess. There are at least three
distinct alarm strings in this suite and the corpus harness greps for one. Grounding: `adconst` and
`dusfold` ship a verbatim compiler-authored diagnosis that is currently discarded *at the moment of
success*, because `stderr_tail` is populated only on parse failure.

**9. `scopex.walk_hlo(x, stage='pre')` / `pre_optimization_hlo(lowered)`. (4 arms.)** The
pre-optimization module is a first-class level and scopex has no accessor for it. Artifact:
`module_NNNN.<name>.before_optimizations.txt`. Grounding: `stackcond` (188x-626x pre-opt against
0.97x optimized), `arity_tree` (`2*nleaves` parameters), `dusfold` and `adconst` (the literal exists
only there — "your module contains a 216 MB literal" is the whole diagnosis in one number).

**10. `record(fn, *args, rss_cap_gb=…, sample_hz=…)` with `peak_rss_gb`, `rss_trajectory`, `killed`,
`stage_at_kill`. (3 arms.)** Compile-time host memory is the resource for a whole family and scopex
cannot see it at all; on `topk` `record` is not merely blind, it takes the caller down. Also
`scopex.dump_progress(dump_dir) -> {'last_snapshot': (seq, pipeline, before_pass),
'passes_completed': n}` so a killed compile localises itself. Grounding: `topk` (both arms, the
entire job), `dusfold` (0.955 GB excess tracks n^3), `adconst`.

**11. Fix `hlo_instructions`' tuple-shape drop. (1 arm, but it was the answer, and it silently loses
every `sort`/`while`/`conditional`/multi-output fusion in the corpus.)** Make the shape group accept
a balanced-paren tuple. Grounding: `argsort` — the `cub_sort_pairs` custom-call.

**12. `scopex.pass_set_diff(src_a, src_b) -> (only_in_a, only_in_b)` and
`scopex.pass_order(src) -> [pass_name]` that works on a compile that never finishes. (4 arms.)**
`pass_timings` already returns the dicts; nobody diffs their keys. Grounding: `argsort`
(`{'estimate-cub-sort-scratch-size'}`), `convT` (`{'autotuner'}`), `jitfib`
(`{'flatten-call-graph'}`), `topk` (19 passes vs 303, divergence at index 19, names
`topk-splitter` — and `pass_timings` cannot produce it because it parses only completed passes and
returns `{}` when the process dies).

**13. `scopex.fusion_decisions(dump_dir)`. (1 arm, decisive there.)** Parse
`module_NNNN.*.priority_fusion_dump.txt` — which is `xla.gpu.FusionProcessDumpProto` text, with
`us_fused`/`us_unfused` per candidate pair and XLA's own veto reasons — into
`{'candidates': n, 'vetoes': Counter(reason), 'timings': [...]}`. Grounding: `xtile`, where 85
identical `not fusing because there are only bitcast users` vetoes against 1 in the control is the
mechanism in the compiler's own words, and scopex hands you the directory and nothing else.

**14. Small, cheap, and each grounded once.** `scopex.jaxpr_sharing(jaxpr) -> (tree_eqns,
distinct_eqns, distinct_subjaxprs, sharing_factor)` — `jitfib`, where the ratio *is* the diagnosis
("your program is a DAG that XLA will expand into a tree") and the DAG count needs an `id()`-keyed
traversal nobody writes by hand. `scopex.cascade(fn, *args) -> {level: n_units}` — `ndtri`, where the
widening ratio down four levels is the finding and today needs four hand-assembled numbers.
`scopex.compare(t_case, t_control) -> {stage: ratio}` — `condrec` and `einsum`, where the per-stage
*ratio* is the whole result. `scopex.ad_census(jaxpr)` — `condrec`, where the 17x jvp/transpose
asymmetry is the actual mechanism and is currently an inference from `attribute(…,'transform')`.
`record` reporting `n_args`/`n_leaves` from `tree_flatten` of the call arguments — `arity_tree`,
where "trace is 124x and you passed 400 arrays instead of 2" is one line record could print
unprompted. `scopex.artifact_digest(dump_dir)` — `einsum`, where md5-equality of the optimized HLO
against the `_pathlit` twin is what *proves* the compiler-side null rather than asserting it.
`dump()` reporting bytes written — `bisect` (944 MB) and `ndtri` (1,089 files).

## Where a maintainer should not bother with scopex at all

**Trace-bound work, past the first command.** `record` tells you it is trace in one call and then
scopex has nothing. Use `cProfile`, `py-spy dump`, or `python -X importtime` as appropriate. On
`einsum` the answer is one frame deep in `opt_einsum.paths._optimal` and no jaxpr-unit attributor
can ever reach it, because the expensive code emits no equations. Measure the *first* trace in a
fresh process; caches inside third-party path solvers survive `jax.clear_caches()`.

**A compile that does not terminate.** Do not call `record` — it has no cap and will take your
process with it. Use an external RSS watchdog plus `XLA_FLAGS=--xla_dump_to=DIR` and read the
highest-numbered snapshot filename, which is what worked on `topk`. `dump()` survives only by luck
(XLA writes incrementally); everything else in scopex needs a `Compiled` object that will never
exist.

**GPU autotuning.** On `convT` every structural level is provably identical down to PTX line counts,
so there is nothing for scopex to compare. Go straight to `TF_CPP_VMODULE=hlo_pass_pipeline=1` and
grep for `autotuner`, or A/B with `--xla_gpu_autotune_level=0`. Until the unit regex is fixed,
scopex's parse of that same log is not merely lossy on these arms, it is inverted.

**When you already suspect a specific pass.** The raw `hlo_pass_pipeline=1` stderr stream is the
ground truth; scopex's parse of it drops the slowest line and double-counts pipelines. Read the log.

**Anything below LLVM IR.** Nothing in scopex counts a `.ll`, a `.ptx`, an object file, a kernel
module or a thunk, and on four arms (`argsort`, `ndtri`, `stackcond`, gather) those counts are the
localisation. Use `--xla_dump_to` and `wc -l`.

**Emitter-stage IR on XLA:CPU.** `dump_flags()` emits only `--xla_dump_to` and
`--xla_dump_hlo_pass_re`, so `dump()` cannot produce `-pre-optimization.mlir` or `mlir-passes.log`.
For the gather family those files hold the entire answer (1,398,109 lines vs 146 at ncycles=8). Set
`--xla_dump_emitter_re='.*'` yourself.

**Kill-switch A/B, which is not a scopex idea and should be.** The fastest route to "which phase owns
the time" on the gather family was six fresh compiles under six XLA flags:
`--xla_disable_all_hlo_passes`, `--xla_llvm_disable_expensive_passes`,
`--xla_backend_optimization_level=0`, `--xla_cpu_opt_preset=FAST_COMPILE`, and
`--xla_cpu_use_fusion_emitters=false`. Only the last one moved (22.60 s -> 0.320 s), and that single
number named the emitter. Five of the six flags cost nothing to try.

**Attribution views on unmarked code.** If nobody in the stack calls `jax.named_scope` in the shape
`scopex.mark` documents, then `author`, `library`, `package`, `role` and `split` are constants. On
this corpus of single-file reproducers they were constants on 28 of 28 arms. `site` and `transform`
are the only views that work unaided.

## Postscript: the two defects that produced confidently wrong answers

Worth separating from the feature gaps, because their failure mode is different in kind. A missing
feature costs you time. These cost you a conclusion.

`_PASS_LINE`'s missing `min` unit made `pass_timings` name `copy-insertion: 13.0 s` on `bisect_m95`
when the truth was `fusion: 172 s`, and made it report a millisecond-scale pass as the top entry on
`switch_ident_1024`, `gemm_shapes_k64` and `convT64_dilate16` when the truth was 94%-98.8% in one
pass. On the ladders (`switch_ident` 64..1024, `gemm_shapes` K=8..64) the tool works on the small
rungs and fails on the large one, which is the worst possible shape for a scaling study.

`walk_stablehlo` returns a length-1 iterator on a 84,845-line module, so it passes an
`if not units` guard and any view built on it reports a one-op program. An empty level is
indistinguishable from a level with nothing to report — the exact failure `Timings.matched` and
`pass_timings`' `n_lines` were invented to prevent, on the one level that has no such guard.

Both are small edits. Both are the difference, on specific arms above, between a diagnosis and a
wrong answer delivered with confidence.

---

# Part 3 — the hardening + prototype round

Everything below was measured *after* Parts 1 and 2 were written, on jax/jaxlib 0.10.2, python 3.12,
CPU unless a backend is named. Part 2's lists are left untouched as the historical record; this part
says which of them still stand.

## A. Routes tried and rejected, with the evidence

These are not "we did not get to it". Each was built, run against a case whose answer is
independently known, and found to give a *plausible wrong answer* — the failure this package exists
to prevent. They are recorded so nobody re-derives them.

### A1. Kill-switch differencing as a time decomposition — NOT VIABLE

Part 2 proposes kill-switch A/B as the route to "which phase owns the time" (it is the last entry
under *Where a maintainer should not bother with scopex at all*). As a **decomposition of seconds**
it is disqualified, and worst exactly where it matters.

A kill switch answers *how much total compile time disappears if this pass never runs*, which
includes every downstream cost the pass **created**. It does not answer *how long did this pass
take*. On `jax#32704` ncycles=8, CPU, baseline backend 12.7 s (three **interleaved** baselines —
12.795 / 12.643 / 12.841 s, so the differences are not drift):

| lever | total | implied by differencing | truth |
|---|---|---|---|
| `--xla_disable_all_hlo_passes=true` | 0.252 s | `hlo_passes` = 12.47 s (97.5%) | 0.023 s |
| `--xla_disable_hlo_passes=fusion` | 0.217 s | the `fusion` pass = 12.53 s | 0.00128 s |

The second row is the surgical lever, the one a reader would trust most, and it overstates the
`fusion` pass by **9,800x**. `fusion` runs in 1.3 ms and produces one `gather_bitcast_fusion` whose
*emission* then costs 12.5 s; deleting the pass deletes the emission and the difference charges all
of it to the pass. So on the one corpus case where the answer is known, this route names
`hlo_passes` at 97.5% when the truth is `emitter` at 99.2%.

Three further defects, each independently disqualifying:

* `--xla_disable_all_hlo_passes=true` **aborts the process** on `jax#2609` ndtri jacrev d4:
  `F hlo_value.h:241] Check failed: values_.size() == 1 (56 vs. 1)`. A fatal `Check`, not an
  exception, so it takes the caller's interpreter down and cannot be run in-process at all.
* Differences go **negative**: `--xla_llvm_disable_expensive_passes=true` measured 11.182 s against a
  10.776 s baseline, i.e. `llvm_opt = -0.406 s`.
* The noise floor swamps every small bucket. Six identical baseline compiles spanned 5.621–6.616 s
  (16.7% spread, 7.5% stdev), and the **first** compile of a session read 10.776 s against a warm
  steady state of 5.967 s — an 80% systematic bias on whichever arm runs first, which is by
  construction the baseline. Three of the four target buckets are under 0.3% of that compile and are
  unresolvable by differencing in principle. (`--xla_backend_optimization_level=0` moved the total by
  −0.005 s, i.e. it is a no-op lever on this case.)

Kill switches remain the right tool for the question they actually answer — *which knob makes this
go away* — which is how `--xla_cpu_use_fusion_emitters=false` (12.7 s → 0.072 s, control 0.084 s)
named the emitter originally. If that ships it must be `scopex.bisect_flags(fn, *args)`, explicitly
labelled a bisection over **causes**, never a decomposition of **time**. One free property worth
relying on: an unknown XLA flag is **fatal** (`F parse_flags_from_env.cc:234] Unknown flag in
XLA_FLAGS`), not a silent no-op, so a lever removed by a future jaxlib crashes loudly.

Superseded by `scopex.backend_split` (§B1), which costs one compile instead of five.

### A2. Per-kernel mtime aggregation for multi-kernel programs — NOT VIABLE

The obvious repair for `backend_split` on programs with many LLVM kernel modules: pair
`.ir-no-opt.ll` / `.ir-with-opt.ll` / `.o` by kernel stem and sum the per-kernel deltas. On the
224-kernel `ndtri` d4 arm, per-kernel llvm-opt intervals sum to 4.400 s and per-kernel codegen to
3.319 s — together **7.72 s, which exceeds the entire 7.40 s backend stage they sit inside**.

Cause: 223 of 223 consecutive kernels begin IR emission before the previous kernel's `.o` is written.
XLA:CPU compiles kernel modules **in parallel across threads**, so wall-clock intervals overlap and
summing them double-counts; they are neither wall seconds nor CPU seconds.

This is why `backend_split` reports one `below_hlo` bucket when `n_kernel_modules > 1` rather than a
per-kernel sum that looks precise and is inflated ~3.2x. The overlap test it keys on is cheap and
decisive: `min(mtime of .ir-with-opt.ll) < max(mtime of .ir-no-opt.ll)` means the phases interleave.
**The coverage guard alone would not have caught this** — ndtri's coverage was 0.919/0.924,
comfortably inside any reasonable band — so the interleave test is load-bearing and not redundant.
Re-measured this round: 223 kernels, `interleaved=True`, `sound=False`, coverage 0.924.

For a per-kernel bill the route is `TF_CPP_VMODULE=cpu_compiler=3,jit_compiler=3`.

### A3. JAX-internal tracing counters — THERE ARE NONE

`jax._src.monitoring` exposes four record functions, and JAX emits exactly three compile-stage
duration events, defined at `jax/_src/dispatch.py:59-61`: `jaxpr_trace_duration`,
`jaxpr_to_mlir_module_duration`, `backend_compile_duration`. Grepping
`record_event_duration_secs`/`record_event` across `jax/_src` finds only those three plus
compilation-cache hit/miss counters. `linear_util.cache` calls `util.register_cache`, but the
registry exists only so `jax.clear_caches()` can walk it (`api.py:2592` → `util.clear_all_caches`);
it exposes `cache_clear`/`cache_info` sizes, not per-trace timings, and nothing is keyed by call
site. There is no finer-grained in-JAX source for trace attribution — hence the CPython-level
profiler in §B2.

### A4. Binary protobuf route for HLO — METHOD EXISTS, SCHEMA DOES NOT

`HloModule.as_serialized_hlo_module_proto()` works (2,087 bytes for a 16-instruction module) and
`from_serialized_hlo_module_proto()` round-trips it. But there is **no schema anywhere**: zero
`*_pb2.py` under site-packages, no `hlo.proto` or `xla_data.proto` on the filesystem, no `protoc`,
`google.protobuf` not importable (not a jax dep), no pip in the venv. Without a schema the wire
format is field **numbers** only — a generic wire walk confirms the data is all there (field 1 =
name, 3 = computations, 17 = stack_frame_index carrying FileNames/FunctionNames/FileLocations/
StackFrames) — but decoding means hardcoding those numbers, checkable against nothing, whose failure
mode on a wrong guess is an empty list. Strictly worse than printed text, which at least names its
fields.

### A5. `HloInstruction.metadata` / `.shape` — NOT AVAILABLE

`HloInstruction`'s complete non-dunder surface on jaxlib 0.10.2 is
`['async_wrapped_root', 'name', 'opcode', 'operands', 'to_string', 'users']`. No `.metadata`, no
`.shape`. Metadata *is* reachable, but only via `to_string()`, which prints
`metadata={op_name="..." stack_frame_id=3}` by default despite the missing attribute.
`HloPrintOptions` exists and has `print_metadata`, but `HloInstruction.to_string()` takes **no
arguments**; only `HloModule.to_string(options)` accepts it. This is why `hlo_shape` and
`hlo_metadata` remain text parsers — see `docs/HARDENING.md`, where the blast radius is now one
known instruction's own string rather than every line of a module.

## B. What was added, and what it is grounded on

### B1. `scopex.backend_split(fn, *args)` — wishlist item 5, and the answer to §A1

Splits `record()['backend']` into `hlo_passes / emitter / llvm_opt / codegen` from dump-artifact
mtimes, in **one** compile. Validated on `jax#32704` ncycles=8 where the answer is independently
known: backend 12.634 s → `hlo_passes` 0.023 (0.2%), **`emitter` 12.536 (99.2%)**, `llvm_opt` 0.028,
`codegen` 0.016, coverage 0.998. It also **scales** with the pathology parameter, which a single
point cannot show — sweeping ncycles 6/7/8/9 the emitter bucket reads 0.773 / 3.110 / 12.870 /
53.352 s (4.02x, 4.14x, 4.15x per added link, matching the 4x/link the issue reports) while the other
three stay flat, and the flattened control's emitter is flat at 0.050–0.055 s across the same sweep.

Re-measured this round on a quieter box: backend 5.823 s, `emitter` **99.12%**, coverage 0.997,
`sound=True`. That agrees with `pass_timeline`'s independent reading of the same case
(`<llvm ir emission>` 99.5%) by a completely different route.

Guards, both of which were added after the prototype reproduced this package's signature bug on its
own first draft: the **interleave** test (§A2) and a **coverage band** of [0.90, 1.10] that *revokes*
`sound` rather than annotating it. The first draft set `sound=True` on `jnp.tanh(x).sum()` at
coverage 0.217 — that program dumps 26 files and **no `.ll` or `.o` at all**, because the optimized
module is a single `kind=kCustom` fusion with `backend_config {"kind":"__ynn_fusion"}`: XLA handed
the whole computation to a library kernel instead of emitting one, so the phases left no timestamped
artifact to measure to. Re-measured this round: coverage 0.196, `sound=False`, two warnings.

Limits, stated in the docstring: the un-spanned head+tail is a fixed ~20–25 ms, so **compiles under
~1 s cannot pass the band** (census of 9 varied small CPU programs: 0/9 sound, coverage 0.20–0.89);
and dumping perturbs what it measures (+5.8% on gather ncycles=8, +36% on ndtri d4 at 729 files), so
the shares are internally consistent but the absolute seconds must not be compared against an
undumped `record()`.

### B2. `scopex.trace_profile(fn, *args)` — wishlist item 7, the biggest hole for trace-bound work

Two implementations, both shipped, with `method` visible in the result. `xplane` uses the python
tracer inside jaxlib (`ProfileOptions.python_tracer_level=1`); `cprofile` is the stdlib cross-check.

On `einsum_optimal_n10` — the arm Part 1 §A1 could not attribute — **94% of trace self-time in
`opt_einsum/paths.py:236 _optimal_iterate`**, a third-party frame containing no JAX code, which is
exactly what the case docstring says attribution must be willing to do. Re-measured this round:
1,727,005 python events, 2.378 s of 2.531 s = 94.0%, with cProfile reading the same frame at 94.6%
of its own run. Two independent instruments, 0.6 percentage points apart.

It also answers *which line of my code*, which the frame table alone does not: `retrace_static_40`
charges **62.0%** to `case_python_retrace_cache_key_storm.py:125 _body` and `jitfib_t20` charges
**44.2%** to `case_trace_time_nested_jit_fib.py:70 fib` — i.e. it attributes tracing that happens
*inside nested jits*, where the stage split has one scalar.

**And it falsifies the stage split.** `jaxpr_trace_duration` is emitted only under
`core.trace_state_clean()` (`jax/_src/pjit.py:528`), so a **top-level `vmap` or `grad` over a jitted
callee** pushes a trace, no enclosing jit event exists, and the metric reads a silent **0.0**.
Measured with a 10-operand `einsum(optimize='optimal')`:

```
jax.jit(inner).lower(*xs)   wall 2.518 s   trace metric 2.511 s
jax.vmap(inner)(*xs)        wall 2.711 s   trace metric 0.001 s   2.651 s MISSING
```

`Timings.matched` stays True (lower and backend are non-zero), so nothing warned, and 98% landed in
`unaccounted` — whose docstring blamed *dispatch, host transfer, or your own code*. It is tracing.
`record()` itself is safe (it always wraps in `jax.jit(fn).lower(...)`); users timing their own
`jax.vmap(jitted)(...)` were not. Now flagged by `Timings.trace_looks_blind`, which points here.

Two hard limits, both in the docstring. **Memory correlates with the thing being measured**: ~950k
events/s and ~0.6 GB RSS per million events, so ~30 s of dense tracing needs ~20 GB — use
`method='cprofile'` above a few seconds. And the xplane tracer records a **basename only** (0 of 527
distinct frame names in one capture contained a `/`); resolving one through `sys.modules` is
ambiguous for 53.7% of names, so frames are classified library-vs-user by a rule whose error is
one-directional — a user file named `core.py` is under-reported as library, and library time is
**never** charged to user code.

### B3. `scopex.jaxpr_sharing(jaxpr)` — wishlist item 14

Reports two notions separately **because they have different fixes**: VALUE duplicates (same
primitive, same params, same operand identities — what CSE collapses) and SHAPE duplicates
(alpha-equivalent sub-jaxprs — what makes XLA emit N computations, and what survives CSE).
Alpha-equivalence is a structural hash memoised on `id()`.

Found the `switch_ident` case exactly: `switch_ident_128` → 130 equations, 128 sub-jaxprs, **127
redundant equations in ONE value group** (`128 x integer_pow`) and **127 redundant sub-jaxprs in ONE
alpha-equivalence class** — i.e. the program is one branch written 128 times. `switch_ident_512` →
511 redundant in one class. `walk` counts the equations and structurally cannot say this.

## C. Which Part 2 dead ends are now closed

Part 2's lists are the historical record and are left as written. Verified this round:

| Part 2 entry | status | evidence measured this round |
|---|---|---|
| `walk_stablehlo` dead, 1 unit on 21/21 arms | **CLOSED** | native `jaxlib.mlir.ir` walk; `examples/marked_framework.py` → **311 units**, 294 named, 226/311 with a resolved `site`, and `library`/`author`/`split` all populated. The legacy regex on the same module still returns 1. |
| `pass_timings` drops any pass over 60 s (`min` unit) | **CLOSED** | unknown units now **raise**; `gather2d_8` → 95 passes, 315 log lines, `unknown_units` empty. A new sibling bug was found by the guard on its first live log: XLA pass names contain **spaces** (`HLO pass: simplification after layout assignment`), and `(?P<name>\S+)` dropped 6 of 384 lines. |
| `pass_timings` total double-counts pipelines | **OPEN** | `simplification` (6.8121 s) is the pipeline containing `constant_folding` (6.8115 s); coverage can read 186%. Only the **ranking** is safe. |
| `hlo_instructions` drops tuple-shaped instructions | **CLOSED** | native enumeration via `xla_client.hlo`; over 2,811 real dump snapshots the old regex undercounted on 895 (31.8%) and missed 1,208 instructions, never overcounted. |
| Operand arity and shape unrepresentable in `Ins` | **OPEN** | still true; `examples/recipes/widest_instruction.py` and `shape_cardinality.py` go around it via `walk`/`hlo_instructions` directly. |
| Below-LLVM-IR artifacts uncounted | **CLOSED** | `codegen_size`, `emitter_*`, and `backend_split` (§B1). |
| Emitter-stage MLIR unreachable from `dump_flags` | **CLOSED** | `dump(emitter=True)` and `scopex.emitter_growth`. |
| Trace-stage attribution impossible | **CLOSED** | §B2. |
| `dump()` reports nothing about what it writes | **PARTLY** | recipes report dump size; `dump()` itself still does not. |
| Marking-contract views constant on this corpus | **UNCHANGED, and correct** | the corpus is unmarked single-file reproducers. `examples/marked_framework.py` is where those views are exercised. |
| XLA's own `slow_operation_alarm` output discarded | **OPEN** | wishlist item 8. XLA printed *"Constant folding an instruction is taking > 1s"* verbatim on stderr during the `adconst` arm and nothing in scopex surfaces it. |
