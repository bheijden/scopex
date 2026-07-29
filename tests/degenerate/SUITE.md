# The degenerate-compile suite

62 case files, **800 `CASES` entries**, every one a `(fn, args, note)` triple that the harness in
`_harness.py` measures in a fresh subprocess. The suite exists so `scopex` can be hammered against
compile pathologies whose *mechanisms* differ, and so a signal that only works on one stage is
caught being useless on the other fifteen.

Run everything on one backend:

```bash
cd /home/bas/scopex/tests/degenerate
JAX_PLATFORMS=cpu python _harness.py --platform cpu            # all 800, serial, ~hours
JAX_PLATFORMS=cpu python _harness.py --platform cpu <names...>  # a subset
```

Environment these numbers were taken on: jax 0.10.2 / jaxlib 0.10.2, python 3.12, x64 enabled by
the harness child, 20-core CPU, one CUDA GPU that was **off-limits** for the whole of this work
(a separate investigation owned it), so every GPU column below is *pending*, never *negative*.

---

## Suite integrity (checked mechanically, this session)

| Check | Result |
| --- | --- |
| files that import and expose `CASES` | **62 / 62** (0 import failures) |
| total discovered cases | **800** |
| duplicate case names across files | **0** |
| orphan `*_control` entries (control with no case) | **0** |
| malformed `CASES` values (not `(callable, tuple, str)`) | **0** |
| `jax.Array` at module scope (recursive scan of module globals and of every `args` tuple) | **0** |
| jax activity during `import` (monitoring listener: compiles, lowerings, device_puts) | **0 events, 0 backends initialised, 9.3 s total import for all 62 files** |
| cases with an exactly-suffixed `<name>_control` | **736 / 800** |
| cases deliberately shipped without a paired control | **64** (see below) |

One cross-file hazard was checked and is **safe as long as the harness keeps its current shape**:
three files write `jax.config.update("jax_num_cpu_devices", ...)` at module scope
(`spmd_reshard_permute` -> 64, `spmd_uneven_shard_cliff` -> 8, `spmd_halo_conv_dim_choice` -> 8).
`discover()` imports all 62 files into the *parent*, so the last writer wins there — but the parent
never compiles, and `_CHILD` imports exactly one case file per measurement, so every child sees its
own file's value. A config write is not a device claim: the probe above confirms zero backends are
initialised at import. `remat_threshold_activation_chain` applies its config *inside* the traced
function for the same reason, and says so.

The 64 unpaired entries are **not** missing controls; each is a variant/contrast arm whose file
documents why it is unpaired, and each lives in a file that also ships properly-suffixed pairs.
Three patterns:

* **size-sweep arms below the harness floor** — `switch_ident_{64,128}` (the file says outright that
  controls exist only where the arm can clear `MIN_COMPILE_S`, since `classify()` checks the floor
  before it ever consults a control), `bisect_m{64..256}` (the cliff *is* the sweep; the paired
  control `bisect_m95_control` is m=96 and is measured once to serve both roles), `topk_pow{19,21,22}`.
* **contrast arms that are themselves the cheap twin** — `ndtri_scan_jacfwd_*`, `seqgrad_N*_scan`,
  `fusion_rolled_n*`, `einsum_optimal_*_pathlit`, `f8chain_{f32,f16,bf16}_*`, `i4chain_i32_d64`,
  `argsort_{u32,f64,bitcast_i32}_*`, `linalg_chol_d32`, `rngdist_{t,poisson,cauchy}_*`,
  `dynwhile_plain_fori_*`, `cumad_cumprod_n65536`.
* **falsification arms** — `adconst_*_{wtraced,live,stack}`, `dusfold_dynidx_300`,
  `seqmask_arange_b64_m50000`, `assocscan64_realwide_32k`, `xtile_issue_lowrank_subtrees`,
  `tilerank_peak{9,11,14}`, `rngimpl_unsafe_rbg_k64`, `convT*_tinyk`.

Read those rows against their siblings in the same results table, not against a `_control` suffix.

---

## Every case file

`Origin` is **mined** (a real issue/PR, URL in the docstring) or **synth** (constructed from
compiler source and then measured — stated as such in the docstring).
`Status`: **cpu-ok** = reproduces on CPU with its control; **cpu-ok(lower)** = reproduces but the
seconds are in `lower_s`, so `_harness.classify()` scores it "no (below floor)" by design;
**cpu-ok(bytes)** = the artifact is memory, not time; **gpu-pending** = written and structurally
verified, mechanism is GPU-side, never run on a GPU here; **no-repro** = deliberate negative,
committed to bound the axis; **weak** = real but under `MIN_VS_CONTROL`.

| File (`case_*.py`) | n | Gap / stage | Mechanism in one line | The one variable the control moves | Platform | Origin | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ad_control_flow_ndtri_scan` | 11 | AD rules x control flow | `jacrev` through `scan` over `ndtri`, which still has no derivative rule, so AD goes through the primitive's expansion | `custom_vjp` with the analytic derivative wrapped around `ndtri` | either (reported CPU) | mined jax#2609 | **cpu-ok 16.3x** (this run: 107.8 s vs 6.6 s at d16) |
| `ad_graph_shape_seq_update` | 10 | AD graph shape / HLO scheduling | gradient of a sequential in-place update is super-linear in N while the jaxpr stays exactly linear | drop `jax.grad`; same loop, same shapes | either | mined jax#17335 | cpu-ok, weak (this run: 6.2x, 71.0 s vs 11.5 s at N=1000) |
| `ad_transpose_bounded_while` | 8 | AD transpose of control flow | an O(1) forward pass whose transpose costs O(max_steps) to trace, lower and compile | one decorator: `jax.grad(loss)` vs `loss` | either (reported CPU) | mined jax#8239 | cpu-ok (this run: 5.9x compile / 5.5x lower at 512) |
| `ad_transpose_cond_recursion` | 8 | AD transpose of control flow | reverse mode through a recursively-halved `lax.cond` tree; the recursion is in Python, the cost is in transpose | one decorator; control grows underneath, so read the ratio's *trend* | either | mined jax#8239 | cpu-ok(lower) 9.6x (this run: 15.2 s vs 1.6 s lower at M=128) |
| `batching_rule_cho_solve_grad` | 12 | batching rule x linalg | compile time of `grad(vmap(cho_solve))` grows with BATCH size, which it must not | drop `value_and_grad`; second axis is the batch sweep itself | GPU reported (V100); CPU informative | mined jax#21313 | gpu-pending |
| `batching_rule_conv_persample` | 10 | batching rule x conv | per-sample conv gradients: compile scales with SAMPLE COUNT, runtime does not | formulation: `grad` of the summed batch instead of per-sample `vmap` | either (reported CPU 2019) | mined jax#1606 | **flat at N=300 in this run** (0.47 s, 1.1x) -- the 2019 number needs N=1000 or a GPU |
| `codegen_multiplicity_switch_branches` | 14 | many computations -> LLVM codegen | `lax.switch` over n identical branches is super-linear in n; nothing dedups the identical computations | the same call with a branch list of length 1 | either | mined jax#4453 | **cpu-ok 194x** (this run: 19.4 s vs 0.10 s at 512 branches) |
| `compilemem_literal_bytes` | 14 | **gap 15** compile-time HOST memory | a closure-captured array becomes a dense literal that exists in ~4 copies at once; the HLO module is 5 instructions | the same array passed as a jit ARGUMENT | either | synth | cpu-ok(bytes+time) |
| `compilemem_peak_live_fanout` | 28 | **gap 15** peak buffer memory | use count decides fusion: a value read twice cannot fuse, so N buffers stay live at byte-identical jaxpr size | one operand token: `acc` vs the parameter `x` | either | synth | cpu-ok(bytes) |
| `const_fold_ad_scatter` | 17 | constant folding, reached through AD | AD turns a constant-index gather into an all-constant scatter, which XLA then folds at compile time | index tables become traced ARGUMENTS | either (reported CPU) | mined pyroki#56 | **cpu-ok 16.1x** (this run: 45.2 s vs 2.8 s at 2^22) |
| `const_fold_dus_reduction` | 7 | constant folding (better controls) | one scalar store into a jit-internal constant, with controls that do not also delete the reduction | in-jit constant kept, only the store's operand hoisted | either | mined jax#12789 | cpu-ok, ratio 39x but 1.97 s -- under the floor (this run) |
| `const_fold_fft_capture` | 8 | constant folding, backend-split | FFT of a closure-captured array: >10 s on GPU, ~0.1 s on CPU for byte-identical HLO | captured array -> jit parameter | **GPU** (CPU is the negative) | mined jax#10596 | gpu-pending |
| `constant_folding_cumsum_seqmask` | 13 | constant folding of a scan ladder | `cumsum` over an all-constant array is evaluated by the compiler, log2(maxlen) levels of it | keep the ladder, make the input a parameter | **GPU** (measured there earlier); CPU flat | mined mlcommons/algorithmic-efficiency#877 | gpu-verified earlier; **CPU flat 0.9x** (this run) |
| `constant_folding_dus` | 12 | constant folding | XLA folds a whole N^3 buffer at compile time to serve one scalar store | hoist the buffer out of the jit | either (reported CPU) | mined jax#12789 | cpu-ok 8.8x but 1.21 s -- under the floor (this run; the 2022 report was 3.31 s) |
| `dtype_expansion_assoc_scan` | 6 | dtype expansion / algebraic simplification | `associative_scan` compiles ~7x slower for complex than for real, from a one-line toggle | `complexify=False` | either | mined jax#18221 | **largely CLOSED: 1.3x in this run** (3.84 vs 2.92 s at 50k) |
| `dtype_expansion_assoc_scan_width` | 7 | dtype expansion, WIDTH axis | same as above plus dtype width and a FLOP-matched arm the sibling lacks | complex->real at matched scan length | either | mined jax#18221 | **largely CLOSED: 1.3x in this run** (7.20 vs 5.64 s at 32k) |
| `dtype_f8_convert_expansion` | 12 | **gap 14** narrow dtypes | `FloatNormalization` inserts convert pairs; `f8_e4m3fn` needs a longer emitter expansion than `f8_e5m2` | one dtype token: `e4m3fn` -> `e5m2` at equal depth | CPU measured | synth | cpu-ok (3.1x) |
| `dtype_int4_subbyte_ops` | 13 | **gap 14** sub-byte dtypes | int4 produces ~4x the HLO of int8 and costs the SAME on CPU | int4 -> int8 (and int32 as a second control) | CPU measured | synth | **no-repro** (CPU) |
| `emitter_flat_cost_topk` | 18 | GPU emitter, flat cost | `lax.top_k` costs a flat multi-second compile at EVERY size; `jnp.sort` does not | `jnp.sort(v)[-k:]` instead of `lax.top_k` — the control is the *larger* program | **GPU** (CPU is the negative) | mined jax#19653 | gpu-pending |
| `export_poly_constraint_chain` | 16 | **gap 8** shape polymorphism | `_DecisionByElimination` pairs each new constraint against every derived combination; transitive DEPTH is exponential at fixed constraint COUNT | the right-hand side of K-1 constraints: a variable -> the constant 2 | either (host Python) | synth | cpu-ok(lower) 66x |
| `export_poly_monomial_blowup` | 10 | **gap 8** shape polymorphism | `_DimExpr.__mul__` distributes eagerly, so `(v0+1)*...*(v_{K-1}+1)` is 2^K monomials and bounding is O(4^K) | `+ 1` on each factor (1 monomial vs 2^K) | either (host Python) | synth | cpu-ok(lower) 441x |
| `ffi_customcall_count` | 16 | **gap 6** custom call / FFI | N identity FFI custom calls interleaved into an N-op chain; compile ~N^2.6 for zero added arithmetic | delete the `pure_callback` line; axis B fixes op count and moves only call count | either | synth | cpu-ok 28x |
| `ffi_target_diversity_lapack` | 16 | **gap 6** FFI target diversity | 6 distinct LAPACK FFI targets vs 1, at equal call count | all K calls to one target | CPU measured | synth | **no-repro** (0.97x) |
| `fusion_pass_barrier_chain` | 12 | fusion pass, INVERTED control | fusion is superlinear in unrolled chain length, and ADDING an `optimization_barrier` makes it FASTER | the barrier: the `_control` arm is the one with the extra op | either | mined xla#7971 / jax#18787 | inconclusive at n2000 (2.92 vs 3.19 s, 0.9x); re-read at n4000 |
| `gather_2d_chain` | 10 | indexing lowering, **CPU-only** | chained 2D fancy indexing compiles exponentially on CPU; flat on GPU (248x vs 1.02x, both measured) | indexing FORM: `data[r, c]` vs `data_flat[r*N + c]` | **CPU-only** | mined jax#32704 | cpu-ok 248x |
| `gpu_autotune_conv_transpose` | 8 | **GPU autotuner** | `lhs_dilate` sends a conv into `GpuConvAlgorithmPicker`, which benchmarks every cuDNN candidate: ~252 s compile, 1.8 ms run | stride 1 instead of 16 (same shapes, same HLO) | **GPU-only by construction** | mined xla#5541 / jax#17464 | gpu-pending |
| `gpu_autotune_gemm_shape_diversity` | 8 | **GPU autotuner** | K distinct GEMM shapes pay K autotunes; K identical shapes pay one | one shape for all K dots, FLOPs matched to 2% | **GPU-only by construction** | mined xla#35955 | gpu-pending |
| `graph_depth_tree_reduce` | 8 | critical-path DEPTH | does depth alone cost, at fixed op count, op kind, dtype and bytes? | `jax.tree.reduce` (chain) vs a balanced tree — depth differs 100x-3100x at equal eqn count | either | mined optax#1498 | **CPU NEGATIVE (this run): 0.8x** -- 95.9 s chain vs 113.4 s tree, depth is not the variable on XLA:CPU |
| `indexing_lowering_gather_chain` | 8 | indexing lowering | the same jax#32704 chain against the reporter's *corrected* flatten (the naive one is not a control at all) | `cols = flat[rows*N + cols]` | **CPU** (GPU measured flat) | mined jax#32704 | cpu-ok |
| `layout_conv_nested_computation` | 10 | **gap 1** layout / transpose folding | `ConvCanonicalization` walks only `entry_computation()`, so the identical conv inside a `scan` body never gets canonicalised and falls to the elemental emitter | `dimension_numbers` NCHW -> NHWC; a second pair moves only WHERE the conv lives | **CPU-only by construction** | synth | cpu-ok 19.7x |
| `layout_conv_spatial_rank` | 8 | **gap 1** layout / dimension_numbers | two extra size-1 spatial axes cross XLA:CPU's `num_spatial_dims in 1..3` Eigen gate, so every conv becomes a hand-emitted loop nest for zero extra MACs | reshape back to rank 4; same bytes, same MAC count | **CPU-only by construction** | synth | cpu-ok 10x |
| `layout_dot_dimension_numbers` | 16 | **gap 1** layout, dot half | permuted `dot_general` dimension_numbers force `DotDecomposer` to insert transposes; jaxpr identical, HLO different | canonical batch-major dimension_numbers | either; CPU weak, GPU is where it should matter | synth | **weak** (1.3-2.5x), gpu-pending |
| `llvm_ptx_unrolled_sample` | 8 | below XLA (LLVM/PTX) | python-unrolled sampling loop, diagnosed in 2020 as exponential *below* XLA | the reporter's own `scan=True` static flag | either (reported CPU+GPU) | mined jax#2777 (closed) | expect no-repro |
| `llvm_scan_unroll_spill` | 12 | **gap 4** LLVM codegen | `scan(unroll=K)` at fixed trip count: same FLOPs, K x the straight-line code and live values in one block; `--xla_backend_optimization_level=0` removes 74-89% of the compile | one integer: `unroll=1` | either; CPU measured | synth | cpu-ok 184x |
| `llvm_unroll_sorted_scatter` | 18 | below XLA (LLVM unroll) | one integer in a scatter's update width (34 vs 35) flips LLVM into full loop unrolling | flag control, plus the width read ACROSS rows | either | mined xla#22233 | **flat at w1091 (0.10 s, 1.2x)** -- no cliff there on CPU/0.10.2; the across-row w34/w35 read is still untested |
| `lowering_arity_pytree` | 14 | **gap 9** lowering as a stage | cost is set by the NUMBER of arrays crossing the jit boundary, and most of it lands before XLA | the container only: N `(dim,)` leaves vs one `(N, dim)` array | either | mined jax#4667 | cpu-ok(lower) |
| `lowering_stage_xla_metadata` | 10 | **gap 9 + 10** | MLIR attributes are uniqued, so cost tracks the number of DISTINCT metadata values, not equation count or byte count | one string literal: `f"v{i:08d}"` vs `"v00000000"` | either | synth | cpu-ok(lower) 5-11x |
| `manymodules_dispatch_constant` | 24 | **gap 12** many small modules | K real XLA compilations happen inside `.lower()` via `ensure_compile_time_eval`; the measured module compiles in 0.1 s after 20-38 s are already spent | an unused static tag held at 0 -> 1 compile + K-1 cache hits | either | synth | cpu-ok(lower) 11-55x |
| `operand_arity_stack_in_cond` | 8 | operand arity | `jnp.stack([y]*N)` on a never-taken branch: ONE instruction with N operands | `jnp.broadcast_to` — same value, 4 operands | either (reported CPU) | mined diffrax#606 | **cpu-ok: >265 s vs 0.15 s control**, XLA's own "Very slow compile?" alarm fired (this run; killed at 265 s) |
| `pallas_kernel_multiplicity` | 16 | **gap 5** Pallas/Triton | K kernels differing in one baked constant are K modules for the *second* compiler; the HLO is identical in both arms | hold the constant fixed so all K kernels hash equal | **GPU** for Triton arms; interp arms run on CPU | synth | gpu-pending (CPU interp = null) |
| `pallas_triton_body_size` | 24 | **gap 5** Pallas/Triton | a `pallas_call` is one opaque custom call: its body is compiled by Triton->LLVM->NVPTX->ptxas, invisible to every XLA-side metric; grid extent costs nothing | the same chain as plain `jnp` ops; and `interpret=True` on a byte-identical kernel | **GPU** for Triton arms; interp arms run on CPU | synth | gpu-pending |
| `pass_selection_argsort_dtype` | 13 | which PASS runs at all | `SortRewriter` swaps in a CUB radix sort only for some dtypes, so one dtype token changes the pass selection (~25x on GPU) | the same lambda on int32 | **GPU** (verified there earlier); CPU is the negative | mined xla#35587 | gpu-verified earlier |
| `primitive_cumulative_ad_rule` | 17 | **gap 13** cumulative ops | `cumlogsumexp` has an associative-scan AD rule where `cumsum` has a closed-form one, so the reverse pass expands | `lax.cumlogsumexp` -> `lax.cumsum` | either; CPU measured | synth | cpu-ok 8-13x |
| `primitive_fft_length_diversity` | 12 | **gap 13** FFT | 256 FFTs of 256 DISTINCT lengths vs 256 of one length: the CPU FFT custom call is keyed by length | `n=BASE+i` -> `n=BASE` | CPU verified; GPU should be stronger | synth | cpu-ok 2.7x |
| `primitive_linalg_expm_vs_lapack` | 16 | **gap 13** linalg | two findings: `expm` *unrolls in jax* (19.7x), while every linalg PRIMITIVE (cholesky/qr/svd/eigh/lu) is compile-time free on CPU | `expm(y)` -> `matrix_power(y, 13)`; cholesky column is the second control | CPU verified | synth | cpu-ok 19.7x + no-repro half |
| `primitive_rng_distribution_expansion` | 24 | **gap 13** RNG | which DISTRIBUTION you sample swings compile ~10x at fixed draw count; threefry-vs-rbg does nothing on CPU | `random.gamma` -> `random.uniform` at the same shape | CPU verified | synth | cpu-ok 4.4-10x + rbg no-repro |
| `primitive_select_and_scatter_poolgrad` | 18 | **gap 13** reduce_window | grad of MAX-pool becomes `select_and_scatter`, which `SelectAndScatterExpander` rewrites into a big loop nest; grad of AVG-pool does not | `lax.max` -> `lax.add` | CPU verified | synth | cpu-ok 12.8x |
| `primitive_while_dyn_trip_batching` | 18 | **gap 13** dynamic while | `_while_loop_batching_rule` fixpoint iteration costs ~K^2 in jax's own Python (35.7 s of LOWERING); XLA does NOT unroll a data-dependent while on CPU | one index in the body: `c[j-1]` -> `c[j]` | either (host Python) | synth | cpu-ok(lower) + unroll no-repro |
| `python_pytree_node_count` | 12 | **gap 10** Python-side cost | cost tracks pytree NODE count (and container kind) at fixed leaf count | same 32 leaves + M empty nodes carried in a LIST | either | synth | inconclusive at M=16000 (1.0x compile / 1.4x lower); read the M=64000 arm |
| `python_retrace_cache_key_storm` | 24 | **gap 10 + 9** | a jitted helper called K times with a varying-but-irrelevant cache key traces K times and lowers K identical MLIR functions; XLA inlines them all | an unused static arg held at 0 / a constant dict key | either | synth | cpu-ok(lower) |
| `remat_threshold_activation_chain` | 18 | **gap 3** rematerialization | `HloRematerialization` is a THRESHOLD pass: nothing below the memory limit, expanding-window block scan above it. CPU absence *verified* (94-pass list enumerated; pass absent) | 0.625x activation bytes (under the limit); plus a flag control and an effort control | **GPU-only** | synth | gpu-pending (CPU absence verified) |
| `shape_diversity_kernel_dedup` | 12 | **gap 2** scheduling / buffer assignment | written to test gap 2 and FALSIFIES it: allocation count and peak temp are identical; what changes is kernel dedup -> N distinct trip counts for LLVM | all parts one length vs all different, same total bytes | CPU measured | synth | cpu-ok(artifacts), weak wall clock |
| `size_cliff_bisection_unroll` | 17 | compiler-heuristic cliff | m=95 does not finish, m=96 compiles in ~10 s: `scan` unroll x nested scan x double vmap | m=96 — one column, ~1% MORE arithmetic | either (reported CPU) | mined jax#10621 | cliff, expect hang at m95 |
| `size_cliff_topk` | 12 | GPU emitter cliff | `top_k` blows up at particular input LENGTHS; n+1 is fine | one character: `n = (1<<20) + 1` | **GPU-only by construction** | mined jax#19653 | gpu-pending (expect no-repro on 0.10.2) |
| `spmd_halo_conv_dim_choice` | 6 | **gap 7** SPMD halo exchange | sharding a conv's SPATIAL dim forces halo exchange (192 synthesised collective-permutes); sharding BATCH keeps it device-local | one token's position inside one `PartitionSpec` | CPU measured (8 fake devices) | synth | **no-repro** (1.2-2.3x) — falsifies "partitioner cost = HLO emitted" |
| `spmd_reshard_permute` | 14 | **gap 7** SPMD reshard | a `PartitionSpec` that permutes which mesh axis owns which dim forces all-to-all + collective-permute synthesis: 72 StableHLO lines -> 2228 HLO lines | every constraint pinned to the SAME spec | CPU (fake devices); backend-independent | synth | cpu-ok 23.9x |
| `spmd_uneven_shard_cliff` | 14 | **gap 7** SPMD divisibility | N=513 on an 8-way mesh makes ragged tiles, so every boundary-dependent op grows iota/compare/select (0 -> 8003 selects) | N=512 (strictly smaller); second control N=520 (strictly LARGER and still faster) | CPU (fake devices) | synth | cpu-ok 7.3x (**weak** vs the 10x bar) |
| `symbolic_tile_rank_blowup` | 10 | XTile fusion emitter | symbolic-tile propagation blows up once intermediates exceed rank ~8 | op-count-matched control with MORE tensordots and rank <= 8 | **GPU** (18.5x measured earlier); CPU path fixed by PR#41174 | mined xla#41173 | gpu-verified earlier, CPU fixed |
| `trace_time_einsum_optimal` | 9 | trace time (third-party) | `einsum(optimize='optimal')` burns ~10 s inside opt_einsum and 0 s in XLA | one keyword: `optimize='auto'`; sharper still, an explicit path literal | platform-independent | mined jax#2583 / PR#25214 | cpu-ok(trace/lower) |
| `trace_time_nested_jit_fib` | 10 | trace time / call-graph inlining | doubly-recursive jit with `static_argnums` costs phi^t; single recursion costs t | `fib(t-1) + fib(t-2)` -> `fib(t-1) + 1` | either | mined jax#22385 | cpu-ok |
| `width_fanout_llvm_codegen` | 8 | **gaps 11 + 2, lands in 4** | fan-out topology alone: spread operands defeat LLVM's CSE/hoisting of a shared transcendental expansion. Every HLO-level metric is EQUAL; object code differs 6.3x | one integer index: `ps[i]` vs `ps[0]` | CPU measured | synth | cpu-ok 4x (**weak**) |
| `width_vs_depth_fixed_opcount` | 8 | **gap 11** width vs depth | W independent chains at exactly 8191 equations for every W: XLA:CPU charges nothing for ILP alone | the W=1 serial chain, same op count | CPU measured | synth | **no-repro** (1.2x) — the negative control for the width axis |

---

## Stage coverage

**Covered, with at least one case that separates from its control:**

| Stage | Files |
| --- | --- |
| tracing / Python front end | `trace_time_einsum_optimal`, `trace_time_nested_jit_fib`, `python_pytree_node_count`, `python_retrace_cache_key_storm`, `lowering_arity_pytree` |
| jax-internal rule expansion (AD, batching, primitives) | `ad_*` (4), `batching_rule_*` (2), `primitive_*` (6), `dtype_expansion_*` (2) |
| symbolic shapes / `jax.export` | `export_poly_monomial_blowup`, `export_poly_constraint_chain` |
| jaxpr -> StableHLO lowering | `lowering_stage_xla_metadata`, `lowering_arity_pytree`, `primitive_while_dyn_trip_batching` |
| per-module dispatch constant (many modules) | `manymodules_dispatch_constant`, `python_retrace_cache_key_storm` |
| HLO simplification / constant folding | `constant_folding_dus`, `const_fold_dus_reduction`, `const_fold_ad_scatter`, `const_fold_fft_capture`, `constant_folding_cumsum_seqmask` |
| HLO expanders (`SelectAndScatterExpander`, `FloatNormalization`, `DotDecomposer`) | `primitive_select_and_scatter_poolgrad`, `dtype_f8_convert_expansion`, `layout_dot_dimension_numbers` |
| layout assignment / transpose folding / conv canonicalisation | `layout_conv_spatial_rank`, `layout_conv_nested_computation`, `layout_dot_dimension_numbers` |
| fusion decisions | `fusion_pass_barrier_chain`, `compilemem_peak_live_fanout`, `shape_diversity_kernel_dedup` |
| scheduling / buffer assignment | `ad_graph_shape_seq_update`, `shape_diversity_kernel_dedup` (falsifies the naive premise) |
| rematerialization (threshold pass) | `remat_threshold_activation_chain` (GPU) |
| SPMD partitioning | `spmd_reshard_permute`, `spmd_uneven_shard_cliff`, `spmd_halo_conv_dim_choice` |
| emitters (elemental, top_k, XTile) | `emitter_flat_cost_topk`, `layout_conv_spatial_rank`, `symbolic_tile_rank_blowup` |
| pass SELECTION (which pass runs at all) | `pass_selection_argsort_dtype`, `remat_threshold_activation_chain` |
| custom call / FFI | `ffi_customcall_count`, `ffi_target_diversity_lapack` |
| second compilers (Pallas/Triton) | `pallas_triton_body_size`, `pallas_kernel_multiplicity` |
| LLVM / NVPTX / ptxas (below XLA) | `llvm_scan_unroll_spill`, `llvm_unroll_sorted_scatter`, `width_fanout_llvm_codegen`, `codegen_multiplicity_switch_branches`, `llvm_ptx_unrolled_sample` |
| GPU autotuners (conv, GEMM) | `gpu_autotune_conv_transpose`, `gpu_autotune_gemm_shape_diversity` |
| compile-time MEMORY as the artifact | `compilemem_literal_bytes`, `compilemem_peak_live_fanout` |
| size/heuristic cliffs (non-monotone in size) | `size_cliff_bisection_unroll`, `size_cliff_topk`, `llvm_unroll_sorted_scatter`, `spmd_uneven_shard_cliff` |
| graph shape at fixed op count (depth, width, fan-out) | `graph_depth_tree_reduce`, `width_vs_depth_fixed_opcount`, `width_fanout_llvm_codegen`, `operand_arity_stack_in_cond` |

**Not covered (no case in the suite touches these):**

1. **Persistent / distributed compilation cache** — every measurement here asserts the cache is off.
   Cache-key computation cost, false misses across jax versions, and cache-write serialisation are
   all unrepresented.
2. **Multi-host / multi-slice SPMD** — the three SPMD files use fake CPU devices on one host. No
   case exercises collective-permute *scheduling* across hosts, `jax.experimental.multihost_utils`,
   or the latency-hiding scheduler.
3. **TPU / Mosaic** — out of scope by declaration (no TPU in this environment); the Pallas files
   cover only the Triton path, so `mosaic`/`tpu_sc` lowering is untouched.
4. **AOT / `jax.export` serialisation round-trips** — the two export files stop at symbolic-shape
   arithmetic; nothing measures `Exported` serialisation, deserialisation or calling convention
   refinement.
5. **`shard_map` / manual collectives** — all sharding cases go through `with_sharding_constraint`
   and automatic partitioning.
6. **Sparse (`jax.experimental.sparse`) and `jax.experimental.jet` / higher-order AD beyond grad-of-grad.**
7. **`jax.lax.platform_dependent`, custom partitioning, and custom-call *layout* constraints** —
   `ffi_*` covers call count and target diversity only; nothing covers FFI attribute payloads,
   aliasing, or user-registered targets with C++ handlers.
8. **Autotuning caches on GPU** (`--xla_gpu_dump_autotune_results_to`) as a *variable* — the two
   autotune cases measure the un-cached path only.
9. **Runtime/thunk construction as its own stage** — `ffi_customcall_count` implicates it (the cost
   survives `--xla_backend_optimization_level=0`) but no case isolates thunk-sequence construction
   from the HLO pipeline.
10. **Compiler *version* as an axis** — `symbolic_tile_rank_blowup` documents a CPU regression that
    was fixed between 0.9.0 and 0.9.1, but the suite pins one jax build and cannot sweep it.

---

## Harness blind spots this suite deliberately exercises

`_harness.classify()` gates on `compile_s >= 3.0` first, then on `compile_s / control.compile_s >= 10`.
Eleven files are *designed* to score "no" under that rule while reproducing perfectly:

* **cost is in `lower_s`, not `compile_s`** — `export_poly_monomial_blowup` (441x),
  `export_poly_constraint_chain` (66x), `manymodules_dispatch_constant` (11-55x),
  `lowering_stage_xla_metadata` (5-11x), `python_retrace_cache_key_storm`, `python_pytree_node_count`,
  `lowering_arity_pytree`, `primitive_while_dyn_trip_batching`, `ad_control_flow_ndtri_scan`.
* **cost is in bytes, not seconds** — `compilemem_peak_live_fanout` (1x -> 50x peak temp at 2.5x wall
  clock), `compilemem_literal_bytes` (memory and time move in *opposite* directions when one literal
  is split into 64).
* **real but under the 10x bar** — `spmd_uneven_shard_cliff` (7.3x), `layout_conv_nested_computation`
  at D=64 (8.1x), `width_fanout_llvm_codegen` (4x), `dtype_expansion_assoc_scan` (~7x).

A tool that only reads `compile_s` fails all of them, which is the point of keeping them.

One harness bug was found and fixed while running the sweep below: `_run_one` did not catch
`subprocess.TimeoutExpired`, so the first case that outran `--timeout` aborted the whole sweep and
discarded every row already measured. `stackcond_n30000` does exactly that (it is still compiling at
265 s). Timeouts are now returned as data and `classify()` reports
`TIMEOUT (>Ns, still compiling)`, which is a *reproduction*, not a failure.

---

## CPU sweep, 2026-07-29 (62 pairs, 124 fresh-process measurements)

```bash
JAX_PLATFORMS=cpu python _harness.py --platform cpu --timeout 300 <62 case+control names>
```

Raw output: `cpu_subset_2026-07-29.log`, `cpu_subset_2026-07-29.json`. One case+control pair from
every file that has any CPU-side story, at the largest size that was not expected to exceed ~60 s.
**Contention caveat**: load average was 11-13 on 20 cores throughout (other agents compiling) and the
GPU was held by 8 foreign processes. Absolute seconds are UPPER BOUNDS; the paired ratios are taken
back to back under the same load and are the numbers to read. Numbers below are `compile_s` unless
`lower` is stated.

**A. Reproduced on `compile_s` — clears both harness bars (14)**

| case | compile | control | x |
| --- | --- | --- | --- |
| `gatherchain2d_9` | 53.51 | 0.15 | **363x** |
| `switch_ident_512` | 19.38 | 0.10 | **194x** |
| `jitfib_t22` | 10.21 | 0.09 | **119x** |
| `gather2d_8` | 15.06 | 0.15 | **101x** |
| `arity_tree_100` | 10.62 | 0.30 | **35x** (also 12.1x on `lower_s`) |
| `linalg_expm_d32` | 4.00 | 0.20 | **20x** |
| `scan_unroll_32` | 3.97 | 0.24 | **17x** |
| `ndtri_scan_jacrev_d16` | 107.81 | 6.63 | **16x** |
| `adconst_idx_2p22` | 45.21 | 2.80 | **16x** |
| `spmd_reshard_d128` | 6.55 | 0.46 | **14x** |
| `poolgrad_max2d_d128` | 3.48 | 0.30 | **12x** |
| `convrank_k256` | 5.27 | 0.47 | **11x** |
| `convscan_nchw_d128` | 3.82 | 0.35 | **11x** |
| `stackcond_n30000` | **>265** (killed) | 0.15 | **>1700x**, and XLA printed its own `Very slow compile?` alarm |

**B. Reproduced on `lower_s`, scored "no (below floor)" by design (9)**

| case | lower | control lower | x | compile (both arms) |
| --- | --- | --- | --- | --- |
| `einsum_optimal_n11` | 56.43 | 0.60 | **94x** | 0.36 / 0.35 |
| `manymod_100` | 37.70 | 1.39 | **27x** | 0.12 / 0.12 |
| `exportpoly_monomials_128` | 9.26 | 0.74 | **13x** | 0.13 / 0.13 |
| `retrace_static_320` | 9.85 | 0.85 | **12x** | 5.58 / 3.09 |
| `dynwhile_fixpoint_k128` | 13.04 | 1.27 | **10x** | 2.84 / 2.75 |
| `condrec_grad_128` | 15.19 | 1.57 | **9.6x** | 1.79 / 0.77 |
| `meta_distinct_1200` | 5.46 | 1.00 | **5.5x** | 4.84 / 4.64 |
| `bwl_grad_512` | 2.77 | 0.50 | **5.5x** | 4.43 / 0.76 |
| `exportpoly_chain_k9` | 3.16 | 0.78 | **4.1x** | 0.13 / 0.13 |

Every arm in group B has a `compile_s` ratio of 1.0-1.8x. A profiler that reads only compile time
reports nine healthy programs here.

**C. Separates, but misses a bar (14)** — ratio first, then why it does not score:

`dusfold_sum_300` 39x but 1.97 s · `litmem_dense_32` 36x but 1.28 s (its real artifact is bytes) ·
`xtile_issue` 13x but 0.91 s (the CPU emitter fix landed; GPU is where this lives) ·
`cumad_lse_n262144` 8.9x · `constfold_ones_352` 8.8x at 1.21 s · `ffi_cb_chain_2048` 7.0x ·
`seqgrad_N1000` 6.2x (71.0 vs 11.5 s) · `rngdist_gamma_k16` 5.3x · `fanout_spread_p512` 4.0x ·
`peaklive_fanout_n256` 3.9x (artifact is peak bytes) · `spmd_uneven_d32` 3.8x (7.3x at d64) ·
`f8chain_e4m3fn_d48` 3.6x · `fftlen_distinct_k128` 2.5x · `shapediv_bitrev_n256` 1.9x.

**D. Flat, exactly as the file predicts (17)** — GPU-side mechanism, or a committed negative:

`topkflat_n2p20_k8` · `topk_pow20_k128` · `argsort_f32_1e6` · `cho_grad_b1024` · `convT_dilate16` ·
`gemm_shapes_k32` · `fftcap_1000000` · `remat_small_flag` · `i4chain_d64` · `lap_div_96` ·
`spmd_halo_conv_d32` · `dotdims_mid_k32` (2.3x) · `widthdepth_w256` (0.9x — the width null) ·
`pallas_interp_d256` · `pallas_kernels_interp_64` · `unrolled_sample_7` (jax#2777 stays closed) ·
`pallas_body_d64`, which fails with exactly the documented
`ValueError: Only interpret mode is supported on CPU backend.`

**E. Flat, and that is NEW information (8)** — the suite's documented status for these is now weaker:

| case | this run | what it means |
| --- | --- | --- |
| `assocscan_complex_50k` | 3.84 vs 2.92 s (1.3x) | jax#18221's ~7x complex/real gap has **largely closed** on 0.10.2/CPU |
| `assocscan64_complex_32k` | 7.20 vs 5.64 s (1.3x) | same, on the width axis |
| `treereduce_chain_n10000` | 95.94 vs 113.43 s (0.8x) | **depth is not the variable** on XLA:CPU at fixed op count — an independent confirmation of `width_vs_depth_fixed_opcount`'s null, from the opposite direction |
| `sorted_scatter_w1091` | 0.10 s, 1.2x | no unroll cliff at w1091 on CPU/0.10.2; the across-row w34/w35 read is still untested |
| `conv_persample_300` | 0.47 s, 1.1x | the 2019 CPU number does not reproduce at N=300 |
| `seqmask_cumsum_b64_m20000` | 0.32 s, 0.9x | CPU flat; the file's own measurement was on GPU |
| `fusion_chain_n2000` | 2.92 vs 3.19 s (0.9x) | the inverted-control effect is not visible at n2000; re-read at n4000 |
| `pytree_keyed_16000` | 1.0x compile / 1.4x lower | needs the M=64000 arm to separate |

**Totals for this sweep**: 23 of 62 sampled cases reproduced decisively on CPU (14 on `compile_s`,
9 on `lower_s`), 14 more separated from their control but under a harness bar, 17 were flat as
designed, and 8 were flat in a way that updates the suite. Zero files failed to import, zero errors
other than the two intended ones (`pallas_body_d64`'s CPU `ValueError`, and `stackcond_n30000` killed
at 265 s of compile).
