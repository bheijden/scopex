"""SYNTHESISED (gap 3). HloRematerialization is a THRESHOLD pass: it does nothing until scheduled
peak memory crosses the device limit, then runs an O(n * block_size * decisions) search. Same
program, 37.5% fewer bytes per activation, and the pass never fires.

    PLATFORM: GPU-ONLY. Written here, VERIFIED LATER -- the GPU was owned by another
    investigation when this file was authored, and nothing in this workflow was allowed to touch
    it. Every number below for the CPU arm is measured; every claim about the GPU arm is a
    prediction with the instrument attached.

    THE CPU ABSENCE IS VERIFIED, NOT ASSUMED. The full XLA:CPU pass list in this build was
    enumerated by running one tiny compile under

        XLA_FLAGS=--xla_dump_to=<dir> TF_CPP_MIN_LOG_LEVEL=0 \
        TF_CPP_VMODULE=hlo_pass_pipeline=1 JAX_PLATFORMS=cpu python -c '<compile anything>' \
        2>&1 | grep -o "HLO pass: [a-zA-Z0-9_-]*" | sort -u

    94 distinct passes, and `rematerialization` is not among them. Corroborating: sweeping
    `jax_memory_fitting_level` over O0/O1/O2/O3 crossed with `jax_memory_fitting_effort` in
    {-1.0, 0.0, +1.0} -- twelve combinations -- changes the CPU peak temp allocation by exactly
    zero bytes. The memory-fitting machinery is compiled in but inert on CPU. So a flat result on
    CPU is a statement about the CPU pipeline and carries no information about this case.

No issue URL: constructed from `xla/hlo/transforms/simplifiers/hlo_rematerialization.cc`. The
audit's justification for the gap is that XLA maintainers added a permanent warning for
rematerialization passes exceeding three minutes, so the pathology is common enough to have earned
its own alarm.

THE MECHANISM, from the source. `HloRematerialization` runs after scheduling. If peak memory is
already under `memory_limit_bytes` it walks the sequence and returns. If it is over, it searches:

    for (auto* start_item = instruction_list.first_skip_node(); ...)
      while (block.size() <= max_block_size) {
        auto cost = GetCostOfRecompute(block, memory_limit_bytes);
        ++effort;
        block.push_back(next_item);
      }

an expanding-window block scan over every instruction, each window re-evaluating
`MemoryReducedIfRematerialized`, repeated until peak fits. So the cost is
O(n * block_size * number_of_remat_decisions) -- and it is ZERO one byte below the limit. That
discontinuity is the whole case. Nothing else in the corpus has a compile cost that is exactly
zero on one side of a data-size boundary and quadratic-ish on the other, with the program
structurally unchanged.

HOW THE PEAK IS MANUFACTURED, and why it is not `optimization_barrier`. XLA refuses to vertically
fuse an elementwise producer with more than one user (fusing would duplicate its code into every
consumer). Each activation here is read TWICE -- once by the next forward step, once by the
reverse-order consumer -- so none of them fuse away and each gets its own buffer. Consuming them
in REVERSE order means activation 0 is needed last, so at the end of the forward pass all `depth`
activations are simultaneously live and peak = depth * activation_bytes. This is exactly the shape
`jax.grad` of a deep chain produces, and exactly the shape rematerialization exists to fix: every
activation is recomputable from its predecessor by two cheap elementwise ops.

That generator is MEASURED ON CPU in `case_compilemem_peak_live_fanout.py`: the two-user form
takes peak temp from 1 buffer to N buffers at byte-identical jaxpr size, 1.0 MiB -> 50.0 MiB. This
file is the same generator scaled until the peak crosses a GPU memory limit. A barrier would have
forced liveness by fiat and changed the instruction mix; the fusion rule does it at matched op
count.

SIZING, AND WHY THE SWEEP IS THE INSTRUMENT. XLA:GPU derives the limit in `GetSchedulerMemoryLimit`
as `module->config().device_memory_size()` when set, else `device_memory_size * 80 / 100`, minus
total I/O size. jaxlib 0.10.2's `ExecutableBuildOptions` exposes no `device_memory_size` field
(checked: the attribute does not exist), so jax cannot set it from Python and the 80% path is the
one to expect -- about 12.8 GiB on a 16 GiB card. But that is a prediction, and
`--xla_gpu_memory_limit_slop_factor` moves it. DO NOT trust the number: the DEPTHS sweep below
spans 3.1 / 6.1 / 9.1 / 11.1 / 13.1 GiB of peak precisely so that the depth at which compile time
jumps IS the measurement of the limit. One data point cannot tell a threshold from a slope.

Read the peak the compiler actually assigned with

    jax.jit(fn).lower(*args).compile().memory_analysis().temp_size_in_bytes

on the target device, and re-centre ACT_ROWS / ACT_FEAT / DEPTHS if the card is not ~16 GiB. The
peak of arm `remat_thresh_d{D}` is (D + 1) * ACT_ROWS * ACT_FEAT * 4 bytes -- confirmed against
the CPU machinery arm, which reports exactly 33 x 4 MiB at depth 32.

WHAT EACH CONTROL ISOLATES. Three axes, and they are not interchangeable.

  * `remat_thresh_d{D}_control` (SIZE): identical depth, identical op count, identical primitives;
    the activation's feature dimension is 8192 -> 5120, i.e. 0.625x the bytes, which drops peak
    below the limit at every depth in the sweep. Structurally the same program doing strictly less
    work. If it compiles in a fraction of the time, the pass fired. At the small depths BOTH arms
    are under the limit and the pair is expected to be FLAT -- that flatness at low D and
    separation at high D is the threshold, and is why the sweep has five points.
  * `remat_flag_d{D}_control` (PASS): a BYTE-IDENTICAL program with
    `jax_compiler_enable_remat_pass = False`, which jax turns into
    `debug_options.xla_disable_hlo_passes = "rematerialization"` (jax/_src/compiler.py:230). This
    holds the program fixed and removes only the pass, separating "the pass ran" from "the program
    is big" -- the size control alone cannot do that.
    !! CAVEAT, stated because it will bite: with the pass off, the executable's peak stays above
    the limit and may fail to ALLOCATE at run time. The harness runs every arm four times, so this
    control can come back as an OOM error rather than a time. An OOM here is itself confirmation
    that remat was load-bearing, but to get a number out of it, run this arm with
    `XLA_PYTHON_CLIENT_PREALLOCATE=false` (or `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`); jax's default
    0.75 preallocation pool is SMALLER than XLA's 80%-of-device remat limit, so the window between
    "over the limit" and "allocatable" is closed by default.
  * `remat_effort_d{D}_control` (BUDGET): the identical over-limit program with
    `jax_memory_fitting_effort` at -1.0 against +1.0. Both arms still rematerialize, so both still
    fit and neither can OOM -- this is the allocation-safe way to move only the search budget. If
    a profiler attributes the effort pair and the flag pair to the same place, it has not localised
    the pass.

`remat_small_flag*` is a fourth, deliberately tiny pair whose only job is to be compilable on CPU:
it validates the config-mutation plumbing used by the flag and effort arms end to end, and it is
EXPECTED TO BE FLAT everywhere, on CPU because the pass is absent and on GPU because it is nowhere
near any limit. It is a machinery check, not a claim. It was run, and it does both jobs:

    remat_small_flag          lower 0.148 s  compile 0.369 s  temp 132.0 MiB
    remat_small_flag_control  lower 0.064 s  compile 0.252 s  temp 132.0 MiB

Flat, as predicted. The 132.0 MiB temp is 33 x 4 MiB, i.e. the generator really does hold all
`depth` activations live at once, which is the property the GPU arms scale up. And the plumbing
was checked directly rather than inferred -- after tracing `remat_small_flag_control` on CPU:

    jax.config.jax_compiler_enable_remat_pass                        -> False
    get_compile_options(...).executable_build_options
        .debug_options.xla_disable_hlo_passes                        -> 'rematerialization'

so the trace-time update does reach XLA's debug options, by name.

HOW THE CONFIG ARMS WORK, and why this is safe. The flag and effort arms cannot set config at
module scope -- the harness imports every case module in one parent process and then hands its
environment to every child, so a module-scope mutation would leak into unrelated cases. They also
cannot use `jax.jit(..., compiler_options=...)`, because the harness owns the jit and calls
`.compile()` with no arguments. So the setting is applied INSIDE the traced function. That works
because jax builds `xla_client.CompileOptions` in `get_compile_options`, called from `.compile()`,
strictly after `.lower()` has traced. Verified on CPU: setting `jax_disable_most_optimizations`
from inside the traced function changes the resulting optimised HLO (61 -> 64 lines), so a
trace-time config update does reach the compiler. Each measurement is its own subprocess running
exactly one arm, so the global mutation cannot reach anything else.

RUNTIME COST, stated plainly. `remat_thresh_d104` allocates ~13 GiB of device memory and the
harness executes it four times. This file needs an idle GPU with at least 16 GiB. It is not
CPU-runnable at these sizes and is not meant to be.

Memory at import: zero. Both arguments are `np.zeros`, which is calloc-backed -- virtual pages
only until jax touches them.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

# Activation geometry. 4096 x 8192 f32 = 128.0 MiB per activation.
ACT_ROWS = 4096
ACT_FEAT = 8192
# Control geometry: 0.625x the bytes, same rank, same op count. 4096 x 5120 f32 = 80.0 MiB.
ACT_FEAT_CTL = 5120

# Peaks: 3.0 / 6.0 / 9.0 / 11.0 / 13.0 GiB. Chosen to bracket both jax's default 12 GiB
# preallocation pool and XLA's predicted ~12.8 GiB limit on a 16 GiB card.
DEPTHS = (24, 48, 72, 88, 104)

# One depth comfortably under any plausible limit and one over, so the flag arm is measured on
# both sides of the threshold rather than only above it.
FLAG_DEPTHS = (72, 104)
EFFORT_DEPTH = 104

# CPU-compilable machinery check: 1024 x 1024 f32 = 4.0 MiB, depth 32 -> 128 MiB peak.
SMALL_ROWS = 1024
SMALL_FEAT = 1024
SMALL_DEPTH = 32

# Two f32 constants that are exact in binary, so the arithmetic is bit-reproducible and no
# rounding difference can be mistaken for a compiler difference between arms.
_SCALE = 1.0009765625        # 1 + 2**-10
_SHIFT = 0.00048828125       # 2**-11


def _chain(depth: int, x):
    """Forward chain of `depth` activations, consumed in REVERSE order.

    Two properties, both load-bearing:

      * every activation is read twice -- by the next forward step and by the reverse consumer --
        so XLA's multi-use fusion guard refuses to fuse it away and it is materialised into its
        own buffer;
      * the reverse consumption order means activation 0 dies last, so all `depth` activations are
        live simultaneously at the end of the forward pass and peak = depth * activation bytes.

    Every activation is recomputable from its predecessor by one multiply-add and one tanh, which
    is exactly the recompute-cheap / store-expensive shape HloRematerialization is written for.
    """
    acts = []
    h = x
    for _ in range(depth):
        h = jnp.tanh(h * _SCALE + _SHIFT)
        acts.append(h)
    g = acts[-1]
    for a in reversed(acts[:-1]):
        g = g * 0.5 + a
    return g.sum()


def _with_config(settings: dict, fn):
    """Apply jax config settings at TRACE time, so `.compile()` sees them.

    Not at module scope: the harness imports every case file in one parent process and passes that
    process's environment and interpreter state assumptions down, so a module-scope mutation would
    silently change unrelated cases. Not via `jit(compiler_options=...)` either: the harness owns
    the jit. Inside the traced function is the one place that is both per-case and early enough --
    `get_compile_options` runs during `.compile()`, after tracing. Verified on CPU (see docstring).
    """

    def wrapped(*args):
        for k, v in settings.items():
            jax.config.update(k, v)
        return fn(*args)

    wrapped.__name__ = f"cfg_{getattr(fn, '__name__', 'fn')}"
    return wrapped


def _zeros(rows: int, feat: int):
    # calloc-backed: no physical pages at import, so discovering CASES is free even though the
    # 4096 x 8192 argument is 128 MiB.
    return np.zeros((rows, feat), dtype=np.float32)


def _gib(depth: int, feat: int) -> float:
    # depth activations plus the one buffer the reverse pass carries. Verified against the CPU
    # machinery arm, which reports exactly 33 x 4 MiB at depth 32.
    return (depth + 1) * ACT_ROWS * feat * 4 / (1 << 30)


CASES: dict = {}

# --- axis 1: SIZE. Same depth, same ops, 0.625x the bytes per activation ----------------------
for _d in DEPTHS:
    CASES[f"remat_thresh_d{_d}"] = (
        functools.partial(_chain, _d),
        (_zeros(ACT_ROWS, ACT_FEAT),),
        f"gap3 SYNTH, GPU-ONLY (unverified, GPU was off-limits): depth={_d} activation chain, "
        f"peak {_gib(_d, ACT_FEAT):.1f} GiB -- above the limit HloRematerialization engages and "
        f"pays an O(n*block*decisions) search; the sweep locates the threshold. Flat on CPU: the "
        f"CPU pipeline has no rematerialization pass (verified, 94 passes enumerated)",
    )
    CASES[f"remat_thresh_d{_d}_control"] = (
        functools.partial(_chain, _d),
        (_zeros(ACT_ROWS, ACT_FEAT_CTL),),
        f"control: identical depth={_d}, identical op count and primitives, feature dim "
        f"{ACT_FEAT}->{ACT_FEAT_CTL} so peak is {_gib(_d, ACT_FEAT_CTL):.1f} GiB -- under any "
        f"plausible limit, so the pass is a no-op walk. Expected FLAT at low depth (both arms "
        f"under) and separated at high depth; that is the threshold",
    )

# --- axis 2: THE PASS ITSELF. Byte-identical program, pass disabled in the control ------------
for _d in FLAG_DEPTHS:
    CASES[f"remat_flag_d{_d}"] = (
        _with_config({"jax_compiler_enable_remat_pass": True},
                     functools.partial(_chain, _d)),
        (_zeros(ACT_ROWS, ACT_FEAT),),
        f"gap3 SYNTH, GPU-ONLY: depth={_d}, peak {_gib(_d, ACT_FEAT):.1f} GiB, remat pass "
        f"explicitly ENABLED (set at trace time so it reaches .compile())",
    )
    CASES[f"remat_flag_d{_d}_control"] = (
        _with_config({"jax_compiler_enable_remat_pass": False},
                     functools.partial(_chain, _d)),
        (_zeros(ACT_ROWS, ACT_FEAT),),
        f"control: BYTE-IDENTICAL program, jax_compiler_enable_remat_pass=False -> "
        f"xla_disable_hlo_passes='rematerialization'. Isolates the pass from the program size. "
        f"CAVEAT: with the pass off the executable's {_gib(_d, ACT_FEAT):.1f} GiB peak may fail "
        f"to allocate when the harness runs it -- use XLA_PYTHON_CLIENT_PREALLOCATE=false; an "
        f"OOM here is itself evidence the pass was load-bearing",
    )

# --- axis 3: SEARCH BUDGET. Identical over-limit program, only the effort dial moves ----------
CASES[f"remat_effort_d{EFFORT_DEPTH}"] = (
    _with_config({"jax_compiler_enable_remat_pass": True, "jax_memory_fitting_effort": 1.0},
                 functools.partial(_chain, EFFORT_DEPTH)),
    (_zeros(ACT_ROWS, ACT_FEAT),),
    f"gap3 SYNTH, GPU-ONLY: depth={EFFORT_DEPTH}, peak {_gib(EFFORT_DEPTH, ACT_FEAT):.1f} GiB, "
    f"jax_memory_fitting_effort=+1.0 -- maximum memory-fitting search budget",
)
CASES[f"remat_effort_d{EFFORT_DEPTH}_control"] = (
    _with_config({"jax_compiler_enable_remat_pass": True, "jax_memory_fitting_effort": -1.0},
                 functools.partial(_chain, EFFORT_DEPTH)),
    (_zeros(ACT_ROWS, ACT_FEAT),),
    f"control: identical program, jax_memory_fitting_effort=-1.0. Both arms still rematerialize "
    f"so both still FIT -- this is the allocation-safe way to move only the search budget, and "
    f"it must not be attributed to the same place as the flag pair",
)

# --- machinery check: small enough to compile on CPU. Expected FLAT everywhere ----------------
CASES["remat_small_flag"] = (
    _with_config({"jax_compiler_enable_remat_pass": True},
                 functools.partial(_chain, SMALL_DEPTH)),
    (np.zeros((SMALL_ROWS, SMALL_FEAT), dtype=np.float32),),
    f"machinery check, NOT a claim: depth={SMALL_DEPTH} at 4 MiB per activation, measured "
    f"132.0 MiB peak temp on CPU, remat pass enabled. CPU-compilable; verified that the "
    f"trace-time config update reaches xla_disable_hlo_passes by name. MEASURED FLAT: 0.369 s "
    f"against the control's 0.252 s, same 132.0 MiB temp",
)
CASES["remat_small_flag_control"] = (
    _with_config({"jax_compiler_enable_remat_pass": False},
                 functools.partial(_chain, SMALL_DEPTH)),
    (np.zeros((SMALL_ROWS, SMALL_FEAT), dtype=np.float32),),
    "control: identical 132 MiB-peak program with the remat pass disabled. MEASURED FLAT on CPU "
    "(0.252 s) -- 132 MiB is under every limit, so the pass has nothing to do even where it "
    "exists, and on CPU it does not exist at all",
)
