"""xla#41173 -- the XLA:CPU XTile fusion emitter hangs/OOMs once a tensordot network's intermediates
exceed rank ~8. The control variable is MAX INTERMEDIATE RANK, and nothing else.

    https://github.com/openxla/xla/issues/41173        (opened 2026-04-19)
    downstream: jax-ml/jax#35646, jax-ml/jax#35958

Reported: upgrading jax 0.9.0 -> 0.9.1 turns a seconds-long CPU compile into a ~9-minute one that
then OOMs. Bisected exactly by the reporter, who profiled it: commit 67f9646 (PR #36510) replaced
the CPU backend's fusion emitter stub -- previously a NO-OP -- with a real tiled emitter that calls
`GetSymbolicTileAnalysis` -> `SymbolicTileAnalysis::AnalyzeComputation`. That path combines
`xla::ConstraintExpression` objects, and the combination degrades catastrophically for tensor rank
above roughly 8. The reporter's own fix is a hard bail-out:

    constexpr int kMaxRank = 8;                       // xla/backends/cpu/codegen/tiled/
    if (instruction->shape().dimensions_size() > kMaxRank)   //   tiled_fusion_emitter.cc
      return Internal("Unsupported fusion in EmitGeneric: tensor rank too large");

That patch names the control variable for us, which is why this case is worth a slot: symbolic
tiling / indexing-map constraint solving is a compiler stage nothing else in the corpus touches,
and it is the stage the whole XTile/Triton direction is built on.

  *** RUN THIS ON GPU. THE CPU ARM IS THE ONE THAT IS NOW FIXED. ***
An earlier revision of this file pinned `jax_platforms="cpu"`, on the strength of the reporter's
statement that XLA:GPU is unaffected. THAT PIN HAS BEEN REMOVED, because it would have produced a
confidently wrong null. Measured in THIS environment (jax 0.10.2, x64, the same machine the harness
runs on) by the search phase, on the verbatim program below:

    cuda   compile 3.319 s   runtime 638.6 us   ratio 5197     control 0.179 s  -> 18.5x   REPRODUCES
    cpu    compile 0.303 s                                                                 fixed

The inversion is dated and has a cause. The reporter's own bail-out --

    constexpr int kMaxRank = 8;    // xla/backends/cpu/codegen/tiled/tiled_fusion_emitter.cc

-- landed for the CPU emitter as PR #41174 (Apr 2026) and is in this build, so on CPU the emitter
now refuses the rank>8 fusion and returns in milliseconds. The GPU tiled path carries no such cap,
so the pathology moved rather than disappeared: it is live on exactly the backend the issue said was
safe. Do not reinstate the pin, and do not copy the issue script's `CUDA_VISIBLE_DEVICES=''` line.
Run `--platform cuda,cpu` and read both: the CPU column is a dated fix, not a non-reproduction.

TWO ARMS, TWO KINDS OF CONTROL
--------------------------------------------------------------------------------------------------
1. `xtile_issue` is the reporter's program transcribed verbatim (29 tensordots over ten 2x2x2 "p"
   tensors and ten 2x2x2x2 "h" tensors; only the variable names are changed). Its scored control,
   `xtile_issue_control`, is the OP-COUNT-MATCHED one measured above at 0.179 s: 30 tensordots over
   the same `p`/`h` tensors, in a contraction order that never leaves rank 3. Same primitive, same
   operands, same dtype, MORE tensordots than the arm it controls (30 vs 29) -- so op count cannot
   explain the 18.5x, and rank is the only surviving variable. The earlier subtree-truncated control
   is kept as `xtile_issue_lowrank_subtrees`, an unpaired probe, because it answers a different
   question (are the low-rank subtrees of the ORIGINAL program cheap?) and it is not op-count
   matched. Its rank profile, computed with numpy so it costs no device time, is

       op       1  2  3  4  5  6  7  8  9 10 11 12 13 ... 19 ... 29
       rank     4  5  4  7  4  5  4  7 10  6  6 10 12 ... 10 ...  0
                                     ^^ first rank>8      ^^ peak rank is 12, not 10

   `xtile_issue_lowrank_subtrees` keeps the SEVENTEEN tensordots whose results are all rank <= 7 -- the
   independent low-rank subtrees, ops 1-8, 10, 11, 14, 16, 17, 20-23 -- on the SAME input tensors,
   and reduces each surviving leaf with `.sum()` instead of combining them. Same dtype, same tensor
   sizes, same contraction style; the twelve dropped ops are exactly the ones that build a rank>8
   intermediate. It has fewer *tensordots* than its case (17 vs 29) but MORE total jaxpr equations
   (32 vs 29, measured: 17 contractions + 8 sums + 7 adds), so a size-based attribution has no
   easier time here than it does with the barrier case.

2. `tilerank_peak*` is a synthetic network with the confound inverted. Each arm alternately grows a
   rank-3 tensor to a peak rank (contract one axis against a fresh 2x2x2 tensor, +1 rank per op)
   and collapses it back to rank 3 (contract two axes, -1 rank per op), for some number of cycles.
   Peak rank and op count are therefore independent knobs, and the controls are chosen to have
   STRICTLY MORE tensordots than the arms they control:

       arm                        peak rank   tensordots
       tilerank_peak10                   10       14
       tilerank_peak10_control            8       20      <- more ops, lower rank
       tilerank_peak12                   12       18
       tilerank_peak12_control            8       30      <- more ops, lower rank

   Every tensor is 2x2x...x2, so the largest object anywhere in the file is 2**14 = 16384 float32s
   and runtime is microseconds in every arm. Rank is the only thing that varies.

MEASURE THE RANK, DO NOT ASSUME IT. `max_intermediate_rank(name)` recomputes each arm's peak rank
from the actual jaxpr, so the control variable is checked rather than asserted in a comment. The
per-arm `note` strings carry the numpy-computed rank and op count for the results table.

WHAT IS UNCERTAIN. The reported nine-minute hang was the PRE-FIX CPU emitter; on this build the
verbatim program lands at 3.319 s on GPU, which clears the 3.0 s floor with almost nothing to spare.
It is the corpus's most marginal reproduction, so a run that comes back at 2.8 s is noise around the
bar rather than a refutation -- read the 18.5x control ratio, which is not marginal at all, and
escalate to the higher-peak synthetic arms if the absolute number matters. The synthetic arms
are the uncertain half: whether XLA:CPU actually forms a *fusion* over these rank-11/12 values, or
peels the tiny dots off into library calls that never reach the tiled emitter, is exactly what the
measurement decides. If the synthetic arms are fast while `xtile_issue` hangs, the trigger needs
more than high rank alone and the note should be believed over the docstring.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

# NO PLATFORM PIN, DELIBERATELY. An earlier revision called
# `jax.config.update("jax_platforms", "cpu")` here on the strength of the issue text. On this build
# that pin selects the ONE backend where the bug has been fixed (kMaxRank=8, PR #41174) and would
# have reported a 0.3 s null for a case that takes 3.3 s against a 0.18 s control on GPU. Platform
# selection belongs to the harness's `--platform`, which labels every number with the backend it
# came from; a case module must never quietly decide that for the corpus.
PLATFORM_PIN = "none (harness decides; see docstring -- GPU is where this reproduces)"

_rng = np.random.default_rng(42)
# Plain numpy at module scope, so importing this file to discover CASES never claims a device.
_P = tuple(_rng.normal(size=(2, 2, 2)).astype("float32") for _ in range(10))
_H = tuple(_rng.normal(size=(2, 2, 2, 2)).astype("float32") for _ in range(10))
# Source tensors for the synthetic arms. 24 of them so a 22-op arm never has to reuse one twice.
_T = tuple(np.random.default_rng(7).normal(size=(2, 2, 2)).astype("float32") for _ in range(24))


# ------------------------------------------------------------------ arm 1: the reporter's program

def _issue(p, h):
    """xla#41173 verbatim. Peak intermediate rank 12; first rank>8 at op 9."""
    td = jnp.tensordot
    a1 = td(p[7], p[8], ((1,), (0,)))
    a2 = td(a1, p[9], ((2,), (0,)))
    a3 = td(p[5], p[6], ((1,), (0,)))
    a4 = td(a2, a3, ((0,), (2,)))
    a5 = td(p[1], p[2], ((1,), (0,)))
    a6 = td(a5, p[0], ((0,), (1,)))
    a7 = td(p[3], p[4], ((1,), (0,)))
    a8 = td(a6, a7, ((1,), (0,)))
    a9 = td(a8, a4, ((2, 5), (2, 4)))                        # rank 10 -- the cliff
    a10 = td(h[1], h[2], ((1,), (0,)))
    a11 = td(h[0], h[9], ((0,), (1,)))
    a12 = td(a11, a10, ((0,), (0,)))
    a13 = td(a9, a12, ((0, 1, 2, 7), (5, 8, 0, 3)))          # rank 12 -- the peak
    a14 = td(h[7], h[8], ((1,), (0,)))
    a15 = td(a13, a14, ((2, 3, 7), (1, 4, 3)))
    a16 = td(h[5], h[6], ((1,), (0,)))
    a17 = td(h[3], h[4], ((1,), (0,)))
    a18 = td(a17, a16, ((3,), (0,)))
    a19 = td(a15, a18, ((0, 1, 2, 3, 7, 9), (1, 3, 5, 8, 0, 7)))
    a20 = td(p[5], p[6], ((1,), (0,)))
    a21 = td(a20, p[4], ((0,), (1,)))
    a22 = td(p[7], p[8], ((1,), (0,)))
    a23 = td(a21, a22, ((1,), (0,)))
    a24 = td(a19, a23, ((4, 5, 7, 8, 9), (4, 6, 3, 0, 1)))
    a25 = td(a24, p[9], ((1, 6), (2, 0)))
    a26 = td(a25, p[0], ((0, 5), (2, 0)))
    a27 = td(a26, p[1], ((0, 4), (2, 0)))
    a28 = td(a27, p[2], ((0, 3), (2, 0)))
    return td(a28, p[3], ((0, 1, 2), (2, 1, 0)))


def _rank3_chain(p, h):
    """THE SCORED CONTROL: 30 tensordots over the same tensors that never leave rank 3.

    Every step contracts two axes of a rank-3 accumulator against two axes of a rank-4 `h`, so the
    result is rank (3-2)+(4-2) = 3 and every intermediate in the program is rank 3, against the
    issue program's peak of 12. (`max_intermediate_rank` reports 4 for this arm, not 3, because it
    scans operands too and the `h` tensors are rank 4; no VALUE PRODUCED here exceeds rank 3.)
    It has MORE tensordots than the arm it controls (30 vs 29), the same
    operands, the same dtype and the same primitive, so every explanation except rank is closed off.
    Measured at 0.179 s on GPU where `_issue` takes 3.319 s.
    """
    acc = p[0]
    for k in range(30):
        acc = jnp.tensordot(acc, h[k % 10], ((1, 2), (0, 1)))
    return acc


def _issue_lowrank_subtrees(p, h):
    """Same tensors, same contractions, truncated to the subtrees that stay at rank <= 7.

    The twelve omitted tensordots are precisely those producing a rank>8 result. Each surviving
    leaf is reduced with `.sum()` so nothing is dead-code-eliminated; the trailing sums are rank-0
    and cannot themselves reach the emitter's bad path.
    """
    td = jnp.tensordot
    b1 = td(p[7], p[8], ((1,), (0,)))
    b2 = td(b1, p[9], ((2,), (0,)))
    b3 = td(p[5], p[6], ((1,), (0,)))
    b4 = td(b2, b3, ((0,), (2,)))                            # rank 7, was op 4
    b5 = td(p[1], p[2], ((1,), (0,)))
    b6 = td(b5, p[0], ((0,), (1,)))
    b7 = td(p[3], p[4], ((1,), (0,)))
    b8 = td(b6, b7, ((1,), (0,)))                            # rank 7, was op 8
    b10 = td(h[1], h[2], ((1,), (0,)))
    b11 = td(h[0], h[9], ((0,), (1,)))
    b14 = td(h[7], h[8], ((1,), (0,)))
    b16 = td(h[5], h[6], ((1,), (0,)))
    b17 = td(h[3], h[4], ((1,), (0,)))
    b20 = td(p[5], p[6], ((1,), (0,)))
    b21 = td(b20, p[4], ((0,), (1,)))
    b22 = td(p[7], p[8], ((1,), (0,)))
    b23 = td(b21, b22, ((1,), (0,)))                         # rank 7, was op 23
    return (b4.sum() + b8.sum() + b10.sum() + b11.sum()
            + b14.sum() + b16.sum() + b17.sum() + b23.sum())


# ---------------------------------------------- arm 2: peak rank and op count as independent knobs

def _network(ts, peak: int, ncycles: int):
    """Grow to `peak` (+1 rank/op), collapse to rank 3 (-1 rank/op), `ncycles` times.

    Ops per cycle is 2*(peak-3), so a low-peak arm can be given MORE tensordots than a high-peak
    one -- which is the whole point, since it makes op count and peak rank independent.
    """
    x, k = ts[0], 1
    for _ in range(ncycles):
        while x.ndim < peak:
            x = jnp.tensordot(x, ts[k % len(ts)], ((x.ndim - 1,), (0,)))
            k += 1
        while x.ndim > 3:
            x = jnp.tensordot(x, ts[k % len(ts)], ((x.ndim - 2, x.ndim - 1), (0, 1)))
            k += 1
    return x


def _shape_probe(peak: int, ncycles: int) -> tuple[int, int]:
    """(op count, peak rank) for a synthetic arm, computed with numpy -- no device, no tracing."""
    x, k, ops, mx = _T[0], 1, 0, 3
    for _ in range(ncycles):
        while x.ndim < peak:
            x = np.tensordot(x, _T[k % len(_T)], ((x.ndim - 1,), (0,)))
            k, ops, mx = k + 1, ops + 1, max(mx, x.ndim)
        while x.ndim > 3:
            x = np.tensordot(x, _T[k % len(_T)], ((x.ndim - 2, x.ndim - 1), (0, 1)))
            k, ops = k + 1, ops + 1
    return ops, mx


def _synth(peak: int, ncycles: int, is_control: bool):
    ops, mx = _shape_probe(peak, ncycles)
    tag = "control: " if is_control else ""
    return (
        functools.partial(_network, peak=peak, ncycles=ncycles),
        (_T,),
        f"{tag}xla#41173 synthetic tensordot network, peak rank {mx}, {ops} tensordots, all dims 2",
    )


CASES = {
    "xtile_issue": (
        _issue, (_P, _H),
        "xla#41173 verbatim: 29 tensordots, peak intermediate rank 12; measured 3.319 s on GPU, "
        "0.303 s on CPU (CPU fixed by kMaxRank=8, PR #41174) -- run on GPU",
    ),
    "xtile_issue_control": (
        _rank3_chain, (_P, _H),
        "control: 30 tensordots (MORE ops than the arm), same tensors, contraction order capped at "
        "rank 3; measured 0.179 s on GPU -> 18.5x",
    ),
    "xtile_issue_lowrank_subtrees": (
        _issue_lowrank_subtrees, (_P, _H),
        "probe (unpaired): the 17 tensordots of the ORIGINAL program that stay at rank <= 7, the "
        "12 rank>8 ops dropped -- 32 jaxpr eqns, more than the 29-op arm",
    ),
    # Sweep: peak rank 8 -> 14, with op counts deliberately anti-correlated with peak rank.
    "tilerank_peak9": _synth(9, 2, False),        # 24 ops
    "tilerank_peak10": _synth(10, 1, False),      # 14 ops
    "tilerank_peak10_control": _synth(8, 2, True),   # 20 ops -- MORE ops than the arm it controls
    "tilerank_peak11": _synth(11, 1, False),      # 16 ops
    "tilerank_peak12": _synth(12, 1, False),      # 18 ops
    "tilerank_peak12_control": _synth(8, 3, True),   # 30 ops -- MORE ops than the arm it controls
    "tilerank_peak14": _synth(14, 1, False),      # 22 ops
}


def max_intermediate_rank(name: str) -> int:
    """Peak rank actually reached by `name`, read off its jaxpr.

    The control variable, measured rather than assumed. Tracing only -- no compilation, so this is
    safe to call on the arms that hang. Expect >= 9 for every non-control arm and <= 8 for both
    controls.
    """
    fn, args, _ = CASES[name]
    jaxpr = jax.make_jaxpr(fn)(*args)
    vs = (v for eqn in jaxpr.eqns for v in list(eqn.invars) + list(eqn.outvars))
    return max((len(v.aval.shape) for v in vs
                if hasattr(v, "aval") and hasattr(v.aval, "shape")), default=0)
