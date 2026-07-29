"""jax#4453 -- lax.switch compile time is super-linear in BRANCH COUNT, even when every branch is
the same one-line function.

    https://github.com/jax-ml/jax/issues/4453     CLOSED as out-of-JAX's-control (hawkinsp
                                                  profiled it and found the majority of the time
                                                  inside LLVM, not inside jax and not inside XLA's
                                                  HLO pipeline)

VERIFIED IN THIS ENVIRONMENT (jax 0.10.2, CUDA, x64) before the case was written:

    branches   lower      compile     run        compile/run
       128     0.199 s     1.308 s    0.92 ms        1421
       512     1.137 s    39.813 s    2.71 ms       14713

512 is 4x the branches of 128 and 30.4x the compile time -- roughly n^2.5. The 512 arm clears all
three bars outright (compile 39.8 s, ratio 14713) with a payload of ONE scalar multiply, so there
are no FLOPs anywhere in this case to confuse a compile-vs-runtime attribution.

WHY THIS EARNS A SLOT NEXT TO THE SCATTER-CHAIN CASE. The corpus's flagship pathology is one
OVERSIZED fusion: a single HLO computation that grows until a fusion/tiling pass chokes on it, and
the in-repo lever that helps it is barrier-splitting -- cut the big computation in half. This case
is the opposite failure and that lever is irrelevant to it. Every one of the n computations here is
two instructions long. Nothing is oversized. The cost is per-computation codegen MULTIPLICITY:
n branch computations plus an n-way conditional are handed to the backend, each is codegen'd
separately, and nothing deduplicates them even though all n are the literally identical body.
Splitting would make it strictly worse. A profiler that reports "your biggest computation is 2
instructions" has said something true and useless.

AND IT IS AN ATTRIBUTION TRAP. All n branches are built from one source line, so every HLO
instruction in the module maps back to the same `x ** 2`. Blaming that line is not an answer -- it
is fast, it appears once, and deleting a copy of it changes nothing. The only useful answer names
the MULTIPLICITY (the length of the branch list), which is a property of the python list
comprehension one line up and of no instruction at all. If scopex can charge cost to "there are 512
of these" rather than to "this is what they do", this case is where that shows.

WHAT EACH CONTROL ISOLATES.

  * `_control` (all pairs) is `lax.switch(i, [body], x)` -- the SAME call to the SAME function with
    the SAME body, differing only in that the branch list has length 1 instead of length n. Since
    all n branches are identical, the two arms are SEMANTICALLY EQUAL: the switch cannot observe
    the index. This is precisely the program a compiler that CSE'd identical computations would
    have ended up with, so the gap between the arms is exactly the cost of the compiler not doing
    that. One list-length edit, nothing else.
  * `switch_ident_{64,128,256,512,1024}` is the scaling axis and doubles as the issue's own control:
    branch count is the only variable across the five, and the in-env 128 -> 512 measurement above
    (30.4x for 4x branches) is what puts this case over the 10x bar without relying on the 1-branch
    arm at all.
  * `opcount_chain_*` is the FALSIFICATION arm and the most important non-obvious entry here. It
    puts n copies of the identical body into ONE computation (an n-long chain of `y = y ** 2`), so
    the instruction COUNT matches the n-branch arm while the computation COUNT is 1. If compile
    time tracked ops, this would be as slow as the switch. The claim of this case is that it will
    be flat -- i.e. the axis is computations, not instructions. It is paired 512-vs-128 so the
    harness prints its scaling next to the switch's; the expectation is ~1x there and ~30x for the
    switch.
  * `switch_distinct_512` makes the branches genuinely different (`x ** 2 + j`, a distinct constant
    per branch) to check that identity is irrelevant -- the mechanism claims nothing dedups, so
    this should cost the SAME as the identical-branch arm. If it costs noticeably more, some
    dedup does exist and the story needs rewriting.
  * `switch_tree_512` is the workaround shape from the issue: a balanced binary tree of 2-way
    switches with 512 leaves, indexed by bit-slicing `i`. EXPECTATION UNCERTAIN and that is the
    point. It has MORE computations than the flat arm (~2n instead of n+1) but none of them is
    wider than 2 branches. If it is fast, the cost is super-linear in the arity of a single
    conditional; if it is just as slow, the cost is linear-ish in total computation count and the
    n^2.5 above comes from somewhere else entirely. Either answer is worth the slot.

STAGE, IF IT REPRODUCES. The 2020 diagnosis put the time in LLVM, below XLA's HLO pass timing.
Confirm rather than assume: compare total compile against XLA's HLO pass profile, and dump with
XLA_FLAGS=--xla_gpu_dump_llvmir. Note that `lower_s` is separately reported by the harness and is
itself super-linear (0.199 s -> 1.137 s), because jax traces each branch; that is jax-side cost and
must not be counted as backend cost.

COST NOTE. `switch_ident_1024` is the expensive entry: at n^2.5 it is ~3.8 minutes, at n^3 ~5.3
minutes, either way inside the harness's 900 s timeout but it dominates this file's wall clock.
Drop it by name if the budget is tight -- 64/128/256/512 already establish the curve.
"""

from __future__ import annotations

import functools

import numpy as np

import jax.lax as lax

# The index is a runtime value, never a python int -- a python int would make `switch` a no-op at
# trace time and there would be no conditional at all. 3 is in range for every branch count here.
_I = np.int32(3)
_X = np.float64(2.0)


def _flat_switch(i, x, nb):
    """jax#4453 verbatim: n branches, every one of them the same one-line body.

    The `_j=j` default is the issue's own spelling. It is not used in the body -- it exists only so
    the comprehension produces n DISTINCT function objects, which is what makes jax trace and emit
    n separate computations instead of reusing one. That is the whole pathology in one line.
    """
    branches = [(lambda y, _j=j: y ** 2) for j in range(nb)]
    return lax.switch(i, branches, x)


def _distinct_switch(i, x, nb):
    """Same shape, but the branches genuinely differ: `+ j` bakes a distinct constant into each.

    Tests whether branch IDENTITY buys anything. The mechanism claims it does not (no dedup), so
    this should land on top of `_flat_switch` at the same n.
    """
    branches = [(lambda y, _j=j: y ** 2 + float(_j)) for j in range(nb)]
    return lax.switch(i, branches, x)


def _one_switch(i, x):
    """CONTROL: the same call to the same function with the same body, branch list length 1.

    Semantically identical to `_flat_switch` at any n, because there all n branches compute the
    same thing, so the index cannot be observed. This is the CSE'd program.
    """
    return lax.switch(i, [lambda y: y ** 2], x)


def _tree_switch(i, x, nb):
    """The issue's suggested workaround: a balanced binary tree of 2-way switches, nb leaves.

    Level k dispatches on bit k of `i`, so the leaf selected is the same one the flat n-way switch
    would have selected. nb must be a power of two. Builds ~2*nb-1 computations, none wider than
    two branches -- the opposite trade from the flat arm.
    """
    depth = nb.bit_length() - 1
    assert 1 << depth == nb, "tree arm needs a power-of-two branch count"

    def build(level):
        if level == 0:
            return lambda y: y ** 2
        lo, hi, bit = build(level - 1), build(level - 1), level - 1

        def node(y, lo=lo, hi=hi, bit=bit):
            return lax.switch((i >> bit) & 1, [lo, hi], y)

        return node

    return build(depth)(x)


def _inline_chain(i, x, nops):
    """FALSIFICATION arm: nops copies of the identical body inside ONE computation.

    Same instruction count as the nops-branch switch, one computation instead of nops+1. `i` is
    accepted and folded in at the end only so both arms take identical arguments and the harness
    lowers them against the same avals.

    The chained value overflows to inf after ~10 squarings. That is irrelevant: `x` is a parameter,
    so nothing is evaluated at compile time, and the harness never inspects the output. Using a
    bounded op instead would change the op MIX, which is the one thing this arm must hold fixed.
    """
    y = x
    for _ in range(nops):
        y = y ** 2
    return y + i


def _mk_switch(nb, note):
    return functools.partial(_flat_switch, nb=nb), (_I, _X), note


CASES = {}

# The scaling axis. Branch count is the only variable across these five.
for _nb in (64, 128, 256, 512, 1024):
    CASES[f"switch_ident_{_nb}"] = _mk_switch(
        _nb,
        f"jax#4453: lax.switch over {_nb} identical `x ** 2` branches "
        f"(in-env: 128 -> 1.31 s, 512 -> 39.81 s compile)",
    )

# Controls only where the arm can plausibly clear the 3.0 s floor -- classify() checks the floor
# first, so a below-floor arm never consults its control and pairing 64/128 would just burn a
# subprocess each. The control is n-independent by construction: it is the 1-branch program.
for _nb in (256, 512, 1024):
    CASES[f"switch_ident_{_nb}_control"] = (
        _one_switch, (_I, _X),
        f"control for n={_nb}: identical call, identical body, branch list of length 1 "
        f"-- semantically equal because all {_nb} branches were the same",
    )

CASES["switch_distinct_512"] = (
    functools.partial(_distinct_switch, nb=512), (_I, _X),
    "512 GENUINELY DIFFERENT branches (`x ** 2 + j`): should match switch_ident_512 if, as the "
    "mechanism claims, nothing dedups identical computations",
)
CASES["switch_distinct_512_control"] = (
    _one_switch, (_I, _X),
    "control: same 1-branch switch",
)

CASES["switch_tree_512"] = (
    functools.partial(_tree_switch, nb=512), (_I, _X),
    "the issue's workaround shape: balanced tree of 2-way switches, 512 leaves, ~2x the "
    "computations of the flat arm but max arity 2 -- EXPECTATION UNCERTAIN, either result informs "
    "whether cost is per-computation or per-arity",
)
CASES["switch_tree_512_control"] = (
    _one_switch, (_I, _X),
    "control: same 1-branch switch",
)

# Falsification: identical instruction count, one computation. Paired 512-vs-128 so its scaling
# prints in the same column as the switch's; expected ~1x against the switch's ~30x.
CASES["opcount_chain_512"] = (
    functools.partial(_inline_chain, nops=512), (_I, _X),
    "falsification arm: 512 copies of the same body in ONE computation -- same instruction count "
    "as switch_ident_512, expected to be flat because the axis is computations, not instructions",
)
CASES["opcount_chain_512_control"] = (
    functools.partial(_inline_chain, nops=128), (_I, _X),
    "control: the same chain at 128 ops, mirroring the 128 -> 512 step that costs the switch 30.4x",
)
