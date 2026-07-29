"""SYNTHESISED (gap 11: PROGRAM DEPTH vs WIDTH). A NEGATIVE RESULT, committed deliberately.

    PLATFORM: CPU (measured here, jax 0.10.2 / jaxlib 0.10.2, JAX_PLATFORMS=cpu).
    GPU UNMEASURED -- the GPU was owned by another investigation when this was written.
    Read the CPU verdict as a statement about XLA:CPU only.

No issue URL. There is no bug report behind this file. It is constructed from the audit's
observation that "the entire existing suite is DEEP (serial chains). Nothing is WIDE: thousands of
mutually independent ops." This file closes that hole by making WIDTH the only free variable in a
program whose op count is held EXACTLY constant, and then reports what happened: on XLA:CPU,
nothing. Raw graph width, by itself, is free.

WHY A NULL RESULT IS WORTH A SLOT. The corpus exists to score a profiler. A profiler that
attributes compile cost by counting graph properties will happily produce a number for "this graph
is 2048 ops wide"; this file is the pair that says that number must be ZERO. It is the negative
control for the whole width axis, and it is the arm that catches a tool which has learned "wide
means expensive" from case_width_fanout_llvm_codegen.py in this directory -- which is ALSO a
width case at fixed op count, and which does reproduce at ~4x. The difference between the two is
not how wide the graph is but whether the wide part shares operands: here the W chains share only
the parameter and cost nothing, there the P consumers each read a different producer and cost 4x.
"Width" is not one variable, and these two files are what separate its components.

THE CONSTRUCTION -- op count is exactly equal, not approximately.

    W chains, each of length D, then a left fold of adds over the W chain ends.

    ops  =  W*D  +  (W - 1)        and we choose   D = TOTAL/W - 1
         =  W*(TOTAL/W - 1) + W - 1
         =  TOTAL - W + W - 1
         =  TOTAL - 1                                    for EVERY W dividing TOTAL.

With TOTAL = 8192 every arm has exactly 8191 jaxpr equations, the same single primitive
(`jnp.maximum` against a distinct f32 constant), the same operand shapes, the same FLOPs and the
same output. W sweeps 1 -> 2048, i.e. from ONE chain of 8191 ops (maximally deep, ILP 1) to 2048
independent chains of 3 (maximally wide, ILP 2048). Nothing else moves. Equation counts were
verified equal at every W, not assumed:

    W=1  8191 eqns    W=16  8191    W=256  8191    W=2048  8191

The distinct per-op constant is load-bearing twice over. It stops CSE from collapsing the W
identical chains into one, and it stops the algebraic simplifier from folding the chain: the same
sweep written with `t = t * c` collapses to 31 HLO lines at W=1 (measured) because
mul(mul(x,a),b) -> mul(x,a*b) telescopes a chain but cannot telescope across chain boundaries --
which would have made the arms structurally different rather than merely differently shaped.

MEASURED, JAX_PLATFORMS=cpu, jax 0.10.2, x = (64,) f32, TOTAL = 8192 (8191 eqns in every arm):

    W       D      compile s    HLO lines
       1   8191      11.805        24601
      16    511      10.507        24576
     256     31      11.748        24096
    2048      3      13.732        20512

Flat. 1.16x end to end across a 2048x change in width, against a 3.4x spread between repeat
measurements of neighbouring arms elsewhere in this run -- i.e. inside the noise. Note also that
the widest arm has 17% FEWER HLO lines than the deepest, so even the sign of any "bigger module"
argument points the wrong way.

A second family was measured and is NOT shipped, because it is not op-count matched: adding one
`lax.optimization_barrier` at each chain end (W extra ops, so 8192 / 8207 / 8447 / 10239 eqns)
forces the W chains to materialise instead of fusing into one kernel. It was equally flat --
13.143 / 8.077 / 13.508 / 14.589 s for W = 1 / 16 / 256 / 2048. So the null is not an artifact of
everything collapsing into a single fusion; breaking the fusions does not create a width cost
either.

WHAT THIS BOUNDS, AND WHAT IT DOES NOT. It bounds XLA:CPU's scheduling, fusion and buffer
assignment: none of them charge for instruction-level parallelism alone at 8k ops. It does NOT
bound (a) GPU, where multi-output fusion sibling merging is a documented exponential and width is
exactly its trigger; (b) width with DISTINCT operands per branch, which is a different program and
does reproduce -- see case_width_fanout_llvm_codegen.py; (c) widths far beyond 8k. TOTAL is capped
at 8192 because the shared baseline is already ~12 s per arm and 32768 would put a null result
over the harness's 900 s timeout for no additional information.

HOW THE HARNESS WILL SCORE IT. `widthdepth_w2048` is above the 3 s floor and will be compared
against `widthdepth_w2048_control`, which is the W=1 program -- byte-identical op count, opposite
shape. Expect "no (1.2x control)". THAT IS THE CORRECT READING and the file is doing its job when
it prints it. The control arm is deliberately the same program for every W so that all four
ratios are measured against one fixed reference.
"""

from __future__ import annotations

import functools

import jax.numpy as jnp
import numpy as np

# NUMPY at module scope. Materialising a jax array here would claim a device at import, before the
# harness has chosen one.
_M = 64
_X = np.linspace(0.3, 0.9, _M).astype(np.float32)

# TOTAL must be a power of two so every W in the sweep divides it exactly; that exactness is what
# makes the op count identical rather than merely similar.
_TOTAL = 8192
# Distinct per-op constants: they defeat CSE across chains and defeat the algebraic simplifier's
# telescoping of a constant chain. Never a jnp array -- see the module-scope rule above.
_C = np.linspace(0.5, 1.5, _TOTAL + 8).astype(np.float32)

# W = 1 is the pure serial chain (the shape the rest of the corpus already covers).
# W = 2048 is 2048 mutually independent chains of length 3 (the shape nothing covers).
_WS = (1, 16, 256, 2048)


def _wide(x, W: int):
    """W independent chains of length TOTAL/W - 1, folded together. Exactly TOTAL-1 ops."""
    D = _TOTAL // W - 1
    outs = []
    k = 0
    for _ in range(W):
        t = x
        for _ in range(D):
            # One primitive, one distinct constant operand. Shapes never change.
            t = jnp.maximum(t, float(_C[k]))
            k += 1
        outs.append(t)
    return functools.reduce(lambda a, b: a + b, outs)


CASES = {}
for _w in _WS:
    _d = _TOTAL // _w - 1
    CASES[f"widthdepth_w{_w}"] = (
        functools.partial(_wide, W=_w),
        (_X,),
        f"gap 11: {_w} independent chains of {_d} ops, exactly {_TOTAL - 1} eqns "
        f"(measured FLAT on CPU -- this is a negative result, see docstring)",
    )
    CASES[f"widthdepth_w{_w}_control"] = (
        functools.partial(_wide, W=1),
        (_X,),
        f"fixed reference for w{_w}: ONE serial chain, the same {_TOTAL - 1} eqns, "
        "same primitive, same shapes, ILP 1 instead of "
        f"{_w}",
    )
