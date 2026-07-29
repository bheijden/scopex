"""SYNTHESISED (gap 5: Pallas / Mosaic). K DISTINCT Pallas kernels against K calls of ONE Pallas
kernel: identical HLO instruction count, identical FLOPs, identical output -- and K separate trips
through the Triton/Mosaic compiler instead of one.

No issue URL -- constructed from the pipeline shape, not mined. Every `pallas_call` whose kernel
jaxpr differs is a fresh module for the second compiler: traced, lowered to Triton IR (or Mosaic),
run through its own LLVM/NVPTX pipeline and handed to ptxas. Kernels that hash identically hit
Pallas's in-process cache and are compiled once. So the compile-time variable is the number of
DISTINCT kernels in the program, not the number of kernel INVOCATIONS -- and nothing in the HLO
module distinguishes the two arms below, because both contain exactly K custom calls of the same
shape.

That property is the whole point. This is the Pallas-shaped version of gap 12 (many small modules
versus one big one), and it is the case where per-instruction attribution is guaranteed to be
wrong: the two arms have the same instructions, the same operand shapes, the same arithmetic and
the same runtime. The only thing that differs is a cache hit rate inside a compiler XLA does not
own and does not time.

WHAT EACH ARM ISOLATES.

  * `pallas_kernels_{K}` -- K kernels whose bodies differ ONLY in one baked-in float constant.
    Same body length, same block spec, same grid, same registers. K distinct Triton compilations.
  * `pallas_kernels_{K}_control` -- the same K `pallas_call`s with the constant held fixed, so
    every kernel is byte-identical and Pallas compiles one. The calls are CHAINED (each consumes
    the previous result) so XLA cannot CSE them away and the HLO instruction count is preserved
    exactly; only the kernel cache key changes.
  * `pallas_kernels_interp_{K}` / `_control` -- the identical pair with `interpret=True`. These run
    on CPU. They do NOT measure the same thing: under interpret the kernels become ordinary HLO, so
    the distinct arm becomes K distinct HLO computations and the shared arm becomes K identical
    ones. That is a real and separate cost (HloCSE, DCE computation bookkeeping) and the pair is
    worth having, but a difference there is NOT evidence about Triton. Read it as its own row.

PLATFORM: GPU for `pallas_kernels_*` (Triton). Those arms do not lower on CPU -- they raise,
verbatim, `ValueError: Only interpret mode is supported on CPU backend.` from
`jax/_src/pallas/pallas_call.py:888`, because jax 0.10.2 ships no CPU Pallas backend. That failure
is expected. The `pallas_kernels_interp_*` arms run on CPU.

STATUS: **UNVERIFIED ON GPU.** The GPU belonged to another investigation when this was written and
was not touched.

WHAT WAS MEASURED ON CPU (JAX_PLATFORMS=cpu, jax 0.10.2, x64 on, single shot, fresh process each):

    arm                                 lower_s   compile_s   HLO lines
    pallas_kernels_interp_1               1.070      0.225         129
    pallas_kernels_interp_1_control       1.146      0.181         129
    pallas_kernels_interp_16              2.187      1.364       1,389
    pallas_kernels_interp_16_control      2.275      0.601       1,389
    pallas_kernels_interp_64              3.833      1.865       5,421
    pallas_kernels_interp_64_control      6.809      3.682       5,421

The HLO instruction counts are IDENTICAL between arm and control at every K, which is the structural
property the file is built on and is now confirmed rather than assumed. The compile times are not
separated: 2.3x at K=16, 0.5x (inverted) at K=64. So under `interpret=True` -- i.e. with the second
compiler removed -- distinct versus shared kernels is NOISE on CPU at these sizes. That is the
correct null result for the interpret rows and it is what makes the GPU rows worth running: if the
GPU pair separates, the separation cannot be attributed to the HLO, because the HLO is the same
size in both arms and does not separate here.

WHAT WOULD FALSIFY IT. If `pallas_kernels_{K}` and its control compile in the same time on GPU,
Pallas is either caching on something coarser than the kernel jaxpr, or the per-kernel constant is
negligible against the launch scaffolding. Either answer is worth having written down, and both are
invisible from HLO.

SIZES. K sweeps 1 -> 64. K=1 pins the per-call constant that both arms share; the slope in K is the
per-DISTINCT-kernel cost. Arrays are 256x256 float32, so runtime is microseconds at every K.

NUMPY at module scope; no device is touched at import.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

ROWS = 256
BLK = 256

# float32 spelled out: the harness enables x64 globally and f64 would change the kernel's register
# budget, which is not the variable under test here.
_SCALE = np.float32(1.0009765625)


def _kernel(x_ref, o_ref, *, bias):
    """Body length, block shape and grid are fixed; `bias` is the only thing that varies.

    `bias` is a python float closed over at trace time, so it lands in the kernel jaxpr as a
    literal. Two kernels with different biases hash differently and are compiled separately; two
    with the same bias are one kernel to Pallas.
    """
    o_ref[...] = jnp.sin(x_ref[...]) * _SCALE + np.float32(bias)


def _one_call(y, bias, interpret):
    spec = pl.BlockSpec((1, BLK), lambda i: (i, 0))
    return pl.pallas_call(
        functools.partial(_kernel, bias=bias),
        grid=(ROWS,),
        in_specs=[spec],
        out_specs=spec,
        out_shape=jax.ShapeDtypeStruct(y.shape, y.dtype),
        interpret=interpret,
    )(y)


def _chain(x, k, distinct, interpret):
    """K chained pallas_calls. `distinct` decides whether the K kernels are K modules or one."""
    y = x
    for i in range(k):
        # 2^-10 steps: representable exactly in f32, so the arms differ in the kernel constant and
        # in nothing else numerically interesting.
        bias = 0.5 + (i / 1024.0 if distinct else 0.0)
        y = _one_call(y, bias, interpret)
    return y


_X = np.zeros((ROWS, BLK), dtype=np.float32)

# 1 pins the constant both arms share; 64 is where a per-distinct-kernel cost of even 0.1 s would
# be unmistakable against it.
KS = (1, 4, 16, 64)

CASES = {}
for _k in KS:
    CASES[f"pallas_kernels_{_k}"] = (
        functools.partial(_chain, k=_k, distinct=True, interpret=False), (_X,),
        f"synthesised gap-5: {_k} DISTINCT Pallas/Triton kernels chained -- {_k} separate trips "
        f"through the Triton pipeline; GPU-ONLY, does not lower on CPU, UNVERIFIED on GPU",
    )
    CASES[f"pallas_kernels_{_k}_control"] = (
        functools.partial(_chain, k=_k, distinct=False, interpret=False), (_X,),
        f"control: the same {_k} chained pallas_calls with a byte-identical kernel -- same HLO "
        f"instruction count, same FLOPs, one Triton compilation instead of {_k}",
    )

    CASES[f"pallas_kernels_interp_{_k}"] = (
        functools.partial(_chain, k=_k, distinct=True, interpret=True), (_X,),
        f"CPU-runnable twin: {_k} distinct kernels under interpret=True -- measures K distinct HLO "
        f"computations, NOT the Triton pipeline; read as its own row",
    )
    CASES[f"pallas_kernels_interp_{_k}_control"] = (
        functools.partial(_chain, k=_k, distinct=False, interpret=True), (_X,),
        f"control: {_k} identical kernels under interpret=True",
    )
