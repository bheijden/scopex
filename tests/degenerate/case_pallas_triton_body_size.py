"""SYNTHESISED (gap 5: Pallas / Mosaic). Compile cost of a `pallas_call` is paid by the KERNEL
BODY, in a compiler XLA does not own, and is invariant in the grid. The control is the identical
arithmetic written as ordinary `jnp` ops, which goes through XLA instead.

No issue URL -- constructed from the pipeline shape, not mined. A `pallas_call` is a single opaque
custom call as far as the HLO module is concerned: XLA sees one instruction whose cost model,
fusion decisions and pass timings tell it nothing about what is inside. The body is compiled by a
SEPARATE toolchain -- Triton -> LLVM -> NVPTX -> ptxas on GPU, Mosaic -> LLO on TPU -- and every
second spent there is invisible to any profiler that reads XLA HLO pass timings, jaxpr size, or
instruction counts. That is the entire reason this gap is worth closing: it is a compile-time
budget with no XLA-side representation at all.

WHAT EACH ARM ISOLATES.

  * `pallas_body_d{D}` vs `pallas_body_d{D}_control` -- the SAME D-op elementwise chain, once
    inside a Pallas kernel and once as plain `jnp`. Identical arithmetic, identical FLOPs,
    identical output array, identical dtype. One goes through Triton/Mosaic, the other through
    XLA's own emitters. The ratio between the two arms at fixed D is a direct read of the two
    pipelines' per-op compile cost against each other, and D sweeps it so a constant offset can be
    told apart from a slope.
  * `pallas_grid_wide` vs `pallas_grid_wide_control` -- the SAME kernel body at the SAME total
    FLOPs, launched as ROWS programs of one row each versus ONE program covering all ROWS rows.
    The grid is a launch dimension, resolved at run time; the block shape is a compile-time
    constant that changes register and shared-memory demand. Prediction: near-equal compile time,
    with any difference attributable to the block shape rather than to the grid extent. If a
    profiler reports the wide-grid arm as more expensive it is confusing a runtime dimension for a
    compile-time one.
  * `pallas_interp_d{D}` -- the SAME kernel with `interpret=True`, which is one keyword and
    replaces the entire Triton/Mosaic pipeline with an HLO lowering of the same kernel. This is the
    tightest control available anywhere in this file: the kernel source, the grid, the block specs
    and the arithmetic are byte-identical, and the only thing that changes is WHICH COMPILER runs.
    It is also the only arm of the pathological family that runs on CPU, which is why it exists.

PLATFORM: GPU for the `pallas_body_*` and `pallas_grid_wide`/`pallas_grid_wide_control` arms
(Triton). Those arms fail to lower on CPU with, verbatim,

    ValueError: Only interpret mode is supported on CPU backend.
      jax/_src/pallas/pallas_call.py:888, cpu_lowering

-- jax 0.10.2 ships no CPU Pallas backend, only `mosaic_gpu`, `triton`, `tpu` and `tpu_sc`. That
failure is expected and is not a bug in the file. The `pallas_interp_*` arms and every `*_control`
arm run on CPU.

STATUS: **UNVERIFIED ON GPU.** The GPU was owned by another investigation when this was written and
was not touched. The numbers to look for on GPU are (a) whether `pallas_body_d{D}` grows faster in D
than its XLA control, and (b) whether `pallas_grid_wide` and its 256x-smaller-grid control agree.

WHAT WAS MEASURED ON CPU (JAX_PLATFORMS=cpu, jax 0.10.2, x64 on, single shot, fresh process each):

    arm                                 lower_s   compile_s   HLO lines
    pallas_interp_d1                      1.702      0.666         125
    pallas_interp_d1_control              0.549      0.203          38
    pallas_interp_d16                     1.597      0.277         170
    pallas_interp_d16_control             1.222      0.753          83
    pallas_interp_d256                    8.126      3.099         890
    pallas_interp_d256_control            1.196      7.490         803
    pallas_grid_wide_interp               1.793      0.631         314
    pallas_grid_wide_interp_control       3.538      1.139         234

Two things fall out of the CPU rows and both are worth knowing before the GPU run.

  * **Pallas LOWERING is expensive and grows with body depth even when no kernel compiler runs at
    all.** 1.70 s at D=1, 1.60 s at D=16, 8.13 s at D=256, against a flat ~1.2 s for the plain-jnp
    control. That cost is python-side, before any backend, and it is present in the GPU arms too --
    so a GPU measurement that only looks at total compile time will silently include it. The
    harness times `lower` and `compile` separately, which is the discriminator.
  * **Under interpret the Pallas arm COMPILES FASTER than the plain chain at D=256** (3.10 s vs
    7.49 s), because the interpreter puts the body inside a grid loop instead of handing XLA one
    768-equation fusion. Same arithmetic, same output, inverted verdict. Do not read this as
    evidence about Triton -- it is evidence that "went through Pallas" and "was expensive" are
    independent.

SIZES. D sweeps 1 -> 256 so a per-kernel constant (flat in D) can be distinguished from per-op
Triton cost (linear in D) from anything superlinear. Arrays are 256x256 float32 -- small enough
that runtime is microseconds and the compile/runtime ratio stays meaningful.

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

# float32 spelled out. The harness enables x64 globally; f64 inside a Pallas kernel changes the
# register budget and therefore the thing being measured.
_SCALE = np.float32(1.0009765625)
_BIAS = np.float32(0.5)


def _chain(y, depth):
    """The payload, shared verbatim by the kernel and by the XLA control."""
    for _ in range(depth):
        y = jnp.sin(y) * _SCALE + _BIAS
    return y


def _kernel(x_ref, o_ref, *, depth):
    o_ref[...] = _chain(x_ref[...], depth)


def _pallas(x, depth, rows_per_block, interpret):
    grid = (ROWS // rows_per_block,)
    spec = pl.BlockSpec((rows_per_block, BLK), lambda i: (i, 0))
    return pl.pallas_call(
        functools.partial(_kernel, depth=depth),
        grid=grid,
        in_specs=[spec],
        out_specs=spec,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=interpret,
    )(x)


def _xla(x, depth):
    """CONTROL: the identical chain as ordinary jnp ops -- same FLOPs, same output, XLA emitters."""
    return _chain(x, depth)


_X = np.zeros((ROWS, BLK), dtype=np.float32)

# 1 isolates the per-kernel constant; 256 is where a per-op Triton cost would dominate it.
DEPTHS = (1, 4, 16, 64, 256)

CASES = {}

for _d in DEPTHS:
    CASES[f"pallas_body_d{_d}"] = (
        functools.partial(_pallas, depth=_d, rows_per_block=1, interpret=False),
        (_X,),
        f"synthesised gap-5: Pallas/Triton kernel with a {_d}-op body -- compiled by Triton, not "
        f"XLA; GPU-ONLY, does not lower on CPU, UNVERIFIED on GPU",
    )
    CASES[f"pallas_body_d{_d}_control"] = (
        functools.partial(_xla, depth=_d), (_X,),
        f"control: the identical {_d}-op chain as plain jnp -- same FLOPs, same output, compiled "
        f"by XLA instead of Triton",
    )

    CASES[f"pallas_interp_d{_d}"] = (
        functools.partial(_pallas, depth=_d, rows_per_block=1, interpret=True),
        (_X,),
        f"same Pallas kernel, {_d}-op body, interpret=True -- one keyword swaps the Triton "
        f"pipeline for an HLO lowering; runs on CPU, tightest control on WHICH COMPILER runs",
    )
    CASES[f"pallas_interp_d{_d}_control"] = (
        functools.partial(_xla, depth=_d), (_X,),
        f"control: the identical {_d}-op chain as plain jnp, depth {_d}",
    )

# Grid axis: same body, same total FLOPs, 256 programs of one row vs 1 program of 256 rows.
_GRID_DEPTH = 64
CASES["pallas_grid_wide"] = (
    functools.partial(_pallas, depth=_GRID_DEPTH, rows_per_block=1, interpret=False),
    (_X,),
    f"grid axis: {ROWS}-program grid, 1 row per block, {_GRID_DEPTH}-op body -- GPU-ONLY, "
    f"UNVERIFIED",
)
CASES["pallas_grid_wide_control"] = (
    functools.partial(_pallas, depth=_GRID_DEPTH, rows_per_block=ROWS, interpret=False),
    (_X,),
    "control: identical kernel and identical total FLOPs as a 1-program grid covering all rows -- "
    "grid extent is a runtime dimension, so these two should compile in the same time",
)
CASES["pallas_grid_wide_interp"] = (
    functools.partial(_pallas, depth=_GRID_DEPTH, rows_per_block=1, interpret=True),
    (_X,),
    f"CPU-runnable twin of pallas_grid_wide: {ROWS}-program grid under interpret=True",
)
CASES["pallas_grid_wide_interp_control"] = (
    functools.partial(_pallas, depth=_GRID_DEPTH, rows_per_block=ROWS, interpret=True),
    (_X,),
    "CPU-runnable twin of pallas_grid_wide_control: 1-program grid under interpret=True",
)
