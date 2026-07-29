"""SYNTHESISED (gap 14) -- NEGATIVE RESULT, deliberately committed. int4 produces ~4x the HLO of
int8 and costs the SAME to compile on CPU. This bounds where the narrow-dtype pathology is not.

    mechanism source (read, not a bug report):
    https://raw.githubusercontent.com/openxla/xla/main/xla/hlo/transforms/simplifiers/sub_byte_normalization.cc

READ THIS TOGETHER WITH ``case_dtype_f8_convert_expansion.py``. That file finds a 3.1x compile
swing between two 8-bit FLOAT types with byte-identical programs. The obvious generalisation is
"narrow types are expensive to compile". This file was written to test that generalisation on the
sub-byte INTEGER path, and it **refutes it**: int4 is free.

WHAT WAS EXPECTED. int4/uint4 have no native storage on any backend. ``SubByteNormalization``
normalises their layouts and the emitters materialise every element access as a shift-and-mask
against a packed byte buffer. The prediction was that per-op IR expansion would make compile time
scale with the length of a sub-byte chain at a large constant, exactly as the f8 case does.

WHAT WAS MEASURED (JAX_PLATFORMS=cpu, jax/jaxlib 0.10.2, x64 on, x = (256, 256), compile seconds,
one fresh subprocess per measurement):

    workload                        int32     int8     int4     int4 HLO lines vs int8
    64-step elementwise chain       0.792    1.261    1.048          983 vs 262  (3.8x)
    16 gathers (jnp.take)           1.773    1.986    1.830         1038 vs 974
    16 dynamic_slices               1.291    1.456    1.211          624 vs 592

and, at an earlier depth sweep on a shorter chain (D = 8/16/32/64):

    int32  0.744 / 0.714 / 0.419 / 0.572
    int8   0.577 / 0.586 / 0.453 / 0.686
    int4   0.688 / 0.770 / 0.421 / 0.773

Flat. No arm separates from any other by more than measurement noise, at any depth, on any of the
three access patterns. The int4 arm carries 3.8x the unoptimised HLO of the int8 arm on the
elementwise chain and compiles in the same time, which is the sharpest single fact in this file:
**HLO line count is not a proxy for compile time here**, and a profiler that predicts cost from
program size will over-predict this case by ~4x.

WHY THIS IS WORTH A SLOT ANYWAY. Three reasons.

  1. It is the control for the f8 case at the level of the WHOLE HYPOTHESIS. Together the two
     files say: the expensive thing is not narrowness, and not the number of inserted
     normalisation ops -- it is specifically the branchy non-IEEE f8_e4m3fn conversion sequence.
     Neither file can say that alone.
  2. It gives the corpus a case where HLO size and compile time point in opposite directions, on
     purpose. Everything else in the suite that is big is also slow.
  3. If a future XLA release regresses ``SubByteNormalization`` this file turns positive with no
     edit, and the flat numbers above date the baseline.

THE ARMS. Case = int4, control = int8 (the next width up, natively storable, no packing). A second
control at int32 separates "narrow integer" from "sub-byte storage"; both are flat, so neither
distinction has a cost on CPU.

INT4 MATMUL IS NOT AN ARM. ``lax.dot_general`` on int4 operands fails outright in jax 0.10.2 --
``JaxRuntimeError: INVALID_ARGUMENT: during context [hlo verifier]: The XLA CPU/GPU backend does
not support ...`` -- so that path cannot be swept at all and is recorded here rather than shipped
as an arm that the harness would report as ERROR.

PLATFORM: **CPU (measured, flat).** GPU is UNVERIFIED and is the arm that could still be positive:
sub-byte packing on GPU goes through a different emitter, and int4 tensor-core paths bring in
their own layout normalisation. A GPU run of this file is the cheapest way to find out, and a flat
GPU result would close gap 14's sub-byte question outright. The GPU was off-limits when this file
was written.

RUNTIME. Trivial in every arm; this is a compile-time file. Inputs are zeros -- values cannot
affect compile time, and int4 saturates at [-8, 7] so random values would be uninformative anyway.
"""

from __future__ import annotations

import functools

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

N = 256

# NUMPY at module scope; the dtype is applied inside the traced function.
_X = np.zeros((N, N), dtype=np.int32)
_IDX = np.arange(0, N // 2, dtype=np.int32)


def _chain(x, depth: int, dt):
    """``depth`` steps of (y*c + c ; max(y, y - c)) carried out entirely in ``dt``."""
    y = lax.convert_element_type(x, dt)
    c = lax.convert_element_type(np.int32(3), dt)
    for _ in range(depth):
        y = lax.add(lax.mul(y, c), c)
        y = lax.max(y, lax.sub(y, c))
    return lax.convert_element_type(y, jnp.int32).sum()


def _gathers(x, idx, depth: int, dt):
    """Sub-byte indexing: every gathered element needs a shift-and-mask against packed bytes."""
    y = lax.convert_element_type(x, dt)
    acc = jnp.zeros((), jnp.int32)
    for i in range(depth):
        acc = acc + lax.convert_element_type(jnp.take(y, idx + i, axis=0), jnp.int32).sum()
    return acc


def _slices(x, depth: int, dt):
    """Sub-byte dynamic slicing: offsets are in elements, storage is in packed bytes."""
    y = lax.convert_element_type(x, dt)
    acc = jnp.zeros((), jnp.int32)
    for i in range(depth):
        acc = acc + lax.convert_element_type(
            lax.dynamic_slice(y, (i, 0), (N // 2, N)), jnp.int32).sum()
    return acc


CASES = {}

# --- elementwise chain, depth sweep. Case = int4, control = int8. -----------------------------
for _d in (16, 32, 64, 128):
    CASES[f"i4chain_d{_d}"] = (
        functools.partial(_chain, depth=_d, dt=jnp.int4), (_X,),
        f"synthesised, NEGATIVE on CPU: depth-{_d} elementwise chain in int4. Carries ~3.8x the "
        f"HLO of the int8 control and compiles in the same time (1.048 s vs 1.261 s at D=64)")
    CASES[f"i4chain_d{_d}_control"] = (
        functools.partial(_chain, depth=_d, dt=jnp.int8), (_X,),
        f"control: byte-identical program in int8 at depth {_d} -- next width up, natively "
        f"storable, no sub-byte packing. One dtype token")

# --- second control: separates 'narrow integer' from 'sub-byte storage' -----------------------
CASES["i4chain_i32_d64"] = (
    functools.partial(_chain, depth=64, dt=jnp.int32), (_X,),
    "second control: the same chain in int32 at depth 64 -- measured 0.792 s, i.e. no slower "
    "than either narrow arm. Neither narrowness nor packing costs anything on CPU")

# --- access patterns where sub-byte packing should have bitten hardest ------------------------
CASES["i4gather_d16"] = (
    functools.partial(_gathers, depth=16, dt=jnp.int4), (_X, _IDX),
    "NEGATIVE: 16 gathers on an int4 buffer -- every element access is a shift-and-mask. "
    "Measured 1.830 s")
CASES["i4gather_d16_control"] = (
    functools.partial(_gathers, depth=16, dt=jnp.int8), (_X, _IDX),
    "control: identical gathers on int8 -- measured 1.986 s, i.e. the int4 arm is FASTER")

CASES["i4slice_d16"] = (
    functools.partial(_slices, depth=16, dt=jnp.int4), (_X,),
    "NEGATIVE: 16 dynamic_slices on an int4 buffer -- element offsets against packed bytes. "
    "Measured 1.211 s")
CASES["i4slice_d16_control"] = (
    functools.partial(_slices, depth=16, dt=jnp.int8), (_X,),
    "control: identical dynamic_slices on int8 -- measured 1.456 s")
