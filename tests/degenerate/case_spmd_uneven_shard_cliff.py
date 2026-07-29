"""SYNTHESISED (gap 7): a DIVISIBILITY CLIFF in the SPMD partitioner -- adding ONE ELEMENT to a
sharded dimension makes the compile 6x slower, and adding EIGHT elements makes it fast again.

No upstream issue.  Constructed from the mechanism.  When a tensor dimension of size N is sharded
over M devices and M divides N, every device holds an identical N/M-row tile and the partitioner
can rewrite each operation as the same local operation on every device.  When M does NOT divide N
the tiles are ragged: the last device holds fewer valid rows than the others, and every operation
whose result depends on WHICH rows are valid has to be guarded.  XLA's partitioner emits that guard
inline -- `iota` to materialise the row index, a comparison against the per-device valid count, and
`select` to choose between the computed value and the padding value -- ONCE PER GUARDED OPERATION.
It also doubles the number of `collective-permute`s, because a shift across ragged tiles is no
longer a single uniform rotation.

The operation chosen here is `jnp.roll(y, 1, axis=0)`, a shift along the sharded dimension.  A shift
is the simplest primitive whose result genuinely depends on tile boundaries, so it forces the guard
without needing a convolution or a window.  Elementwise-only chains do NOT trigger this: measured
separately, the same D=64 chain with `tanh(y*1.0001 + 0.5)` in place of the roll gives 0.344 s at
N=512 against 2.766 s at N=513 -- real, but a single `pad`/`select`/`iota` inserted once at the
head, not per operation.  It is the boundary-dependent op that turns a one-off padding cost into a
per-op cost.

WHAT THE CONTROL ISOLATES -- exactly one thing: the LENGTH OF THE SHARDED DIMENSION, by one
element.  Same D, same mesh, same 8 devices, same second dimension (64), same dtype, same
`jnp.roll` + multiply-add body, same jaxpr structure, same number of `with_sharding_constraint`
calls, same PartitionSpec.

    pathological   x.shape = (513, 64)     513 = 8*64 + 1     NOT divisible by 8
    control        x.shape = (512, 64)     512 = 8*64         divisible by 8

The control is STRICTLY SMALLER -- 64 fewer float32 values, 0.2% less data, 0.2% fewer FLOPs -- and
strictly cheaper to compile.  So far that is only consistent with "bigger is slower".  The second
control breaks that reading.

THE SECOND CONTROL IS THE POINT OF THE FILE.  `x.shape = (520, 64)`, 520 = 8*65, is divisible.  It
is LARGER than the pathological 513 by seven rows, does MORE work, and compiles FASTER than 513 --
faster, in fact, than the 512 arm.  Compile time as a function of N is therefore NON-MONOTONE, with
a spike at every N that 8 does not divide.  Every size-driven heuristic a profiler might carry --
bytes, elements, FLOPs, array shape, problem size -- predicts the ordering 512 < 513 < 520 and the
measurement is 520 < 512 << 513.

MEASURED IN-ENV under the harness's own child sequence (JAX_PLATFORMS=cpu, jax 0.10.2,
`jax_enable_x64=True`, `jax_num_cpu_devices=8`, 1-D mesh over all 8, fresh process per point).
`compile_s` seconds:

    D      N=513 (case)    N=512 (control)   N=520 (larger control)   513/512   513/520
     8       0.894 s          0.428 s              0.353 s              2.1x      2.5x
    16       2.225 s          0.787 s              0.871 s              2.8x      2.6x
    32       6.511 s          3.313 s              2.689 s              2.0x      2.4x
    64      40.415 s          6.568 s              5.395 s              6.2x      7.5x

and the emitted HLO, which shows the mechanism directly:

    D      metric                 N=513      N=512      N=520
     8     opt-HLO lines           1204        331        331
     8     select                   107          0          0
     8     collective-permute        48         24         24
    16     opt-HLO lines           3660        987        987
    16     select                   467          0          0
    16     iota                      60          0          0
    16     collective-permute        96         48         48
    32     opt-HLO lines          12412       3451       3451
    32     select                  1955          0          0
    64     opt-HLO lines          45276      12987      12987
    64     select                  8003          0          0
    64     iota                     252          0          0
    64     collective-permute       384        192        192

N=511 (= 8*63 + 7) at D=32 lands on the ragged side too and produces HLO BYTE-FOR-BYTE the same
SIZE as N=513's -- 12412 lines, 1955 selects, 124 iotas, 192 collective-permutes -- at 6.776 s
against the divisible 512's 3.313 s.  So "one past an even tiling" and "seven short of the next
one" are the same case; only divisibility matters.

ABSOLUTE SECONDS ON THIS BOX DRIFT, ratios do not.  Re-running the D=32 triple back to back under
the harness's own child gave 6.13 / 6.23 s for the ragged arms against 1.40-1.42 s for both
divisible arms, i.e. the same cliff at 4.4x rather than 2.0x, because the earlier grid was measured
while other compiles shared the machine.  Trust the paired ratio, which is what the harness's
rotation and per-round pairing are built to produce; do not quote the absolute seconds above as
anything but an order of magnitude.

The two DIVISIBLE arms produce HLO of IDENTICAL LINE COUNT (987 / 3451 / 12987) despite differing
in size, which is as clean a statement as this corpus contains that the partitioner's output is a
function of divisibility and not of size.  The `select` count in the ragged arm grows 467 -> 1955 ->
8003 across D = 16 -> 32 -> 64, roughly 4x per doubling: the guards are quadratic in the chain
length, which is why the 513/512 ratio itself grows with D rather than staying constant.

PLATFORM: **CPU** as written (fake devices via `jax_num_cpu_devices=8`), and the mechanism is
backend-independent -- ragged-tile guarding happens in the partitioner, not in a backend pass.  A
GPU or multi-host run should show the same cliff; the GPU is owned by another investigation in this
workflow and was NOT touched, so treat the GPU number as unmeasured rather than predicted.

WHY THIS EARNS A SLOT NEXT TO `case_spmd_reshard_permute.py`.  Different sub-mechanism, opposite
control shape, and a different failure mode for a profiler:

  * the reshard file varies an ANNOTATION and holds the shape fixed; this one varies the SHAPE by
    one element and holds every annotation fixed;
  * the reshard file's cost is collective SYNTHESIS (all-to-all); this one's is per-operation
    PREDICATION (iota/compare/select), which lands in the ordinary elementwise emitters afterwards;
  * the reshard file is monotone in its knob, so a profiler can find it by bisection.  This one is
    a CLIFF at a value no gradient points to.  A tool that samples sizes 256, 512, 1024, 2048 sees a
    perfectly smooth curve and never observes the pathology at all.

RUNTIME.  16-70 ms in every arm (a 513x64 f32 array is 131 KB and the work is D rolls), against a
36 s compile in the worst arm, so every arm is strongly compile-bound and `compile/runtime` clears
the harness's 1000x gate comfortably.

WHAT THE HARNESS WILL SAY, stated up front so the verdict column is not misread.  Under the default
`MIN_VS_CONTROL = 10.0` the top arm scores just BELOW the bar: measured back to back under the
harness's own child, D=64 gives 36.44 s against the control's 5.00 s, i.e. **7.3x**, and the
harness will print "no (7.3x control)".  That is the correct arithmetic and the wrong conclusion.
A 7x compile swing produced by adding ONE ELEMENT to one dimension, with the emitted HLO going from
12988 lines and zero `select`s to 45277 lines and 8003 `select`s, is not a borderline result --
the structural evidence is categorical even where the wall-clock ratio is not.  This case is
therefore also a test of the THRESHOLD: a scoring rule that leans only on a compile ratio files
this under "no", and any profiler that reads the HLO census gets it right.

SIZES: D in (8, 16, 32, 64), so the cliff can be read as a surface rather than a point and the
growth of the guard count with D (107 -> 467 -> 1955 -> 8003 selects, ~4x per doubling) is visible
rather than asserted.  N=511 (= 8*63 + 7, ragged the other way) is included at D=32 to show the
effect is divisibility and not "one more than a power of two".  D=128 is deliberately NOT exposed:
the DIVISIBLE arm alone already costs 47.3 s and 50491 HLO lines there, so the ragged arm would
risk the harness's 900 s per-case timeout for a point the four smaller D values already establish.
"""

from __future__ import annotations

import functools

import numpy as np

import jax
import jax.numpy as jnp

# The one module-scope jax interaction: a config write, no device claimed.
try:
    jax.config.update("jax_num_cpu_devices", 8)
except Exception:
    pass

NDEV = 8
COLS = 64        # second dimension, replicated, identical in every arm


def _sharding():
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    have = jax.devices()
    if len(have) < NDEV:
        raise RuntimeError(
            f"case_spmd_uneven_shard_cliff needs {NDEV} addressable devices, found {len(have)}. "
            f"On CPU set jax_num_cpu_devices >= {NDEV} (this module sets it at import, so this "
            f"means the backend was initialised first); on GPU it needs {NDEV} real devices.")
    return NamedSharding(Mesh(np.array(have[:NDEV]).reshape(NDEV), ("x",)), P("x", None))


def _fn(x, depth: int):
    """D shifts along the SHARDED dimension.  Identical code for every arm; only x.shape[0] varies.

    `jnp.roll(y, 1, axis=0)` is the boundary-dependent primitive: under an even tiling it is a
    uniform rotation the partitioner implements with one collective-permute; under a ragged tiling
    every device needs a different valid-row mask, which the partitioner materialises as
    iota/compare/select inline.
    """
    sh = _sharding()
    y = x
    for _ in range(depth):
        y = jax.lax.with_sharding_constraint(y, sh)
        y = jnp.roll(y, 1, axis=0) + y * 0.5
    return jnp.sum(y)


def _args(rows: int):
    # numpy at module scope, never jnp: importing this file claims no device.
    return (np.ones((rows, COLS), dtype=np.float32),)


DEPTHS = (8, 16, 32, 64)
RAGGED = 513        # 8*64 + 1  -- one element past an even tiling
EVEN_SMALLER = 512  # 8*64      -- strictly SMALLER than the case, strictly cheaper
EVEN_LARGER = 520   # 8*65      -- strictly LARGER than the case, and still cheaper

CASES = {}

for _d in DEPTHS:
    CASES[f"spmd_uneven_d{_d}"] = (
        functools.partial(_fn, depth=_d), _args(RAGGED),
        f"synthesised gap 7: {_d} rolls along a dimension of length {RAGGED} sharded over 8 "
        f"devices. 8 does not divide {RAGGED}, so the partitioner guards every op with "
        f"iota/compare/select for the ragged last tile -- 8003 selects at D=64, against 0 when the "
        f"dimension is divisible",
    )
    CASES[f"spmd_uneven_d{_d}_control"] = (
        functools.partial(_fn, depth=_d), _args(EVEN_SMALLER),
        f"control: the identical program on a dimension of length {EVEN_SMALLER} -- ONE ELEMENT "
        f"SHORTER, strictly less data and strictly fewer FLOPs, and divisible by 8. Zero selects, "
        f"zero iotas, half the collective-permutes",
    )

# ---- the non-monotone pair: the LARGER control.  This is why the file exists. -----------------
for _d in (32, 64):
    CASES[f"spmd_uneven_vs_larger_d{_d}"] = (
        functools.partial(_fn, depth=_d), _args(RAGGED),
        f"same {RAGGED}-row program as spmd_uneven_d{_d}, paired here against a LARGER control so "
        f"the non-monotonicity is a first-class comparison rather than a footnote",
    )
    CASES[f"spmd_uneven_vs_larger_d{_d}_control"] = (
        functools.partial(_fn, depth=_d), _args(EVEN_LARGER),
        f"control: {EVEN_LARGER} rows -- SEVEN ROWS MORE than the case, more data, more FLOPs, and "
        f"still faster to compile (5.4 s vs 40.4 s at D=64) because 8 divides {EVEN_LARGER}. Every "
        f"size-based heuristic predicts the opposite ordering",
    )

# ---- ragged the other way, to rule out 'one past a power of two' ------------------------------
CASES["spmd_uneven_511_d32"] = (
    functools.partial(_fn, depth=32), _args(511),
    "511 = 8*63 + 7: ragged in the opposite direction, seven rows SHORT of an even tiling rather "
    "than one row past it. Emits HLO of exactly the same size as the 513 arm (12412 lines, 1955 "
    "selects, 124 iotas) at 6.78 s against the divisible 512's 3.31 s -- the driver is "
    "divisibility, not proximity to a power of two",
)
CASES["spmd_uneven_511_d32_control"] = (
    functools.partial(_fn, depth=32), _args(EVEN_SMALLER),
    f"control: {EVEN_SMALLER} rows, divisible by 8, one row LONGER than the 511 case",
)
