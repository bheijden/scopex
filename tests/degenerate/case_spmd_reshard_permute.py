"""SYNTHESISED (gap 7): the SPMD partitioner manufactures a 55x-larger HLO module from a
BYTE-IDENTICAL StableHLO input, and the whole cost is invisible to anything that reads the jaxpr,
the StableHLO, or the op count.

No upstream issue.  Constructed from the mechanism.  jax 0.10.2 runs Shardy (`jax_use_shardy_
partitioner` defaults to True) followed by XLA's SPMD partitioner; both run whenever a program
carries shardings, REGARDLESS OF HOW MANY DEVICES ARE REAL.  CPU can be given as many addressable
devices as you like -- this file uses `jax.config.update("jax_num_cpu_devices", 64)` -- so an
entire compiler stage that the corpus has never touched becomes testable on a machine with no
accelerator at all.

MECHANISM.  A `with_sharding_constraint` whose PartitionSpec differs from the incoming sharding is
a RESHARD.  The partitioner has to synthesise the data movement that turns one tiling of a tensor
into another.  When the two tilings differ by a PERMUTATION of which mesh axis owns which tensor
dimension, no single collective does it: the partitioner emits a sequence of `all-to-all`s (one per
mesh axis that has to move) wrapped in `bitcast`/`transpose`/`copy` reshaping, plus
`collective-permute`s for the residual device relabeling.  Every one of those instructions is
CREATED BY THE PASS.  Nothing upstream of the partitioner knows they are coming.

    D = number of reshards in the chain          <- swept
    mesh device count                            <- swept (8 vs 64)
    mesh FACTORISATION at fixed device count     <- swept (4x4x4 vs 2x2x2x2x2x2)

WHAT THE CONTROL ISOLATES -- exactly one thing: the PartitionSpec argument.

    pathological   with_sharding_constraint(y, NamedSharding(mesh, spec_for(i)))   i-th permutation
    control        with_sharding_constraint(y, NamedSharding(mesh, spec_for(0)))   always the 0th

Same D, same mesh, same device count, same tensor shape and dtype, same number of
`with_sharding_constraint` calls, same `jnp.sin` chain, same final `jnp.sum`.  The control's
constraints are all satisfied by the incoming sharding after the first one, so every reshard after
the first is a no-op and the partitioner emits nothing.  The case's constraints each demand a
different axis-to-dimension assignment, so every one is a real reshard.

VERIFIED IDENTICAL UPSTREAM OF THE PARTITIONER (JAX_PLATFORMS=cpu, jax 0.10.2), D=32, 64 devices:

                                case arm      control arm
    jaxpr equations                 65             65        <- identical
    StableHLO lines                 72             72        <- identical
    optimised HLO lines           2227             97        <- 23x, all of it made by the pass
    all-to-all                     234              0
    collective-permute              42              0
    bitcast                       1088              0
    transpose                      633              0
    copy                           211              0

That table is the reason this case is in the corpus.  Every pre-partitioning metric -- equation
count, primitive histogram, StableHLO size, source-line attribution, FLOP count -- is EQUAL between
the two arms.  A profiler that ranks by jaxpr size, or that diffs the lowered module, reports
"these two programs are the same" while one of them takes 8x longer to compile at this D and 24x
longer at D=128.  The only signal that separates them is a per-PASS timing that names the SPMD
partitioner, or an HLO census taken AFTER partitioning rather than before.

PLATFORM: **CPU** as written (fake devices via `jax_num_cpu_devices`), and the mechanism is
backend-independent -- the partitioner is not a backend pass.  Re-verified with GSPMD instead of
Shardy (`JAX_USE_SHARDY_PARTITIONER=0`): 3.834 s vs 0.382 s at D=64/64 devices, i.e. the same
10x, so the effect is a property of SPMD partitioning and not of Shardy specifically.  A multi-GPU
or multi-host run should show the same shape; a real GPU run is NOT done here (the GPU is owned by
another investigation) but nothing in the mechanism is CUDA-specific.

MEASURED IN-ENV under the harness's own child sequence (JAX_PLATFORMS=cpu, jax 0.10.2,
`jax_enable_x64=True` as the harness sets it, `jax_num_cpu_devices=64`, one fresh process per
point).  `compile_s` seconds, rank-6 f32 tensor of shape (8,)*6:

  PRIMARY AXIS -- number of reshards D, mesh (4,4,4) = 64 devices, 3 mesh axes:

    D      case      control     ratio    opt-HLO lines    all-to-alls
      8    0.998 s   0.325 s      3.1x        388/74            42
     16    4.053 s   0.978 s      4.1x       1120/82           114
     32    4.574 s   0.573 s      8.0x       2228/98           234
     64   12.018 s   0.811 s     14.8x       5294/130          534
    128   19.087 s   0.799 s     23.9x      10576/194         1074

  Case grows ~linearly in D (1.0 -> 19.1 s over 16x of D) while the control is FLAT (0.33 -> 0.80),
  so the RATIO grows and there is no size at which the two converge.  All-to-alls track D exactly --
  ~8.4 per reshard, which is the 3 mesh axes moving through a rank-6 tensor.  This is a large
  per-reshard CONSTANT, not a superlinear blowup, and the file claims no more than that.

  The same sweep with x64 OFF is uniformly cheaper but has the same shape (0.93 / 1.83 / 2.97 /
  6.35 / 10.42 s case against 0.20 / 0.33 / 0.30 / 0.44 / 0.64 s control), so the pathology is not
  an x64 artefact -- it is roughly 1.8x larger under x64, which is worth knowing given that gap 14
  asks whether these effects survive a dtype change.

  SECOND AXIS -- device count at FIXED program.  D=64, 3 mesh axes, only the mesh size changes:

    mesh (2,2,2)  =  8 devices    4.945 s   (control 0.999 s)    2059 HLO lines,  274 a2a
    mesh (4,4,4)  = 64 devices   12.018 s   (control 0.811 s)    5294 HLO lines,  534 a2a

  Same jaxpr, same StableHLO, same number of reshards; 8x the devices costs 2.4x the compile.  The
  partitioner's output size depends on the device count, which is a compile-time axis with no
  source-level counterpart at all.

  THIRD AXIS -- mesh FACTORISATION at fixed device count, and it is NON-MONOTONE, which is the
  most useful single fact in this file.  D=128, 64 devices in both:

    mesh (4,4,4)          3 axes    19.087 s    10576 HLO lines   1074 all-to-all
    mesh (2,2,2,2,2,2)    6 axes     0.880 s      977 HLO lines      0 all-to-all

  Twenty-two times faster with the SAME 64 devices, the same program, the same op count and MORE
  mesh axes to permute.  With 6 axes of size 2 across a rank-6 tensor every dimension is sharded and
  a permutation of the axes degenerates into a device relabeling the partitioner implements without
  any all-to-all at all.  So "more sharding" is not monotonically "more compile": a profiler
  heuristic that ranks by mesh rank, by number of sharded dimensions, or by sharding-annotation
  count gets the SIGN wrong here.  The 6-axis arm is the fastest thing in the file and is also the
  one that looks most heavily sharded.

WHY THE MESH IS BUILT INSIDE `fn`.  Module scope holds numpy only, per the corpus rule; a `Mesh`
needs `jax.devices()`, which claims a backend.  `_mesh()` is therefore called at TRACE time, and
the only module-scope jax interaction is the `jax_num_cpu_devices` config write, which touches no
device.  If the run has fewer devices than an arm needs, `_mesh()` raises with the required config
rather than silently degrading to a 1-device mesh that would make the case vanish -- an arm that
quietly stops testing anything is worse than an arm that errors.

RUNTIME IS NOT NEGLIGIBLE HERE, and the harness's `compile/runtime >= 1000` heuristic would score
this case WRONG if it were applied.  The tensor is only 8^6 = 262144 f32 = 1 MB and the work is D
`sin`s, but the synthesised all-to-alls are real data movement between 64 fake CPU devices, all of
them multiplexed onto the same physical cores.  Measured first-run times: 0.07 s at D=8, 0.80 s at
D=64, 0.52 s at D=128 for the case arms against 0.07-0.15 s for the controls, so the case arm's
compile/runtime ratio is only ~15-37.  This is precisely the situation the harness's `classify()`
docstring warns about, and it is why THE CONTROL OUTRANKS THE RATIO: `compile_s` is 23.9x the
control's while `runtime_s` is ~7x, and the compile gap is the one that is not explained by the
program doing more work.  The `mesh6d_d128` arm is the extreme version -- 2.85 s of runtime against
0.88 s of compile -- and it is not a pathology at all; it is the FAST arm, and a ratio-based rule
would rank it identically to the 19 s one.

SIZES: D in (8, 16, 32, 64, 128), five points across 16x, plus the two cross-axis arms.  The claim
is about scaling in the number of reshards, so a single D would not distinguish "linear in D with a
big constant" (which is what the data say) from "superlinear in D" (which they do not).

WHICH ARMS CLEAR THE HARNESS BARS (`MIN_COMPILE_S = 3.0`, `MIN_VS_CONTROL = 10.0`): `d64` (12.0 s,
14.8x) and `d128` (19.1 s, 23.9x) clear both.  `d16` and `d32` clear the floor but not the ratio
(4.1x, 8.0x); `d8`, `dev8_d64` and both `mesh6d_d128` arms clear neither.  That spread is
deliberate -- the file is a scaling claim, and the low-D arms exist to show where the effect is
still invisible, which is the information a bisecting profiler needs.
"""

from __future__ import annotations

import functools
import itertools
import math

import numpy as np

import jax
import jax.numpy as jnp

# The one module-scope jax interaction: a config write, no device claimed.  64 is the maximum any
# arm in this file needs; the 8-device arm takes a SUB-MESH of the first 8, so every arm runs under
# the same process-wide device count and the device-count axis is not confounded by it.
try:
    jax.config.update("jax_num_cpu_devices", 64)
except Exception:                       # already-initialised backend, or a non-CPU platform
    pass

RANK = 6        # tensor rank; must be >= number of mesh axes
SIDE = 8        # length of every tensor dimension -> 8^6 = 262144 f32 = 1 MB

# numpy at module scope, never jnp: importing this file claims no device.
_X = np.ones((SIDE,) * RANK, dtype=np.float32)


def _mesh(shape: tuple[int, ...]):
    """Mesh over the FIRST prod(shape) devices, built at trace time.

    Raises rather than degrading: a silently under-provisioned mesh would turn every arm of this
    file into a no-op and the file would report "does not reproduce" for the wrong reason.
    """
    from jax.sharding import Mesh

    need = math.prod(shape)
    have = jax.devices()
    if len(have) < need:
        raise RuntimeError(
            f"case_spmd_reshard_permute needs {need} addressable devices, found {len(have)}. "
            f"On CPU set jax_num_cpu_devices >= {need} (this module sets 64 at import, so this "
            f"means the backend was initialised before import); on GPU this arm needs {need} "
            f"real devices.")
    return Mesh(np.array(have[:need]).reshape(shape), tuple(f"m{i}" for i in range(len(shape))))


def _spec_for(mesh, i: int):
    """The i-th assignment of mesh axes to tensor dimensions.

    Consecutive i give DIFFERENT assignments, so consecutive constraints are reshards.  i=0 always
    gives the same one, which is what the control uses.
    """
    from jax.sharding import PartitionSpec as P

    axes = mesh.axis_names
    perms = list(itertools.permutations(range(RANK), len(axes)))
    spec = [None] * RANK
    for axis, dim in zip(axes, perms[i % len(perms)]):
        spec[dim] = axis
    return P(*spec)


def _fn(x, mesh_shape: tuple[int, ...], depth: int, permute: bool):
    """D `sin`s, each followed by one `with_sharding_constraint`.

    `permute` is the ONLY difference between the arms: True walks through distinct axis-to-dimension
    assignments (every constraint is a reshard), False pins every constraint to the 0th assignment
    (no constraint after the first moves any data).
    """
    from jax.sharding import NamedSharding

    mesh = _mesh(mesh_shape)
    y = x
    for i in range(depth):
        y = jnp.sin(y)
        y = jax.lax.with_sharding_constraint(
            y, NamedSharding(mesh, _spec_for(mesh, i if permute else 0)))
    return jnp.sum(y)


def _mk(mesh_shape, depth, permute):
    return functools.partial(_fn, mesh_shape=mesh_shape, depth=depth, permute=permute), (_X,)


MESH_3D_64 = (4, 4, 4)          # 64 devices, 3 mesh axes -- the primary axis
MESH_3D_8 = (2, 2, 2)           #  8 devices, 3 mesh axes -- the device-count comparison
MESH_6D_64 = (2, 2, 2, 2, 2, 2)  # 64 devices, 6 mesh axes -- the non-monotone comparison

DEPTHS = (8, 16, 32, 64, 128)

CASES = {}

# ---- primary axis: number of reshards, at 64 devices on a 3-axis mesh -------------------------
for _d in DEPTHS:
    CASES[f"spmd_reshard_d{_d}"] = (
        *_mk(MESH_3D_64, _d, True),
        f"synthesised gap 7: {_d} reshards on a 4x4x4 (64-device) mesh, rank-6 f32 tensor. Each "
        f"with_sharding_constraint permutes which mesh axis owns which tensor dim, so the SPMD "
        f"partitioner synthesises ~8 all-to-alls per reshard. jaxpr and StableHLO are identical "
        f"to the control's; the optimised HLO is up to 55x bigger",
    )
    CASES[f"spmd_reshard_d{_d}_control"] = (
        *_mk(MESH_3D_64, _d, False),
        f"control: identical program, identical mesh, identical {_d} with_sharding_constraint "
        f"calls -- every one pinned to the SAME PartitionSpec, so no reshard is ever needed and "
        f"the partitioner emits zero collectives",
    )

# ---- second axis: device count, program held fixed --------------------------------------------
CASES["spmd_reshard_dev8_d64"] = (
    *_mk(MESH_3D_8, 64, True),
    "device-count axis: the d64 program verbatim on a 2x2x2 (8-device) mesh instead of 4x4x4. "
    "Same jaxpr, same StableHLO, same 64 reshards, same 3 mesh axes; 8x fewer devices. Measured "
    "4.95 s against the 64-device arm's 12.02 s -- the partitioner's output size depends on device "
    "count, which no source-level metric sees",
)
CASES["spmd_reshard_dev8_d64_control"] = (
    *_mk(MESH_3D_8, 64, False),
    "control for the 8-device arm: same 2x2x2 mesh, all constraints pinned to one PartitionSpec",
)

# ---- third axis: mesh factorisation at fixed device count.  NON-MONOTONE ----------------------
CASES["spmd_reshard_mesh6d_d128"] = (
    *_mk(MESH_6D_64, 128, True),
    "mesh-factorisation axis, and the file's most useful data point: the d128 program on 64 "
    "devices factored as 2x2x2x2x2x2 (6 axes) instead of 4x4x4 (3 axes). Same devices, same "
    "program, MORE sharded dimensions -- and 22x FASTER (0.88 s vs 19.09 s, zero all-to-alls), "
    "because with all six dims sharded 2-ways a permutation degenerates to device relabeling. Any "
    "heuristic that ranks by mesh rank or sharded-dimension count gets the sign wrong here",
)
CASES["spmd_reshard_mesh6d_d128_control"] = (
    *_mk(MESH_6D_64, 128, False),
    "control for the 6-axis arm: same 2^6 mesh, all constraints pinned to one PartitionSpec. "
    "Measured 0.80 s against its case arm's 0.88 s -- indistinguishable, which is the "
    "non-monotonicity stated plainly: this pair is NOT a pathology and must not score as one",
)
