"""SYNTHESISED (gap 7), AND IT DOES **NOT** REPRODUCE ON CPU -- committed deliberately, because a
negative with a clean control bounds where the pathology is not, and this one falsifies the most
obvious wrong hypothesis a profiler could form about `case_spmd_reshard_permute.py`.

THE HYPOTHESIS THIS CASE KILLS.  Its sibling file shows the SPMD partitioner turning a 72-line
StableHLO module into a 10576-line HLO module and taking 24x longer to compile than its control.
The tempting inference is "the partitioner's cost is the size of the HLO it emits".  This file
holds a mirror to that: the partitioner emits 5.3x more HLO here too, including 192 synthesised
`collective-permute`s, and it costs almost nothing.  Post-partitioning instruction count is
therefore NOT a sufficient predictor, and any tool that ranks SPMD cost by "how much HLO did the
partitioner add" will over-report this case and mis-rank it against the reshard case.

MECHANISM (what it was built to test).  Sharding the SPATIAL dimension of a convolution means each
device holds an interior slab of the signal and needs its neighbours' edge elements to compute the
window at its boundary.  XLA's partitioner handles this with a HALO EXCHANGE: `collective-permute`
to fetch the halo, `pad`/`slice`/`select` to splice it in, per convolution.  Sharding the BATCH
dimension instead makes every convolution purely local and the partitioner emits nothing.  That is
a different partitioner code path from the general reshard the sibling file exercises, which is why
it earned its own probe rather than another arm over there.

WHAT THE CONTROL ISOLATES -- exactly one thing: which axis of the PartitionSpec carries the mesh
name.

    pathological   P(None, None, 'x')     shard the spatial dim W  -> halo exchange per conv
    control        P('x',  None, None)    shard the batch   dim N  -> conv is device-local

Same D convolutions, same `tanh`s, same kernel, same 1-D 8-device mesh, same (8, 16, 2048) f32
input, same number of `with_sharding_constraint` calls, same jaxpr (25 / 49 / 97 equations at
D = 8 / 16 / 32, identical between arms).  One token moves position inside one `PartitionSpec`.

MEASURED IN-ENV (JAX_PLATFORMS=cpu, jax 0.10.2, x64 on, `jax_num_cpu_devices=8`, fresh process per
point).  `compile_s` seconds, and the HLO the partitioner produced:

    D     spatial (case)   batch (control)   ratio    HLO lines    collective-permute   select
     8      0.485 s           0.215 s        2.3x      405 / 107          48 / 0         8 / 0
    16      1.023 s           0.688 s        1.5x      757 / 163          96 / 0        16 / 0
    32      0.820 s           0.696 s        1.2x     1461 / 275         192 / 0        32 / 0

The ratio SHRINKS with D and the absolute compile time is non-monotone (1.02 s at D=16, 0.82 s at
D=32).  Nothing here clears the harness's 3 s floor or its 10x-vs-control bar, and it should not:
the honest reading is that on the CPU backend the halo-exchange rewrite is cheap per convolution
even though it triples the instruction count.  A 6 per-conv `collective-permute` expansion is a
constant-factor rewrite; the sibling file's all-to-all reshard is not.

WHY THIS IS NOT DEAD WEIGHT.  Three uses:

  1.  It is the DISCRIMINATOR described above -- same pass, similar HLO growth, no compile cost.
  2.  It bounds the mechanism: if scopex reports an SPMD signal here at the same strength it
      reports one on `case_spmd_reshard_permute.py`, its SPMD signal is measuring instruction
      count, not partitioner work.
  3.  The GPU arm is genuinely unmeasured.  The GPU is owned by another investigation in this
      workflow and was not touched.  Convolution partitioning is one of the places where the two
      backends most plausibly diverge (cuDNN layout constraints interact with the sharded spatial
      dimension in a way the CPU path has no analogue for), and the corpus already contains one
      case that is 248x on CPU and completely flat on GPU -- the reverse is equally possible.  The
      file is written so that a later GPU run is a one-line invocation.

PLATFORM: **CPU measured and NEGATIVE (1.2-2.3x, below every threshold).  GPU UNVERIFIED** -- the
mesh needs 8 devices, so a GPU run needs 8 real GPUs or a multi-host setup; this box has one and it
is off-limits here.  Treat any GPU number as unknown, not as predicted.

SIZES: D in (8, 16, 32).  Three points are enough to show the ratio going the wrong way, which is
the finding.  W=2048 over 8 devices gives 256 elements per shard against a 15-wide kernel, so the
halo is 7 elements each side -- large enough to be real, small enough that the partitioner does not
give up and all-gather the whole tensor (verified: zero `all-gather` in the emitted HLO).

RUNTIME is 0.13-0.43 s in both arms (real convolutions on 8 fake CPU devices sharing physical
cores), so `compile/runtime` is single-digit here.  The control is what carries the verdict, as the
harness's own `classify()` docstring insists.
"""

from __future__ import annotations

import functools

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

# The one module-scope jax interaction: a config write, no device claimed.
try:
    jax.config.update("jax_num_cpu_devices", 8)
except Exception:
    pass

NDEV = 8
BATCH, CHAN, WIDTH = 8, 16, 2048
KERNEL = 15          # halo of 7 each side against 256 elements per shard

# numpy at module scope, never jnp.
_A = np.ones((BATCH, CHAN, WIDTH), dtype=np.float32)
_K = np.ones((KERNEL, CHAN, CHAN), dtype=np.float32) / (KERNEL * CHAN)


def _sharding(spatial: bool):
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    have = jax.devices()
    if len(have) < NDEV:
        raise RuntimeError(
            f"case_spmd_halo_conv_dim_choice needs {NDEV} addressable devices, found {len(have)}. "
            f"On CPU set jax_num_cpu_devices >= {NDEV} (this module sets it at import, so this "
            f"means the backend was initialised first); on GPU it needs {NDEV} real devices.")
    mesh = Mesh(np.array(have[:NDEV]).reshape(NDEV), ("x",))
    # THE ONLY DIFFERENCE BETWEEN THE ARMS is which position holds 'x'.
    return NamedSharding(mesh, P(None, None, "x") if spatial else P("x", None, None))


def _fn(a, k, depth: int, spatial: bool):
    sh = _sharding(spatial)
    y = a
    for _ in range(depth):
        y = jax.lax.with_sharding_constraint(y, sh)
        y = lax.conv_general_dilated(y, k, (1,), "SAME", dimension_numbers=("NCW", "WIO", "NCW"))
        y = jnp.tanh(y)
    return jnp.sum(y)


DEPTHS = (8, 16, 32)

CASES = {}

for _d in DEPTHS:
    CASES[f"spmd_halo_conv_d{_d}"] = (
        functools.partial(_fn, depth=_d, spatial=True), (_A, _K),
        f"synthesised gap 7, NEGATIVE ON CPU: {_d} convolutions with the SPATIAL dim sharded over "
        f"8 devices, so the partitioner inserts a halo exchange (6 collective-permutes per conv) "
        f"per convolution. Emits 3.8-5.3x the HLO of the control for only 1.2-2.3x the compile -- "
        f"the case exists to show that partitioner-emitted instruction count does NOT predict "
        f"partitioner cost. GPU arm unverified (needs 8 real devices)",
    )
    CASES[f"spmd_halo_conv_d{_d}_control"] = (
        functools.partial(_fn, depth=_d, spatial=False), (_A, _K),
        f"control: the same {_d} convolutions with the BATCH dim sharded instead of the spatial "
        f"dim -- one token moves inside one PartitionSpec. Identical jaxpr, identical shapes, "
        f"identical FLOPs; the partitioner emits zero collectives",
    )
