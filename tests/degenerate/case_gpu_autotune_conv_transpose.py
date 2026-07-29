"""xla#5541 / jax#17464 -- conv_transpose spends ~252 s COMPILING and 1.8 ms running.

    https://github.com/jax-ml/jax/issues/17464

Reported: ``lax.conv_transpose(x, k, strides=(16,16), padding="VALID")`` with
x=(1,8,8,128), k=(32,32,128,32) took ~252 s to compile and ~1.8 ms to execute on GPU.
The same call with ``strides=(1,1)`` compiles in about a second.

WHY THIS CASE EARNS ITS PLACE IN THE CORPUS.  Every other reproduced case in the corpus burns
compile time TRANSFORMING IR -- more nodes, more passes, superlinear pass cost.  This one burns
it RUNNING KERNELS.  ``conv_transpose`` with stride s lowers to a plain convolution with
``lhs_dilate = s``, and XLA:GPU's ``GpuConvAlgorithmPicker`` benchmarks every candidate cuDNN
algorithm on the real buffers at compile time to pick a winner.  Several algorithms are
pathologically slow on a 16x-dilated input, and the picker times all of them, to completion,
with no early abort.  hawkinsp confirmed the diagnosis on the issue; akuegel added that
``heuristics_mode_a`` and ``heuristics_mode_b`` return identical candidate lists, so every
algorithm is benchmarked TWICE.

That makes it the adversarial case for scopex.  A profiler that attributes compile time to HLO
passes will see a module with a handful of instructions and one convolution, and will report a
flat, uninteresting profile -- while 99.9% of the wall clock sits inside one autotuning step of
one backend pass.  If scopex cannot say "the time is in autotuning this conv", it is blind to an
entire mechanism class.

MEASURED IN THIS ENVIRONMENT (jax 0.10.2, x64, the harness's own machine), by the search phase, on
a LARGER input than the issue's -- these are the numbers this file is anchored to, and the
``convT64_*`` arms below are that exact program:

    lhs (1,64,64,128) f32, kernel (32,32,128,32) f32, padding SAME, output reduced with .sum()

    strides=(16,16)                 compile 69.846 s   run 450.7 ms      <- convT64_dilate16
    strides=(1,1),  same kernel     compile  1.853 s   run   3.44 ms     -> 37.7x   (control)
    strides=(16,16), kernel 3x3     compile  3.712 s   run   5.56 ms     -> 18.8x   (probe)

Two things about that table matter for reading the results.  First, 37.7x on a ONE-TOKEN change is
as clean an attribution target as the corpus has.  Second, ``compile/runtime`` for the slow arm is
only 155, well under the corpus's 1000 -- because stride-16 upsampling makes the output 64x bigger
and the RUN gets genuinely expensive.  That is not a defect in the case: ``classify()`` ranks the
control comparison above the ratio for exactly this situation, and a ratio computed against 450 ms
of real convolution says nothing about whether compilation was pathological.  The ``convT_dilate*``
arms (8x8 input) exist alongside as the low-runtime variant: same mechanism, ~64x less output, so
if they stay slow they clear the ratio bar too and the case is proven twice over.

CONTROLS.  Three exist, in decreasing order of purity:

  (1) FLAG -- the same program under ``XLA_FLAGS=--xla_gpu_autotune_level=0``.  This is the pure
      control: the program, the HLO, the shapes and the dtypes are byte-identical, and only the
      compiler's willingness to benchmark changes.  It CANNOT be expressed as a CASES entry,
      because XLA flags are read once when the backend initialises and the harness gives every
      case the same environment.  Run it by hand against the same file:

          XLA_FLAGS=--xla_gpu_autotune_level=0 python _harness.py convT_dilate16

      If ``convT_dilate16`` collapses to ~1 s under that flag and stays slow without it, the
      mechanism is autotuning and nothing else.  This is worth doing once; it is the single
      cleanest attribution experiment in the corpus.

  (2) STRIDE (the ``_control`` entries below) -- ``strides=(1,1)`` instead of ``(16,16)``.  Same
      op, same operand shapes, same dtypes, same padding, same number of HLO instructions; the
      only difference is the ``lhs_dilate`` field on the conv window, which is exactly the field
      the slow cuDNN algorithms choke on.  This is the semantic twin and the one the harness
      scores automatically.

  (3) KERNEL -- ``convT_dilate16_tinyk`` keeps ``strides=(16,16)`` but shrinks the kernel from
      32x32 to 3x3.  It is not a control, it is a discriminator: dilation is unchanged, so if
      this arm is ALSO fast then the cost is a function of the candidate kernels' cost at that
      kernel size, not of dilation alone.

DTYPE.  Arrays are explicitly float32.  x64 is on globally in the harness, and an f64
convolution takes a completely different (non-cuDNN) code path on GPU, which would silently
destroy the pathology.

PLATFORM.  This case is GPU-ONLY BY CONSTRUCTION.  ``GpuConvAlgorithmPicker`` does not exist in
the CPU backend, so on CPU the mechanism cannot be present and a fast CPU arm proves nothing
about the case.  That makes ``--platform cpu`` worth running anyway, as a negative control on the
MECHANISM rather than on the program: if the 16x arm is also slow on CPU, then the cost is not
autotuning and this file's whole story is wrong.

VERIFIED AT TRACE TIME (CPU, jax 0.10.2, no execution): every arm is a single-equation jaxpr, and
the dilation is visibly in the output shapes -- ``convT_dilate16`` gives (1,144,144,32) where
144 = 7*16 + 32, against (1,39,39,32) for its stride-1 control.  One instruction, ~250 s: that
size mismatch between program and compile time is the entire point of the case.

UNCERTAINTY.  The issue is from jax 0.4.x.  Since then XLA:GPU has moved much conv selection to
the cuDNN graph API and heuristics-only modes, and recent builds cap or skip autotuning for
some configurations.  If the 16x arm does not clear the floor, scale up: ``convT_dilate32``
(64x64 kernel, stride 32) is here for exactly that reason, and beyond it the next knobs are more
input channels or a larger spatial input.
"""

from __future__ import annotations

import functools

import numpy as np

from jax import lax

_rng = np.random.default_rng(17464)

# Cached by shape so the 64x64x128x32 kernel (67 MB) is materialised once, not once per arm.
_ARRAYS: dict[tuple[int, ...], np.ndarray] = {}


def _arr(*shape: int) -> np.ndarray:
    """A float32 host array of the given shape, shared between arms that need the same one."""
    if shape not in _ARRAYS:
        _ARRAYS[shape] = _rng.standard_normal(shape, dtype=np.float32)
    return _ARRAYS[shape]


def _conv_transpose(x, k, strides):
    return lax.conv_transpose(x, k, strides=strides, padding="VALID")


def _mk(kernel_hw: int, stride: int, note: str):
    x = _arr(1, 8, 8, 128)
    k = _arr(kernel_hw, kernel_hw, 128, 32)
    fn = functools.partial(_conv_transpose, strides=(stride, stride))
    return fn, (x, k), note


def _conv_transpose_same_sum(x, k, strides):
    """The exact form the 69.846 s measurement was taken on: SAME padding, explicit NHWC/HWIO, sum.

    The trailing ``.sum()`` is not decoration -- without it the stride-16 arm returns a
    (1,1024,1024,32) f32 array (134 MB) per call, and it is what the measured 450.7 ms runtime
    includes.  ``dimension_numbers`` is spelled out rather than left to default so the two files'
    arms cannot silently disagree about layout.
    """
    return lax.conv_transpose(x, k, strides=strides, padding="SAME",
                              dimension_numbers=("NHWC", "HWIO", "NHWC")).sum()


def _mk64(kernel_hw: int, stride: int, note: str):
    """The VERIFIED configuration: 64x64x128 input, SAME padding, reduced output."""
    x = _arr(1, 64, 64, 128)
    k = _arr(kernel_hw, kernel_hw, 128, 32)
    fn = functools.partial(_conv_transpose_same_sum, strides=(stride, stride))
    return fn, (x, k), note


CASES = {
    # --- VERIFIED pair: the configuration actually measured at 69.846 s vs 1.853 s in this env ---
    "convT64_dilate16": _mk64(
        32, 16,
        "xla#5541 VERIFIED here: 64x64 input, k=32x32, strides=(16,16), SAME -- compile 69.846 s, "
        "run 450.7 ms (ratio 155; judge on the 37.7x control, not the ratio)",
    ),
    "convT64_dilate16_control": _mk64(
        32, 1,
        "control (VERIFIED 1.853 s): identical program and shapes, strides=(1,1) so no lhs_dilate "
        "-> 37.7x",
    ),
    "convT64_dilate16_tinyk": _mk64(
        3, 16,
        "probe (VERIFIED 3.712 s): dilation kept at 16, kernel 32x32->3x3 -- 18.8x below the slow "
        "arm, so the autotune cost is a function of candidate cost at that kernel size too",
    ),

    # --- primary pair: 32x32 kernel, stride 16 vs stride 1 -------------------------------------
    "convT_dilate16": _mk(
        32, 16,
        "xla#5541: conv_transpose k=32x32 strides=(16,16) -> lhs_dilate=16, cuDNN picker times "
        "every candidate",
    ),
    "convT_dilate16_control": _mk(
        32, 1,
        "control: identical op/shapes/dtypes, strides=(1,1) so no lhs_dilate and a normal "
        "algorithm wins fast",
    ),

    # --- scaled pair, in case the 16x arm has been fixed or capped upstream ---------------------
    "convT_dilate32": _mk(
        64, 32,
        "scaled: k=64x64 strides=(32,32); more dilation and a bigger candidate set if 16x is "
        "below the floor",
    ),
    "convT_dilate32_control": _mk(
        64, 1,
        "control: k=64x64 strides=(1,1), same 67 MB kernel, no dilation",
    ),

    # --- discriminator, not a control ----------------------------------------------------------
    "convT_dilate16_tinyk": _mk(
        3, 16,
        "probe (no auto-pair): strides=(16,16) kept, kernel shrunk 32x32->3x3, so a fast result "
        "means kernel size not dilation drives the autotune cost",
    ),
}
