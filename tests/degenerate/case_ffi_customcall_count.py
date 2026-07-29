"""SYNTHESISED (gap 6: custom call / FFI). Interleaving N IDENTITY FFI custom calls into an N-op
elementwise chain multiplies compile time by 28x and turns it superlinear -- and the cost is NOT in
LLVM, which is what separates this case from every other one in the corpus that also gets slow.

No issue URL -- constructed from the mechanism, not mined. `jax.pure_callback` is the cheapest way
to obtain a genuine FFI custom call from pure python: it lowers to
`stablehlo.custom_call @xla_ffi_python_cpu_callback` (verified in this environment), which is an
FFI-registered target exactly like a hand-written `jax.ffi.ffi_call` handler, with no C++ to build.
The python function is the identity, so the arm adds ZERO arithmetic: every extra second is the
price of the custom-call boundary itself.

MEASURED IN THIS ENVIRONMENT (JAX_PLATFORMS=cpu, jax/jaxlib 0.10.2, x64 on, x = 64 f32, one fresh
process per point, single shot -- see the noise warning below).

AXIS A -- one identity custom call after every `sin`,`mul` pair, N pairs:

    N        control (no calls)   with N custom calls    HLO lines (control / case)
     512          0.784 s               1.730 s              1,054 /  5,145
    1024          1.527 s               4.413 s              2,078 / 10,265
    2048          2.886 s              28.875 s              4,126 / 20,505
    4096          5.355 s             151.756 s              8,222 / 40,985

28.3x at N=4096. The control grows LINEARLY in N (0.78 -> 5.36 s, a factor 6.8 for a factor 8 in
size); the custom-call arm grows as roughly N^2.6 (1.73 -> 151.76, a factor 88). HLO instruction
count grows by only 5x between the arms while compile time grows by 28x, so this is not "the module
got bigger".

AXIS B -- the sharper one. TOTAL CHAIN LENGTH HELD FIXED at 2048 `sin`,`mul` pairs; only the SPACING
of the custom calls changes, so the arithmetic, the FLOPs, the output and the source of the payload
are identical in every row:

    one call every M ops     custom calls    compile      HLO lines
      (none)                        0        2.429 s        4,126
      M = 2048                      1        2.491 s        4,129
      M =  512                      4        1.408 s        4,153
      M =  128                     16        1.159 s        4,249
      M =   32                     64        1.178 s        4,633
      M =    8                    256        1.374 s        6,169
      M =    2                  1,024        3.966 s       12,313
      M =    1                  2,048       18.582 s       20,505

**THE CURVE IS U-SHAPED.** Sixteen custom calls make the identical program compile 2.1x FASTER than
zero custom calls, because they chop one enormous fusion into pieces the backend handles cheaply.
Two thousand of them make it 16x SLOWER. The minimum is in the middle, at M=128. One knob, fixed op
count, fixed FLOPs, non-monotone by a factor of 16 -- any attribution that is monotone in "number of
custom calls" or in "module size" gets the middle of this table backwards, and the two ends of it
have opposite signs.

THE STAGE, MEASURED not inferred. Recompiling the N=2048 arm with the LLVM optimiser off:

    XLA_FLAGS=<none>                             compile 19.090 s
    XLA_FLAGS=--xla_backend_optimization_level=0 compile 21.034 s
    XLA_FLAGS=--xla_backend_optimization_level=1 compile 26.415 s

Disabling LLVM optimisation does not help at all. Contrast `case_llvm_scan_unroll_spill`, measured
the same way in the same environment, where the same flag removes 89% of the compile. So the two
files are a matched pair for stage attribution: same harness, same flag, opposite answers. Whatever
is expensive here lives ABOVE LLVM -- in the HLO pipeline, in buffer assignment/scheduling around
opaque unaliasable results, or in building the executable's thunk sequence. A profiler that lumps
"slow compile" into one bucket cannot tell these two apart; one that reads only HLO pass timings
will be right here and wrong there.

WHY THE CUSTOM CALL AND NOT JUST "AN OPAQUE OP". `lax.optimization_barrier` was measured as a third
arm at every N (0.865 / 1.588 / 4.074 / 5.410 s) and tracks the control almost exactly -- XLA folds
the barriers away and the optimised module ends up 2 instructions larger than the control's. So the
28x is not "something blocked fusion"; it is specific to the custom call. That arm is not shipped
as a case because it is indistinguishable from the control; it is recorded here as the reason the
control below is the right one.

CONTROLS.
  * AXIS A control: the same chain with the `pure_callback` line deleted. Same N, same arithmetic,
    same output, same dtype; one line of source.
  * AXIS B control: the same 2048-pair chain with no custom calls at all, paired against each
    spacing. Because the payload is identical in every axis-B row, the whole family is also
    readable as a sweep against each other, which is where the U-shape shows up.

NOISE WARNING. These are single-shot numbers from a shared machine. The N=2048 case arm was
measured at 28.875 s, 18.582 s and 19.090 s in three separate runs -- a 55% spread. The ORDERING
and the ratios are stable; individual absolute seconds are not. Use the harness's rounds/median
rather than the table above when a number matters.

PLATFORM: EITHER, measured on CPU. `xla_ffi_python_cpu_callback` is the CPU target; on GPU the
equivalent callback target is used and the same structure should hold, but GPU was off-limits when
this was written so the GPU arm is UNVERIFIED.

RUNTIME. The case arms really do call back into python N times per execution, so runtime is not
negligible at N=4096 and the compile/runtime ratio will be lower than for a pure-XLA case. The
control comparison, not the ratio, is the verdict here -- which is exactly the ordering the
harness's `classify` uses.

NUMPY at module scope; no device is touched at import.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

# Tiny payload: the point is the number of custom-call boundaries, not the arithmetic across them.
ELEMS = 64

# float32 spelled out -- the harness enables x64 globally, and f64 would double every buffer in a
# case whose subject is how many buffers there are.
_SCALE = np.float32(1.0009765625)


def _identity(a):
    """The FFI payload. Deliberately does NOTHING: the arm adds no arithmetic, only a boundary."""
    return a


def _cb(y):
    return jax.pure_callback(_identity, jax.ShapeDtypeStruct(y.shape, y.dtype), y)


def _chain(x, n, with_calls):
    """n `sin`,`mul` pairs, optionally with one identity custom call after each."""
    y = x
    for _ in range(n):
        y = jnp.sin(y) * _SCALE
        if with_calls:
            y = _cb(y)
    return y


# Axis B holds this many pairs regardless of spacing, so every axis-B arm has identical arithmetic.
SPACING_LEN = 2048


def _spaced(x, m):
    """SPACING_LEN pairs with one identity custom call every m ops. m=0 means none."""
    y = x
    for i in range(SPACING_LEN):
        y = jnp.sin(y) * _SCALE
        if m and (i + 1) % m == 0:
            y = _cb(y)
    return y


_X = np.zeros(ELEMS, dtype=np.float32)

# 512 is below the turn-up and is here to anchor the linear part of the curve; 4096 is where the
# superlinearity is unmistakable (151 s against a 5 s control) and is also the largest that fits
# comfortably inside the harness's 900 s timeout at the observed growth rate.
CHAIN_NS = (512, 1024, 2048, 4096)

# 2048 = a single custom call at the very end; 1 = one after every op. 128 is the MINIMUM of the
# U-shape and is the row that makes the curve non-monotone rather than merely superlinear.
SPACINGS = (2048, 128, 8, 1)

CASES = {}

for _n in CHAIN_NS:
    CASES[f"ffi_cb_chain_{_n}"] = (
        functools.partial(_chain, n=_n, with_calls=True), (_X,),
        f"synthesised gap-6: {_n}-op elementwise chain with {_n} identity FFI custom calls "
        f"(pure_callback -> xla_ffi_python_cpu_callback) -- cost is NOT in LLVM "
        f"(--xla_backend_optimization_level=0 does not help)",
    )
    CASES[f"ffi_cb_chain_{_n}_control"] = (
        functools.partial(_chain, n=_n, with_calls=False), (_X,),
        f"control: the identical {_n}-op chain with the pure_callback line deleted -- same "
        f"arithmetic, same output, one line of source",
    )

for _m in SPACINGS:
    CASES[f"ffi_cb_spacing_{_m}"] = (
        functools.partial(_spaced, m=_m), (_X,),
        f"synthesised gap-6: {SPACING_LEN}-op chain, FIXED arithmetic, one identity custom call "
        f"every {_m} ops ({SPACING_LEN // _m} calls) -- read the four spacings against each other, "
        f"the curve is U-shaped with its minimum near m=128",
    )
    CASES[f"ffi_cb_spacing_{_m}_control"] = (
        functools.partial(_spaced, m=0), (_X,),
        f"control: the identical {SPACING_LEN}-op chain with no custom calls at all",
    )
