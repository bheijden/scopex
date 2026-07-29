"""jax#10596 -- FFT of a CLOSURE-CAPTURED array: >10 s compile on GPU; as an ARGUMENT, ~0.1 s.

    https://github.com/jax-ml/jax/issues/10596         CLOSED (filed 2022-05-06, labelled bug)

REPORTED. Two snippets that differ only in how the array reaches the FFT:

    pulse = jax.device_put(np.random.rand(8000))
    def f():        return jnp.fft.fft(pulse)     # captured  -> jit on GPU took >10 s
    def f(pulse):   return jnp.fft.fft(pulse)     # parameter -> jit on GPU took ~0.1 s

The same captured version on CPU compiled in ~0.1 s. 8000-10000 elements; the array is trivial.

MECHANISM UNDER TEST. A captured device array is not an input, it is a CONSTANT: it is inlined into
the HLO as a literal. The GPU pipeline then has a constant-valued FFT sitting in front of it and
tries to evaluate it at compile time. Turning the identical array into a parameter turns the literal
into a parameter and the fold has nothing to fire on. Same family as jax#12789, reached through a
different binding (capture vs parameter) and a different primitive (fft).

The CPU/GPU asymmetry is itself the attribution signal, and is why this file must be run on BOTH
platforms: >10 s on GPU against ~0.1 s on CPU for byte-identical HLO says the cost is in a
backend-specific pass, not in shared HLO simplification. A CPU-only "does not reproduce" would be a
statement about the CPU pipeline, not about the case.

WHAT THE CONTROL ISOLATES. `<name>` captures the array; `<name>_control` takes it as an argument.
Same math, same primitive, same dtype, same length, same result. The single bit that differs is
constant-vs-parameter. This is the same axis jax#14080 probes, so it is probed once, here.

SECOND CONTROL (the discriminator that makes an attribution, not just a repro). `fftcap_sin_*`
captures a constant of the SAME size and shape and applies a cheap elementwise op instead of the
FFT. If the FFT arm is slow and the sin arm is fast, the cost is "GPU constant-folding an FFT", not
"a big literal in the HLO" -- these two explanations are otherwise indistinguishable and predict
opposite things about where scopex should point.

SIZES. 8_000 is the reported size and may well fold instantly on a 2026 pipeline; the mechanism
under test is captured-constant -> compile-time evaluation, not that particular length, so the sweep
runs to 1_000_000 before the case may be called dead. 1e6 float64 in, 1e6 complex128 out = 24 MB,
and the folded literal that ends up in the optimised module is 16 MB. It is capped there
deliberately: at 1e7 the harness's own `hlo_modules()[0].to_string()` would materialise a
multi-hundred-MB Python string for the folded constant and the measurement machinery, not the case,
would become the bottleneck.

CAPTURE MECHANISM. The issue captures `jax.device_put(...)`; this file captures a host numpy array,
because building CASES must not touch the accelerator at import. Verified equivalent -- under
JAX 0.10.2 `jax.make_jaxpr` gives a byte-identical jaxpr for `fft(np_array)` and
`fft(jax.device_put(np_array))`: both become the same constvar, no `device_put` equation is staged
out, and both reach HLO as the same literal.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

# Host-side only: importing this module to discover CASES must never claim a device.
_rng = np.random.default_rng(0)


def _mk_fft(n: int, captured: bool):
    arr = _rng.random(n)                      # float64 (x64 is on) -> complex128 out
    if captured:
        def fn():
            return jnp.fft.fft(arr)           # CONSTANT: inlined into the HLO as a literal
        return fn, (), f"jax#10596 fft of CAPTURED constant, n={n}"

    def fn(x):
        return jnp.fft.fft(x)                 # PARAMETER: nothing to fold
    return fn, (arr,), f"control: fft of ARGUMENT, n={n}"


def _mk_sin(n: int, captured: bool):
    """Discriminator: same captured literal, a primitive that is cheap to fold."""
    arr = _rng.random(n)
    if captured:
        def fn():
            return jnp.sin(arr)
        return fn, (), f"discriminator: sin of CAPTURED constant (same size as fft), n={n}"

    def fn(x):
        return jnp.sin(x)
    return fn, (arr,), f"control: sin of ARGUMENT, n={n}"


CASES = {}
for _n in (8_000, 100_000, 1_000_000):
    CASES[f"fftcap_{_n}"] = _mk_fft(_n, True)
    CASES[f"fftcap_{_n}_control"] = _mk_fft(_n, False)

CASES["fftcap_sin_1000000"] = _mk_sin(1_000_000, True)
CASES["fftcap_sin_1000000_control"] = _mk_sin(1_000_000, False)
