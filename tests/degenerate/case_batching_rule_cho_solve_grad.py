"""jax#21313 -- compile time of grad(vmap(cho_solve)) grows with the BATCH size, which it must not.

    https://github.com/jax-ml/jax/issues/21313         (filed 2024-05-20, jax 0.4.26, V100)

REPORTED. Compile time of `value_and_grad(lambda p: vmap(func)(p).mean())` where `func` does
`cho_factor` then `cho_solve` on a (nobs, nobs) matrix built from the parameters. nobs=256, ngp=64:

    nbatch     16     32     64    128    256    512   1024   2048
    compile  0.532  0.507  0.516  0.580  0.652  0.822  1.7    2.75

vmap should widen a leading axis and change nothing else, so compile time should be FLAT in nbatch.
It is flat from 16 to 512 and then turns over: there is a KNEE, not a slope. A knee is a much better
attribution target than a smooth ramp, because below it the program is not merely smaller, it is
CORRECT -- same source, same primitives, same everything but one integer -- so whatever scopex
points at above the knee has to be absent below it.

SUSPECTED MECHANISM (hypothesis, not a maintainer diagnosis -- the issue carries none). The reverse
mode of cho_factor/cho_solve materialising something per batch element: most plausibly the
triangular-solve transpose rule, or a batched-Cholesky fallback that XLA:GPU unrolls above a
threshold. This file does not assert which; it provides the arms that separate them.

TWO CONTROLS, ON TWO DIFFERENT AXES.

  1. BATCH SIZE. `cho_grad_b16` vs `cho_grad_b2048` -- one integer, same source text. This axis is
     read off the sweep rather than from a `_control` suffix, because both arms are cases.

  2. THE GRADIENT. `<name>_control` is the same program with `jax.value_and_grad` DROPPED: the
     forward `vmap(func)(pars).mean()` at the SAME nbatch, same geometry, same primitives. If
     forward compile stays flat while grad compile climbs, the pathology is in the transpose and not
     in vmap or in Cholesky -- structurally the same finding as our scatter-chain-plus-`jnp.sum`
     result (a reduction/transpose stage, not the payload, being the whole cost) on a completely
     unrelated primitive family.

HONEST STATUS -- THIS ONE MAY NOT CLEAR THE BAR AND IS HERE ANYWAY. The reported numbers do NOT
pass: 2.75 s is under the 3.0 s floor and 2048-vs-16 is ~5x, under the 10x gate. It is in the corpus
because the trend past the knee is superlinear and it is cheap to push further, so the sweep is
extended to 2048 at the reported geometry and to 16384 at a narrow geometry. If it has flattened on
0.10.2 that is a clean negative result about a batching rule that was fixed, and it should be
dropped fast rather than re-litigated.

MEMORY (this is why the sweep is shaped the way it is, not the reported shape scaled up). The
per-example matrix is (nobs, nobs) float64, so the batched `ftf` alone is nbatch*nobs^2*8 bytes and
reverse mode keeps several such buffers live. At the reported nobs=256, nbatch=2048 is already
1 GB per buffer, which is the ceiling on a 16 GB card. To reach batch sizes far past the knee
without an OOM -- an OOM at run time would throw away the compile number we came for -- the sweep
also carries a NARROW geometry (nobs=64, ngp=16, 32 KB per example) at nbatch 16 and 16384. The
narrow arms change the per-example work but not the question, which is whether compile time depends
on the batch axis at all.

PLATFORM. Reported on GPU (V100). Run CPU too: a GPU-only pathology localises the cost to the
XLA:GPU lowering of the linalg ops rather than to the shared batching rule, which is exactly the
distinction scopex is supposed to make.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _geometry(nobs: int, ngp: int):
    """Host-side only -- numpy throughout, so importing this module claims no device."""
    t = np.linspace(0, 1, nobs)
    f = np.arange(1, ngp + 1, dtype=np.float64)
    fmat = np.zeros((nobs, 2 * ngp), dtype=np.float64)
    fmat[:, ::2] = np.sin(2.0 * np.pi * f * t[:, np.newaxis])
    fmat[:, 1::2] = np.cos(2.0 * np.pi * f * t[:, np.newaxis])
    return fmat, np.identity(nobs, dtype=np.float64), np.ones(nobs, dtype=np.float64)


def _mk(nobs: int, ngp: int, nbatch: int, grad: bool, tag: str):
    fmat, one, ones = _geometry(nobs, ngp)

    def func(pars):
        # Verbatim from the issue: the explicit jnp.diag is kept even though
        # fmat @ diag(p**2) @ fmat.T == (fmat * p**2) @ fmat.T, because rewriting it would
        # change the primitives under test.
        ftf = fmat @ jnp.diag(pars ** 2) @ fmat.T + one
        cf = jax.scipy.linalg.cho_factor(ftf)
        b = jax.scipy.linalg.cho_solve(cf, ones)
        return b.mean()

    def fwd(pars):
        return jax.vmap(func)(pars).mean()

    fn = jax.value_and_grad(fwd) if grad else fwd
    pars = np.random.default_rng(0).standard_normal((nbatch, 2 * ngp))
    what = "value_and_grad(vmap(cho_solve))" if grad else "control: FORWARD ONLY, no grad"
    return fn, (pars,), f"jax#21313 {what}, nobs={nobs} ngp={ngp} nbatch={nbatch} [{tag}]"


CASES = {}

# Reported geometry. The knee was between 512 and 1024; 2048 is the largest that fits alongside
# reverse-mode's live buffers on a 16 GB card.
for _nb in (16, 512, 1024, 2048):
    CASES[f"cho_grad_b{_nb}"] = _mk(256, 64, _nb, True, "reported geometry")
    CASES[f"cho_grad_b{_nb}_control"] = _mk(256, 64, _nb, False, "reported geometry")

# Narrow geometry: 64x64 per example, so nbatch can go 8x past the reported maximum on 0.5 GB.
for _nb in (16, 16_384):
    CASES[f"cho_grad_narrow_b{_nb}"] = _mk(64, 16, _nb, True, "narrow")
    CASES[f"cho_grad_narrow_b{_nb}_control"] = _mk(64, 16, _nb, False, "narrow")
