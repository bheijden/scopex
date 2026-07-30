"""Compile a spread of small programs in ONE process with vmodule on, so the stderr log
contains every HLO pass XLA ran for any of them. Union of pass names is the point; the
timings are not.

Run as:  TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE=hlo_pass_pipeline=1 python probe_passes.py
"""
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
import functools
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

jax.config.update("jax_enable_x64", True)

R = np.random.RandomState(0)


def run(name, fn, *args):
    try:
        jax.block_until_ready(jax.jit(fn)(*args))
        print(f"PROBE-OK {name}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"PROBE-FAIL {name}: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr, flush=True)


x = jnp.asarray(R.randn(64, 64), jnp.float32)
v = jnp.asarray(R.randn(4096), jnp.float32)
iv = jnp.asarray(R.randint(0, 1000, 4096), jnp.int32)

run("elementwise_dot", lambda a: jnp.sum(jnp.tanh(a) @ a), x)
run("reduce_window", lambda a: lax.reduce_window(a, -jnp.inf, lax.max, (2, 2), (2, 2), "VALID"), x)
run("sort_f32", lambda a: jnp.sort(a)[:8].sum(), v)
run("argsort_f32", lambda a: jnp.argsort(a)[:8].sum(), v)
run("argsort_i32", lambda a: jnp.argsort(a)[:8].sum(), iv)
run("topk", lambda a: lax.top_k(a, 16)[0].sum(), v)
run("gather", lambda a, i: a[i % 4096].sum(), v, iv)
run("scatter", lambda a, i: a.at[i % 4096].add(1.0).sum(), v, iv)
run("while_loop", lambda a: lax.while_loop(lambda s: s[0] < 8, lambda s: (s[0] + 1, s[1] * 1.01),
                                           (0, a))[1].sum(), x)
run("scan", lambda a: lax.scan(lambda c, _: (c * 1.01, c.sum()), a, None, length=8)[1].sum(), x)
run("cond", lambda a: lax.cond(a.sum() > 0, lambda z: z * 2, lambda z: z / 2, a).sum(), x)
run("conv", lambda a: lax.conv_general_dilated(
    a.reshape(1, 1, 64, 64), jnp.ones((4, 1, 3, 3), jnp.float32), (1, 1), "SAME").sum(), x)
run("cholesky", lambda a: jnp.linalg.cholesky(a @ a.T + 64 * jnp.eye(64)).sum(), x)
run("eigh", lambda a: jnp.linalg.eigh(a @ a.T)[0].sum(), x)
run("svd", lambda a: jnp.linalg.svd(a, compute_uv=False).sum(), x)
run("triangular_solve", lambda a: lax.linalg.triangular_solve(
    jnp.tril(a) + 64 * jnp.eye(64), a).sum(), x)
run("fft", lambda a: jnp.abs(jnp.fft.fft(a.astype(jnp.complex64))).sum(), v)
run("rng", lambda k: jax.random.normal(k, (256, 256)).sum(), jax.random.PRNGKey(0))
run("rng_poisson", lambda k: jax.random.poisson(k, 3.0, (256,)).sum(), jax.random.PRNGKey(0))
run("bf16_convert", lambda a: (a.astype(jnp.bfloat16) * 2).astype(jnp.float32).sum(), x)
run("f64", lambda a: jnp.sum(a.astype(jnp.float64) ** 2), x)
run("int4", lambda a: (a.astype(jnp.int8) // 2).sum(), iv)
run("concat_slice", lambda a: jnp.concatenate([a, a[::2]])[3:100].sum(), v)
run("dynamic_slice", lambda a, i: lax.dynamic_slice(a, (i[0] % 100,), (64,)).sum(), v, iv)
run("grad", jax.grad(lambda a: jnp.sum(jnp.tanh(a) @ a)), x)
run("remat", jax.grad(lambda a: jnp.sum(jax.checkpoint(lambda z: jnp.tanh(z) @ z)(a))), x)
run("vmap_batched", jax.vmap(lambda a: jnp.sum(jnp.tanh(a) * a)), jnp.stack([x] * 8))
run("optimization_barrier", lambda a: lax.optimization_barrier(a * 2).sum(), x)
run("select_and_scatter", jax.grad(lambda a: lax.reduce_window(
    a, -jnp.inf, lax.max, (2, 2), (2, 2), "VALID").sum()), x)
run("cumsum", lambda a: jnp.cumsum(a).sum(), v)
run("cumlogsumexp", lambda a: lax.cumlogsumexp(a).sum(), v)
run("where_select", lambda a: jnp.where(a > 0, a, -a).sum(), x)
run("pad_rev", lambda a: jnp.pad(a, 3)[::-1].sum(), x)
run("bitcast", lambda a: lax.bitcast_convert_type(a, jnp.int32).sum(), v)
run("iota_broadcast", lambda a: (jnp.arange(64, dtype=jnp.float32)[:, None] + a).sum(), x)
run("triton_like_dot_chain", lambda a: functools.reduce(lambda z, _: jnp.tanh(z @ a), range(4), a).sum(), x)
run("integer_dot", lambda a: (a[:, None] * a[None, :]).sum(), iv[:64])
run("complex", lambda a: jnp.abs(a.astype(jnp.complex64) * 1j).sum(), v)
run("erf_special", lambda a: (jax.scipy.special.erf(a) + jax.scipy.special.digamma(jnp.abs(a) + 1)).sum(), v)
run("logistic_expm1", lambda a: (jax.nn.sigmoid(a) + jnp.expm1(a) + jnp.log1p(jnp.abs(a))).sum(), v)

# Collectives / SPMD: needs several devices.
try:
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P
    devs = jax.devices()
    if len(devs) >= 2:
        mesh = Mesh(np.asarray(devs[:2]), ("d",))
        s = NamedSharding(mesh, P("d"))
        xs = jax.device_put(jnp.asarray(R.randn(256, 256), jnp.float32), s)
        run("spmd_allreduce", lambda a: jnp.sum(a @ a.T), xs)
        run("spmd_reshard", lambda a: jax.lax.with_sharding_constraint(
            a, NamedSharding(mesh, P(None, "d"))).sum(), xs)
except Exception as e:
    print(f"PROBE-FAIL spmd: {e}", file=sys.stderr, flush=True)

print("PROBE-DONE", file=sys.stderr, flush=True)
