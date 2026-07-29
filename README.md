# scopex

Attribute a jitted JAX program's compilation artifacts back to the code that wrote them.

When a `jax.jit` compile takes 40 seconds, the question is *which code caused that*. JAX and XLA
already emit enough to answer it — a name stack on every equation, metadata on every HLO
instruction, per-stage timers, a fusion decision log. The information is there, spread across four
representations that name things differently. `scopex` reads all of them and links them to one
record per unit of your program.

One dependency, `jax`. It asks **nothing** of the libraries whose programs you profile.

## Install

```bash
uv add scopex
uv pip install -e .        # from a checkout
```

## Sixty seconds

```python
import scopex

print(scopex.record(my_fn, x))          # which STAGE: trace / lower / backend
```
```
stage        seconds   share
----------------------------
trace          0.144   12.1%
lower          0.515   43.2%
backend        0.527   44.2%
unaccounted    0.006    0.5%
WALL           1.192
```

Then attribute the structure:

```python
units = list(scopex.walk(jax.make_jaxpr(my_fn)(x)))
print(scopex.table(scopex.attribute(units, "site")))
```

## Read this next

**[The blueprint](docs/blueprint_scopex.html)** is the documentation — the differential method, the
two routes (with and without instrumenting a framework), the naming contract, the ways of asking
that silently return nothing, the API, and what scopex will not tell you.

**[examples/marked_framework.py](examples/marked_framework.py)** is a runnable framework-plus-user
model. Every number quoted in the blueprint is its actual output:

```bash
python examples/marked_framework.py
```

## The short version of the caveats

- **Measure a control.** A number from one program is not a diagnosis. A near-identical *fast*
  variant turns every view into a difference, and the difference is the pathology.
- **The device is an axis.** Which backend you compile for decides which passes run at all. One
  known pathology measures 248× on CPU and completely flat on GPU — same code, same sizes.
- **It does not rank lines by compile seconds.** It attributes structure. On a known pathology all
  eight offending instructions share one correct source line whose per-instruction interventions
  span a 19.6× range. Correct is not the same as actionable.
- **Two accessors silently return nothing** where they look authoritative: `Lowered.as_text()`
  defaults to `debug_info=False`, and `compiler_ir(...)` drops locations entirely. Use
  `scopex.stablehlo_text()` / `scopex.hlo_text()`. The blueprint has the full table.

## Correctness

`scopex.walk` is verified equal in count to
`jax._src.jaxpr_util.all_eqns(revisit_inner_jaxprs=True)` across 13 nesting constructs — `pjit`,
`scan`, `cond`, `switch`, `while`, `fori`, `custom_jvp`, `custom_vjp`, `remat`, `named_scope`,
`vmap`, `grad(pjit)`, `grad(scan)` — all attributing an identical payload to the same source line.

```python
ours, jaxs, equal = scopex.verify_parity(jaxpr)
```

Run it on your own program. Disagreement is a bug in scopex.

## License

MIT.
