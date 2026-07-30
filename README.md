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

**[examples/recipes/](examples/recipes/)** is one runnable file per question — *which stage owns the
wall*, *which pass grew the module*, *who is slow inside tracing*, *where did the seconds go when the
passes account for none of them*. Each names the case it was derived on and the conditions under
which its knob stops working. Blueprint §7 routes them.

**[docs/HARDENING.md](docs/HARDENING.md)** is for anyone changing a parser: every remaining piece of
compiler text scopex reads, what prints it, what happens when that changes, and how the two
self-checks (`scopex.conformance()`, `scopex.selftest()`) cover it.

**[docs/DEFICITS.md](docs/DEFICITS.md)** is for anyone deciding whether to trust a number scopex just
produced: one section per instrument saying what validates it, **where that validation stops
working**, what the instrument structurally cannot see, and what it costs in extra compiles. It also
carries the ship call for each — including the one thing that was measured, found 49–62% correct
exactly where it would be used, and deliberately **not** built.

**[docs/INVESTIGATIONS.md](docs/INVESTIGATIONS.md)** is the case record — 15 investigations, what each
instrument did and did not show, and the routes that were tried and rejected with the evidence.

## The short version of the caveats

- **Measure a control.** A number from one program is not a diagnosis. A near-identical *fast*
  variant turns every view into a difference, and the difference is the pathology.
- **The device is an axis.** Which backend you compile for decides which passes run at all. One
  known pathology measures 248× on CPU and completely flat on GPU — same code, same sizes.
- **It does not rank lines by compile seconds.** It attributes structure. On a known pathology all
  eight offending instructions share one correct source line whose per-instruction interventions
  span a 19.6× range. Correct is not the same as actionable.
- **Two accessors silently return nothing** where they look authoritative: `Lowered.as_text()`
  defaults to `debug_info=False`, and printing `compiler_ir(...)` shows no locations — that one is
  MLIR's `Operation.__str__` default, not the IR, which carries them. Use
  `scopex.stablehlo_text()` / `scopex.hlo_text()`, or `scopex.stablehlo_module()` for the module
  itself. The blueprint has the full table and why the wrong explanation cost a parser.

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
