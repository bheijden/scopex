# scopex

Attribute a jitted JAX program's compilation artifacts back to the code that wrote them.

When a `jax.jit` compile takes 40 seconds, the question is *which code caused that*. JAX and XLA
already emit enough to answer it — a name stack on every equation, metadata on every HLO
instruction, per-stage timers, a fusion decision log. The information is there and it is spread
across four representations that name things differently. `scopex` reads all of them and links them
to one record per unit of your program.

It has one dependency, `jax`. It asks **nothing** of the libraries whose programs you profile.

## Install

```bash
uv add scopex
# or, from a checkout:
uv pip install -e .
```

## Start here: which stage is slow?

Everything else is guessing until you know this. Attribution explains time spent in tracing and in
HLO passes; it says nothing about time spent in kernel autotuning.

```python
import scopex

print(scopex.record(my_fn, x, y))
```

```
stage        seconds   share
---------------------------
trace          0.412    2.8%
lower          0.331    2.2%
backend       13.847   93.1%
unaccounted    0.281    1.9%
WALL          14.871
```

These come from `jax.monitoring`, which JAX emits itself — not sampled, not estimated.

## Then: who wrote the code that produced all this?

```python
import jax, scopex

jaxpr = jax.make_jaxpr(my_fn)(x, y)
units = list(scopex.walk(jaxpr))

print(scopex.table(scopex.attribute(units, "site")))
```

```
site                          count    share
--------------------------------------------
/home/me/model/nrtl.py:312     2841    17.1%
/home/me/model/nrtl.py:272     1904    11.5%
/home/me/model/nrtl.py:292     1655    10.0%
<no-frame>                     2686    16.2%
```

That `<no-frame>` bucket is real and stays visible. Some equations genuinely have no user frame —
JAX's own `jaxpr_util.source_locations` reports the same bucket. Filling it in by borrowing the
caller's line turns a 16.2% honest gap into a reported 0% and quietly inflates whatever file
happens to be above it.

Swap the view for a different question — `"kind"`, `"transform"`, `"scope_path"`, `"file"`,
`"depth"`, or any callable you write:

```python
scopex.attribute(units, "transform")            # how much is under vmap / jvp / transpose
scopex.attribute(units, lambda u: u.kind if u.transforms else None)
scopex.crosstab(units, rows="file", cols="transform")
```

## Marking your library so the user/library split is visible

**Optional.** Everything above works on any JAX program, from any library, unmodified.

If you *maintain* a framework, marking it makes one more question answerable: of this compile, how
much came from your code and how much from the code your users wrote against it?

The contract is a **naming convention, not an API** — so your framework depends on `jax` and never
imports `scopex`:

```
<pkg>:<role>              e.g.  dflux:lib.solve
<pkg>:<role>.<detail>     e.g.  dflux:user.MyColumn.residual
```

`role` is `lib` or `user`. `pkg` namespaces it, so two marked frameworks in one program can't
collide.

Wrap your own internals as you like, and wrap the user's implementations of your interface at the
point you call them. The whole integration is one decorator:

```python
import scopex

@scopex.mark_framework("mylib", ("residual", "cell", "operator"))
class Block:
    ...
```

Or, with no scopex dependency at all — this is exactly what the decorator installs:

```python
import jax

class Block:
    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)                     # note: a REAL super(), see below
        if cls.__module__.split(".")[0] != "mylib":         # a FOREIGN subclass is a user's
            for name in ("residual", "cell", "operator"):
                fn = cls.__dict__.get(name)
                if fn is None:
                    continue
                def wrap(fn=fn, tag=f"mylib:user.{cls.__qualname__}.{name}"):
                    def marked(*a, **k):
                        with jax.named_scope(tag):
                            return fn(*a, **k)
                    return marked
                setattr(cls, name, wrap())
```

Both are tested to produce identical output. The `super()` call matters: it must be the real
zero-argument form inside a class body, or the `super(base, cls)` form closing over the *defining*
class. Writing `super(cls, cls).__init_subclass__` — where `cls` is the subclass being created —
resolves straight back into the wrapper and recurses until the stack blows.

Then:

```python
scopex.attribute(units, "split")     # user / library / <unmarked>
scopex.attribute(units, "author")    # the full nesting: Col.residual/Col.cell
scopex.attribute(units, "library")   # which of your subsystems
```

### If your library has no classes

The subclass hook only reaches frameworks whose extension point *is* subclassing. A library of
plain functions — an optimiser taking a user's update rule, an ODE solver taking a vector field, a
linear solver taking a matvec — has nothing to hook, and **the failure is not silence, it is a
wrong answer**:

```python
# measured, on a diffrax-shaped solver calling an unmarked user vector field
unmarked        split={'library': 4}          # 100% of the USER's code credited to the framework
mark_callable   split={'user': 3, 'library': 1}
```

So mark at the point of ingestion:

```python
def diffeqsolve(term, ...):
    term = scopex.mark_callable(term, "diffrax", "vector_field")
```

### The split is binary, and that has a cost

`user` vs `library` is decided by one comparison: `cls.__module__.split(".")[0] != pkg`. Ecosystem
middleware — a flaxformer module under flax, say — is *foreign*, so it reads as user-authored, and
the user's own twenty-line subclass is one row among hundreds. `scopex` warns the first time a
framework marks subclasses from more than one package root. When that fires, use `by="author"` or
`by="file"`, which keep the roots apart.

### Two frameworks at once

Marks nest and none is overwritten. A unit inside a dflux hook inside a flax module carries all
four:

```python
u.marks     # (('flax','lib','apply'), ('flax','user','UserMod.__call__'),
            #  ('dflux','lib','solve'), ('dflux','user','UserBlock.residual'))
u.packages  # ('flax', 'dflux')
```

That is why every accessor returns the full ordered sequence. A design carrying ownership in one
metadata key loses the outer namespace here, and there is no residual signal to detect the loss.

### Two rules that are not negotiable

**Never put `/` in a scope name.** JAX joins name-stack entries with `/`. Two nested scopes `"a"`
then `"b"` and one scope named `"a/b"` both render to `'a/b'`, and below the jaxpr only the rendered
string survives — so the `/` is unrecoverable. Use `.` inside names. Everything else measured safe
verbatim to optimized HLO: `:` `.` `-` `_` spaces, brackets, parens, mixed case, digits, non-ASCII.

**Deciding by package, not by file path.** `cls.__module__.split(".")[0]` survives refactors; a
file-path rule breaks the first time someone moves a module.

### What marking costs

Nothing measurable. jax 0.10.2, 60-leaf trace, 21 rounds, order-rotated and paired within round:
one scope per leaf 1.029×, two nested scopes per leaf 1.007×, with a per-round range of 0.485–1.157
— the effect is below this measurement's noise. Building the representation costs 1.7 µs/equation
unmarked, 3.5 µs/equation marked.

## Getting text out without hitting a silent trap

Every accessor below exists because the obvious call returns a plausible, empty answer. Measured on
jax 0.10.2 by counting a known scope name:

| call | found | |
|---|---|---|
| jaxpr `source_info.name_stack` | 4 | correct |
| `Lowered.as_text()` | **0** | trap — defaults to `debug_info=False` |
| `Lowered.as_text(debug_info=True)` | 4 | correct |
| `Lowered.compiler_ir('stablehlo')` | **0** | trap — drops locations |
| `Lowered.compiler_ir('hlo')` | **0** | trap — drops metadata |
| `Compiled.as_text()` | 9 | correct |
| `executable.hlo_modules()[0].to_string()` | 9 | correct |

```python
scopex.stablehlo_text(lowered)     # StableHLO WITH locations
scopex.hlo_text(compiled)          # optimized HLO WITH metadata
scopex.check_env()                 # warns about settings that make a measurement lie
```

Two more, both of which have produced wrong conclusions:

- `TF_CPP_VMODULE` alone is a **silent no-op** — importing jax sets `TF_CPP_MIN_LOG_LEVEL=1`, which
  suppresses all VLOG. Use `scopex.vmodule_env()`, which sets both, in a *subprocess* environment.
- `optimization_barrier` is **erased** before the optimized HLO exists. Counting it there always
  returns 0 and is not a survival check; count it pre-optimization.

## Going deeper than the stage split

`record()` gives you one `backend` number covering HLO passes, autotuning and codegen — which want
opposite responses. Two instruments split it, and **both fail silently if enabled at the wrong
moment**, so scopex guards both.

### Per-pass timings — needs a subprocess

```python
r = scopex.pass_timings("""
import jax, jax.numpy as jnp
jax.jit(my_fn).lower(x).compile()
""")
r["passes"]     # {'simplification': 0.00117, 'layout-assignment': 0.00065, ...}  93 passes
r["n_lines"]    # 832 -- if this is ~0, vmodule never took effect
```

A subprocess is not laziness. `TF_CPP_VMODULE` is read by the C++ logging layer when the shared
library loads — during `import jax`. Measured: set before the import, **829 log lines**; set after,
**0**. There is no in-process route. And `TF_CPP_VMODULE` alone does nothing regardless, because
importing jax sets `TF_CPP_MIN_LOG_LEVEL=1`, which suppresses every VLOG — `vmodule_env()` sets both.

### XLA's own dumps — in-process, but only before the first compile

```python
with scopex.dump() as d:          # RAISES if the backend is already up
    jax.jit(my_fn).lower(x).compile()
# d holds before/after HLO per pass, and on GPU the priority-fusion decision log
```

`XLA_FLAGS` is read when the XLA backend is first initialised. Measured: set before `import jax`,
**30 dump files**; set after the import but before any compile, **30**; set after the first
compile, **0 — silently**. So `dump()` raises rather than hand you an empty directory you would
read as "nothing to see".

`dump()` also warns that the priority-fusion decision log is **GPU-only** — 77 dump files on GPU
including `priority_fusion_dump.txt`, 27 on CPU without it. Its absence is not evidence that no
fusion happened.

Equivalent env vars, if you would rather set them yourself:

```bash
XLA_FLAGS="--xla_dump_to=/tmp/d"                    # before the first compile
TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE=hlo_pass_pipeline=1   # before `import jax`
```

## What it will not tell you

- **It does not rank lines by compile seconds.** It attributes *structure* — equations,
  instructions, fusion decisions. Time attribution below the stage split needs per-pass timing, and
  even that does not divide cleanly by source line.
- **Correct attribution is not actionable attribution.** On a known scatter pathology all eight
  scatters attribute to one source line, and per-instruction interventions on that line span a
  19.6× range in effect. The line is 100% correct and 0% sufficient.
- **It will not find your pathology for you.** It builds the map. Deciding what to change is still
  yours, and interventions we have tried (barrier placement, region bisection) declined on every
  real program we tested them on.

## Correctness

`scopex.walk` is verified equal in count to `jax._src.jaxpr_util.all_eqns(revisit_inner_jaxprs=True)`
on every corpus case, including a payload buried inside `pjit`, `pjit×3`, `scan`, nested `scan`,
`cond`, `switch`, `while`, `fori`, `custom_jvp`, `custom_vjp`, `remat`, `named_scope`, `vmap`,
`grad(pjit)` and `grad(scan)` — 16/16 attribute an identical payload to the same source line.

```python
ours, jaxs, equal = scopex.verify_parity(jaxpr)
```

Run it on your own program. If it disagrees, that is a bug in `scopex`.

## License

MIT.
