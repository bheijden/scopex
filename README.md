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
point you call them:

```python
import jax

class Block:
    """Your framework's extension point."""
    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if cls.__module__.split(".")[0] != "mylib":        # a FOREIGN subclass is a user's
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

That is the whole integration. `scopex` offers sugar for it (`scopex.mark_subclasses("mylib",
(...))`) but the fifteen lines above are equivalent and keep your dependency list at `jax`.

Then:

```python
scopex.attribute(units, "split")     # user / library / <unmarked>
scopex.attribute(units, "author")    # the full nesting: Col.residual/Col.cell
scopex.attribute(units, "library")   # which of your subsystems
```

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

### If your library has no classes

The subclass hook only reaches frameworks whose extension point *is* subclassing. A library of
plain functions has nothing to hook, and should mark its own entry points directly or rely on the
unmarked fallback — source-line and package attribution from tracebacks, which needs no marking at
all.

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
