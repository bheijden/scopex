# Captured pass logs

Raw stderr from `scopex.pass_timings`' own child process, plus the `jax.monitoring` numbers from
**the same compile**, gzipped as JSON. They exist so `tests/test_coverage_guard.py` can demonstrate
the coverage guard firing on real compiler output rather than on a hand-written line.

| fixture | arm | platform | backend | pass lines | note |
|---|---|---|---|---|---|
| `min_unit_gpu.json.gz` | `switch_ident_1024` (jax#4453) | CUDA | 75.71 s | 299 | `copy-insertion` prints **`1.21 min`** — 1 line of 299, 95.5% of the compile |
| `convT64_dilate16_gpu.json.gz` | `convT64_dilate16` (xla#5541 / jax#17464) | CUDA | 50.89 s | 640 | autotune-bound, `autotuner` is 99.6% of backend; **21 interleaved log threads** |

Captured on jax/jaxlib 0.10.2, python 3.12, x64, RTX 4090 Laptop (sm_8.9), with `nvidia-smi`
reporting no other compute process. Only lines containing `hlo_pass_pipeline.cc:` are kept — that is
everything the parsers read, and dropping the rest cannot flatter the cross-check, because XLA's
`#called` / `cumulative:` / `max:` counters live on the kept lines themselves.

## Why two fixtures and not the one the brief asked for

`convT64_dilate16` is the arm that produced this package's worst bug: the autotuner took 72.5 s,
XLA printed `1.19 min`, the parser knew only `us`/`ms`/`s`, and `pass_timings` returned a plausible
ranking topped by a pass worth 0.12 s.

Re-measured here on an **idle** GPU, the same arm's autotuner takes 50.7 s and XLA prints `50.7 s`.
XLA switches to `min` above 60 s, so on this machine, today, that arm does not reproduce the bug at
all. The original reading was taken with ten foreign GPU processes at 100% utilisation.

That is worth stating plainly: **the same program, code and compiler produce a catastrophic silent
failure on a loaded machine and none whatsoever on an idle one.** A regression test pinned to
`convT64_dilate16` would have passed on this box throughout the entire lifetime of the bug. So the
min-unit fixture is `switch_ident_1024`, whose slowest pass clears 60 s with margin, and
`convT64_dilate16` is kept as the positive control — a compile the HLO passes really do explain,
where coverage reads 0.996 and the ranking is the answer.

`test_min_unit_fixture_actually_contains_a_min_unit_line` asserts the min-unit fixture still
contains a `min` line, so a fixture regenerated on faster hardware fails loudly instead of quietly
ceasing to test anything.

## Regenerating

Both were produced by driving `scopex.flags._child_source` directly — the same child `pass_timings`
runs, so the capture is byte-identical to what the instrument sees:

```python
import os, subprocess, sys, tempfile
sys.path.insert(0, "examples/recipes")
import _cases
from scopex import flags

fd, tmp = tempfile.mkstemp(suffix=".py")
os.write(fd, _cases.src("switch_ident_1024").encode()); os.close(fd)
env = dict(os.environ, JAX_PLATFORMS="cuda", **flags.vmodule_env("hlo_pass_pipeline=1"))
p = subprocess.run([sys.executable, "-c", flags._child_source(tmp)],
                   capture_output=True, text=True, env=env, timeout=3600)
log = p.stderr + p.stdout          # keep `hlo_pass_pipeline.cc:` lines; read backend_s from the
                                   # line starting with flags._SENTINEL
```

Compile serially and on an otherwise idle device. Absolute seconds under contention are upper
bounds; the `fidelity` cross-check is immune either way, since both of its numbers come from the
same log.
