"""Loader for the degenerate corpus, so a recipe's ``__main__`` is one line.

Every file in ``tests/degenerate/`` exposes ``CASES: {name: (fn, args, note)}``. That is the only
thing this module knows. It exists so each recipe can say

    fn, args = _cases.load("stackcond_n3000")

and stay a recipe rather than becoming a test harness.

``run_in_subprocess`` is here for the same reason and is NOT a convenience. Three scopex
instruments are once-per-process by construction:

* :func:`scopex.dump` RAISES if XLA's backend is already up, because ``XLA_FLAGS`` is read when the
  backend is first initialised and setting it later is a silent no-op. Two arms => two processes.
* :func:`scopex.pass_timings` already forks for you (``TF_CPP_VMODULE`` is read at ``import jax``).
* :func:`scopex.record` registers a ``jax.monitoring`` listener that cannot be removed. Measured on
  jax 0.10.2: this does NOT inflate later readings, because each call's listener writes into its
  own accumulator -- three consecutive records on the same program read total/wall 0.92, 0.96, 0.96.
  So record IS safe to call twice in one process, and the recipes here do. What is NOT safe is a
  hand-rolled timing loop without ``jax.clear_caches()``; see ``stage_split.py``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import sys

CORPUS = pathlib.Path(__file__).resolve().parents[2] / "tests" / "degenerate"

_MODS: dict[str, object] = {}


def _module(path: pathlib.Path):
    if str(path) not in _MODS:
        spec = importlib.util.spec_from_file_location(f"corpus_{path.stem}", path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _MODS[str(path)] = m
    return _MODS[str(path)]


def find(name: str) -> pathlib.Path:
    """The corpus file defining ``name``. Raises with the candidates if there is none."""
    for f in sorted(CORPUS.glob("case_*.py")):
        # Cheap text pre-filter: importing all 62 case files costs seconds and one of them
        # precomputes an einsum path.
        stem = name.rstrip("0123456789").rstrip("_")
        if stem.split("_")[0] not in f.read_text():
            continue
        try:
            if name in getattr(_module(f), "CASES", {}):
                return f
        except Exception:
            continue
    for f in sorted(CORPUS.glob("case_*.py")):
        try:
            if name in getattr(_module(f), "CASES", {}):
                return f
        except Exception:
            continue
    raise KeyError(f"no case named {name!r} under {CORPUS}")


def load(name: str):
    """``(fn, args)`` for a corpus case. ``args`` is a tuple, ready for ``fn(*args)``."""
    fn, args, _note = _module(find(name)).CASES[name]
    return fn, tuple(args) if isinstance(args, (list, tuple)) else (args,)


def note(name: str) -> str:
    return _module(find(name)).CASES[name][2]


# ── running one arm in its own process ──────────────────────────────────────────────────────────
# The child prints one line beginning with the sentinel and nothing else is parsed, so XLA's own
# chatter on stderr cannot be mistaken for data.
SENTINEL = "__SCOPEX_RECIPE__"

_PREAMBLE = f'''
import json, os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, {str(pathlib.Path(__file__).resolve().parent)!r})
import jax
jax.config.update("jax_enable_x64", True)
import scopex, _cases
def emit(d):
    print({SENTINEL!r} + json.dumps(d, default=str))
'''


def run_in_subprocess(body: str, *, platform: str = "cpu", timeout: int = 1800,
                      env_extra: dict | None = None) -> dict:
    """Run ``body`` in a fresh interpreter; return whatever it passed to ``emit``.

    ``body`` may use ``jax``, ``scopex``, ``_cases`` and ``emit``. Raises with the child's stderr
    tail if it never emitted -- an empty dict here would be exactly the silent-failure mode this
    package exists to avoid.
    """
    env = dict(os.environ)
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    if platform:
        env["JAX_PLATFORMS"] = platform
    env.update(env_extra or {})
    p = subprocess.run([sys.executable, "-c", _PREAMBLE + body],
                       capture_output=True, text=True, timeout=timeout, env=env)
    for line in p.stdout.splitlines():
        if line.startswith(SENTINEL):
            return json.loads(line[len(SENTINEL):])
    raise RuntimeError(f"child emitted nothing (rc={p.returncode}).\n"
                       f"stderr tail:\n{p.stderr[-2000:]}\nstdout tail:\n{p.stdout[-800:]}")


def src(name: str, *, tail: str = "") -> str:
    """Python SOURCE that cold-compiles corpus case ``name``.

    :func:`scopex.pass_timings` takes source text and not a function, and that is not an oversight:
    ``TF_CPP_VMODULE`` is read by the C++ logging layer when the shared library loads, i.e. during
    ``import jax``, so the compile has to happen in a process that does not exist yet. A live python
    closure cannot be sent there. Every recipe built on ``pass_timings`` therefore takes source or a
    corpus name rather than ``(fn, args)``.
    """
    return (f'import os, importlib.util\n'
            f'os.environ.setdefault("JAX_ENABLE_X64", "1")\n'
            f'import jax\n'
            f'jax.config.update("jax_enable_x64", True)\n'
            f'spec = importlib.util.spec_from_file_location("case_mod", {str(find(name))!r})\n'
            f'm = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n'
            f'fn, args, _ = m.CASES[{name!r}]\n'
            f'jax.jit(fn).lower(*args).compile()\n' + tail)


def backend_seconds(name: str, *, platform: str = "cpu", timeout: int = 1800) -> dict:
    """``scopex.record`` on one corpus arm, in a fresh process. The denominator for coverage."""
    return run_in_subprocess(
        f'fn, args = _cases.load({name!r})\n'
        f't = scopex.record(fn, *args)\n'
        f'emit({{k: round(t.get(k, 0.0), 4) for k in ("trace", "lower", "backend", "wall")}} '
        f'| {{"matched": t.matched, "regime": scopex.regime(t)}})\n',
        platform=platform, timeout=timeout)


def gpu_busy() -> tuple[bool, str]:
    """``(busy, description)`` from nvidia-smi. Recipes derived on GPU refuse to publish absolute
    seconds measured while somebody else holds the device."""
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        used, total, util = (int(x) for x in q.stdout.strip().splitlines()[0].split(","))
        a = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        pids = [x.strip() for x in a.stdout.splitlines() if x.strip()]
        d = f"GPU {used}/{total} MiB, {util}% util, {len(pids)} other compute process(es)"
        return (bool(pids) or used / max(1, total) > 0.15), d
    except Exception as e:                                                   # pragma: no cover
        return False, f"no nvidia-smi ({type(e).__name__})"
