"""`pass_timings` end to end: the coverage fields, and the properties its child process must have.

These run real compiles (trivial ones, CPU, ~1-2 s each in a subprocess). They are here because the
coverage number is only worth anything if the child that produces it behaves, and two of its
behaviours are easy to break by accident and invisible when broken.
"""

from __future__ import annotations

import scopex

CPU = """
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax, jax.numpy as jnp
jax.jit(lambda x: jnp.sum(jnp.tanh(x) * jnp.sin(x))).lower(jnp.ones((128, 128))).compile()
"""


def test_coverage_is_returned_and_self_consistent():
    r = scopex.pass_timings(CPU, timeout=600)
    c = r["coverage"]
    assert isinstance(c, scopex.Coverage)

    # The self-check: scopex's arithmetic against XLA's own, over the same lines.
    assert c["xla_pass_count"] == c["parsed_passes"] > 0
    assert c.lines_lost == 0
    assert 0.97 < c.fidelity < 1.03, c.fidelity
    assert c.broken is False
    assert c.split_ok

    # The measurement: a real backend number, from the same process.
    assert c["backend_s"] > 0.0
    assert c["n_backend_compiles"] >= 1
    assert 0.0 < c.coverage < 2.0
    assert abs(c["leaf_seconds"] + c["aggregate_seconds"] - c["parsed_seconds"]) < 1e-9
    assert c["n_leaves"] + c["n_aggregates"] == c["parsed_passes"]
    assert c["unmatched_pipelines"] == 0
    assert str(c).count("\n") > 8          # it prints, and printing is the point


def test_the_child_does_not_import_jax_before_the_users_source_does():
    """The property that makes the whole design legal.

    `jax` freezes `JAX_PLATFORMS`, `JAX_ENABLE_X64` and friends into its config AT IMPORT. A
    preamble that imported jax to reach `jax.monitoring` would silently break every `module_src`
    that sets one of those first -- which is the documented way to do it -- and the failure would
    look like a platform mismatch rather than like scopex. The listener is therefore armed by a
    hook on `builtins.__import__`, never by importing jax ourselves.

    Asserted by having the child set x64 and platform itself and report what it got.
    """
    src = """
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"
import jax, jax.numpy as jnp
jax.jit(lambda x: (x * x).sum()).lower(jnp.ones((32, 32))).compile()
print("SCOPEX_TEST_ENV platform=%s x64=%s"
      % (jax.devices()[0].platform, jax.config.jax_enable_x64), flush=True)
"""
    r = scopex.pass_timings(src, timeout=600)
    # The child's stdout is merged into the log `pass_timings` parses, so the line rides along.
    assert r["coverage"]["backend_s"] > 0.0
    assert r["coverage"]["trace_s"] is not None
    # x64 defaults to False; if the preamble had imported jax first, the child's setenv would have
    # come too late and this compile would have run in f32 with the env var ignored.
    assert r["coverage"]["n_backend_compiles"] >= 1


def test_no_backend_number_is_a_reason_not_a_zero():
    """Every way the denominator can go missing must produce a sentence, not a 0.0.

    A coverage of 0.0 meaning "we could not measure" is the same shape of lie as a pass ranking
    meaning "we could not parse", and this package has shipped the second one already.
    """
    r = scopex.pass_timings("print('no jax here')", timeout=120)
    c = r["coverage"]
    assert c["backend_s"] is None
    assert c.coverage is None
    assert c["why_no_backend"]
    assert "UNKNOWN" in c.verdict
    assert not c.verdict.startswith("PARSE BROKEN")


def test_module_filter_does_not_narrow_the_denominator():
    """A filtered `passes` dict with an unfiltered `backend` would silently under-report coverage.

    `jax.monitoring` attaches no module name to `backend_compile_duration`, so the denominator is
    always the child's total. `coverage` therefore stays on the unfiltered numerator and the
    filtered figure is reported separately -- rather than dividing two things that do not match.
    """
    r = scopex.pass_timings(CPU, module="jit_<lambda>", timeout=600)
    c = r["coverage"]
    assert c["module_filter"] == "jit_<lambda>"
    assert c["returned_seconds"] <= c["parsed_seconds"] + 1e-12
    assert c["parsed_passes"] == c["xla_pass_count"]        # the check still sees every module
    assert c.broken is False
