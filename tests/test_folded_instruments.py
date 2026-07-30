"""The three instruments folded in from the prototype round, plus the monitor fixes they forced.

    jaxpr_sharing    duplicate subexpressions -- why a small program becomes a big module
    trace_profile    WHERE inside the trace stage the seconds went
    backend_split    the backend stage split into phases, from dump-artifact mtimes

Structure follows the rest of the suite: everything that can be checked without a compile is, and
the two things that genuinely need a COLD XLA backend run in a fresh subprocess rather than skipping
themselves depending on suite ordering.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import scopex
import scopex.sharing
import scopex.tracing
import scopex.phases
from scopex.phases import backend_split  # noqa: F401
from scopex.sharing import jaxpr_sharing  # noqa: F401
from scopex.sharing import struct_hash
from scopex.tracing import trace_profile  # noqa: F401

# ══════════════════════════════════════════════════════════════════════════════════════════════
# jaxpr_sharing
# ══════════════════════════════════════════════════════════════════════════════════════════════


def _switch_jaxpr(n: int):
    import jax
    import jax.numpy as jnp
    return jax.make_jaxpr(
        lambda x: jax.lax.switch(jnp.int32(0), [lambda y: y ** 2] * n, x))(1.0)


def test_sharing_finds_n_identical_switch_branches():
    """The codegen-multiplicity family in one number.

    Measured on the corpus arm ``switch_ident_128``: 130 equations, 128 sub-jaxprs, 127 redundant
    equations in ONE value group (128 x integer_pow) and 127 redundant sub-jaxprs in ONE
    alpha-equivalence class -- i.e. the program is one branch written 128 times. ``walk`` counts
    the equations and cannot say that; this is the instrument that can.
    """
    s = scopex.sharing.jaxpr_sharing(_switch_jaxpr(8))
    assert s["n_subjaxprs"] == 8
    assert s["subjaxpr_dup"] == 7, "8 alpha-equivalent branches means 7 are redundant"
    assert len(s["subjaxpr_groups"]) == 1, "all 8 must land in ONE alpha-equivalence class"
    assert s["value_dup_eqns"] == 7
    assert s["value_groups"][0][0] == 8 and s["value_groups"][0][1] == "integer_pow"


def test_sharing_scales_with_branch_count_rather_than_reporting_a_constant():
    """A census that returned a plausible constant would be this package's signature bug."""
    a, b = scopex.sharing.jaxpr_sharing(_switch_jaxpr(4)), scopex.sharing.jaxpr_sharing(_switch_jaxpr(16))
    assert (a["subjaxpr_dup"], b["subjaxpr_dup"]) == (3, 15)


def test_sharing_reports_zero_on_a_program_with_no_duplicates():
    """Zero must be reachable, or a non-zero result means nothing."""
    import jax
    import jax.numpy as jnp
    s = scopex.sharing.jaxpr_sharing(jax.make_jaxpr(lambda x: jnp.sin(x) + jnp.cos(x) * 2.0)(1.0))
    assert s["value_dup_eqns"] == 0 and s["subjaxpr_dup"] == 0
    assert s.redundant_fraction == 0.0


def test_struct_hash_is_alpha_equivalence_not_identity():
    """Two branches differing only in variable names must hash EQUAL; differing in structure or in
    a primitive parameter must not. Both directions, because only one of them is the interesting
    failure and shipping either alone proves nothing."""
    import jax
    same_a = jax.make_jaxpr(lambda p: p * 2.0 + 1.0)(1.0)
    same_b = jax.make_jaxpr(lambda q: q * 2.0 + 1.0)(1.0)
    diff_param = jax.make_jaxpr(lambda p: p * 3.0 + 1.0)(1.0)
    diff_struct = jax.make_jaxpr(lambda p: p * 2.0)(1.0)
    assert struct_hash(same_a) == struct_hash(same_b)
    assert struct_hash(same_a) != struct_hash(diff_param)
    assert struct_hash(same_a) != struct_hash(diff_struct)


def test_sharing_rejects_a_non_jaxpr_instead_of_returning_an_empty_census():
    with pytest.raises(TypeError, match="not a jaxpr"):
        scopex.sharing.jaxpr_sharing([1, 2, 3])


# ══════════════════════════════════════════════════════════════════════════════════════════════
# trace_profile
# ══════════════════════════════════════════════════════════════════════════════════════════════


def _slow_trace_fn():
    """A function whose TRACE is the expensive part and whose runtime is not.

    Python-level work inside the traced function, so the cost is unambiguously in tracing rather
    than in lowering or in XLA.
    """
    import jax.numpy as jnp

    def fn(x):
        acc = x
        for _ in range(120):
            acc = jnp.sin(acc) + jnp.cos(acc)
        return acc.sum()
    return fn


@pytest.mark.parametrize("method", ["xplane", "cprofile"])
def test_trace_profile_returns_a_populated_profile(method):
    import numpy as np
    p = scopex.tracing.trace_profile(_slow_trace_fn(), np.ones((4,), np.float32), method=method)
    assert p.method == method, "the method must be visible in the result -- the two are not equal"
    assert p.n_events > 0
    assert p.frames, "a profile with no frames is indistinguishable from an instant trace"
    assert p.traced_wall > 0.0
    name, secs = p.top_frame
    assert name and secs >= 0.0


def test_trace_profile_charges_time_to_a_named_frame_rather_than_one_bucket():
    """The point of the instrument: real frame names, not a single total."""
    import numpy as np
    p = scopex.tracing.trace_profile(_slow_trace_fn(), np.ones((4,), np.float32))
    assert len(p.frames) > 5
    assert any("jax" in f[0] or ".py" in f[0] for f in p.frames[:10])


def test_cprofile_method_has_absolute_paths_and_declares_it_has_no_sites():
    """cProfile keeps no stacks, so ``sites`` is empty BY CONSTRUCTION -- and the object says so
    rather than silently returning an empty table that reads as 'no user code was involved'."""
    import numpy as np
    p = scopex.tracing.trace_profile(_slow_trace_fn(), np.ones((4,), np.float32), method="cprofile")
    assert p.sites == []
    assert "no stacks" in str(p)
    assert any(f[0].startswith("/") for f in p.frames), "cProfile records full absolute paths"


def test_trace_profile_rejects_an_unknown_method():
    import numpy as np
    with pytest.raises(ValueError, match="xplane"):
        scopex.tracing.trace_profile(_slow_trace_fn(), np.ones((4,), np.float32), method="perf")


def test_a_missing_xplane_plane_raises_instead_of_reporting_zeros():
    """The guard that matters. Every per-frame number would read 0.0, which is exactly what an
    instant trace looks like, so the instrument must refuse rather than return."""
    from scopex import tracing

    class _FakePlane:
        name = "/device:GPU:0"
        lines = ()

    class _FakePD:
        planes = (_FakePlane(),)

        @staticmethod
        def from_file(_p):
            return _FakePD()

    import jax.profiler
    real = jax.profiler.ProfileData
    jax.profiler.ProfileData = _FakePD
    try:
        with pytest.raises(tracing.TraceProfileError, match="host:CPU"):
            tracing._python_line("nonexistent.pb")
    finally:
        jax.profiler.ProfileData = real


def test_a_plane_without_a_python_line_raises():
    """python_tracer_level failing to take effect must not read as an instant trace."""
    from scopex import tracing

    class _Line:
        name = "XLA Modules"

    class _Host:
        name = "/host:CPU"
        lines = (_Line(),)

    class _FakePD:
        planes = (_Host(),)

        @staticmethod
        def from_file(_p):
            return _FakePD()

    import jax.profiler
    real = jax.profiler.ProfileData
    jax.profiler.ProfileData = _FakePD
    try:
        with pytest.raises(tracing.TraceProfileError, match="python_tracer_level"):
            tracing._python_line("nonexistent.pb")
    finally:
        jax.profiler.ProfileData = real


class _E:
    def __init__(self, s, e, n):
        self.start_ns, self.end_ns, self.name = s, e, n


def test_self_time_is_recursion_safe_where_inclusive_time_is_not():
    """A frame that calls ITSELF is the case that breaks naive accounting, and it is the shape of
    every recursive trace in the corpus (``fib``, ``_optimal_iterate``).

    Outer ``f`` spans 0-100 and contains an inner ``f`` spanning 10-60. Aggregated by name, ``f``
    exclusively occupies 100 ns of wall -- outer contributes 50 (its span minus its child) and inner
    contributes 50. INCLUSIVE time double-counts the overlap to 150, which is why self time is the
    number this instrument ranks on.
    """
    from scopex.tracing import _walk_events
    self_ns, incl_ns, calls, _us, _ui = _walk_events(
        [_E(0, 100, "$f.py:1 f"), _E(10, 60, "$f.py:1 f")], set())
    assert self_ns["$f.py:1 f"] == 100, "self time must equal the exclusive wall span"
    assert incl_ns["$f.py:1 f"] == 150, "inclusive time double-counts recursion -- by design"
    assert calls["$f.py:1 f"] == 2


def test_self_time_is_exclusive_of_children():
    """The other direction: a child's time must be subtracted from its parent, or every caller
    would rank above the callee that is actually slow."""
    from scopex.tracing import _walk_events
    self_ns, _incl, _calls, _us, _ui = _walk_events(
        [_E(0, 100, "$a.py:1 caller"), _E(10, 60, "$b.py:1 callee")], set())
    assert self_ns["$a.py:1 caller"] == 50
    assert self_ns["$b.py:1 callee"] == 50


def test_time_is_charged_to_the_nearest_enclosing_user_frame():
    """The per-site table: a library frame's cost belongs to the line of YOUR code that called it.
    This is what charged 94% of the einsum case to opt_einsum while naming the user site."""
    from scopex.tracing import _walk_events
    _s, _i, _c, user_self, _ui = _walk_events(
        [_E(0, 100, "$mine.py:7 build"), _E(10, 90, "$lib.py:3 solve")], {"lib.py"})
    assert user_self["$mine.py:7 build"] == 100, "library self time rolls up to the user frame"
    assert "$lib.py:3 solve" not in user_self


def test_library_misclassification_is_one_directional():
    """The xplane tracer records only a BASENAME, so a resolver would be wrong ~half the time. The
    rule instead never charges library time to user code; it may under-report user code."""
    from scopex.tracing import _is_user
    assert _is_user("$mycode.py:12 fn", set()) is True
    assert _is_user("$core.py:12 fn", {"core.py"}) is False
    assert _is_user("$builtins len", set()) is False, "no ':' means no file, so not user code"
    assert _is_user("not-a-frame", set()) is False


# ══════════════════════════════════════════════════════════════════════════════════════════════
# backend_split
# ══════════════════════════════════════════════════════════════════════════════════════════════


def test_kernel_key_pairs_the_three_phases_of_one_kernel():
    """XLA names the .ll and the .o of the SAME kernel differently (the .o gains an ``obj-file.``
    infix). If they did not pair, every program would look like it had more kernel modules than it
    has, and the interleave guard would fire on single-kernel programs -- taking away the only code
    path allowed to report ``sound=True``."""
    from scopex.phases import _kernel_key
    a = _kernel_key("module_0004.jit_program.broadcast_multiply_fusion_kernel_module.ir-no-opt.ll")
    b = _kernel_key("module_0004.jit_program.broadcast_multiply_fusion_kernel_module"
                    ".ir-with-opt.ll")
    c = _kernel_key("module_0004.jit_program.obj-file.broadcast_multiply_fusion_kernel_module.o")
    assert a[0] == b[0] == c[0] == ("module_0004.jit_program",
                                    "broadcast_multiply_fusion_kernel_module")
    assert (a[1], b[1], c[1]) == ("ir-no-opt", "ir-with-opt", "obj")


def test_kernel_key_does_not_merge_same_named_kernels_from_different_modules():
    """A dump holds JAX's warm-up modules alongside your program. Keying on the kernel name alone
    would merge them, under-count kernel modules, and could hand a multi-kernel program the
    single-kernel code path."""
    from scopex.phases import _kernel_key
    a = _kernel_key("module_0002.jit_warmup.wrapped_reduce_kernel_module.ir-no-opt.ll")
    b = _kernel_key("module_0004.jit_program.wrapped_reduce_kernel_module.ir-no-opt.ll")
    assert a[0] != b[0]


def test_kernel_key_keeps_a_dot_disambiguated_kernel_name():
    """Delegating to _parse.emitter_dump_name inherits its hardening: XLA appends `.<n>` when
    kernel names collide, and a dot-free kernel group reads `265` as the kernel -- not empty, not
    an error, just the disambiguator where the name should be."""
    from scopex.phases import _kernel_key
    k = _kernel_key("module_0000.jit__f_scan.__compute_module_add_bitcast_fusion.265.ir-no-opt.ll")
    assert k[0][1] == "__compute_module_add_bitcast_fusion.265"


def test_kernel_key_ignores_files_that_are_not_llvm_artifacts():
    from scopex.phases import _kernel_key
    assert _kernel_key("module_0004.jit_top.0004.fusion.after_x.before_y.txt") is None
    assert _kernel_key("priority_fusion_dump.txt") is None
    # An emitter MLIR dump is a real emitter artifact but not one of the three timed phases.
    assert _kernel_key("module_0002.jit_scatter.wrapped_scatter-pre-optimization.mlir") is None


def test_backend_split_refuses_when_the_backend_is_already_up():
    """The silent no-op this package exists to prevent: XLA reads --xla_dump_to when its backend is
    first initialised, so calling this late would compute a split from an EMPTY directory. By the
    time this test runs, other tests have compiled."""
    import jax.numpy as jnp
    import numpy as np
    jnp.ones(1) + 1                                   # guarantee the backend is up
    with pytest.raises(RuntimeError, match="already initialised"):
        scopex.phases.backend_split(lambda x: x * 2, np.ones((2,), np.float32))


# The live half. A COLD backend is required, so it runs in a fresh interpreter. The program is
# deliberately tiny: this asserts the SHAPE of the answer and the guards, not a pathology.
_LIVE_BS = r'''
import warnings
import numpy as np
import jax.numpy as jnp
import scopex
import scopex.sharing
import scopex.tracing
import scopex.phases

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    r = scopex.phases.backend_split(lambda x: (jnp.sin(x) * jnp.cos(x)).sum(),
                             np.ones((64, 64), np.float32))

assert r["n_pass_snapshots"] > 0, "no snapshots means the dump silently did not happen"
assert "hlo_passes" in r
assert set(r) >= {"backend", "coverage", "sound", "n_kernel_modules", "interleaved", "warnings"}
# A tiny compile CANNOT pass the coverage band -- the un-spanned head/tail is a fixed ~20-25 ms.
# That is the documented floor, and `sound` must reflect it rather than being optimistic.
assert r.sound is False, f"tiny compile claimed sound with coverage {r.coverage}"
assert r["warnings"], "coverage outside the band must warn"
assert any(w.category is RuntimeWarning for w in caught), "the warning must reach the caller"
# soundness and coverage are never allowed to disagree
assert not (r.sound and not (0.90 <= r.coverage <= 1.10))
# every reported bucket is a real span, so none may be negative
assert all(r[k] >= 0.0 for k in ("hlo_passes", "emitter", "llvm_opt", "codegen", "below_hlo")
           if k in r)
assert r.top[0] in ("hlo_passes", "emitter", "llvm_opt", "codegen", "below_hlo")
print("LIVE-BS-OK", r["n_kernel_modules"], r["n_pass_snapshots"], round(r.coverage, 3))
'''


def test_live_backend_split_shape_and_guards():
    p = subprocess.run([sys.executable, "-c", _LIVE_BS], capture_output=True, text=True,
                       timeout=900)
    assert "LIVE-BS-OK" in p.stdout, (p.stdout or "") + "\n" + p.stderr[-4000:]


# ══════════════════════════════════════════════════════════════════════════════════════════════
# the monitor fixes these instruments forced
# ══════════════════════════════════════════════════════════════════════════════════════════════


def test_record_removes_its_listeners():
    """monitor.py stated in two places that jax.monitoring has no public deregister. It does --
    ``unregister_event_duration_listener`` (note: no ``_secs``, unlike the register function) and
    ``unregister_event_listener``. Verified by count across a record()."""
    import numpy as np
    from jax._src import monitoring as _m
    before = (len(_m.get_event_duration_listeners()), len(_m.get_event_listeners()))
    scopex.record(lambda x: x * 2, np.ones((4,), np.float32))
    after = (len(_m.get_event_duration_listeners()), len(_m.get_event_listeners()))
    assert before == after, f"record leaked listeners: {before} -> {after}"


def test_repeated_record_does_not_accumulate_listeners():
    import numpy as np
    from jax._src import monitoring as _m
    for _ in range(3):
        scopex.record(lambda x: x + 1.0, np.ones((4,), np.float32))
    assert len(_m.get_event_duration_listeners()) == 0


def test_trace_looks_blind_fires_on_the_trace_state_clean_signature():
    """jax emits jaxpr_trace_duration only under core.trace_state_clean(), so a top-level
    vmap/grad over a jitted callee makes the metric a silent 0.0 while tracing really happens.
    Measured on jax 0.10.2/CPU with a 10-operand einsum(optimize='optimal'):
    jit().lower() -> wall 2.518 s / trace 2.511 s, but vmap() -> wall 2.711 s / trace 0.001 s.
    ``matched`` stays True, so nothing else in the module notices."""
    t = scopex.Timings({"trace": 0.001, "lower": 0.03, "backend": 0.03, "wall": 2.711})
    assert t.matched is True, "the point is that the existing guard does NOT fire"
    assert t.trace_looks_blind is True
    assert "trace_profile" in str(t)


def test_trace_looks_blind_stays_quiet_on_a_healthy_split():
    healthy = scopex.Timings({"trace": 2.511, "lower": 0.004, "backend": 0.003, "wall": 2.518})
    assert healthy.trace_looks_blind is False
    backend_bound = scopex.Timings({"trace": 0.008, "lower": 0.029, "backend": 6.42, "wall": 6.60})
    assert backend_bound.trace_looks_blind is False, "backend-bound is not trace-blind"
    tiny = scopex.Timings({"trace": 0.0, "lower": 0.0, "backend": 0.0, "wall": 0.0})
    assert tiny.trace_looks_blind is False
