"""The emitter level, opcode deltas, and the pre-optimization module.

The parser half runs with no jax at all -- it replays frozen captures through ``scopex._parse``,
so it separates "somebody edited the parser" from "the compiler moved". The live half compiles one
small program in a fresh subprocess, because ``scopex.dump`` needs a cold backend and a test that
skips itself depending on suite ordering is not a test.
"""

from __future__ import annotations

import pytest

from scopex import _parse

# ── the filename grammar: the ways it has actually been wrong ────────────────────────────────────


def test_emitter_dump_name_keeps_a_dot_disambiguated_kernel():
    """XLA appends `.<n>` when kernel names collide. Reading the kernel as dot-free reported the
    disambiguator as the kernel, on 22 files of one real dump, with no error."""
    d = _parse.emitter_dump_name(
        "module_0000.jit__f_scan.__compute_module_add_bitcast_fusion.265-pre-optimization.mlir")
    assert d == {"module": "module_0000.jit__f_scan", "kind": "pre-optimization",
                 "kernel": "__compute_module_add_bitcast_fusion.265"}


def test_emitter_dump_name_keeps_a_hyphenated_kernel():
    d = _parse.emitter_dump_name(
        "module_0002.jit_scatter.wrapped_reduce-window_kernel_module-post-lowering.mlir")
    assert d["kernel"] == "wrapped_reduce-window_kernel_module"


def test_emitter_dump_name_rejects_a_per_pass_hlo_snapshot():
    """The two grammars share their first two fields; overlap would report a PASS as a kernel."""
    assert _parse.emitter_dump_name(
        "module_0004.jit_top.0004.fusion.after_pipeline-start.before_priority-fusion.txt") is None


@pytest.mark.parametrize("name,kind", [
    ("module_0002.jit_scatter.wrapped_scatter.mlir-passes.log", "passes-log"),
    ("module_0002.jit_scatter.wrapped_scatter.ir-no-opt.ll", "ir-no-opt"),
    ("module_0002.jit_scatter.obj-file.wrapped_scatter.o", "obj"),
    ("module_0002.jit_elem.1.ptx", "ptx"),
])
def test_emitter_dump_name_kinds(name, kind):
    assert _parse.emitter_dump_name(name)["kind"] == kind


# ── the MLIR pass log ────────────────────────────────────────────────────────────────────────────


def test_mlir_pass_dumps_reads_every_header_shape():
    d = _parse.mlir_pass_dumps(_parse.SAMPLE_MLIR_PASS_LOG)
    assert [x.pass_name for x in d] == [
        "SimplifyArithPass", "Inliner", "xla::cpu::ModuleCallbackPass"]
    assert "pipeline=inline{" in d[1].pass_spec          # nested braces survive
    assert [x.scope for x in d] == ["func.func", "builtin.module", "builtin.module"]


def test_mlir_op_lines_counts_the_forms_that_have_no_result():
    ops = _parse.mlir_op_lines(_parse.SAMPLE_MLIR_PASS_LOG)
    assert "llvm.return" in ops and "return" in ops     # zero-operand terminators
    assert "scf.while" in ops                           # `%1:2 = ` multi-result
    assert not any(o.startswith("#") for o in ops)      # attribute aliases are not operations


def test_a_torn_log_is_damaged_input_and_not_a_broken_parser():
    """XLA writes this log torn on multi-kernel modules -- measured, 2 of 19 real logs, 582 and
    560 NUL bytes. The reader must lose exactly the torn snapshot and SAY it did."""
    dmg = _parse.mlir_log_damage(_parse.SAMPLE_MLIR_PASS_LOG_TORN)
    assert dmg["torn"] == 1 and dmg["nul_bytes"] == 1
    assert len(_parse.mlir_pass_dumps(_parse.SAMPLE_MLIR_PASS_LOG_TORN)) == 2
    assert _parse.mlir_log_damage(_parse.SAMPLE_MLIR_PASS_LOG)["torn"] == 0


def test_a_moved_header_format_still_raises():
    """The torn-log allowance must not become a licence to under-report. A header that is COMPLETE
    but no longer matches is a parser bug and has to stay loud."""
    bad = _parse.SAMPLE_MLIR_PASS_LOG.replace("operation: @", "operation ON @")
    with pytest.raises(_parse.ParseError):
        _parse.mlir_pass_dumps(bad)


# ── the flag ─────────────────────────────────────────────────────────────────────────────────────


def test_dump_flags_sends_the_dump_kind_and_never_a_kernel_name():
    """--xla_dump_emitter_re is partial-matched against the fixed tag `mlir-fusion`, not against a
    kernel name; forwarding a user string yields an empty emitter level and no error."""
    from scopex.flags import dump_flags
    f = dump_flags("/tmp/x", fusion=False, emitter=True)["XLA_FLAGS"]
    assert "--xla_dump_emitter_re=mlir-fusion" in f
    assert dump_flags("/tmp/x", fusion=False)["XLA_FLAGS"].count("emitter") == 0


# ── the live half ────────────────────────────────────────────────────────────────────────────────
#
# In a SUBPROCESS, and not because it is slow. XLA reads its dump flags when the backend is first
# initialised, so `scopex.dump` needs a cold process -- and a `skipif(backend_initialized())` would
# evaluate at COLLECTION time, pass on its own, and then quietly skip whenever the suite happens to
# run another test first. A test that disappears depending on ordering is the same failure mode
# this package exists to prevent, so it gets its own process and always runs.

_LIVE_SRC = r'''
import os, sys, tempfile
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import scopex
import scopex.levels
import scopex.artifacts
import scopex.emitters
from scopex.artifacts import boundaries_in
from scopex.emitters import emitter_files  # noqa: F401
from scopex.emitters import emitter_growth  # noqa: F401
from scopex.emitters import emitter_summary  # noqa: F401
from scopex.artifacts import hlo_at
from scopex.artifacts import opcode_census  # noqa: F401
from scopex.artifacts import opcode_delta  # noqa: F401
from scopex.levels import pre_optimization_hlo  # noqa: F401
from scopex.levels import pre_optimization_text  # noqa: F401
import jax
import jax.numpy as jnp

def case(x):                       # a gather chain: reaches the MLIR emitters on CPU
    idx = jnp.arange(8)
    y = x
    for _ in range(3):
        y = y[idx][:, idx]
    return y.sum()

# numpy, not jnp: creating a jax array initialises the backend, and XLA reads its dump flags
# exactly once, when the backend comes up. `jnp.ones` here would empty the dump silently.
x = np.ones((8, 8), np.float32)
d = tempfile.mkdtemp(prefix="scopex-test-emit-")

with scopex.dump(d, passes=".*", fusion=False, emitter=True):
    jax.jit(case).lower(x).compile()

inv = scopex.emitters.emitter_files(d)
assert inv["n_files"] > 0, "no emitter files at all -- did --xla_dump_emitter_re move?"
ks = scopex.emitters.emitter_growth(d)
assert ks and all(k.module.startswith("module_") for k in ks)
assert any(k.steps for k in ks), "every kernel came back with an empty MLIR pass curve"
s = scopex.emitters.emitter_summary(d)
assert s["distinct_passes"] > 10, s

# a dump compared with ITSELF must be exactly zero everywhere -- the identity check that catches a
# boundary selector silently reading two different modules.
dd = scopex.artifacts.opcode_delta(d, d, at="after")
assert all(delta == 0 for _, _, _, delta in dd["delta"]), dd["delta"]
assert not dd["case_only"] and not dd["control_only"]
assert dd["case_total"] == dd["control_total"] > 0

bs = boundaries_in(d)
assert bs[0] == "before" and bs[-1] == "after" and len(bs) > 2, bs
assert len(hlo_at(d, "before")) and len(hlo_at(d, "after"))
try:
    hlo_at(d, "no-such-pass-anywhere")
except KeyError:
    pass
else:
    raise AssertionError("hlo_at silently fell back to a boundary nobody asked for")

# the pre-optimization module needs no dump, and carries provenance the obvious accessor throws away
m = scopex.levels.pre_optimization_hlo(case, x)
t = m.to_string()
assert "op_name=" in t and "stack_frame_id=" in t
lossy = jax.jit(case).lower(x).compiler_ir("hlo").as_hlo_text()
assert "op_name=" not in lossy, "as_hlo_text() has started keeping metadata; update the docs"
units = list(scopex.walk_hlo(t, level="hlo_pre"))
assert units and any(u.path for u in units)
assert scopex.artifacts.opcode_census(m)["gather"] >= 1

# TRAPS['barrier_erased'] says to count opt-barrier in the PRE-optimization module. That now needs
# no dump, and the opcode has to spell itself the way XLA prints it.
def barrier(v):
    return jnp.sum(jax.lax.optimization_barrier(v))
assert scopex.artifacts.opcode_census(scopex.levels.pre_optimization_text(barrier, x))["opt-barrier"] == 1

print("LIVE-OK", len(ks), s["distinct_passes"], len(bs))
'''


def test_live_emitter_level_and_opcode_delta_and_pre_optimization():
    import subprocess
    import sys
    p = subprocess.run([sys.executable, "-c", _LIVE_SRC], capture_output=True, text=True,
                       timeout=600)
    assert "LIVE-OK" in p.stdout, (p.stdout or "") + "\n" + p.stderr[-4000:]
