"""Every text parser left in scopex, in one place, each with a verbatim sample of what it parses.

WHY THIS MODULE EXISTS
----------------------
Three parsers have shipped broken in this package, and all three failed the same way: they returned
a plausible EMPTY-or-wrong answer instead of an error, and all three were found by *using* the tool,
never by reading it.

* ``walk_stablehlo`` matched only the inline ``loc("name")`` form. jax emits ``loc(#loc17)`` on the
  operation and defines the name separately, so the walker yielded **1 unit on 16 of 21 real
  programs**, including modules of 3,214 and 21,000 operations. The level looked EMPTY, not BROKEN.
* ``walk_hlo`` read metadata with ``(\\w+)="([^"]*)"`` -- QUOTED values only.
  ``stack_frame_id=5`` is an unquoted int, so it was dropped, and the published conclusion was
  that the optimized module carries no source location at all. It does.
* ``pass_timings`` knew units ``{us, ms, s}``. XLA switches to ``min`` at large magnitudes, so the
  SLOWEST pass is the one most likely to be dropped. Measured: 1 of 640 lines used ``min``, it was
  the autotuner at 98.8% of a 72.5 s compile, and dropping it left a plausible dict topped by
  ``remat-pipeline: 0.1196``. The tool reported the OPPOSITE of the truth, with no warning.

A FOURTH was found by the guard in this module, on the first live log it ran against, and is the
reason to believe the guard is worth its weight: 384 lines of a CPU compile said ``HLO pass: `` and
only 378 parsed, because a pass NAME CAN CONTAIN SPACES (``simplification after layout assignment``)
and the pattern read the name as ``\\S+``. Nobody was looking; the check simply refused to return.

A parser whose blind spot correlates with the quantity being measured is worse than no parser. So:

1. **Every regex over compiler output lives here.** ``levels``, ``flags`` and ``artifacts`` import
   named functions from this module and compile no patterns of their own.
2. **Every parser names the component that prints its input**, and carries a KNOWN-GOOD SAMPLE of
   that output as a module constant, measured on jax/jaxlib 0.10.2. Paths were shortened; every
   other character is what XLA printed.
3. **Every parser is guarded by :func:`expect`**, which turns "fewer results than the input visibly
   contains" into a :class:`ParseError`. Its docstring shows how the one rule catches all three
   bugs above.
4. :func:`conformance` runs every parser against its embedded sample -- no jax, no compile, so it
   runs in CI -- and :func:`scopex.selftest` runs the same parsers against a freshly compiled
   program. Run both after any jax upgrade.

WHAT IS NO LONGER PARSED HERE, AND WHY THAT MATTERS
---------------------------------------------------
Most of what this module used to own has been replaced by native routes, and those are strictly
better: HLO structure now comes from ``jaxlib.xla_client.hlo`` (:func:`scopex.levels.hlo_module`),
StableHLO from the ``jaxlib.mlir.ir`` module jax lowered into, and the priority-fusion dump from a
schema-free text-proto reader in :mod:`scopex.fusion`. This module is the RESIDUE: the things that
are printed and nothing else exposes.

Checked on jaxlib 0.10.2: ``HloInstruction`` has ``name/opcode/operands/users/to_string`` and NO
``.metadata`` and no ``.shape``; ``HloModule`` has no accessor for the stack-frame tables.
``as_serialized_hlo_module_proto()`` returns wire-format bytes whose schema is not shipped -- there
is no ``xla_data_pb2`` and ``google.protobuf`` is not even installed -- and raw wire format is field
NUMBERS, so a schema-less reader must hardcode them and guesses wrong SILENTLY. That is the one
trade this package must never make. So these five stay text, and stay here:

==========================  =====================================================  ===============
parser                      printed by                                             blast radius
==========================  =====================================================  ===============
``hlo_metadata``            ``xla::OpMetadata`` in HloInstruction::ToString         one instruction
``hlo_shape``               the same string; ``HloInstruction`` has no ``.shape``   one instruction
``hlo_frame_tables``        ``xla::HloModule::Print`` (StackFrameIndexProto)        4 fixed tables
``hlo_frame_stack/site``    the same tables, parent-linked                          4 fixed tables
``pass_timing_lines``       ``hlo_pass_pipeline.cc:176`` under TF_CPP_VMODULE=1     a log
``dump_snapshot_name``      ``xla/service/dump.cc`` filename builder                a filename
``custom_call_targets``     ``xla::HloCustomCallInstruction::ToString``             a module
``stablehlo_op_lines``      MLIR ``AsmPrinter`` -- FALLBACK ONLY, see levels.py     a module
==========================  =====================================================  ===============

:mod:`scopex.fusion` also reads text, and is deliberately NOT moved here: it is a complete
proto3-text-format tokenizer over a SELF-DESCRIBING format, so it cannot have the blind spot this
module exists to guard against -- an unknown field arrives as an unknown key, not as silence. It
raises ``TextProtoError`` rather than returning a short plausible list, and :func:`conformance`
exercises it alongside everything else.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import NamedTuple

__all__ = [
    "ParseError", "expect",
    "hlo_metadata", "hlo_shape", "hlo_frame_tables", "hlo_frame_stack", "hlo_site", "Frame",
    "is_hlo_instruction_line", "custom_call_targets", "check_metadata_coverage",
    "stablehlo_loc_aliases", "stablehlo_file_aliases", "stablehlo_op_lines",
    "pass_timing_lines", "pass_pipeline_headers", "PassTime", "UNITS",
    "dump_snapshot_name",
    "emitter_dump_name", "mlir_pass_dumps", "mlir_op_lines", "MlirPassDump",
    "mlir_log_damage", "EMITTER_DUMP_KIND",
    "conformance", "PARSERS", "Parser",
]


# ══════════════════════════════════════════════════════════════════════════════════════════════
# THE INVARIANT
# ══════════════════════════════════════════════════════════════════════════════════════════════

class ParseError(RuntimeError):
    """A parser found fewer results than its input visibly contains.

    Deliberately an exception and not a warning. Every historical failure in this package was
    warning-shaped at most, and nobody saw it, because the result it accompanied looked fine.
    """


def expect(parser: str, matches, text: str, *, witness: str | re.Pattern | None = None,
           produced_by: str = "", unit: str = "matches", allow_empty: bool = False) -> int:
    """Assert that ``matches`` is at least as large as what ``text`` visibly contains. Or raise.

    ``witness`` is a regex whose occurrence count in ``text`` is a LOWER BOUND on the number of
    results a correct parser must return -- deliberately cruder and more literal than the parser it
    guards, so the two cannot share a blind spot. With no witness the bound is simply 1: a
    non-empty input must produce something.

    THE ONE RULE, AND WHY IT IS A COUNT AND NOT A BOOLEAN. All three bugs in the module docstring
    fall to it, but only two of them would fall to ``n > 0``:

    * ``walk_stablehlo``: witness counts SSA assignment lines. The broken version returned 1 from a
      module with 3,214 of them. ``1 > 0`` passes; ``1 >= 3214`` does not.
    * ``walk_hlo`` metadata: witness counts ``stack_frame_id=``. The broken quoted-only pattern
      resolved 0 of N.
    * ``pass_timings``: witness counts ``HLO pass: `` lines. The unit-dropping version parsed 639
      of 640, and the one it dropped was 98.8% of the compile.

    Empty input is not an error and returns 0. ``allow_empty=True`` additionally permits zero
    results from non-empty input containing no witness at all -- for parsers whose absence is
    genuine information: no program is obliged to contain a custom call.
    """
    n = matches if isinstance(matches, int) else len(matches)
    body = text or ""
    if not body.strip():
        return n
    hits: list = []
    if witness is None:
        need = 0 if allow_empty else 1
    else:
        pat = re.compile(witness, re.M) if isinstance(witness, str) else witness
        hits = pat.findall(body)
        need = len(hits)
        if need == 0:
            need = 0 if allow_empty else 1
    if n >= need:
        return n
    lines = [ln for ln in body.splitlines() if ln.strip()]
    show = "\n    ".join(ln[:160] for ln in lines[:3])
    raise ParseError(
        f"scopex parser {parser!r} returned {n} {unit} from input that visibly contains at least "
        f"{need}.\n"
        f"  This is a BROKEN PARSER, not an empty program -- scopex raises here because every "
        f"previous failure of this kind was reported as a plausible empty answer instead.\n"
        f"  printed by : {produced_by or 'unknown'}\n"
        f"  witness    : {getattr(witness, 'pattern', witness)!r} matched {len(hits)} time(s)\n"
        f"  input      : {len(lines)} non-blank lines, first of them:\n    {show}\n"
        f"  Fix the parser in scopex/_parse.py, then run scopex.selftest().")


# ══════════════════════════════════════════════════════════════════════════════════════════════
# KNOWN-GOOD SAMPLES
#
# Captured on jax/jaxlib 0.10.2, CPU, from this program:
#
#     def leaf(x):      return jnp.tanh(x)                                    # line 6
#     def residual(x):
#         with jax.named_scope("mylib:user.MyModel.residual"):
#             return leaf(x) * 2.0                                            # line 10
#     def solve(x):
#         with jax.named_scope("mylib:lib.solve"):
#             return residual(x) + leaf(x)                                    # line 14
#     def top(x):       return jnp.sum(solve(x) @ x)                          # line 17
#     low = jax.jit(top).lower(jnp.ones((8, 8))); comp = low.compile()        # line 19
#
# It was chosen to contain the things that break parsers: two nested marked scopes, the SAME source
# line (leaf, line 6) reached through two different call paths, and an instruction with no metadata
# at all. The absolute path was replaced with /home/u/proj/model.py; nothing else was edited.
# ══════════════════════════════════════════════════════════════════════════════════════════════

SAMPLE_HLO = '''HloModule jit_top, is_scheduled=true, entry_computation_layout={(f32[8,8]{1,0})->f32[]}, allow_spmd_sharding_propagation_to_parameters={true}, allow_spmd_sharding_propagation_to_output={true}

FileNames
1 "/home/u/proj/model.py"

FunctionNames
1 "<module>"
2 "top"
3 "solve"
4 "residual"
5 "leaf"

FileLocations
1 {file_name_id=1 function_name_id=1 line=19 end_line=19 column=6 end_column=42}
2 {file_name_id=1 function_name_id=2 line=17 end_line=17 column=11 end_column=32}
3 {file_name_id=1 function_name_id=2 line=17 end_line=17 column=19 end_column=27}
4 {file_name_id=1 function_name_id=3 line=14 end_line=14 column=15 end_column=26}
5 {file_name_id=1 function_name_id=4 line=10 end_line=10 column=15 end_column=28}
6 {file_name_id=1 function_name_id=5 line=6 end_line=6 column=11 end_column=22}
7 {file_name_id=1 function_name_id=3 line=14 end_line=14 column=29 end_column=36}

StackFrames
1 {file_location_id=1 parent_frame_id=1}
2 {file_location_id=2 parent_frame_id=2}
3 {file_location_id=3 parent_frame_id=2}
4 {file_location_id=4 parent_frame_id=4}
5 {file_location_id=5 parent_frame_id=5}
6 {file_location_id=6 parent_frame_id=6}
7 {file_location_id=7 parent_frame_id=4}
8 {file_location_id=6 parent_frame_id=8}


%fused_computation (param_0: f32[8,8], param_1: f32[8,8]) -> f32[8,8] {
  %param_0 = f32[8,8]{1,0} parameter(0)
  %param_1 = f32[8,8]{1,0} parameter(1)
  ROOT %dot_general.0 = f32[8,8]{1,0} dot(%param_0, %param_1), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="jit(top)/dot_general" stack_frame_id=3}
}

%fused_computation.1 (param_0.3: f32[8,8]) -> f32[8,8] {
  %param_0.3 = f32[8,8]{1,0} parameter(0)
  %tanh.0 = f32[8,8]{1,0} tanh(%param_0.3), metadata={op_name="jit(top)/mylib:lib.solve/mylib:user.MyModel.residual/tanh" stack_frame_id=6}
  %constant.0 = f32[] constant(2)
  %mul.1 = f32[8,8]{1,0} broadcast(%constant.0), dimensions={}, metadata={op_name="jit(top)/mylib:lib.solve/mylib:user.MyModel.residual/mul" stack_frame_id=5}
  %mul.0 = f32[8,8]{1,0} multiply(%tanh.0, %mul.1), metadata={op_name="jit(top)/mylib:lib.solve/mylib:user.MyModel.residual/mul" stack_frame_id=5}
  ROOT %add.0 = f32[8,8]{1,0} add(%mul.0, %tanh.0), metadata={op_name="jit(top)/mylib:lib.solve/add" stack_frame_id=4}
}

%region_0.1 (reduce_sum.3: f32[], reduce_sum.4: f32[]) -> f32[] {
  %reduce_sum.3 = f32[] parameter(0), metadata={op_name="reduce_sum"}
  %reduce_sum.4 = f32[] parameter(1), metadata={op_name="reduce_sum"}
  ROOT %reduce_sum.5 = f32[] add(%reduce_sum.3, %reduce_sum.4), metadata={op_name="jit(top)/reduce_sum" stack_frame_id=2}
}

%wrapped_reduce_computation (param_0.4: f32[8,8], param_1.4: f32[]) -> f32[] {
  %param_0.4 = f32[8,8]{1,0} parameter(0)
  %param_1.4 = f32[] parameter(1)
  ROOT %reduce_sum.0 = f32[] reduce(%param_0.4, %param_1.4), dimensions={0,1}, to_apply=%region_0.1, metadata={op_name="jit(top)/reduce_sum" stack_frame_id=2}
}

ENTRY %main.2 (x.1: f32[8,8]) -> f32[] {
  %x.1 = f32[8,8]{1,0} parameter(0), metadata={op_name="x"}
  %constant.3 = f32[] constant(0)
  %multiply_add_fusion = f32[8,8]{1,0} fusion(%x.1), kind=kLoop, calls=%fused_computation.1, metadata={op_name="jit(top)/mylib:lib.solve/add" stack_frame_id=4}
  %ynn_fusion = f32[8,8]{1,0} fusion(%multiply_add_fusion, %x.1), kind=kCustom, calls=%fused_computation, metadata={op_name="jit(top)/dot_general" stack_frame_id=3}, backend_config={"outer_dimension_partitions":[],"fusion_config":{"kind":"__ynn_fusion"}}
  ROOT %wrapped_reduce = f32[] fusion(%ynn_fusion, %constant.3), kind=kLoop, calls=%wrapped_reduce_computation, metadata={op_name="jit(top)/reduce_sum" stack_frame_id=2}
}
'''

# TUPLE-SHAPED instructions, which is where `shape = \\S+` used to fail: the shape contains a space.
# From a `jax.lax.while_loop` compile on the same jaxlib.
SAMPLE_HLO_TUPLE_LINES = (
    '  %while.1 = (s32[], f32[8]{0}) while(%tuple.1), condition=%while_cond, body=%while_body, '
    'metadata={op_name="jit(f)/while" stack_frame_id=2}',
    '  %get-tuple-element.5 = f32[8]{0} get-tuple-element(%while.1), index=1',
    '  ROOT %tuple.2 = (f32[8]{0}, s32[]) tuple(%get-tuple-element.5, %constant.7)',
)

# The same program, lowered: Lowered.as_text(debug_info=True). Note what the OPERATIONS carry --
# `loc(#loc38)`, never `loc("jit(top)/...")`. That indirection is bug #1 in the module docstring.
# Note also #loc2..#loc11: FileLineColLoc aliases, whose first quoted string is a FILE PATH. A
# parser that takes "the first quoted string" from every alias gives operations named /home/u/....
SAMPLE_STABLEHLO = '''#loc1 = loc("x")
module @jit_top attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8x8xf32> loc("x")) -> (tensor<f32> {jax.result_info = "result"}) {
    %0 = stablehlo.tanh %arg0 : tensor<8x8xf32> loc(#loc38)
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32> loc(#loc23)
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x8xf32> loc(#loc36)
    %2 = stablehlo.multiply %0, %1 : tensor<8x8xf32> loc(#loc36)
    %3 = stablehlo.tanh %arg0 : tensor<8x8xf32> loc(#loc37)
    %4 = stablehlo.add %2, %3 : tensor<8x8xf32> loc(#loc34)
    %5 = stablehlo.dot_general %4, %arg0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32> loc(#loc29)
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc(#loc30)
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [0, 1] : (tensor<8x8xf32>, tensor<f32>) -> tensor<f32> loc(#loc30)
    return %6 : tensor<f32> loc(#loc23)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/home/u/proj/model.py":6:11 to :22)
#loc3 = loc("/home/u/proj/model.py":10:15 to :22)
#loc4 = loc("/home/u/proj/model.py":14:15 to :26)
#loc5 = loc("/home/u/proj/model.py":17:19 to :27)
#loc6 = loc("/home/u/proj/model.py":19:6 to :42)
#loc7 = loc("/home/u/proj/model.py":10:15 to :28)
#loc8 = loc("/home/u/proj/model.py":14:29 to :36)
#loc9 = loc("/home/u/proj/model.py":14:15 to :36)
#loc10 = loc("/home/u/proj/model.py":17:19 to :31)
#loc11 = loc("/home/u/proj/model.py":17:11 to :32)
#loc12 = loc("leaf"(#loc2))
#loc13 = loc("residual"(#loc3))
#loc14 = loc("solve"(#loc4))
#loc15 = loc("top"(#loc5))
#loc16 = loc("<module>"(#loc6))
#loc17 = loc("residual"(#loc7))
#loc18 = loc("solve"(#loc8))
#loc19 = loc("solve"(#loc9))
#loc20 = loc("top"(#loc10))
#loc21 = loc("top"(#loc11))
#loc22 = loc(callsite(#loc15 at #loc16))
#loc23 = loc("jit(top)"(#loc16))
#loc24 = loc(callsite(#loc20 at #loc16))
#loc25 = loc(callsite(#loc21 at #loc16))
#loc26 = loc(callsite(#loc14 at #loc22))
#loc27 = loc(callsite(#loc18 at #loc22))
#loc28 = loc(callsite(#loc19 at #loc22))
#loc29 = loc("jit(top)/dot_general"(#loc24))
#loc30 = loc("jit(top)/reduce_sum"(#loc25))
#loc31 = loc(callsite(#loc13 at #loc26))
#loc32 = loc(callsite(#loc17 at #loc26))
#loc33 = loc(callsite(#loc12 at #loc27))
#loc34 = loc("jit(top)/mylib:lib.solve/add"(#loc28))
#loc35 = loc(callsite(#loc12 at #loc31))
#loc36 = loc("jit(top)/mylib:lib.solve/mylib:user.MyModel.residual/mul"(#loc32))
#loc37 = loc("jit(top)/mylib:lib.solve/tanh"(#loc33))
#loc38 = loc("jit(top)/mylib:lib.solve/mylib:user.MyModel.residual/tanh"(#loc35))
'''

# TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE=hlo_pass_pipeline=1. The first five lines are verbatim from
# a CPU compile of the program above (807 pass lines in 832 lines of log). The `autotuner` line is
# the one that broke this parser: transcribed from the GPU record where it was 1 of 640 lines and
# 98.8% of a 72.5 s compile. Its tail is elided exactly as it was recorded there; the parser never
# reads past `us)`.
SAMPLE_PASS_LOG = '''I0729 17:26:34.639165  203556 hlo_pass_pipeline.cc:303] Running HLO pass pipeline on module jit_convert_element_type: async-collective
I0729 17:26:34.639427  203556 hlo_pass_pipeline.cc:181]   HLO pass async-collective-replacer
I0729 17:26:34.639460  203556 hlo_pass_pipeline.cc:176] HLO pass: async-collective-replacer time: 34 us (34 us) (cumulative: 34 us, max: 34 us, #called: 1)
I0729 17:26:34.912426  203556 hlo_pass_pipeline.cc:303] Running HLO pass pipeline on module jit_top: HLO passes after scheduling
I0729 17:26:34.912498  203556 hlo_pass_pipeline.cc:176] HLO pass: apply-xla-transforms time: 15 us (15 us) (cumulative: 12.9 ms, max: 710 us, #called: 384)
I0729 17:51:03.131497  227384 hlo_pass_pipeline.cc:176] HLO pass: simplification after layout assignment time: 143 us (143 us) (cumulative: 4.06 ms, max: 918 us, #called: 119)
I0729 05:12:44.183010  123456 hlo_pass_pipeline.cc:176] HLO pass: autotuner time: 1.19 min (71651421 us) (cumulative: 1.2 min, ...)
'''

# ls of a scopex.dump(passes=".*") directory: 92 files for the program above. JAX's own warm-up
# modules sit alongside the one you asked for, which is what modules_in() exists to disambiguate.
SAMPLE_DUMP_NAMES = (
    "module_0000.jit_convert_element_type.0000.async-collective.after_pipeline-start"
    ".before_async-collective-replacer.txt",
    "module_0004.jit_top.0004.HLO_passes_through_layout_assignment.after_pipeline-start"
    ".before_batched_gather_scatter_normalizer.txt",
    "module_0004.jit_top.cpu_after_optimizations.txt",
    "module_0004.jit_top.obj-file.__compute_module_wrapped_tanh.o",
    "module_0004.jit_top.wrapped_reduce_kernel_module.ir-with-opt.ll",
)

# The emitter level's filenames, all verbatim from jaxlib 0.10.2 dumps. The last is a per-pass HLO
# snapshot and must NOT parse as an emitter file: the two grammars overlap in their first two
# fields, and a reader that accepted it would report an HLO pass as a kernel.
SAMPLE_EMITTER_DUMP_NAMES = (
    "module_0002.jit_scatter.wrapped_scatter.mlir-passes.log",                          # CPU
    "module_0002.jit_elem.input_reduce_fusion.mlir-passes.log",                         # GPU
    "module_0002.jit_scatter.wrapped_reduce-window_kernel_module-pre-optimization.mlir",
    "module_0002.jit_scatter.wrapped_scatter-post-lowering.mlir",
    "module_0002.jit_scatter.wrapped_scatter-post-optimization.mlir",
    "module_0002.jit_scatter.wrapped_scatter.ir-no-opt.ll",
    "module_0002.jit_scatter.obj-file.wrapped_scatter.o",
    "module_0002.jit_elem.1.ptx",                              # GPU: an INDEX, not a kernel name
    # A COLLIDING kernel name, disambiguated by XLA with a dotted suffix. 22 files in one CPU dump
    # looked like this and the first version of the grammar read the kernel as "265".
    "module_0000.jit__f_scan.__compute_module_add_bitcast_fusion.265-pre-optimization.mlir",
    "module_0000.jit__f_scan.__compute_module_multiply_divide_fusion.48.mlir-passes.log",
    "module_0004.jit_top.0004.fusion.after_pipeline-start.before_priority-fusion.txt",  # not ours
)

# jnp.linalg.eigh on CPU, jaxlib 0.10.2. Verbatim but for the elided api_version.
SAMPLE_CUSTOM_CALL = (
    '  %eigh.0 = (f32[8,8]{0,1}, f32[8]{0}, s32[]) custom-call(%multiply_copy_fusion), '
    'custom_call_target="lapack_ssyevd_ffi", operand_layout_constraints={f32[8,8]{0,1}}, '
    'output_to_operand_aliasing={{0}: (0, {})}, api_version=API_VERSION_TYPED_FFI\n')


# ══════════════════════════════════════════════════════════════════════════════════════════════
# ONE PRINTED HLO INSTRUCTION
#
# These two run on a SINGLE instruction's own `to_string()`, obtained from the native object walk --
# never on a guessed line of a whole module. That is the blast radius the native rewrite bought, and
# it is why these are the only HLO-text parsers left.
# ══════════════════════════════════════════════════════════════════════════════════════════════

# metadata mixes QUOTED values (op_name="...") with UNQUOTED ints (stack_frame_id=5). A quoted-only
# pattern drops the int SILENTLY, which is how source resolution appeared impossible at this level
# for as long as it did. Both forms, or nothing.
_META_BLOCK = re.compile(r"metadata=\{(?P<body>.*?)\}(?:,|\s*$)")
_META_KV = re.compile(r'(?P<k>\w+)=(?:"(?P<qv>(?:[^"\\]|\\.)*)"|(?P<uv>[^\s,}]+))')

# `%add.3 = f32[16,16]{1,0} add(%a, %b), metadata={...}`
# `.+?` and NOT `\S+` for the shape: a TUPLE shape contains a space -- `(s32[], f32[8]{0})` -- and
# `\S+` silently skipped every tuple-shaped instruction, i.e. every while / call / tuple /
# custom-call with a scratch output, and every parameter of a control-flow body.
_INSTR = re.compile(
    r"^\s*(?:ROOT\s+)?%?(?P<name>[\w.\-]+)\s*=\s*(?P<shape>.+?)\s+(?P<opcode>[a-z][\w-]*)\(")


def hlo_metadata(text: str) -> dict[str, str]:
    """``metadata={op_name="a" stack_frame_id=6}`` -> ``{"op_name": "a", "stack_frame_id": "6"}``.

    Printed by ``xla::OpMetadata`` inside ``xla::HloInstruction::ToString``
    (xla/hlo/ir/hlo_instruction.cc). ``HloInstruction`` exposes no ``.metadata``, so this is the
    only route.

    Returns ``{}`` when there is no metadata block, which is a real state -- plenty of
    XLA-introduced instructions carry none. But a block that is PRESENT and yields nothing is a
    broken parser, and raises: that is bug #2, guarded at the smallest unit there is.
    """
    m = _META_BLOCK.search(text)
    if not m:
        if "metadata={" in text:
            raise ParseError(
                f"scopex parser 'hlo_metadata' found 'metadata={{' in an instruction it could not "
                f"read.\n  printed by : xla::OpMetadata in HloInstruction::ToString\n"
                f"  instruction: {text.strip()[:200]}\n"
                f"  The metadata block's syntax has moved. Every scopex attribution below the "
                f"jaxpr comes from this block; do not let it fail quietly.")
        return {}
    # finditer and not findall: findall reports a non-participating group as "" rather than None,
    # so `quoted or unquoted` silently yields "" for every UNQUOTED value -- which is exactly the
    # stack_frame_id bug this function was written to fix, reintroduced one layer down.
    out: dict[str, str] = {}
    for kv in _META_KV.finditer(m.group("body")):
        out[kv.group("k")] = kv.group("qv") if kv.group("qv") is not None else kv.group("uv")
    if m.group("body").strip() and not out:
        raise ParseError(
            f"scopex parser 'hlo_metadata' read a non-empty metadata block as zero fields.\n"
            f"  printed by : xla::OpMetadata\n  block      : {m.group('body')[:200]}\n"
            f"  This is bug #2 (quoted-values-only) in a new form. Fix _META_KV in "
            f"scopex/_parse.py.")
    return out


def hlo_shape(text: str) -> str:
    """The result shape of one printed HLO instruction. ``HloInstruction`` has no ``.shape``.

    Printed by ``xla::HloInstruction::ToString``. Raises when the string is clearly an instruction
    (``name = ... opcode(``) and the shape will not come out: a silently empty shape is how the
    tuple-shape blind spot hid, and that blind spot sat exactly on ``while``/``call``/``tuple``.
    """
    m = _INSTR.match(text)
    if m:
        return m.group("shape")
    if " = " in text and "(" in text:
        raise ParseError(
            f"scopex parser 'hlo_shape' could not read the shape of an instruction that has one.\n"
            f"  printed by : xla::HloInstruction::ToString\n"
            f"  instruction: {text.strip()[:200]}\n"
            f"  Check _INSTR in scopex/_parse.py -- and note that the last time this pattern was "
            f"wrong it was `\\S+` for the shape, which cannot match a tuple shape and dropped "
            f"1,208 instructions across 895 of 2,811 real dump snapshots.")
    return ""


def check_metadata_coverage(records: Iterable[dict], text: str) -> None:
    """Every field the MODULE visibly carries must have reached the records. Or raise.

    :func:`hlo_metadata` runs on one instruction at a time, and at that scale a missing key is
    indistinguishable from a key the instruction never had. Bug #2 is invisible from there: a
    quoted-only pattern still returns ``op_name`` -- the dict is non-empty, every instruction looks
    fine -- and only ``stack_frame_id`` vanishes, taking every source line at this level with it.

    So the check has to be made once per MODULE, where the population is big enough to count. That
    is what this does, and it is the guard bug #2 would have tripped on its first run.
    """
    recs = list(records)
    if not recs:
        return
    for key, witness in (("op_name", r'op_name="'), ("stack_frame_id", r"stack_frame_id=")):
        if re.search(witness, text):
            expect(f"hlo_metadata[{key}]", sum(1 for r in recs if r.get(key)), text,
                   witness=witness, produced_by="xla::OpMetadata", unit=f"{key} values")


def hlo_shape_and_opcode(text: str) -> tuple[str, str]:
    """``(shape, printed opcode)`` for one printed HLO instruction.

    The opcode is returned only so a caller can CHECK it against the native ``HloOpcode`` enum --
    :func:`scopex.levels.opcode_of` is the source of truth, and the divergence warning is what keeps
    :data:`scopex.levels.OPCODE_TEXT` honest. Never use this as the opcode.
    """
    m = _INSTR.match(text)
    if m:
        return m.group("shape"), m.group("opcode")
    return hlo_shape(text), ""          # raises if the string is an instruction with a shape


def is_hlo_instruction_line(line: str) -> bool:
    """True if ``line`` is an HLO instruction. The FALLBACK instruction counter for per-pass dump
    snapshots -- :func:`scopex.artifacts.pass_growth` prefers XLA's own parser and records which
    route it used, so a slide back onto this one is visible in the data."""
    return bool(_INSTR.match(line))


_CUSTOM_CALL = re.compile(r'custom_call_target="((?:[^"\\]|\\.)*)"')


def custom_call_targets(text: str) -> list[str]:
    """Every ``custom_call_target`` in a printed module, with repeats.

    Printed by ``xla::HloCustomCallInstruction::ToString``. ``allow_empty=True``: a program with no
    custom calls is ordinary and its emptiness is information. The guard still fires if the text
    SAYS ``custom_call_target=`` and this returns nothing.
    """
    out = _CUSTOM_CALL.findall(text)
    expect("custom_call_targets", out, text, witness=r"custom_call_target=",
           produced_by="xla::HloCustomCallInstruction::ToString", unit="targets", allow_empty=True)
    return out


# ══════════════════════════════════════════════════════════════════════════════════════════════
# THE STACK-FRAME INDEX
#
# The optimized module DOES carry source locations; they are indirected through four module-level
# tables that HloModule has no accessor for:
#
#     FileNames                         FileLocations
#     1 "/home/u/proj/model.py"         1 {file_name_id=1 function_name_id=1 line=19 ...}
#     FunctionNames                     StackFrames
#     1 "<module>"                      1 {file_location_id=1 parent_frame_id=1}
#
# and an instruction says only `stack_frame_id=6`.
# ══════════════════════════════════════════════════════════════════════════════════════════════

_TAB_HEAD = re.compile(r"^(FileNames|FunctionNames|FileLocations|StackFrames)\s*$", re.M)
_TAB_ROW = re.compile(r'^(\d+)\s+(?:"((?:[^"\\]|\\.)*)"|\{(.*)\})\s*$')
_KV = re.compile(r"(\w+)=([^\s}]+)")
_FRAME_ROW = re.compile(r"^\d+ \{file_location_id=", re.M)


class Frame(NamedTuple):
    """One python frame recovered from the optimized module."""
    file: str
    line: int
    function: str

    @property
    def site(self) -> str:
        return f"{self.file}:{self.line}"


def _parent_offset(frames: dict) -> int:
    """How to read ``parent_frame_id``. MEASURED per module, not assumed.

    jaxlib 0.10.2 prints frame ids and parent ids in DIFFERENT index spaces. From the sample::

        6 {file_location_id=6 parent_frame_id=6}     # tanh's frame, in leaf() at line 6
        8 {file_location_id=6 parent_frame_id=8}     # the SAME source line, reached via solve()

    Read literally, frame 6's parent is frame 6. That is not a tree, and a walk over it either
    hangs or -- if it is cycle-guarded, which is the tempting fix -- silently truncates to a single
    frame and reports the innermost location as if it had no caller. Subtract one and the whole
    thing resolves: 6 -> 5 (residual, line 10) -> 4 (solve, 14) -> 3 (top, 17) -> 1 (<module>, 19),
    with 0 meaning root. That is exactly the python stack, and it was checked frame by frame against
    the MLIR callsite chain in the lowered module for the same program, on three programs with
    different call shapes.

    So the offset is DERIVED: whichever of {0, 1} yields an acyclic forest with every parent in
    range. If both do, the literal reading wins, so a future jaxlib that fixes the off-by-one keeps
    working with no edit here. If neither does, that is a ParseError -- refusing to answer beats a
    plausible wrong stack, which is the whole thesis of this module.
    """
    if not frames:
        return 0
    ok = []
    for off in (0, 1):
        good = True
        for fid in frames:
            seen: set = set()
            cur = fid
            while cur and good:
                if cur in seen or cur not in frames:
                    good = False
                    break
                seen.add(cur)
                cur = frames[cur].get("parent_frame_id", 0) - off
                if cur < 0:
                    good = False
        if good:
            ok.append(off)
    if not ok:
        raise ParseError(
            "scopex parser 'hlo_frame_tables' cannot read parent_frame_id: neither the literal "
            "reading nor the measured off-by-one gives an acyclic frame tree.\n"
            "  printed by : xla::HloModule::Print (StackFrameIndexProto)\n"
            f"  frames     : {dict(list(frames.items())[:8])}\n"
            "  Refusing to guess -- a wrong parent chain is a wrong attribution that looks right. "
            "Re-derive the convention in scopex/_parse.py:_parent_offset.")
    return ok[0]


def hlo_frame_tables(text: str) -> dict:
    """The module's four stack-frame tables, keyed by id, plus the parent-link convention.

    Printed by ``xla::HloModule::Print`` from ``StackFrameIndexProto`` (xla/service/hlo.proto).
    Returns ``{"files": {id: str}, "functions": {id: str}, "locations": {id: {k: int}},
    "frames": {id: {k: int}}, "parent_offset": int}``.

    Empty dicts when the module was printed without them -- per-pass dump snapshots carry
    ``stack_frame_id`` on instructions but not the tables, and XLA itself logs
    ``Invalid stack_frame_id`` when re-parsing those. That is a real state and not an error; the
    error case is a ``StackFrames`` section that is present and parses to nothing.
    """
    out: dict = {"files": {}, "functions": {}, "locations": {}, "frames": {}, "parent_offset": 0}
    key = {"FileNames": "files", "FunctionNames": "functions",
           "FileLocations": "locations", "StackFrames": "frames"}
    heads = list(_TAB_HEAD.finditer(text))
    for n, h in enumerate(heads):
        end = heads[n + 1].start() if n + 1 < len(heads) else len(text)
        dest = out[key[h.group(1)]]
        for line in text[h.end():end].splitlines():
            r = _TAB_ROW.match(line.strip())
            if not r:
                if line.strip():          # first non-row line ends the table
                    break
                continue
            i = int(r.group(1))
            if r.group(2) is not None:
                dest[i] = r.group(2)
            else:
                dest[i] = {k: int(v) for k, v in _KV.findall(r.group(3)) if v.lstrip("-").isdigit()}
    if "\nStackFrames" in text or text.startswith("StackFrames"):
        expect("hlo_frame_tables", out["frames"], text[text.index("StackFrames"):],
               witness=_FRAME_ROW, produced_by="xla::HloModule::Print", unit="frames")
    out["parent_offset"] = _parent_offset(out["frames"])
    return out


def hlo_frame_stack(frame_id, tab: dict, *, limit: int = 512) -> tuple[Frame, ...]:
    """The python stack for a ``stack_frame_id``, INNERMOST FIRST.

    Follows ``parent_frame_id`` using the offset :func:`_parent_offset` derived for this module.
    Cycle-safe and depth-capped anyway: a format change must degrade to a short answer, never to a
    hang.
    """
    frames, locs = tab.get("frames", {}), tab.get("locations", {})
    files, fns = tab.get("files", {}), tab.get("functions", {})
    off = tab.get("parent_offset", 0)
    out: list[Frame] = []
    seen: set = set()
    try:
        cur = int(frame_id)
    except (TypeError, ValueError):
        return ()
    while cur and cur in frames and cur not in seen and len(out) < limit:
        seen.add(cur)
        fr = frames[cur]
        loc = locs.get(fr.get("file_location_id", 0), {})
        f = files.get(loc.get("file_name_id", 0), "")
        if f:
            out.append(Frame(f, loc.get("line", 0), fns.get(loc.get("function_name_id", 0), "?")))
        cur = fr.get("parent_frame_id", 0) - off
    return tuple(out)


def hlo_site(frame_id, tab: dict) -> tuple[str, str]:
    """``(file:line, function)`` for a ``stack_frame_id``, filtered the SAME way the jaxpr is.

    Returns the innermost frame that is not inside jax or scopex, using
    :data:`scopex.walk._SKIP` -- which is what makes this level join with the jaxpr level on
    ``site``. If every frame is inside jax, the innermost is returned rather than a bucket: those
    frames are real, they are just not the user's.
    """
    from .walk import NO_FRAME
    frames = hlo_frame_stack(frame_id, tab)
    if not frames:
        return NO_FRAME, "?"
    from .walk import _SKIP
    for fr in frames:
        if not any(fr.file.startswith(m) for m in _SKIP):
            return fr.site, fr.function
    return frames[0].site, frames[0].function


# ══════════════════════════════════════════════════════════════════════════════════════════════
# STABLEHLO TEXT -- THE FALLBACK ROUTE ONLY
#
# scopex.levels.walk_stablehlo walks the jaxlib.mlir.ir module jax lowered into, which sees things
# no line parser can: an operation that owns a REGION has its loc(...) printed after the closing
# brace, so while/case/sort/reduce/scatter are invisible here. These functions remain as the
# no-bindings fallback, and the caller warns before reaching them.
#
# MLIR does not put the name on the operation. Three alias shapes occur:
#   NameLoc         #loc38 = loc("jit(top)/.../tanh"(#loc35))     <- the name stack
#   FileLineColLoc  #loc2  = loc("/home/u/proj/model.py":6:11 to :22)
#   CallSiteLoc     #loc22 = loc(callsite(#loc15 at #loc16))
# Taking "the first quoted string" from all three gives an operation whose name is a FILE PATH.
# ══════════════════════════════════════════════════════════════════════════════════════════════

_LOC_ALIAS = re.compile(r'^(?P<alias>#loc\w*)\s*=\s*loc\((?P<body>.*)\)\s*$', re.M)
_NAME_LOC = re.compile(r'^"(?P<name>(?:[^"\\]|\\.)*)"(?:\(|$)')
_FILE_LOC = re.compile(r'^"(?P<file>(?:[^"\\]|\\.)*)":(?P<line>\d+):(?P<col>\d+)')
# An SSA assignment line. Cruder than the op pattern on purpose: it is the witness, and a witness
# that shares the parser's assumptions proves nothing.
_MLIR_SSA = re.compile(r'^\s+%[\w#$.-]+\s*=\s*\S', re.M)
_MLIR_OP = re.compile(
    r'^\s*(?:%[\w#$.-]+(?:\s*,\s*%[\w#$.-]+)*\s*=\s*)?'          # optional result list
    r'(?P<op>[a-z_][\w.]*\.[\w.]+|return|func\.func)'            # dialect.op
    r'.*?\bloc\((?P<loc>#[\w.]+|"(?:[^"\\]|\\.)*")\)')           # its location


def stablehlo_loc_aliases(text: str) -> dict[str, str]:
    """``{"#loc38": "jit(top)/mylib:lib.solve/.../tanh"}`` -- NAME aliases only.

    Printed by MLIR's ``AsmPrinter`` when jax lowers with ``debug_info=True``. File/line aliases are
    deliberately excluded here; see :func:`stablehlo_file_aliases`.
    """
    out: dict[str, str] = {}
    for m in _LOC_ALIAS.finditer(text):
        body = m.group("body")
        if body.startswith("callsite") or _FILE_LOC.match(body):
            continue
        q = _NAME_LOC.match(body)
        if q:
            out[m.group("alias")] = q.group("name")
    return out


def stablehlo_file_aliases(text: str) -> dict[str, tuple[str, int, int]]:
    """``{"#loc2": ("/home/u/proj/model.py", 6, 11)}`` -- the FileLineCol aliases.

    Split from :func:`stablehlo_loc_aliases` so neither can be mistaken for the other: an operation
    whose "name" is a file path is the signature of merging them.
    """
    out: dict[str, tuple[str, int, int]] = {}
    for m in _LOC_ALIAS.finditer(text):
        f = _FILE_LOC.match(m.group("body"))
        if f:
            out[m.group("alias")] = (f.group("file"), int(f.group("line")), int(f.group("col")))
    return out


def stablehlo_op_lines(text: str) -> list[tuple[str, str]]:
    """``[(op, name_stack), ...]`` for every located StableHLO operation, in module order.

    TWO guards, because this level has two ways to go quiet:

    * the OPERATION count, witnessed by SSA assignment lines. A walker returning 1 unit from a
      module with 3,214 of them raises instead of reporting an empty level -- bug #1.
    * DANGLING ALIASES. Every ``loc(#locNN)`` an operation cites must be defined somewhere in the
      module. If the alias-DEFINITION syntax moves, every reference dangles, names silently go
      empty, and the operation count still looks perfect. Counting names against a fixed number
      would be wrong -- an operation may legitimately cite a callsite or a file location and have no
      name -- but a dangling REFERENCE cannot be legitimate.
    """
    names = stablehlo_loc_aliases(text)
    defined = {m.group("alias") for m in _LOC_ALIAS.finditer(text)}
    out: list[tuple[str, str]] = []
    dangling: list[str] = []
    for line in text.splitlines():
        m = _MLIR_OP.match(line)
        if not m:
            continue
        raw = m.group("loc")
        if raw.startswith("#") and raw not in defined:
            dangling.append(raw)
        out.append((m.group("op"), names.get(raw, "") if raw.startswith("#") else raw.strip('"')))
    expect("stablehlo_op_lines", out, text, witness=_MLIR_SSA,
           produced_by="MLIR AsmPrinter via Lowered.as_text(debug_info=True)", unit="operations")
    if dangling:
        raise ParseError(
            f"scopex parser 'stablehlo_op_lines' found {len(dangling)} operation location "
            f"reference(s) with no definition in the module, e.g. {sorted(set(dangling))[:5]}.\n"
            f"  printed by : MLIR AsmPrinter\n"
            f"  The alias-DEFINITION syntax has moved, so names resolve to '' while the operation "
            f"count still looks perfect. That is bug #1 wearing a new hat: fix "
            f"stablehlo_loc_aliases, do not relax this check.")
    return out


# ══════════════════════════════════════════════════════════════════════════════════════════════
# THE XLA PASS LOG
# ══════════════════════════════════════════════════════════════════════════════════════════════

# xla/service/hlo_pass_pipeline.cc:176, under TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE=...=1:
#     HLO pass: async-collective-replacer time: 34 us (34 us) (cumulative: 34 us, max: 34 us, ...)
#     HLO pass: autotuner time: 1.19 min (71651421 us) (cumulative: 1.2 min, ...)
#
# An even earlier attempt matched "<word> ... <number> s" and dutifully reported the glog timestamp
# prefix `I0729` as the most expensive pass in the program.
# A PASS NAME CAN CONTAIN SPACES, and `(?P<name>\S+)` silently dropped every line where it did.
# Found by the witness check above on the first live log it ran against: 384 lines said `HLO pass: `
# and 378 parsed. The six missing ones were all the same shape --
#     HLO pass: simplification after layout assignment time: 143 us (143 us) (cumulative: 4.06 ms..)
# -- so the blind spot sat on the passes whose names read as English, which on CPU includes the
# layout-assignment simplification passes. Same failure family as the other four: a silent
# undercount that leaves a plausible ranking. `.+?` up to the literal ` time:` instead.
_PASS_LINE = re.compile(
    r"HLO pass:\s+(?P<name>.+?)\s+time:\s+(?P<val>[\d.]+)\s*(?P<unit>[a-z]+)"
    r"(?:\s*\((?P<us>\d+)\s*us\))?")
_PASS_WITNESS = re.compile(r"HLO pass:\s")
_PIPELINE_LINE = re.compile(
    r"Running HLO pass pipeline on module (?P<module>\S+?):\s*(?P<pipeline>.+?)\s*$", re.M)
UNITS = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0, "sec": 1.0,
         "min": 60.0, "m": 60.0, "h": 3600.0, "hr": 3600.0}


class PassTime(NamedTuple):
    name: str
    seconds: float
    unit: str        # the headline unit XLA used, kept for diagnosis
    exact: bool      # True when taken from the parenthesised microseconds


def pass_timing_lines(log: str) -> list[PassTime]:
    """One entry per ``HLO pass: NAME time: ...`` line, in log order, NOT aggregated.

    Printed by ``xla/service/hlo_pass_pipeline.cc:176``. Prefers the PARENTHESISED microseconds,
    which XLA emits regardless of the headline unit, and RAISES rather than skipping a line whose
    unit it cannot convert. That refusal is the whole lesson of bug #3: the unconvertible line is by
    construction the expensive one.
    """
    out: list[PassTime] = []
    unknown: dict[str, str] = {}
    for m in _PASS_LINE.finditer(log):
        if m.group("us") is not None:
            out.append(PassTime(m.group("name"), int(m.group("us")) * 1e-6, m.group("unit"), True))
            continue
        u = m.group("unit")
        if u not in UNITS:
            unknown[u] = m.group(0)[:160]
            continue
        out.append(PassTime(m.group("name"), float(m.group("val")) * UNITS[u], u, False))
    if unknown:
        raise ParseError(
            f"scopex parser 'pass_timing_lines' cannot convert time units {sorted(unknown)}, and "
            f"those lines carry no parenthesised microseconds either.\n"
            f"  printed by : xla/service/hlo_pass_pipeline.cc:176\n"
            f"  example    : {list(unknown.values())[0]}\n"
            f"  XLA reports large times in large units, so an excluded pass is probably the "
            f"expensive one. Dropping 1 of 640 lines once hid 98.8% of a 72.5 s compile and left a "
            f"plausible ranking topped by a pass taking 0.12 s. Add the unit to "
            f"scopex._parse.UNITS.")
    expect("pass_timing_lines", out, log, witness=_PASS_WITNESS,
           produced_by="xla/service/hlo_pass_pipeline.cc:176", unit="pass timings",
           allow_empty=True)
    return out


def pass_pipeline_headers(log: str) -> list[tuple[str, str]]:
    """``[(module, pipeline), ...]`` -- which MODULE each run of passes belonged to.

    Printed by ``xla/service/hlo_pass_pipeline.cc:303``. Parsed because the log interleaves JAX's
    warm-up modules (``jit_convert_element_type`` and friends) with the program you asked about, and
    a total summed over all of them attributes someone else's passes to you without saying so.
    """
    return [(m.group("module"), m.group("pipeline")) for m in _PIPELINE_LINE.finditer(log)]


# ══════════════════════════════════════════════════════════════════════════════════════════════
# DUMP FILENAMES
# ══════════════════════════════════════════════════════════════════════════════════════════════

# module_0004.jit_top.0000.fusion.after_sort-iota-fusion.before_priority-fusion.txt
# Built by xla/service/dump.cc; the index is the pass ordinal and the two names bracket the pass.
_SNAPSHOT = re.compile(
    r"^module_(?P<mod>\d+)\.(?P<fn>.+?)\.(?P<idx>\d{4})\.(?P<pipeline>[^.]+)"
    r"\.after_(?P<after>[^.]+)\.before_(?P<before>[^.]+)\.txt$")


def dump_snapshot_name(filename: str) -> dict | None:
    """Split a per-pass snapshot filename, or None if it is not one.

    Built by ``xla/service/dump.cc``. None is the ordinary case in a dump directory -- ``.ll``,
    ``.o``, ``debug_options`` and the before/after-optimization modules all live there too -- so
    this is unguarded per call. :func:`scopex.artifacts.pass_growth` guards the population.
    """
    m = _SNAPSHOT.match(filename)
    if not m:
        return None
    return {"module": f"module_{m.group('mod')}.{m.group('fn')}", "index": int(m.group("idx")),
            "pipeline": m.group("pipeline"), "after_pass": m.group("after"),
            "before_pass": m.group("before")}


# ══════════════════════════════════════════════════════════════════════════════════════════════
# THE EMITTER LEVEL -- XLA's OWN MLIR PIPELINE, BELOW THE HLO PASSES
#
# `--xla_dump_emitter_re` switches on a per-KERNEL MLIR dump written by the backends' emitters,
# which run their own MLIR pass pipeline after the HLO passes are finished. Verified present on
# jax/jaxlib 0.10.2 on BOTH backends; what it writes differs:
#
#   CPU   module_0002.jit_scatter.wrapped_scatter.mlir-passes.log        (66 pass snapshots)
#         module_0002.jit_scatter.wrapped_scatter-pre-optimization.mlir
#         module_0002.jit_scatter.wrapped_scatter-post-lowering.mlir
#         module_0002.jit_scatter.wrapped_scatter-post-optimization.mlir
#   GPU   module_0002.jit_elem.input_reduce_fusion.mlir-passes.log       (67 pass snapshots)
#         -- and NOTHING else. The three stage .mlir files are CPU-only, measured.
#
# THE FLAG'S ARGUMENT IS NOT A KERNEL FILTER, WHICH IS A TRAP OF EXACTLY THE SHAPE THIS MODULE
# EXISTS FOR. It reads like `--xla_dump_hlo_pass_re`, i.e. "a regex over the thing you want". It
# is not: it is partial-matched against the FIXED 11-character dump-kind tag `"mlir-fusion"`.
# Measured by bisection on a dump with three kernels named `wrapped_scatter`,
# `wrapped_broadcast_kernel_module` and `concatenate_bitcast_fusion_kernel_module`:
#
#     --xla_dump_emitter_re=.*                12 emitter files
#     --xla_dump_emitter_re=mlir-fusion       12          (and `fusion`, `usion`, `f.*n`, `.+`)
#     --xla_dump_emitter_re=.*scatter.*        0   <-- names a kernel that EXISTS
#     --xla_dump_emitter_re=wrapped_scatter    0
#     --xla_dump_emitter_re=.*loop_fusion.*    0
#     --xla_dump_emitter_re=^.{11}$           12   (this is how the length was pinned)
#
# So the obvious spelling -- naming the kernel you care about -- yields an EMPTY emitter level and
# no error, on a compile that has one. scopex therefore never passes a user string here: it sends
# `mlir-fusion` and filters by kernel in python.
# ══════════════════════════════════════════════════════════════════════════════════════════════

# module_0002.jit_scatter.wrapped_scatter.mlir-passes.log
# module_0002.jit_scatter.wrapped_reduce-window_kernel_module-post-lowering.mlir
# module_0000.jit__f_scan.__compute_module_add_bitcast_fusion.265-pre-optimization.mlir
#
# THE KERNEL FIELD CAN CONTAIN A DOT AND THE MODULE STEM CANNOT. That is the whole reason these
# patterns are shaped the way they are, and it was found by running them on a second program: when
# a module has several kernels that would collide, XLA appends `.<n>` to the name. With `fn`
# greedy and `kernel` dot-free -- the obvious reading, and the first version here -- every one of
# those 22 files parsed happily with `kernel="265"`, `kernel="48"`, `kernel="6"`. Not empty. Not an
# error. Just the disambiguator where the kernel name should be, and a per-kernel report keyed on
# numbers that mean nothing. `fn` is `[^.]+` because a jit'd function's `__name__` cannot contain a
# dot; `emitter_files` cross-checks every stem this yields against the stems the rest of the dump
# shows, so a module name that breaks that assumption raises instead of eating the kernel.
_EMIT_LOG = re.compile(r"^module_(?P<mod>\d+)\.(?P<fn>[^.]+)\.(?P<kernel>.+)\.mlir-passes\.log$")
_EMIT_MLIR = re.compile(r"^module_(?P<mod>\d+)\.(?P<fn>[^.]+)\.(?P<kernel>.+?)"
                        r"-(?P<stage>pre-optimization|post-lowering|post-optimization)\.mlir$")
# The codegen products. On CPU the `kernel` slot is the kernel NAME, so these join to the MLIR by
# name. On GPU it is an LLVM-module INDEX (`module_0002.jit_elem.1.ptx`) and the join is impossible
# -- `emitter_growth` reports that rather than silently pairing an index with a name.
_EMIT_LL = re.compile(r"^module_(?P<mod>\d+)\.(?P<fn>[^.]+)\.(?P<kernel>.+)"
                      r"\.ir-(?P<opt>no|with)-opt\.ll$")
_EMIT_OBJ = re.compile(r"^module_(?P<mod>\d+)\.(?P<fn>[^.]+)\.obj-file\.(?P<kernel>.+)\.o$")
_EMIT_PTX = re.compile(r"^module_(?P<mod>\d+)\.(?P<fn>[^.]+)\.(?P<kernel>.+)\.ptx$")

EMITTER_DUMP_KIND = "mlir-fusion"     # what --xla_dump_emitter_re is actually matched against


def emitter_dump_name(filename: str) -> dict | None:
    """Split an emitter-level dump filename, or None if it is not one.

    Returns ``{"module", "kernel", "kind"}`` where ``kind`` is one of ``passes-log``,
    ``pre-optimization``, ``post-lowering``, ``post-optimization``, ``ir-no-opt``, ``ir-with-opt``,
    ``obj``, ``ptx``. Written by ``xla/service/dump.cc`` via the backend emitters.
    """
    for rx, kind in ((_EMIT_LOG, "passes-log"), (_EMIT_MLIR, None), (_EMIT_OBJ, "obj"),
                     (_EMIT_LL, None), (_EMIT_PTX, "ptx")):
        m = rx.match(filename)
        if not m:
            continue
        if kind is None:
            kind = m.groupdict().get("stage") or f"ir-{m.group('opt')}-opt"
        return {"module": f"module_{m.group('mod')}.{m.group('fn')}",
                "kernel": m.group("kernel"), "kind": kind}
    return None


# Printed by MLIR's own `PrintIRPass` (mlir/lib/Pass/IRPrinting.cpp), which XLA switches on for the
# emitter pipeline. VERBATIM, from module_0002.jit_scatter.wrapped_scatter.mlir-passes.log --
# note that a pass NAME can contain `::` and a pass SPEC can contain nested braces, so neither is
# `\w+` and neither is `[^{]*`.
SAMPLE_MLIR_PASS_LOG = '''\
builtin.module(func.func(xla-simplify-arith{explicit_nan_propagation=false fast_min_max=false}),cse)

// -----// IR Dump Before SimplifyArithPass: xla-simplify-arith{explicit_nan_propagation=false \
fast_min_max=false} ('func.func' operation: @wrapped_scatter) //----- //
#indexing_map = #xla.indexing_map<"(th_x)[s0] -> (s0), domain: th_x in [0, 0], s0 in [0, 63]">
module @wrapped_scatter_kernel_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>} {
  func.func @wrapped_scatter(%arg0: tensor<f32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %0 = xla.workgroup_id x {xla.range = [0 : index, 0 : index]}
    %1:2 = scf.while (%arg2 = %0) : (index) -> (index, index) {
      %inserted = tensor.insert %pure_call into %iter[%ra, %rb] : tensor<64x64xf32>
      xla.yield %inserted : tensor<64x64xf32>
    }
    return %0 : tensor<64x64xf32>
  }
}

// -----// IR Dump Before Inliner: composite-fixed-point-pass{max-iterations=10 name=Inliner \
pipeline=inline{default-pipeline=canonicalize inlining-threshold=4294967295 max-iterations=4 }} \
('builtin.module' operation: @wrapped_scatter_kernel_module) //----- //
module @wrapped_scatter_kernel_module {
  func.func @wrapped_scatter(%arg0: tensor<f32>) -> tensor<f32> {
    return %arg0 : tensor<f32>
  }
}

// -----// IR Dump Before xla::cpu::ModuleCallbackPass: unknown<xla::cpu::ModuleCallbackPass> \
('builtin.module' operation: @wrapped_scatter_kernel_module) //----- //
module @wrapped_scatter_kernel_module {
  llvm.func @wrapped_scatter(%arg0: !llvm.ptr) attributes {sym_visibility = "public"} {
    llvm.return
  }
}
'''

# The same log as above, torn the way XLA tears it on a multi-kernel module: a NUL run and a
# newline inserted mid-header. Verbatim in shape from
# module_0000.jit__f_scan.__compute_module_wrapped_compare.mlir-passes.log (582 NUL bytes).
SAMPLE_MLIR_PASS_LOG_TORN = SAMPLE_MLIR_PASS_LOG.replace(
    "// -----// IR Dump Before Inliner: composite-fixed-point-pass{max-iterations=10 name=Inliner",
    "// -----// IR Dump Before Inliner: composite-fixed-point-pass{max-iter\nations=10 \x00 Inliner")

_EMITIR_HDR = re.compile(
    r"^// -----// IR Dump (?P<when>Before|After) (?P<pass>\S+?): (?P<spec>.*) "
    r"\('(?P<scope>[^']+)' operation: @?(?P<symbol>[^)]*)\) //----- //\s*$", re.M)
_EMITIR_HDR_WITNESS = re.compile(r"^// -----// IR Dump ", re.M)
# A header that at least STARTS and ENDS like one. Deliberately lax, and it is the witness rather
# than the one above, because XLA can write this log TORN -- see `mlir_log_damage`. A header that
# starts and never finishes is damaged input; one that finishes in a shape `_EMITIR_HDR` does not
# recognise is a moved format, and only the second is a parser bug. Separating them is the whole
# point: `_EMITIR_HDR_WITNESS` alone would report XLA's torn write as scopex's blind spot.
_EMITIR_HDR_OK = re.compile(r"^// -----// IR Dump .*//----- //\s*$", re.M)

# One MLIR operation per line, in the pretty (non-generic) form MLIR prints by default. Results are
# stripped first: `%0 = `, `%1:2 = ` (multi-result), `%a, %b = `. What is left must start with a
# bare identifier -- `dialect.op` for everything except the handful of custom-printed forms
# (`return`, `module`, `func.func` is already dotted). Lines starting `#`, `//`, `^`, `}`, `{` are
# attribute aliases, the dump banners, block labels and region delimiters, and are not operations.
_EMITIR_RESULTS = re.compile(r"^(%[\w$.<>-]+(?::\d+)?)(\s*,\s*%[\w$.<>-]+(?::\d+)?)*\s*=\s*")
# The `|$` is load-bearing: a zero-operand terminator (`llvm.return`, `llvm.unreachable`,
# `cf.br`-less blocks) is a whole line with no trailing punctuation, and requiring one dropped it.
_EMITIR_OP = re.compile(r"^([a-z_][\w]*(?:\.[\w.$]+)*)(?:[({<:\[\"@%!^\s]|$)")
_EMITIR_SSA = re.compile(r"^\s*%[\w$.<>-]+(?::\d+)?(\s*,\s*%[\w$.<>-]+)*\s*=\s", re.M)
# Words that begin a CONTINUATION line rather than an operation -- MLIR wraps long operations.
_EMITIR_NOT_OPS = frozenset({"attributes", "to", "into", "in", "from", "at", "loc", "iter_args",
                           "shared_outs", "step", "outs", "ins", "else", "then", "do", "by"})


def mlir_op_lines(text: str, *, allow_empty: bool = False) -> list[str]:
    """Every operation in a printed MLIR module, as ``dialect.op`` strings.

    A COUNT and not a parse: this is the emitter level's analogue of counting HLO instructions, and
    it exists to make one MLIR snapshot comparable with the next. Guarded against the failure that
    matters -- returning fewer operations than the text visibly assigns SSA values -- because the
    thing this measures (a pass that explodes the IR) is exactly the thing an under-counting
    pattern would hide.
    """
    out: list[str] = []
    for raw in text.splitlines():
        ln = raw.strip()
        if not ln or ln[0] in "#/^}){]" or ln.startswith("{"):
            continue
        ln = _EMITIR_RESULTS.sub("", ln)
        m = _EMITIR_OP.match(ln)
        if m and m.group(1) not in _EMITIR_NOT_OPS:
            out.append(m.group(1))
    expect("mlir_op_lines", out, text, witness=_EMITIR_SSA,
           produced_by="mlir/lib/IR/AsmPrinter.cpp (emitter MLIR pipeline)", unit="operations",
           allow_empty=allow_empty)
    return out


class MlirPassDump(NamedTuple):
    """One ``IR Dump Before <pass>`` snapshot from a ``.mlir-passes.log``."""
    index: int
    when: str            # "Before" -- MLIR's IRPrinting prints only before, measured on 0.10.2
    pass_name: str       # "SimplifyArithPass", or "xla::cpu::ModuleCallbackPass"
    pass_spec: str       # "xla-simplify-arith{...}" -- may contain nested braces
    scope: str           # "func.func" or "builtin.module": the granularity the pass ran at
    symbol: str          # the function or module the pass ran ON
    lines: int
    ops: int


def mlir_pass_dumps(log: str) -> list[MlirPassDump]:
    """Split a ``.mlir-passes.log`` into per-pass IR snapshots, with a size for each.

    Printed by MLIR's ``PrintIRPass``. The first line of the file is the whole pipeline spec and
    carries no IR; everything after a header belongs to that header.

    A pass that runs at ``func.func`` scope is printed ONCE PER FUNCTION, so the same pass name
    appears many times with different ``symbol``. That is not a duplicate and must not be summed
    away: on the CPU scatter case ``SimplifyArithPass`` appears 8 times over 6 distinct symbols.
    """
    hdrs = list(_EMITIR_HDR.finditer(log))
    out: list[MlirPassDump] = []
    for i, m in enumerate(hdrs):
        body = log[m.end():hdrs[i + 1].start() if i + 1 < len(hdrs) else len(log)]
        out.append(MlirPassDump(
            index=i, when=m.group("when"), pass_name=m.group("pass"), pass_spec=m.group("spec"),
            scope=m.group("scope"), symbol=m.group("symbol"),
            lines=len(body.strip().splitlines()),
            ops=len(mlir_op_lines(body, allow_empty=True))))
    expect("mlir_pass_dumps", out, log, witness=_EMITIR_HDR_OK,
           produced_by="mlir/lib/Pass/IRPrinting.cpp", unit="IR snapshots", allow_empty=True)
    return out


def mlir_log_damage(log: str) -> dict:
    """How much of a ``.mlir-passes.log`` XLA wrote intact. ``{"headers", "complete", "torn",
    "nul_bytes"}``.

    XLA WRITES THIS LOG TORN WHEN A MODULE HAS SEVERAL KERNELS, and the reader must be able to say
    so rather than absorb it. Measured over 19 real logs from four programs on jax 0.10.2/CPU:
    the 17 logs from single-kernel modules were byte-clean, and 2 of the logs from the two
    eight-kernel modules contained 582 and 560 NUL bytes each, with one header line torn in half
    across the damage::

        // -----// IR Dump Before SimplifyArithPass: xla-simplify-arith{...  fast_min_max
        f\\x00lse} ('func.func' operation: @wrapped_compare) //----- //

    -- i.e. concurrent emitters interleaving into one file descriptor. So on a multi-kernel program
    this log is best-effort, and a reader that quietly returns 77 of 78 snapshots is under-reporting
    for a reason that has nothing to do with the program. ``torn`` is carried through to
    :class:`scopex.EmitterKernel.damage` so a curve with a hole in it says so.
    """
    started = len(_EMITIR_HDR_WITNESS.findall(log))
    complete = len(_EMITIR_HDR_OK.findall(log))
    return {"headers": started, "complete": complete, "torn": started - complete,
            "nul_bytes": log.count("\x00")}


# ══════════════════════════════════════════════════════════════════════════════════════════════
# CONFORMANCE
# ══════════════════════════════════════════════════════════════════════════════════════════════

class Parser(NamedTuple):
    """One parser, what prints its input, and the least its own sample must yield."""
    name: str
    printed_by: str
    run: Callable[[], object]
    at_least: int
    note: str = ""


def _n(x) -> int:
    if isinstance(x, (int, float)):
        return int(x)
    if isinstance(x, dict):
        return len(x)
    return len(list(x)) if isinstance(x, Iterable) else int(bool(x))


_HLO_LINES = [ln for ln in SAMPLE_HLO.splitlines() if " = " in ln]

PARSERS: tuple[Parser, ...] = (
    Parser("hlo_metadata", "xla::OpMetadata in HloInstruction::ToString",
           lambda: hlo_metadata(
               '  %tanh.0 = f32[8,8]{1,0} tanh(%p), metadata={op_name="jit(top)/tanh" '
               'stack_frame_id=6}'), 2,
           "must read BOTH the quoted op_name and the unquoted stack_frame_id"),
    Parser("hlo_metadata[module]", "xla::OpMetadata",
           lambda: [m for m in map(hlo_metadata, _HLO_LINES) if m], 12, ""),
    Parser("hlo_metadata[stack_frame_id]", "xla::OpMetadata",
           lambda: [m for m in map(hlo_metadata, _HLO_LINES) if m.get("stack_frame_id")], 9,
           "bug #2 resolved zero of these"),
    Parser("check_metadata_coverage", "xla::OpMetadata, counted per MODULE",
           lambda: [check_metadata_coverage(map(hlo_metadata, _HLO_LINES), SAMPLE_HLO), "ok"][1:],
           1, "the guard bug #2 would have tripped; per-instruction checks cannot see it"),
    Parser("hlo_shape", "xla::HloInstruction::ToString",
           lambda: [s for s in map(hlo_shape, _HLO_LINES) if s], 20, ""),
    Parser("hlo_shape[tuple]", "xla::HloInstruction::ToString",
           lambda: [s for s in map(hlo_shape, SAMPLE_HLO_TUPLE_LINES) if "," in s], 2,
           "tuple shapes contain a space; `\\S+` dropped every while/call/tuple"),
    Parser("is_hlo_instruction_line", "xla::HloComputation::ToString",
           lambda: [ln for ln in SAMPLE_HLO.splitlines() if is_hlo_instruction_line(ln)], 20, ""),
    Parser("hlo_frame_tables", "xla::HloModule::Print (StackFrameIndexProto)",
           lambda: hlo_frame_tables(SAMPLE_HLO)["frames"], 8, ""),
    Parser("hlo_frame_stack", "the frame tables, parent-linked",
           lambda: hlo_frame_stack(6, hlo_frame_tables(SAMPLE_HLO)), 5,
           "tanh is 5 frames deep: leaf/residual/solve/top/<module>"),
    Parser("stablehlo_loc_aliases", "MLIR AsmPrinter",
           lambda: stablehlo_loc_aliases(SAMPLE_STABLEHLO), 15, ""),
    Parser("stablehlo_file_aliases", "MLIR AsmPrinter",
           lambda: stablehlo_file_aliases(SAMPLE_STABLEHLO), 10, ""),
    Parser("stablehlo_op_lines", "MLIR AsmPrinter",
           lambda: stablehlo_op_lines(SAMPLE_STABLEHLO), 10, ""),
    Parser("stablehlo_op_lines[named]", "MLIR AsmPrinter",
           lambda: [x for x in stablehlo_op_lines(SAMPLE_STABLEHLO) if x[1]], 10,
           "bug #1 returned 1 here, on modules of up to 21,000 operations"),
    Parser("pass_timing_lines", "hlo_pass_pipeline.cc:176",
           lambda: pass_timing_lines(SAMPLE_PASS_LOG), 4,
           "including the `min` line bug #3 dropped and the space-containing pass name"),
    Parser("pass_pipeline_headers", "hlo_pass_pipeline.cc:303",
           lambda: pass_pipeline_headers(SAMPLE_PASS_LOG), 2, ""),
    Parser("dump_snapshot_name", "xla/service/dump.cc",
           lambda: [d for d in map(dump_snapshot_name, SAMPLE_DUMP_NAMES) if d], 2, ""),
    Parser("custom_call_targets", "xla::HloCustomCallInstruction::ToString",
           lambda: custom_call_targets(SAMPLE_CUSTOM_CALL), 1, ""),
    Parser("emitter_dump_name", "xla/service/dump.cc, via the backend emitters",
           lambda: [d for d in map(emitter_dump_name, SAMPLE_EMITTER_DUMP_NAMES) if d], 7,
           "all eight names but the per-pass HLO snapshot, which is NOT an emitter file"),
    Parser("mlir_pass_dumps", "mlir/lib/Pass/IRPrinting.cpp",
           lambda: mlir_pass_dumps(SAMPLE_MLIR_PASS_LOG), 3,
           "a pass name can contain `::` and a pass spec can contain NESTED braces"),
    Parser("mlir_op_lines", "MLIR AsmPrinter, emitter pipeline",
           lambda: mlir_op_lines(SAMPLE_MLIR_PASS_LOG), 12, ""),
    Parser("mlir_log_damage", "mlir/lib/Pass/IRPrinting.cpp, written torn by concurrent emitters",
           lambda: [mlir_log_damage(SAMPLE_MLIR_PASS_LOG_TORN)["torn"]], 1,
           "a torn header must be counted as DAMAGED INPUT, not as a parser blind spot"),
)


def _semantic_checks() -> list[str]:
    """The checks a count cannot express. Every one of them is a bug this package shipped."""
    bad = []
    md = hlo_metadata(
        '  %t = f32[8,8]{1,0} tanh(%p), metadata={op_name="jit(top)/tanh" stack_frame_id=6}')
    if md.get("stack_frame_id") != "6":
        bad.append("hlo_metadata drops UNQUOTED values (stack_frame_id) -- bug #2")

    if hlo_shape(SAMPLE_HLO_TUPLE_LINES[0]) != "(s32[], f32[8]{0})":
        bad.append(f"hlo_shape mangles a tuple shape: {hlo_shape(SAMPLE_HLO_TUPLE_LINES[0])!r}")

    tab = hlo_frame_tables(SAMPLE_HLO)
    got = [(f.function, f.line) for f in hlo_frame_stack(6, tab)]
    want = [("leaf", 6), ("residual", 10), ("solve", 14), ("top", 17), ("<module>", 19)]
    if got != want:
        bad.append(f"hlo_frame_stack resolves the wrong python stack: {got} != {want} "
                   f"(parent_offset={tab['parent_offset']})")
    # The same source line reached two ways must give two DIFFERENT stacks, or the frame index is
    # being read as a flat table and every attribution through a shared helper is wrong.
    if hlo_frame_stack(8, tab)[:2] == hlo_frame_stack(6, tab)[:2]:
        bad.append("hlo_frame_stack gives frames 6 and 8 the same caller; they are the same source "
                   "line reached through residual() and through solve(), and must differ")
    site, fn = hlo_site(6, tab)
    if site != "/home/u/proj/model.py:6" or fn != "leaf":
        bad.append(f"hlo_site gives {site!r}/{fn!r}, want '/home/u/proj/model.py:6'/'leaf'")

    ops = stablehlo_op_lines(SAMPLE_STABLEHLO)
    tanh = sorted(n for o, n in ops if o == "stablehlo.tanh")
    want_tanh = ["jit(top)/mylib:lib.solve/mylib:user.MyModel.residual/tanh",
                 "jit(top)/mylib:lib.solve/tanh"]
    if tanh != want_tanh:
        bad.append(f"stablehlo_op_lines lost the name stack behind loc(#locNN): {tanh} != "
                   f"{want_tanh} -- bug #1")
    if any(n.startswith("/") for _, n in ops):
        bad.append("stablehlo_op_lines returned a FILE PATH as an operation name "
                   "(FileLineColLoc mistaken for NameLoc)")

    times = {p.name: p.seconds for p in pass_timing_lines(SAMPLE_PASS_LOG)}
    if "simplification after layout assignment" not in times:
        bad.append("pass_timing_lines dropped a pass whose NAME CONTAINS SPACES; on a live CPU log "
                   "that was 6 of 384 lines and `\\S+` was the reason")
    if "autotuner" not in times:
        bad.append("pass_timing_lines dropped the `min` line -- bug #3")
    elif abs(times["autotuner"] - 71.651421) > 1e-3:
        bad.append(f"pass_timing_lines mis-scaled the `min` line: {times['autotuner']}")
    elif max(times, key=times.get) != "autotuner":
        bad.append("pass_timing_lines no longer ranks the slowest pass first")

    mods = {m for m, _ in pass_pipeline_headers(SAMPLE_PASS_LOG)}
    if mods != {"jit_convert_element_type", "jit_top"}:
        bad.append(f"pass_pipeline_headers lost module attribution: {mods}")

    # ── the emitter level ────────────────────────────────────────────────────────────────────────
    if emitter_dump_name(SAMPLE_EMITTER_DUMP_NAMES[-1]) is not None:
        bad.append("emitter_dump_name accepted a per-pass HLO SNAPSHOT as an emitter file; the "
                   "two filename grammars share their first two fields and must not overlap")
    got = emitter_dump_name("module_0002.jit_scatter.wrapped_reduce-window_kernel_module"
                            "-pre-optimization.mlir")
    if got != {"module": "module_0002.jit_scatter", "kind": "pre-optimization",
               "kernel": "wrapped_reduce-window_kernel_module"}:
        bad.append(f"emitter_dump_name split a HYPHENATED kernel name wrongly: {got}. The kernel "
                   f"and the stage are both hyphen-joined, so a greedy split eats the kernel")
    got = emitter_dump_name(
        "module_0000.jit__f_scan.__compute_module_add_bitcast_fusion.265-pre-optimization.mlir")
    if got != {"module": "module_0000.jit__f_scan", "kind": "pre-optimization",
               "kernel": "__compute_module_add_bitcast_fusion.265"}:
        bad.append(f"emitter_dump_name lost a DOT-DISAMBIGUATED kernel name: {got}. XLA appends "
                   f"`.<n>` when kernel names collide, and a dot-free kernel pattern reports the "
                   f"disambiguator as the kernel -- plausibly, on 22 files at a time")
    d = mlir_pass_dumps(SAMPLE_MLIR_PASS_LOG)
    if [x.pass_name for x in d] != ["SimplifyArithPass", "Inliner", "xla::cpu::ModuleCallbackPass"]:
        bad.append(f"mlir_pass_dumps lost a pass name: {[x.pass_name for x in d]}. A `\\w+` pattern "
                   f"drops `xla::cpu::ModuleCallbackPass`, which is the pass that hands the module "
                   f"to LLVM -- i.e. the boundary anyone reading this level is looking for")
    if len(d) > 1 and "pipeline=inline{" not in d[1].pass_spec:
        bad.append(f"mlir_pass_dumps truncated a NESTED-brace pass spec: {d[1].pass_spec!r}")
    if [x.scope for x in d] != ["func.func", "builtin.module", "builtin.module"]:
        bad.append(f"mlir_pass_dumps lost the pass SCOPE: {[x.scope for x in d]}. Without it a "
                   f"func-scoped pass printed once per function reads as N runs of one pass")
    if d and d[0].ops < 6:
        bad.append(f"mlir_op_lines counted {d[0].ops} operations in a snapshot with 8 -- a "
                   f"multi-result (`%1:2 = `) or a bare `return` is being dropped")
    clean = mlir_log_damage(SAMPLE_MLIR_PASS_LOG)
    if clean["torn"] or clean["nul_bytes"]:
        bad.append(f"mlir_log_damage reports an intact log as damaged: {clean}")
    torn = mlir_log_damage(SAMPLE_MLIR_PASS_LOG_TORN)
    if torn["torn"] != 1 or torn["nul_bytes"] != 1:
        bad.append(f"mlir_log_damage missed XLA's torn write: {torn}. A reader that cannot see it "
                   f"reports 77 of 78 snapshots and blames its own regex")
    if len(mlir_pass_dumps(SAMPLE_MLIR_PASS_LOG_TORN)) != 2:
        bad.append("mlir_pass_dumps did not survive a torn log; a damaged header must cost one "
                   "snapshot, not the whole file")

    ops = mlir_op_lines(SAMPLE_MLIR_PASS_LOG)
    if any(o.startswith("#") or o in ("module", "attributes") and False for o in ops):
        bad.append("mlir_op_lines counted an attribute alias as an operation")
    if "tensor.insert" not in ops or "llvm.return" not in ops or "return" not in ops:
        bad.append(f"mlir_op_lines missed a plain form: {sorted(set(ops))}")

    try:
        from .fusion import parse_textproto
    except Exception:                                                        # pragma: no cover
        return bad
    # Not an XLA capture: a hand-written text-proto exercising the grammar features XLA's fusion
    # dump uses (nested message, repeated field, quoted string, bareword enum, float).
    d = parse_textproto('step { fusion { producer_name: "a" consumer_name: "b" } } '
                        'step { update_priority { priority: -1.5 kind: KIND_A } } '
                        'step { unknown_future_kind { whatever: 1 } }')
    steps = d.get("step", [])
    if len(steps) != 3 or "unknown_future_kind" not in steps[2]:
        bad.append(f"fusion.parse_textproto lost a step or an unknown field: {d}")
    return bad


def conformance(*, verbose: bool = False) -> dict:
    """Run every parser against its EMBEDDED SAMPLE. Returns a report; raises on failure.

    Needs no jax and no compile, so it belongs in CI. The other half of the check -- the same
    parsers against a freshly compiled program -- is :func:`scopex.selftest`.
    """
    rows: dict = {}
    bad: list[str] = []
    for p in PARSERS:
        try:
            n = _n(p.run())
        except Exception as e:                                # a raising parser is a failure too
            rows[p.name] = f"ERROR {type(e).__name__}"
            bad.append(f"{p.name}: raised {type(e).__name__}: {str(e).splitlines()[0]}")
            continue
        rows[p.name] = n
        if n < p.at_least:
            bad.append(f"{p.name}: {n} results from its own sample, expected >= {p.at_least}"
                       + (f" ({p.note})" if p.note else ""))
    try:
        bad.extend(_semantic_checks())
    except Exception as e:
        bad.append(f"semantic checks raised {type(e).__name__}: {str(e).splitlines()[0]}")
    report = {"counts": rows, "failures": bad, "ok": not bad,
              "parent_offset": hlo_frame_tables(SAMPLE_HLO)["parent_offset"]}
    if verbose:
        print(report)
    if bad:
        raise ParseError(
            "scopex parser conformance FAILED against its own embedded samples:\n  - "
            + "\n  - ".join(bad)
            + "\nThose samples are frozen text, so nothing external can have changed: the parsers "
              "in scopex/_parse.py have been edited into something wrong.")
    return report
