"""GAP 9 (jaxpr -> StableHLO lowering as its own stage), with a slice of GAP 10 (Python-side cost
that is not jaxpr size). SYNTHESISED.

No issue URL. This case was constructed from the mechanism, not mined. There is no bug report
behind it and none is claimed.

THE POINT. Every other case in this corpus is dominated by what XLA does to the module. This one
is dominated by what JAX does to build the module, and it is engineered so that *the module is the
same either way*. Both arms emit:

    identical jaxpr equation counts, identical primitives, identical shapes, identical dtypes
    identical StableHLO CHARACTER COUNT  (657190 chars at E=1200, measured, byte-for-byte equal)
    identical optimised HLO line counts  (2441 lines at E=1200, measured)
    identical FLOPs and identical runtime

and they differ by 5-11x in `.lower()`. Anything that reads the emitted program -- op counts, HLO
text size, pass timings, module bytes -- is blind to this case by construction. That is the whole
reason it is here: it is the negative control for every program-size-based heuristic in the tool.

MECHANISM. `jax.experimental.xla_metadata.set_xla_metadata(**kv)` attaches a key/value dict to
every jaxpr equation traced inside the context. At lowering, `jax/_src/interpreters/mlir.py` turns
each equation's metadata into an MLIR `DictionaryAttr` of `StringAttr`s hung off the op as
`mhlo.frontend_attributes`. MLIR attributes are UNIQUED: a `StringAttr` for a string the context
has already seen is a hash lookup returning the existing pointer, while a new string is an
allocation plus an insertion into the uniquer, plus the dictionary attribute itself is uniqued on
the sorted key/value pointer list. So the cost is set by the number of DISTINCT metadata values,
not by the number of annotated equations and not by the number of bytes.

The independent variable is therefore *distinctness*, which is invisible in every size metric.

WHAT THE CONTROL ISOLATES. `meta_distinct_E` annotates each of the E equations with a distinct
9-character tag `v00000000, v00000001, ...`. `meta_distinct_E_control` annotates each of the same E
equations with the SAME 9-character tag `v00000000`. Same number of `set_xla_metadata` contexts,
same number of annotated equations, same key, same value length, same total metadata bytes in the
serialised module. One string constant differs: `f"v{i:08d}"` versus `"v00000000"`.

MEASURED IN-ENV BEFORE COMMITTING (jax 0.10.2, JAX_PLATFORMS=cpu, x64 on, x = (8,8) f64,
3 interleaved repeats per point, medians, trace and lower timed separately):

      E     distinct trace   distinct lower  |  control trace   control lower  |  lower ratio
    300         0.132 s          0.493 s     |     0.120 s         0.063 s     |     7.8x
    600         0.558 s          1.029 s     |     0.546 s         0.274 s     |     3.8x
   1200         1.337 s          1.186 s     |     0.573 s         0.201 s     |     5.9x
   2400         5.212 s          4.398 s     |     1.284 s         0.393 s     |    11.2x

CONFIRMED IN FRESH PROCESSES (one process per measurement, arms interleaved, E=1200), which is how
the harness measures and is the number to trust:

    meta_distinct_1200            lower 7.153 / 8.013 s     compile 5.009 / 3.951 s   hlo 2445
    meta_distinct_1200_control    lower 1.366 / 0.997 s     compile 4.906 / 3.934 s   hlo 2445
    scopename_distinct_1200       lower 1.184 s             compile 5.302 s           hlo 2445
    scopename_distinct_1200_ctrl  lower 1.969 s             compile 7.161 s           hlo 2445

-> metadata arm 6.4x on `lower_s`, 1.0x on `compile_s`, identical HLO line count.
-> name-scope arm INVERTED (the control is slower), i.e. negative, as described below.

In a separate single-shot run at E=1200 the BACKEND stage was likewise flat -- 4.609 s for the
distinct arm against 5.031 s for the control, the control nominally *slower*. So the split is:

    lowering (and tracing)   5-11x apart
    backend compile          1.0x apart, within noise

THE COST IS SPLIT ACROSS TWO PRE-BACKEND STAGES, WHICH IS WHY THIS FILE IS TAGGED FOR TWO GAPS.
`jax.make_jaxpr` alone -- pure tracing, no MLIR at all -- was 6.841 s against 1.394 s at E=2400
with the equation counts verified identical (4801 in both arms), so roughly half the excess is
Python-side bookkeeping in `set_xla_metadata` and the equation parameter dicts, and the other half
is MLIR attribute uniquing during lowering. Any tool that reports a single number for "pre-XLA" is
merging two mechanisms here; the trace/lower split in the table above is the ground truth for
separating them.

This is the case the harness's separate `lower_s` column exists for. A profiler that instruments
`backend_compile` reports nothing here at all. Note also that the harness's own verdict function
keys on `compile_s`, so this case will very likely be scored "no (below floor)" or "no (Nx
control)" by `_harness.classify` even when it is reproducing perfectly -- read `lower_s`.

SIZE SWEEP. E in (300, 600, 1200, 2400). The claim being tested is that lowering cost grows with
the distinct-value count at a large constant, not that it is superlinear; four points are enough to
show the ratio persists rather than being one lucky size. The trace-side numbers above hint at
superlinearity between E=1200 and E=2400 (1.337 -> 5.212 s for a 2x size step) which the sweep will
either confirm or refute.

THE THIRD ARM IS A DELIBERATE NEGATIVE RESULT. `scopename_distinct_E` / `_control` do the exactly
analogous thing with `jax.named_scope` -- E equations under a distinct scope name each, versus E
equations under one shared scope name -- because `source_info_to_location` builds an
`ir.Location.name` per equation out of `str(name_stack)` and the same uniquing argument ought to
apply. IT DOES NOT. Measured at E=1200 with 400-character scope names:

    distinct scope names          lower 1.025 s
    identical long scope names    lower 0.651 s
    identical short scope names   lower 1.420 s     <-- ordering inverts; this is noise
    no scope at all               lower 0.942 s

Flat, with the arms out of order. Location names are evidently either cached or cheap enough not to
matter, and a separate length sweep of the metadata arm (value length 4 / 40 / 400 / 4000 chars at
E=600) was also flat, confirming the cost tracks distinct-value COUNT and not bytes. Both negatives
are kept in the file, wired as a proper pair, because a tool that fires on the name-scope arm is
firing on "there is metadata" rather than on the mechanism, and this is how you catch that.

PLATFORM: either. Lowering is jaxpr -> StableHLO in Python and MLIR; no backend is involved.
Measured on CPU; the GPU arm is unverified only because the GPU was off-limits when this was
written, not because anything device-specific is suspected.

MEMORY: negligible. The largest array is (8,8) f64 = 512 bytes. The cost is entirely host-side
compile work.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.xla_metadata import set_xla_metadata

# 9 characters, so `f"v{i:08d}"` and the shared constant below have EXACTLY equal length and the
# serialised module has exactly equal size in the two arms.
SHARED_TAG = "v00000000"

# Scope names for the negative-result arms.  Long on purpose: if length mattered at all, 400
# characters x E equations would show it.
SCOPE_LEN = 400
SHARED_SCOPE = "s" + "0" * SCOPE_LEN


def _chain(y):
    """One annotated-free equation pair.  Cheap at runtime, exactly 2 primitives."""
    return y * 1.0000001 + 0.5


def _meta(x, nmeta: int, distinct: bool):
    y = x
    for i in range(nmeta):
        with set_xla_metadata(tag=(f"v{i:08d}" if distinct else SHARED_TAG)):
            y = _chain(y)
    return jnp.sum(y)


def _scope(x, nmeta: int, distinct: bool):
    y = x
    for i in range(nmeta):
        with jax.named_scope(f"s{i:0{SCOPE_LEN}d}" if distinct else SHARED_SCOPE):
            y = _chain(y)
    return jnp.sum(y)


# numpy at module scope: importing this file claims no device.
X = np.ones((8, 8), dtype=np.float64)

SIZES = (300, 600, 1200, 2400)
SCOPE_AT = (1200,)          # the negative arm needs one size, not a sweep

CASES = {}

for _e in SIZES:
    CASES[f"meta_distinct_{_e}"] = (
        functools.partial(_meta, nmeta=_e, distinct=True), (X,),
        f"gap9 synth: {_e} equations each carrying a DISTINCT xla_metadata value -> {_e} distinct "
        f"MLIR StringAttr/DictionaryAttr uniquings during jaxpr->StableHLO lowering",
    )
    CASES[f"meta_distinct_{_e}_control"] = (
        functools.partial(_meta, nmeta=_e, distinct=False), (X,),
        f"control: the same {_e} equations, the same {_e} metadata dicts, the same key and the "
        f"same 9-char value LENGTH -- one shared value, so MLIR uniquing hits every time. "
        f"Byte-identical StableHLO size and identical optimised HLO",
    )

for _e in SCOPE_AT:
    CASES[f"scopename_distinct_{_e}"] = (
        functools.partial(_scope, nmeta=_e, distinct=True), (X,),
        f"gap9 synth, MEASURED NEGATIVE: {_e} equations under {_e} distinct {SCOPE_LEN}-char "
        f"jax.named_scope names -> distinct ir.Location.name per op. Flat on CPU (1.03 s vs "
        f"0.65 s control, and 1.42 s for a SHORT shared name -- ordering inverts). Kept as the "
        f"discriminator: a tool that fires here is firing on 'metadata exists', not the mechanism",
    )
    CASES[f"scopename_distinct_{_e}_control"] = (
        functools.partial(_scope, nmeta=_e, distinct=False), (X,),
        f"control: the same {_e} equations under ONE shared {SCOPE_LEN}-char scope name",
    )
