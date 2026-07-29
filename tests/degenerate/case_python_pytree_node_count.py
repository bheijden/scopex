"""GAP 10 (Python-side compile cost that is NOT jaxpr size), SYNTHESISED.

No issue URL. Constructed from the mechanism. Nothing here is a bug -- flattening a `dict` pytree
has to sort its keys to get a deterministic leaf order -- it is a workload shape in which the
compile bill is set by a quantity that no program-level metric records.

WHAT MAKES IT DIFFERENT FROM `case_lowering_arity_pytree`. That case (jax#4667) varies the NUMBER
OF LEAVES, so the jaxpr, the StableHLO and the HLO all grow together with the independent variable
and every size metric moves at once. This case holds the leaf count FIXED at 32 and varies only the
number and container type of the NON-LEAF nodes. Every arm in this file produces:

    the same 11584 jaxpr equations       (verified: identical across all twelve arms)
    the same 32 array leaves of shape (4,) f64
    the same StableHLO, the same HLO, the same FLOPs, the same answer

and the extreme arms are ~4.7x apart. Leaf count, equation count and op count are constant by
construction, so any heuristic built on them draws a flat line across this file.

MECHANISM. `jax.tree.map` flattens each input tree and unflattens the result, once per call, and
the walk covers the whole treedef rather than just the leaves. `None` is an EMPTY pytree node in
JAX -- one node, zero leaves -- so a state tree can be made node-heavy and leaf-light without
changing a single equation. That shape is not contrived: frozen-parameter masks, optional optimiser
slots, disabled adapters and `None` gradients all put exactly it into real training code, and the
tree is re-flattened on every `tree.map` of the update step.

The expensive part turns out to be the CONTAINER, not the node count. `dict` flatten sorts its keys
every time; `list` does not.

MEASURED IN-ENV BEFORE COMMITTING (jax 0.10.2, JAX_PLATFORMS=cpu, x64 on, one FRESH PROCESS per
measurement with the backend pre-warmed, arms interleaved; 32 leaves of shape (4,), R=120
`tree.map` rounds, M extra empty `None` nodes). `lower` is `jit(fn).lower(*args)`, i.e. tracing
plus jaxpr->StableHLO:

    container      M          lower              compile        eqns   HLO lines
    list           0    4.616 / 3.661 s     3.482 / 4.792 s    11584     8396
    list        4000    3.821 s             5.623 s            11584     8396
    list       16000    3.807 s             5.232 s            11584     8396
    list       64000    5.054 s             3.921 s            11584     8396
    dict        4000    3.667 s             4.913 s            11584     8396
    dict       16000    7.109 s             8.013 s            11584     8396
    dict       64000   23.485 / 24.084 s    5.063 / 5.638 s    11584     8396

Read down the two blocks and the attribution is unambiguous:

  * the LIST block is FLAT: 3.7-5.1 s from M=0 to M=64000. Adding 64000 empty nodes to a list costs
    essentially nothing. NODE COUNT ON ITS OWN IS A NEGATIVE RESULT, recorded deliberately because
    it bounds where the cost is not.
  * the DICT block, at the SAME node counts and with everything else identical, runs 3.67 / 7.11 /
    23.8 s. Excess over the ~3.8 s baseline is 0 / 3.3 / 20.0 s for M = 4k / 16k / 64k -- roughly
    6x per 4x step in M, i.e. clearly SUPERLINEAR, consistent with a sort. At M=64000 the two
    containers are 4.7x apart.
  * the backend column is flat and unordered across every row (3.5-8.0 s) and the optimised HLO is
    8396 lines in all seven rows. The cost is entirely host-side.

An earlier in-process sweep suggested the list arm also grew with M (2.4 s -> 9.1 s over M=0 ->
64000). That measurement was contaminated -- `jax.make_jaxpr` caches on (fn, treedef, avals), so
re-timing the same arguments in one process reads back a cache hit, and arm order then decides who
pays. The fresh-process numbers above supersede it. This is the same lesson `_harness` records in
its own docstring and it bit this file too.

WHAT EACH CONTROL ISOLATES.

  pytree_keyed_M           32 array leaves + M empty nodes carried in a DICT (keys sorted on every
                           flatten)
  pytree_keyed_M_control   the same 32 leaves + the same M empty nodes carried in a LIST. Same node
                           count, same leaves, same 11584 equations, same 8396-line HLO. The ONLY
                           difference is whether flatten sorts. Measured 1.0x / 1.9x / 4.7x at
                           M = 4000 / 16000 / 64000 -- so the sweep is load-bearing here: at the
                           smallest size the case does not reproduce at all.

  pytree_nodes_M           32 array leaves + M empty nodes in a LIST
  pytree_nodes_M_control   32 array leaves + ZERO extra nodes in a list. Container held fixed, node
                           count varied. MEASURED FLAT (1.0x at every M) -- this pair is the
                           negative arm, and a tool that fires on it has found "the tree is big"
                           rather than the mechanism.

SIZE SWEEP: M in (4000, 16000, 64000), a 16x range, and it is not decoration. The dict/list ratio
goes 1.0x -> 1.9x -> 4.7x across it, so a single-size version of this file at M=4000 would have
been filed "does not reproduce". The two arms have different exponents by construction: the list
walk is linear and cheap enough to vanish into noise, while `dict` flatten sorts M keys on each of
the 3R flattens.

PLATFORM: either. This is `jax.tree_util` and Python; the backend never sees the difference and the
backend column is measurably flat. Measured on CPU only because the GPU was owned by another
investigation when this was written.

MEMORY: negligible on device (32 x 4 f64 = 1 KB). The M=64000 arms build 64k-entry Python
containers at `CASES` construction, costing ~1.5 s of import time and a few MB of host memory --
relevant because `_harness.discover()` imports every case file in the directory.

CAVEAT: `_harness.classify` renders its verdict on `compile_s`, which is flat here by design, so
the harness will score this case "no". The 4.7x is in `lower_s`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

NLEAF = 32       # array leaves, held FIXED across every arm
LEAF = 4         # elements per leaf
ROUNDS = 120     # tree.map applications; multiplies the per-flatten cost


def _core() -> dict:
    """The 32 real leaves. numpy at module scope: importing this file claims no device."""
    return {f"w{i:05d}": np.zeros(LEAF, dtype=np.float64) for i in range(NLEAF)}


def _build(nextra: int, container: str):
    """State tree: 32 array leaves plus `nextra` EMPTY nodes (None), in a dict or a list.

    `None` is an empty pytree node: it adds a node and no leaf, so the jaxpr is unchanged.
    """
    if container == "dict":
        extra = {f"z{j:06d}": None for j in range(nextra)}
    else:
        extra = [None] * nextra
    return (_core(), extra)


def _update(state, params):
    for _ in range(ROUNDS):
        state = jax.tree.map(lambda a, b: a * 0.999 + b * 1e-3, state, params)
    return sum(jnp.sum(v) for v in jax.tree.leaves(state))


SIZES = (4000, 16000, 64000)

# Built once and shared by every arm that needs it, so import time stays near the 1.5 s the
# 64000-entry containers already cost.
_EMPTY_LIST_ARGS = (_build(0, "list"), _build(0, "list"))

CASES = {}

for _m in SIZES:
    _dict_args = (_build(_m, "dict"), _build(_m, "dict"))
    _list_args = (_build(_m, "list"), _build(_m, "list"))

    CASES[f"pytree_keyed_{_m}"] = (
        _update, _dict_args,
        f"gap10 synth: {NLEAF} array leaves + {_m} EMPTY (None) nodes in a DICT, {ROUNDS} "
        f"tree.map rounds -> {_m} keys sorted on every flatten. 11584 equations, unchanged",
    )
    CASES[f"pytree_keyed_{_m}_control"] = (
        _update, _list_args,
        f"control: the same {NLEAF} leaves and the same {_m} empty nodes in a LIST -- same node "
        f"count, same equations, same HLO, no key sort. Measured 4.7x apart at M=64000",
    )
    CASES[f"pytree_nodes_{_m}"] = (
        _update, _list_args,
        f"gap10 synth, MEASURED NEGATIVE: {_m} empty nodes in a LIST against none at all. "
        f"Container held fixed, node count varied -- flat on CPU (5.05 s vs 4.5-5.8 s)",
    )
    CASES[f"pytree_nodes_{_m}_control"] = (
        _update, _EMPTY_LIST_ARGS,
        f"control: the identical {NLEAF} leaves with ZERO extra nodes. The discriminator: a tool "
        f"that fires on this pair is reacting to tree size, not to the key sort",
    )
