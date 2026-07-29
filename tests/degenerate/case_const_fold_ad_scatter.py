"""pyroki#56 -- AD turns a constant-index gather into an ALL-CONSTANT scatter, which XLA then folds.

    https://github.com/chungmin99/pyroki/issues/56

WHAT THE ISSUE REPORTS.  Compiling a pyroki/jaxls inverse-kinematics solve stalls, and XLA names the
instruction itself:

    Constant folding an instruction is taking > 1s: ... scatter-add
    op_name: vmap(vmap(transpose(vmap(jvp(jit(multiply))))))/jit(multiply)/scatter-add
    source:  jaxlie/_so3.py:476

``jaxlie.SO3.multiply`` writes the quaternion product as
``jnp.sum(signs * q_outer[..., terms_i, terms_j], axis=-1)`` with NumPy-literal index tables.
Reported on CPU.

WHY THIS CASE EARNS ITS PLACE IN THE CORPUS.  The corpus already has constant folding
(``case_constant_folding_dus.py``, jax#12789).  What it does not have is constant-ness with THIS
provenance.  In jax#12789 the user wrote the literal: ``jnp.ones((500,500,500))`` is right there in
the source, and pointing at it is a matter of reading the program.  Here NOTHING THE USER WROTE IS
CONSTANT.  The user wrote a gather -- indexing a live array by a fixed table.  Reverse-mode AD
transposes that gather into a scatter-add, and the transpose rule supplies the operand (a zeros
broadcast) and, wherever the incoming cotangent is a seed rather than data (the ``.sum()`` of a
loss, or a jacobian's identity basis), the updates as well.  So all three scatter operands are
compile-time constants that AD MANUFACTURED, in a library the user did not write, reached through a
transpose the user did not ask for.  Attribution has to point through an AD transformation into a
third-party source line -- and XLA's own alarm gives the ground truth to check the answer against.

GROUND TRUTH MEASURED IN THIS ENVIRONMENT (CPU, jax 0.10.2, lowering only, no execution).  A
provenance walk over the lowered stablehlo -- for each ``stablehlo.scatter``, are all three operands
rooted in ``stablehlo.constant`` rather than in a ``%arg``? -- gives:

    arm                                        scatters   ALL-CONSTANT   biggest const scatter
    adconst_idx_*   grad(sum(W * x[IDX]))             1        1          M elements (the knob)
      _control      IDX passed in as an argument      1        0          --
      _wtraced      W   passed in as an argument      1        0          --
    adconst_quat_*  jacrev(vmap(quaternion chain))  L-1        1          64 * L * B^2 elements
      _control      index tables passed as arguments L-1       0          --
      _stack        pre-optimisation jaxlie formula   0        0          --
    adconst_quat_permap  vmap(jacrev(chain))        L-1        1          512 elements, FIXED

and, for the exact configurations shipped below: 262 144 elements at M = 2**18, 4 194 304 at
M = 2**22, and 131 072 / 2 097 152 / 4 194 304 for (L,B) = (8,16) / (8,64) / (4,128).

Two facts from that table are the design of this file.  First, exactly ONE scatter per program is
all-constant -- the one at the top of the reverse graph, where the cotangent is still the seed;
every other scatter's updates are data and are therefore left alone.  Second, the constant one's
size is a knob: for the distilled arm it is M, and for the faithful quaternion arm it is
``64 * L * B**2``, QUADRATIC in the batch (verified: 32 768 / 131 072 / 524 288 elements at
B = 8 / 16 / 32 with L = 8).

THE DISTILLED ARM (``adconst_idx_*``), which is what should carry the file.  Four lines:

    IDX = <random constant int array, M entries>;  W = <random constant f64 array, M entries>
    f  = lambda x: jnp.sum(W * x[IDX])
    jax.grad(f)

Verified by lowering at M = 4096: the jaxpr contains ``scatter-add(broadcast(0.0), IDX, W*1.0)``,
all three operands constant, and ``main()`` TAKES NO ARGUMENTS AT ALL -- the gradient of a fixed
linear form is a compile-time constant, so XLA is entitled to evaluate the whole scatter and emit a
literal.  Its evaluator does that one update index at a time.  M is therefore a clean dial on
exactly the quantity the mechanism predicts, with the program held at ten equations.

AND IT FIRES.  Verified by compiling the distilled arm in this environment (CPU, jax 0.10.2).  The
POST-optimisation HLO is, in its entirety:

    ENTRY %main.2 () -> f64[4096] {
      %constant = f64[4096]{0} constant({...}),
          metadata={op_name="jit(<lambda>)/transpose(jvp())/scatter-add"}
      ROOT %copy = f64[4096]{0} copy(%constant)
    }

The scatter is GONE: XLA evaluated it at compile time and replaced the whole gradient with a
literal.  Two things follow.  (1) The executable does no work at all, so ``compile/runtime`` is
whatever the harness's timer floor allows -- the control comparison, not the ratio, is the test
here.  (2) The surviving constant carries ``op_name="...transpose(jvp())/scatter-add"`` in its
metadata, which is GROUND TRUTH for attribution in the same way XLA's slow-compile alarm is: the
compiler itself records that this literal came from a transpose, and any answer scopex gives can be
checked against it.  Note what that op_name does NOT contain: a user source line.  The provenance
is a transformation, not a location.

CPU compile times measured while checking the above (M / lower / compile): 2**10 / 0.14 s / 0.02 s,
2**16 / 0.02 s / 0.23 s, 2**18 / 0.02 s / 0.82 s.  Growth is roughly linear in M once past a floor,
which extrapolates to ~13 s at M = 2**22 -- an EXTRAPOLATION, not a measurement, and the reason the
ladder tops out there rather than lower.

THE FAITHFUL ARM (``adconst_quat_*``) reproduces jaxlie's formulation exactly -- the same
``jnp.sum(signs * outer[..., terms_i, terms_j], axis=-1)``, verified against the textbook Hamilton
product -- composed into an L-link chain (a kinematic chain, which is what pyroki is doing) and
differentiated the way jaxls does, as a Jacobian over a batch of factors.  ``jacrev(vmap(chain))``
is the arrangement that reproduces the reported ``vmap(vmap(transpose(vmap(jvp))))`` op_name AND
lets the constant grow, because jacrev's identity seed spans the whole batch.

``adconst_quat_permap`` is ``vmap(jacrev(chain))`` -- the per-factor Jacobian, arguably closer to
what jaxls actually builds.  Verified: its all-constant scatter is 512 elements at EVERY batch size,
because jax hoists the batch-invariant seed out of the vmap.  It is in the file as a NEGATIVE
control on the size story: if it also compiles slowly, folding size is not the mechanism.

WHAT EACH CONTROL ISOLATES.

  * ``_control`` (auto-paired): the index tables become traced ARGUMENTS.  Verified: the same
    scatters with the same shapes are emitted and ZERO of them are all-constant; the lowered module
    grows by 1 line out of 23 (distilled arm) and by 10-26 lines out of 203-473 (quaternion arms),
    all of it index-clamping on parameters instead of on literals.  The op graph is the same; only
    constant-ness differs.  This is the tightest control in the file.
  * ``_wtraced`` (distilled arm only): indices stay CONSTANT, the update values become an argument.
    Verified all-constant = 0.  It separates "constant indices" from "constant everything", which
    matters because only the latter lets the folder fire.
  * ``_stack``: the pre-optimisation jaxlie formulation, an explicit ``jnp.stack`` of the four
    components.  Identical numerics (verified exactly equal, and both verified against the textbook
    Hamilton product), no advanced indexing, hence NO GATHER and verified zero scatters.  Note what
    this control costs: 676 stablehlo lines against 203 for the gather version at the shipped
    (L,B) = (4,128), and 1600 against 447 at L = 8.  THE FAST ARM IS 3.3x THE BIGGER PROGRAM.  Any
    "compile time follows program size" heuristic gets the sign wrong here, which is a second
    reason to keep this arm.

UNCERTAINTY, stated plainly.  The mechanism is CONFIRMED live in this environment -- the
all-constant scatter is emitted at the sizes tabulated and XLA does fold it -- so the only open
question is magnitude: whether M = 2**22 buys enough folding to clear the 3 s floor.  The CPU trend
above says yes by roughly 4x; that is an extrapolation and it may not hold, and XLA additionally
refuses to fold outputs above ~45M elements, which is why every arm here is kept well below that
ceiling.  A flat result at M = 2**22 would mean the folder has been given a cheaper path or a
tighter guard since the report, which is a result worth having and retires a family of candidates.
The reporter ran on CPU and constant folding is a host-side pass, so it should carry to GPU -- but
``--platform cpu,cuda`` is the way to check rather than assume, and the quaternion arms in
particular have real runtime work on GPU that the distilled arms do not.

MEMORY.  Distilled arms: M f64 plus M i32, 50 MB at the largest M.  Quaternion arms: the dense
Jacobian of a batched map is (4LB)^2 f64, which is why L is dropped to 4 when B is raised to 128 --
every configuration below holds ~130 MB or less.
"""

from __future__ import annotations

import functools

import numpy as np

import jax
import jax.numpy as jnp

# --------------------------------------------------------------------------------------------
# Distilled arm: constant-index advanced indexing under grad.  M = the number of constant scatter
# updates, which is the quantity the mechanism says the fold cost tracks.  2**22 = 4.2M stays well
# under XLA's ~45M-element constant-folding ceiling.
# --------------------------------------------------------------------------------------------
M_LADDER = (1 << 18, 1 << 20, 1 << 22)

# Faithful arm: (chain length L, batch B).  The all-constant scatter is 64 * L * B**2 elements.
QUAT_SHAPES = ((8, 16), (8, 64), (4, 128))

# jaxlie's quaternion product tables, in jaxlie's own layout: entry [r, c] of the gathered array is
# outer[terms_i[r, c], terms_j[r, c]], and the row sum with `signs` is the r-th product component.
# Verified against the textbook Hamilton product for random quaternions.
TERMS_I = np.array([[0, 1, 2, 3]] * 4, dtype=np.int32)
TERMS_J = np.array([[0, 1, 2, 3],
                    [1, 0, 3, 2],
                    [2, 3, 0, 1],
                    [3, 2, 1, 0]], dtype=np.int32)
SIGNS = np.array([[1., -1., -1., -1.],
                  [1., 1., 1., -1.],
                  [1., -1., 1., 1.],
                  [1., 1., -1., 1.]])


# --------------------------------------------------------------------------------------------
# distilled
# --------------------------------------------------------------------------------------------
# CONSTANT-NESS COMES FROM CLOSING OVER, NOT FROM PASSING IN.  Everything in the harness's ``args``
# tuple becomes a traced jit PARAMETER, so an index table handed over as an argument is by
# construction not a constant -- that is precisely how the controls below are built, and it is why
# the pathological arms must capture their tables from the enclosing scope instead.

def _mk_grad_const(idx, w):
    """SLOW: idx and w are captured constants, so AD's transpose emits scatter(const, const, const)."""
    def f(x):
        return jax.grad(lambda v: jnp.sum(w * v[idx]))(x)
    return f


def _mk_grad_idx_traced(w):
    """CONTROL: identical scatter, index operand is a PARAMETER, so the folder cannot fire."""
    def f(x, idx):
        return jax.grad(lambda v: jnp.sum(w * v[idx]))(x)
    return f


def _mk_grad_w_traced(idx):
    """CONTROL on the other operand: indices stay constant, update values become a parameter."""
    def f(x, w):
        return jax.grad(lambda v: jnp.sum(w * v[idx]))(x)
    return f


def _mk_grad_live(idx, w):
    """Variant with a live data path, so the executable is not a pure constant.

    The pure arm's ``main()`` takes no arguments once folded, which invites the objection that the
    whole program was constant rather than the scatter.  Here ``tanh`` keeps a genuine data
    dependence in the output while the scatter's three operands stay exactly as constant as before.
    """
    def f(x):
        return jax.grad(lambda v: jnp.sum(w * v[idx]) + jnp.sum(jnp.tanh(v)))(x)
    return f


# --------------------------------------------------------------------------------------------
# faithful: jaxlie SO3.multiply
# --------------------------------------------------------------------------------------------
def _mul_gather(q0, q1, ti, tj):
    """jaxlie/_so3.py:476, verbatim in shape: outer product, constant-index gather, row sum."""
    outer = q0[..., :, None] * q1[..., None, :]
    return jnp.sum(SIGNS * outer[..., ti, tj], axis=-1)


def _mul_stack(q0, q1):
    """CONTROL: the pre-optimisation formulation -- same numbers, no indexing, so no gather."""
    w0, x0, y0, z0 = q0[..., 0], q0[..., 1], q0[..., 2], q0[..., 3]
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    return jnp.stack([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                      w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                      w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                      w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1], axis=-1)


def _chain(qs, ti, tj):
    """An L-link chain of quaternion products; returns every partial product (a kinematic chain)."""
    out = [qs[0]]
    for i in range(1, qs.shape[0]):
        out.append(_mul_gather(out[-1], qs[i], ti, tj))
    return jnp.stack(out)


def _chain_stack(qs):
    out = [qs[0]]
    for i in range(1, qs.shape[0]):
        out.append(_mul_stack(out[-1], qs[i]))
    return jnp.stack(out)


def _batchjac_const(qs):
    """SLOW: Jacobian of the batched chain. jacrev's identity seed is constant and spans the batch,
    so the top scatter is scatter(const, const, const) with 64*L*B**2 elements."""
    return jax.jacrev(lambda q: jax.vmap(_chain, in_axes=(0, None, None))(q, TERMS_I, TERMS_J))(qs)


def _batchjac_traced(qs, ti, tj):
    """CONTROL: identical op graph, index tables arrive as arguments, nothing is all-constant."""
    return jax.jacrev(lambda q: jax.vmap(_chain, in_axes=(0, None, None))(q, ti, tj))(qs)


def _batchjac_stack(qs):
    """CONTROL: no gather at all -- and 3.6x the stablehlo lines of the arm it controls."""
    return jax.jacrev(lambda q: jax.vmap(_chain_stack)(q))(qs)


def _permap_jac(qs):
    """NEGATIVE control on the size story: per-factor Jacobian. Still has exactly one all-constant
    scatter, but it is 512 elements at every B because jax hoists the batch-invariant seed."""
    return jax.vmap(lambda q: jax.jacrev(_chain)(q, TERMS_I, TERMS_J))(qs)


def _permap_jac_traced(qs, ti, tj):
    return jax.vmap(lambda q: jax.jacrev(_chain)(q, ti, tj), in_axes=(0,))(qs)


# --------------------------------------------------------------------------------------------
_TABLES: dict[int, tuple] = {}


def _idx_tables(m: int):
    """(x, idx, w) for size m, built once -- importing this file must not allocate three times."""
    if m not in _TABLES:
        rng = np.random.default_rng(56)
        # np.zeros is calloc-backed, so the (unused, immediately dropped) input costs no physical
        # pages until jax transfers it.
        _TABLES[m] = (np.zeros(m),
                      rng.integers(0, m, size=m, dtype=np.int32),
                      rng.standard_normal(m))
    return _TABLES[m]


def _quat_args(lgt: int, b: int):
    return (np.random.default_rng(56).standard_normal((b, lgt, 4)),)


CASES: dict = {}

for _m in M_LADDER:
    _e = _m.bit_length() - 1
    _x, _i, _w = _idx_tables(_m)
    CASES[f"adconst_idx_2p{_e}"] = (
        _mk_grad_const(_i, _w), (_x,),
        f"pyroki#56 distilled: grad of sum(W*x[IDX]) with CAPTURED (constant) IDX -- AD "
        f"manufactures scatter-add(const, const, const), M=2**{_e} updates for XLA to fold",
    )
    CASES[f"adconst_idx_2p{_e}_control"] = (
        _mk_grad_idx_traced(_w), (_x, _i),
        f"control: identical scatter, IDX arrives as an argument so the index operand is not "
        f"constant and the folder skips it, M=2**{_e}",
    )

_e = M_LADDER[-1].bit_length() - 1
_x, _i, _w = _idx_tables(M_LADDER[-1])
CASES[f"adconst_idx_2p{_e}_wtraced"] = (
    _mk_grad_w_traced(_i), (_x, _w),
    f"control on the OTHER operand: indices stay constant, update VALUES arrive as an argument. "
    f"Separates 'constant indices' from 'constant everything'. M=2**{_e}",
)
CASES[f"adconst_idx_2p{_e}_live"] = (
    _mk_grad_live(_i, _w), (_x,),
    f"same all-constant scatter but with a live tanh path, so the folded program is not a pure "
    f"constant and the executable still does real work. M=2**{_e}",
)

for _l, _b in QUAT_SHAPES:
    _tag = f"adconst_quat_L{_l}B{_b}"
    _n_const = 64 * _l * _b ** 2
    CASES[_tag] = (
        _batchjac_const, _quat_args(_l, _b),
        f"pyroki#56 faithful: jacrev(vmap({_l}-link jaxlie quaternion chain)), B={_b} -- one "
        f"all-constant scatter of {_n_const} elements (verified by lowering)",
    )
    CASES[f"{_tag}_control"] = (
        _batchjac_traced, _quat_args(_l, _b) + (TERMS_I, TERMS_J),
        f"control: same op graph, terms_i/terms_j as traced arguments -- 0 all-constant scatters, "
        f"L={_l} B={_b}",
    )

_L, _B = QUAT_SHAPES[-1]
CASES[f"adconst_quat_L{_L}B{_B}_stack"] = (
    _batchjac_stack, _quat_args(_L, _B),
    f"formulation control: pre-optimisation jaxlie stack formula, identical numerics, no gather "
    f"hence no scatter -- and 3.3x MORE stablehlo lines (676 vs 203) than the arm it controls "
    f"(L={_L} B={_B})",
)

_L, _B = QUAT_SHAPES[1]
CASES[f"adconst_quat_permap_L{_L}B{_B}"] = (
    _permap_jac, _quat_args(_L, _B),
    f"negative control on the SIZE story: vmap(jacrev(chain)) keeps exactly one all-constant "
    f"scatter but at a fixed 512 elements for any B (L={_L} B={_B})",
)
CASES[f"adconst_quat_permap_L{_L}B{_B}_control"] = (
    _permap_jac_traced, _quat_args(_L, _B) + (TERMS_I, TERMS_J),
    f"control: per-factor Jacobian with traced index tables, L={_L} B={_B}",
)
