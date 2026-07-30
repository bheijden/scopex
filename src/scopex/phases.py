"""Split ``record()['backend']`` into the phases XLA does not name for you.

:func:`scopex.record` gives one number for the whole backend compile. That number covers the HLO
pass pipeline, the backend's IR emitter, LLVM's own optimiser and object codegen -- four phases that
want four different responses, and only the first of which XLA's pass timer can see. On the corpus's
signature case (jax#32704, chained 2-D gather, CPU) ``pass_timings`` accounts for **0.1%** of the
compile; the other 99.9% is below the pass pipeline and every count-based instrument in this package
reads it as a null.

:func:`backend_split` reaches it in ONE compile, from the mtimes of the files XLA already writes
under ``--xla_dump_to``. No kill switches, no differencing, and it cannot go negative.

WHY NOT KILL-SWITCH DIFFERENCING, WHICH WAS THE PROPOSED IMPLEMENTATION
----------------------------------------------------------------------
Because it measures the wrong quantity, and it does so worst exactly where it matters. A kill switch
answers "how much total compile time disappears if this pass never runs", which includes every
downstream cost the pass CREATED. It does not answer "how long did this pass take".

Measured on jax#32704 ncycles=8, CPU, baseline backend 12.7 s (three INTERLEAVED baselines --
12.795 / 12.643 / 12.841 s -- so the differences below are not drift):

===========================================  ========  =========================  =========
lever                                        total     implied by differencing    truth
===========================================  ========  =========================  =========
``--xla_disable_all_hlo_passes=true``        0.252 s   hlo_passes = 12.47 s       0.023 s
``--xla_disable_hlo_passes=fusion``          0.217 s   the fusion pass = 12.53 s  0.00128 s
===========================================  ========  =========================  =========

The second row is the surgical lever, the one a reader would trust most, and it overstates the
``fusion`` pass by **9,800x**. The mechanism is not subtle: ``fusion`` runs in 1.3 ms and produces
one ``gather_bitcast_fusion`` whose EMISSION then costs 12.5 s. Deleting the pass deletes the
emission, and the difference charges all of it to the pass. So on the one corpus case where the
answer is independently known, kill-switch differencing reports ``hlo_passes`` at 97.5% of the
backend when the true answer is ``emitter`` at 99.2%. It names the wrong phase, confidently, with no
warning -- this package's signature failure, in a new place.

Three further defects, each independently disqualifying:

* ``--xla_disable_all_hlo_passes=true`` **aborts the process** on jax#2609 (ndtri jacrev d4):
  ``F hlo_value.h:241] Check failed: values_.size() == 1 (56 vs. 1)``. That is a fatal ``Check``,
  not an exception, so it takes the caller's interpreter down and cannot be run in-process at all.
* Differences go **negative**: ``--xla_llvm_disable_expensive_passes=true`` measured 11.182 s
  against a 10.776 s baseline, i.e. ``llvm_opt = -0.406 s``.
* The noise floor swamps every small bucket. Six identical baseline compiles spanned 5.621-6.616 s
  (16.7% spread, 7.5% stdev), and the FIRST compile of a session read 10.776 s against a warm steady
  state of 5.967 s -- an 80% systematic bias on whichever arm runs first, which is by construction
  the baseline. Three of the four target buckets are under 0.3% of that compile and are unresolvable
  by differencing in principle.

Kill switches remain the right tool for the question they actually answer -- *which knob makes this
go away* -- which is how ``--xla_cpu_use_fusion_emitters=false`` (12.7 s -> 0.072 s) named the
emitter in the original investigation. That is a bisection over CAUSES, not a decomposition of TIME,
and it must not ship under a name that suggests otherwise. One free property worth relying on if you
build it: an unknown XLA flag is FATAL (``F parse_flags_from_env.cc:234] Unknown flag in
XLA_FLAGS``), not a silent no-op, so a lever removed by a future jaxlib crashes loudly rather than
quietly returning a plausible zero.

THE PRECONDITION THAT MAKES THIS SOUND, AND THE ONE THAT DOES NOT
-----------------------------------------------------------------
XLA:CPU emits, optimises and object-codegens each LLVM *kernel module* as a unit, and does several
of them CONCURRENTLY. The three artifacts of kernel k+1 are written while kernel k is still being
optimised, so GLOBAL phase boundaries (max mtime over every ``.ir-no-opt.ll``, then over every
``.ir-with-opt.ll``, ...) describe a real ordering only when there is exactly ONE kernel module.

Measured on jax 0.10.2 / CPU:

* jax#32704 gather chain, ncycles=8 -- **1** kernel module, no overlap, the split is sound.
* jax#2609 ndtri jacrev d4 -- **224** kernel modules, 223/223 consecutive pairs overlap, and the
  split is NOT DEFINED. The naive global-boundary reading reports ``llvm_opt = 0.309 s`` while the
  per-kernel intervals sum to 4.400 s -- and even that sum is wrong, because the kernels compile in
  parallel so their wall-clock intervals overlap each other: 4.400 + 3.319 = 7.72 s exceeds the
  7.40 s backend stage it is supposed to sit inside. Wall-clock interval sums there are neither wall
  seconds nor CPU seconds, so this module reports ``sound=False`` and collapses the tail into one
  ``below_hlo`` bucket rather than ship a per-kernel sum that looks precise and is inflated 3.2x.

That is the whole design, and it is not conservatism for its own sake: multi-kernel is EXACTLY the
regime where LLVM and codegen are the answer (INVESTIGATIONS records the ORC JIT at 52% of the ndtri
compile), so a splitter that quietly mislabels there would have its blind spot aligned with the
quantity it is measuring -- the defect this package exists to prevent.

For a per-kernel bill on multi-kernel programs the route is
``TF_CPP_VMODULE=cpu_compiler=3,jit_compiler=3``, which is a separate piece of work.
"""

from __future__ import annotations

import os
import pathlib
import time
import warnings

from ._parse import emitter_dump_name

__all__ = ["backend_split", "BackendSplit"]

# The three phases, in the order XLA runs them, keyed by `_parse.emitter_dump_name`'s `kind`.
_PHASES = ("ir-no-opt", "ir-with-opt", "obj")

# Coverage outside this band revokes `sound`. See backend_split for why the two are not allowed to
# disagree.
_COVERAGE_BAND = (0.90, 1.10)


def _kernel_key(f: str):
    """``((module, kernel), phase_kind)`` for an LLVM artifact filename, else ``None``.

    The filename grammar is NOT re-parsed here. It is quarantined in :mod:`scopex._parse`, which
    already carries the verbatim samples and the reason each group is shaped the way it is -- and
    which was hardened against a real bug this function would otherwise have reintroduced: XLA
    appends ``.<n>`` to a kernel name when several would collide, so a dot-free ``kernel`` group
    reads ``265`` as the kernel on every such file. Not empty, not an error, just a disambiguator
    where the name should be.

    The key is ``(module, kernel)`` and not ``kernel`` alone because one dump holds several modules
    (JAX's warm-up ``jit_convert_element_type`` and friends alongside your program) and identically
    named kernels in two of them must not merge into one -- merging would under-count kernel modules
    and could hand a multi-kernel program the single-kernel code path, which is the one place this
    module is allowed to claim ``sound=True``.
    """
    d = emitter_dump_name(f)
    if d is None or d["kind"] not in _PHASES:
        return None
    return (d["module"], d["kernel"]), d["kind"]


class BackendSplit(dict):
    """The phase buckets, plus everything you need in order to distrust them.

    ``sound`` and ``coverage`` are the guard. ``sound`` is True only when the phases are separable
    in principle (one LLVM kernel module, no interleaving) AND the buckets actually add up to the
    backend stage. Read them before reading the seconds.
    """

    @property
    def sound(self) -> bool:
        return bool(self.get("sound"))

    @property
    def coverage(self) -> float:
        """Fraction of the backend stage the buckets account for."""
        return self.get("coverage", 0.0)

    @property
    def top(self) -> tuple[str, float]:
        """``(phase, seconds)`` for the largest bucket."""
        cand = {k: self[k] for k in
                ("hlo_passes", "emitter", "llvm_opt", "codegen", "below_hlo") if k in self}
        if not cand:                                                          # pragma: no cover
            return ("<none>", 0.0)
        k = max(cand, key=lambda x: cand[x])
        return (k, cand[k])

    def __str__(self) -> str:
        b = self.get("backend", 0.0)
        rows = [f"backend {b:.3f} s   ({self['n_kernel_modules']} LLVM kernel module(s), "
                f"{self['n_pass_snapshots']} pass snapshots)   sound={self.sound}",
                f"{'phase':14s} {'seconds':>9s} {'share':>7s}"]
        rows.append("-" * len(rows[-1]))
        for k in ("hlo_passes", "emitter", "llvm_opt", "codegen", "below_hlo"):
            if k in self:
                v = self[k]
                rows.append(f"{k:14s} {v:9.3f} {100 * v / max(1e-9, b):6.1f}%")
        rows.append(f"{'(unaccounted)':14s} {self['unaccounted']:9.3f} "
                    f"{100 * self['unaccounted'] / max(1e-9, b):6.1f}%")
        rows.append(f"coverage {self.coverage:.3f}")
        for w in self.get("warnings", []):
            rows.append(f"  ! {w}")
        return "\n".join(rows)


def backend_split(fn, *args, dump_dir: str | None = None, **kwargs) -> BackendSplit:
    """Compile ``fn(*args)`` ONCE with dumping on and split the backend stage by artifact mtime.

    Returns ``hlo_passes / emitter / llvm_opt / codegen`` in seconds when the program has exactly
    one LLVM kernel module, and ``hlo_passes / below_hlo`` when it has more -- see the module
    docstring for why a finer split is not defined in that case. Always check ``.sound`` and
    ``.coverage`` first.

    Must run BEFORE the first compile in the process: XLA reads ``--xla_dump_to`` when its backend
    first initialised, and setting it later is a silent no-op that would leave this function
    is first initialised, and setting it later is a silent no-op that would leave this function
    so does this.

    Validated on jax#32704 (chained 2-D gather, CPU, N=500/nsamples=2e5), where the answer is
    independently known from an XLA kill-switch sweep. At ncycles=8: backend 12.634 s, of which
    ``hlo_passes`` 0.023 (0.2%), ``emitter`` 12.536 (99.2%), ``llvm_opt`` 0.028 (0.2%), ``codegen``
    0.016 (0.1%), coverage 0.998. The bulk lands on EMISSION, not on passes, which is the required
    outcome. It also SCALES with the pathology parameter, which a single point cannot show: sweeping
    ncycles 6/7/8/9 the emitter bucket reads 0.773 / 3.110 / 12.870 / 53.352 s (4.02x, 4.14x, 4.15x
    per added link, matching the 4x/link the issue reports for total compile time) while the other
    three buckets stay flat (hlo_passes 0.0176 -> 0.0213, llvm_opt 0.0238 -> 0.0303, codegen
    0.0167 -> 0.0177). The flattened control's emitter bucket is flat at 0.050-0.055 s across the
    same sweep, so the emitter ratio at ncycles=9 is 970x against ``hlo_passes`` at 1.09x.

    THE NUMBERS ARE PERTURBED BY THE MEASUREMENT. Writing per-pass snapshots costs wall time, and it
    lands inside the ``hlo_passes`` bucket. Measured: gather ncycles=8 backend 10.78 s undumped vs
    11.40 s dumped (+5.8%); ndtri d4 5.43 s vs 7.40 s (+36%, 729 files). ``backend`` here is the
    DUMPED compile's own number, so the shares are internally consistent -- but do not compare these
    absolute seconds against an undumped :func:`scopex.record`.

    THERE IS A FLOOR. The un-spanned head and tail (module setup before the first snapshot, buffer
    assignment, thunk emission) are a fixed ~20-25 ms regardless of program, so a compile under
    a second cannot pass the coverage band at all. Census of 9 varied small CPU programs: 0 of 9
    sound, coverage 0.20-0.89. That is the honest reading, not a bug -- this instrument is for
    compiles whose seconds you are trying to explain.

    ``scopex.pass_timings`` is an independent second witness for the ``hlo_passes`` bucket alone (it
    costs one more subprocess compile): measured agreement within 1.1-1.5x on three programs --
    gather ncycles=8 0.0186 s vs 0.012-0.023 s here, ndtri d4 2.983 s vs 3.270 s. It is blind to
    everything after the pass pipeline, which is precisely the 99.2% on the validation case.
    """
    from . import artifacts, flags, monitor

    if flags.backend_initialized():
        raise RuntimeError(
            "XLA's backend is already initialised, so --xla_dump_to would be ignored SILENTLY and "
            "this split would be computed from an empty directory. Call backend_split before the "
            "first compile in the process, or use a fresh one.")

    with flags.dump(path=dump_dir, passes=".*", fusion=False, keep=True) as d:
        t0 = time.perf_counter()
        t = monitor.record(fn, *args, **kwargs)
        wall = time.perf_counter() - t0

    backend = t.get("backend", 0.0)
    steps = artifacts.pass_growth(d)
    warns: list[str] = []

    # ── kernel-module census, and the overlap test that decides whether a split exists ───────────
    kern: dict[tuple[str, str], dict[str, float]] = {}
    for f in os.listdir(d):
        k = _kernel_key(f)
        if k:
            kern.setdefault(k[0], {})[k[1]] = (pathlib.Path(d) / f).stat().st_mtime
    n_kern = len(kern)

    interleaved = False
    if n_kern > 1:
        no = [v[_PHASES[0]] for v in kern.values() if _PHASES[0] in v]
        wo = [v[_PHASES[1]] for v in kern.values() if _PHASES[1] in v]
        ob = [v[_PHASES[2]] for v in kern.values() if _PHASES[2] in v]
        # Any kernel starting a later phase before every kernel finished the earlier one means the
        # phases are concurrent and global boundaries are meaningless. This test is cheap, decisive,
        # and NOT redundant with the coverage band -- ndtri's coverage was 0.919, comfortably inside
        # any reasonable band, while its phases interleaved 223/223.
        if (wo and no and min(wo) < max(no)) or (ob and wo and min(ob) < max(wo)):
            interleaved = True

    out = BackendSplit()
    out["backend"] = backend
    out["wall"] = wall
    out["dump_dir"] = d
    out["n_kernel_modules"] = n_kern
    out["n_pass_snapshots"] = len(steps)
    out["interleaved"] = interleaved

    if not steps:
        raise RuntimeError(
            f"no per-pass snapshots in {d}; the mtime split has no HLO-pass boundary to measure "
            f"from. This is what a dump that silently did not happen looks like, and an empty "
            f"result here would read as 'the backend did nothing'.")

    # ── hlo_passes: first snapshot mtime -> last snapshot mtime ──────────────────────────────────
    # Deliberately the SPAN, not the sum of per-gap deltas: XLA snapshots only passes that CHANGED
    # the module (23 snapshots against 95 timed passes on the gather case), so passes that changed
    # nothing fall inside the gaps either way and the span does not depend on how many there were.
    # The per-gap LABELS are wrong for the same reason, which is why this module deliberately does
    # not expose them -- use scopex.pass_timeline if you want the gap-by-gap view with that caveat.
    first, last = steps[0].mtime, steps[-1].mtime
    out["hlo_passes"] = last - first

    def latest(suffix):
        ts = [v[suffix] for v in kern.values() if suffix in v]
        return max(ts) if ts else None

    if n_kern == 0:
        # NO .ll AND NO .o WERE WRITTEN. Do not read this as "there was nothing below HLO".
        out["below_hlo"] = 0.0
        out["sound"] = False
        warns.append(
            "0 LLVM kernel modules: this compile wrote no .ll and no .o, so everything after the "
            "last HLO pass is UNMEASURABLE by mtime -- there is no artifact to measure to, and the "
            "0.0 below_hlo here is an absence of evidence, not a measured zero. On jax 0.10.2/CPU "
            "the usual cause is that XLA routed the module to a library kernel (a kCustom fusion "
            "with backend_config kind=__ynn_fusion) rather than emitting one: measured on "
            "jnp.tanh(x).sum(), which dumps 26 files and not one LLVM module. Check "
            "scopex.custom_calls() and the optimized module's fusion kinds.")
    elif n_kern == 1 and not interleaved:
        a, b, c = (latest(s) for s in _PHASES)
        out["emitter"] = (a - last) if a else 0.0
        out["llvm_opt"] = (b - a) if (a and b) else 0.0
        out["codegen"] = (c - b) if (b and c) else 0.0
        out["sound"] = True
    else:
        endall = max((max(v.values()) for v in kern.values()), default=last)
        out["below_hlo"] = endall - last
        out["sound"] = False
        warns.append(
            f"{n_kern} LLVM kernel modules, compiled concurrently (phase mtimes interleave: "
            f"{interleaved}). emitter / llvm_opt / codegen are NOT separable from mtimes here, so "
            f"they are reported as one 'below_hlo' bucket. Measured on jax#2609 ndtri jacrev d4: "
            f"the global-boundary split would say llvm_opt=0.309 s while per-kernel intervals "
            f"sum to 4.400 s, and per-kernel sums themselves exceed the backend stage they sit "
            f"inside. Use TF_CPP_VMODULE=cpu_compiler=3,jit_compiler=3 for a per-kernel bill.")

    acct = sum(out.get(k, 0.0) for k in
               ("hlo_passes", "emitter", "llvm_opt", "codegen", "below_hlo"))
    out["accounted"] = acct
    out["unaccounted"] = backend - acct
    out["coverage"] = acct / max(1e-9, backend)
    if not (_COVERAGE_BAND[0] <= out["coverage"] <= _COVERAGE_BAND[1]):
        # `sound` and `coverage` are not allowed to disagree. An earlier draft set sound=True on a
        # program whose buckets covered 21% of the backend stage -- the exact shape of every bug in
        # this package's history: a confident-looking answer that is mostly missing. Failing the
        # band REVOKES soundness; it does not merely annotate it.
        out["sound"] = False
        warns.append(
            f"coverage {out['coverage']:.3f}: the buckets account for {100 * out['coverage']:.0f}% "
            f"of the {backend:.3f} s backend stage, so {out['unaccounted']:.3f} s happened outside "
            f"every mtime boundary (module setup before the first snapshot, buffer assignment, "
            f"thunk emission, library-kernel dispatch -- none of which write a timestamped file). "
            f"sound=False. Note the floor: the un-spanned head and tail are ~20-25 ms whatever the "
            f"program, so compiles under ~1 s cannot pass this band at all.")
    out["warnings"] = warns
    for w in warns:
        warnings.warn(w, RuntimeWarning, stacklevel=2)
    return out
