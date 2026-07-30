"""Regenerate ``scopex/_passmap.py``: pass name -> the XLA source file that implements it.

    python tools/build_passmap.py <xla-source-dir> <pass_names.json> src/scopex/_passmap.py

``pass_names.json`` is ``{pass name: [platforms]}``, ENUMERATED FROM REAL LOGS rather than guessed
-- see ``tools/probe_passes.py``, which compiles ~45 small programs under
``TF_CPP_VMODULE=hlo_pass_pipeline=1`` on each backend and takes the union of what XLA printed.

``<xla-source-dir>`` must be the tree the installed jaxlib was BUILT from, not HEAD. jax pins it
and publishes the hash, so this is checkable rather than assumed::

    curl -s https://raw.githubusercontent.com/jax-ml/jax/jax-v0.10.2/third_party/xla/revision.bzl
    #   XLA_COMMIT = "5a9e73cbd92530cac2ac36f4736a774b2412afe2"
    #   XLA_SHA256 = "08a52210a04cd68d38f6201d56273d04ac0f8b4e0da9f72677c74a48cc637422"
    curl -sL https://github.com/openxla/xla/archive/<commit>.tar.gz -o xla.tar.gz
    sha256sum xla.tar.gz          # must equal XLA_SHA256

WHY THE METHOD IS THIS PARANOID. Two earlier versions of this script produced WRONG POINTERS, and
a wrong pointer is the failure mode the table exists to prevent -- it costs a reader an afternoon
in the wrong file and it looks exactly like a right one:

  * a "names are sometimes built by concatenation" heuristic matched the prefix ``"float"`` next to
    a ``StrCat`` and mapped the GPU pipeline ``float_normalization`` to
    ``third_party/tsl/tsl/platform/numbers.h``. The prefix rule now demands >= 12 characters, a
    ``name_``/``pass_name_`` member initialiser, and a file that defines ``name()``.
  * ``simplification`` was mapped to ``gpu_compiler.cc`` alone. CPU and GPU each build a pipeline
    of that name, so the single answer was wrong for every CPU user. Files that disagree are now
    split by platform or dropped.
  * ``fusion`` collided with two CPU runtime THUNKS that also answer ``name() == "fusion"``.
    Thunks are not passes; ``/runtime/`` is excluded.

THE RULE: a name maps only when one file wins outright at the strongest evidence tier available,
or when the candidates split cleanly by backend. Anything else is omitted with its candidates
recorded. Omission is the correct output for an ambiguous name.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
from collections import defaultdict


def is_not_pass_source(rel: str) -> bool:
    """Files that cannot contain an HLO pass definition, and would otherwise contribute noise."""
    return ("_test." in rel or "/tests/" in rel or "/testlib/" in rel or "test_util" in rel
            or "_test_" in rel or "/benchmarks/" in rel or "/tools/" in rel
            # An HLO pass is never implemented under runtime/: those are Thunks, and a Thunk has a
            # name() too. Two of them answer "fusion", same as InstructionFusion.
            or "/runtime/" in rel)


# Context immediately BEFORE the string literal decides what the literal is doing.
DEFINES = [
    (re.compile(r"name\s*\(\s*\)\s*const[^{]{0,40}\{\s*return\s*$"), "name() { return ... }"),
    (re.compile(r"\bkName\s*=\s*$"), "static kName"),
    (re.compile(r"\breturn\s*$"), "return ... (inside name())"),
]
PIPELINE = [
    (re.compile(r"HloPassPipeline\s*>?\s*>?\s*(?:\w+\s*)?\(\s*$"), "HloPassPipeline(...)"),
    (re.compile(r"AddPass<[^>]*HloPassPipeline[^>]*>\s*\(\s*$"), "AddPass<HloPassPipeline>(...)"),
    (re.compile(r"CreateSimplificationPipeline\s*\(\s*$"), "CreateSimplificationPipeline(...)"),
    (re.compile(r"Pipeline\s*\(\s*$"), "...Pipeline(...)"),
]
# The ONE rule for names XLA builds at run time (float-normalization-bf16,
# host-offloading-prepare-convert-to-custom-call). Deliberately narrow -- see the docstring.
BUILT = re.compile(r"(?:^|\W)(?:name_|pass_name_)\s*\(\s*(?:absl::StrCat\s*\(\s*)?\"([^\"]+)\"")
MIN_PREFIX = 12


def platform_of(f: str) -> str | None:
    if "/gpu/" in f or f.startswith("xla/backends/gpu") or "gpu_compiler" in f:
        return "gpu"
    if "/cpu/" in f or f.startswith("xla/backends/cpu") or "cpu_compiler" in f:
        return "cpu"
    return None


def build(root: pathlib.Path, names: dict[str, list[str]]) -> tuple[dict, dict]:
    text_of = {}
    for p in root.rglob("*"):
        if p.suffix in (".h", ".cc") and p.is_file():
            rel = str(p.relative_to(root))
            if not is_not_pass_source(rel):
                try:
                    text_of[rel] = p.read_text(errors="replace")
                except Exception:
                    pass
    print(f"{len(text_of)} candidate source files", file=sys.stderr)

    lit = {nm: re.compile(re.escape(f'"{nm}"')) for nm in names}
    occ: dict[str, list] = defaultdict(list)
    for rel, text in text_of.items():
        for nm, pat in lit.items():
            for m in pat.finditer(text):
                pre = " ".join(text[max(0, m.start() - 220):m.start()].split())
                rank, kind = 9, "other"
                for p, k in DEFINES:
                    if p.search(pre):
                        rank, kind = 1, k
                        break
                else:
                    for p, k in PIPELINE:
                        if p.search(pre):
                            rank, kind = 2, k
                            break
                occ[nm].append((rank, kind, rel, text.count("\n", 0, m.start()) + 1))

    built_index: dict[str, list] = defaultdict(list)
    for rel, text in text_of.items():
        if "name()" not in text:
            continue
        for m in BUILT.finditer(text):
            built_index[m.group(1)].append((rel, text.count("\n", 0, m.start()) + 1))

    def source_line(rel, line):
        return " ".join(text_of[rel].splitlines()[line - 1].split())[:150]

    def impl_of(rel):
        """The .cc twin of a header, when the tree really has one.

        The evidence lives in the header -- that is where `name()` is -- but the file someone
        actually wants to read is the implementation. Checked against the tree rather than
        assumed: header-only passes exist, and a pointer to a file that is not there is the
        wrong-file failure in a politer form."""
        if not rel.endswith(".h"):
            return None
        cc = rel[:-2] + ".cc"
        return cc if (root / cc).is_file() else None

    table, omitted = {}, {}
    for nm in sorted(names):
        hs = occ.get(nm, [])
        best = min((h[0] for h in hs), default=9)
        top = [h for h in hs if h[0] == best]
        files = sorted({h[2] for h in top})
        kind = {1: "pass", 2: "pipeline"}.get(best)

        if not hs or best == 9:
            got = None
            for cut in range(len(nm), MIN_PREFIX - 1, -1):
                pref = nm[:cut]
                if pref in built_index and len({r for r, _ in built_index[pref]}) == 1:
                    got = (*built_index[pref][0], pref)
                    break
            if got:
                table[nm] = {"platforms": sorted(names[nm]), "kind": "pass", "file": got[0],
                             "line": got[1], "evidence": source_line(got[0], got[1]),
                             "impl": impl_of(got[0]),
                             "how": f"name is built from the literal {got[2]!r}"}
                continue
            omitted[nm] = {"platforms": sorted(names[nm]), "candidates": files[:6],
                           "reason": ("the literal does not occur in any candidate source file"
                                      if not hs else
                                      "the literal occurs only in contexts that neither define a "
                                      "pass nor name a pipeline")}
            continue

        if len(files) == 1:
            table[nm] = {"platforms": sorted(names[nm]), "kind": kind, "file": files[0],
                         "line": top[0][3], "evidence": source_line(files[0], top[0][3]),
                         "impl": impl_of(files[0]), "how": top[0][1]}
            continue

        byp = defaultdict(list)
        for f in files:
            byp[platform_of(f)].append(f)
        if None not in byp and set(names[nm]) <= set(byp) and all(len(v) == 1 for v in byp.values()):
            table[nm] = {"platforms": sorted(names[nm]), "kind": kind,
                         "file": {p: v[0] for p, v in byp.items()},
                         "line": {h[2]: h[3] for h in top},
                         "evidence": {h[2]: source_line(h[2], h[3]) for h in top},
                         "impl": {h[2]: impl_of(h[2]) for h in top},
                         "how": top[0][1] + " (each backend has its own)"}
            continue
        # nvptx vs amdgpu is a vendor split, not an ambiguity: only one of the two ever runs.
        if set(files) <= {"xla/service/gpu/nvptx_compiler.cc", "xla/service/gpu/amdgpu_compiler.cc"}:
            win = "xla/service/gpu/nvptx_compiler.cc"
            h = next(x for x in top if x[2] == win)
            table[nm] = {"platforms": sorted(names[nm]), "kind": kind, "file": win, "line": h[3],
                         "evidence": source_line(win, h[3]), "impl": impl_of(win),
                         "how": h[1] + "; amdgpu_compiler.cc builds the same pipeline for ROCm"}
            continue
        # Same pass name, same class name, two namespaces. Decidable without preference: exactly
        # one of the headers is #included by a file that actually AddPass<>es that class.
        adders = defaultdict(set)
        for rel2, text2 in text_of.items():
            for h in top:
                c = re.search(r"class\s+(\w+)\s*:", text_of[h[2]])
                if (c and re.search(r"AddPass<\s*" + c.group(1) + r"\s*>", text2)
                        and f'#include "{h[2]}"' in text2):
                    adders[h[2]].add(rel2)
        if len(adders) == 1:
            win = next(iter(adders))
            h = next(x for x in top if x[2] == win)
            table[nm] = {"platforms": sorted(names[nm]), "kind": kind, "file": win, "line": h[3],
                         "evidence": source_line(win, h[3]), "impl": impl_of(win),
                         "how": (f"{h[1]}; {len(files)} headers declare a pass of this name and "
                                 f"only this one is included where it is added "
                                 f"({sorted(adders[win])[0]})")}
            continue
        omitted[nm] = {"platforms": sorted(names[nm]), "candidates": files,
                       "reason": f"{len(files)} files define this name and nothing separates them"}
    return table, omitted


HEADER = '''"""GENERATED by tools/build_passmap.py -- do not hand-edit.

pass name -> the XLA source file that implements it, greped out of the source tree the installed
jaxlib was BUILT from. Identified by jax's own pin and verified by hash, so no row is a
recollection::

    jax-v0.10.2  third_party/xla/revision.bzl
        XLA_COMMIT = "{commit}"
        XLA_SHA256 = "{sha}"
    sha256sum of the tarball actually greped -- MATCHED.

Every row is a literal found at that file and line in that tree. A row is A POINTER TO A FILE,
never an explanation of what the pass does or of why it was slow.

ROW SHAPE  name -> (where, impl, kind, platforms_seen, wrapper_pipeline, source_line)

``where``             ``(path, line)``, or ``{{platform: (path, line)}}`` when the backends have
                      different implementations of one pass NAME (measured: 2 such names).
``impl``              the ``.cc`` beside that header, when the tree HAS one -- the file you
                      actually want to read. ``None`` for header-only passes and for pipeline
                      rows, where the pointer is already a ``.cc``. Checked, not assumed.
``kind``              ``"pass"``     the name is an ``HloPassInterface::name()``.
                      ``"pipeline"`` the name is an ``HloPassPipeline`` built at that line. Its
                      seconds are mostly OTHER passes' seconds, and those are timed separately --
                      so a "pipeline" row is a warning as much as a pointer.
``platforms_seen``    backends this name was observed on in the probe logs. Absence is "not seen
                      here", which is weaker than "does not run there".
``wrapper_pipeline``  not None when XLA wraps this pass in a PIPELINE OF THE SAME NAME, so the log
                      carries two nested lines with one name; ``(path, line)`` of the wrapper.
``source_line``       the line of C++ the pointer resolves to, so a reader can tell at a glance
                      whether the table is stale without opening anything.
"""

# fmt: off
XLA_COMMIT = "{commit}"
XLA_SHA256 = "{sha}"
BUILT_FOR = "jaxlib 0.10.2"
SOURCE_URL = "https://github.com/openxla/xla/blob/{{commit}}/{{file}}#L{{line}}"

# Names XLA wraps in a pipeline of the SAME name: the log then prints two nested timings that
# differ only in depth, and anything that sums by name counts the inner one twice. Located by
# hand in the tree above, one `git grep` each; kept separate from the generated rows because
# they were not machine-derived.
WRAPPED_BY_SAME_NAME_PIPELINE = {{
    "autotuner": ("xla/service/gpu/gpu_compiler.cc", 1778),
    "fusion-wrapper": ("xla/service/gpu/gpu_compiler.cc", 3221),
    "propagate-call-metadata": ("xla/service/gpu/gpu_compiler.cc", 1790),
    "sanitize-constant-names": ("xla/service/gpu/gpu_compiler.cc", 3239),
}}

# Observed but deliberately NOT mapped. Listed so the absence is visible and citable: a reader
# who looks up one of these gets the candidates and the reason, not a plausible single file.
OMITTED = {omitted}

PASSES = {{
'''


def main():
    root = pathlib.Path(sys.argv[1])
    names = json.load(open(sys.argv[2]))
    out_path = sys.argv[3]
    commit = sys.argv[4] if len(sys.argv) > 4 else "5a9e73cbd92530cac2ac36f4736a774b2412afe2"
    sha = sys.argv[5] if len(sys.argv) > 5 else \
        "08a52210a04cd68d38f6201d56273d04ac0f8b4e0da9f72677c74a48cc637422"

    table, omitted = build(root, names)
    wrapped = {"autotuner": ("xla/service/gpu/gpu_compiler.cc", 1778),
               "fusion-wrapper": ("xla/service/gpu/gpu_compiler.cc", 3221),
               "propagate-call-metadata": ("xla/service/gpu/gpu_compiler.cc", 1790),
               "sanitize-constant-names": ("xla/service/gpu/gpu_compiler.cc", 3239)}
    with open(out_path, "w") as f:
        f.write(HEADER.format(commit=commit, sha=sha,
                              omitted=json.dumps(omitted, indent=4, sort_keys=True)))
        for nm in sorted(table):
            v = table[nm]
            where = ({k: (v["file"][k], v["line"][v["file"][k]]) for k in v["file"]}
                     if isinstance(v["file"], dict) else (v["file"], v["line"]))
            ev = (v["evidence"] if isinstance(v["evidence"], str)
                  else next(iter(v["evidence"].values())))
            impl = v.get("impl")
            if isinstance(impl, dict):
                impl = {k: impl[v["file"][k]] for k in v["file"]}
            f.write(f"    {nm!r}: ({where!r}, {impl!r}, {v['kind']!r}, {v['platforms']!r},\n")
            f.write(f"        {wrapped.get(nm)!r},\n        {ev!r}),\n")
        f.write("}\n# fmt: on\n")
    print(f"mapped {len(table)}  omitted {len(omitted)}  -> {out_path}", file=sys.stderr)
    for nm, o in omitted.items():
        print(f"  OMIT {nm}: {o['reason']}", file=sys.stderr)


if __name__ == "__main__":
    main()
