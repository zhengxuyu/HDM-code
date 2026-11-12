"""The layering, as a test rather than a diagram.

A boundary that is only described rots the first time someone is in a hurry. These
read the import graph of `src/` and fail on the five ways it has actually gone
wrong on this project: the self-routing block reaching into the prompt stack for a
format constant, live code importing a compatibility shim instead of the module
behind it, a layer importing the layer above it, a cycle, and **a launcher naming
a module that is not there** -- which is the one that cost a day of cluster time,
because a launcher's import is checked by running it and running it means a job.
"""

import ast
import collections
import importlib.util
import pathlib
import re

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
# Only an actual import statement, so prose naming a module that moved -- which
# is how three of the comments below explain themselves -- is not a false
# failure. A backtick after `import` is enough to make it prose.
IMPORTS = re.compile(r"from\s+(litjev[\w.]*)\s+import|(?:^|[\"\s;(])import\s+(litjev[\w.]*)", re.M)

# Lower may not import higher. A module not named here is unconstrained.
LAYERS = {
    "litjev.formats": 0,
    "litjev.data.schema": 0,
    "litjev.readout.scoring": 0,
    "litjev.run.streaming": 0,
    "litjev.core.block_utils": 1,
    "litjev.core.prompts_utils": 1,
    "litjev.core.pipeline_utils": 1,
    "litjev.core.block": 2,
    "litjev.core.prompts": 2,
    "litjev.core.objective": 3,
    "litjev.core.pipeline": 4,
}

# Siblings that must stay independent of each other, in either direction. The
# layer numbers cannot express this: two modules at the same depth are unordered,
# but "change one module without touching the others" means the block must not
# know how a prompt is built and the prompts must not know what the block is.
INDEPENDENT = (
    ("litjev.core.block", "litjev.core.prompts"),
    ("litjev.core.block_utils", "litjev.core.prompts_utils"),
    ("litjev.core.block", "litjev.core.pipeline_utils"),
    ("litjev.core.prompts", "litjev.core.pipeline_utils"),
    ("litjev.core.objective", "litjev.core.prompts"),
    ("litjev.core.objective", "litjev.core.pipeline"),
)

# Re-exports of the pre-f82b516 module names. Nothing in src/ may use them: a
# shim that live code imports is not a shim, it is a layer.
#
# **Two things these are widely believed to be, and are not.** They do not make
# the old flat names resolve: `litjev.heads` has raised ModuleNotFoundError since
# f82b516 created these in the same commit that deleted `src/litjev/heads.py`,
# because nothing aliases `litjev.heads` to `litjev._legacy.heads`. And they are
# not only for launchers outside this repository: eighteen scripts in `scripts/`
# import them by their `_legacy` names, including `alf_rollout.py` and
# `browser_pmpo.py`, so they cannot be deleted by moving external callers.
SHIMS = ("litjev._legacy.heads", "litjev._legacy.routing", "litjev._legacy.prompting", "litjev._legacy.decision")


def module_name(path):
    return str(path.relative_to(SRC).with_suffix("")).replace("/", ".").removesuffix(".__init__")


def import_graph():
    """Module-level imports only.

    An import inside a function is a deliberate escape hatch -- it is how
    `provenance.check_imported_tree` asks where `litjev` was imported from without
    the package importing itself at load time, and how `cli.py` keeps startup off
    the critical path. These rules are about load-time coupling, so a deferred
    import is not an edge. They are returned separately so they stay visible.
    """
    graph = {}
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text())
        top, deferred = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for inner in ast.walk(node):
                    if isinstance(inner, ast.ImportFrom) and inner.module and inner.level == 0:
                        deferred.add(inner.module)
                    elif isinstance(inner, ast.Import):
                        deferred.update(a.name for a in inner.names)
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                top.add(node.module)
            elif isinstance(node, ast.Import):
                top.update(a.name for a in node.names)
        graph[module_name(path)] = (path, top, deferred)
    return graph


GRAPH = import_graph()


def test_formats_imports_nothing_from_litjev():
    """It is the module every layer may import, so it may import none of them.

    `core.block` used to take SLOT_FORMAT from `litjev.backends.slots`, which imports the
    prompt builders, so the self-routing block depended on the whole prompt stack
    for one version string.
    """
    _, deps, deferred = GRAPH["litjev.formats"]
    deps = deps | deferred
    assert not [d for d in deps if d.startswith("litjev")], \
        "litjev.formats must import nothing from litjev"


@pytest.mark.parametrize("module", sorted(LAYERS))
def test_no_module_imports_a_layer_above_it(module):
    own, (_path, deps, _deferred) = LAYERS[module], GRAPH[module]
    above = sorted(d for d in deps if LAYERS.get(d, -1) > own)
    assert not above, f"{module} (layer {own}) imports {above}, which sit above it"


@pytest.mark.parametrize("left,right", INDEPENDENT)
def test_independent_modules_do_not_import_each_other(left, right):
    for a, b in ((left, right), (right, left)):
        deps = GRAPH[a][1] | GRAPH[a][2]
        assert b not in deps, (
            f"{a} imports {b}; they are meant to be changeable one at a time. "
            "If they need to agree on something, that something is a format "
            "constant and belongs in litjev.formats."
        )


@pytest.mark.parametrize("name", sorted(GRAPH))
def test_nothing_in_src_imports_a_compatibility_shim(name):
    path, deps, deferred = GRAPH[name]
    deps = deps | deferred
    if name in SHIMS:
        return
    used = sorted(d for d in deps if d in SHIMS)
    assert not used, (
        f"{name} imports {used}; import the module behind the shim instead, "
        "so the shim can be deleted once the launchers outside this repo are moved"
    )


def launcher_imports():
    """Every `litjev` module an sbatch, shell or python launcher imports.

    Text, not AST: the sbatch and shell launchers carry their python inside
    heredocs and `python -c` strings, which is exactly where the broken import
    was, so an AST pass over `scripts/*.py` would have read past it.
    """
    found = {}
    for path in sorted(SCRIPTS.iterdir()):
        if path.suffix not in {".py", ".sh", ".sbatch"}:
            continue
        for from_form, import_form in IMPORTS.findall(path.read_text()):
            found.setdefault(from_form or import_form, set()).add(path.name)
    return found


LAUNCHER_IMPORTS = launcher_imports()


@pytest.mark.parametrize("module", sorted(LAUNCHER_IMPORTS))
def test_every_module_a_launcher_imports_exists(module):
    """The check that could not fail where it mattered, made able to fail here.

    `serve_decision.sbatch` imported `litjev.heads`, which f82b516 had moved, so
    under `set -e` the decision service died at its own pre-flight in seconds and
    the browser job's health check died with it. Two more callers of the same gone
    name were *verifying* things: `sync.sh` ran `from litjev.heads import ...` on
    every single launch and `build_htm_env.sbatch` listed it among the importables
    a built environment must have. Both passed throughout, because both resolved
    against the shared root's editable install, which still points at the old
    `src/` -- so the tree being launched had no such module while two checks of
    that tree reported success.

    This runs against the tree it is in. A module that moves takes every launcher
    naming it red here, in a second, instead of at the head of a queue.
    """
    assert importlib.util.find_spec(module) is not None, (
        f"{module} is imported by {', '.join(sorted(LAUNCHER_IMPORTS[module]))} "
        f"and does not exist in this tree"
    )


def test_the_launcher_scan_reaches_inside_the_sbatch_heredocs():
    """Or the test above is green over a set that excludes where the bug was.

    `litjev.cli` is only ever named inside the `uv run python -c` strings of the
    sbatch files; if the scan were AST-over-`.py` it would find none of them, and
    an empty parametrisation is a passing test.
    """
    assert "litjev.cli" in LAUNCHER_IMPORTS
    assert any(name.endswith(".sbatch") for name in LAUNCHER_IMPORTS["litjev.cli"])
    assert "litjev.core.objective" in LAUNCHER_IMPORTS
    assert LAUNCHER_IMPORTS["litjev.core.objective"] >= {"sync.sh"}


def test_no_script_opens_a_records_file_without_the_checkpoint_guard():
    """The coverage the first version of that guard did not have.

    `ea76b23` put the `partial` check in `load_records`, and nine scripts never
    reach it: each wants one file's columns rather than a merge, so each called
    `np.load` directly and read `metadata` itself. `free_baselines.py`, which
    produces the fitted-nothing table, and `table3.py` were both among them, so
    the guard protected one reader of ten while the two that feed published
    tables stayed open. exps found it by listing the call sites rather than by
    reading the commit.

    So this counts the call sites instead of trusting that they were all moved. A
    records file is opened through `open_records` or `load_records`; a bare
    `np.load` on one is what this forbids. The allowance is by file, not by
    pattern, because `depth_probe` and `settling_analysis` load arrays that are
    not records and have no metadata to check.
    """
    import re

    # Loads an .npz that is not a records file: no `metadata`, nothing to guard.
    NOT_RECORDS = {"depth_walk.py", "settling_analysis.py", "depth_probe.py",
                   "content_vs_coded.py", "position_bias.py"}
    offenders = []
    for path in sorted(SCRIPTS.glob("*.py")):
        if path.name in NOT_RECORDS:
            continue
        text = path.read_text()
        for line in re.findall(r"^.*np\.load\(.*$", text, re.M):
            if "open_records" in line or "load_records" in line:
                continue
            offenders.append(f"{path.name}: {line.strip()}")
    assert not offenders, (
        "these open a records file without the checkpoint guard; use "
        "`litjev.data.collect.open_records`:\n  " + "\n  ".join(offenders))


def test_the_shims_only_re_export():
    """A shim with logic in it is a second implementation waiting to drift."""
    for shim in SHIMS:
        path = GRAPH[shim][0]
        body = ast.parse(path.read_text()).body
        offenders = [type(n).__name__ for n in body
                     if not isinstance(n, (ast.Import, ast.ImportFrom, ast.Expr))]
        assert not offenders, f"{shim} defines {offenders}; it should only re-export"


def test_the_import_graph_has_no_cycles():
    edges = {name: {d for d in deps if d in GRAPH} for name, (_, deps, _d) in GRAPH.items()}
    colour = collections.defaultdict(int)
    stack = []

    def walk(node):
        colour[node] = 1
        stack.append(node)
        for nxt in sorted(edges[node]):
            if colour[nxt] == 1:
                raise AssertionError("cycle: " + " -> ".join(stack[stack.index(nxt):] + [nxt]))
            if colour[nxt] == 0:
                walk(nxt)
        stack.pop()
        colour[node] = 2

    for node in sorted(edges):
        if colour[node] == 0:
            walk(node)
