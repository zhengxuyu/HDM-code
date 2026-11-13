"""Which code a run actually ran, asked at run time rather than at submission.

`scripts/launch.sh` writes the record when a job is submitted and copies it into
the run directory. The job imports its code when it *starts*, which on a busy
partition is hours later, and every `scripts/sync.sh` in between rsyncs over the
one shared source tree on the cluster with `--delete`. A job that waited through
a sync runs code its record does not name, and nothing in its output says so.

The fix is for a sync to land in a directory named after the tree and for a job
to run in that directory, so that the code a record names cannot be replaced
under it. `scripts/sync.sh` and `scripts/launch.sh` do that.

What is left here is the half only the process itself can answer: which directory
`hde` was actually imported from. The shared environment holds an editable
install pointing at the old shared tree, so a job can sit in the right directory
and import the wrong code, which is the same failure wearing a disguise.
"""

import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path

UNKNOWN = "unknown"
# Written by scripts/sync.sh beside the pushed tree; its first line is
# `commit <sha>`. On the cluster this is the only statement of what the tree is,
# because rsync does not carry .git.
RECORD = "REVISION"


def commit_in(path):
    """The commit a `REVISION` file names, or None if it does not name one."""
    try:
        text = Path(path).read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("commit "):
            # `commit ` with nothing after it is a record that names none, not a
            # crash: a half-written REVISION must not take the process down.
            parts = line.split(None, 1)
            return parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
    return None


def tree_id_in(path):
    """The tree id a `REVISION` file names, or None if it does not name one.

    `scripts/sync.sh` writes `tree-id <commit>-d<diff sha>-u<untracked sha>`, and
    the commit alone is not an identity: 29636 ran from a dirty tree
    (`-d523cd8bf78ef4c07`), so two runs at one commit with different working
    changes would be indistinguishable by commit.
    """
    try:
        text = Path(path).read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("tree-id "):
            parts = line.split(None, 1)
            return parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
    return None


def code_identity(root=None):
    """What identifies the code this process runs from, for a corpus signature.

    `REVISION` first here, unlike `commit_now`: its tree id names the working
    changes as well as the commit, and the places runs actually happen have no
    `.git` at all. That is not a corner case -- `scripts/sync.sh` rsyncs, so the
    pushed tree is never a checkout, and a `git rev-parse HEAD` there fails and
    was recording `None` into every ALFWorld signature. A field meant to stop two
    corpora that mean different things from pooling was therefore blank in the
    one environment every such run happens in. exps read it off 29636's live
    corpus; my own code had been asserting it worked.

    Falls back to the commit, then to git, then to `UNKNOWN` with a line on
    stderr. A dirty local checkout is identified by its commit alone, which this
    cannot improve on from inside the process.
    """
    here = Path(root) if root else Path.cwd()
    found = tree_id_in(here / RECORD) or commit_in(here / RECORD)
    if found:
        return found
    try:
        found = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here,
                               capture_output=True, text=True,
                               check=True).stdout.strip()
        if found:
            return found
    except (OSError, subprocess.CalledProcessError):
        pass
    print(f"warning: neither {here / RECORD} nor git says which code this is "
          f"running, so the signature will record {UNKNOWN!r} and two corpora "
          f"collected by different code will pool", file=sys.stderr, flush=True)
    return UNKNOWN


def commit_now(root=None):
    """The commit of the tree this process is running from.

    git first, because in a checkout it is the tree itself speaking, and a
    `REVISION` that arrived by some other route can be stale. On the cluster
    there is no `.git`, so the file is what answers -- and it is the file a sync
    replaces, which is what makes a swapped tree visible at all.

    Returns `"unknown"` and says so on stderr when neither answers. A row labelled
    `unknown` cannot be traced to code, so it is worth a line in the log rather
    than a silent default.
    """
    here = Path(root) if root else Path.cwd()
    try:
        found = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True,
                               text=True, check=True).stdout.strip()
        if found:
            return found
    except (OSError, subprocess.CalledProcessError):
        pass
    from_record = commit_in(here / RECORD)
    if from_record:
        return from_record
    print(f"warning: neither git nor {here / RECORD} says which commit this is running, "
          f"so rows will be labelled {UNKNOWN!r} and cannot be traced to code",
          file=sys.stderr, flush=True)
    return UNKNOWN


def check_imported_tree(submit_dir=None, module=None, announce=False):
    """Refuse to run when the code being imported is not the code that was submitted.

    `scripts/launch.sh` submits out of `$REMOTE/code/<tree-id>`, a directory
    nothing writes over. That is only worth anything if the process imports from
    it. The shared environment holds an editable install of this package pointing
    at the old shared `src/`, which takes effect on any `import hde` that
    `PYTHONPATH` has not put the submit directory ahead of -- so the job would sit
    in the right directory and run the wrong code, which is the failure the
    directories were built to end, wearing a disguise.

    Returns the directory the package was imported from. Nothing to check against
    -- no `$SLURM_SUBMIT_DIR`, so not a Slurm job -- returns None.
    """
    where = submit_dir or os.environ.get("SLURM_SUBMIT_DIR")
    if not where:
        return None
    if module is None:
        import hde

        module = hde
    imported = Path(module.__file__).resolve().parent
    submitted = Path(where).resolve()
    if announce:
        print(f"hde imported from {imported}", file=sys.stderr, flush=True)
        print(f"environment {describe_environment()}", file=sys.stderr, flush=True)
    if submitted not in imported.parents and imported != submitted:
        raise SystemExit(
            f"this job was submitted from {submitted} and is importing hde from "
            f"{imported}. The tree it runs from is not the tree it was recorded "
            f"against, so its numbers would belong to code nothing here names. Set "
            f"PYTHONPATH to the submitted tree's src, which scripts/launch.sh does.")
    return imported


# The cluster has two of them -- a .venv on torch 2.11 and an hdm-env on 2.9.1 --
# and which one a job used was written down nowhere. A number produced under one
# is not the same number produced under the other, and a run that cannot name its
# environment cannot be repeated.
RECORDED = ("torch", "transformers", "sglang", "math-verify", "numpy")


def describe_environment(extra=()):
    """The interpreter and the versions that move a result, without importing them.

    `importlib.metadata` reads the installed distributions' metadata, so this
    costs nothing and works for a package that is absent or that would take
    thirty seconds to import. A package that is not installed is recorded as
    None, which is itself a fact about the environment.

    `extra` is for a venue whose own packages decide what a run even means --
    `browsergym-workarena` carries the task registry, and the same task name in
    two releases is two different tasks. Those belong in the run that uses them
    rather than in every header, so the caller names them.
    """
    facts = {"python": sys.executable}
    for name in (*RECORDED, *(n for n in extra if n not in RECORDED)):
        try:
            facts[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            facts[name] = None
    return facts
