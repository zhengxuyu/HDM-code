"""The provenance record, exercised by running the script that writes it.

`sync.sh` was read and flagged green while it crashed on every call: a rename
left two references to the old variable and `set -u` killed it before any rsync,
so nothing could be submitted for as long as it stood. Reading it was not
enough, and neither was `bash -n`, which parses without executing.

So these run it. `SYNC_DRY_RUN=1` writes the record and exits before it needs a
cluster, which is the only part worth asserting on anyway: the record is what a
result's provenance is read from.
"""
import re
import subprocess

import pytest
from stubs import environment as stub_environment
from stubs import network_stubs as stubs_for

SYNC = "scripts/sync.sh"
LAUNCH = "scripts/launch.sh"
REVISION = "scripts/revision.sh"


def repo(tmp_path, root):
    """A checkout with the script in it, so the script runs against a real git tree."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "scripts").mkdir()
    # revision.sh too: sync.sh sources it for the name of the directory it pushes
    # into, and launch.sh sources the same file so the two cannot drift apart.
    for name in ("sync.sh", "revision.sh"):
        target = tmp_path / "scripts" / name
        target.write_bytes((root / "scripts" / name).read_bytes())
        target.chmod(0o755)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "kept.py").write_text("x = 1\n")
    for command in (["git", "add", "-A"],
                    ["git", "-c", "user.email=t@t", "-c", "user.name=t",
                     "commit", "-qm", "first"]):
        subprocess.run(command, cwd=tmp_path, check=True)
    return tmp_path


def record(tmp_path):
    """The record the script writes, asked for by name.

    Named rather than defaulted: the default is a temp path outside the tree, so
    that a dry run cannot leave a record where launch.sh would copy it beside a
    job as that job's provenance.
    """
    out = tmp_path / "written-here"
    result = subprocess.run(["bash", "scripts/sync.sh"], cwd=tmp_path,
                            env={"SYNC_DRY_RUN": "1", "PATH": "/usr/bin:/bin:/usr/local/bin",
                                 "HOME": str(tmp_path), "SYNC_REVISION_OUT": str(out)},
                            capture_output=True, text=True)
    assert result.returncode == 0, f"sync.sh failed: {result.stderr}"
    return out.read_text()


def test_a_clean_tree_is_recorded_clean(tmp_path, pytestconfig):
    written = record(repo(tmp_path, pytestconfig.rootpath))
    assert "tree clean" in written
    assert "uncommitted paths" not in written
    assert "untracked paths" not in written


def test_an_untracked_launcher_does_not_make_the_tree_dirty(tmp_path, pytestconfig):
    """The whole point of the change that broke this: a launcher is not a modification."""
    where = repo(tmp_path, pytestconfig.rootpath)
    (where / "scripts" / "run.sbatch").write_text("#!/bin/sh\necho hi\n")
    written = record(where)
    assert "tree clean" in written, "an untracked file must not be reported as a diff"
    assert "untracked paths" in written
    assert "scripts/run.sbatch" in written


def test_a_tracked_change_is_recorded_dirty_and_named(tmp_path, pytestconfig):
    where = repo(tmp_path, pytestconfig.rootpath)
    (where / "src" / "kept.py").write_text("x = 2\n")
    written = record(where)
    assert "dirty diff-sha256" in written
    assert "uncommitted paths" in written
    assert "src/kept.py" in written


def test_the_untracked_digest_changes_with_the_file(tmp_path, pytestconfig):
    """A launcher recorded by name alone could be edited between runs unnoticed."""
    where = repo(tmp_path, pytestconfig.rootpath)
    launcher = where / "scripts" / "run.sbatch"
    launcher.write_text("one\n")
    first = record(where)
    launcher.write_text("two\n")
    second = record(where)
    assert first != second, "the record must change when the launcher does"


def test_the_scratch_files_are_removed_however_the_script_ends(tmp_path, pytestconfig):
    """`set -e` exits without reaching any rm, which is most of how this ends.

    Observed through TMPDIR: the diff and the record are both mktemp'd there, so
    what is left in it afterwards is what the script failed to clean up. The
    record asked for by name is written elsewhere and is not scratch.
    """
    where = repo(tmp_path, pytestconfig.rootpath)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    result = subprocess.run(["bash", "scripts/sync.sh"], cwd=where,
                            env={"SYNC_DRY_RUN": "1", "PATH": "/usr/bin:/bin:/usr/local/bin",
                                 "HOME": str(where), "TMPDIR": str(scratch),
                                 "SYNC_REVISION_OUT": str(tmp_path / "written-here")},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    left = sorted(p.name for p in scratch.iterdir())
    assert left == [], f"left behind in TMPDIR: {left}"


def test_a_dry_run_leaves_nothing_in_the_tree(tmp_path, pytestconfig):
    """Its record must not be mistakable for a submitted job's provenance."""
    where = repo(tmp_path, pytestconfig.rootpath)
    result = subprocess.run(["bash", "scripts/sync.sh"], cwd=where,
                            env={"SYNC_DRY_RUN": "1", "PATH": "/usr/bin:/bin:/usr/local/bin",
                                 "HOME": str(where)},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not (where / "REVISION").exists(), "a dry run wrote into the tree"
    written = subprocess.run(["git", "status", "--porcelain"], cwd=where,
                             capture_output=True, text=True).stdout
    assert written == "", f"a dry run changed the tree: {written!r}"


@pytest.mark.skipif(not subprocess.run(["which", "shellcheck"], capture_output=True).stdout,
                    reason="shellcheck is not installed")
def test_shellcheck_is_clean_on_the_scripts_that_write_records(pytestconfig):
    """Not the fault that broke this file -- see the test below -- but the rest."""
    result = subprocess.run(["shellcheck", "-S", "warning", SYNC, LAUNCH, REVISION],
                            cwd=pytestconfig.rootpath, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout


# Environment, so unassigned here is correct. Anything else all-caps and
# unassigned is a rename that did not finish.
FROM_ENVIRONMENT = {"HOME", "PATH", "USER", "SHELL", "TMPDIR", "PWD", "LANG",
                    # Slurm sets these in the job environment, not the script.
                    "SLURM_JOB_ID", "SLURM_SUBMIT_DIR", "SLURM_ARRAY_TASK_ID"}
REFERENCE = re.compile(r"\$\{?([A-Z_][A-Z0-9_]*)\}?")
# An assignment can follow a `;`, a pipe or an opening brace, not just a newline:
# `DATASET=$1; SCRIPT=$2` binds two.
ASSIGNMENT = re.compile(r"(?:^|[;&|{(]\s*|\s)(?:export\s+|local\s+)?([A-Z_][A-Z0-9_]*)=", re.M)
# `:+` belongs here with `:-`, `:=` and `:?`: all four are legal under set -u on an
# unset name. The launchers pass optional flags as ${VAR:+--flag "$VAR"}, which is
# why that idiom survives an unset variable where a bare $VAR aborts the job.
SELF_DEFAULTING = re.compile(r"\$\{([A-Z_][A-Z0-9_]*):[-=?+]")


LAUNCHERS = ["scripts/collect.sbatch", "scripts/collect_sglang.sbatch",
             "scripts/table1_sglang.sbatch"]


@pytest.mark.parametrize("script", [SYNC, LAUNCH, REVISION] + LAUNCHERS)
def test_every_variable_these_scripts_read_is_one_they_set(pytestconfig, script):
    """The fault that broke this file, which shellcheck does not report.

    `DIRTY` survived a rename to `TRACKED` at two lines and `set -u` killed the
    script on every call. SC2154 stays quiet on it twice over: once because the
    name is all-caps, which it treats as possibly environment, and again because
    every use sat behind `[ -n "$DIRTY" ]`, which it reads as a deliberate check
    for unset. Verified against the broken revision -- it reports nothing, at
    any severity, with check-unassigned-uppercase on. So this is checked here.
    """
    text = (pytestconfig.rootpath / script).read_text()
    unassigned = (set(REFERENCE.findall(text)) - set(ASSIGNMENT.findall(text))
                  - set(SELF_DEFAULTING.findall(text)) - FROM_ENVIRONMENT)
    assert not unassigned, (f"{script} reads {sorted(unassigned)} without setting them; "
                            f"under set -u that is a crash on every call")


# Flags a CLI accepts that its launcher does not pass, with the reason. FIXED is
# "the launcher means to decide this": the cluster's hardware, the lockfile's and
# the manifest's pins, a flag the launcher's own choices make inapplicable. OPEN
# is "nobody has needed it from the cluster yet" -- honest rather than designed,
# and the first person bitten should move it into the argv instead of arguing.
# A flag in neither set fails, which is the point: four flags reached the CLI and
# not the cluster before this existed, and each was found at the head of a queue.
NOT_IN_ARGV = {
    "scripts/collect.sbatch": {
        "FIXED": {"--dtype", "--device-map", "--revision", "--dataset-revision",
                  "--phase", "--categories"},
        # Defaults to 5, so it is on, and unreachable means every failed run
        # leaves nothing. Three 27B shards died twice over at ~28 minutes with
        # no partial file, which at batch 10 is one checkpoint per 2.3 hours.
        "OPEN": {"--checkpoint-every"},
    },
    "scripts/collect_sglang.sbatch": {
        # --unified-prompt is a no-op here: the CLI ORs it with backend == "sglang".
        "FIXED": {"--dtype", "--device-map", "--revision", "--dataset-revision",
                  "--unified-prompt"},
        "OPEN": {"--fast-only", "--code-offset", "--checkpoint-every"},
    },
    "scripts/table1_sglang.sbatch": {
        # --normalisation is ignored under --readout coded, which this launcher
        # hardcodes; --offset is fixed because --phase is how a disjoint sample is
        # asked for, which datasets.py explains.
        "FIXED": {"--dtype", "--device-map", "--revision", "--dataset-revision",
                  "--normalisation", "--offset"},
        # --margin-taus is the parameter-free margin rule the paper's novelty is
        # measured against. It has never been reachable from here, so no column of
        # Table 1 carries it as a policy arm. Open rather than fixed, and the one
        # entry in this table that is a question for the paper rather than for a
        # launcher.
        "OPEN": {"--margin-taus"},
    },
}
ENTRY_POINT = {
    "scripts/collect.sbatch": "collect",
    "scripts/collect_sglang.sbatch": "collect",
    "scripts/table1_sglang.sbatch": "bench_routing",
}


@pytest.mark.parametrize("launcher", sorted(NOT_IN_ARGV))
def test_every_cli_flag_is_reachable_from_the_cluster(pytestconfig, launcher):
    """The failure these launchers' own comments warn about, made able to fail here.

    "Every flag the CLI grows has to be reachable from here or it does not exist
    on the cluster" -- and it had happened four times on one launcher alone by
    the time this was written, most recently `--batch-size`, added the day three
    27B collections died 530 MiB short on an 80 GiB card with no way to ask for a
    smaller batch. Two more were found on the other launchers by running this
    audit by hand: `--batch-size` there too, and `--rows`, whose absence is why a
    filesystem outage cost 3.4 GPU-hours of resumable work.

    A flag missing from an argv is only correct when the launcher means to fix it,
    so every exception is named and a new flag fails until someone decides which
    it is.
    """
    import ast

    source = (pytestconfig.rootpath / "src/litjev/cli.py").read_text()
    entry = ENTRY_POINT[launcher]
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == entry)
    defined = {
        argument.value
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument"
        for argument in node.args
        if isinstance(argument, ast.Constant)
        and isinstance(argument.value, str)
        and argument.value.startswith("--")
    }
    # Comments stripped first. These launchers explain themselves at length and
    # name flags while doing it, so scanning the whole file lets a comment stand
    # in for an argv entry -- which it did, silently, until a deletion test on
    # --budget failed to fail.
    argv = "\n".join(
        re.sub(r"#.*$", "", line)
        for line in (pytestconfig.rootpath / launcher).read_text().splitlines()
    )
    passed = set(re.findall(r"(--[a-z0-9-]+)", argv))
    allowed = NOT_IN_ARGV[launcher]["FIXED"] | NOT_IN_ARGV[launcher]["OPEN"]
    unreachable = sorted(defined - passed - allowed)
    assert not unreachable, (
        f"{entry} accepts {unreachable} and {launcher} never passes them, so they do not "
        f"exist on the cluster. Add them to its argv, or to NOT_IN_ARGV[{launcher!r}] "
        "under FIXED with the reason the launcher decides it, or OPEN if nobody needs it yet."
    )


def test_the_not_in_argv_table_has_no_stale_entries(pytestconfig):
    """An exception that names a flag the CLI no longer has is a comment, not a check."""
    import ast

    source = (pytestconfig.rootpath / "src/litjev/cli.py").read_text()
    stale = {}
    for launcher, groups in NOT_IN_ARGV.items():
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == ENTRY_POINT[launcher])
        defined = {
            argument.value
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument"
            for argument in node.args
            if isinstance(argument, ast.Constant)
            and isinstance(argument.value, str)
            and argument.value.startswith("--")
        }
        gone = sorted((groups["FIXED"] | groups["OPEN"]) - defined)
        if gone:
            stale[launcher] = gone
    assert not stale, f"NOT_IN_ARGV names flags these CLIs no longer accept: {stale}"


def push(where, ssh_cases=(), **extra):
    """Run the real push against a stubbed cluster, and return what it sent."""
    stubs, log = stubs_for(where, ssh_cases=ssh_cases)
    result = subprocess.run(["bash", "scripts/sync.sh"], cwd=where,
                            env=stub_environment(stubs, where, **extra),
                            capture_output=True, text=True)
    assert result.returncode == 0, f"sync.sh failed: {result.stderr}"
    return result, (log.read_text() if log.exists() else "")


ABSENT = [('*"test -f"*', "exit 1")]      # no code directory for this tree yet
PRESENT = [('*"test -f"*', "exit 0")]     # one is already there


def test_the_code_lands_in_a_directory_named_after_the_tree(tmp_path, pytestconfig):
    where = repo(tmp_path, pytestconfig.rootpath)
    result, reached = push(where, ssh_cases=ABSENT)
    identity = subprocess.run(
        ["bash", "-c", ". scripts/revision.sh; tree_id"], cwd=where,
        capture_output=True, text=True, check=True).stdout.strip()
    assert identity, "revision.sh produced no id"
    assert f"code/{identity}" in reached, reached
    assert f"code/{identity}" in result.stdout, result.stdout


def test_the_push_is_staged_and_renamed(tmp_path, pytestconfig):
    """A push that dies halfway must not leave something launch.sh would accept."""
    _, reached = push(repo(tmp_path, pytestconfig.rootpath), ssh_cases=ABSENT)
    staged = [line for line in reached.splitlines() if ".partial." in line]
    assert staged, reached
    assert any("mv -T" in line for line in reached.splitlines()), reached
    # REVISION is what launch.sh tests for, so it must be the last thing written
    # into the staging directory and must never be written straight to the final
    # name ahead of the rest.
    sent = [line for line in reached.splitlines() if line.startswith("rsync")]
    revision = [i for i, line in enumerate(sent) if line.rstrip().endswith("REVISION")]
    assert len(revision) == 1 and revision[0] == len(sent) - 1, sent


def test_a_tree_already_on_the_cluster_is_not_pushed_again(tmp_path, pytestconfig):
    """The id covers tracked and untracked files, not ignored ones, so rsyncing
    into a directory a job is reading could still change it underneath."""
    result, reached = push(repo(tmp_path, pytestconfig.rootpath), ssh_cases=PRESENT)
    assert "already on the cluster" in result.stdout, result.stdout
    assert "rsync" not in reached, reached


def test_the_shared_tree_is_left_alone_unless_it_is_asked_for(tmp_path, pytestconfig):
    """A running job imported hours ago; pushing over what it reads only breaks it."""
    where = repo(tmp_path, pytestconfig.rootpath)
    _, reached = push(where, ssh_cases=ABSENT)
    assert not [line for line in reached.splitlines()
                if line.startswith("rsync") and "/code/" not in line], reached

    _, asked = push(repo(tmp_path / "again", pytestconfig.rootpath),
                    ssh_cases=ABSENT, SYNC_SHARED="1")
    assert [line for line in asked.splitlines()
            if line.startswith("rsync") and "/code/" not in line], asked


def test_a_directory_left_by_a_dead_push_is_reported_here(tmp_path, pytestconfig):
    """`mv -T` fails with ENOTEMPTY against one, and discarding the staged tree
    would leave a state nothing recovers from: sync says nothing, and every launch
    says the tree was never pushed and to run sync."""
    where = repo(tmp_path, pytestconfig.rootpath)
    # A cluster where the final check fails, which is what a directory holding no
    # REVISION does. The earlier `test -f` is what decides to push at all, so the
    # two cases have to be told apart by which command they are in.
    stubs, log = stubs_for(where, ssh_cases=[
        ('*"test -f"*', "exit 1"),
        ('*"! -f"*', 'echo "no REVISION" >&2; exit 1')])
    result = subprocess.run(["bash", "scripts/sync.sh"], cwd=where,
                            env=stub_environment(stubs, where),
                            capture_output=True, text=True)
    assert result.returncode != 0, "a push that did not land reported success"
    sent = log.read_text()
    assert "REVISION" in sent and "died part way" in sent, sent
