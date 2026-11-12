"""What `launch.sh` refuses to submit, exercised by running it.

`sbatch --export` joins `VAR=value` pairs with commas and has no escape, so a
value holding one is cut at it and the rest are read as names of variables to
export from the environment. The script's comment said it refused that; the case
below the comment matched whitespace only, so `TEMPERATURES=0.1,0.3,0.6,1.0` was
accepted and the job swept one temperature of four without saying so.

These run the script. The refusal happens before the sync and before any ssh, and
`ssh` and `rsync` are stubbed here so that a test which does reach them records
the fact instead of reaching the cluster.
"""
import os
import subprocess

import pytest
from stubs import environment as stub_environment
from stubs import network_stubs

LAUNCH = "scripts/launch.sh"


def tree(tmp_path, root):
    """A checkout holding the script, and stubs for everything off the machine."""
    (tmp_path / "scripts").mkdir()
    for name in ("launch.sh", "sync.sh", "revision.sh"):
        target = tmp_path / "scripts" / name
        target.write_bytes((root / "scripts" / name).read_bytes())
        target.chmod(0o755)
    (tmp_path / "job.sbatch").write_text("#!/bin/sh\necho hi\n")

    stubs, log = network_stubs(tmp_path)
    return tmp_path, log, stubs


def run(tmp_path, root, *arguments, **environment):
    where, log, stubs = tree(tmp_path, root)
    result = subprocess.run(["bash", "scripts/launch.sh", *arguments], cwd=where,
                            env=stub_environment(stubs, where, **environment),
                            capture_output=True, text=True)
    return result, (log.read_text() if log.exists() else "")


@pytest.mark.parametrize("pair", ["TEMPERATURES=0.1,0.3,0.6,1.0", "A=1,2", "LIST=a,b"])
def test_a_comma_in_a_value_is_refused(tmp_path, pytestconfig, pair):
    """The sweep that ran one of four values. It must not submit."""
    result, reached = run(tmp_path, pytestconfig.rootpath,
                          "math-500", "job.sbatch", "sweep", pair)
    assert result.returncode == 2, f"submitted anyway: {result.stdout}{result.stderr}"
    assert pair in result.stderr, result.stderr
    assert "comma" in result.stderr
    assert reached == "", f"reached the network before refusing: {reached!r}"


def test_whitespace_in_a_value_is_refused(tmp_path, pytestconfig):
    result, reached = run(tmp_path, pytestconfig.rootpath,
                          "math-500", "job.sbatch", "spaced", "FILES=a.json b.json")
    assert result.returncode == 2
    assert "whitespace" in result.stderr
    assert reached == ""


def test_too_few_arguments_is_refused(tmp_path, pytestconfig):
    result, reached = run(tmp_path, pytestconfig.rootpath, "math-500", "job.sbatch")
    assert result.returncode == 2
    assert "usage" in result.stderr
    assert reached == ""


def test_a_pair_without_either_gets_through_the_guard(tmp_path, pytestconfig):
    """Otherwise the guard could refuse everything and the tests above would pass."""
    result, reached = run(tmp_path, pytestconfig.rootpath,
                          "math-500", "job.sbatch", "one", "TEMPERATURE=0.1",
                          SKIP_SYNC="1")
    assert "comma" not in result.stderr and "whitespace" not in result.stderr
    assert "TEMPERATURE=0.1" in reached, (
        f"the pair never reached sbatch: {reached!r} {result.stderr}")
    assert ",TEMPERATURE=0.1 job.sbatch" in reached, reached


def test_several_pairs_are_joined_with_commas_by_the_script(tmp_path, pytestconfig):
    """Which is why a comma inside one cannot be told apart from the separator."""
    _, reached = run(tmp_path, pytestconfig.rootpath, "math-500", "job.sbatch", "two",
                     "A=1", "B=2", SKIP_SYNC="1")
    assert ",A=1,B=2 job.sbatch" in reached, reached


def test_the_job_is_submitted_out_of_the_directory_named_after_the_tree(
        tmp_path, pytestconfig):
    """Not out of the shared tree every later sync rsyncs over with --delete."""
    _, reached = run(tmp_path, pytestconfig.rootpath, "math-500", "job.sbatch", "one",
                     SKIP_SYNC="1")
    submit = [line for line in reached.splitlines() if "sbatch --parsable" in line]
    assert len(submit) == 1, reached
    assert "cd ~/litjev-systemtwo/code/" in submit[0], submit[0]


def test_the_job_is_pointed_at_its_own_src_and_the_shared_environment(
        tmp_path, pytestconfig):
    """The editable install in the shared environment points at the shared tree,
    so a job in the right directory still imports the wrong code without this."""
    _, reached = run(tmp_path, pytestconfig.rootpath, "math-500", "job.sbatch", "one",
                     SKIP_SYNC="1")
    assert "PYTHONPATH=/remote/litjev/code/" in reached, reached
    assert "/src," in reached or "/src " in reached, reached
    assert "UV_PROJECT_ENVIRONMENT=/remote/litjev/.venv" in reached, reached
    # Without this only the two scripts that call the check by hand are covered.
    assert "LITJEV_REQUIRE_TREE=1" in reached, reached


def test_a_tree_that_was_never_pushed_is_refused(tmp_path, pytestconfig):
    """SKIP_SYNC=1 used to copy whatever REVISION sat at the shared root into the
    record, so a job could be recorded against a commit nobody had pushed."""
    where, log, _ = tree(tmp_path, pytestconfig.rootpath)
    # A cluster where `test -f` fails, which is what an absent code directory is.
    stubs, log = network_stubs(where, ssh_cases=[('*"test -f"*', "exit 1")])
    result = subprocess.run(
        ["bash", "scripts/launch.sh", "math-500", "job.sbatch", "one"], cwd=where,
        env=stub_environment(stubs, where, SKIP_SYNC="1"),
        capture_output=True, text=True)
    assert result.returncode == 3, f"{result.returncode}: {result.stdout}{result.stderr}"
    assert "never" in result.stderr and "pushed" in result.stderr, result.stderr
    assert "sbatch" not in log.read_text(), "submitted a tree that is not there"


@pytest.mark.skipif(not subprocess.run(["which", "shellcheck"], capture_output=True).stdout,
                    reason="shellcheck is not installed")
def test_shellcheck_is_clean(pytestconfig):
    result = subprocess.run(["shellcheck", "-S", "warning", LAUNCH],
                            cwd=pytestconfig.rootpath, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout


def test_every_variable_the_launcher_reads_is_one_it_sets(pytestconfig):
    """The fault that killed sync.sh, checked here too: same shell, same `set -u`."""
    from test_sync_record import ASSIGNMENT, FROM_ENVIRONMENT, REFERENCE, SELF_DEFAULTING

    text = (pytestconfig.rootpath / LAUNCH).read_text()
    unassigned = (set(REFERENCE.findall(text)) - set(ASSIGNMENT.findall(text))
                  - set(SELF_DEFAULTING.findall(text)) - FROM_ENVIRONMENT)
    assert not unassigned, f"{LAUNCH} reads {sorted(unassigned)} without setting them"


def test_the_stub_would_have_caught_a_submission(tmp_path, pytestconfig):
    """The refusal tests assert on an empty log, so an always-empty log would pass."""
    where, log, stubs = tree(tmp_path, pytestconfig.rootpath)
    subprocess.run(["ssh", "anything"], env={**os.environ, "PATH": f"{stubs}:/usr/bin:/bin"},
                   capture_output=True, check=True)
    assert log.exists() and "ssh anything" in log.read_text()
