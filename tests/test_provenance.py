"""Which code a run says it ran, and whether it is telling the truth.

Jobs used to import from one shared tree on the cluster that every sync rsynced
over with `--delete`, so a sync landing while a job ran swapped the code under it
and the job's record still named the commit it was submitted at. One run took
four such syncs. `scripts/sync.sh` now pushes into a directory named after the
tree and `scripts/launch.sh` submits out of it; what is left for the process
itself is to check it is importing from there.
"""
import pathlib
import subprocess
import sys
import types

import pytest

from hde.run.provenance import (UNKNOWN, check_imported_tree, code_identity,
                                   commit_in, commit_now, tree_id_in)


def record(tmp_path, text):
    (tmp_path / "REVISION").write_text(text)
    return tmp_path / "REVISION"


def test_the_commit_is_read_from_the_record(tmp_path):
    written = record(tmp_path, "commit abc123def\ntree-id abc123def\ntree clean\n")
    assert commit_in(written) == "abc123def"


def test_a_record_that_names_no_commit_is_not_guessed_at(tmp_path):
    assert commit_in(record(tmp_path, "tree clean\n")) is None
    assert commit_in(record(tmp_path, "commit \n")) is None
    assert commit_in(tmp_path / "absent") is None


def test_a_checkout_answers_from_git(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "a").write_text("x")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "first"], cwd=tmp_path, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path,
                          capture_output=True, text=True, check=True).stdout.strip()
    # A stale record beside a real checkout must not win: the tree itself is the
    # authority where there is one, and a REVISION can arrive by any route.
    record(tmp_path, "commit 0000000000000000000000000000000000000000\n")
    assert commit_now(tmp_path) == head


def test_a_cluster_directory_answers_from_the_record(tmp_path):
    """rsync does not carry .git, so on the cluster the file is all there is."""
    written = record(tmp_path, "commit feedface\ntree clean\n")
    assert written.exists()
    assert commit_now(tmp_path) == "feedface"


def test_neither_one_says_so_out_loud(tmp_path, capsys):
    assert commit_now(tmp_path) == UNKNOWN
    assert "warning" in capsys.readouterr().err


def package(tmp_path, name="hde"):
    where = tmp_path / name
    where.mkdir(parents=True)
    (where / "__init__.py").write_text("")
    return types.SimpleNamespace(__file__=str(where / "__init__.py"))


def test_importing_from_the_submitted_tree_is_accepted(tmp_path):
    submitted = tmp_path / "code" / "abc123"
    module = package(submitted / "src")
    assert check_imported_tree(submitted, module) == (submitted / "src" / "hde")


def test_importing_from_anywhere_else_stops_the_job(tmp_path):
    """The editable install in the shared environment points at the shared tree."""
    submitted = tmp_path / "code" / "abc123"
    submitted.mkdir(parents=True)
    module = package(tmp_path / "shared" / "src")
    with pytest.raises(SystemExit) as raised:
        check_imported_tree(submitted, module)
    assert "shared" in str(raised.value) and "PYTHONPATH" in str(raised.value)


def test_outside_slurm_there_is_nothing_to_check(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_SUBMIT_DIR", raising=False)
    assert check_imported_tree() is None


def test_the_submit_directory_comes_from_slurm(monkeypatch, tmp_path):
    submitted = tmp_path / "code" / "abc123"
    monkeypatch.setenv("SLURM_SUBMIT_DIR", str(submitted))
    module = package(submitted / "src")
    assert check_imported_tree(module=module) == (submitted / "src" / "hde")


def test_the_environment_is_named_without_importing_it():
    """Two cluster environments carry different torch, and nothing recorded which."""
    from hde.run.provenance import RECORDED, describe_environment

    facts = describe_environment()
    assert facts["python"] == sys.executable
    assert set(facts) == {"python", *RECORDED}
    assert facts["torch"], "torch is installed here and must be named"
    # A package that is absent is a fact about the environment, not an error, and
    # asking for its version must not drag it into the process.
    assert "sglang" not in sys.modules
    assert facts["sglang"] is None or isinstance(facts["sglang"], str)


def test_the_tree_id_names_the_working_changes_and_not_only_the_commit(tmp_path):
    """A commit is not an identity for a tree a sync pushed.

    29636 ran from `42b7b4f472e4-d523cd8bf78ef4c07-u0d2142dc06aa8f0b`: the same
    commit with different uncommitted work is a different tree, and a signature
    that recorded only the commit would call the two alike.
    """
    written = record(tmp_path, "commit 42b7b4f472e44d9099e9ad8e1850dc6c184589c7\n"
                               "tree-id 42b7b4f472e4-d523cd8bf78ef4c07-u0d2142dc06aa8f0b\n"
                               "tree dirty diff-sha256 523cd8bf78ef4c07 untracked 4\n")
    assert tree_id_in(written) == "42b7b4f472e4-d523cd8bf78ef4c07-u0d2142dc06aa8f0b"
    assert code_identity(tmp_path) == tree_id_in(written)
    assert tree_id_in(record(tmp_path, "commit abc\n")) is None, "no tree-id line"
    assert tree_id_in(record(tmp_path, "tree-id \n")) is None


def test_code_identity_answers_where_runs_actually_happen(tmp_path, capsys):
    """The defect this exists for: the pushed tree is not a checkout.

    `hdm_rl_loop.py` called `git rev-parse HEAD` in the job's cwd, which
    `sync.sh` rsyncs, so there is no `.git` and the collection signature
    recorded `None` -- in every ALFWorld run there has ever been. A field whose
    whole purpose is to keep two corpora that mean different things from pooling
    was blank exactly where it was needed.
    """
    assert not (tmp_path / ".git").exists()
    assert code_identity(tmp_path) == UNKNOWN, "and loudly, not as None"
    assert "warning" in capsys.readouterr().err
    assert code_identity(tmp_path) is not None

    # With the record there, which is the real cluster case.
    record(tmp_path, "commit feedface\ntree-id feedface-dcafe-u0\n")
    assert code_identity(tmp_path) == "feedface-dcafe-u0"


def test_the_collection_signature_records_the_code_it_ran(tmp_path, monkeypatch):
    """And the launcher asks for it, rather than re-deriving it.

    This calls the launcher's own function from a directory that is not a
    checkout -- the shape every cluster run has -- so a return to `git
    rev-parse` here goes red instead of writing `None` into a 15-hour corpus.
    """
    import importlib.util

    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "hdm_rl_loop_under_test", root / "scripts" / "hdm_rl_loop.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    record(tmp_path, "commit feedface\ntree-id feedface-d99-u88\n")
    monkeypatch.chdir(tmp_path)
    assert module._code() == "feedface-d99-u88"

    (tmp_path / "REVISION").unlink()
    assert module._code() == UNKNOWN
