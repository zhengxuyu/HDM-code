"""Resuming a run that died, which is the only state these files are read in."""
import json

import pytest

from hde.run.streaming import Rows, completed, write_json_atomic


def test_nothing_written_yet_resumes_from_the_start(tmp_path):
    assert completed(tmp_path / "absent.jsonl") == {}


def test_a_line_the_job_died_inside_is_dropped(tmp_path):
    path = tmp_path / "partial.jsonl"
    path.write_text(json.dumps({"id": "a", "v": 1}) + "\n" + '{"id": "b", "v":')
    done = completed(path)
    assert set(done) == {"a"}, "a half-written row must not count as done"


def test_rows_are_readable_before_the_run_ends(tmp_path):
    path = tmp_path / "rows.jsonl"
    with Rows(path) as rows:
        rows.append({"id": "a"})
        assert set(completed(path)) == {"a"}, "a row must be on disk before the next one"
        rows.append({"id": "b"})
    assert set(completed(path)) == {"a", "b"}


def test_resuming_appends_rather_than_truncating(tmp_path):
    path = tmp_path / "rows.jsonl"
    with Rows(path, resume=True) as rows:
        rows.append({"id": "a"})
    with Rows(path, resume=True) as rows:
        rows.append({"id": "b"})
    assert set(completed(path)) == {"a", "b"}


def test_resume_is_by_identity_not_by_count(tmp_path):
    """A skipped question would otherwise shift every later question's gold answer."""
    path = tmp_path / "rows.jsonl"
    with Rows(path) as rows:
        rows.append({"id": "q3"})
        rows.append({"id": "q7"})
    assert set(completed(path)) == {"q3", "q7"}


def test_appending_to_an_existing_file_without_resuming_is_refused(tmp_path):
    """Two runs writing one path interleaved, and nothing downstream could separate them."""
    import pytest

    path = tmp_path / "rows.jsonl"
    with Rows(path, config={"temperature": 0.1}) as rows:
        rows.append({"id": "a"})
    with pytest.raises(FileExistsError, match="without --resume"):
        Rows(path, config={"temperature": 0.7})


def test_resuming_into_a_different_run_is_refused(tmp_path):
    import pytest

    path = tmp_path / "rows.jsonl"
    with Rows(path, config={"temperature": 0.1, "width": 10}) as rows:
        rows.append({"id": "a"})
    with pytest.raises(ValueError, match="different run"):
        Rows(path, config={"temperature": 0.7, "width": 10}, resume=True)


def test_resuming_into_the_same_run_is_allowed(tmp_path):
    path = tmp_path / "rows.jsonl"
    config = {"temperature": 0.1, "width": 10}
    with Rows(path, config=config) as rows:
        rows.append({"id": "a"})
    with Rows(path, config=config, resume=True) as rows:
        rows.append({"id": "b"})
    assert set(completed(path)) == {"a", "b"}


def test_a_row_appended_after_a_kill_is_not_glued_to_the_fragment(tmp_path):
    """Dropping a half-written line on read is not enough; the next write lands on it."""
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"header": {"t": 1}}) + "\n"
                    + json.dumps({"id": "a"}) + "\n" + '{"id": "b", "v":')
    with Rows(path, config={"t": 1}, resume=True) as rows:
        rows.append({"id": "c"})
    assert set(completed(path)) == {"a", "c"}, "the row after a kill must survive"


def test_the_header_is_not_mistaken_for_a_row(tmp_path):
    path = tmp_path / "rows.jsonl"
    with Rows(path, config={"t": 1}) as rows:
        rows.append({"id": "a"})
    assert set(completed(path)) == {"a"}


def test_a_result_file_holds_the_whole_payload_or_nothing(tmp_path):
    """The final name is written by a rename, so there is no partial state under it.

    A result file is read by the reporting scripts and by the paper. Truncated
    JSON under the final name either parses as far as it goes or fails in a way
    that says nothing about which run produced it.
    """
    path = tmp_path / "result.json"
    write_json_atomic(path, {"coverage": 0.42, "rows": [{"id": "q1"}]})
    assert json.loads(path.read_text())["coverage"] == 0.42
    assert path.read_text().endswith("\n")
    assert list(tmp_path.iterdir()) == [path], "the scratch file must not survive"


def test_a_write_that_fails_halfway_leaves_the_previous_result_intact(tmp_path):
    """json.dump writes as it walks, so a failure partway has already emitted text."""
    path = tmp_path / "result.json"
    write_json_atomic(path, {"run": "first"})
    with pytest.raises(TypeError):
        write_json_atomic(path, {"rows": [1, 2, 3], "head": object()})
    assert json.loads(path.read_text()) == {"run": "first"}, (
        "a failed write replaced a good result")


def test_the_scratch_file_is_beside_the_target(tmp_path):
    """Where the rename is atomic, and where a dead write is visible.

    `os.replace` is only atomic within one filesystem, so the scratch file cannot
    live in the system temp directory: a results path is often a mount that is
    not on. It is also left behind on a failure rather than cleaned up, so a
    directory of results says which run died while writing, under a name nothing
    reads as a result.
    """
    path = tmp_path / "result.json"
    write_json_atomic(path, {"run": "first"})
    with pytest.raises(TypeError):
        write_json_atomic(path, {"head": object()})
    left = sorted(p.name for p in tmp_path.iterdir())
    assert left == ["result.json", "result.json.partial"], left
