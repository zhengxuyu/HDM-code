"""Write a row as it is produced, and pick up where a dead run left off.

A run that writes its record once, at the end, loses everything if it dies at
hour twenty, and the slow mode's tokens are the expensive part of these runs.
Appending a row per question costs nothing and makes a death a pause.

The resume is by question id and not by count, because a run that skipped a
question -- an exception, a filtered row -- would otherwise resume at the wrong
place and silently score one question against another's gold answer.
"""

import json
import os
from pathlib import Path


def write_json_atomic(path, payload, indent=1):
    """Write a result file so that a killed job leaves no result file at all.

    `json.dump(payload, open(path, "w"))` is closed and flushed by CPython as the
    statement ends, so the risk is not an unflushed buffer. It is the window
    during the dump: a job killed inside it leaves a truncated file under the
    final name, and truncated JSON read by the reporting scripts is a result that
    parses as far as it goes or fails in a way that says nothing about why. The
    rename is atomic, so the name either holds the whole payload or nothing.

    The scratch file sits beside the target, because `os.replace` is atomic only
    within one filesystem and a results path is often a mount the system temp
    directory is not on. A failed or killed write leaves it there, under a name
    nothing reads as a result, which is how a run that died mid-write says so.
    """
    target = Path(path)
    scratch = target.with_name(target.name + ".partial")
    with open(scratch, "w") as handle:
        json.dump(payload, handle, indent=indent)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(scratch, target)
    return target


def completed(path):
    """The ids already written, from a partial file. Empty when there is none.

    A truncated final line is expected: the file was being appended to when the
    job was killed. It is dropped, and that question is done again.
    """
    rows = {}
    if not path or not Path(path).exists():
        return rows
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # the line the job died in the middle of
            if "header" in row:
                continue
            if "id" in row:
                rows[row["id"]] = row
    return rows


class Rows:
    """Append rows to a JSONL file, flushing each one.

    Flushing per row is the point: a buffered row is a row that a killed job
    never wrote.

    The file carries a header describing the run that opened it, and resuming
    into it is refused unless the header matches. Without that, pointing two
    runs at one path silently interleaves them -- the same question id appeared
    twice in one file, once from a run at temperature 0.1 and once at 0.7 -- and
    nothing downstream could tell which row belonged to which.
    """

    def __init__(self, path, config=None, resume=False):
        self.path = path
        self.handle = None
        if not path:
            return
        existing = Path(path)
        if existing.exists() and existing.stat().st_size:
            if not resume:
                raise FileExistsError(
                    f"{path} already holds rows. Appending to it without --resume would "
                    f"mix two runs in one file; pass --resume, or write somewhere else")
            self._check_header(existing, config)
        # A handle held open for the life of the run, by design: the point is to
        # append and flush per row. A `with` here would close it at the end of
        # __init__ and there would be nothing to append to.
        self.handle = open(path, "a")  # noqa: SIM115
        # A job killed mid-line leaves a partial one. Appending straight onto it
        # glues the next row to a fragment, and both are then unreadable.
        if existing.exists() and existing.stat().st_size:
            with open(path, "rb") as reader:
                reader.seek(-1, 2)
                if reader.read(1) != b"\n":
                    self.handle.write("\n")
                    self.handle.flush()
        elif config is not None:
            self.append({"header": config})

    @staticmethod
    def _check_header(path, config):
        """Resuming into a file written by a different run is refused, not merged."""
        if config is None:
            return
        with open(path) as handle:
            first = handle.readline().strip()
        try:
            header = json.loads(first).get("header")
        except json.JSONDecodeError:
            header = None
        if header is None:
            raise ValueError(
                f"{path} has no header, so what produced it cannot be checked against "
                f"this run. It was written before headers existed, or by something else")
        differing = {key: (header.get(key), value) for key, value in config.items()
                     if header.get(key) != value}
        if differing:
            raise ValueError(
                f"{path} was written by a different run and resuming would mix them. "
                f"Differing: {differing}")

    def append(self, row):
        if self.handle is None:
            return
        self.handle.write(json.dumps(row) + "\n")
        self.handle.flush()

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
