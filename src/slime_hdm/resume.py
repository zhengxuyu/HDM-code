"""Episode bookkeeping for a run that will be killed partway, because it will be.

The hang exps captured in 28832 is a greenlet blocked inside Playwright. It
cannot be unwound, only exited, so the guard that catches it ends the process and
a launcher re-invokes the script. That makes a restart the expected path rather
than an edge case, and it puts one requirement on every phase that runs episodes:
the work already done has to be on disk, and the restart has to know **which**
episodes it has -- not how many.

How many is the trap. code-reviewer found it in the first version of this: resume
at `len(rows)` after episode five was dropped and ten were attempted, and the
restart begins at nine. It re-runs episode nine, which is now in the corpus
twice, and it never retries five. Every index is written with its row, and a
restart is the set difference.

An attempt is marked **before** the episode runs, because the failure being
guarded against leaves nothing behind: a hang that takes the process would
otherwise be invisible, and a launcher loop would retry that index until the
job's time limit. The attempt number, rather than a count of rows, is what the
cap reads -- the marker and the row that follows it share a number, so they count
once.
"""

import json
import time
from pathlib import Path

import numpy as np

# The key of the row that records what settings a file was written under. It
# carries no episode, so every count and index here steps over it.
SIGNATURE = "signature"


def jsonable(value):
    """numpy arrays into lists, so a row can be written and read back."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"cannot serialise {type(value).__name__}")


def append_row(path, row):
    """One row, flushed, so a process taken mid-episode leaves it behind.

    A partial line left by the kill is **closed first**. Appending straight onto
    a fragment glues the next row to it, and both are then unreadable -- so one
    lost episode became a lost run, which is the opposite of what this file is
    for. `hde.streaming.Rows` has done this since 27862 and I rebuilt a worse
    version of it instead of reusing it; code-reviewer reproduced the
    consequence.
    """
    path = Path(path)
    partial = False
    if path.exists() and path.stat().st_size:
        with open(path, "rb") as reader:
            reader.seek(-1, 2)
            partial = reader.read(1) != b"\n"
    with open(path, "a") as handle:
        if partial:
            handle.write("\n")
        handle.write(json.dumps(row, default=jsonable) + "\n")
        handle.flush()


def read_rows(path):
    """Every row written so far, in the order they were written.

    **A line that will not parse is dropped, wherever it is.** The guard this
    file exists for kills the process at an arbitrary instant, and a corpus row
    carrying 25 steps of hidden state is about 700 KB -- far past the 8 KB
    buffer, so part of it reaches the file before the exit does. The next
    `append_row` closes that fragment with a newline, which means the fragment is
    no longer the **last** line: dropping only the last one would have raised on
    it from the second resume onwards, turning one lost episode into a lost run.
    This is `hde.run.streaming.completed`'s rule, and it is the right one for the
    same reason.

    Dropping loses nothing but that episode: the index it was playing has no
    usable row, so `resume_plan` asks for it again.
    """
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"{path}: line {number} does not parse ({len(line)} bytes) and is "
                  f"dropped -- the episode it held is one the kill that wrote it "
                  f"cost, and it will be played again", flush=True)
    return rows


def episode_rows(rows, key):
    """The rows carrying a finished episode, as distinct from the markers.

    `key` is the field that only a real episode has. The rest -- an attempt that
    started, an episode that was dropped or given up on, the signature -- exist
    for the arithmetic here and are not data.

    Presence and not truthiness: an episode whose step list came back empty is
    still an episode that was played, and reading it as a marker would leave its
    index pending for as many attempts as the cap allows. Nothing writes such a
    row today -- a result with no steps is recorded `no_steps` instead -- so this
    is the weaker assumption, not a fix for a live bug.
    """
    return [row for row in rows if key in row]


def indices_of(rows, key):
    """Which episode indices already have a usable row.

    A row with no index is from the pre-streaming `.npz`, where the only knowable
    thing is how many episodes it held; those are numbered by position, which
    reproduces exactly what the old count-based resume did.
    """
    done, position = set(), 0
    for row in episode_rows(rows, key):
        index = row.get("index")
        if index is None:
            index, position = position, position + 1
        done.add(int(index))
    return done


def attempts_by_index(rows, key):
    """How many times each episode index has been started.

    From the recorded `attempt` number, so the marker written when an attempt
    begins and the row written when it ends do not count as two.
    """
    counts, position = {}, 0
    for row in rows:
        index = row.get("index")
        if index is None:
            if key not in row:
                continue
            index, position = position, position + 1
        index = int(index)
        counts[index] = max(counts.get(index, 0), int(row.get("attempt", 0)) + 1)
    return counts


def resume_plan(rows, episodes, max_attempts, key):
    """(to run, given up) for one phase, from what its file already holds.

    An index is run until it yields a usable row or has been started
    `max_attempts` times, and is then given up on -- a task that takes the
    process every time is a fact about the environment, and without the cap a
    launcher loop spins on it for the length of the job.
    """
    done = indices_of(rows, key)
    attempts = attempts_by_index(rows, key)
    run, given_up = [], []
    for index in range(episodes):
        if index in done:
            continue
        if max_attempts and attempts.get(index, 0) >= max_attempts:
            given_up.append(index)
        else:
            run.append(index)
    return run, given_up


def checked_rows(path, signature):
    """The rows in `path`, after setting aside any written under other settings.

    One `--output` reused with a changed flag -- a different split, seed, budget
    or group, or a corpus the heads were refitted on -- would otherwise be
    resumed onto, and the run would answer a different question in half its cells
    with nothing downstream able to tell. The stale file is **moved aside** rather
    than deleted, which keeps the evidence, and rather than raised on, which would
    stop a launcher loop that is otherwise doing the right thing.

    A file with no signature row at all is also stale: it predates this check, so
    what it was written under is unknown, and unknown is the case this exists for.
    """
    rows = read_rows(path)
    if signature is None:
        return rows
    stored = next((row[SIGNATURE] for row in rows if SIGNATURE in row), None)
    if rows and stored != signature:
        spare = Path(f"{path}.stale-{int(time.time())}")
        Path(path).rename(spare)
        print(f"{path} was written under different settings, so it is not resumed "
              f"onto. Moved to {spare.name}; what differs: "
              f"{_differences(stored, signature)}", flush=True)
        rows = []
    if not rows:
        append_row(path, {SIGNATURE: signature})
    return [row for row in rows if SIGNATURE not in row]


def _differences(stored, signature):
    """Which settings differ, for the message. Not a diff of values -- names."""
    if not isinstance(stored, dict):
        return "no signature was recorded"
    keys = sorted(set(stored) | set(signature))
    changed = [key for key in keys if stored.get(key) != signature.get(key)]
    return ", ".join(changed) if changed else "nothing named, so the order changed"
