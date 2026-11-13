"""Which MiniWoB tasks a click-only agent could possibly complete, decided before the run.

The gate's fast mode offers `click('bid')` and nothing else, so a task needing
typing, a dropdown or a drag is unwinnable for it whatever the model does. Reading
that split off the scores afterwards -- "the typing tasks came back at zero" --
would be circular: it is the same number twice. So the split is computed here,
from the pinned pages, committed, and read by the gate as given.

The rule, and it is deliberately narrow: a task is **beyond click** when its page
contains a text field (`<input>` that is not a checkbox, radio or button, or a
`<textarea>`) or a `<select>`. Nothing else counts.

What that rule gets wrong, stated because it decides how the result may be read:
it misses the tasks needing pointer geometry -- a drag, a drag-select, a click at a
computed point -- which `click(bid)` cannot express either, and leaves them in the
click group. Those are not detected because the signals for them are noisy: a
jquery-ui include fires on pages that never drag. So the error runs one way only --
the click group holds tasks that are also impossible -- which **depresses** that
group's success rather than inflating it. A contrast measured there survives the
contamination; an absence of one there does not prove the ceiling.

`pointer_markers` records those signals per task **without using them**, so the
size of the contamination is visible instead of guessed. It is 18, and it went 8
then 12 then 18 as two readers looked harder, so treat 18 as a lower bound as
well.

From `01b43b1` those 18 are their own group rather than part of `click`, and that
change is dated **after** 28816's first episodes rather than before: this script's
earlier rule said they stay in the click group. See the `regrouped` field in the
output for how the two readings have to be reported.

    python scripts/miniwob_action_split.py <miniwob-pages-dir> [-o out.json]

The pages are the pinned MiniWoB++ checkout that `scripts/build_browsergym.sbatch`
clones; the directory is the one `MINIWOB_URL` points at.
"""

import argparse
import json
import re
from pathlib import Path

# An <input> that is not one of the clickable kinds, or any <textarea>.
TEXT_FIELD = re.compile(
    r"<textarea\b|<input\b(?![^>]*type\s*=\s*[\"']?(?:checkbox|radio|button|submit|reset)\b)",
    re.I)
DROPDOWN = re.compile(r"<select\b", re.I)
# Recorded, never used: see the module docstring. Each is a different way a task
# can need the pointer rather than an element, and they were found one reader at a
# time -- the first pass caught only the library include.
POINTER_MARKERS = {
    "pointer_library": re.compile(r"<canvas\b|\.draggable\(|\.sortable\(|\.slider\(", re.I),
    "rich_text": re.compile(r"\bquill\b", re.I),
    "selection": re.compile(r"getSelection|createRange", re.I),
    "coordinates": re.compile(r"clientX|clientY|offsetX|offsetY|Math\.(?:atan2?|sqrt)", re.I),
    "mouse_handlers": re.compile(r"mousedown|mousemove|mouseup", re.I),
}


def classify(html):
    """What a page needs beyond clicking, by the narrow rule."""
    needs = []
    if TEXT_FIELD.search(html):
        needs.append("text")
    if DROPDOWN.search(html):
        needs.append("select")
    return needs


def split(pages, names):
    """Every task, classified. Missing pages are an error, not an empty class."""
    tasks, missing = {}, []
    for name in sorted(names):
        page = Path(pages) / f"{name}.html"
        if not page.exists():
            missing.append(name)
            continue
        html = page.read_text(errors="replace")
        tasks[name] = {
            "needs": classify(html),
            "pointer_markers": sorted(kind for kind, pat in POINTER_MARKERS.items()
                                      if pat.search(html)),
        }
    if missing:
        raise SystemExit(f"no page for {len(missing)} task(s), e.g. {missing[:3]}; "
                         f"is {pages} the MiniWoB++ html/miniwob directory?")
    return tasks


def default_output():
    """Beside the gate module, which is the only place a run can read it from."""
    from slime_htm.envs.gate import miniwob_split_path

    return miniwob_split_path()


def registered_names():
    import browsergym.miniwob  # noqa: F401  registers the tasks
    import gymnasium

    return [name.split(".", 1)[1] for name in gymnasium.registry
            if name.startswith("browsergym/miniwob.")]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pages", help="the MiniWoB++ html/miniwob directory")
    parser.add_argument("-o", "--output", default=str(default_output()),
                        help="defaults beside the gate, which is where a run reads "
                             "it from: experiments/ is not synced to the cluster and "
                             "is not in the image")
    args = parser.parse_args()

    names = registered_names()
    tasks = split(args.pages, names)
    click = sorted(name for name, row in tasks.items() if not row["needs"])
    beyond = sorted(name for name, row in tasks.items() if row["needs"])
    contaminated = [name for name in click if tasks[name]["pointer_markers"]]
    record = {
        "rule": "beyond click when the page has a text field or a <select>; "
                "pointer-only tasks are not detected and stay in the click group, "
                "which depresses its success rather than inflating it",
        "click_tasks_with_pointer_markers": contaminated,
        "miniwob_commit": "33c3b4ddef8c6eb67c57a29663d844b1eda7e614",
        "tasks": tasks,
        "groups": {"click": click, "beyond-click": beyond},
    }
    Path(args.output).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"{len(tasks)} tasks: {len(click)} click, {len(beyond)} beyond click")
    print(f"  of the click group, at least {len(contaminated)} need the pointer and "
          f"are impossible too, so {len(click) - len(contaminated)} are usable")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
