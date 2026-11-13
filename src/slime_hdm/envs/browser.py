"""The page as a decision: what a browser step offers, and how it is described.

Shared by the demo and by the WorkArena gate, in one place because the rule that
decides a candidate set is the rule that decides what the fast mode can answer.
Two copies of it would be two experiments measuring different things under one
name, which this repository has spent enough time on.

Nothing here imports torch or a model: a browser runs where its own dependencies
are, and `inference.client` is the whole of the decision side it needs. The clock
this page can have drawn into it is the demo's business and stays there.
"""

import re


def element_names(observation):
    """bid -> what the element says it is, read off the accessibility tree."""
    from browsergym.utils.obs import flatten_axtree_to_str

    names = {}
    try:
        tree = flatten_axtree_to_str(observation["axtree_object"])
    except Exception:  # noqa: BLE001  an axtree this cannot read means no names
        return names
    for line in tree.splitlines():
        line = line.strip()
        if not line.startswith("[") or "]" not in line:
            continue
        bid, _, rest = line[1:].partition("]")
        rest = rest.strip()
        if rest:
            names[bid] = rest[:90]
    return names


# BrowserGym gives this environment **twenty** actions, not the thirteen this
# comment used to claim. Thirteen was `bid`(11) + `chat`(1) + `noop`(1), added up
# rather than read: `gym.make` passes no `action_mapping`, so the default
# `HighLevelActionSet` is built, and it also carries `infeas`(1), `nav`(3) and
# `tab`(3). Read off `env.unwrapped.action_mapping.__self__.action_set`:
#
#   chat    send_msg_to_user
#   infeas  report_infeasible
#   bid     clear click dblclick drag_and_drop fill focus hover press scroll
#           select_option upload_file
#   nav     go_back go_forward goto
#   tab     new_tab tab_close tab_focus
#   always  noop
#
# `candidates` constructs two of them. The split that matters is not two-way but
# three, and it is **measured** in `browsergym-actions.json` rather than derived
# here -- an earlier version of this comment re-derived it and put `tab_focus`
# and `report_infeasible` on the wrong sides:
#
#   selectable_now          2   click scroll
#   selectable_in_principle 13   clear dblclick drag_and_drop focus go_back
#                                go_forward hover new_tab noop press
#                                select_option tab_close tab_focus
#   needs_content            5   fill goto report_infeasible send_msg_to_user
#                                upload_file
#
# The middle thirteen are the ones that matter for any proposal to widen the fast
# mode: they can be enumerated **with no change to what the fast mode is** -- no
# generation, still one index into a list -- so they are missing because nobody
# wrote the code, not because of a design constraint. Their absence is why
# `candidates` filters to the window at all ("what is on screen, since there is
# no scroll action"). The last five are a different question, in `text_actions`,
# and they do change the fast mode.

# Which actions cannot be chosen without producing text. Taken from the measured
# classification in `browsergym-actions.json` rather than restated, because the
# two sides of this set are what the routing metric counts and a hand-copied list
# would drift from the file that is cross-checked against upstream signatures.
#
# The distinction is **the parameter's source, not its count**: `scroll(dx, dy)`
# takes two arguments and needs no content because both are constants, while
# `goto(url)` takes one and cannot be enumerated from the page.
NEEDS_CONTENT = frozenset({"fill", "goto", "report_infeasible", "send_msg_to_user",
                           "upload_file"})


def action_name(text):
    """The action a written or offered line calls, or None if it calls nothing."""
    match = re.match(r"\s*([a-z_]+)\s*\(", text or "")
    return match.group(1) if match else None


def fast_expressible_types():
    """The action names `candidates()` can ever construct, derived not listed.

    **The user's call on 2026-10-06**: the metric's predicate is "the fast mode
    cannot express this", not "this needs generated text". The two differ, and
    `drag_and_drop` is the difference -- it needs no text (two bids, both on the
    page) and the fast mode still cannot offer it, because enumerating it is
    `|bids|^2`: `drag-cube` has 31 bids, so 961 candidates for 52 slots.

    Derived from the two things `candidates()` actually emits -- a `click` per
    kept element, plus whatever `extra` holds, which on every path the
    environment builds is `content_free_actions()`. Hand-listing `{"click",
    "scroll"}` would drift the first time an extra is added, and drifting *open*
    is the dangerous direction: the metric would read 0 while the fast mode had
    started taking actions it was supposed not to.
    """
    return frozenset({"click"} | {action_name(action)
                                  for action in content_free_actions(scrolled=True)})


def fast_expressible(text):
    """Could the fast mode have produced this action's *type* at all?

    False for the eight whitelisted types it cannot offer: `fill`,
    `select_option`, `send_msg_to_user`, `goto`, `go_back`, `hover`, `press`,
    `drag_and_drop`. This is the **type** question; whether the fast mode was
    offered this exact action at this step is the instance question, and the
    step rows answer that separately as `outside_fast_set`.

    `go_back()` is in the False set and is the one that could change cheaply: its
    arity is 0, so it is a constant like `scroll` and costs one slot. It is
    False here because `content_free_actions()` does not offer it, not because
    it could not be offered -- and this function reports what the fast mode can
    do, not what it could be made to do.
    """
    return action_name(text) in fast_expressible_types()


def needs_content(text):
    """Does taking this action require having produced text?

    This is the predicate the routing metric is built on: **an action needing
    content is one the fast mode cannot express**, no matter how the candidate
    set is built, because the fast mode generates nothing. So a step where such
    an action is taken is a step that had to be escalated, and one taken under
    the fast mode would mean the two modes' action sets had come apart.

    Unparseable text is **not** content-needing: it is not an action at all, and
    counting it here would put a parser failure into a routing statistic.
    """
    return action_name(text) in NEEDS_CONTENT


SCROLL_STEP = 400
CONTENT_FREE = (f"scroll(0, {SCROLL_STEP})", f"scroll(0, -{SCROLL_STEP})")


def content_free_actions(scrolled=False):
    """Actions belonging to no element that need nothing written.

    Scrolling up is useless before the page has been moved and costs one of the 52
    candidate slots, so it is offered only after a scroll. `drag_and_drop` is left
    out: it needs two element ids, so it is a cross product rather than a constant,
    and the tasks needing it need pointer geometry as well.
    """
    return CONTENT_FREE if scrolled else CONTENT_FREE[:1]


def browser_failure(problem):
    """Whether this exception came from the browser rather than from our code.

    Judged by the exception's own **module**, not by its message or by catching
    `Exception`: playwright raises `playwright._impl._errors.*` and browsergym
    wraps those. Catching everything here would swallow a bug in the candidate
    rule or the action parser and report it as an environment failure, which is
    what makes a broken run look like a flaky one -- and this repository has
    spent a day on the opposite mistake, a flaky environment looking like a
    result.

    The case this exists for: `TargetClosedError` raised **from** `env.step`
    rather than as an orphaned future after a kill. 28853's log showed the
    orphaned kind, which does not propagate; the raised kind escapes
    `except StepTimeout`, takes the process with exit 1 and no stack-file growth,
    and a launcher following the artefact rule then **stops a 48-hour run** over
    one dead browser. code-reviewer named it as the case to watch; it is cheaper
    to handle than to watch for.
    """
    module = type(problem).__module__ or ""
    return module.split(".")[0] in ("playwright", "browsergym")


class StepTimeout(Exception):
    """Something in the browser did not come back. Not an outcome the model made."""


def step_deadline(seconds, stack_file=None, hang_multiple=1.5):
    """Interrupt a browser call that does not return, as a context manager.

    **Two layers, because one is not enough and we have the stack to prove it.**

    `SIGALRM` fires at the next bytecode boundary, which is enough for a blocked
    IO call that has released the GIL. It is **not** enough for the hang exps
    captured in 28832: `browsergym.core.env.post_step` calls Playwright's
    `context.cookies()`, which blocks in the greenlet bridge
    (`playwright/_impl/_sync_base.py:113`), and it sat there for over 300 s against
    a 120 s deadline. A Python signal cannot unwind a greenlet waiting on
    Playwright's own loop. I had said the deadline was a net and not a fix; that
    stack is which case it misses.

    So the second layer is `faulthandler.dump_traceback_later(..., exit=True)`, a
    **C-level** timer thread that dumps every thread's stack and `_exit`s even when
    the main thread never returns to the interpreter -- code-reviewer verified it
    against a blocking `select`. Process exit is the one thing a blocked greenlet
    cannot prevent. It is armed later than SIGALRM (`hang_multiple`) so a
    recoverable hang is still recorded as `step_timeout` and the run continues;
    only an unrecoverable one takes the process, and then with a diagnosis rather
    than a silence.

    SIGALRM exists only in the main thread on Unix; elsewhere that layer is absent
    rather than an error, and the faulthandler layer still applies.
    """
    import contextlib
    import faulthandler
    import signal

    @contextlib.contextmanager
    def deadline():
        if seconds <= 0:
            yield
            return
        handle = None
        if stack_file is not None:
            handle = open(stack_file, "a")  # noqa: SIM115  closed in finally
            faulthandler.dump_traceback_later(seconds * hang_multiple, exit=True,
                                              file=handle)
        armed = False
        previous = None
        if hasattr(signal, "SIGALRM"):
            def ring(_signal, _frame):
                raise StepTimeout(f"the call did not return within {seconds}s")

            try:
                previous = signal.signal(signal.SIGALRM, ring)
                signal.setitimer(signal.ITIMER_REAL, seconds)
                armed = True
            except ValueError:  # not the main thread
                armed = False
        try:
            yield
        finally:
            if armed:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous)
            if handle is not None:
                faulthandler.cancel_dump_traceback_later()
                handle.close()

    return deadline()


def process_tree():
    """pid -> (parent pid, command), read from `/proc`. Empty where there is none."""
    from pathlib import Path

    tree = {}
    root = Path("/proc")
    for entry in root.iterdir() if root.is_dir() else []:
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            command = (entry / "comm").read_text().strip()
        except OSError:
            continue
        # "pid (comm) state ppid ...", and comm can contain spaces and parentheses,
        # so the parent is the second field after the *last* closing parenthesis.
        tail = stat.rsplit(")", 1)[-1].split()
        if len(tail) >= 2 and tail[1].isdigit():
            tree[int(entry.name)] = (int(tail[1]), command)
    return tree


def descendants(pid, tree=None):
    """Every process below `pid`, deepest last.

    Direct children are not enough: the tree is python -> the `node` driver ->
    chromium, so killing only our children killed the driver and **orphaned
    Chromium**, which kept its memory for the rest of the job.
    """
    tree = process_tree() if tree is None else tree
    children = {}
    for child, (parent, command) in tree.items():
        children.setdefault(parent, []).append((child, command))
    found, frontier = [], [pid]
    while frontier:
        current = frontier.pop()
        for child, command in children.get(current, []):
            found.append((child, command))
            frontier.append(child)
    return found


def kill_browser_children():
    """Kill the browser processes below this one, for when `close` will not return.

    After a step has hung, `env.close()` is a call to the same wedged browser, so
    waiting on it turns one dead episode into a dead run -- which is how 28816
    ended. Reads `/proc` rather than depending on psutil, and reports what it
    killed instead of raising: this runs while something is already wrong.

    Killing nothing is a possible and not an exceptional outcome -- no `/proc`, or
    a process the walk does not see -- which is why clearing browsergym's cached
    driver belongs to the *timeout* in `close_env` and not to a successful kill.
    """
    import os
    import signal as signals

    killed = []
    for pid, command in descendants(os.getpid()):
        if not any(name in command.lower() for name in ("chrom", "node", "playwright")):
            continue
        try:
            os.kill(pid, signals.SIGKILL)
            killed.append(f"{command}:{pid}")
        except OSError:
            continue
    return killed


def forget_global_playwright(stop_timeout=20.0, stack_file=None):
    """Stop browsergym's cached Playwright and drop it, so the next env starts fresh.

    One module global, started once by `_get_global_playwright` and shared by
    every env in the process, so a driver that has failed must not stay cached:
    the next `gym.make` would use a dead connection and raise an error the gate
    does not catch.

    **Dropping the reference is not enough, and that cost a RED.** The first
    version only set the global to None, so the old `sync_playwright`'s event loop
    stayed registered on the main thread and the next `.start()` raised "It looks
    like you are using Playwright Sync API inside the asyncio loop." With
    recycling on by default that crashed every run at episode 26, and the same
    call in `close_env` made the orphan recovery crash the episode after it.
    Reproduced directly: two `.start()` calls in one thread raise, and
    `stop()`-then-`start()` works -- including after the driver process has been
    killed, where `stop()` returns at once.

    The stop is under a deadline because it is called while something is already
    wrong, and a stop that hangs must not become the new way a run dies.
    """
    try:
        import browsergym.core as core

        cached = getattr(core, "_PLAYWRIGHT", None)
        if cached is not None and hasattr(cached, "stop"):
            try:
                # With the stack file: a `stop()` blocked in the greenlet bridge is
                # the likeliest *second* hang, and without it that one had no exit
                # either. code-reviewer caught both call sites.
                with step_deadline(stop_timeout, stack_file=stack_file):
                    cached.stop()
            except Exception:  # noqa: BLE001  a stop that fails must still drop it
                pass
        core._set_global_playwright(None)
    except Exception:  # noqa: BLE001  nothing here may raise during a recovery
        return False
    return True


def close_env(env, seconds=30.0, stack_file=None):
    """Close a browser, and kill it if it will not close. Returns a problem or None."""
    try:
        with step_deadline(seconds, stack_file=stack_file):
            env.close()
    except StepTimeout:
        killed = kill_browser_children()
        # Cleared on the timeout, not on a successful kill. My first version tied
        # it to the kill, reasoning that a slow close might leave a healthy
        # driver -- but a close that *timed out* is not a slow close that
        # finished, it is a driver that failed to close. And the case the first
        # version missed is the worse one: a timeout that kills nothing leaves the
        # cache pointing at that driver, and then every step of the next episode
        # times out. code-reviewer inverted it; the cost is one fresh driver start
        # after something has already gone wrong.
        dropped = forget_global_playwright(stack_file=stack_file)
        return (f"close did not return within {seconds}s; killed "
                f"{len(killed)} browser process(es): {', '.join(killed) or 'none found'}"
                f"; cached driver {'dropped' if dropped else 'could not be dropped'}")
    except Exception as problem:  # noqa: BLE001  closing must not end a run
        return f"close raised {type(problem).__name__}: {problem}"
    return None


def elements(observation):
    """bid -> (role, name), from the same flattened tree `element_names` reads.

    The role is the first word of a tree line -- `[42] textbox 'Search'` -- and it
    is what says whether an element can be typed into rather than clicked. The
    click-only candidate set never needed it.
    """
    found = {}
    for bid, rest in element_names(observation).items():
        role, _, name = rest.partition(" ")
        found[bid] = (role.lower(), name.strip().strip("'\"").strip())
    return found


# Roles that take typed text, and roles that take one of a fixed set of options.
# Disjoint on purpose. `combobox` is both in ARIA -- an autocomplete field is one
# and a `<select>` flattens to one -- and the role alone cannot tell them apart.
# It counts as selectable because that is what WebArena's validators check: a
# `select.selectedIndex` against an `exact_match` is a real `<select>`, and a
# `fill` on it does nothing. A searchable combobox therefore loses its typing,
# which is the error that costs less.
TYPABLE = ("textbox", "searchbox", "spinbutton")
SELECTABLE = ("combobox", "listbox", "menu")
# The quoted or otherwise marked-out parts of a goal, which is where a WebArena
# task puts the value to type: "search for 'jaw bruxism'", "set the routing engine
# to Bicycle". Nothing here is generated -- a span is a substring of the goal.
_QUOTED = re.compile(r"[\"'\u201c\u2018]([^\"'\u201d\u2019]{2,60})[\"'\u201d\u2019]")
_AFTER_TO = re.compile(r"\b(?:to|as|with|named|called|titled)\s+([A-Za-z0-9][\w .@/-]{1,40})")
_TITLED = re.compile(r"\b([A-Z][\w-]+(?: [A-Z][\w-]+){0,3})\b")
_NUMBER = re.compile(r"\b(\d[\d.,:/-]{0,20})\b")


def goal_spans(goal, limit=4):
    """The literal values a goal names, most explicit first.

    This is the whole of how a token-free mode can type: the content comes out of
    the goal it was given, so choosing an action is still choosing an index in an
    enumerated set and nothing is generated. What it cannot do is supply a value
    the goal does not contain -- a task whose answer must be read off the page and
    sent back is out of reach whatever this returns, which is 31% of WebArena and
    the reason the action space is a decision rather than a patch.
    """
    spans, seen = [], set()
    for pattern in (_QUOTED, _AFTER_TO, _TITLED, _NUMBER):
        for match in pattern.finditer(goal or ""):
            span = match.group(1)
            if pattern is _TITLED and match.start(1) == 0:
                # A goal's first word is capitalised because it starts a sentence,
                # not because it is a value. Dropping the *match* dropped the value
                # with it: "Fork ChatGPT." matches as one phrase from position 0,
                # so skipping it returned nothing at all. Strip the verb instead.
                _, _, span = span.partition(" ")
            span = span.strip().strip(".,;:")
            key = span.lower()
            if span and key not in seen:
                seen.add(key)
                spans.append(span)
            if len(spans) >= limit:
                return spans
    return spans


def text_actions(observation, goal, *, spans=4, limit=12):
    """Enumerated actions that are not clicks: typing, choosing, answering.

    **Declined for the fast mode on 2026-10-05 and unused.** Kept because the
    decision rests on what it measures: with it the fast mode reaches 90 of
    WebArena's 694 scoreable tasks by typing values out of their goals, against
    50 that need nothing typed, and per site the best group goes from 32 tasks to
    51 (`scripts/webarena_span_coverage.py`). The user declined it so that the
    line between the cheap and the expensive mode is exactly "does this step need
    content produced", which is a cleaner thing to report than two action spaces
    chosen to be nearly equal. Nothing calls this; the gate asserts it does not.

    Returns `(offered, dropped)`, the same shape as `candidates`, because this cap
    cuts actions too and a cut nobody counts is a loss nobody can read. The first
    version returned a bare list and said the `dropped` count measured the loss --
    it did not: `candidates` reports only what *it* cuts, and three typable fields
    with four spans already fill all twelve slots here.

    The answer actions get their slots reserved rather than appended. They were
    last, so on any form page they were the first thing the cap removed, which
    left the one action class that can finish an answer-scored task unofferable
    exactly where a form made the page busy.

    Passed to `candidates` as `extra`. **Off unless asked for**: the gate's
    pre-registered runs are click-only, and a candidate set that changed under them
    would void them.
    """
    values = goal_spans(goal, limit=spans)
    if not values:
        return [], 0
    # The answer channel, which belongs to no element and is reserved first.
    answers = [(f"send_msg_to_user('{value}')",
                f"send_msg_to_user('{value}')  -- answer") for value in values]
    room = max(0, limit - len(answers))

    found = elements(observation)
    fields = []
    for bid, (role, name) in found.items():
        if role in TYPABLE:
            kind = "fill"
        elif role in SELECTABLE:
            kind = "select_option"
        else:
            continue
        for value in values:
            action = f"{kind}('{bid}', '{value}')"
            label = f"{action}  -- {role} {name}" if name else f"{action}  -- {role}"
            fields.append((action, label))
    return fields[:room] + answers, max(0, len(fields) - room)


def candidates(env, observation, limit, *, selector=None, extra=()):
    """The actions offered at this step, each with what it is on the page.

    BrowserGym marks every element it has given a browsing id and says which are
    clickable, but on a real site most of those are not on the page. An English
    Wikipedia article reports 3476 clickable elements; 232 are rendered, and ten
    are article links inside the window. The fast mode reads its answer from
    single-token codes and this tokenizer yields 52, so the filtering is what
    decides whether the page arrives intact or truncated to its chrome:

        rendered            has a box with area, so a person could hit it
        inside `selector`   the article, not the navigation around it
        in the window       what is on screen, since there is no scroll action

    Each action is labelled with the element's name. Without that the choice is
    between bare ids, and the page text that would explain them is cut by its
    own truncation: on the Kant article the model reasoned "options A-F click
    ids 1173-1178 not visible in provided page snippet... we don't have labels",
    and then guessed. The label is what the element already announces to a
    screen reader, so this gives away nothing a person would not see.

    `extra` are actions that belong to no element, which is how the agent can
    leave the site it is on. They are enumerated like everything else, so the
    fast mode still generates nothing to choose one.
    """
    properties = observation.get("extra_element_properties", {})
    names = element_names(observation)
    page = env.unwrapped.page
    inside = None
    if selector:
        try:
            inside = set(page.evaluate(
                "s => Array.from(document.querySelectorAll(s + ' [bid]'))"
                ".map(e => e.getAttribute('bid'))", selector))
        except Exception:  # noqa: BLE001  whatever the page raises, fall back
            inside = None
        # A selector that matches nothing is a page it was not written for, such
        # as the bookshop after Wikipedia. Fall back rather than offer no action.
        if not inside:
            inside = None
    # **The bboxes and the viewport are in different coordinate systems.**
    # BrowserGym reports element boxes scaled to match the *zoomed* screenshot --
    # `extract_dom_extra_properties(dom, scale_factor=...)` in `core/env.py` --
    # while `viewport_size` stays unzoomed. MiniWoB sets `_bgym_scale_factor =
    # 1.5` for its VLMs, so comparing a scaled `y` against an unscaled height
    # declared the bottom `1 - 1/1.5 = 33%` of every page to be below the fold.
    #
    # Measured on all 54 tasks of the click group, first step of each:
    #
    #     click candidates      205 -> 271   (+32%)
    #     of those, labelled     155 -> 207   (+34%)
    #     tasks with no click     12 -> 4
    #     tasks losing some       34 of 54
    #
    # `click-test` -- "click the button", the simplest task in the venue -- was
    # one of the twelve: its one button sits at a true `y` of 145 inside a
    # 214-high viewport, reported as 217.5, and the fast mode was offered
    # nothing but `scroll`. So the gate's `outside_fast_set` rate was measuring
    # this filter as much as it was measuring the policy.
    #
    # `getattr` with 1.0 is how browsergym reads the factor in all three of its
    # own call sites, so a venue that sets no zoom is unaffected.
    scale = getattr(page, "_bgym_scale_factor", 1.0)
    height = page.viewport_size["height"] * scale if page.viewport_size else None

    kept = []
    for bid, prop in properties.items():
        if not prop.get("clickable"):
            continue
        box = prop.get("bbox")
        if not box or box[2] <= 1 or box[3] <= 1:
            continue
        if inside is not None and bid not in inside:
            continue
        if height is not None and (box[1] >= height or box[1] + box[3] <= 0):
            continue
        label = names.get(bid)
        kept.append((f"click('{bid}')",
                     f"click('{bid}')  -- {label}" if label else f"click('{bid}')"))
    room = max(0, limit - len(extra))
    # Returned, not swallowed. The readout needs single-token codes and this
    # tokenizer yields 52, so a busy page is truncated -- and an episode that
    # failed because the right element was cut is not an episode the model got
    # wrong. `dropped` is recorded per step for exactly that reading.
    return kept[:room] + [(action, action) for action in extra], max(0, len(kept) - room)


# What a free-form slow mode is allowed to have written, and how many arguments
# each takes. A whitelist rather than "whatever parses": the slow mode is a
# language model writing code-shaped text, and an action we did not intend is a
# step we cannot interpret afterwards.
ACTION_ARITY = {
    "click": 1,
    "fill": 2,
    "select_option": 2,
    "send_msg_to_user": 1,
    "scroll": 2,
    "goto": 1,
    "go_back": 0,
    "hover": 1,
    "press": 2,
    # **Added 2026-10-06 on the user's call** that the action set is one and the
    # same for every policy with no extra construction. Leaving it out *was* an
    # extra construction: the environment accepts it, it needs no generated text
    # (two bids, both on the page), and the stated reason for excluding it --
    # "two element ids, a cross product rather than a constant" -- is a reason
    # the **fast** mode cannot enumerate it, not a reason the slow mode cannot
    # write it. The slow mode writes free text; it enumerates nothing.
    #
    # What it cost: on the five drag tasks the slow mode wrote
    # `send_msg_to_user('I cannot complete the drag because the triangles and
    # black box are not exposed as actionable elements and no drag action is
    # available.')` **48 times**, which is a correct diagnosis of our whitelist.
    # exps found 28853's unfinished episodes concentrating on the same tasks.
    "drag_and_drop": 2,
}

# How many *leading* arguments of each action name an element. **One entry per
# key in `ACTION_ARITY`, and `parse_action` subscripts it** -- code-reviewer's
# nit on 82e937e, and the right one: the first version was a hand list inside
# the function with `.get(name, 0)`, so a new bid-taking action added to
# `ACTION_ARITY` alone would have skipped the bid check silently. That **fails
# open**, on the one guard that stops a written action naming an element the
# page does not have. Subscripting makes the omission a `KeyError` at the first
# parse instead.
BID_ARGUMENTS = {
    "click": 1,
    "fill": 1,
    "select_option": 1,
    "send_msg_to_user": 0,
    "scroll": 0,
    "goto": 0,
    "go_back": 0,
    "hover": 1,
    "press": 1,
    # The only one with two, which is why the hand list was wrong to begin with.
    "drag_and_drop": 2,
}
# The last call on a line, allowing a label before it and a comment after: both
# "Final: click('42')" and "click('42')  # done" are how a model actually writes a
# conclusion, and refusing them ends an episode over phrasing.
_CALL = re.compile(r"([a-z_][\w]*)\s*\((.*)\)\s*(?:#.*)?$", re.S)
# scroll takes numbers; everything else takes strings.
NUMERIC = ("scroll",)
# Playwright's named keys, plus any single character, plus a modifier combination.
_KEY_NAMES = frozenset({
    "Enter", "Tab", "Escape", "Backspace", "Delete", "Insert", "Home", "End",
    "PageUp", "PageDown", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
    "Space", "Shift", "Control", "Alt", "Meta", "CapsLock", "NumLock",
    *(f"F{number}" for number in range(1, 13)),
    *(f"Digit{number}" for number in range(10)),
})
_MODIFIERS = frozenset({"Shift", "Control", "Alt", "Meta", "ControlOrMeta"})


def known_key(key):
    """Whether Playwright would recognise this as a key.

    `press('42', 'NotAKey')` parsed cleanly before, so a model inventing a key
    name spent a step on an action the browser cannot perform and the row said it
    pressed something.
    """
    if not key:
        return False
    *modifiers, last = key.split("+")
    if any(modifier not in _MODIFIERS for modifier in modifiers):
        return False
    return len(last) == 1 or last in _KEY_NAMES


def _literal(node):
    """The value of a literal argument node, or None if it is not one.

    One helper for both sides, because the same rule is needed twice -- once on
    what the model wrote and once on what we rendered -- and I wrote it twice:
    the second copy refused `scroll(0, -400)`, a constant this module itself
    offers, because a negative number is a `UnaryOp` over a `Constant`.

    Returns `(value, None)` or `(None, reason)`. bool is excluded before int, since
    it is a subclass: `fill('42', True)` would otherwise type the word True.
    """
    import ast

    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))
            and not isinstance(node.operand.value, bool)):
        return -node.operand.value, None
    if not isinstance(node, ast.Constant):
        return None, "an argument is not a literal"
    value = node.value
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None, f"{type(value).__name__} is not a value to type"
    return value, None


def _arguments(call_text, name):
    """The argument values of a written call, read by `ast` and not by a regex.

    Returns `(values, problem)`. Every argument has to be a literal: a model that
    wrote `fill('42', some_variable)` has written something this environment
    cannot interpret, and guessing would invent a value it never chose.
    """
    import ast

    try:
        tree = ast.parse(call_text, mode="eval")
    except SyntaxError as problem:
        return None, f"the written action does not parse: {problem.msg}"
    call = tree.body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        return None, "what was written is not a single call"
    if call.func.id != name or call.keywords:
        return None, "the call's name or keywords are not what was written"
    values = []
    for argument in call.args:
        value, problem = _literal(argument)
        if problem is not None:
            return None, problem
        values.append(value if isinstance(value, str) else str(value))
    return values, None


def _verified(rendered, name, count):
    """The rendered call, checked as a syntax tree rather than trusted as a string.

    This is the guard that matters, because `browsergym.core.action.base` runs the
    action with `exec` -- its own docstring says "WARNING: this is not safe!" -- so
    a string that re-renders into two statements runs two statements. Rendering
    with `repr` closes the hole; parsing the result is what proves it closed,
    rather than a comment claiming it did.
    """
    import ast

    try:
        tree = ast.parse(rendered, mode="eval")
    except SyntaxError as problem:
        return f"the action does not parse: {problem.msg}"
    call = tree.body
    if not isinstance(call, ast.Call):
        return "the action is not a single call"
    if not isinstance(call.func, ast.Name) or call.func.id != name:
        return "the action calls something other than its own name"
    if call.keywords or len(call.args) != count:
        return "the action's arguments changed when it was rendered"
    for argument in call.args:
        value, problem = _literal(argument)
        if problem is not None:
            return problem
        if name in NUMERIC:
            if not isinstance(value, (int, float)):
                return f"{name} takes numbers"
        elif not isinstance(value, str):
            return f"{name} takes strings"
    return None


def parse_action(text, allowed_bids=None):
    """The action a free-form slow mode wrote, or a reason it is not one.

    Returns `(action, problem)`: exactly one is None. **Nothing falls back to a
    click.** An unparsable or out-of-range action is a fact about that step --
    the model wrote something we cannot execute -- and turning it into the
    nearest clickable element would record a step the model did not choose and
    attribute its outcome to thinking.

    This is the one place in the project where an answer is parsed out of
    generated text, and it is here because the alternative is worse: reading the
    slow mode's answer off the enumerated codes means escalating cannot buy a
    typed action, so a step needing text is unwinnable at every price. The
    guardrails are the whitelist above, the arity, and the bid having to be one
    the page actually offered.
    """
    if not text:
        return None, "nothing was written"
    # The last fenced or bare call in the text: a reasoning trace ends with its
    # conclusion, and an action named mid-thought is usually a rejected option.
    # Backticks and markdown emphasis come off: a fenced block or a bolded
    # conclusion is phrasing, and an episode should not end over it. Prose after
    # the call is left refused, because "I will click('42') now, or maybe not" is
    # genuinely ambiguous.
    cleaned = text.strip().strip("`*").strip()
    lines = [line.strip().strip("`*").strip() for line in cleaned.splitlines()]
    for line in reversed([line for line in lines if line]):
        # search, not match: "Final: click('42')" and "Action: fill(...)" are how a
        # model writes a conclusion, and refusing them ended an episode over
        # phrasing rather than over a decision.
        match = _CALL.search(line)
        if not match:
            continue
        name, inside = match.group(1), match.group(2).strip()
        if name not in ACTION_ARITY:
            return None, f"{name!r} is not an action this environment takes"
        # Python's own parser reads the arguments, rather than a regex over
        # quotes. Two regexes tried it and both were wrong: `findall` returned ''
        # for a group that did not participate, so every double-quoted argument
        # was empty; then `finditer` read the first `""` of `"""x"""` as an empty
        # string and dropped the value. Parsing is not executing -- `exec` is the
        # unsafe part and comes later, on the string we render ourselves -- so
        # there is no reason to approximate it.
        arguments, problem = _arguments(f"{name}({inside})", name)
        if problem is not None:
            return None, problem
        if len(arguments) != ACTION_ARITY[name]:
            return None, (f"{name} takes {ACTION_ARITY[name]} argument(s), "
                          f"{len(arguments)} written")
        # Subscripted, not `.get(..., 0)`: an action in `ACTION_ARITY` with no
        # entry here must fail loudly rather than skip the bid check.
        for bid in arguments[:BID_ARGUMENTS[name]]:
            if allowed_bids is not None and bid not in allowed_bids:
                return None, f"{bid!r} is not an element on this page"
        if name == "press" and not known_key(arguments[1]):
            return None, f"{arguments[1]!r} is not a key Playwright knows"
        if name in NUMERIC:
            try:
                values = [int(argument) for argument in arguments]
            except ValueError:
                return None, f"{name} takes numbers, got {arguments}"
            rendered = f"{name}({', '.join(str(value) for value in values)})"
        else:
            # repr, never an f-string with quotes around it: browsergym runs the
            # action with `exec`, so `fill("42", "x'); click('13")` rendered as
            # `fill('42', 'x'); click('13')` executed two actions, and a payload
            # ending `__import__('os').system(...)` executed anything. The page
            # text a model reads is not ours -- reddit, gitlab -- so this was
            # reachable by prompt injection and not only by a confused model.
            rendered = f"{name}({', '.join(repr(argument) for argument in arguments)})"
        problem = _verified(rendered, name, len(arguments))
        if problem is not None:
            return None, problem
        return rendered, None
    return None, "no action call was written"


def state_text(observation, goal, trail, history=6, chars=4000):
    """The goal, what has been tried, and the page as an accessibility tree."""
    from browsergym.utils.obs import flatten_axtree_to_str

    lines = [f"Goal: {goal}", ""]
    if trail:
        lines += ["Recent actions:", *trail[-history:], ""]
    try:
        tree = flatten_axtree_to_str(observation["axtree_object"])
    except Exception:  # noqa: BLE001  a prompt without the page beats no prompt
        tree = ""
    lines += ["Page:", tree[:chars]]
    return "\n".join(lines)
