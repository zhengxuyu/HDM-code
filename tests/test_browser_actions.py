"""The enumerated non-click actions: what a token-free mode can type, and cannot.

Option (b) of the action-space decision in `experiments/browser-gate/README.md`.
It is implemented and **not enabled**: the gate's pre-registered runs are
click-only, and a candidate set that changed under them would void them.

The thing these tests are really about is the boundary. Nothing here generates:
every value offered is a substring of the goal the environment handed over. So a
task naming its value ("set the routing engine to Bicycle") is reachable, and a
task whose answer has to be read off the page and sent back is not -- which is
31% of WebArena and the reason the decision belongs to the user rather than to a
patch.
"""

import json

import pytest

from slime_htm.envs.browser import goal_spans, text_actions


def axtree(monkeypatch, lines):
    """Stub the flattened accessibility tree, which is what `elements` parses."""
    from slime_htm.envs import browser

    monkeypatch.setattr(browser, "element_names",
                        lambda observation: dict(lines))


def test_a_value_the_goal_names_is_offered_and_one_it_does_not_is_not():
    assert goal_spans('search for "jaw bruxism" in the forum') == ["jaw bruxism"]
    assert goal_spans("Set the routing engine to Bicycle") == ["Bicycle"]
    assert goal_spans("How many orders were placed in 2023?") == ["2023"]

    # The first word of a goal is capitalised because it starts a sentence, not
    # because it is a value.
    assert "Set" not in goal_spans("Set the routing engine to Bicycle")
    assert "How" not in goal_spans("How many orders were placed in 2023?")

    # Nothing is invented: every span is a substring of the goal.
    goal = "Create a new issue titled Bug in checkout"
    assert all(span in goal for span in goal_spans(goal))

    assert goal_spans("") == [] and goal_spans(None) == []


def test_typing_is_offered_for_a_field_and_choosing_for_a_dropdown(monkeypatch):
    """A `<select>` is not a click, which is a third of what the click-only agent
    could not do on WebArena -- `select.selectedIndex` checks were in its
    validators."""
    axtree(monkeypatch, {"11": "textbox 'Search'", "22": "combobox 'Engine'",
                         "33": "button 'Go'"})
    offered, dropped = text_actions({}, "Set the routing engine to Bicycle")
    actions = [action for action, _ in offered]
    assert dropped == 0, "three elements and one span fit"

    assert "fill('11', 'Bicycle')" in actions
    assert "select_option('22', 'Bicycle')" in actions
    assert not any(action.startswith("fill('33'") for action in actions), \
        "a button is not typable"
    assert all("--" in shown for _, shown in offered), "each carries what it is"


def test_the_answer_channel_is_offered_last(monkeypatch):
    """A step that answers ends the episode, and the page usually has to be read
    first, so it should not sit at the top of the list where a slot prior lands --
    but it keeps a reserved slot rather than being cut, which is a separate test."""
    axtree(monkeypatch, {"11": "textbox 'Search'"})
    offered, _ = text_actions({}, 'find the page about "jaw bruxism"')
    actions = [action for action, _ in offered]

    assert actions[-1] == "send_msg_to_user('jaw bruxism')"
    assert actions[0].startswith("fill(")


def test_the_cross_product_is_capped(monkeypatch):
    """Six fields and four spans is twenty-four actions, which would crowd out the
    clicks most steps actually need -- they share the 52 single-token codes."""
    axtree(monkeypatch, {str(bid): "textbox 'Field'" for bid in range(10)})
    goal = 'type "alpha" then "beta" then "gamma" then "delta" somewhere'

    assert len(goal_spans(goal)) == 4
    assert len(text_actions({}, goal)[0]) == 12, "the default cap"
    assert len(text_actions({}, goal, limit=5)[0]) == 5
    assert len(text_actions({}, goal, spans=1)[0]) <= 12


def test_no_value_in_the_goal_means_no_text_actions(monkeypatch):
    """The honest boundary: a task whose value must come from the page cannot be
    reached by substrings of its goal, and offering an empty `fill` would be
    worse than offering nothing."""
    axtree(monkeypatch, {"11": "textbox 'Search'"})
    assert text_actions({}, "click the thing") == ([], 0)
    assert text_actions({}, "") == ([], 0)


def uses(source, name):
    """Does this source **call** `name`, or import it -- as opposed to mentioning it?

    These two tests used `name not in source`, which cannot tell a call from a
    comment. A comment added in the `needs_content` work explained *why* the
    content-needing half stays off, named `text_actions` to do so, and turned both
    tests red. The substring check was asserting "does not call it" while
    enforcing "does not say it", which is the first defect shape in this
    repository: a name stating a property the check does not have.
    """
    import ast
    import textwrap

    # `inspect.getsource` of a nested function keeps its indentation, which
    # `ast.parse` rejects. Dedent rather than branch on it: the first draft here
    # had `source if ... else source`, two identical arms, which ruff caught.
    tree = ast.parse(textwrap.dedent(source), mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name) and target.id == name:
                return True
            if isinstance(target, ast.Attribute) and target.attr == name:
                return True
        if isinstance(node, ast.ImportFrom):
            if any(alias.name == name for alias in node.names):
                return True
        if isinstance(node, (ast.Name, ast.Attribute)) and not isinstance(
                getattr(node, "ctx", None), ast.Store):
            # a bare reference, e.g. passing the function as a value
            if getattr(node, "id", None) == name or getattr(node, "attr", None) == name:
                return True
    return False


def test_they_are_off_unless_asked_for(monkeypatch):
    """The gate's pre-registered runs are click-only. These arrive through
    `candidates(..., extra=...)`, which the gate does not pass, so no run changes
    under them until someone decides it should."""
    import inspect

    from slime_htm.envs import browser, gate

    assert inspect.signature(browser.candidates).parameters["extra"].default == ()
    source = inspect.getsource(gate.run_episode)
    # The gate now passes the *content-free* extras -- scroll, which needs nothing
    # written -- and still not the content-needing ones. Those are the two halves
    # of browsergym's action set and only the second changes what the fast mode is.
    assert "content_free_actions" in source, "the gate offers the content-free half"
    assert not uses(source, "text_actions"), "and not the half that needs content"


def test_the_budget_still_measures_what_the_extras_cost():
    """`candidates` reserves room for `extra` and reports the rest as dropped, so
    turning typing on makes its price visible rather than silently cutting the
    clicks."""
    import inspect

    from slime_htm.envs.browser import candidates

    source = inspect.getsource(candidates)
    assert "room = max(0, limit - len(extra))" in source
    assert "max(0, len(kept) - room)" in source


def test_a_goal_that_is_a_verb_then_a_value_still_offers_the_value():
    """`_TITLED` matches "Fork ChatGPT" as one phrase from position 0, and the
    first version skipped the whole match to drop the sentence-initial verb -- so
    `goal_spans("Fork ChatGPT.")` returned nothing at all. The verb is stripped
    now, not the match. code-reviewer reproduced this; it is the kind of bug a
    skip hides and a strip cannot."""
    assert goal_spans("Fork ChatGPT.") == ["ChatGPT"]
    assert goal_spans("Fork MetaSeq.") == ["MetaSeq"]
    assert goal_spans("Create MyProject now") == ["MyProject"]
    # And the verb itself is still not offered.
    assert "Fork" not in goal_spans("Fork ChatGPT.")
    assert "Create" not in goal_spans("Create MyProject now")


def test_the_answer_actions_keep_their_slots_on_a_busy_page(monkeypatch):
    """They were appended last, so three typable fields and four spans filled all
    twelve slots and cut them -- removing the one action class that can finish an
    answer-scored task, exactly where a form made the page busy."""
    axtree(monkeypatch, {str(bid): "textbox 'Field'" for bid in range(6)})
    goal = 'type "alpha" then "beta" then "gamma" then "delta"'

    offered, dropped = text_actions({}, goal)
    actions = [action for action, _ in offered]

    assert len(offered) == 12, "the cap still holds"
    answers = [action for action in actions if action.startswith("send_msg_to_user")]
    assert len(answers) == 4, "one per span, reserved rather than appended"
    assert dropped == 24 - 8, "and the fields it could not carry are counted"


def test_the_internal_cut_is_returned_because_candidates_cannot_see_it():
    """The first version returned a bare list and claimed `dropped` measured the
    loss. It did not: `candidates` reports only what *it* cuts, so everything this
    cap removed was invisible. Same shape as `candidates` now."""
    import inspect

    from slime_htm.envs.browser import candidates

    assert len(text_actions({}, "")) == 2, "(offered, dropped), even when empty"
    assert text_actions({}, "") == ([], 0)

    # The signatures agree, so a caller cannot forget which one returns a tuple.
    for function in (candidates, text_actions):
        assert "dropped" in inspect.getsource(function)


def coverage_module():
    """`scripts/webarena_span_coverage.py`, whose rules are pure functions."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/webarena_span_coverage.py"
    spec = importlib.util.spec_from_file_location("webarena_span_coverage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_span_is_not_coverage_just_because_it_shares_characters():
    """The three real cases code-reviewer found, each scored as covered by
    `value in span or span in value` and none of them produced by typing the span.
    They are the reason the rule is per channel rather than one containment."""
    covered_by = coverage_module().covered_by

    # The span `books` does not put a whole reddit URL anywhere.
    assert not covered_by(
        "http://www.reddit.com/f/books/59396/apple-books-has-a-free-audiobook",
        "must_include", "answer", ["books", "URLs", "10"])
    # Sending a date that happens to contain a 1 is a validator loophole, not an
    # answer: the answer channel needs equality.
    assert not covered_by("1", "must_include", "answer",
                          ["a11yproject on 3/5/2023", "3/5/2023"])
    # Three names are not the sentence that has to be sent.
    assert not covered_by(
        "The order is Princeton University, Yale University, Harvard University",
        "must_include", "answer",
        ["Princeton University", "Yale University", "Harvard University"])


def test_the_three_coverage_rules_do_what_their_channel_needs():
    covered_by = coverage_module().covered_by

    # Answer channel: equality, case and whitespace insensitive.
    assert covered_by("Bicycle", "must_include", "answer", ["bicycle"])
    assert not covered_by("Bicycle", "must_include", "answer", ["Bicycle lane"])

    # Page, exact_match: the field must end up equal to it.
    assert covered_by("1", "exact_match", "page", ["1"])
    assert not covered_by("1", "exact_match", "page", ["item 1"])

    # Page, must_include: typing a superstring does put the value on the page --
    # but on word boundaries, or a one-character requirement is met by anything.
    assert covered_by("mouth guard", "must_include", "page",
                      ["a mouth guard review"])
    assert not covered_by("1", "must_include", "page", ["a11yproject"])

    # And a span inside the value is never coverage, in either channel.
    assert not covered_by("Harvard University", "must_include", "page", ["Harvard"])


def test_an_answer_scored_task_is_not_reachable_by_typing_the_goal():
    """The result the rule produces, stated as the property it is about: a task
    whose answer must be read off the page cannot be answered with a substring of
    its own goal, whatever the validator would accept."""
    module = coverage_module()
    config = {
        "intent": "How many commits did kilian make to a11yproject on 3/5/2023?",
        "eval": {"eval_types": ["string_match"],
                 "reference_answers": {"must_include": ["1"]}},
    }
    from slime_htm.envs.browser import goal_spans

    assert module.classify(config, goal_spans(config["intent"])) == "none"


def test_a_search_results_url_is_a_typed_value_not_a_click():
    """A `url_match` task has no required content, which the first count read as
    "needs no typing". `Search for "usb wifi"` is graded on
    `catalogsearch/result/?q=usb+wifi`, and a query parameter goes into a box. 16
    of the 66 were this, which overstated the click-only ceiling by 16."""
    module = coverage_module()

    assert module.query_values(
        "__SHOPPING__/catalogsearch/result/?q=usb+wifi") == ["usb wifi"]
    assert module.query_values("__MAP__/search?query=Carnegie%20Museum") == [
        "Carnegie Museum"]
    assert module.query_values("__GITLAB__/dashboard/projects") == []
    assert module.query_values("") == []

    search = {"intent": 'Search for "usb wifi"',
              "eval": {"eval_types": ["url_match"],
                       "reference_url": "__SHOPPING__/catalogsearch/result/?q=usb+wifi"}}
    from slime_htm.envs.browser import goal_spans

    assert module.classify(search, goal_spans(search["intent"])) == "exact"
    assert module.classify(search, []) == "none", "without the span it is unreachable"

    # And the query has to match exactly: a longer phrase types a different query.
    longer = {"intent": 'Search for "batteries for iphone 13"',
              "eval": {"eval_types": ["url_match"],
                       "reference_url": "__SHOPPING__/catalogsearch/result/?q=iphone+13"}}
    assert module.classify(longer, goal_spans(longer["intent"])) == "none"


def test_a_written_action_is_parsed_strictly_or_refused():
    """The slow mode writing its action is the whole point: reading its answer off
    the enumerated codes meant escalating bought a longer thought and no larger
    action space, so a step whose right action is text was unwinnable at every
    price. This is the one place the project parses an answer out of generated
    text, so the guardrails are the test."""
    from slime_htm.envs.browser import parse_action

    # What it must accept, including a conclusion after a trace and a code fence.
    assert parse_action("I should type the query.\nfill('93', 'usb wifi')",
                        {"93"}) == ("fill('93', 'usb wifi')", None)
    assert parse_action('click("42")', {"42"}) == ("click('42')", None)
    assert parse_action("```\nselect_option(\"88\", \"Bicycle\")\n```",
                        {"88"}) == ("select_option('88', 'Bicycle')", None)
    assert parse_action("go_back()") == ("go_back()", None)
    # A rejected option named mid-thought must not win over the conclusion.
    action, _ = parse_action("click('9') looks wrong, so\nsend_msg_to_user('3 commits')")
    assert action == "send_msg_to_user('3 commits')"

    # What it must refuse, and never as a click.
    for text, bids, expected in (
        ("fill('99', 'x')", {"93"}, "not an element"),
        ('evaluate("alert(1)")', None, "not an action"),
        ("fill('93')", {"93"}, "takes 2 argument"),
        ("I will think about it", None, "no action call"),
        ("", None, "nothing was written"),
    ):
        action, problem = parse_action(text, bids)
        assert action is None and expected in problem, text
        assert problem and "click" not in problem.lower() or "not an" in problem


def test_double_quoted_arguments_survive_the_parser():
    """`re.findall` returns '' for a group that did not participate and a match
    object returns None, and the first version used findall -- so every
    double-quoted argument became an empty string and `click("42")` was refused
    as "'' is not an element on this page"."""
    from slime_htm.envs.browser import parse_action

    for text in ('click("42")', "click('42')", 'click( "42" )'):
        assert parse_action(text, {"42"}) == ("click('42')", None), text
    assert parse_action('fill("42", "two words")', {"42"}) == (
        "fill('42', 'two words')", None)


def test_the_whitelist_is_the_rule_and_not_whatever_parses():
    """The slow mode writes code-shaped text, so an action we did not intend is a
    step nobody can interpret afterwards."""
    from slime_htm.envs.browser import ACTION_ARITY, parse_action

    assert set(ACTION_ARITY) == {"click", "fill", "select_option", "send_msg_to_user",
                                 "scroll", "goto", "go_back", "hover", "press",
                                 "drag_and_drop"}
    for forbidden in ("eval(\"1\")", "exec('x')", "open('/etc/passwd')",
                      "page.evaluate('x')", "__import__('os')"):
        action, problem = parse_action(forbidden)
        assert action is None, forbidden

    # **The whitelist and the sentence shown to the model must name the same
    # actions.** They were two separate strings, and that is how `drag_and_drop`
    # could have been allowed while the instructions still said it may not be
    # written -- the model would never have tried, which looks identical to it
    # being forbidden. The gate's five drag tasks have the slow mode writing
    # `send_msg_to_user('... no drag action is available.')` 48 times, which was
    # a correct reading of a sentence that was wrong. `SLOW_INSTRUCTIONS` is now
    # built from `ACTION_ARITY` and raises for a key it has no shape for.
    from slime_htm.envs.browser_environment import SLOW_INSTRUCTIONS

    for name in ACTION_ARITY:
        assert f"{name}(" in SLOW_INSTRUCTIONS, \
            f"{name} is allowed but the model is never told it may write it"


def test_every_whitelisted_action_is_either_offered_to_the_fast_mode_or_not():
    """The metric's predicate, which the user chose on 2026-10-06 to be "the fast
    mode cannot express this" rather than "this needs generated text".

    The two differ and `drag_and_drop` is the difference: it needs no text --
    two bids, both already on the page -- and the fast mode still cannot offer
    it, because enumerating it is `|bids|^2` and `drag-cube` has 31 bids, so 961
    candidates for 52 slots. Under the narrower predicate a drag routed to the
    fast mode would not have counted as a misfit.
    """
    from slime_htm.envs.browser import (
        ACTION_ARITY,
        fast_expressible,
        fast_expressible_types,
        needs_content,
    )

    # Derived from what `candidates()` emits, not listed, so an added extra
    # widens it rather than leaving the metric reading 0 while the fast mode has
    # started taking actions it was supposed not to.
    assert fast_expressible_types() == {"click", "scroll"}

    cannot = {name for name in ACTION_ARITY if name not in fast_expressible_types()}
    assert cannot == {"drag_and_drop", "fill", "go_back", "goto", "hover", "press",
                      "select_option", "send_msg_to_user"}

    # The two predicates must not be the same function. `drag_and_drop` and
    # `go_back` are in the gap: no generated text, still not expressible.
    gap = {name for name in cannot if not needs_content(f"{name}()")}
    assert gap == {"drag_and_drop", "go_back", "hover", "press", "select_option"}, \
        "the wider predicate has to be strictly wider, or the user's choice is a no-op"

    assert fast_expressible("click('13')") and fast_expressible("scroll(0, 400)")
    assert not fast_expressible("drag_and_drop('13', '14')")
    assert not fast_expressible("go_back()")


def test_a_written_argument_cannot_become_a_second_statement():
    """The RED on 48f74ab, and the most serious defect in this project to date.

    `browsergym.core.action.base` runs the action with `exec` -- its own docstring
    says "WARNING: this is not safe!" -- and the first version re-rendered every
    argument as `f"'{argument}'"`. A double-quoted argument containing a single
    quote therefore closed the string and opened a statement. The page text a model
    reads is not ours (reddit, gitlab), so this was reachable by prompt injection,
    not only by a confused model. code-reviewer found it with these payloads."""
    from slime_htm.envs.browser import parse_action

    for payload in ("""fill("42", "x'); click('13")""",
                    """fill("42", "x'); __import__('os').system('id'); ('")"""):
        action, problem = parse_action(payload, {"42", "13"})
        assert problem is None, problem
        # It survives as ONE call whose argument is ONE string literal: executing
        # it types the payload as text, which is what typing text means.
        import ast

        call = ast.parse(action, mode="eval").body
        assert isinstance(call, ast.Call) and call.func.id == "fill"
        assert len(call.args) == 2
        assert all(isinstance(argument, ast.Constant) and isinstance(argument.value, str)
                   for argument in call.args)
        assert "__import__" not in action or call.args[1].value.count("__import__") == 1
        assert ";" not in action.split(",")[0], "nothing escaped into the call itself"


def test_an_apostrophe_in_a_typed_value_survives():
    """The same bug refused correct answers as well as admitting bad ones: the
    rendered `fill('42', 'O'Reilly')` was a SyntaxError, so a right answer failed."""
    from slime_htm.envs.browser import parse_action

    action, problem = parse_action('fill("42", "O\'Reilly")', {"42"})
    assert problem is None and action == '''fill('42', "O'Reilly")'''
    action, problem = parse_action('send_msg_to_user("it\'s 3")')
    assert problem is None and action == '''send_msg_to_user("it's 3")'''

    import ast

    for text in (action, '''fill('42', "O'Reilly")'''):
        assert ast.parse(text, mode="eval")  # parses, so exec would not raise


def test_a_key_the_browser_does_not_know_is_refused():
    """`press('42', 'NotAKey')` parsed cleanly before, so a model inventing a key
    spent a step on an action the browser cannot perform while the row said it
    pressed something."""
    from slime_htm.envs.browser import known_key, parse_action

    assert parse_action('press("42", "Enter")', {"42"}) == ("press('42', 'Enter')", None)
    assert parse_action('press("42", "Control+a")', {"42"})[1] is None
    assert parse_action('press("42", "a")', {"42"})[1] is None
    assert parse_action('press("42", "NotAKey")', {"42"})[0] is None
    assert known_key("ArrowDown") and known_key("F5") and not known_key("Wibble")
    assert not known_key("Wrong+a"), "a modifier has to be a modifier"


def test_a_labelled_or_commented_conclusion_is_still_an_action():
    """How a model actually writes a conclusion. Refusing these ends an episode
    over phrasing rather than over a decision -- and the refusal rate would read
    as the model being incapable."""
    from slime_htm.envs.browser import parse_action

    for text in ('Final: click("42")', 'Action: fill("42", "q")',
                 "click('42')  # done", "**click('42')**", "```\nclick('42')\n```"):
        action, problem = parse_action(text, {"42"})
        assert problem is None, f"{text!r}: {problem}"
    # A value containing a # is not a comment.
    assert parse_action("fill('42', 'issue #3')", {"42"}) == (
        "fill('42', 'issue #3')", None)


def test_scroll_is_rendered_as_numbers():
    """It takes numbers, and `scroll('0', '200')` would reach the browser as
    strings."""
    from slime_htm.envs.browser import parse_action

    assert parse_action("scroll(0, 200)") == ("scroll(0, 200)", None)
    assert parse_action("scroll('a', 'b')")[0] is None


def test_the_arguments_are_read_by_python_and_not_by_a_regex():
    """Two regexes tried this and both were wrong in a way that lost the value:
    `findall` returned '' for a group that did not participate, so every
    double-quoted argument was empty; then `finditer` read the first `""` of
    `\"\"\"x\"\"\"` as an empty string and dropped the text. Parsing is not
    executing -- `exec` is the unsafe part, and it runs on the string we render
    ourselves -- so there was never a reason to approximate Python's own parser."""
    from slime_htm.envs.browser import parse_action

    assert parse_action('fill("42", """x""")', {"42"}) == ("fill('42', 'x')", None)
    assert parse_action("fill('42', '''multi''')", {"42"}) == (
        "fill('42', 'multi')", None)
    # Escapes are the parser's business too, and a regex would have kept the slash.
    assert parse_action(r'fill("42", "a\tb")', {"42"})[0] == "fill('42', 'a\\tb')"
    assert parse_action('fill("42", "a" "b")', {"42"})[0] == "fill('42', 'ab')", \
        "adjacent literals concatenate, as they do in Python"


def test_an_argument_that_is_not_a_literal_is_refused():
    """A model that wrote `fill('42', query)` has written something this
    environment cannot interpret, and guessing would invent a value it never
    chose."""
    from slime_htm.envs.browser import parse_action

    for text in ('fill("42", some_variable)', 'click(bid)',
                 'fill("42", 1 + 1)', 'fill("42", f"{x}")'):
        action, problem = parse_action(text, {"42"})
        assert action is None, text
        assert "literal" in problem or "parse" in problem, (text, problem)


def test_the_content_free_half_of_the_action_set_is_offered_and_parses():
    """browsergym's `chat`+`bid` set has 13 actions and `candidates` built one.
    They split into those needing content -- fill, select_option,
    send_msg_to_user, press, upload_file -- and those needing none. The second
    half can be enumerated for the fast mode with no change to what it is: no
    generation, still one index into a list. They were missing because nobody
    added them, and their absence is why `candidates` filters to the window at
    all."""
    from slime_htm.envs.browser import CONTENT_FREE, content_free_actions, parse_action

    assert content_free_actions() == ("scroll(0, 400)",), "scrolling up does nothing yet"
    assert content_free_actions(scrolled=True) == CONTENT_FREE
    for action in CONTENT_FREE:
        parsed, problem = parse_action(action)
        assert problem is None and parsed == action, (action, problem)


def test_a_negative_number_is_a_literal():
    """`scroll(0, -400)` is one of this module's own constants and was refused as
    "not a literal", because a negative number is a `UnaryOp` over a `Constant`.
    I had written the rule twice and fixed only the first copy; it is one helper
    now, used on what the model wrote and on what we rendered."""
    from slime_htm.envs.browser import parse_action

    assert parse_action("scroll(0, -400)") == ("scroll(0, -400)", None)
    assert parse_action("scroll(-1, -2)") == ("scroll(-1, -2)", None)


def test_a_value_that_is_not_text_or_a_number_is_refused():
    """`fill('42', b'x')` typed the characters b, quote, x, quote; `None` typed the
    word None. bool is excluded before int, since it is a subclass."""
    from slime_htm.envs.browser import parse_action

    for text, named in (("fill('42', b'x')", "bytes"), ("fill('42', None)", "NoneType"),
                        ("fill('42', True)", "bool")):
        action, problem = parse_action(text, {"42"})
        assert action is None and named in problem, (text, problem)
    assert parse_action("fill('42', 3.5)", {"42"})[0] == "fill('42', '3.5')"


def test_the_declined_option_stays_declined_and_says_so():
    """The user declined the content-needing actions for the fast mode on
    2026-10-05. The module keeps them, because the decision rests on what they
    measure, and nothing may call them: a future caller wiring them in silently
    would change the pre-registered comparison without changing the plan."""
    import inspect

    from slime_htm.envs import browser, browser_environment, gate

    assert "Declined for the fast mode" in browser.text_actions.__doc__
    for module in (gate, browser_environment):
        source = inspect.getsource(module)
        assert not uses(source, "text_actions"), \
            f"{module.__name__} must not call it"


def test_the_hang_guard_is_told_apart_from_other_exits_by_the_stack_file(tmp_path):
    """**The launcher contract depends on this and the exit code cannot carry it.**

    `faulthandler.dump_traceback_later(exit=True)` always exits 1 -- exps measured
    it -- so "the hang guard fired, resume" and "a refusal, stop" share a status
    and a launcher cannot follow the table by status alone. faulthandler takes no
    other code, so the signal has to be an artefact: the run's stack file **grew**.

    Both halves are measured here, because the first probe I wrote for this got
    the answer backwards. It blocked in `select.select`, which SIGALRM *can*
    interrupt, so `StepTimeout` propagated and exited 1 with an empty stack file
    -- and had I believed it I would have told code-reviewer its scheme did not
    work. The faulthandler layer is reached only where the signal layer is absent
    or cannot land, which off the main thread is the former and in the greenlet
    bridge is the latter."""
    import subprocess
    import sys
    from pathlib import Path

    source = Path(__file__).resolve().parent.parent / "src"

    def run(mode, stacks):
        script = tmp_path / f"{mode}.py"
        script.write_text(
            "import sys, threading, time\n"
            f"sys.path.insert(0, {str(source)!r})\n"
            "from slime_htm.envs.browser import StepTimeout, step_deadline\n"
            f"stacks = {str(stacks)!r}\n"
            "def blocked():\n"
            "    with step_deadline(0.4, stack_file=stacks):\n"
            "        time.sleep(30)\n"
            f"if {mode!r} == 'hang':\n"
            "    t = threading.Thread(target=blocked, daemon=True)\n"
            "    t.start(); t.join(20); print('NOT REACHED')\n"
            "else:\n"
            "    with step_deadline(5.0, stack_file=stacks):\n"
            "        pass\n"
            "    print('returned')\n")
        done = subprocess.run([sys.executable, str(script)], capture_output=True,
                              text=True, timeout=90)
        grew = Path(stacks).stat().st_size if Path(stacks).exists() else 0
        return done.returncode, grew

    code, grew = run("hang", tmp_path / "hang-stacks.log")
    assert code == 1, "faulthandler always exits 1, which is the problem"
    assert grew > 0, "and the stack file growing is what says it was the hang guard"

    code, grew = run("clean", tmp_path / "clean-stacks.log")
    assert code == 0 and grew == 0, "a call that returns writes nothing"


def test_the_contract_distinguishes_the_hang_exit_by_the_artefact_not_the_code():
    """So a launcher can be written from the table. exps was matching on the text
    of a message instead, which a reword breaks silently."""
    from conftest import repo_path

    contract = repo_path("experiments/browser-gate/README.md").read_text()
    row = next((line for line in contract.splitlines()
                if line.startswith("| exit 1 +")), None)
    assert row, "the table has a row keyed on exit 1 plus the artefact"
    assert "stacks.log" in row and "grew" in row
    assert "resume" in row
    plain = next((line for line in contract.splitlines()
                  if line.startswith("| exit 1 |")), None)
    assert plain and "did not grow" in plain, "and the other half of exit 1 says stop"


def test_the_documented_action_inventory_matches_browsergym():
    """`docs/datasets.md` 1.4 states 35 actions defined, 32 reachable through a
    subset, and `check`/`uncheck`/`noop` reachable through none. Those are
    **upstream** numbers, so they drift when browsergym is upgraded and nothing
    would notice -- the defect shape this repository has paid for repeatedly.

    Skipped where browsergym is absent (it is a dev extra and not installed in
    every checkout), which means the guard runs in the container and on the
    cluster, where the numbers were measured in the first place."""
    import ast
    import inspect

    pytest.importorskip("browsergym.core.action.functions")
    from browsergym.core.action import functions
    from browsergym.core.action.highlevel import ACTION_SUBSETS

    defined = {node.name for node in ast.parse(inspect.getsource(functions)).body
               if isinstance(node, ast.FunctionDef)
               and not node.name.startswith("_")}
    offered = {f.__name__ for subset in ACTION_SUBSETS.values() for f in subset}

    assert len(defined) == 35, "the documented inventory, so a change is visible"
    assert len(offered) == 32
    assert sorted(defined - offered) == ["check", "noop", "uncheck"], \
        "no venue preset offers a semantic checkbox action; tasks click instead"

    # The free-text five: the actions whose argument cannot be enumerated from
    # the page, which is the line between the two modes.
    free_text = {"fill", "send_msg_to_user", "keyboard_type",
                 "keyboard_insert_text", "report_infeasible"}
    assert free_text <= defined
    # And the three that the doc's older split called content-bearing but whose
    # arguments **are** enumerable: options come from the <select>, keys from a
    # finite set, a file path from the environment.
    assert {"select_option", "press", "upload_file"} <= defined


def test_the_finite_key_set_is_ours_and_is_what_the_doc_points_at():
    """1.4 says `press`'s argument is enumerable because the key names are a
    finite set "we maintain ourselves". That claim is about our code, so it can
    be checked without browsergym."""
    from slime_htm.envs.browser import CONTENT_FREE, known_key

    assert known_key("Enter") and known_key("a") and known_key("Control+a")
    assert not known_key("NotAKey"), "which is why press's argument is checkable"
    assert not known_key("")
    # And the fast mode's content-free set really is only scrolling, as 1.3 says.
    assert all(action.startswith("scroll(") for action in CONTENT_FREE)
    assert len(CONTENT_FREE) == 2


def test_the_doc_explains_bid_before_it_uses_it():
    """The user asked what `bid` meant, which is the answer to "is this document
    readable by someone who did not write it". It is used from the action-subset
    table onward and was never defined -- and the subset *named* `bid` is not the
    same thing as the HTML attribute named `bid`, which is the kind of collision
    that needs saying rather than noticing."""
    from conftest import repo_path

    doc = repo_path("docs/datasets.md").read_text()
    where = doc.index("### 1.2 ")
    section = doc[where:doc.index("### 1.3 ")]

    assert "browsergym id" in section, "what the letters stand for"
    assert "BROWSERGYM_ID_ATTRIBUTE" in section, "and where it is defined"
    # The generation rule, which is what makes `a48` legible rather than opaque.
    assert "iframe" in section and "aA" in section
    # The property with consequences: it is not stable across a navigation, which
    # is why parse_action checks against the current step's set only.
    assert "不是稳定标识符" in section
    assert "parse_action" in section
    # And the naming collision with the subset of the same name.
    assert "子集名" in section
    # Used before it is explained would defeat the point.
    assert doc.index("browsergym id") < doc.index("### 1.3 "), \
        "defined before the table that leans on it"


def action_lists():
    from conftest import repo_path

    return json.loads(
        repo_path("src/slime_htm/envs/browsergym-actions.json").read_text())


def test_the_action_lists_cover_every_action_exactly_once():
    """The user asked for both lists enumerated into a file: what can be chosen
    with no change to the code, and what cannot. A list that quietly omits an
    action is the failure mode, so the three buckets must partition all 35."""
    found = action_lists()
    buckets = ("selectable_now", "selectable_in_principle", "needs_content")
    names = [row["action"] for bucket in buckets for row in found[bucket]]

    assert len(names) == 35, "every action browsergym defines appears"
    assert len(set(names)) == 35, "and none appears twice"
    assert found["counts"] == {b: len(found[b]) for b in buckets}


def test_the_split_is_by_where_the_values_come_from():
    """`candidates_from` is the only criterion: an action with a source is
    choosable, one without has to be generated. If a row could be moved between
    buckets without changing its source, the split would be an opinion."""
    found = action_lists()

    for row in found["selectable_now"] + found["selectable_in_principle"]:
        assert row["candidates_from"], f"{row['action']} has no source"
    for row in found["needs_content"]:
        # None, or a source marked partial -- `goto` can take page hrefs but not
        # an arbitrary URL, and partial is not enumerable.
        source = row["candidates_from"]
        assert source is None or "部分" in source, row["action"]


def test_selectable_now_is_only_what_the_code_really_offers():
    """Two actions, not twenty-five. `docs/datasets.md` 1.5 splits by whether a
    value can be enumerated, which reads as though `select_option` and `press`
    were usable today -- they are not, because nobody builds their candidates.
    That distinction is the whole point of a third bucket."""
    from slime_htm.envs.browser import CONTENT_FREE

    found = action_lists()
    now = {row["action"] for row in found["selectable_now"]}

    assert now == {"click", "scroll"}
    assert set(found["wired_up"]) == now
    assert all(action.startswith("scroll(") for action in CONTENT_FREE)
    # And the biggest bucket is the one that needs enumeration code, not a mode
    # change -- which is the cost of adding any of them.
    assert len(found["selectable_in_principle"]) == 25
    assert {"select_option", "press", "hover", "clear", "tab_focus"} <= {
        row["action"] for row in found["selectable_in_principle"]}


def test_the_free_text_actions_are_the_ones_that_divide_the_two_modes():
    found = action_lists()
    needs = {row["action"] for row in found["needs_content"]}

    assert {"fill", "send_msg_to_user", "keyboard_type", "keyboard_insert_text"} \
        <= needs, "the free-text four"
    # Not content in the strict sense, but still not enumerable from the page.
    assert {"upload_file", "mouse_upload_file", "goto"} <= needs
    # `report_infeasible` is in needs_content and the file says why a degenerate
    # enumerable version exists, so nobody re-derives it.
    reason = next(r for r in found["needs_content"]
                  if r["action"] == "report_infeasible")
    assert "退化版" in reason["note"]


def test_the_option_trap_is_recorded_rather_than_rediscovered():
    """`<option>` tags do get a bid -- `standard_html` marks 125 standard tags --
    so "just click the option" looks like it should replace `select_option`. It
    does not: a collapsed `<select>`'s options have no box in the viewport, so
    `candidates()`'s visibility filter removes them."""
    found = action_lists()

    assert "option" in found["filtered_out"]
    assert "select_option" in found["filtered_out"]["option"]
    select = next(r for r in found["selectable_in_principle"]
                  if r["action"] == "select_option")
    # And why it is not usable today: the values are in the tree, the pairing of
    # option to <select> needs the hierarchy, and we read the flattened string.
    assert "拍平" in select["note"]


def test_the_arity_table_agrees_with_installed_browsergym():
    """The classification is ours; the arities are upstream's. If a signature
    changes under us the file is wrong, and nothing else would notice."""
    pytest.importorskip("browsergym.core.action.functions")
    from conftest import load_script

    module = load_script("browsergym_actions")
    signatures = module.upstream_signatures()
    assert signatures is not None and len(signatures) == 35

    _, mismatched = module.build(signatures)
    assert mismatched == [], f"arities disagree with upstream: {mismatched}"


def test_the_doc_points_at_the_enumerated_file_rather_than_restating_it():
    """A table in prose and a file on disk are two copies, and the prose one
    drifts. 1.6 names the file and the generator; the counts it quotes are
    asserted against the file here, so a regeneration that changes them fails
    rather than leaving the doc wrong."""
    from conftest import repo_path

    doc = repo_path("docs/datasets.md").read_text()
    found = action_lists()
    section = doc[doc.index("### 1.6 "):doc.index("### 1.7 ")]

    assert "src/slime_htm/envs/browsergym-actions.json" in section
    assert "scripts/browsergym_actions.py" in section
    for bucket, count in found["counts"].items():
        assert f"**{count}**" in section, f"{bucket} count {count} is quoted"
    assert f"{sum(found['counts'].values())}" in section, "and the total"
    # The distinction the third bucket exists for, stated rather than implied.
    assert "概念上可枚举" in section and "现有代码不改" in section


def test_the_shared_budget_is_what_decides_whether_a_set_fits():
    """The user asked whether the candidate sets can get very large. They can,
    and the answer is not per-action: all 52 slots are **shared**, so adding the
    cheap actions is what overflows first, not the expensive ones.

    Measured on 28853's corpus, MiniWoB click, 317 steps."""
    found = action_lists()
    budget = {row["label"]: row for row in found["shared_budget"]}

    assert found["slots"] == 52
    # Constants alone are safe; the element-indexed ones are what overflow.
    assert budget["只加常量动作"]["fits"] is True
    overflow = budget["常量 + 和 click 同循环的"]
    assert overflow["fits"] is False
    # 3.9x, not 4x. The doc said "four times over" before this assertion was
    # written, which is the same habit as every other unchecked multiple today:
    # 205/52 = 3.94. Overstating a problem is as wrong as understating it.
    assert overflow["worst_total"] == 205
    assert 3.9 < overflow["worst_total"] / found["slots"] < 4.0, \
        "nearly four times over on MiniWoB, the smallest venue there is"
    # And it is monotone: every larger set is worse.
    totals = [row["worst_total"] for row in found["shared_budget"]]
    assert totals == sorted(totals), "more actions, never fewer candidates"


def test_the_three_shapes_of_overflow_are_recorded_with_their_numbers():
    """A cross product, a page-independent constant, and a square. They need
    different answers, so the file keeps them apart rather than calling them all
    'too many'."""
    found = action_lists()
    by_name = {row["action"]: row
               for row in found["selectable_in_principle"]}

    # Cross product: the key set multiplies the element count.
    press = by_name["press"]
    assert press["candidates_formula"] == "E x K"
    assert press["at_measured_scale"]["median"] > found["slots"], \
        "over the limit on a median page, not only a worst one"
    assert press["build_cost"] == "needs_a_subset"

    # Page-independent constant: the least intuitive one, since it takes a
    # single argument and still never fits.
    keys = by_name["keyboard_press"]
    assert keys["candidates_formula"] == "K"
    assert keys["at_measured_scale"]["median"] == \
        keys["at_measured_scale"]["worst"], "page size does not enter it"
    assert keys["at_measured_scale"]["fits"] is False

    # Square: fits only up to seven elements.
    drag = by_name["drag_and_drop"]
    assert drag["candidates_formula"] == "E^2"
    assert drag["build_cost"] == "quadratic"
    assert 7 * 6 <= found["slots"] < 8 * 7, "the seven-element ceiling"

    # And choosing the key subset is flagged as a pre-registration, because it
    # changes which tasks are reachable.
    assert "预注册" in found["build_cost_kinds"]["needs_a_subset"]


def test_the_measured_scale_says_which_venue_it_came_from():
    """37 of 52 on a median page is comfortable and 205 on a worst page is not --
    both from MiniWoB. WebArena is an order of magnitude larger, and a number
    quoted without its venue would be read as general."""
    found = action_lists()
    scale = found["measured_scale"]

    assert scale["venue"] == "miniwob click" and scale["steps"] == 317
    assert scale["candidates_at_limit"] == 0, "nothing hit 52 in the real run"
    assert scale["elements_median"] == 2 and scale["elements_max"] == 14
    assert "Wikipedia" in scale["webarena_note"], "and the larger venue is named"


class _Page:
    """Enough of a Playwright page for `candidates()`: a viewport and a zoom."""

    def __init__(self, height, scale=None):
        self.viewport_size = {"width": 332, "height": height}
        if scale is not None:
            self._bgym_scale_factor = scale

    def evaluate(self, *_args):  # pragma: no cover  only `selector` calls this
        raise AssertionError("no selector is passed here")


class _Env:
    def __init__(self, page):
        self.unwrapped = type("U", (), {"page": page})()


def _click_test_observation():
    """`miniwob.click-test` as browsergym actually reports it.

    Taken from a probe against the installed package, 2026-10-06. The viewport
    is 332x214 and `_bgym_scale_factor` is 1.5, so the one button's true DOM
    rect of `[49, 145, 48, 48]` arrives multiplied: `[73.5, 217.5, 72, 72]`.
    Its true bottom edge is 193, well inside 214 -- it is on screen.
    """
    boxes = {
        "0": ([0.0, 0.0, 498.0, 315.0], False),
        "9": ([0.0, 0.0, 498.0, 315.0], False),
        "10": ([0.0, 0.0, 240.0, 315.0], False),
        "11": ([0.0, 0.0, 240.0, 75.0], False),
        "12": ([0.0, 75.0, 240.0, 217.5], False),
        "13": ([73.5, 217.5, 72.0, 72.0], True),
    }
    return {"extra_element_properties":
            {bid: {"bbox": box, "clickable": click}
             for bid, (box, click) in boxes.items()},
            "axtree_object": None}


@pytest.fixture
def unlabelled(monkeypatch):
    """`candidates()` labels each element from the accessibility tree, which
    needs browsergym installed. These two tests are about the geometry filter,
    so the labelling step is stubbed out rather than skipping them where the
    package is absent -- the filter is ours and testable without a browser."""
    from slime_htm.envs import browser

    monkeypatch.setattr(browser, "element_names", lambda observation: {})


def test_the_viewport_filter_compares_the_two_coordinate_systems_browsergym_uses(
        unlabelled):
    """The bboxes are zoomed and the viewport is not, so the filter has to scale.

    This is the defect, not a style point: on `click-test` -- one button, the
    simplest task in the venue -- the unscaled comparison offered the fast mode
    **nothing but `scroll`**, and across the 54 tasks of the click group it
    removed 32% of all click candidates on 34 of them. Every `outside_fast_set`
    number the gate reported was partly measuring this.
    """
    from slime_htm.envs.browser import candidates

    observation = _click_test_observation()

    offered, dropped = candidates(_Env(_Page(214, scale=1.5)), observation, 52)
    assert [action for action, _ in offered] == ["click('13')"], (
        "the button is at a true y of 145 in a 214-high viewport; scaling the "
        "height by 1.5 is what puts the reported 217.5 back on screen")
    assert dropped == 0

    # The bug, pinned: the same page with the factor ignored offers nothing.
    bare = _Page(214)
    assert not hasattr(bare, "_bgym_scale_factor")
    offered_unscaled, _ = candidates(_Env(bare), observation, 52)
    assert offered_unscaled == [], (
        "this is what the gate ran: a visible button judged below the fold")


def test_a_venue_that_sets_no_zoom_is_unaffected_by_the_scale_read(unlabelled):
    """`getattr(..., 1.0)` is how browsergym reads the factor in its own three
    call sites. WebArena sets none, so the fix must be a no-op there -- an
    element genuinely below the fold stays filtered."""
    from slime_htm.envs.browser import candidates

    observation = {"axtree_object": None, "extra_element_properties": {
        "1": {"bbox": [0.0, 10.0, 50.0, 20.0], "clickable": True},
        "2": {"bbox": [0.0, 900.0, 50.0, 20.0], "clickable": True},
    }}
    offered, _ = candidates(_Env(_Page(800)), observation, 52)
    assert [action for action, _ in offered] == ["click('1')"], \
        "no factor means scale 1.0, and 900 is still past 800"


def test_needs_content_is_the_parameter_source_not_the_argument_count():
    """The predicate the routing metric counts, taken from the measured file.

    `scroll(dx, dy)` takes two arguments and needs no content because both are
    constants; `goto(url)` takes one and cannot be enumerated from the page. An
    earlier version of this split, re-derived by hand in a source comment, put
    `tab_focus` and `report_infeasible` on the wrong sides.
    """
    from slime_htm.envs.browser import NEEDS_CONTENT, action_name, needs_content

    found = action_lists()
    measured = {row["action"] for row in found["needs_content"]}
    # The file classifies all 35 browsergym actions; the environment offers 20.
    ENV = {"send_msg_to_user", "report_infeasible", "clear", "click", "dblclick",
           "drag_and_drop", "fill", "focus", "hover", "press", "scroll",
           "select_option", "upload_file", "go_back", "go_forward", "goto",
           "new_tab", "tab_close", "tab_focus", "noop"}
    assert NEEDS_CONTENT == measured & ENV, \
        "the constant must be the measured classification restricted to our 20"

    assert needs_content("fill('13', 'Bicycle')")
    assert needs_content("send_msg_to_user('42')")
    assert not needs_content("click('13')")
    assert not needs_content("scroll(0, 400)"), "two arguments, both constants"
    assert not needs_content("press('13', 'Enter')"), "a key is one of 43 names"

    # A parser failure is not an action, so it is not content-needing: counting it
    # here would put a parse error into a routing statistic.
    assert action_name("I think we should click the button") is None
    assert not needs_content("I think we should click the button")
    assert not needs_content(None) and not needs_content("")


def test_the_fast_mode_cannot_take_a_content_needing_action_today():
    """`unfit_fast_steps` is an invariant now and a learning signal later.

    Every action the fast mode can be offered comes from `candidates()`, which
    emits `click('<bid>')` plus whatever `extra` holds, and the only `extra` the
    environment passes is `content_free_actions()`. So a step with
    `needs_content and not escalated` cannot occur -- the metric reads 0 because
    the fast mode is incapable, **not** because a router learned anything. This
    test is what makes that claim checkable rather than asserted, and it is what
    starts failing if anyone widens the candidate set without widening the
    metric's reading.
    """
    from slime_htm.envs.browser import (
        CONTENT_FREE,
        content_free_actions,
        needs_content,
    )

    for action in CONTENT_FREE + content_free_actions(scrolled=True):
        assert not needs_content(action), f"{action} is offered to the fast mode"
    assert not needs_content("click('whatever')"), "the other thing it is offered"

    observation = {"axtree_object": None, "extra_element_properties": {
        "7": {"bbox": [0.0, 10.0, 50.0, 20.0], "clickable": True}}}
    offered, _ = candidates_without_labels(observation)
    assert offered, "the fixture has to offer something for this to mean anything"
    assert not any(needs_content(action) for action, _ in offered)


def candidates_without_labels(observation, height=800):
    """`candidates()` over a fake page, with the AXTree label step removed."""
    from unittest.mock import patch

    from slime_htm.envs import browser

    with patch.object(browser, "element_names", lambda _observation: {}):
        return browser.candidates(_Env(_Page(height)), observation, 52)


def test_a_drag_cannot_name_an_element_the_page_does_not_have():
    """`drag_and_drop` is the only whitelisted action with two bids, and checking
    only `arguments[0]` would let the target be anything.

    That matters more than it looks: browsergym runs the action with `exec`, so
    an unknown target raises *inside* the environment, which the gate reads as a
    crashed step rather than as a refused action -- and a refused action is a
    fact about the model's choice while a crash is a fact about us.
    """
    from slime_htm.envs.browser import parse_action

    assert parse_action("drag_and_drop('13', '14')", {"13", "14"}) == (
        "drag_and_drop('13', '14')", None)

    action, problem = parse_action("drag_and_drop('13', '99')", {"13", "14"})
    assert action is None and problem == "'99' is not an element on this page", \
        "the second bid is checked too, not just the first"

    action, problem = parse_action("drag_and_drop('99', '14')", {"13", "14"})
    assert action is None and "'99'" in problem, "and so is the first"

    action, problem = parse_action("drag_and_drop('13')", {"13", "14"})
    assert action is None and "2 argument(s), 1 written" in problem


def test_a_bid_taking_action_cannot_be_added_to_only_one_table():
    """code-reviewer's nit on 82e937e, and it fails open, which is why it is a
    test rather than a comment.

    The bid count was a hand-written dict inside `parse_action` read with
    `.get(name, 0)`. An action added to `ACTION_ARITY` alone would then have
    **skipped the bid check silently** -- the one guard that stops a written
    action naming an element the page does not have. `BID_ARGUMENTS` now has one
    entry per key and `parse_action` subscripts it, so the omission raises.
    """
    from slime_htm.envs.browser import ACTION_ARITY, BID_ARGUMENTS, parse_action

    assert set(BID_ARGUMENTS) == set(ACTION_ARITY), (
        f"one entry per action, so an omission cannot default to zero: "
        f"{sorted(set(ACTION_ARITY) ^ set(BID_ARGUMENTS))}")
    for name, count in BID_ARGUMENTS.items():
        assert 0 <= count <= ACTION_ARITY[name], f"{name} cannot have {count} bids"

    # Subscripted, so a whitelisted action with no entry raises rather than
    # parsing without a bid check. Simulated by removing one.
    missing = {k: v for k, v in BID_ARGUMENTS.items() if k != "click"}
    import slime_htm.envs.browser as browser
    saved = browser.BID_ARGUMENTS
    browser.BID_ARGUMENTS = missing
    try:
        with pytest.raises(KeyError, match="click"):
            parse_action("click('42')", {"42"})
    finally:
        browser.BID_ARGUMENTS = saved

    assert parse_action("click('42')", {"42"}) == ("click('42')", None)
