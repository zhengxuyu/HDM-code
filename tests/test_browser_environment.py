"""The browser adapter, without a browser.

What can be wrong without one is the part that matters: whether the actions it
publishes are the ones it can take, whether an unoffered action is refused rather
than guessed, whether an episode that offers nothing stops, and whether the seed
and the task list are what the caller asked for. The gate and this share one
candidate rule on purpose -- two rules for what a step offers is the defect this
repository has already had four times.
"""
import pytest

from slime_hdm import Environment
from slime_hdm.envs.browser_environment import BrowserEnvironment


class FakePage:
    """A browsergym env: a goal, a few elements, and a reward at the end."""

    def __init__(self, steps_to_finish=2, offer=("11", "22", "33")):
        self.steps_to_finish = steps_to_finish
        self.offer = list(offer)
        self.taken = []
        self.closed = False
        self.seeds = []

    def _observation(self):
        return {"goal": "find the submit button",
                "extra_element_properties": {
                    bid: {"clickable": True, "visibility": 1.0, "bbox": [0, 0, 10, 10]}
                    for bid in self.offer},
                "axtree_object": {"nodes": []}}

    def reset(self, seed=None):
        self.seeds.append(seed)
        self.taken = []
        return self._observation(), {}

    def step(self, action):
        self.taken.append(action)
        done = len(self.taken) >= self.steps_to_finish
        return self._observation(), (1.0 if done else 0.0), done, False, {}

    def close(self):
        self.closed = True


def browser(monkeypatch, pages, **kwargs):
    """A BrowserEnvironment over fake pages, with the candidate rule stubbed to the
    element ids the page offers -- the real one needs an AXTree."""
    import sys
    import types

    from slime_hdm.envs import browser as rule

    made = []
    gym = types.ModuleType("gymnasium")

    def make(task, **kwargs):
        page = pages[task]
        made.append(task)
        # Recorded on the page rather than in `made`, which existing tests read
        # as a list of task names.
        page.made_with = kwargs
        return page

    gym.make = make
    gym.registry = dict.fromkeys(pages)
    monkeypatch.setitem(sys.modules, "gymnasium", gym)
    def stub_candidates(env, observation, limit, extra=(), selector=None):
        """The element ids the page offers, under the **real** budget rule.

        `room = limit - len(extra)` and `dropped = len(kept) - room`, copied from
        `browser.candidates`, because the first version of this stub sliced to
        `limit` and then appended the extras -- so it published `limit + 1`
        actions where the real rule publishes `limit`, and a test of the budget
        was measuring the stub. A double that does not keep its original's
        contract tests nothing.
        """
        kept = [(f"click('{bid}')", f"click('{bid}')  -- element {bid}")
                for bid in observation.get("extra_element_properties", {})]
        room = max(0, limit - len(extra))
        return (kept[:room] + [(action, action) for action in extra],
                max(0, len(kept) - room))

    monkeypatch.setattr(rule, "candidates", stub_candidates)
    monkeypatch.setattr(rule, "state_text",
                        lambda observation, goal, trail, history=6, chars=4000:
                        f"{goal} | after {trail}")
    import slime_hdm.envs.browser_environment as module
    monkeypatch.setattr(module, "candidates", rule.candidates)
    monkeypatch.setattr(module, "state_text", rule.state_text)
    return BrowserEnvironment(list(pages), **kwargs), made


def _playwright_error(message):
    """An exception that looks to `browser_failure` exactly like playwright's.

    Built rather than imported: playwright is a dev extra, and the production
    check asks which **module** raised, so a double has to carry a module name
    from that package or it is testing nothing.
    """
    error = type("TargetClosedError", (Exception,), {})
    error.__module__ = "playwright._impl._errors"
    return error(message)


def test_it_satisfies_the_protocol_the_rollout_drives():
    assert isinstance(BrowserEnvironment(["one"]), Environment)
    with pytest.raises(ValueError, match="no tasks"):
        BrowserEnvironment([])


def test_the_actions_published_are_the_ones_it_can_take(monkeypatch):
    """The model reads `click('11')  -- element 11` because a bare browsing id is
    not a choice anybody can make, and the page takes `click('11')`. The rollout
    hands back a string it was given, so the mapping has to exist here."""
    pages = {"a": FakePage()}
    env, _ = browser(monkeypatch, pages)

    state, actions = env.reset(0)
    assert "find the submit button" in state
    assert actions == ["click('11')  -- element 11", "click('22')  -- element 22",
                       "click('33')  -- element 33", "scroll(0, 400)"]

    env.step(actions[1])
    assert pages["a"].taken == ["click('22')"], "the page got the action, not the label"


def test_an_action_it_never_offered_is_refused_rather_than_guessed(monkeypatch):
    """Clicking the nearest element instead would hide a caller that is not
    choosing from what it was given, behind a plausible episode."""
    env, _ = browser(monkeypatch, {"a": FakePage()})
    env.reset(0)

    with pytest.raises(ValueError, match="not one of the 4 actions"):
        env.step("click('99')")
    with pytest.raises(ValueError, match="not one of"):
        env.step("click('11')")  # the real action, not the published string


def test_a_page_with_no_clickable_element_can_still_scroll(monkeypatch):
    """This is the defect 28845 was running with, and it is the reason the
    content-free actions default on.

    `candidates` filters to the viewport **because** there was no scroll action,
    so without one the fast mode is sealed into the first screen. A page whose
    first screen has no clickable element then offered nothing at all, the
    rollout broke before taking a single decision, and the collector dropped the
    episode as `no_steps` -- exps measured 13.8% of 28845's episodes going that
    way, and every surviving episode was choosing from one screen.

    `gate.run_episode` passed these actions and `BrowserEnvironment` did not,
    which made the CDPO run's fast mode weaker than the one its own precondition
    gate had measured."""
    env, _ = browser(monkeypatch, {"a": FakePage(offer=())})
    state, actions = env.reset(0)

    assert actions == ["scroll(0, 400)"], "scrolling is a choice, so the episode runs"
    assert env.ended is None, "and it has not ended"
    assert state, "the state is still reported, so the row says what was seen"
    # Up only after the page has moved: it is useless before and costs a slot.
    _, actions, _, _ = env.step("scroll(0, 400)")
    assert actions == ["scroll(0, 400)", "scroll(0, -400)"]


def test_an_episode_with_nothing_to_choose_from_still_stops(monkeypatch):
    """With the content-free actions switched off there is a genuinely empty set,
    and it must still be reported rather than raise. The gate's pre-registered
    click-only arm is what reads this path."""
    env, _ = browser(monkeypatch, {"a": FakePage(offer=())}, content_free=False)
    state, actions = env.reset(0)
    assert actions == []
    assert env.ended == "no_candidates"
    assert state, "the state is still reported, so the row says what was seen"


def test_the_reward_arrives_once_and_the_actions_stop_with_it(monkeypatch):
    """Offering a choice after the reward has arrived would add a step that cannot
    have earned anything, and the rollout would score it."""
    env, _ = browser(monkeypatch, {"a": FakePage(steps_to_finish=2)})
    _, actions = env.reset(0)

    state, actions, done, reward = env.step(actions[0])
    assert (done, reward) == (False, 0.0) and actions
    state, actions, done, reward = env.step(actions[0])
    assert (done, reward) == (True, 1.0)
    assert actions == [], "no choice after the episode is over"
    assert env.ended == "done"


def test_one_task_per_episode_and_the_previous_page_is_closed(monkeypatch):
    """A browsergym env is built for a task id, so an episode is a new env. Not
    closing the last one leaks a chromium per episode."""
    pages = {"a": FakePage(), "b": FakePage()}
    env, made = browser(monkeypatch, pages)

    env.reset(0)
    env.reset(1)
    assert made == ["a", "b"]
    assert pages["a"].closed and not pages["b"].closed
    assert env.task == "b"
    assert env.episodes == 2
    with pytest.raises(ValueError, match="episode must be in"):
        env.reset(2)


def test_the_seed_is_the_instances_and_reaches_the_page(monkeypatch):
    """A second seed over the same tasks is a second environment, which is the
    discipline the gate's rows enforce; it keeps "episode 7" naming one thing."""
    pages = {"a": FakePage(), "b": FakePage()}
    env, _ = browser(monkeypatch, pages, seed=3)
    env.reset(0)
    env.reset(1)
    assert pages["a"].seeds == [3] and pages["b"].seeds == [3]


def test_dropped_candidates_are_counted_because_the_protocol_has_no_slot(monkeypatch):
    """`StepRecord.candidates` counts what was offered, which is capped at the
    budget by construction and so can never show what the budget cost. `dropped`
    is that quantity, and it accumulates over the episode."""
    env, _ = browser(monkeypatch, {"a": FakePage(steps_to_finish=3)}, limit=2)
    _, actions = env.reset(0)

    assert len(actions) == 2, "the budget caps what is offered"
    # One of the two slots is the scroll, so two of the three elements are cut.
    assert actions[-1] == "scroll(0, 400)"
    assert env.dropped_total == 2
    env.step(actions[0])
    assert env.dropped_total == 4, "it accumulates, per step"

    fresh, _ = browser(monkeypatch, {"a": FakePage()}, limit=52)
    fresh.reset(0)
    assert fresh.dropped_total == 0


def test_extra_actions_pass_through_to_the_candidate_set(monkeypatch):
    """The seam the enumerated text actions will arrive through: actions that
    belong to no element. `extra` is additional to the content-free ones rather
    than instead of them."""
    env, _ = browser(monkeypatch, {"a": FakePage()}, extra=("goto('http://x')",))
    _, actions = env.reset(0)

    assert "goto('http://x')" in actions
    assert "scroll(0, 400)" in actions, "and the content-free ones are still there"
    env.step("goto('http://x')")
    assert env.task == "a"

    off, _ = browser(monkeypatch, {"a": FakePage()}, content_free=False)
    _, plain = off.reset(0)
    assert all("click(" in action for action in plain), \
        "content_free=False is the only way to get the click-only set"


def stub_group(monkeypatch, tasks):
    """A venue whose group resolves to `tasks`, with no browsergym and no env."""
    import sys
    import types

    from slime_hdm.envs import browser_environment as module
    from slime_hdm.envs import gate

    monkeypatch.setattr(gate, "group_tasks", lambda venue, group: list(tasks))
    monkeypatch.setattr(gate, "missing_environment", lambda venue, environ=None: [])
    monkeypatch.setitem(sys.modules, "browsergym", types.ModuleType("browsergym"))
    monkeypatch.setitem(sys.modules, "browsergym.miniwob",
                        types.ModuleType("browsergym.miniwob"))
    return module


class Args:
    """What the RL loop passes: whatever flags happen to be set, and no others."""


def test_training_and_evaluation_get_different_tasks_on_defaults(monkeypatch):
    """The defect this replaced a docstring with: both offsets defaulted to 0,
    both counts to 0 meaning every task, both seeds to 0, so on defaults an RL run
    would have reported its training set as its evaluation. The docstring said
    they "cannot silently become the same episodes" and nothing enforced it."""
    module = stub_group(monkeypatch, [f"browsergym/miniwob.t{i:02d}" for i in range(10)])
    args = Args()

    training = module.build(args)
    evaluating = module.build(args, evaluation=True)

    assert training.tasks and evaluating.tasks
    assert not set(training.tasks) & set(evaluating.tasks), "disjoint on defaults"
    assert set(training.tasks) | set(evaluating.tasks) == {
        f"browsergym/miniwob.t{i:02d}" for i in range(10)}


def test_both_sides_span_the_group_rather_than_splitting_it_in_half(monkeypatch):
    """Task names cluster by family, so the first quarter would evaluate on
    whichever families sort first while training on the others."""
    from slime_hdm.envs.browser_environment import split_tasks

    level = [f"{family}-{index:02d}" for family in "abcde" for index in range(4)]
    training, evaluating = split_tasks(level)

    assert len({name.split("-")[0] for name in evaluating}) >= 4, "eval spans families"
    assert len({name.split("-")[0] for name in training}) == 5
    assert not set(training) & set(evaluating)


def test_a_group_too_small_to_split_is_refused(monkeypatch):
    from slime_hdm.envs.browser_environment import split_tasks

    with pytest.raises(ValueError, match="too small"):
        split_tasks(["one"])
    with pytest.raises(ValueError, match="fraction must be"):
        split_tasks([f"t{i}" for i in range(10)], fraction=0.0)


def test_overlap_takes_an_explicit_flag(monkeypatch):
    """A deliberate debug run can have it; a default run cannot get it by
    accident, which is the whole point."""
    module = stub_group(monkeypatch, [f"browsergym/miniwob.t{i:02d}" for i in range(10)])
    args = Args()
    args.hdm_allow_eval_overlap = True

    training = module.build(args)
    evaluating = module.build(args, evaluation=True)
    assert training.tasks == evaluating.tasks, "asked for, and granted"


def test_the_offered_action_set_is_signable_so_a_policy_change_invalidates_a_resume(
        monkeypatch):
    """exps found this gap. `collection_signature` carried the venue, the seed and
    the splits but **not the action set**, so making `content_free` the default
    changed the policy and left the signature byte-identical -- a resume would
    have read episodes collected by the sealed fast mode into a corpus the fixed
    one was extending, and printed nothing, because nothing differed.

    A set rather than the one flag: `content_free` in the dict fixes this
    instance; the next change to the fast mode would hit the same silence."""
    env, _ = browser(monkeypatch, {"a": FakePage()})
    sealed, _ = browser(monkeypatch, {"a": FakePage()}, content_free=False)

    assert env.offered_actions != sealed.offered_actions, \
        "the two fast modes do not share a signature"
    assert env.offered_actions["content_free"] == ["scroll(0, -400)", "scroll(0, 400)"]
    assert sealed.offered_actions["content_free"] == []

    # Both halves of the scroll pair, always: the up-scroll is only *offered*
    # after the page moves, so reading the live pair would make the signature
    # depend on when in the episode it was taken.
    before = env.offered_actions
    env.reset(0)
    env.step("scroll(0, 400)")
    assert env.offered_actions == before, "stable within an episode"

    # And the other things a policy may choose from are in it too.
    wider, _ = browser(monkeypatch, {"a": FakePage()}, limit=8,
                       extra=("goto('http://x')",))
    assert wider.offered_actions["limit"] == 8
    assert wider.offered_actions["extra"] == ["goto('http://x')"]


def test_the_cdpo_signature_carries_the_action_set():
    """The guard is only a guard if the run writes it."""
    from conftest import script_source

    source = script_source("browser_pmpo")
    assert '"offered_actions": environment(training, args.seed).offered_actions' \
        in source


def test_a_browser_that_dies_ends_the_episode_instead_of_the_run(monkeypatch):
    """code-reviewer named this as the case to watch, and it is cheaper to handle.

    28853's log showed `TargetClosedError` as **orphaned futures** after a kill,
    which do not propagate. The same error raised **from** `step` escapes
    `except StepTimeout`, exits 1 without growing the stack file -- and a
    launcher following the artefact rule reads exit 1 with no growth as "stop",
    ending a 48-hour run over one dead browser."""
    class DeadBrowser(FakePage):
        def step(self, action):
            raise _playwright_error("Target page, context or browser has been closed")

    env, _ = browser(monkeypatch, {"a": DeadBrowser()})
    _, actions = env.reset(0)
    state, offered, done, reward = env.step(actions[0])

    assert done is True and reward == 0.0
    assert offered == []
    assert env.ended == "browser_error:TargetClosedError", "named, not just dropped"
    assert env.complete is False, "so the collector drops and retries it"
    assert state, "and the row still says what was seen"


def test_our_own_bugs_still_propagate(monkeypatch):
    """The reason the check is by module and not `except Exception`: swallowing
    everything would report a bug in the candidate rule or the action parser as
    an environment failure, and a broken run would look like a flaky one."""
    class OurBug(FakePage):
        def step(self, action):
            raise KeyError("axtree_txt")

    env, _ = browser(monkeypatch, {"a": OurBug()})
    _, actions = env.reset(0)
    with pytest.raises(KeyError, match="axtree_txt"):
        env.step(actions[0])


def test_a_browser_that_dies_on_reset_is_handled_the_same_way(monkeypatch):
    class DeadOnReset(FakePage):
        def reset(self, seed=None):
            raise _playwright_error("Browser closed")

    env, _ = browser(monkeypatch, {"a": DeadOnReset()})
    state, actions = env.reset(0)
    assert actions == [] and state is not None
    assert env.ended == "browser_error:TargetClosedError"
    assert env.complete is False


def test_the_bid_marking_reaches_gym_make_and_the_signature(monkeypatch):
    """`tags_to_mark` decides which elements get a bid, so it is both an
    argument browsergym must receive and a field the signature must carry.

    browsergym's default, `standard_html`, marks only the 125 standard HTML
    tags, so an `<svg>`'s `<path>`, `<text>` and `<circle>` get no bid -- and an
    action names a bid, so **neither mode can touch them**. Measured at seed 0:
    `click-pie` 0 candidates -> 10, `ascending-numbers` 0 -> 5,
    `grid-coordinate` 0 -> 25, and on twelve HTML tasks nothing changes at all.

    In the signature for the same reason `content_free` is: a resume across this
    would read episodes collected by a policy that could not see four tasks into
    a corpus extending one that can, and nothing in the dict would differ.
    """
    page = FakePage()
    environment, _ = browser(monkeypatch, {"task": page})

    environment.reset(0)
    assert page.made_with.get("tags_to_mark") == "all", \
        "the default has to arrive at gym.make, not sit in our constructor"
    assert environment.offered_actions["tags_to_mark"] == "all"

    other = FakePage()
    conservative, _ = browser(monkeypatch, {"task": other},
                              tags_to_mark="standard_html")
    conservative.reset(0)
    assert other.made_with.get("tags_to_mark") == "standard_html"
    assert conservative.offered_actions["tags_to_mark"] == "standard_html"
    assert conservative.offered_actions != environment.offered_actions, \
        "two settings must not produce the same signature"
