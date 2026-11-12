"""The browser adapter: BrowserGym as `slime_htm.Environment`, for the RL loop.

The gate (`slime_htm.envs.gate`) has its own loop because it runs two constant
policies and trains nothing. This is the other half: the same venues, the same
candidate rule, behind the protocol the rollout drives. Without it the RL loop is
an ALFWorld loop -- `scripts/htm_rl_loop.py` imports `AlfworldEnvironment` by
name -- and the method is not about ALFWorld.

Three things are worth stating because they are not obvious from the protocol.

**The actions published are what the model reads, and stepping needs what the
page takes.** `candidates` returns both: `click('42')` for the environment and
`click('42')  -- Submit` for the prompt, because a bare browsing id is not a
choice anybody can make. The rollout hands back one of the strings it was given
(`actions[argmax]`), so this maps that string back to the action and **refuses an
action it never offered** rather than clicking whatever is nearest.

**One task per episode.** A browsergym environment is built for a task id, so
`reset(episode)` builds a new one and closes the last. The seed is fixed for the
instance: a second seed over the same tasks is a second environment, which is the
discipline the gate's rows already enforce, and it keeps "episode 7" naming one
thing.

**The dropped count has nowhere to go in the protocol**, and it is the quantity
that says whether the page offered more than the readout can encode. So it is
kept here and exposed, for the caller to put in `EpisodeResult.metadata`:
`StepRecord.candidates` counts what was offered, which is capped at the budget by
construction and so can never show the loss.
"""

from slime_htm.envs.browser import (
    ACTION_ARITY,
    StepTimeout,
    browser_failure,
    candidates,
    close_env,
    content_free_actions,
    parse_action,
    state_text,
    step_deadline,
)


# What the free-form slow mode is told it may write. The whitelist is
# `browser.ACTION_ARITY`; this is the half the model sees.
# **Built from `ACTION_ARITY`, not retyped beside it.** The two were separate
# strings and would drift: `drag_and_drop` went into the whitelist and this
# sentence would still have told the model it may not write one, so the model
# would never have tried -- which is indistinguishable from it being forbidden.
# That is not hypothetical. The gate's five drag tasks have the slow mode writing
# `send_msg_to_user('I cannot complete the drag because the triangles and black
# box are not exposed as actionable elements and no drag action is available.')`
# **48 times**: a correct reading of a sentence that was wrong.
def _slow_instructions():
    shapes = {"click": "click('bid')", "fill": "fill('bid', 'text')",
              "select_option": "select_option('bid', 'option')",
              "send_msg_to_user": "send_msg_to_user('answer')",
              "scroll": "scroll(x, y)", "goto": "goto('url')",
              "go_back": "go_back()", "hover": "hover('bid')",
              "press": "press('bid', 'key')",
              "drag_and_drop": "drag_and_drop('from_bid', 'to_bid')"}
    unnamed = sorted(set(ACTION_ARITY) - set(shapes))
    if unnamed:
        raise RuntimeError(
            f"{unnamed} are in ACTION_ARITY with no shape to show the model. A "
            f"whitelisted action the instructions do not name is one the model "
            f"will never write, which is not the same as one it may not write")
    listed = ", ".join(shapes[name] for name in ACTION_ARITY)
    return ("Decide the next action on this page and write it as a single call on "
            "its own line, nothing after it. You may write: " + listed + ". A bid "
            "is the number in brackets beside an element. Unlike the enumerated "
            "choice, you are not restricted to clicking.")


SLOW_INSTRUCTIONS = _slow_instructions()


class BrowserEnvironment:
    """One browsergym episode at a time, as `slime_htm.Environment`."""

    def __init__(self, tasks, *, seed=0, limit=52, state_chars=6000, headless=True,
                 history=6, extra=(), content_free=True, cycle=False,
                 step_timeout=120.0, stack_file=None, tags_to_mark="all"):
        if not tasks:
            raise ValueError("no tasks; a group that resolves to nothing is not an "
                             "environment")
        self._tasks = list(tasks)
        self._seed = seed
        self._limit = limit
        self._state_chars = state_chars
        self._headless = headless
        self._history = history
        self._extra = tuple(extra)
        # **On by default, and that default is the fix.** `candidates` filters to
        # the viewport, and its own docstring gives the reason: "in the window --
        # what is on screen, *since there is no scroll action*". So without the
        # content-free actions the fast mode is sealed into the first screen: it
        # cannot reach anything below the fold, and a page whose first screen has
        # no clickable element offers nothing at all, which ends the episode
        # before a single decision is taken.
        #
        # `gate.run_episode` passed `content_free_actions(scrolled)` and
        # `BrowserEnvironment` did not, and `browser_pmpo` builds its environments
        # here -- so 28845 ran a *weaker* fast mode than the gate that was its
        # precondition, and exps measured it as 13.8% of episodes dropped
        # `no_steps`. The drop was the visible part; the invisible part was every
        # other episode choosing from one screen. Third time a guarantee lived on
        # the gate's path and not on this one, so it defaults true rather than
        # being passed by the caller who remembers.
        self._content_free = bool(content_free)
        # **Which elements get a bid at all**, and browsergym's default excludes
        # the ones four tasks are made of. `standard_html` marks only the 125
        # standard HTML tags, so an `<svg>`'s `<path>`, `<text>` and `<circle>`
        # never receive a bid -- and an element without a bid cannot be clicked
        # by either mode, because every action names a bid. Measured at
        # `seed=0`, first step:
        #
        #                      standard_html        all
        #     click-pie            0 候选        10 候选, 5 带标签
        #     ascending-numbers    0            5
        #     grid-coordinate      0           25, 25 带标签
        #     click-pie-nodelay    0           10
        #
        # And the labels that arrive are the ones the goal names: click-pie's
        # `<text>` elements carry 'd', 'Q', 'M', 'N' for "click on the item
        # labeled M", and ascending-numbers' carry '1'..'5'.
        #
        # **It costs nothing where it changes nothing.** On twelve HTML tasks --
        # including the three busiest, `order-food` (137 bids), `social-media`
        # (133) and `number-checkboxes` (78) -- the bid count, the candidate
        # count and the labelled count are all byte-identical under the two
        # settings, because those pages contain no non-standard element. The
        # largest candidate set in the venue stays 29, well inside the 52 slots.
        #
        # **Unmeasured on WebArena**, where real sites carry web components and
        # SVG icons that `all` would also mark. That venue is not running, so
        # this has to be measured there before a run, not assumed from here.
        self._tags_to_mark = tags_to_mark
        self._scrolled = False
        # Cycling is for **collection**: a MiniWoB task is parameterised by its
        # seed, so a second pass over the same tasks at seed+1 is new data rather
        # than a replay, and asking for 100 episodes from 40 tasks is a reasonable
        # thing to want. It is **off** for evaluation, where a second pass would
        # weight some tasks twice in the mean and the count must be exactly one
        # pass. The first version had neither and simply raised past the end --
        # round 0 died at episode 40 before any head was trained.
        self._cycle = bool(cycle)
        # The same two-layer guard the gate has, because this is the path the CDPO
        # run uses and the gate's is not on it. 28832's hang was
        # `env.step` -> `post_step` -> `context.cookies()`, which is the call the
        # rollout makes **every step**: without this, the first hang in a
        # 540-episode run freezes it until the time limit.
        self._step_timeout = step_timeout
        self._stack_file = stack_file
        self._complete = True
        self._env = None
        self._observation = None
        self._goal = ""
        self._trail = []
        self._offered = {}
        self._dropped_total = 0
        self._written_refusals = 0
        self._ended = None
        self._task = None

    @property
    def episodes(self):
        """One pass over the tasks. A cycling environment can serve more than this."""
        return len(self._tasks)

    @property
    def bids(self):
        """Every element id this page has, which is what a written action is
        checked against -- not just the clickable ones the fast mode was offered,
        because the point of writing an action is to reach the others."""
        return set((self._observation or {}).get("extra_element_properties", {}))

    def slow_instructions(self):
        """What the free-form slow mode is told it may write, for this step."""
        return SLOW_INSTRUCTIONS

    def canonical_action(self, text):
        """What `text` would execute as, or why it is not an action. Executes nothing.

        `act_freely` parses and then acts; this is the parse alone, so a probe can
        learn what the slow mode chose without taking its step. It matters because
        the written form and the enumerated form of the same action are different
        strings -- a model writing `click("42")` or backticking it means the same
        click as the candidate `click('42')` -- and comparing them raw can only
        ever manufacture disagreement, never agreement. exps found that the
        agreement rate was doing exactly that, with the bias all one way.
        """
        return parse_action(text, self.bids)

    def act_freely(self, text):
        """Execute what the slow mode wrote, or report why it is not an action.

        Returns `(result, problem)` where `result` is what `step` returns. The
        refusal is not a click: an unexecutable action is a fact about the step,
        and substituting the nearest element would record a step the model did
        not choose and credit its outcome to thinking.
        """
        action, problem = parse_action(text, self.bids)
        if problem is not None:
            self._written_refusals += 1
            return None, problem
        # Published or not, it is a real action on this page: that is the whole
        # point of the free-form path, so the map is extended rather than checked.
        self._offered.setdefault(action, action)
        return self.step(action), None

    @property
    def written_refusals(self):
        """How many written actions this episode could not execute."""
        return self._written_refusals

    @property
    def tasks(self):
        """Exactly which tasks this environment serves, for the run's record."""
        return list(self._tasks)

    @property
    def task(self):
        """The task the current episode is playing."""
        return self._task

    @property
    def dropped_total(self):
        """Candidates the budget could not carry, this episode. See the module note."""
        return self._dropped_total

    @property
    def complete(self):
        """Whether the last episode produced an outcome the model is answerable for.

        False after a step that did not return: the collector must not keep such
        an episode, since training on it would score a browser that stopped
        answering as a failure the policy caused.
        """
        return self._complete

    @property
    def ended(self):
        """Why the last episode stopped: done, no_candidates, or None while running."""
        return self._ended

    def reset(self, episode):
        import gymnasium as gym

        if episode < 0 or (not self._cycle and episode >= len(self._tasks)):
            raise ValueError(
                f"episode must be in [0, {len(self._tasks)}), got {episode}. This "
                f"environment does not cycle, which is deliberate for evaluation: a "
                f"second pass would weight some tasks twice in the mean")
        self.close()
        position, pass_number = divmod(episode, len(self._tasks))[::-1]
        self._task = self._tasks[position]
        self._complete = True
        # A new seed per pass, so the second time round a task draws a different
        # instance of it rather than replaying the first.
        self._env = gym.make(self._task, headless=self._headless,
                             wait_for_user_message=False,
                             tags_to_mark=self._tags_to_mark)
        try:
            with step_deadline(self._step_timeout, stack_file=self._stack_file):
                self._observation, _ = self._env.reset(seed=self._seed + pass_number)
        except StepTimeout:
            self._ended, self._complete = "reset_timeout", False
            self._observation = {}
            self._offered = {}
            return self._state(), []
        except Exception as problem:  # noqa: BLE001  re-raised unless it is the browser
            if not browser_failure(problem):
                raise
            # A browser that died on reset is not the policy's failure and not a
            # reason to end the run: the episode is dropped and retried like a
            # timeout. Our own bugs still propagate, because `browser_failure`
            # asks which module raised.
            self._ended = f"browser_error:{type(problem).__name__}"
            self._complete = False
            self._observation, self._offered = {}, {}
            return self._state(), []
        self._goal = self._observation.get("goal", "")
        self._trail = []
        # Scrolling up is useless before the page has moved and costs one of the
        # 52 candidate slots, so it is offered only after a scroll -- per
        # episode, which is why this resets here.
        self._scrolled = False
        self._dropped_total = 0
        self._written_refusals = 0
        self._ended = None
        return self._publish()

    def step(self, action):
        if self._env is None:
            raise RuntimeError("step before reset")
        chosen = self._offered.get(action)
        if chosen is None:
            # Never guess. A published action that comes back unrecognised means the
            # caller is not choosing from what it was given, and clicking the
            # nearest element would hide that behind a plausible episode.
            raise ValueError(
                f"{action!r} is not one of the {len(self._offered)} actions this step "
                f"offered; the rollout must return a string it was handed")
        self._scrolled = self._scrolled or chosen.startswith("scroll(")
        try:
            with step_deadline(self._step_timeout, stack_file=self._stack_file):
                self._observation, reward, terminated, truncated, _ = self._env.step(
                    chosen)
        except StepTimeout:
            # The episode ends here and is **not** the policy's failure: a browser
            # that stopped answering is not a decision. `complete` is False so the
            # collector drops it rather than training on a manufactured zero.
            self._ended, self._complete = "step_timeout", False
            self._offered = {}
            return self._state(), [], True, 0.0
        except Exception as problem:  # noqa: BLE001  re-raised unless it is the browser
            if not browser_failure(problem):
                raise
            # A browser that **died** rather than hung: `TargetClosedError` and
            # its kin come out of `step`, not as the orphaned futures 28853's log
            # showed after a kill. Escaping here would exit 1 without growing the
            # stack file, which a launcher following the artefact rule reads as
            # "stop" -- ending a 48-hour run over one dead browser. Same handling
            # as a timeout: dropped, retried, and the reason recorded with the
            # exception's name so the rows say which failure it was.
            self._ended = f"browser_error:{type(problem).__name__}"
            self._complete = False
            self._offered = {}
            return self._state(), [], True, 0.0
        self._trail.append(chosen)
        done = bool(terminated or truncated)
        if done:
            self._ended = "done" if terminated else "truncated"
            self._offered = {}
            # No actions when the episode is over: the rollout stops on an empty
            # list, and offering a choice after the reward has arrived would add a
            # step that cannot have earned anything.
            return self._state(), [], True, float(reward or 0.0)
        state, actions = self._publish()
        return state, actions, False, float(reward or 0.0)

    def close(self):
        """Close under the same guard, and kill what will not close.

        `close_env` handles the deadline, the kill and dropping browsergym's
        cached driver; a close blocked in the greenlet bridge is the likeliest
        *second* hang, so it gets the stack file too.
        """
        if self._env is not None:
            trouble = close_env(self._env, self._step_timeout,
                                stack_file=self._stack_file)
            if trouble:
                print(f"  {self._task}: {trouble}", flush=True)
            self._env = None

    @property
    def offered_actions(self):
        """What this environment lets a policy choose from, for a run's signature.

        exps found the gap this closes. `collection_signature` carried the venue,
        the group, the seed, the budget and the splits -- but **not the action
        set**, so making `content_free` the default changed the policy and left
        the signature byte-identical. A resume would have read episodes collected
        by the sealed fast mode back into a corpus the fixed one was extending,
        and printed nothing, because nothing in the dict differed. That is the
        exact failure the signature exists to prevent.

        A property rather than the one flag: `content_free` in the dict would fix
        this instance, and the next change to the fast mode -- `extra`, the
        candidate limit, option (b)'s text actions -- would hit the same silence.
        This changes whenever what a policy may choose changes.
        """
        return {
            "limit": self._limit,
            "extra": sorted(self._extra),
            # Both, not `content_free_actions(self._scrolled)`: the scrolled half
            # varies within an episode and would make the signature depend on when
            # it was read.
            "content_free": sorted(content_free_actions(True)) if self._content_free
                            else [],
            # In the dict because it decides which elements exist to be chosen,
            # which is the same kind of change `content_free` was: a resume
            # across it would read episodes collected by a policy that could not
            # see the SVG tasks into a corpus extending one that can. The
            # viewport fix did not need this -- it changed the filter, not the
            # set of actions the environment names -- and `commit_now()` was
            # what guarded it. This one belongs here.
            "tags_to_mark": self._tags_to_mark,
        }

    def _publish(self):
        """The state text and the actions, from the one shared candidate rule."""
        extra = self._extra
        if self._content_free:
            extra = extra + content_free_actions(self._scrolled)
        offered, dropped = candidates(self._env, self._observation, self._limit,
                                      extra=extra)
        self._dropped_total += dropped
        self._offered = {}
        for action, shown in offered:
            # First occurrence wins. Browsing ids are unique, so two identical shown
            # strings would mean the page changed under us rather than a real tie.
            self._offered.setdefault(shown, action)
        if not offered:
            self._ended = "no_candidates"
        return self._state(), list(self._offered)

    def _state(self):
        return state_text(self._observation or {}, self._goal, self._trail,
                          history=self._history, chars=self._state_chars)


EVAL_FRACTION = 0.25


def split_tasks(found, fraction=EVAL_FRACTION):
    """Partition a group into training and evaluation tasks, disjointly.

    Every kth task goes to evaluation, so both sides span the group: task names
    cluster by family, and taking the first quarter would evaluate on whichever
    families sort first while training on the others.

    This exists because the alternative was a docstring. `build` used to say that
    training and evaluation "take different task slices of the same group, so the
    two cannot silently become the same episodes", and nothing enforced it: both
    offsets defaulted to 0, both counts to 0 meaning every task, both seeds to 0,
    so on defaults an RL run would have reported its training set as its
    evaluation. code-reviewer reproduced it with empty args.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    step = max(2, round(1 / fraction))
    evaluation = [task for index, task in enumerate(found) if index % step == 0]
    training = [task for index, task in enumerate(found) if index % step != 0]
    if not evaluation or not training:
        raise ValueError(
            f"a group of {len(found)} task(s) cannot be split at {fraction}; it is "
            f"too small to train and evaluate on different tasks")
    return training, evaluation


def build(args, evaluation=False):
    """The factory `--htm-environment-path slime_htm.envs.browser_environment:build` names.

    Training and evaluation get disjoint task sets, enforced here rather than
    asserted in a comment: the two lists are built together and their intersection
    is checked. `--htm-allow-eval-overlap` is the escape hatch for a deliberate
    debug run and has to be passed on purpose.
    """
    from importlib import import_module

    from slime_htm.envs.gate import VENUES, chosen_tasks, group_tasks, missing_environment

    venue = getattr(args, "htm_venue", "miniwob")
    group = getattr(args, "htm_group", "click")
    absent = missing_environment(venue)
    if absent:
        raise RuntimeError(f"{venue} needs {', '.join(absent)} in the environment")
    import_module(VENUES[venue]["registers"])

    found = group_tasks(venue, group)
    overlap_allowed = bool(getattr(args, "htm_allow_eval_overlap", False))
    if overlap_allowed:
        training = evaluating = found
    else:
        training, evaluating = split_tasks(
            found, getattr(args, "htm_eval_fraction", EVAL_FRACTION))

    side = evaluating if evaluation else training
    count = (getattr(args, "htm_eval_games", 0) if evaluation
             else getattr(args, "htm_games", 0))
    offset = (getattr(args, "htm_eval_offset", 0) if evaluation
              else getattr(args, "htm_offset", 0))
    stride = getattr(args, "htm_task_stride", 0)
    tasks = chosen_tasks(side[offset:], count, stride)
    if not overlap_allowed:
        # The partition makes this impossible; the check is here because the
        # property is what matters and a future offset or stride could break it.
        other = chosen_tasks(
            (training if evaluation else evaluating)[
                getattr(args, "htm_offset", 0) if evaluation
                else getattr(args, "htm_eval_offset", 0):],
            getattr(args, "htm_games", 0) if evaluation
            else getattr(args, "htm_eval_games", 0),
            stride)
        shared = sorted(set(tasks) & set(other))
        if shared:
            raise ValueError(
                f"{len(shared)} task(s) are in both the training and evaluation sets, "
                f"e.g. {shared[0]}; an RL run would report its training set as its "
                f"evaluation. Pass --htm-allow-eval-overlap only to debug")
    return BrowserEnvironment(
        tasks,
        seed=getattr(args, "htm_eval_seed", 0) if evaluation else getattr(args, "htm_seed", 0),
        limit=getattr(args, "htm_candidates", 52),
        state_chars=getattr(args, "htm_state_chars", 6000),
        headless=not getattr(args, "htm_no_headless", False),
        history=getattr(args, "htm_history", 6),
    )
