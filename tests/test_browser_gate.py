"""The browser gate's logic, without a browser, a model or any server.

What can be wrong without any of those is all of it except the stepping: whether
the two policies are paired on the same task, whether the four-way split follows
from the pairing, whether a truncated candidate set is recorded as such, and
whether the verdict is the one the thresholds say rather than one a reader picks
afterwards. ALFWorld is why the verdict is in the code: that venue's numbers were
read as a disappointment about the method for two rounds of RL before anybody said
they were a fact about the venue.
"""
import json
from pathlib import Path

import pytest

from slime_htm.envs.gate import PASS, four_way, run_episode, step_record, verdict


def episode(success, tokens=0, seconds=1.0, dropped=0, outside=False):
    """An episode as `four_way` requires it, including `used_outside_action`.

    That key used to be absent here and read with `.get()` in production, which
    is the shape that let 28832 report a reasoning effect it never measured: its
    resumed pairs came back from rows that `episode_record` had not serialised
    the key into, `.get()` returned `None`, and `not None` counted each of those
    helps as reasoning. Production now subscripts it, so a fixture missing it
    fails here rather than producing a plausible wrong split.
    """
    return {"success": success, "tokens": tokens, "seconds": seconds,
            "dropped_total": dropped, "used_outside_action": outside}


def test_the_split_comes_from_the_pairing_and_not_from_two_accuracies():
    """Two runs at 0.5 can mean every task is answered by exactly one mode or by
    both on half of them. A router has everything to do in the first case and
    nothing in the second, and only the pairing tells them apart."""
    disjoint = [(episode(True), episode(False)), (episode(False), episode(True))]
    overlapping = [(episode(True), episode(True)), (episode(False), episode(False))]

    one, two = four_way(disjoint), four_way(overlapping)
    assert one["never"] == two["never"] == 0.5
    assert one["always"] == two["always"] == 0.5
    assert one["both_right"] == 0.0 and one["oracle"] == 1.0
    assert two["both_right"] == 0.5 and two["oracle"] == 0.5


def test_the_oracle_is_the_better_of_the_two_modes_per_task():
    pairs = [(episode(True), episode(False)), (episode(False), episode(True)),
             (episode(False), episode(False))]
    split = four_way(pairs)
    assert split["oracle"] == pytest.approx(2 / 3)
    assert split["helps"] == pytest.approx(1 / 3)
    assert split["hurts"] == pytest.approx(1 / 3)
    assert split["both_wrong"] == pytest.approx(1 / 3)


def test_alfworlds_shape_fails_the_gate_and_says_which_shape_it_is():
    """always 0.575 against never 0.619 is the venue this gate exists to refuse.
    It cost two rounds of RL to learn that after the router was built.

    `pass` is tri-state since MIN_PAIRS, and `not None` is True, so every failure
    here is asserted as `is False`: otherwise a fixture that shrank below twenty
    pairs would pass this test by returning no verdict at all.
    """
    pairs = ([(episode(True), episode(False))] * 5
             + [(episode(True), episode(True))] * 5
             + [(episode(False), episode(False))] * 2)
    call = verdict(four_way(pairs), min_pairs=1)  # the rule, not the run size
    assert call["pass"] is False
    assert any("constant one" in reason for reason in call["reasons"])


def test_a_level_beyond_the_fast_mode_fails_differently_from_a_useless_one():
    """Both near zero is not "thinking has no value"; it is "there is no gap to
    route". WorkArena++ reports frontier models doing poorly on L2 and L3, so this
    is the failure this venue is most likely to produce."""
    floor = verdict(four_way([(episode(False), episode(False))] * 20))
    assert floor["pass"] is False
    assert any("beyond the fast mode" in reason for reason in floor["reasons"])

    ceiling = verdict(four_way([(episode(True), episode(True))] * 20))
    assert ceiling["pass"] is False
    assert any("without thinking" in reason for reason in ceiling["reasons"])


def test_scienceworlds_shape_passes():
    """The venue that was worth running: 64 episodes, thinking rescuing 15 and
    spoiling 5, and most episodes unchanged by it. Its own score was a progress
    measure in [0, 1] rather than a success bit, so this is its shape and not its
    numbers. The oracle margin there was 0.034, which is why that is reported and
    is not a pass condition."""
    pairs = ([(episode(True), episode(True))] * 25
             + [(episode(False), episode(True))] * 15
             + [(episode(True), episode(False))] * 5
             + [(episode(False), episode(False))] * 19)
    split = four_way(pairs)
    call = verdict(split)
    assert call["pass"], call["reasons"]
    assert split["always"] > split["never"]
    assert call["oracle_over_always"] < 0.2, "a small oracle margin does not fail"


def test_a_truncated_candidate_set_is_recorded_rather_than_swallowed():
    """The readout needs single-token codes and this tokenizer supplies 52, so a
    busy page is cut. An episode that failed because the right element was dropped
    is not an episode the model got wrong, and nothing else in the record says so."""
    offered = [("click(\'1\')", "click -- Submit"), ("click(\'2\')", "click -- Cancel")]
    row = step_record("t", "never", 0, offered, 7, 1, [0.1, 0.9], 0.5, 0, False)
    assert row["candidates"] == 2 and row["dropped"] == 7
    assert row["chosen_index"] == 1 and row["chosen_action"] == "click(\'2\')"
    assert row["probabilities"] == [0.1, 0.9], "the whole distribution, for position_bias"


def test_the_gate_refuses_a_policy_it_cannot_run():
    """It has no head and cannot route. A third policy name is a mistake, not a
    default that silently becomes one of the two."""
    with pytest.raises(ValueError, match="never and always"):
        run_episode(None, None, "t", "routed", budget=8, max_steps=1, limit=52,
                    seed=0, state_chars=100)


def test_the_thresholds_are_fixed_in_the_code_and_not_passed_in_at_report_time():
    """A gate whose pass condition is chosen after the numbers are in is not a
    gate. The defaults live here; `thresholds` exists so a test can state a
    different rule, not so a run can."""
    assert PASS == {"always_over_never": 0.05, "both_right_min": 0.20,
                    "spread_min": 0.05, "sign_test_p": 0.10}
    pairs = [(episode(True), episode(True))] * 5 + [(episode(False), episode(True))] * 5
    # min_pairs=1 because this is about the rule and not about how many tasks a
    # real group needs: at ten pairs there is no verdict by default.
    assert verdict(four_way(pairs), min_pairs=1)["pass"] is True
    strict = verdict(four_way(pairs), {"always_over_never": 0.9}, min_pairs=1)
    assert strict["pass"] is False


def test_an_accuracy_margin_of_one_task_does_not_pass_on_its_own():
    """At twenty tasks a margin of 0.05 is a single rescued task. The sign test is
    what refuses it, and it refuses it because one discordant pair out of one is
    p = 0.5 -- not because the margin is small, which it is not."""
    one = ([(episode(True), episode(True))] * 10
           + [(episode(False), episode(True))] * 1
           + [(episode(False), episode(False))] * 9)
    split = four_way(one)
    assert split["always"] - split["never"] == pytest.approx(0.05), "the margin passes"
    call = verdict(split)
    assert call["pass"] is False
    assert call["sign_test_p"] == pytest.approx(0.5)
    assert any("sign test" in reason for reason in call["reasons"])


def test_the_rescued_count_is_the_accuracy_margin_and_not_a_second_condition():
    """The first version of the rule counted `helps - hurts` and called it a paired
    test. It is `(always - never) * n` exactly, so it restated the margin. This is
    the identity, in the code, so the mistake cannot come back quietly."""
    pairs = ([(episode(True), episode(True))] * 6
             + [(episode(False), episode(True))] * 9
             + [(episode(True), episode(False))] * 4
             + [(episode(False), episode(False))] * 1)
    split = four_way(pairs)
    rescued = (split["helps"] - split["hurts"]) * split["n"]
    assert rescued == pytest.approx((split["always"] - split["never"]) * split["n"])


def test_the_sign_test_reads_the_split_and_not_the_difference():
    """Which is the whole reason it says something the margin cannot. 23 against 20
    has a difference of three and is nothing; 4 against 0 has a difference of four
    and is worth having. A difference-of-three rule would have taken the first."""
    from slime_htm.envs.gate import sign_test

    assert sign_test(3, 0) == pytest.approx(0.125), "not 0.11, which the plan claimed"
    assert sign_test(4, 0) == pytest.approx(0.0625)
    assert sign_test(15, 5) == pytest.approx(0.0207, abs=1e-4)
    assert sign_test(23, 20) == pytest.approx(0.3804, abs=1e-4)
    assert sign_test(0, 0) == 1.0, "no disagreement is no evidence"

    thin = ([(episode(True), episode(True))] * 7
            + [(episode(False), episode(True))] * 23
            + [(episode(True), episode(False))] * 20)
    call = verdict(four_way(thin))
    assert call["pass"] is False, "a difference of three over 50 tasks is not an effect"
    assert any("sign test" in reason for reason in call["reasons"])


def test_the_row_id_says_what_kind_of_row_it_is():
    """Without the kind, a step row at index 0 and an episode row at seed 0 are both
    `task|policy|0`, and `streaming.completed` keys on the id -- so one would have
    silently shadowed the other in every resumed run."""
    from slime_htm.envs.gate import episode_record

    step = step_record("t", "never", 0, [("click", "click")], 0, 0, [1.0], 0.1, 0, False)
    outcome = episode_record({"task": "t", "policy": "never", "seed": 0, "reward": 1.0,
                              "success": True, "steps": 1, "done": True,
                              "truncated": False, "tokens": 0, "seconds": 1.0,
                              "dropped_total": 0, "ended": "done",
                              "used_outside_action": False, "unexecutable": 0,
                              "complete": True})
    assert step["id"] != outcome["id"]
    assert step["kind"] == "step" and outcome["kind"] == "episode"


def test_a_resumed_run_rebuilds_its_pairs_from_the_rows_and_not_from_memory():
    """Otherwise a resume reports a split over the tasks this process happened to
    reach and calls it the level. The outcome rows are what make that possible, and
    they were not being written at all."""
    from slime_htm.envs.gate import episode_record, rebuild_pairs

    def row(task, policy, success, outside=False):
        return episode_record({"task": task, "policy": policy, "seed": 0,
                               "reward": 1.0 if success else 0.0, "success": success,
                               "steps": 2, "done": True, "truncated": False,
                               "tokens": 0, "seconds": 1.0, "dropped_total": 0,
                               "ended": "done", "used_outside_action": outside,
                               "unexecutable": 0, "complete": True})

    done = {}
    for one in (row("a", "never", True), row("a", "always", True),
                row("b", "never", False)):
        done[one["id"]] = one
    pairs, missing = rebuild_pairs(done, ["a", "b", "c"])
    assert len(pairs) == 1, "only the task with both policies is a pair"
    assert missing == [("b", ["always"]), ("c", ["always", "never"])]
    assert four_way(pairs)["both_right"] == 1.0

    # **`used_outside_action` has to survive the round trip, or a resume
    # mislabels the split.** This is the defect that produced 28832's reasoning
    # effect: `episode_record` did not serialise the key, `rebuild_pairs` reads
    # these rows, and `.get()` turned the absence into "not outside" -- which
    # `four_way` counts as `helps_by_reasoning`. The rows file said 14 reach and
    # the JSON said 12 reach / 2 reasoning for the same seed.
    rescued = {}
    for one in (row("d", "never", False), row("d", "always", True, outside=True)):
        rescued[one["id"]] = one
    reach_pairs, _ = rebuild_pairs(rescued, ["d"])
    split = four_way(reach_pairs)
    assert split["helps_by_reach"] == 1.0 and split["helps_by_reasoning"] == 0.0, \
        "a rescue that acted outside the candidate set is reach, after a resume too"

    inside = {}
    for one in (row("e", "never", False), row("e", "always", True, outside=False)):
        inside[one["id"]] = one
    reasoning_pairs, _ = rebuild_pairs(inside, ["e"])
    other = four_way(reasoning_pairs)
    assert other["helps_by_reasoning"] == 1.0 and other["helps_by_reach"] == 0.0


def test_a_re_run_episode_does_not_collide_with_the_one_it_replaces():
    """A kill mid-episode leaves step rows. The next attempt writes its own, and
    without an attempt number the two interleave under the same ids."""
    from slime_htm.envs.gate import attempts_so_far

    first = step_record("t", "never", 0, [("click", "click")], 0, 0, [1.0], 0.1, 0, False)
    done = {first["id"]: first}
    assert attempts_so_far(done, "t", "never") == 1
    assert attempts_so_far(done, "t", "always") == 0
    second = step_record("t", "never", 0, [("click", "click")], 0, 0, [1.0], 0.1, 0,
                         False, attempt=1)
    assert second["id"] != first["id"]


def test_the_tasks_measured_are_a_stride_through_the_level_not_its_first_names():
    """`registered` sorts, and WorkArena's names group by family, so the first
    twenty are a few families rather than a sample of the level -- a verdict from
    them is a verdict about whichever families sort first."""
    from slime_htm.envs.gate import chosen_tasks

    level = [f"{family}.{index:02d}" for family in "abcde" for index in range(10)]
    first = level[:10]
    strided = chosen_tasks(level, 10, 0)
    assert len({name.split(".")[0] for name in first}) == 1, "the first ten are one family"
    assert len({name.split(".")[0] for name in strided}) == 5, "the stride spans them"
    assert len(strided) == 10
    assert chosen_tasks(level, 0, 0) == level, "zero means the whole level"
    assert chosen_tasks(level, 99, 0) == level
    assert chosen_tasks(level, 10, 0) == chosen_tasks(level, 10, 0), "deterministic"


def test_the_image_writes_a_revision_the_provenance_reader_can_parse(tmp_path):
    """The image copies no `.git`, so `REVISION` is the only thing that can say
    which code ran, and `commit_in` reads only a line that starts with `commit `.
    A bare sha -- the obvious way to write this file -- reads as no record at all,
    which would label every row `unknown` after a clean-looking build. So the test
    takes the format from the Dockerfile itself rather than restating it."""
    import re
    from pathlib import Path

    from litjev.run.provenance import commit_in

    dockerfile = Path(__file__).resolve().parents[1] / "src/slime_htm/envs/Dockerfile"
    text = dockerfile.read_text()
    written = re.search(r"RUN printf '([^']+)' \"\$GIT_COMMIT\" > /src/REVISION", text)
    assert written, "the Dockerfile no longer writes REVISION the way this test reads it"

    sha = "0123456789abcdef0123456789abcdef01234567"
    revision = tmp_path / "REVISION"
    revision.write_text(written.group(1).replace("\\n", "\n") % sha)
    assert commit_in(revision) == sha

    bare = tmp_path / "bare" / "REVISION"
    bare.parent.mkdir()
    bare.write_text(sha + "\n")
    assert commit_in(bare) is None, "a bare sha is not a record; hence the prefix"

    assert "ARG GIT_COMMIT" in text
    assert 'test -n "$GIT_COMMIT"' in text, "an empty build arg must fail the build"


def test_the_gate_refuses_to_run_when_nothing_says_which_commit_it_is():
    """A verdict against pre-registered thresholds is worth nothing if it cannot
    be tied to a reviewed commit, and `commit_now()` returns 'unknown' rather than
    raising -- so the refusal has to be here, before the browser starts."""
    from litjev.run.provenance import UNKNOWN
    from slime_htm.envs.gate import require_commit

    sha = "0123456789abcdef0123456789abcdef01234567"
    assert require_commit(sha) == sha
    with pytest.raises(SystemExit) as refused:
        require_commit(UNKNOWN)
    assert "REVISION" in str(refused.value)


def test_each_venue_records_the_packages_that_decide_what_a_task_name_means():
    """A venue's own package carries its task registry, so the same name in two
    releases is two different tasks. `describe_environment`'s standing list is
    about the model's numerics and covers none of them, and each venue's packages
    must be pinned in the image and recorded in the header -- both, or a run
    cannot be compared with the next one."""
    from pathlib import Path

    from litjev.run.provenance import RECORDED, describe_environment
    from slime_htm.envs.gate import VENUES

    dockerfile = Path(__file__).resolve().parents[1] / "src/slime_htm/envs/Dockerfile"
    text = dockerfile.read_text()
    for venue, spec in VENUES.items():
        packages = spec["packages"]
        facts = describe_environment(extra=packages)
        for name in (*RECORDED, *packages):
            assert name in facts, f"{venue}: {name} is not recorded"
        assert len(facts) == len({*RECORDED, *packages}) + 1, "no name recorded twice"
        venue_package = f"browsergym-{venue}"
        assert venue_package in packages, f"{venue} does not record its own package"
        assert f"{venue_package}==" in text, f"{venue_package} must be pinned in the image"
    assert "browsergym-core==" in text, "the core they all share must be pinned too"


def workarena_registry():
    """The id shape `browsergym-workarena==0.5.3` actually registers.

    Measured, not invented: 716 ids, all under one `browsergym/workarena.servicenow.`
    prefix, of which 341 end `-l2`, 341 end `-l3`, 33 are the atomic L1 list, and
    one -- `update-private` -- is registered and unsuffixed but is not an L1 task.
    """
    atomic = [f"workarena.servicenow.atomic-{i:02d}" for i in range(33)]
    comp = [f"workarena.servicenow.comp-{i:03d}-{level}"
            for level in ("l2", "l3") for i in range(341)]
    return atomic, comp + atomic + ["workarena.servicenow.update-private"]


def fake_workarena(monkeypatch):
    import sys
    import types

    atomic, every = workarena_registry()
    package = types.ModuleType("browsergym")
    module = types.ModuleType("browsergym.workarena")
    module.workarena_tasks_l1 = atomic
    module.workarena_tasks_all = every
    package.workarena = module
    gym = types.ModuleType("gymnasium")
    gym.registry = {f"browsergym/{name}": object() for name in every}
    monkeypatch.setitem(sys.modules, "browsergym", package)
    monkeypatch.setitem(sys.modules, "browsergym.workarena", module)
    monkeypatch.setitem(sys.modules, "gymnasium", gym)
    return gym


def test_a_level_cannot_be_selected_by_a_name_prefix(monkeypatch):
    """The reason the CLI no longer takes one. WorkArena puts the level in a
    *suffix*: every id shares `browsergym/workarena.servicenow.`, so one prefix
    matches all 716 tasks and no per-level prefix exists. The old `--prefixes`
    would have run L1's thresholds over the whole benchmark, three times."""
    from slime_htm.envs.gate import group_tasks

    gym = fake_workarena(monkeypatch)
    shared = "browsergym/workarena.servicenow."
    assert all(task.startswith(shared) for task in gym.registry)
    by_prefix = [task for task in gym.registry if task.startswith(shared)]
    assert len(by_prefix) == 716, "one prefix, the whole benchmark"

    assert len(group_tasks('workarena', "l1")) == 33
    assert len(group_tasks('workarena', "l2")) == len(group_tasks('workarena', "l3")) == 341
    assert sum(len(group_tasks('workarena', level)) for level in ("l1", "l2", "l3")) == 715, \
        "715, not 716: the unsuffixed building block is in no level"


def test_l1_is_the_packages_own_list_and_not_the_tasks_without_a_suffix(monkeypatch):
    """`update-private` is registered, has no level suffix, and is not an L1
    benchmark task. Deriving L1 from 'no suffix' would quietly add it, and the
    gate would report a level the package does not define."""
    from slime_htm.envs.gate import group_tasks

    fake_workarena(monkeypatch)
    l1 = group_tasks('workarena', "l1")
    assert "browsergym/workarena.servicenow.update-private" not in l1
    assert all(not task.endswith(("-l2", "-l3")) for task in l1)
    assert l1 == sorted(l1)


def test_an_id_the_package_names_but_does_not_register_is_an_error(monkeypatch):
    """`browsergym/` + name is an assumption about the package's id scheme. If it
    changes, the run must die here and not at the first `gym.make`."""
    import sys

    from slime_htm.envs.gate import group_tasks

    fake_workarena(monkeypatch)
    sys.modules["gymnasium"].registry.pop("browsergym/workarena.servicenow.atomic-00")
    with pytest.raises(RuntimeError, match="id scheme has changed"):
        group_tasks('workarena', "l1")


def test_an_unknown_level_is_refused_before_the_venue_is_imported():
    """So a typo costs nothing, and the error is the same on a machine with no
    browsergym installed as on the one that runs the gate."""
    from slime_htm.envs.gate import VENUES, group_tasks

    assert VENUES["workarena"]["groups"] == ("l1", "l2", "l3")
    with pytest.raises(ValueError, match="l4"):
        group_tasks('workarena', "l4")


def gate_output(seed, outcomes, level="l1", **header):
    """A gate result file: `outcomes` maps a task to (never_success, always_success)."""
    episodes = []
    for task, (never, always) in outcomes.items():
        for policy, success in (("never", never), ("always", always)):
            episodes.append({**episode(success), "task": task, "policy": policy,
                             "seed": seed})
    return {"header": {"seed": seed, "venue": "workarena", "groups": [level], "budget": 512,
                       "max_steps": 25, **header},
            "tasks": {level: sorted(outcomes)},
            "episodes": episodes,
            # The per-group splits the run wrote; `combine_results` recomputes them
            # from the episodes rather than averaging them, which is the point.
            "groups": {}}


def test_two_seeds_of_a_level_combine_into_one_verdict():
    """L1 has 33 tasks and no more, so the only way to buy power there is another
    seed. Two runs must become one verdict, not two verdicts and an average."""
    from slime_htm.envs.gate import combine_results

    first = gate_output(0, {"a": (False, True), "b": (True, True), "c": (True, False)})
    second = gate_output(1, {"a": (False, True), "b": (True, True), "c": (False, True)})
    combined = combine_results([first, second])

    split = combined["groups"]["l1"]
    assert split["pairs"] == 6, "six paired episodes"
    assert split["tasks"] == 3, "from three tasks -- reported, because they are not six"
    assert split["seeds"] == [0, 1]
    assert split["helps"] == pytest.approx(3 / 6)
    assert split["hurts"] == pytest.approx(1 / 6)
    assert split["both_right"] == pytest.approx(2 / 6)
    assert "verdict" in split


def test_combining_the_same_seed_twice_is_refused():
    """It is the same episodes read twice, which would double every count and make
    the sign test significant on no new evidence."""
    from slime_htm.envs.gate import combine_results

    one = gate_output(0, {"a": (False, True)})
    with pytest.raises(ValueError, match="same seed appears twice"):
        combine_results([one, gate_output(0, {"a": (False, True)})])


def test_runs_that_are_not_the_same_gate_are_refused():
    """A combined verdict over two budgets or two step limits is a verdict about
    nothing. The seed is the one field allowed to differ, and the environment,
    which records the node."""
    from slime_htm.envs.gate import combine_results

    base = gate_output(0, {"a": (False, True)})
    with pytest.raises(ValueError, match="not the same gate"):
        combine_results([base, gate_output(1, {"a": (False, True)}, budget=1024)])
    with pytest.raises(ValueError, match="not the same gate"):
        combine_results([base, gate_output(1, {"a": (False, True)}, max_steps=10)])

    differing_node = gate_output(1, {"a": (False, True)})
    differing_node["header"]["environment"] = {"python": "elsewhere"}
    assert combine_results([base, differing_node])["combined_from"] == [0, 1]


def test_an_incomplete_pair_is_dropped_rather_than_counted_half():
    """A killed run leaves one policy's episode without its partner. Counting it
    would put an unpaired episode into a paired statistic."""
    from slime_htm.envs.gate import combine_results

    half = gate_output(0, {"a": (False, True), "b": (True, True)})
    half["episodes"] = [e for e in half["episodes"]
                        if not (e["task"] == "b" and e["policy"] == "always")]
    split = combine_results([half])["groups"]["l1"]
    assert split["pairs"] == 1 and split["tasks"] == 1


class StubEnv:
    """Two candidates, three steps, and a reward that depends on which was chosen.

    The never policy takes candidate 0 and the always policy candidate 1, so the
    stub decides the outcome by the action and the episode is paired without the
    environment knowing anything about policies.
    """

    def __init__(self, task):
        self.task = task
        self.steps = 0

    def _observation(self):
        # The element ids a written action is checked against: always-think writes
        # its action now, and `parse_action` refuses a bid the page does not have.
        return {"goal": f"do {self.task}",
                "extra_element_properties": {"a": {"clickable": True},
                                             "b": {"clickable": True}}}

    def reset(self, seed=None):
        self.steps = 0
        return {**self._observation(), "seed": seed}, {}

    def step(self, action):
        self.steps += 1
        reward = 1.0 if "'b'" in action else 0.0
        done = self.steps >= 2
        return self._observation(), reward, done, False, {}

    def close(self):
        pass


class StubDecider:
    def health(self):
        return {"model": "stub", "head": None}

    def fast(self, state, actions):
        return {"probabilities": [0.7, 0.3]}

    def slow(self, state, actions, budget):
        return {"probabilities": [0.2, 0.8], "generated_tokens": 11}

    def solve(self, state, instructions, budget):
        """What always-think now does: write the action rather than pick one. The
        stub writes a click on the second element, so the paired comparison is
        the same as before -- never takes the first, always takes the second."""
        return {"text": "I will take the second one.\nclick('b')",
                "generated_tokens": 11, "truncated": False}


def gate_in_a_box(monkeypatch, tmp_path, tasks=("t-a", "t-b", "t-c")):
    """Everything `main()` needs except a browser, a model and an instance."""
    import sys
    import types

    from slime_htm.envs import browser

    names = [f"workarena.servicenow.{task}" for task in tasks]
    package = types.ModuleType("browsergym")
    venue = types.ModuleType("browsergym.workarena")
    venue.workarena_tasks_l1 = names
    venue.workarena_tasks_all = names
    package.workarena = venue
    gym = types.ModuleType("gymnasium")
    gym.registry = {f"browsergym/{name}": object() for name in names}
    gym.make = lambda task, **kwargs: StubEnv(task)
    monkeypatch.setitem(sys.modules, "browsergym", package)
    monkeypatch.setitem(sys.modules, "browsergym.workarena", venue)
    monkeypatch.setitem(sys.modules, "gymnasium", gym)
    monkeypatch.setattr("inference.DecisionClient", lambda url: StubDecider())
    monkeypatch.setattr(browser, "candidates",
                        lambda env, observation, limit, extra=(): (
                            [("click('a')", "click('a')"),
                             ("click('b')", "click('b')")]
                            + [(action, action) for action in extra], 0))
    monkeypatch.setattr(browser, "state_text",
                        lambda observation, goal, trail, chars: f"{goal} after {trail}")
    # The venue's own variables, which `main` now refuses to run without. The stub
    # never reads them; the point is that the refusal is in front of the run.
    from slime_htm.envs.gate import VENUES
    for name in VENUES["workarena"]["requires_env"]:
        monkeypatch.setenv(name, "stub")
    # And the commit, **stubbed rather than inherited from the working
    # directory**. `main` calls `require_commit(commit_now())`, and `commit_now`
    # reads `Path.cwd()`: run from the repository root it finds the checkout and
    # these tests passed, run from anywhere else it returns `unknown` and the gate
    # correctly refuses. The refusal is the designed behaviour -- a verdict that
    # cannot be tied to a reviewed commit is not evidence -- and it has its own
    # test at `require_commit`. These tests are about the loop, so they say which
    # commit they are rather than depending on where pytest was invoked.
    monkeypatch.setattr("litjev.run.provenance.commit_now",
                        lambda root=None: "0" * 40)
    return str(tmp_path / "gate.json"), str(tmp_path / "gate.rows.jsonl")


def run_gate(monkeypatch, output, rows, *extra):
    import sys

    from slime_htm.envs.gate import main

    argv = ["gate.py", "--decision-url", "http://stub", "--venue", "workarena", "--groups", "l1",
            "--output", output, "--rows", rows, *extra]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    return json.loads(Path(output).read_text())


def test_the_level_loop_runs_and_resumes_without_losing_a_pair(monkeypatch, tmp_path):
    """`main()`'s own loop, which no test reached before: `--resume` raised
    `unhashable type: 'dict'` on the first task it had already finished, and a
    resumed pair never reached the output JSON that `--combine` reads."""
    from slime_htm.envs.gate import combine_results

    output, rows = gate_in_a_box(monkeypatch, tmp_path)
    first = run_gate(monkeypatch, output, rows)
    assert len(first["episodes"]) == 6, "three tasks, both policies"
    assert first["groups"]["l1"]["helps"] == pytest.approx(1.0), "always takes b"

    # A kill: the header, one complete pair, and one episode of the next task.
    lines = Path(rows).read_text().splitlines()
    header = lines[0]
    episodes = [line for line in lines if '"kind": "episode"' in line]
    assert len(episodes) == 6
    survived = [header, *episodes[:3]]
    Path(rows).write_text("\n".join(survived) + "\n")

    second = run_gate(monkeypatch, output, rows, "--resume")
    assert len(second["episodes"]) == 6, "the resumed pair is in the output too"
    # Everything except the clock: a resumed run reuses the kept pair's recorded
    # seconds and measures the rest itself, so the timings are allowed to differ
    # and nothing a verdict is made of is.
    outcome = ("n", "never", "always", "oracle", "both_right", "helps", "hurts",
               "both_wrong", "verdict")
    for key in outcome:
        assert second["groups"]["l1"][key] == first["groups"]["l1"][key], key
    assert {e["task"] for e in second["episodes"]} == {t["task"] for t in first["episodes"]}

    # And the output a resumed run writes is one `--combine` can read back.
    combined = combine_results([second])["groups"]["l1"]
    for key in ("never", "always", "both_right", "helps", "hurts", "oracle"):
        assert combined[key] == pytest.approx(second["groups"]["l1"][key]), key
    assert combined["pairs"] == 3 and combined["tasks"] == 3


def test_a_half_pair_in_the_rows_file_is_finished_rather_than_skipped(monkeypatch,
                                                                     tmp_path):
    """The killed-mid-task case: `never` is on disk and `always` is not. The task
    must be completed, not treated as done and not run twice."""
    output, rows = gate_in_a_box(monkeypatch, tmp_path, tasks=("t-a", "t-b"))
    run_gate(monkeypatch, output, rows)

    lines = Path(rows).read_text().splitlines()
    header = lines[0]
    episodes = [line for line in lines if '"kind": "episode"' in line]
    # Keep t-a's pair and only the first of t-b's two episodes.
    Path(rows).write_text("\n".join([header, *episodes[:3]]) + "\n")
    kept_policies = {json.loads(line)["policy"] for line in episodes[2:3]}

    second = run_gate(monkeypatch, output, rows, "--resume")
    assert len(second["episodes"]) == 4
    second_b = [e for e in second["episodes"] if e["task"].endswith("t-b")]
    assert {e["policy"] for e in second_b} == {"never", "always"}
    assert kept_policies <= {e["policy"] for e in second_b}


def test_an_argument_error_does_not_need_the_venue_installed(monkeypatch):
    """`--groups l4` must be an argument error anywhere, not a ModuleNotFoundError
    on whichever machine happens to lack chromium. This broke once by moving the
    venue imports earlier, and the exit code is in the launcher's contract."""
    import sys

    from slime_htm.envs.gate import main

    monkeypatch.setattr(sys, "argv",
                        ["gate.py", "--decision-url", "unused", "--venue", "workarena", "--groups", "l4"])
    with pytest.raises(SystemExit) as refused:
        main()
    assert refused.value.code == 2, "argparse's exit code, not an import failure"


def webarena_configs():
    """WebArena's task file in miniature, with the shape the real one has.

    Measured against `browsergym-webarena==0.14.3`: 812 tasks, ids that are bare
    integers, `sites` lists that are usually one site and sometimes two, and 118
    tasks whose `reference_answers` carry a `fuzzy_match` key -- the ones an LLM
    grades.
    """
    def task(task_id, sites, fuzzy=False):
        refs = {"fuzzy_match": "whatever"} if fuzzy else {"must_include": "x"}
        return {"task_id": task_id, "sites": sites,
                "eval": {"eval_types": ["string_match"], "reference_answers": refs}}

    return [task(0, ["gitlab"]), task(1, ["gitlab"]), task(2, ["gitlab"], fuzzy=True),
            task(3, ["shopping"]), task(4, ["shopping"], fuzzy=True),
            task(5, ["gitlab", "reddit"]), task(6, ["map", "wikipedia"]),
            task(7, ["wikipedia", "map"]), task(8, ["reddit"])]


def fake_webarena(monkeypatch, configs=None):
    import sys
    import types

    from slime_htm.envs import gate

    configs = webarena_configs() if configs is None else configs
    monkeypatch.setattr(gate, "_webarena_configs", lambda: configs)
    gym = types.ModuleType("gymnasium")
    gym.registry = {f"browsergym/webarena.{c['task_id']}": object() for c in configs}
    monkeypatch.setitem(sys.modules, "gymnasium", gym)
    return gym


def test_a_webarena_site_is_a_join_and_not_a_name_pattern(monkeypatch):
    """WebArena's ids are `browsergym/webarena.0` to `.811` -- bare integers. The
    site, the validator and the difficulty are in the task file and nowhere in
    the name, so unlike WorkArena the grouping cannot come from the registry."""
    from slime_htm.envs.gate import group_tasks

    gym = fake_webarena(monkeypatch)
    assert all(task.split(".")[-1].isdigit() for task in gym.registry), \
        "nothing in an id says which site it is"

    assert group_tasks("webarena", "gitlab") == ["browsergym/webarena.0",
                                                 "browsergym/webarena.1"]
    assert group_tasks("webarena", "shopping") == ["browsergym/webarena.3"]


def test_the_model_judged_tasks_are_excluded(monkeypatch):
    """118 of the 812 carry a fuzzy_match reference answer and need OPENAI_API_KEY
    to grade. A routing score that depends on a third party's model version cannot
    be reproduced after that version is deprecated, so they are out -- and the
    exclusion is in the code, before any score was seen."""
    from slime_htm.envs.gate import group_tasks, judged_by_a_model

    fake_webarena(monkeypatch)
    configs = webarena_configs()
    assert sum(map(judged_by_a_model, configs)) == 2
    assert judged_by_a_model({"eval": {"reference_answers": {"fuzzy_match": "a"}}})
    assert not judged_by_a_model({"eval": {"reference_answers": {"must_include": "a"}}})
    assert not judged_by_a_model({}), "a task with no eval block is not model-judged"

    assert "browsergym/webarena.2" not in group_tasks("webarena", "gitlab")
    assert "browsergym/webarena.4" not in group_tasks("webarena", "shopping")


def test_a_task_needing_two_sites_is_in_neither_site_group(monkeypatch):
    """Each site is a container to stand up, so a task spanning two needs both.
    Folding it into one site's group would make that group's verdict depend on a
    server the run was not told to start."""
    from slime_htm.envs.gate import group_tasks

    fake_webarena(monkeypatch)
    assert "browsergym/webarena.5" not in group_tasks("webarena", "gitlab")
    assert "browsergym/webarena.5" not in group_tasks("webarena", "reddit")
    assert group_tasks("webarena", "multi") == ["browsergym/webarena.5",
                                                "browsergym/webarena.6",
                                                "browsergym/webarena.7"]
    assert group_tasks("webarena", "reddit") == ["browsergym/webarena.8"], \
        "a single-site reddit task is still reddit's"


def test_a_group_that_resolves_to_nothing_is_refused(monkeypatch):
    """`wikipedia` is this case against the real package: every task naming it
    also needs another site, so the single-site group is empty. Empty would
    otherwise reach `four_way([])` and be printed as a verdict over nothing."""
    from slime_htm.envs.gate import group_tasks

    fake_webarena(monkeypatch)
    with pytest.raises(RuntimeError, match="resolves to no tasks"):
        group_tasks("webarena", "wikipedia")


def test_a_group_emptied_by_the_exclusion_says_so_instead(monkeypatch):
    """A different diagnosis from the one above and worth its own message: the
    site has tasks, and the gate is refusing to score them. Silently empty would
    read as "this site does not exist"."""
    from slime_htm.envs.gate import group_tasks

    only_judged = [{"task_id": 9, "sites": ["map"],
                    "eval": {"reference_answers": {"fuzzy_match": "x"}}}]
    fake_webarena(monkeypatch, configs=only_judged)
    with pytest.raises(RuntimeError, match="model-judged"):
        group_tasks("webarena", "map")


def test_the_webarena_task_file_is_found_in_a_namespace_package(monkeypatch):
    """`webarena` ships without an `__init__`, so its spec has no origin and the
    directory is in `submodule_search_locations`. Reading `spec.origin` alone
    reported "libwebarena is not installed" about a package that was installed."""
    import types

    from slime_htm.envs import gate

    directory = Path(__file__).parent
    assert not (directory / "test.raw.json").exists(), "the fixture needs it absent"
    spec = types.SimpleNamespace(origin=None, submodule_search_locations=[str(directory)])
    monkeypatch.setattr("importlib.util.find_spec", lambda name: spec)
    with pytest.raises(RuntimeError, match="layout has changed"):
        gate._webarena_configs()


def test_every_venue_offers_groups_and_a_way_to_get_their_tasks():
    """The venue table is the only place a venue is defined, so a venue missing a
    piece of it would fail at a different point for each venue."""
    from slime_htm.envs.gate import VENUES

    assert set(VENUES) == {"webarena", "workarena", "miniwob"}
    for venue, spec in VENUES.items():
        assert spec["groups"], venue
        assert callable(spec["tasks"]), venue
        assert spec["registers"].startswith("browsergym."), venue
        assert f"browsergym-{venue}" in spec["packages"], venue


def action_split():
    from slime_htm.envs.gate import miniwob_split_path

    path = miniwob_split_path()
    return path, json.loads(path.read_text())


def test_the_miniwob_split_is_committed_and_partitions_every_task():
    """The split decides which MiniWoB results may be read as a measurement, so it
    is a file in the repository and not a computation at run time. Derived from the
    scores instead -- "the typing tasks came back at zero" -- it would be the same
    number twice."""
    _, record = action_split()
    click = record["groups"]["click"]
    beyond = record["groups"]["beyond-click"]
    pointer = record["groups"]["pointer"]

    assert len(record["tasks"]) == 125, "every registered MiniWoB task is classified"
    assert set(click) | set(beyond) | set(pointer) == set(record["tasks"]), \
        "the split covers them"
    assert not set(click) & set(beyond) and not set(click) & set(pointer)
    assert len(click) == 54 and len(pointer) == 18 and len(beyond) == 53
    assert all(not record["tasks"][name]["needs"] for name in click)
    assert all(record["tasks"][name]["needs"] for name in beyond)
    assert all(record["tasks"][name]["pointer_markers"] for name in pointer), \
        "the pointer group is exactly the tasks a click cannot express"


def test_the_split_puts_the_typing_tasks_where_a_reader_would_expect():
    """Spot checks, because a rule that is wrong on `enter-text` is wrong."""
    _, record = action_split()
    beyond, click = set(record["groups"]["beyond-click"]), set(record["groups"]["click"])

    for typing in ("enter-text", "enter-password", "login-user", "book-flight"):
        assert typing in beyond, f"{typing} needs typing"
    for clicking in ("click-button", "click-link", "click-checkboxes", "click-tab"):
        assert clicking in click, f"{clicking} is a click task"


def test_the_split_records_the_signals_it_refuses_to_use():
    """Pointer tasks are not detected, because the signals are noisy -- a jquery-ui
    include fires on pages that never drag. The error runs one way, so impossible
    tasks stay in the click group and depress its success, and the markers are how
    a reader sees the size of that contamination instead of guessing it.

    It was guessed, twice: 8 on the first pass, then 12, then 18 as two readers
    looked harder. Hence the named list in the file rather than a count in prose,
    and hence the four kinds beyond the library include."""
    _, record = action_split()
    click = record["groups"]["click"]
    contaminated = record["click_tasks_with_pointer_markers"]

    assert set(contaminated) == set(record["groups"]["pointer"]), \
        "they are their own group now, not a contamination of click"
    assert all(record["tasks"][name]["pointer_markers"] for name in contaminated)
    assert len(contaminated) == 18, "54 click tasks and 18 pointer ones"
    assert not set(contaminated) & set(click), "click is clean now"
    assert "depresses" in record["rule"], "the direction of the error is recorded"

    # The four kinds the first pass missed, each a different way to need a pointer.
    kinds = {kind for name in contaminated
             for kind in record["tasks"][name]["pointer_markers"]}
    assert {"pointer_library", "rich_text", "selection", "coordinates"} <= kinds
    for named, kind in (("text-editor", "rich_text"), ("highlight-text", "selection"),
                        ("highlight-text-2", "selection"), ("circle-center", "coordinates")):
        assert kind in record["tasks"][named]["pointer_markers"], named
        assert named in contaminated, f"{named} is in the pointer group"


def test_the_miniwob_pin_is_the_same_everywhere_it_appears():
    """Three places name the MiniWoB commit: the split that was computed from it,
    the image that clones it, and the cluster build that does the same without a
    container. The tasks are identified by file name and the repo renames them, so
    a split computed from one commit and pages served from another is a silent
    mismatch."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    _, record = action_split()
    pin = record["miniwob_commit"]
    assert len(pin) == 40 and all(c in "0123456789abcdef" for c in pin)

    for named in ("src/slime_htm/envs/Dockerfile",
                  "scripts/build_browsergym.sbatch",
                  "scripts/miniwob_action_split.py"):
        assert pin in (root / named).read_text(), f"{named} names a different commit"


def test_miniwob_offers_no_pooled_group():
    """Condition from review, made structural rather than promised: over 125 tasks
    that mix clicking and typing, one pooled verdict would restate the action space
    and be read as a venue verdict."""
    from slime_htm.envs.gate import VENUES

    assert VENUES["miniwob"]["groups"] == ("click", "pointer", "beyond-click")
    assert "all" not in VENUES["miniwob"]["groups"]


def test_a_missing_split_file_is_a_refusal_with_instructions(tmp_path):
    """Not an empty group: the file is the pre-registration, so running without it
    would be running without one."""
    from slime_htm.envs.gate import miniwob_tasks

    with pytest.raises(RuntimeError, match="miniwob_action_split"):
        miniwob_tasks("click", split_path=str(tmp_path / "absent.json"))


def test_every_venue_declares_the_environment_it_needs():
    """browsergym checks none of these until a task is constructed, so an unset
    variable costs a queue wait and a GPU allocation before it says anything. The
    contract in the README lists them; this is the same list in code."""
    from slime_htm.envs.gate import VENUES, missing_environment

    expected = {
        "miniwob": {"MINIWOB_URL"},
        "workarena": {"SNOW_INSTANCE_URL", "SNOW_INSTANCE_UNAME", "SNOW_INSTANCE_PWD"},
        "webarena": {"WA_SHOPPING", "WA_SHOPPING_ADMIN", "WA_REDDIT", "WA_GITLAB",
                     "WA_WIKIPEDIA", "WA_MAP", "WA_HOMEPAGE"},
    }
    for venue, names in expected.items():
        assert set(VENUES[venue]["requires_env"]) == names, venue
        assert missing_environment(venue, environ={}) == list(VENUES[venue]["requires_env"])
        full = dict.fromkeys(names, "set")
        assert missing_environment(venue, environ=full) == []

    # An empty string is not a value: a launcher that exports an unset variable
    # passes it through as "" and browsergym would then build a task with no URL.
    assert missing_environment("miniwob", environ={"MINIWOB_URL": ""}) == ["MINIWOB_URL"]
    assert "WA_FULL_RESET" not in VENUES["webarena"]["requires_env"], "it is optional"


def test_main_refuses_before_the_run_when_the_venue_environment_is_unset(monkeypatch,
                                                                        tmp_path):
    """The refusal has to be in `main`'s path, not only in the helper: that is where
    a launcher's missing export actually arrives."""
    output, rows = gate_in_a_box(monkeypatch, tmp_path)
    monkeypatch.delenv("SNOW_INSTANCE_PWD")

    with pytest.raises(SystemExit) as refused:
        run_gate(monkeypatch, output, rows)
    # 1, not argparse's 2: the contract reserves 2 for a bad argument and puts a
    # broken environment at 1, and a launcher branches on the code.
    assert refused.value.code != 2
    assert "SNOW_INSTANCE_PWD" in str(refused.value)
    assert not Path(output).exists(), "nothing was written"


def test_check_env_runs_where_a_launcher_can_use_it(monkeypatch, capsys):
    """It has to work on a login node: no venue package installed, no decision
    service, nothing allocated. The repo venv has no browsergym, so this test is
    that machine. Without it the check only saves the time up to the first
    `gym.make`, which is already inside an allocated job -- code-reviewer's point."""
    import sys

    from slime_htm.envs.gate import main

    # No --groups: it asks about the environment, and a launcher should not have to
    # name a group to ask. Requiring it made the documented command exit 2.
    monkeypatch.setattr(sys, "argv", ["gate.py", "--venue", "miniwob", "--check-env"])
    monkeypatch.setenv("MINIWOB_URL", "file:///opt/miniwob/miniwob/html/miniwob/")
    main()
    out = capsys.readouterr().out
    assert "set" in out and "MINIWOB_URL" in out
    assert "every required variable is set" in out

    monkeypatch.delenv("MINIWOB_URL")
    with pytest.raises(SystemExit) as refused:
        main()
    assert refused.value.code != 2, "a broken environment is exit 1 by the contract"
    assert "MINIWOB_URL" in str(refused.value)


def test_the_commands_that_run_nothing_need_no_decision_url(monkeypatch):
    """`--list`, `--check-env` and `--combine` touch no decision service, so
    requiring the flag made every documented first command carry
    `--decision-url unused`, which is a thing a reader copies and then wonders
    about."""
    import sys

    from slime_htm.envs.gate import main

    monkeypatch.setenv("MINIWOB_URL", "file:///pages/")
    monkeypatch.setattr(sys, "argv", ["gate.py", "--venue", "miniwob", "--check-env"])
    main()

    # And it is still required for a run.
    monkeypatch.setattr(sys, "argv", ["gate.py", "--venue", "miniwob",
                                      "--groups", "click"])
    with pytest.raises(SystemExit) as refused:
        main()
    assert refused.value.code == 2, "a missing required argument is an argument error"

    # --list does need groups, because it prints them; a run does too.
    for argv in (["gate.py", "--venue", "miniwob", "--list"],
                 ["gate.py", "--venue", "miniwob", "--decision-url", "http://stub"]):
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as refused:
            main()
        assert refused.value.code == 2, argv


def test_the_split_is_found_from_any_working_directory(monkeypatch, tmp_path):
    """It was read by a cwd-relative path under `experiments/`, and neither place a
    run happens has that directory: `scripts/sync.sh` rsyncs `src scripts tests
    configs`, and the image copies `pyproject.toml`, `README.md` and `src/`. So
    `--list` died before anything ran, and the error told the reader to regenerate
    the file into the same unreachable place."""
    from slime_htm.envs.gate import miniwob_split_path, miniwob_tasks

    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "experiments").exists(), "a cwd with no experiments/"
    assert miniwob_split_path().exists(), "found from the module, not the cwd"

    import sys
    import types

    names = json.loads(miniwob_split_path().read_text())["tasks"]
    package = types.ModuleType("browsergym")
    venue = types.ModuleType("browsergym.miniwob")
    package.miniwob = venue
    gym = types.ModuleType("gymnasium")
    gym.registry = {f"browsergym/miniwob.{name}": object() for name in names}
    monkeypatch.setitem(sys.modules, "browsergym", package)
    monkeypatch.setitem(sys.modules, "browsergym.miniwob", venue)
    monkeypatch.setitem(sys.modules, "gymnasium", gym)
    assert len(miniwob_tasks("click")) == 54


def test_the_split_ships_where_the_code_ships():
    """The two places a run reads it from: a tree rsynced by sync.sh, and the
    image. Both carry `src/`; neither carries `experiments/`. This asserts the
    file is under src and that the sync and the image would bring it."""
    from pathlib import Path

    from slime_htm.envs.gate import miniwob_split_path

    root = Path(__file__).resolve().parents[1]
    relative = miniwob_split_path().relative_to(root)
    assert relative.parts[0] == "src", f"{relative} is outside the synced tree"

    sync = (root / "scripts/sync.sh").read_text()
    assert "for tree in src scripts tests configs" in sync, \
        "sync.sh no longer syncs the trees this test reasons about"
    assert "COPY src /src/src" in (root / "src/slime_htm/envs/Dockerfile").read_text()


def test_a_run_that_sees_no_actions_aborts_instead_of_reporting_near_zero():
    """An episode ending `no_candidates` offered nothing to choose from. One is a
    fact about a page; the first three in a row are a broken observation, and the
    alternative to aborting is a full run reporting "both near zero" about the
    environment rather than about thinking. Aborting is not a verdict, so it
    cannot bias one."""
    from slime_htm.envs.gate import BLIND_EPISODES, blind_run

    assert BLIND_EPISODES == 3
    assert blind_run(["no_candidates"] * 3)
    assert "nothing was measured" in blind_run(["no_candidates"] * 3)
    assert blind_run(["no_candidates"] * 2) is None, "two is not yet a pattern"
    assert blind_run([]) is None
    # A real one among them is enough: the environment is working.
    assert blind_run(["no_candidates", "done", "no_candidates"]) is None
    assert blind_run(["done", "no_candidates", "no_candidates", "no_candidates"]) is None, \
        "only the first episodes decide, so a later cluster is a result not a fault"


def test_the_split_is_in_the_built_wheel_not_only_in_an_editable_install():
    """The cluster and the image both install editable, so the file is reachable
    there whatever the build says. A wheel is the case that would drop it, and
    hatchling includes everything under a listed package -- verified by building
    one: it carries `slime_htm/envs/miniwob-action-split.json`. This pins the
    mechanism, so narrowing `packages` or moving to a backend that needs explicit
    package-data fails here."""
    import tomllib
    from pathlib import Path

    from slime_htm.envs.gate import miniwob_split_path

    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    packages = config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert config["build-system"]["build-backend"] == "hatchling.build", \
        "another backend may need the .json declared as package data"

    inside = miniwob_split_path().relative_to(root)
    assert any(inside.is_relative_to(Path(p)) for p in packages), \
        f"{inside} is not under any packaged directory: {packages}"


def test_always_think_writes_its_action_and_reaches_what_clicking_cannot(monkeypatch,
                                                                        tmp_path):
    """The design error this fixes. `decider.slow` reasons and then reads the
    answer off the same click-only candidate list, so always-think had exactly
    never-think's action space: a step whose right action is typing was
    unwinnable at every price, and the gate would have reported "both near zero"
    about a venue when it was reporting a restriction of ours."""
    from slime_htm.envs.gate import run_episode

    class TypingPage(StubEnv):
        """A page that is finished by filling a box, not by clicking."""

        def _observation(self):
            # `box` has to be here on *every* instance: `parse_action` checks the
            # written bid against the page, and setting this on one object was
            # this test's own first bug.
            return {"goal": 'search for "usb wifi"',
                    "extra_element_properties": {"a": {"clickable": True},
                                                 "box": {"clickable": True}}}

        def step(self, action):
            self.steps += 1
            if action.startswith("fill('box'"):
                return self._observation(), 1.0, True, False, {}
            done = self.steps >= 3
            return self._observation(), 0.0, done, False, {}

    class TypingDecider(StubDecider):
        def solve(self, state, instructions, budget):
            return {"text": "The box needs the query.\nfill('box', 'usb wifi')",
                    "generated_tokens": 20, "truncated": False}

    from slime_htm.envs import browser

    monkeypatch.setattr(browser, "candidates",
                        lambda env, observation, limit, extra=(): (
                            [("click('a')", "click('a')")]
                            + [(action, action) for action in extra], 0))
    monkeypatch.setattr(browser, "state_text",
                        lambda observation, goal, trail, chars: f"{goal} {trail}")

    page = TypingPage("typing")
    common = dict(budget=64, max_steps=4, limit=52, seed=0, state_chars=500)
    never = run_episode(page, StubDecider(), "t", "never", **common)
    always = run_episode(TypingPage("typing"), TypingDecider(), "t", "always", **common)

    assert never["success"] is False, "clicking cannot finish it, at any price"
    assert always["success"] is True, "writing the action can"
    assert always["steps"] == 1
    assert always["step_rows"][0]["written_action"] == "fill('box', 'usb wifi')"


def test_an_unexecutable_written_action_is_recorded_and_ends_the_episode(monkeypatch):
    """Not a click. Substituting the nearest element would record a step the model
    did not choose and credit thinking with its outcome; continuing past it would
    attribute the rest of the episode to a decision nobody made."""
    from slime_htm.envs import browser
    from slime_htm.envs.gate import run_episode

    class Nonsense(StubDecider):
        def solve(self, state, instructions, budget):
            return {"text": "I think I should give up.", "generated_tokens": 5,
                    "truncated": False}

    monkeypatch.setattr(browser, "candidates",
                        lambda env, observation, limit, extra=(): (
                            [("click('a')", "click('a')")]
                            + [(action, action) for action in extra], 0))
    monkeypatch.setattr(browser, "state_text",
                        lambda observation, goal, trail, chars: "state")

    page = StubEnv("t")
    run = run_episode(page, Nonsense(), "t", "always", budget=64, max_steps=4,
                      limit=52, seed=0, state_chars=500)

    assert run["ended"] == "unexecutable_action"
    assert run["steps"] == 1 and run["success"] is False
    assert "no action call" in run["step_rows"][0]["unexecutable"]
    assert page.steps == 0, "nothing was executed on the page"


def test_the_unexecutable_count_is_reported_per_episode_and_group():
    """Asked for in review: a parser too strict for how the model phrases its
    conclusion would otherwise read as the model being incapable. The count makes
    the two distinguishable from the output alone."""
    from slime_htm.envs.gate import format_groups, four_way

    def episode_with(bad):
        return {**episode(False), "unexecutable": bad}

    pairs = [(episode_with(0), episode_with(2)), (episode_with(0), episode_with(1))]
    split = four_way(pairs)
    assert split["unexecutable_steps"] == 3
    assert "bad" in format_groups({"click": split})
    assert "    3" in format_groups({"click": split})


def test_helps_is_split_by_whether_the_slow_mode_reached_outside_the_fast_set():
    """The confound exps raised: the two policies differ in reasoning *and* in
    action space, so `always - never` alone cannot say which won, and nothing in
    PASS would flag it. It is separable from data the rows already carry -- the
    slow mode's written action either was or was not one the fast mode was
    offered."""
    from slime_htm.envs.gate import four_way

    def rescued(by_reach):
        return {**episode(True), "used_outside_action": by_reach}

    pairs = [(episode(False), rescued(True)),    # typed something: reach
             (episode(False), rescued(False)),   # clicked better: reasoning
             (episode(True), episode(True))]
    split = four_way(pairs)

    assert split["helps"] == pytest.approx(2 / 3)
    assert split["helps_by_reach"] == pytest.approx(1 / 3)
    assert split["helps_by_reasoning"] == pytest.approx(1 / 3)
    assert split["helps_by_reach"] + split["helps_by_reasoning"] == \
        pytest.approx(split["helps"]), "the two account for all of it"


def test_the_regrouping_is_recorded_as_dated_after_the_data():
    """The condition on 01b43b1, and it is about reading rather than code. The
    committed rule said the pointer tasks *stay in* the click group; 28816 ran
    under that rule, and the decision to read them apart came after its first 28
    episodes. Naming them in advance is not pre-registering the decision to
    analyse them separately -- and the file has to say so, because the file is
    what a reader will believe."""
    _, record = action_split()
    note = record["regrouped"]

    assert "after" in note and "28816" in note
    assert "stay in the click group" in note, "the superseded rule is quoted"
    assert "primary" in note and "secondary" in note
    assert "cannot be combined" in note, "the action set changed, so it is a new run"


def test_combine_refuses_runs_whose_fast_action_set_changed():
    """What enforces point 3 is not the note: the rows header carries the commit,
    and `--combine` refuses any header difference but the seed and the node. A run
    from before `scroll` was offered measured a different fast mode."""
    from slime_htm.envs.gate import combine_results

    before = gate_output(0, {"a": (False, True)}, commit="c8ed86e")
    after = gate_output(1, {"a": (False, True)}, commit="01b43b1")
    with pytest.raises(ValueError, match="not the same gate"):
        combine_results([before, after])


def test_a_step_that_never_returns_is_recorded_and_not_scored(monkeypatch):
    """28816 blocked inside step 14 of `click-collapsible-nodelay` and produced
    nothing for 3h35m until slurm took the job; exps found it from the streamed
    rows -- fourteen step rows, no episode row. A bigger --time buys a longer
    silence, so the step has a deadline. The episode is marked incomplete rather
    than failed: a browser that stopped answering is not a decision the model
    made, and scoring it as a loss biases whichever policy hit the hang."""
    import time

    from slime_htm.envs import browser
    from slime_htm.envs.gate import run_episode

    class Hangs(StubEnv):
        def step(self, action):
            time.sleep(3)
            raise AssertionError("not reached: the deadline fires first")

    monkeypatch.setattr(browser, "candidates",
                        lambda env, observation, limit, extra=(): (
                            [("click('a')", "click('a')")], 0))
    monkeypatch.setattr(browser, "state_text",
                        lambda observation, goal, trail, chars: "state")

    run = run_episode(Hangs("t"), StubDecider(), "t", "never", budget=8, max_steps=4,
                      limit=52, seed=0, state_chars=200, step_timeout=0.5)

    assert run["ended"] == "step_timeout"
    assert run["complete"] is False
    assert "0.5s" in run["timed_out"]
    assert run["steps"] == 1, "the step that blocked is still recorded"


def test_an_incomplete_pair_is_dropped_from_the_split_and_counted():
    """Not scored as a loss, and not silently absent: a group with many of these
    has no verdict, and the number is what says so."""
    from slime_htm.envs.gate import four_way

    def done(success):
        return {**episode(success), "complete": True}

    hung = {**episode(False), "complete": False}
    split = four_way([(done(False), done(True)), (hung, done(True)),
                      (done(True), done(True))])

    assert split["n"] == 2, "the hung pair is not one of the two scored"
    assert split["incomplete_pairs"] == 1
    assert split["helps"] == pytest.approx(0.5), "and does not dilute the rest"


def test_a_page_that_will_not_load_is_incomplete_too(monkeypatch):
    """`env.reset` sits outside the episode clock, so an unbounded page load was
    invisible in the rows as well as unbounded. Recommended in review, and the
    same non-outcome as a step that will not return."""
    import time

    from slime_htm.envs import browser
    from slime_htm.envs.gate import run_episode

    class SlowToLoad(StubEnv):
        def reset(self, seed=None):
            time.sleep(3)
            raise AssertionError("not reached: the deadline fires first")

    monkeypatch.setattr(browser, "candidates",
                        lambda env, observation, limit, extra=(): ([], 0))
    monkeypatch.setattr(browser, "state_text",
                        lambda observation, goal, trail, chars: "state")

    run = run_episode(SlowToLoad("t"), StubDecider(), "t", "never", budget=8,
                      max_steps=4, limit=52, seed=0, state_chars=200,
                      step_timeout=0.5)

    assert run["ended"] == "reset_timeout"
    assert run["complete"] is False and run["steps"] == 0
    assert run["step_rows"] == []


def test_closing_a_wedged_browser_does_not_wait_forever(monkeypatch):
    """After a step has hung, `env.close()` is a call to the same wedged browser,
    so waiting on it turns one dead episode into a dead run -- which is how 28816
    ended. The processes are killed and the problem is returned, not raised: this
    runs when something is already wrong."""
    import time

    from slime_htm.envs import browser
    from slime_htm.envs.browser import close_env

    class Wedged:
        def close(self):
            time.sleep(3)
            raise AssertionError("not reached")

    killed = []
    monkeypatch.setattr(browser, "kill_browser_children",
                        lambda: killed.append(True) or ["chrome:123"])

    trouble = close_env(Wedged(), seconds=0.4)
    assert trouble and "did not return within 0.4s" in trouble
    assert "chrome:123" in trouble and killed, "it killed rather than waited"

    class Fine:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    env = Fine()
    assert close_env(env, seconds=5) is None and env.closed

    class Raises:
        def close(self):
            raise RuntimeError("already gone")

    assert "already gone" in close_env(Raises(), seconds=5)


def test_the_whole_browser_tree_is_killed_not_just_our_children():
    """The tree is python -> the `node` driver -> chromium, so killing only direct
    children killed the driver and orphaned Chromium, which kept its memory for the
    rest of the job. Found in review."""
    from slime_htm.envs.browser import descendants

    tree = {100: (1, "python"), 200: (100, "node"), 300: (200, "chrome"),
            400: (300, "chrome_crashpad"), 500: (1, "unrelated")}
    found = dict(descendants(100, tree))

    assert set(found) == {200, 300, 400}, "grandchildren and deeper, not just 200"
    assert 500 not in found, "and nothing outside the tree"
    assert descendants(500, tree) == []


def test_a_close_that_times_out_clears_the_cached_driver_even_if_it_killed_nothing(
        monkeypatch):
    """browsergym keeps one module-global Playwright that every env in the process
    shares, so a driver that has failed must not stay cached: the next `gym.make`
    would use a dead connection and raise an error the gate does not catch.

    I first tied the clearing to a successful kill, reasoning that a slow close
    might leave a healthy driver. Review inverted it, correctly: a close that
    *timed out* is not a slow close that finished, and the case I missed is the
    worse one -- a timeout that kills nothing leaves the cache pointing at that
    driver and every step of the next episode times out."""
    import sys
    import types

    from slime_htm.envs import browser

    core = types.ModuleType("browsergym.core")
    core._PLAYWRIGHT = "a live driver"

    def _set_global_playwright(value):
        core._PLAYWRIGHT = value

    core._set_global_playwright = _set_global_playwright
    package = types.ModuleType("browsergym")
    package.core = core
    monkeypatch.setitem(sys.modules, "browsergym", package)
    monkeypatch.setitem(sys.modules, "browsergym.core", core)

    import os

    monkeypatch.setattr(browser, "descendants",
                        lambda pid, tree=None: [(999, "chrome")])
    # `os` is imported inside the function, so patch the module it reaches.
    monkeypatch.setattr(os, "kill", lambda pid, signal: None)

    import time

    class Wedged:
        def close(self):
            time.sleep(3)

    trouble = browser.close_env(Wedged(), seconds=0.3)
    assert core._PLAYWRIGHT is None and "dropped" in trouble

    # And the case that motivated the inversion: a timeout that kills nothing.
    core._PLAYWRIGHT = "a live driver"
    monkeypatch.setattr(browser, "descendants", lambda pid, tree=None: [])
    trouble = browser.close_env(Wedged(), seconds=0.3)
    assert core._PLAYWRIGHT is None, "the failed driver must not stay cached"
    assert "none found" in trouble, "nothing was killed, and it still cleared"

    # A close that simply succeeds leaves the cache alone.
    core._PLAYWRIGHT = "a live driver"

    class Fine:
        def close(self):
            return None

    assert browser.close_env(Fine(), seconds=5) is None
    assert core._PLAYWRIGHT == "a live driver"


def test_the_run_continues_after_a_browser_had_to_be_killed(monkeypatch, tmp_path):
    """What the kill is for, end to end: the episode after a forced kill gets a
    working env and the run reaches its verdict. Bounded by `--resume` is not the
    same as continuing."""
    import time

    output, rows = gate_in_a_box(monkeypatch, tmp_path, tasks=("t-a", "t-b"))

    import sys

    made = []
    wedged = {"count": 0}
    gym = sys.modules["gymnasium"]
    plain_make = gym.make

    def make(task, **kwargs):
        env = plain_make(task, **kwargs)
        made.append(task)
        if task.endswith("t-a") and wedged["count"] == 0:
            wedged["count"] += 1

            def wedge():
                time.sleep(3)

            env.close = wedge
        return env

    gym.make = make
    from slime_htm.envs import browser

    monkeypatch.setattr(browser, "kill_browser_children", lambda: ["chrome:1"])

    result = run_gate(monkeypatch, output, rows, "--step-timeout", "0.4")

    assert len(made) == 4, "both tasks, both policies: the run did not stop"
    assert len(result["episodes"]) == 4
    assert result["groups"]["l1"]["n"] == 2, "and both pairs were scored"


def test_a_probe_cannot_print_a_verdict(monkeypatch):
    """exps's two-task probe printed a full FAIL with all four threshold lines, in
    a file indistinguishable from a real run's. Below the power table's smallest
    row there is no verdict -- the split is still reported, because a probe's
    numbers are worth reading; what is refused is the word FAIL on two tasks."""
    from slime_htm.envs.gate import MIN_PAIRS, format_groups, four_way, verdict

    assert MIN_PAIRS == 20

    tiny = four_way([(episode(False), episode(True)), (episode(False), episode(True))])
    call = verdict(tiny)
    assert call["pass"] is None
    assert "below the 20" in call["reasons"][0]
    assert len(call["reasons"]) == 1, "and not the four threshold lines"
    printed = format_groups({"click": tiny}).splitlines()
    assert printed[1].rstrip().endswith("--"), "the verdict column says no verdict"
    assert "FAIL" not in printed[1], "and does not say FAIL on the row"

    # At the floor it decides again: twenty tasks both modes answer is the
    # "answerable without thinking" failure, which is a verdict and not a refusal.
    enough = [(episode(True), episode(True))] * MIN_PAIRS
    call = verdict(four_way(enough))
    assert call["pass"] is False
    assert any("without thinking" in reason for reason in call["reasons"])

    # It can only withhold a verdict, never flip one: the same split read at the
    # floor and below it agrees about `pass` wherever both give one.
    assert verdict(four_way(enough), min_pairs=1)["pass"] is False


def test_the_driver_is_recycled_every_n_episodes(monkeypatch, tmp_path):
    """exps's hypothesis, bounded rather than argued: 28816 hung at episode 29,
    after 28 make/close cycles against browsergym's one process-global Playwright.
    A two-episode probe cannot test accumulated state, so no cheap probe will
    settle it -- and the episode count is the only variable here, which is what
    makes this a bound rather than a guess at a cause."""
    import sys
    import types

    output, rows = gate_in_a_box(monkeypatch, tmp_path, tasks=("t-a", "t-b", "t-c"))

    core = types.ModuleType("browsergym.core")
    core._PLAYWRIGHT = "driver"
    drops = []

    def _set_global_playwright(value):
        core._PLAYWRIGHT = value
        drops.append(value)

    core._set_global_playwright = _set_global_playwright
    package = sys.modules["browsergym"]
    package.core = core
    monkeypatch.setitem(sys.modules, "browsergym.core", core)

    # Six episodes: three tasks, two policies. Recycle every two.
    result = run_gate(monkeypatch, output, rows, "--recycle-driver", "2")
    assert len(result["episodes"]) == 6
    assert len(drops) == 3, "after episodes 2, 4 and 6"
    assert result["header"]["recycle_driver"] == 2, "recorded in the header"

    # Zero is the default, and means one driver for the whole run: an unevidenced
    # mitigation is opt-in, because defaulting this to 25 turned a hypothesis exps
    # had explicitly labelled "shape, not finding" into a crash in every run.
    import inspect

    from slime_htm.envs.gate import main as gate_main

    source = inspect.getsource(gate_main)
    assert '"--recycle-driver", type=int, default=0' in source, "off by default"

    drops.clear()
    second = str(tmp_path / "again.json")
    run_gate(monkeypatch, second, str(tmp_path / "again.rows.jsonl"),
             "--recycle-driver", "0")
    assert drops == []


def test_no_verdict_would_pass_a_loose_failure_assertion():
    """Why every failure in this file is asserted as `is False`.

    `pass` became tri-state with MIN_PAIRS, and `not None` is True, so
    `assert not call["pass"]` passes on "no verdict" as happily as on FAIL. A
    fixture that shrank below twenty pairs would then keep its test green while
    testing nothing. Review caught the hazard; this is it written down, because
    the next person to add a verdict test will not have read that message."""
    from slime_htm.envs.gate import four_way, verdict

    nothing = verdict(four_way([(episode(False), episode(True))] * 2))
    assert nothing["pass"] is None
    assert not nothing["pass"], "the trap: a loose assertion passes here"
    assert nothing["pass"] is not False, "and the identity is what refuses it"

    real = verdict(four_way([(episode(False), episode(False))] * 20))
    assert real["pass"] is False, "while a real failure satisfies both"
    assert not real["pass"]


def test_an_always_think_row_does_not_claim_an_index_it_never_had(monkeypatch):
    """always-think writes its action; it does not choose from the candidate set.
    The first version put `chosen = 0` and a uniform distribution in the row, so
    every `always` step claimed `chosen_action` was whatever happened to be offered
    first -- a candidate the model never saw the inside of. exps found it only
    because one row's parse failed and the two fields disagreed: `chosen_action:
    click('13')` beside `'et' is not an element on this page`. A row that cannot
    say what the model wrote is worse than one that says nothing, and
    `position_bias.py` reads `probabilities`."""
    from slime_htm.envs import browser
    from slime_htm.envs.gate import run_episode

    monkeypatch.setattr(browser, "candidates",
                        lambda env, observation, limit, extra=(): (
                            [("click('a')", "click('a')"), ("click('b')", "click('b')")],
                            0))
    monkeypatch.setattr(browser, "state_text",
                        lambda observation, goal, trail, chars: "state")

    never = run_episode(StubEnv("t"), StubDecider(), "t", "never", budget=8,
                        max_steps=2, limit=52, seed=0, state_chars=200)
    always = run_episode(StubEnv("t"), StubDecider(), "t", "always", budget=8,
                         max_steps=2, limit=52, seed=0, state_chars=200)

    # never chose an index, and its row says which.
    assert never["step_rows"][0]["chosen_index"] == 0
    assert never["step_rows"][0]["chosen_action"] == "click('a')"
    assert never["step_rows"][0]["probabilities"] == [0.7, 0.3]

    # always wrote one, and its row says that instead of inventing an index.
    row = always["step_rows"][0]
    assert row["chosen_index"] is None
    assert row["chosen_action"] is None
    assert row["probabilities"] == [], "no distribution it never produced"
    assert row["written_action"] == "click('b')", "what it actually wrote"


def test_nothing_claims_to_read_the_step_distribution_that_does_not():
    """The comment said the distribution is "the quantity `position_bias.py`
    reads". That script takes `litjev-collect` .npz records through `load_records`
    and has never seen a gate row, so no position statistic was exposed to the
    uniform placeholders -- and I had told exps the fix protected it. Twelfth
    instance of prose asserting a property the code does not have, and the first
    where the false claim made a defect sound worse than it was."""
    import inspect
    from pathlib import Path

    from slime_htm.envs.gate import step_record

    source = inspect.getsource(step_record)
    assert "Nothing reads this yet" in source
    script = (Path(__file__).resolve().parents[1] / "scripts/position_bias.py").read_text()
    assert "load_records" in script and "rows" not in script.split('"""')[1], \
        "position_bias reads collect records, not gate rows"


def test_the_readme_does_not_repeat_the_false_reader_claim():
    """The sentence was in the README first and was copied into the code comment,
    which is how it survived -- so correcting only the comment would have left the
    source. A claim about who reads a file is cheap to check and was never checked."""
    from pathlib import Path

    readme = (Path(__file__).resolve().parents[1]
              / "experiments/browser-gate/README.md").read_text()
    assert "`position_bias.py` reads these\nrows" not in readme
    assert "Nothing reads it yet" in readme


def test_an_episode_hung_too_often_is_written_off_not_retried(monkeypatch, tmp_path):
    """The hang exps captured takes the **process** -- a greenlet blocked in
    Playwright cannot be unwound, only exited -- so a launcher loops `--resume`.
    Without a cap, a task that hangs every time loops forever. Written off as
    `process_hang`, incomplete, scored against neither policy."""
    output, rows = gate_in_a_box(monkeypatch, tmp_path, tasks=("t-a", "t-b"))
    first = run_gate(monkeypatch, output, rows)
    assert len(first["episodes"]) == 4

    # Rewrite the rows as if t-b/never had been started three times and never
    # finished: three step rows at three attempt numbers, no episode row.
    lines = Path(rows).read_text().splitlines()
    header = lines[0]
    kept = [line for line in lines
            if '"kind": "episode"' in line and '"task": "browsergym/workarena.servicenow.t-a"' in line]
    tries = []
    for attempt in range(3):
        tries.append(json.dumps({
            "id": f"step:browsergym/workarena.servicenow.t-b|never|{attempt}|0",
            "kind": "step", "attempt": attempt,
            "task": "browsergym/workarena.servicenow.t-b", "policy": "never",
            "step": 0, "candidates": 2, "dropped": 0, "chosen_index": 0,
            "chosen_action": "click('a')", "probabilities": [0.7, 0.3],
            "escalated": False, "generated_tokens": 0, "seconds": 0.1}))
    Path(rows).write_text("\n".join([header, *kept, *tries]) + "\n")

    second = run_gate(monkeypatch, output, rows, "--resume", "--max-attempts", "3")
    written_off = [e for e in second["episodes"]
                   if e["task"].endswith("t-b") and e["policy"] == "never"]
    assert written_off and written_off[0]["ended"] == "process_hang"
    assert written_off[0]["complete"] is False
    # And its pair is dropped from the split rather than scored as a loss.
    assert second["groups"]["l1"]["incomplete_pairs"] >= 1


def test_the_step_guard_has_a_layer_a_blocked_greenlet_cannot_prevent():
    """exps's stack from 28832: `post_step` -> Playwright `context.cookies()`,
    blocked in the greenlet bridge, 300 s past a 120 s deadline. A Python signal
    cannot unwind that. `faulthandler.dump_traceback_later(..., exit=True)` is a
    C-level timer that dumps every thread and `_exit`s even when the main thread
    never returns to the interpreter -- verified here off the main thread, where
    SIGALRM is unavailable, in which case only that layer can act."""
    import inspect

    from slime_htm.envs.browser import step_deadline

    source = inspect.getsource(step_deadline)
    assert "dump_traceback_later" in source and "exit=True" in source
    assert "cancel_dump_traceback_later" in source, "and disarmed after the step"
    assert "hang_multiple" in source, "armed later than SIGALRM, so a recoverable"
    assert "greenlet" in source, "with the case it exists for named"

    # Zero disables both layers rather than arming one of them.
    import time
    with step_deadline(0):
        time.sleep(0.01)


def test_the_launcher_contract_names_the_checkout_requirement():
    """`commit_now` reads `Path.cwd()`, so the gate exits 1 from any directory
    that is neither a checkout nor carries a REVISION file. Deliberate -- a
    verdict that cannot be tied to a reviewed commit is not evidence -- but a
    launcher trips over it rather than reading about it, so it belongs in the
    contract table. Five tests were passing only because pytest happened to run
    from the checkout, which is how this surfaced."""
    from conftest import repo_path

    contract = repo_path("experiments/browser-gate/README.md").read_text()
    row = next((line for line in contract.splitlines()
                if line.startswith("| cwd |")), None)
    assert row, "the contract table has a row for the working directory"
    assert "REVISION" in row and "require_commit" in row
    assert "exits 1" in row, "and says what happens, not just what is needed"


def test_the_contract_tells_the_launcher_not_to_resume_on_exit_three():
    """The guard exists to stop for a person. A launcher that loops to survive the
    hang guard's process exit would relaunch it, drop three more, abort again and
    spend every index's attempt cap -- so the code has to be in the table the
    launcher is written from, not only in the loop."""
    from conftest import repo_path

    from slime_htm.loop import ENVIRONMENT_BROKEN

    contract = repo_path("experiments/browser-gate/README.md").read_text()
    row = next((line for line in contract.splitlines()
                if line.startswith(f"| exit {ENVIRONMENT_BROKEN} |")), None)
    assert row, f"the contract table has a row for exit {ENVIRONMENT_BROKEN}"
    assert "must not resume" in row
    assert "ENVIRONMENT_BROKEN" in row, "and names the constant it comes from"


def test_both_paths_make_their_environment_the_same_way():
    """The gate and `BrowserEnvironment` must not disagree about `tags_to_mark`.

    `BrowserEnvironment` defaults it to `"all"`; the gate's own `gym.make` did
    not pass it, so the gate would have run on browsergym's `standard_html` --
    four SVG tasks with no bid at all -- while the CDPO path, whose precondition
    the gate is, ran on `all`. code-reviewer found it. It is the same shape as
    `content_free`, which lived on the gate's path and not the other one, and
    this pins the pair rather than the instance.
    """
    import ast
    import inspect
    import textwrap

    from slime_htm.envs import browser_environment, gate

    def make_kwargs(module):
        tree = ast.parse(textwrap.dedent(inspect.getsource(module)))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "make"):
                return {kw.arg for kw in node.keywords}
        return None

    gate_kwargs = make_kwargs(gate)
    env_kwargs = make_kwargs(browser_environment)
    assert gate_kwargs is not None and env_kwargs is not None, "both must call gym.make"
    assert "tags_to_mark" in gate_kwargs, "the gate must pass it, not inherit a default"
    assert "tags_to_mark" in env_kwargs
    assert gate_kwargs == env_kwargs, (
        f"the two paths build the environment differently: "
        f"gate {sorted(gate_kwargs)} vs environment {sorted(env_kwargs)}")


def test_the_header_records_every_argument_the_environment_was_built_with():
    """code-reviewer's condition on 9da34cf.

    `header` is both the run's recorded config and what
    `Rows(..., config=header, resume=...)` checks. An argument that reaches
    `gym.make` and not the header leaves the output unable to say which
    environment produced it, and lets `--resume` at one commit mix rows taken
    under two different settings. So the test is over the *set*, not over
    `tags_to_mark`: the next argument added to `gym.make` alone fails here.
    """
    import ast
    import inspect
    import textwrap

    from slime_htm.envs import gate

    source = textwrap.dedent(inspect.getsource(gate.main))
    tree = ast.parse(source)

    made, header = set(), set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "make"):
            made = {kw.arg for kw in node.keywords}
        if isinstance(node, ast.Dict):
            keys = {k.value for k in node.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            if "commit" in keys and "venue" in keys:
                header = keys

    assert made, "gate.main must call gym.make"
    assert header, "gate.main must build the header dict"
    # `headless` is the one exception and it is deliberate: it changes the
    # viewport, so it belongs in the record too -- but it is a machine property
    # rather than a policy one, and `describe_environment` already carries the
    # machine. Named here so the exception is a decision rather than a gap.
    missing = made - header - {"headless", "wait_for_user_message"}
    assert not missing, (
        f"these reach gym.make but are not recorded in the header: {sorted(missing)}")


def test_a_row_too_old_to_answer_the_split_fails_instead_of_answering_wrongly():
    """The regression for 28832, pinned on the access rather than on the writer.

    `episode_record` now serialises `used_outside_action`, so a round trip keeps
    it -- but rows already on disk from before that do not have it, and the whole
    defect was that `.get()` turned the absence into a *plausible answer*: `None`
    is falsy, so every resumed help was counted as `helps_by_reasoning`. 28832
    reported 2 of those and its own step rows say 0.

    So `four_way` must subscript. A file too old to answer this question has to
    fail loudly and be recomputed from the step rows' `outside_fast_set`, which
    is present in every archived gate run.
    """
    old = {"success": True, "tokens": 0, "seconds": 1.0, "dropped_total": 0}
    assert "used_outside_action" not in old
    lost = dict(old, success=False)

    with pytest.raises(KeyError, match="used_outside_action"):
        four_way([(lost, old)])

    # And the same pair with the key present answers, so the failure above is
    # about the missing key rather than about the shape of the fixture.
    answered = four_way([(dict(lost, used_outside_action=False),
                          dict(old, used_outside_action=True))])
    assert answered["helps_by_reach"] == 1.0


def test_the_routing_fit_metric_uses_the_wider_predicate_the_user_chose():
    """`unfit_fast_steps` counts "the fast mode cannot express this", not "this
    needs generated text". The user chose the wider one on 2026-10-06.

    The distinguishing case is a `drag_and_drop` taken under the fast mode: it
    needs no generated text, so the narrow predicate would score it as a fine
    fit, and the wide one scores it as the misfit it is. A mutation swapping the
    two passed every test before this one existed, which is why the counters are
    now a function rather than five expressions inside `run_episode`.
    """
    from slime_htm.envs.gate import routing_fit

    drag_under_fast = [{"escalated": False, "fast_expressible": False,
                        "needs_content": False}]
    assert routing_fit(drag_under_fast)["unfit_fast_steps"] == 1, (
        "a drag chosen by the fast mode is a misfit even though it needs no text; "
        "the narrow predicate would read 0 here")

    click_under_fast = [{"escalated": False, "fast_expressible": True,
                         "needs_content": False}]
    assert routing_fit(click_under_fast)["unfit_fast_steps"] == 0

    # And the three-way split of why each escalation was needed, which is the
    # part that varies. 28831/28832 give 118 / 454 / 137 over 709 always steps.
    mixed = [
        {"escalated": True, "fast_expressible": False},
        {"escalated": True, "fast_expressible": True, "outside_fast_set": True},
        {"escalated": True, "fast_expressible": True, "outside_fast_set": False},
        {"escalated": False, "fast_expressible": True},
    ]
    fit = routing_fit(mixed)
    assert fit["escalated_type"] == 1, "a type the fast mode can never offer"
    assert fit["escalated_element"] == 1, "a type it has, an element it was not offered"
    assert fit["escalated_redundant"] == 1, "already on the fast menu; bought nothing"
    assert fit["fast_steps"] == 1 and fit["unfit_fast_steps"] == 0
