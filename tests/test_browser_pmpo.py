"""The CDPO-against-baselines comparison, without a model or a browser.

What can be wrong without either is what decides whether the comparison means
anything: whether the two heads are fitted to identical rows, whether the
baseline is given everything the method gets except its loss, and whether the
policies are read on tasks no head trained on.
"""
import numpy as np
import pytest

from litjev._legacy.heads import HeadMetadata, build_features
from slime_htm.head_trainer import (
    HeadRLConfig,
    flatten_episodes,
    train_head_on_episodes,
    train_head_on_smeared_returns,
)


def metadata():
    return HeadMetadata(model_id="m", revision="r", hidden_size=8, feature_layers=(-1,))


def corpus(count=24, steps=4, seed=0):
    rng = np.random.default_rng(seed)
    episodes = []
    for index in range(count):
        episodes.append({
            "terminal_reward": 1.0 if index % 3 == 0 else 0.0,
            "head_features": [
                {"step": step, "hidden": rng.normal(size=8),
                 "probabilities": rng.random(4), "candidates": 4,
                 "escalated": bool(step % 2), "generated_tokens": 0,
                 "behaviour_prob": 0.5}
                for step in range(steps)],
        })
    return episodes


def test_both_heads_are_fitted_to_identical_rows():
    """The comparison has to be between two losses, not two collections. Both
    trainers call the same `flatten_episodes`, and the loop hands the corpus to
    the baseline through `on_corpus` rather than collecting again."""
    episodes = corpus()
    config = HeadRLConfig()
    features, actions, behaviour, rewards, spent, index = flatten_episodes(
        episodes, config)

    _, pmpo = train_head_on_episodes(metadata(), episodes, config, seed=0)
    _, smeared = train_head_on_smeared_returns(metadata(), episodes, config, seed=0)

    assert pmpo["decisions"] == smeared["decisions"] == len(features)
    assert pmpo["decisions_per_episode"] == smeared["decisions_per_episode"]


def test_the_baseline_gets_the_same_head_and_differs_only_in_the_loss():
    """The single-question comparison failed because the baseline carried a class
    weighting the method did not. This asserts the trunk, the width and the
    decision rule are shared, so only the loss differs."""
    import inspect

    episodes = corpus()
    config = HeadRLConfig()
    head, head_report = train_head_on_smeared_returns(metadata(), episodes, config, seed=0)
    other, other_report = train_head_on_episodes(metadata(), episodes, config, seed=0)

    # **Structural equality, which is what this test was always about** -- the
    # trunk, the width, the layer and the readout. It used to compare the whole
    # metadata, which also compared the `training` block; each trainer now writes
    # its own fit there, so an equality over everything would be asserting that
    # two different fits recorded the same provenance, which is the fault
    # `with_own_provenance` exists to stop.
    from dataclasses import replace

    assert replace(head.metadata, training={}) == replace(other.metadata, training={})
    assert sum(p.numel() for p in head.parameters()) == \
        sum(p.numel() for p in other.parameters())
    # And the provenance does differ, because the two objectives are not the same
    # fit. Asserted here so the loosening above cannot hide a regression to one
    # shared block.
    assert head.metadata.training != other.metadata.training
    assert head.metadata.training["objective"].startswith("smeared")
    assert "clipped_fraction_mean" in other.metadata.training
    # Both are read through the same gain at evaluation. Checked by driving the
    # rule rather than by looking for the formula's text, which is what this did
    # and which went stale the moment the gain became a log ratio: the assertion
    # then failed for being out of date rather than for anything being wrong.
    import torch

    from litjev.core.block import escalation_gain

    features = np.stack([
        build_features(np.asarray(step["hidden"]),
                       np.asarray(step["probabilities"])[: step["candidates"]], "choice")
        for episode in episodes for step in episode["head_features"]
    ]).astype(np.float32)
    for fitted, report in ((head, head_report), (other, other_report)):
        outcome = torch.softmax(torch.as_tensor(fitted(torch.as_tensor(features))), dim=-1)
        served = np.array([escalation_gain(row) for row in outcome.detach().numpy()])
        assert report["escalation_rate_learned"] == pytest.approx(
            float((served > config.lambda_).mean())), (
            "the rate a head reports must be the rate the serving rule gives")

    source = inspect.getsource(train_head_on_smeared_returns)
    assert "class_weight" not in source, "nothing may handicap the baseline"


def test_the_smeared_label_is_success_not_an_invented_counterfactual():
    """The return is one number, so the only per-step target it supplies is "the
    mode this step took was right", with the whole episode agreeing. Calling a
    successful escalation "fast wrong, slow right" would put a counterfactual in
    the baseline that the data does not contain."""
    config = HeadRLConfig()

    # One succeeding episode: the behaviour's own decisions are the target.
    winner = [{"terminal_reward": 1.0, "head_features": [
        {"step": 0, "hidden": np.zeros(8), "probabilities": np.ones(4) / 4,
         "candidates": 4, "escalated": True, "generated_tokens": 0,
         "behaviour_prob": 0.5}]}]
    _, report = train_head_on_smeared_returns(metadata(), winner, config, epochs=1)
    assert report["cloned_escalation_rate"] == 1.0
    assert report["episodes_succeeded"] == 1.0

    # One failing episode: the complement is.
    loser = [{**winner[0], "terminal_reward": 0.0}]
    _, report = train_head_on_smeared_returns(metadata(), loser, config, epochs=1)
    assert report["cloned_escalation_rate"] == 0.0
    assert report["episodes_succeeded"] == 0.0


def test_the_loop_hands_its_corpus_out_before_training():
    """`on_corpus` exists so the baseline sees the rows the method was fitted to.
    Without it the only way to fit a baseline is a second collection, and the
    comparison becomes about the sample."""
    import inspect

    from slime_htm.loop import run

    source = inspect.getsource(run)
    assert "on_corpus(index, episodes)" in source
    assert source.index("on_corpus(index, episodes)") < source.index(
        "train_head_on_episodes"), "handed out before the method trains on it"


def test_the_policies_are_read_on_tasks_no_head_trained_on():
    """`split_tasks` is what makes that structural, and the runner must use it
    rather than evaluating on the whole group -- which is also why the gate's
    paired episodes are not baseline (C): the gate runs all 54 click tasks."""
    import inspect

    from slime_htm.envs.browser_environment import split_tasks

    training, evaluating = split_tasks([f"t{i:02d}" for i in range(54)])
    assert not set(training) & set(evaluating)
    assert len(evaluating) >= 13

    source = (__import__("pathlib").Path(__file__).resolve().parents[1]
              / "scripts/browser_pmpo.py").read_text()
    assert "split_tasks(found, args.eval_fraction)" in source
    assert "constant_policy" in source, "(C) is measured here, not taken from the gate"
    assert inspect.getdoc(split_tasks)


def test_a_run_with_no_episodes_refuses_rather_than_reporting():
    """Nothing to compare is not a comparison."""
    with pytest.raises(ValueError, match="no decisions"):
        flatten_episodes([], HeadRLConfig())


def script_defaults():
    """The script's own argparse defaults, read from its source."""
    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "scripts/browser_pmpo.py").read_text()
    found = {}
    for name, value in re.findall(r'"--([a-z-]+)", type=(?:int|float), default=([\d.]+)',
                                  source):
        found[name.replace("-", "_")] = float(value)
    return found, source


def test_no_reset_goes_out_of_range_on_the_scripts_defaults():
    """The RED on 193de20: `--episodes 100` and `--eval-episodes 54` indexed past
    the 40 training tasks, so round 0 died at episode 40 before any head was
    trained. code-reviewer reproduced it with the real split and environment, so
    this runs the same path."""
    from slime_htm.envs.browser_environment import BrowserEnvironment, split_tasks

    defaults, _ = script_defaults()
    tasks = [f"browsergym/miniwob.t{index:02d}" for index in range(54)]
    training, evaluating = split_tasks(tasks, defaults["eval_fraction"])

    collected = int(defaults["episodes"]) or 3 * len(training)
    evaluated = int(defaults["eval_episodes"]) or len(evaluating)

    # Collection cycles, so any count is in range and a second pass is a new seed.
    collect = BrowserEnvironment(training, seed=0, cycle=True)
    for episode in range(collected):
        position, pass_number = divmod(episode, len(training))[::-1]
        assert 0 <= position < len(training)
        assert pass_number == episode // len(training)

    # Evaluation does not cycle, and the count is exactly one pass.
    assert evaluated == len(evaluating)
    evaluate = BrowserEnvironment(evaluating, seed=0, cycle=False)
    with pytest.raises(ValueError, match="does not cycle"):
        evaluate.reset(len(evaluating))
    assert collect.episodes == len(training)


def test_the_held_out_set_clears_the_floor_this_repository_refuses_below():
    """`MIN_PAIRS = 20` is the gate's refusal; a headline result should not be held
    to a weaker standard. At the first default of 0.25 the held-out set was 14
    tasks -- one task being 7 percentage points -- which exps raised before the run
    rather than after."""
    from slime_htm.envs.browser_environment import split_tasks
    from slime_htm.envs.gate import MIN_PAIRS

    defaults, source = script_defaults()
    tasks = [f"t{index:02d}" for index in range(54)]
    _, evaluating = split_tasks(tasks, defaults["eval_fraction"])

    assert len(evaluating) >= MIN_PAIRS, "the comparison can reach significance"
    assert defaults["min_eval_tasks"] == MIN_PAIRS
    assert "--min-eval-tasks" in source, "and a smaller group is refused at startup"
    # The old default would not have cleared it, which is why it is not the default.
    _, small = split_tasks(tasks, 0.25)
    assert len(small) < MIN_PAIRS


def test_each_episode_is_written_as_it_finishes_not_at_the_end_of_the_round():
    """exps asked twice what survives a kill, and the second answer was still
    wrong: the `.npz` was written **after** a round completed, so at `--rounds 1`
    -- now the default -- a death during collection left nothing and the restart
    began at episode zero. From tonight's hang rate, about 1% an episode over 194
    episodes, that is a **57% chance** the collection dies, and the new guard makes
    it the expected path because exiting is its whole job.

    Streamed per episode now, which is why the gate survived both of tonight's
    hangs with one episode lost each."""
    import inspect

    from slime_htm.loop import corpus_path, load_corpus, run

    source = inspect.getsource(run)
    assert "append(" in source, "appended per episode, through `resume.append_row`"
    assert "np.savez_compressed" not in source, "not at the end of the round"
    assert "indices=pending" in source, "and a restart runs the ones it is missing"
    assert "first=len(" not in source, "by index and never by count"
    assert str(corpus_path("out", 0)).endswith("episodes-round0.jsonl")
    assert load_corpus(corpus_path("out", 99)) == [], "a missing stream is empty"


def test_a_written_corpus_can_be_trained_on_after_a_round_trip(tmp_path):
    """So a resume or a re-analysis uses the rows the round actually collected."""
    import json

    from slime_htm.loop import _jsonable, load_corpus

    episodes = corpus(count=4, steps=3)
    path = tmp_path / "episodes-round0.npz"
    np.savez_compressed(path, episodes=np.array(
        json.dumps(episodes, default=_jsonable), dtype=object))

    back = load_corpus(path)
    assert len(back) == len(episodes)
    features, *_ = flatten_episodes(back, HeadRLConfig())
    assert features.shape[0] == 4 * 3
    head, report = train_head_on_episodes(metadata(), back, HeadRLConfig(), seed=0)
    assert report["decisions"] == 12


def test_the_help_text_says_the_number_the_code_uses():
    """It said "two passes" while the code did three. Eleventh instance of the same
    pattern in this repository -- prose asserting a property the code does not have
    -- and the first in a `--help` string, which is the copy a launcher reads."""
    import re

    _, source = script_defaults()
    help_text = re.search(r'"--episodes".*?help="(.*?)"\)', source, re.S).group(1)
    passes = re.search(r"\*\*(\w+)\*\* passes", help_text)
    assert passes and passes.group(1) == "three"
    assert "3 * len(training)" in source, "and the code is what it describes"


def test_the_engine_cannot_be_in_the_browsers_process():
    """28840 died at the first scoring call of the first episode: Playwright's sync
    API runs an event loop on the calling thread and SGLang's in-process `Engine`
    drives its own with `run_until_complete`. The gate never met it because it has
    always gone over HTTP with `urllib`, which needs no loop -- so the project had
    the right architecture for a different stated reason, and the conflict was
    written down nowhere."""
    _, source = script_defaults()

    assert "SGLangScorer" not in source, "no in-process engine beside Playwright"
    assert "ClientScorer" in source and "--decision-url" in source
    assert "event loop" in source, "and the reason is in the flag's help"


def test_the_client_scorer_matches_the_in_process_one():
    """`rollout_episodes` calls `fast`, `slow` and `solve` and does not care which
    it has. A surface that drifted would fail at episode one, in a job."""
    import inspect

    from slime_htm import ClientScorer, StepScorer

    for name in ("fast", "slow", "solve"):
        mine = inspect.signature(getattr(ClientScorer, name))
        theirs = inspect.signature(getattr(StepScorer, name))
        assert list(mine.parameters) == list(theirs.parameters), name

    # **And how many values each returns**, which this test did not check and
    # which is where the surface actually drifted: both scorers return three from
    # `slow`, the rollout unpacked two, and every test double here returned two,
    # so the suite agreed with itself and not with the code that runs.
    import numpy as np

    class Field:
        logits = [0.0, 1.0]
        hidden = np.zeros(4, dtype=np.float32)
        generated_tokens = 5
        reasoning = "because"

    class Provider:
        def score(self, context, schema):
            return (Field(),)

        def think(self, context, schema, fields, budget):
            return (Field(),)

        def solve(self, context, instructions, budget):
            return "click(1)", 5, False

    scorer = StepScorer(Provider())
    assert len(scorer.fast("ctx", ["a", "b"])) == 2
    assert len(scorer.slow("ctx", ["a", "b"], 8)) == 3
    assert len(scorer.solve("ctx", "do it", 8)) == 3


def test_the_fast_path_asks_for_the_hidden_state_and_refuses_without_it():
    """CDPO trains on the readout state at every step, so a service that cannot
    supply it cannot collect for training. Failing at the probe costs one request;
    failing at episode one costs a job."""
    from slime_htm import ClientScorer

    class Service:
        def __init__(self, hidden):
            self.hidden = hidden
            self.asked = []

        def fast(self, state, actions, want_hidden=False):
            self.asked.append(want_hidden)
            answer = {"probabilities": [0.6, 0.4]}
            if want_hidden and self.hidden is not None:
                answer["hidden_state"] = self.hidden
            return answer

    good = Service([0.1] * 16)
    probabilities, hidden = ClientScorer(good).fast("state", ["a", "b"])
    assert good.asked == [True], "it asks, every step"
    assert len(hidden) == 16 and len(probabilities) == 2

    with pytest.raises(RuntimeError, match="no hidden state"):
        ClientScorer(Service(None)).fast("state", ["a", "b"])


def test_one_hidden_state_is_not_the_48mb_the_service_exists_to_avoid():
    """The figure in serve_decision.sbatch is sglang's own server encoding a state
    for **every prompt position**. This ships one vector, and the distinction is
    what makes the fix a port rather than a design conflict."""
    from inference.client import _decode_hidden
    from inference.decision_service import decode_hidden, encode_hidden

    vector = np.random.default_rng(0).normal(size=5128).astype(np.float32)
    encoded = encode_hidden(vector)

    assert len(encoded["data"]) < 30 * 1024, "about 27 KB, against a step of seconds"
    assert np.array_equal(decode_hidden(encoded), vector), "exact, not float16"
    # The client decodes with the standard library only: an environment runs where
    # its own dependencies are, and ScienceWorld's had a JVM and no numpy.
    assert np.array_equal(np.array(_decode_hidden(encoded), dtype=np.float32), vector)
    assert encode_hidden(None) is None and decode_hidden(None) is None


def test_the_startup_probe_checks_the_width_and_not_only_the_presence():
    """Recommended in review, in the spirit of everything else today: a service
    holding another model or another head would otherwise collect a whole round --
    hours -- and fail only when `DecisionHead(metadata)` met wrong-width features.
    The probe costs one request."""
    _, source = script_defaults()

    assert "metadata.hidden_dim" in source, "the probe compares, not just prints"
    # Both numbers in the message: "a shape mismatch" without them tells a
    # launcher nothing about which side to change.
    assert "metadata.hidden_size} x " in source
    assert "different model or a different head" in source

    # And the relationship it checks is the one the head actually uses.
    meta = HeadMetadata(model_id="m", revision="r", hidden_size=4096,
                        feature_layers=(-1, 24))
    assert meta.hidden_dim == 4096 * 2
    features = build_features(np.zeros(meta.hidden_dim, dtype=np.float32),
                              np.array([0.5, 0.5]), "choice")
    assert features.shape[0] == meta.input_dim


def test_thinking_is_priced_so_pmpo_is_not_trained_to_escalate_everywhere():
    """exps's first reason for holding 28842, verified from their numbers and mine.
    `value = reward - cost_weight * spent`, so at `cost_weight = 0` the return is
    pure reward, thinking is free, and the reward-maximising policy is to think
    everywhere -- CDPO would be trained to do the opposite of what it is for. And
    28831 measured hurts = 0, so predicted gain is non-negative almost everywhere
    and `gain > lambda` at lambda 0 escalates almost everywhere too."""
    defaults, source = script_defaults()

    assert defaults["cost_weight"] > 0, "a zero price trains it to escalate always"
    assert "think everywhere" in source, "and the flag's help says why"

    # The mechanism, so the default is not the only thing holding it.
    episodes = corpus(count=4, steps=4)
    from slime_htm.head_trainer import priced_value

    _, _, _, rewards, spent, _ = flatten_episodes(episodes, HeadRLConfig())
    free = priced_value(rewards, spent, 0.0)
    priced = priced_value(rewards, spent, 0.05)
    assert (free > priced).any(), "a priced value is lower where steps escalated"
    # The price discounts rather than subtracts, so what is constant is the ratio
    # and not the difference: two escalations per four-step episode at 0.05 keeps
    # exp(-0.1) of the value, not 0.1 less of it.
    escalating = free > 0
    assert np.allclose(priced[escalating] / free[escalating], np.exp(-0.1))


def test_the_run_produces_a_curve_because_the_accuracy_ceiling_is_shared():
    """exps's second reason: 28831 has hurts = 0, so always-think **is** the oracle
    -- `oracle` and `always` are both 0.4815 -- and no router can beat it on
    accuracy. A tie at the top is the best accuracy outcome available, which makes
    cost the only axis a win can live on, and two heads with one ceiling are
    distinguished by their accuracy-cost curve rather than by one point."""
    defaults, source = script_defaults()

    assert "--lambdas" in source and "0.0,0.1,0.2" in source
    assert 'price_{price:g}' in source, "every price is its own reported row"
    assert "ceiling" in source and "oracle == always" in source, \
        "and the file says what it cannot show"


def test_one_round_by_default_because_later_rounds_lose_the_declines():
    """Round 0 explores at `--explore-prob`, so its corpus has balanced
    escalate/decline decisions. Later rounds collect under the learned head, and
    with hurts = 0 that head escalates almost everywhere, so coverage collapses
    exactly where it has to learn to decline. CDPO is off-policy by construction,
    which is what makes one collection sufficient rather than a compromise."""
    import inspect

    from slime_htm.rollout import rollout_episodes

    defaults, source = script_defaults()
    assert defaults["rounds"] == 1
    assert "coverage would collapse" in source

    # The loop explores only while there is no head, which is the mechanism.
    loop_source = inspect.getsource(rollout_episodes)
    assert "escalate_prob is not None" in loop_source


def summariser():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/browser_pmpo.py"
    spec = importlib.util.spec_from_file_location("browser_pmpo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.summarise


class FakeResult:
    def __init__(self, reward, escalations, steps, ended="done"):
        self.reward = reward
        self.escalations = escalations
        self.steps = [0] * steps
        self.finished = True
        self.generated_tokens = 0
        # Shares `EpisodeResult`'s fields, including the one `summarise` counts:
        # a double that drops a field its original has is how a test passes
        # against a shape nothing produces.
        self.ended = ended


def test_the_headline_metric_is_reward_net_of_what_thinking_cost():
    """The user's correction: the metric was wrong, not the venue. A reader given
    accuracy and escalation side by side is left to combine them,
    while the objective already says how. `value = reward - price x escalations`
    is the quantity CDPO optimises, and it separates two policies that score the
    same and pay differently."""
    summarise = summariser()

    # Same accuracy, half the escalations.
    always = [FakeResult(1.0, 5, 5)] * 12 + [FakeResult(0.0, 5, 5)] * 15
    router = [FakeResult(1.0, 2, 5)] * 12 + [FakeResult(0.0, 2, 5)] * 15

    assert summarise(always)["reward"] == summarise(router)["reward"], "a tie"
    # At a price of nothing, a tie is all there is -- which is why price 0 cannot
    # discriminate and was the defect exps held the run over.
    assert summarise(always, 0.0)["value"] == summarise(router, 0.0)["value"]
    # At any price, declining where it is safe wins on the same accuracy.
    for price in (0.05, 0.1, 0.2):
        assert summarise(router, price)["value"] > summarise(always, price)["value"]

    exact = summarise(router, 0.1)
    assert exact["value"] == pytest.approx(12 / 27 - 0.1 * 2)
    assert exact["escalation_rate"] == pytest.approx(2 / 5)


def test_the_constant_policies_are_scored_at_every_price_without_rerunning():
    """Their behaviour does not depend on the price, so their value at each price
    is arithmetic on the same episodes. Re-running them per price would be another
    54 episodes a price for no new information."""
    summarise = summariser()
    _, source = script_defaults()

    always = [FakeResult(1.0, 5, 5)] * 2
    assert summarise(always, 0.0)["value"] > summarise(always, 0.2)["value"]
    assert "re-scored rather than" in source, "and the script says so"
    # Once as a definition and once as a call: not once per price.
    assert source.count("constant_policy(") == 2


def test_no_summary_claims_a_value_without_a_price():
    """A value is only defined against a price, and a row carrying one without
    saying which price would be unreadable."""
    summarise = summariser()

    assert "value" not in summarise([FakeResult(1.0, 1, 1)])
    assert "price" not in summarise([FakeResult(1.0, 1, 1)])
    assert summarise([], 0.1) == {"episodes": 0}


def test_the_value_a_decision_is_judged_by_moves_with_the_price_it_is_asked_about():
    """The user's correction, end to end. The reward was "did the episode succeed",
    priced once at a fixed `cost_weight`, while the price the policy term asked
    about was drawn per decision -- so escalating on a high-advantage episode was
    rewarded whatever price it was being asked about. One price now, in both."""
    from slime_htm.head_trainer import flatten_episodes, priced_value

    # One success that thought at every step, one failure that thought at none.
    thought = {"terminal_reward": 1.0, "head_features": [
        {"step": i, "hidden": np.zeros(8), "probabilities": np.ones(4) / 4,
         "candidates": 4, "escalated": True, "generated_tokens": 0,
         "behaviour_prob": 0.5} for i in range(4)]}
    declined = {"terminal_reward": 0.0, "head_features": [
        {**step, "escalated": False} for step in thought["head_features"]]}

    _, _, _, rewards, spent, _ = flatten_episodes([thought, declined], HeadRLConfig())
    cheap = priced_value(rewards, spent, 0.05)
    dear = priced_value(rewards, spent, 0.30)

    assert cheap[0] > cheap[4], "cheap thinking: the success is worth more"
    # A dear price cannot push the success below the failure, because the discount
    # is multiplicative and the failure is zero whatever it spent. That inversion
    # was what a linear price allowed, and removing it is the point: cost alone
    # should not make a right answer worth less than a wrong one.
    #
    # What a dear price does instead, and what the head is trained on, is leave
    # far less of the win standing.
    assert dear[0] > dear[4] == 0.0
    # Exactly, rather than by some bound: raising the price by 0.25 over four
    # escalations leaves exp(-1.0) of what the cheap price left.
    assert dear[0] / cheap[0] == pytest.approx(np.exp(-(0.30 - 0.05) * 4))
    # Which is the ordering that makes one head serve every price, and it is
    # invisible if the price is not in the reward.
    assert priced_value(rewards, spent, 0.0)[0] > priced_value(rewards, spent, 0.0)[4]


def test_the_unpriced_baseline_clones_success_and_not_cheapness():
    """With `price=None` it smears the **raw** reward: a practitioner with no
    counterfactual clones what worked, and "worked" means the task was completed,
    not that it was completed cheaply. The priced variant is a separate baseline
    rather than a replacement, so the run can say which of the two a win is
    against."""
    import inspect

    from slime_htm.head_trainer import train_head_on_smeared_returns

    source = inspect.getsource(train_head_on_smeared_returns)
    assert "rewards > 0" in source, "the unpriced branch"
    assert "if price is None:" in source, "and it is a branch, not the only path"


def test_there_are_two_cross_entropy_baselines_and_why():
    """Unpriced says the price in the objective mattered. **Priced** optimises the
    same quantity CDPO does, fitted by cross-entropy instead of a clipped policy
    gradient, so a win over it is attributable to what CDPO actually adds -- the
    per-decision importance ratio and the clip. Without the second, the sharpest
    objection to a CDPO win is that CDPO was handed the price and the baseline was
    not; code-reviewer reached the same conclusion independently."""
    from slime_htm.head_trainer import train_head_on_smeared_returns

    defaults, source = script_defaults()
    assert '"smeared_ce", None' in source and '"priced_ce"' in source
    assert "importance\n    # ratio and the clip" in source or "importance" in source

    episodes = corpus(count=12, steps=4)
    config = HeadRLConfig()
    _, unpriced = train_head_on_smeared_returns(metadata(), episodes, config, epochs=3)
    _, priced = train_head_on_smeared_returns(metadata(), episodes, config, epochs=3,
                                              price=0.125)

    assert unpriced["price"] is None and priced["price"] == 0.125
    assert unpriced["objective"] == "smeared_return_cross_entropy"
    assert priced["objective"] == "smeared_priced_value_cross_entropy"


def test_the_priced_baseline_labels_by_the_same_advantage_sign_pmpo_uses():
    """So the two differ in how they fit it and not in what they are fitting."""
    import inspect

    from slime_htm.head_trainer import train_head_on_smeared_returns

    source = inspect.getsource(train_head_on_smeared_returns)
    assert "priced_value(rewards, spent, prices)" in source
    assert "value > value.mean()" in source, "the sign of CDPO's own advantage"
    # And the price is drawn per decision and redrawn per epoch, as CDPO does, so
    # the two differ in the loss and not also in price marginalisation.
    assert "draw_prices(len(features), config, generator)" in source
    assert source.count("draw_prices") >= 2, "once to start, once per epoch"


def test_the_table_has_a_column_for_every_policy_compared():
    """A reader should not have to join two files to see the comparison."""
    _, source = script_defaults()

    assert '"never", "always", "smeared_ce", "priced_ce", "pmpo"' in source
    assert "winner" in source, "and which one won at that price"


def test_the_comparison_is_paired_by_task_and_carries_a_p_value():
    """What turns the table into a claim. Episodes are paired -- the same 27
    held-out tasks at the same seed under each policy -- so the test is within
    task, and it is the gate's own exact one-sided sign test on the discordant
    pairs. A mean difference over 27 tasks with no test is a number, and this
    repository refuses a verdict without one elsewhere."""
    paired = None
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/browser_pmpo.py"
    spec = importlib.util.spec_from_file_location("browser_pmpo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paired = module.paired_test

    cheap = [FakeResult(1.0, 1, 5)] * 12 + [FakeResult(0.0, 1, 5)] * 15
    dear = [FakeResult(1.0, 5, 5)] * 12 + [FakeResult(0.0, 5, 5)] * 15

    # At a price of nothing there is nothing to find, and it says so.
    flat = paired(cheap, dear, 0.0)
    assert flat["ahead"] == 0 and flat["tied"] == 27 and flat["sign_test_p"] == 1.0

    # At a price, the cheaper policy is ahead on every task.
    won = paired(cheap, dear, 0.1)
    assert won["ahead"] == 27 and won["behind"] == 0
    assert won["sign_test_p"] < 1e-6

    # And it is symmetric: the loser's test refuses.
    lost = paired(dear, cheap, 0.1)
    assert lost["ahead"] == 0 and lost["sign_test_p"] == 1.0

    # The one-sided value is not a test of "no difference", and reading it as one
    # put p = 0.42 and p = 0.063 into the paper where the two-sided values are
    # 0.83 and 0.125. So both tails are returned, and the two-sided one is what a
    # question without a pre-specified direction gets quoted. These four pairs are
    # the ones that were actually misreported, checked against the closed form
    # 2 * sum(C(n, k) for k <= min(ahead, behind)) / 2**n, clipped at one.
    for ahead, behind, one_sided, two_sided in [
        (4, 5, 0.7461, 1.0),  # AnyJev vs fast-only: a one-sided 0.75 read as "same"
        (5, 7, 0.8062, 0.7744),
        (12, 10, 0.4159, 0.8318),  # the "thinking is a wash" test
        (6, 1, 0.0625, 0.125),  # 0.063 reads as marginal; 0.125 does not
    ]:
        mine = [FakeResult(1.0, 1, 5)] * ahead + [FakeResult(0.0, 1, 5)] * behind
        theirs = [FakeResult(0.0, 1, 5)] * ahead + [FakeResult(1.0, 1, 5)] * behind
        got = paired(mine, theirs, 0.0)
        assert (got["ahead"], got["behind"]) == (ahead, behind), got
        assert abs(got["sign_test_p"] - one_sided) < 5e-4, got
        assert abs(got["sign_test_p_two_sided"] - two_sided) < 5e-4, got
        # Where the one-sided test looks in the observed direction the relation
        # is exact doubling, which is what makes quoting the one-sided value a
        # halving of the p: by symmetry P(X >= ahead) = P(X <= behind) at
        # n = ahead + behind. It does NOT hold the other way round -- at 5 ahead
        # and 7 behind the one-sided 0.806 exceeds the two-sided 0.774, because a
        # one-sided test of a direction that lost is above 0.5 by construction.
        # An earlier version of this test asserted two-sided >= one-sided
        # unconditionally and that pair is the counterexample.
        if ahead >= behind:
            assert abs(got["sign_test_p_two_sided"]
                       - min(1.0, 2 * got["sign_test_p"])) < 1e-12
        else:
            assert got["sign_test_p"] > 0.5

    # With nothing discordant neither tail is defined and both say so, rather
    # than the two-sided doubling an undefined 1.0 into 2.0.
    assert flat["sign_test_p_two_sided"] == 1.0


def test_every_head_is_fitted_at_several_seeds_and_the_reads_are_rationed():
    """A one-seed difference over 27 tasks cannot tell the objective from training
    noise. Training is CPU-seconds and reading costs episodes, so every head is
    fitted at every seed and only the reads are rationed: the full curve at the
    first seed, the others at the middle price."""
    defaults, source = script_defaults()

    assert defaults["train_seeds"] == 3
    assert "prices if seed == seeds[0] else [middle]" in source
    assert "seed_spread" in source, "and the spread is reported"
    assert "training noise" in source, "with the reason in the flag's help"


def test_the_priced_baseline_marginalises_over_the_price_as_pmpo_does():
    """code-reviewer's condition: passing one fixed midpoint left the two differing
    in the loss **and** in price marginalisation, so a CDPO win could not be
    credited to the importance ratio and the clip -- which is what this file's own
    sentence claimed. `price="drawn"` takes a price per decision from the same
    `draw_prices` and redraws it every epoch."""
    from slime_htm.head_trainer import train_head_on_smeared_returns

    _, source = script_defaults()
    assert '("priced_ce", "drawn")' in source, "not a fixed midpoint"

    config = HeadRLConfig(lambda_range=(0.0, 0.25))
    episodes = corpus(count=12, steps=4)
    _, drawn = train_head_on_smeared_returns(metadata(), episodes, config,
                                             epochs=3, price="drawn")
    assert drawn["price"] == "drawn"
    assert drawn["objective"] == "smeared_priced_value_cross_entropy"

    # Same seed, same draws: the baseline is reproducible against the method.
    _, again = train_head_on_smeared_returns(metadata(), episodes, config,
                                             epochs=3, price="drawn", seed=0)
    assert again["cloned_escalation_rate"] == drawn["cloned_escalation_rate"]


def test_one_primary_test_is_pre_registered_and_the_rest_are_labelled():
    """Twelve sign tests read uncorrected would let one p < 0.05 in twelve be
    reported as the result. The primary is named in the file and in the output
    before the run."""
    from conftest import load_script

    _, source = script_defaults()
    module = load_script("browser_pmpo")

    # **Which comparison is primary, asked of the function rather than of the
    # source.** This assertion used to read `'"against": "priced_ce"' in source`
    # and broke when the dict became a call -- a string test that noticed a
    # refactor and would not have noticed the opponent changing to `smeared_ce`.
    cell = [module.Played(episode=0, reward=1.0, finished=True,
                          steps=[{"escalated": False, "generated_tokens": 0}])]
    found = module.primary_verdict({("pmpo", 0.1): cell,
                                    ("priced_ce", 0.1): cell}, 0.1, seed=0)
    assert found["mine"] == "pmpo"
    assert found["against"] == "priced_ce", "the comparison that isolates CDPO"
    assert "reverse" in found, "and the other direction, so a loss has a threshold"

    assert '"primary_test"' in source
    assert "PRIMARY" in source and "(secondary)" in source, "and the rest are marked"
    assert "p < 0.05 in twelve" in source, "with the reason stated"
    assert "Twelve sign tests" in source, "and the count named"


def test_a_partial_collection_round_trips_and_resumes(tmp_path):
    """What makes a launcher loop terminate: the episodes already streamed are
    read back and the rollout restarts after them."""
    import json

    from slime_htm.loop import _jsonable, corpus_path, load_corpus

    stream = tmp_path / "episodes-round0.jsonl"
    episodes = corpus(count=5, steps=3)
    with open(stream, "a") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode, default=_jsonable) + "\n")

    back = load_corpus(stream)
    assert len(back) == 5, "five already collected, so the restart asks for the rest"
    features, *_ = flatten_episodes(back, HeadRLConfig())
    assert features.shape[0] == 15, "and they are trainable, not just present"

    # A truncated last line -- a kill mid-write -- loses that episode and no more.
    # It cannot raise: the guard exits the process at an arbitrary instant and a
    # row of 25 steps is far past the write buffer, so a resume that raised here
    # would raise on every later resume too.
    with open(stream, "a") as handle:
        handle.write('{"terminal_reward": 1.0, "head_feat')
    assert len(load_corpus(stream)) == 5, "the truncated row is dropped, the rest stay"
    assert corpus_path(tmp_path, 0) == stream

    # **And then the resume appends.** code-reviewer: `append_row` wrote straight
    # onto the fragment, gluing the next row to it -- so the fragment stopped
    # being the *last* line, and from the second resume on every read raised. One
    # lost episode became a lost run, which is what this is here to prevent. The
    # partial line is closed first, as `litjev.streaming.Rows` has always done.
    from slime_htm.resume import append_row

    for episode in corpus(count=2, steps=3, seed=7):
        append_row(stream, episode)
    back = load_corpus(stream)
    assert len(back) == 7, "five, the fragment dropped, and the two appended after it"
    features, *_ = flatten_episodes(back, HeadRLConfig())
    assert features.shape[0] == 21, "and all seven are trainable"


def test_the_rollout_path_has_the_same_guard_as_the_gate(monkeypatch, tmp_path):
    """code-reviewer: `browser_pmpo` never calls `gate.run_episode`, so the gate's
    guard was not on the path the CDPO run uses -- and 28832's hang was
    `env.step` -> `post_step` -> `cookies()`, the call the rollout makes every
    step. An episode whose step does not return ends **incomplete** and is dropped
    by the collector, because training on it would score a browser that stopped
    answering as a failure the policy caused."""
    import time

    from slime_htm.envs.browser_environment import BrowserEnvironment

    class Hangs:
        def __init__(self):
            self.closed = False

        def reset(self, seed=None):
            return {"goal": "g", "extra_element_properties": {"a": {"clickable": True}}}, {}

        def step(self, action):
            time.sleep(3)
            raise AssertionError("not reached")

        def close(self):
            self.closed = True

    import sys
    import types

    gym = types.ModuleType("gymnasium")
    gym.make = lambda task, **kwargs: Hangs()
    gym.registry = {"t": object()}
    monkeypatch.setitem(sys.modules, "gymnasium", gym)
    # The names `browser_environment` imported, not the ones in `browser`: the
    # module binds them at import, so patching the source module would leave the
    # real `candidates` in place and it reaches for browsergym.
    from slime_htm.envs import browser_environment as module

    monkeypatch.setattr(module, "candidates",
                        lambda env, observation, limit, extra=(): (
                            [("click('a')", "click('a')")], 0))
    monkeypatch.setattr(module, "state_text",
                        lambda observation, goal, trail, history=6, chars=4000: "s")

    env = BrowserEnvironment(["t"], step_timeout=0.4,
                             stack_file=str(tmp_path / "stacks.log"))
    _, actions = env.reset(0)
    state, actions, done, reward = env.step(actions[0])

    assert done is True and reward == 0.0, "the episode ends here"
    assert actions == [], "and offers nothing further"
    assert env.ended == "step_timeout"
    assert env.complete is False, "so the collector drops it"


class Toy:
    """Two steps and a terminal reward, with an episode index it can refuse to finish.

    `incomplete` names the episode indices whose step does not return -- the
    `step_timeout` shape -- so `complete` is False there, exactly as
    `BrowserEnvironment` reports it after the deadline fires.
    """

    def __init__(self, incomplete=(), seen=None, once=(), die_at=None):
        self.incomplete, self.seen = set(incomplete), seen if seen is not None else []
        self.once, self.die_at = set(once), die_at
        self.episode, self.position, self.complete = 0, 0, True
        self.ended, self.failing = None, False

    def reset(self, episode):
        if self.die_at is not None and episode == self.die_at:
            # What the guard does to a hung process: it exits. The episodes
            # already streamed are all the next process has.
            raise SystemExit(f"killed at episode {episode}")
        self.episode, self.position = episode, 0
        self.complete, self.ended = True, None
        # `incomplete` fails every time; `once` fails the first time only.
        self.failing = episode in self.incomplete or episode in self.once
        self.once.discard(episode)
        self.seen.append(episode)
        return f"task {episode}", ["a", "b"]

    def step(self, action):
        self.position += 1
        if self.failing:
            self.complete, self.ended = False, "step_timeout"
            return "", [], True, 0.0
        done = self.position >= 2
        return f"task {self.episode}", [] if done else ["a", "b"], done, float(done)

    def close(self):
        pass


class ToyScorer:
    def fast(self, context, actions):
        return np.array([0.6, 0.4]), np.arange(8, dtype=np.float32)

    def slow(self, context, actions, budget):
        return np.array([0.9, 0.1]), budget, "reasoning"


def test_a_dropped_episode_is_retried_by_index_and_duplicates_none(tmp_path):
    """code-reviewer, on 4c872a2: resume was by **count**, and the stream holds only
    the episodes that were kept. Ten attempted with episode 5 dropped leaves nine
    rows, so a count-based restart began at 9 -- re-running episode 9, which the
    corpus then held twice, and never retrying 5. Every row carries its index and
    what is still missing is the set difference.

    The retry happens in **this** process, before the corpus is handed to the
    trainers: a corpus a later resume could still add to would mean the
    evaluation cells cached by this process were played by a head fitted to fewer
    episodes than the ones cached by the next."""
    from slime_htm.loop import LoopConfig, corpus_path, load_corpus, run, trainable

    config = LoopConfig(rounds=1, episodes_per_round=6, explore_prob=1.0,
                        head=HeadRLConfig(epochs=1),
                        output=str(tmp_path / "out"))
    seen = []
    run(metadata(), lambda index, evaluation: Toy(once={2, 4}, seen=seen),
        ToyScorer(), config, seed=0)

    assert seen == [0, 1, 2, 3, 4, 5, 2, 4], "the two drops are retried, by index"
    kept = trainable(load_corpus(corpus_path(tmp_path / "out", 0)))
    indices = sorted(row["index"] for row in kept)
    assert indices == [0, 1, 2, 3, 4, 5], "all six"
    assert len(indices) == len(set(indices)), "once each, no duplicate in the corpus"


def test_a_killed_collection_resumes_where_it_stopped(tmp_path):
    """The guard that catches the Playwright greenlet works by **exiting**, so this
    is the expected path. What the next process must not do is start at zero --
    81 episodes is most of a day -- or restart at a count, which after a drop is a
    different episode than the one it is missing."""
    from slime_htm.loop import LoopConfig, corpus_path, load_corpus, run, trainable

    config = LoopConfig(rounds=1, episodes_per_round=6, explore_prob=1.0,
                        head=HeadRLConfig(epochs=1),
                        output=str(tmp_path / "out"))
    first = []
    with pytest.raises(SystemExit):
        run(metadata(), lambda index, evaluation: Toy(once={1}, seen=first, die_at=4),
            ToyScorer(), config, seed=0)

    stream = corpus_path(tmp_path / "out", 0)
    assert first == [0, 1, 2, 3], "episode 1 was dropped and 4 took the process"
    assert [row["index"] for row in trainable(load_corpus(stream))] == [0, 2, 3]

    second = []
    run(metadata(), lambda index, evaluation: Toy(seen=second), ToyScorer(),
        config, seed=0)

    assert second == [1, 4, 5], "the dropped one and the ones never reached"
    indices = sorted(row["index"] for row in trainable(load_corpus(stream)))
    assert indices == [0, 1, 2, 3, 4, 5] and len(indices) == len(set(indices))


def test_an_index_that_fails_every_time_is_given_up_on(tmp_path):
    """Without the cap a launcher loop re-invoking this script retries a task that
    takes the process for as long as the job lives. The attempt is recorded
    **before** the episode runs, which is the only way a failure that leaves
    nothing behind gets counted."""
    from slime_htm.loop import LoopConfig, corpus_path, load_corpus, run, trainable

    config = LoopConfig(rounds=1, episodes_per_round=3, explore_prob=1.0,
                        max_attempts=2, head=HeadRLConfig(epochs=1),
                        output=str(tmp_path / "out"))
    stream = corpus_path(tmp_path / "out", 0)
    attempted = []
    for _ in range(4):
        run(metadata(), lambda index, evaluation: Toy(incomplete={1}, seen=attempted),
            ToyScorer(), config, seed=0)

    assert attempted == [0, 1, 2, 1], "index 1 is tried twice and then given up on"
    rows = load_corpus(stream)
    assert sorted(row["index"] for row in trainable(rows)) == [0, 2]
    given = [row for row in rows if row.get("ended") == "given_up"]
    assert [row["index"] for row in given] == [1], "recorded once, not once per resume"


def test_the_evaluation_resumes_too_because_it_is_the_larger_half(tmp_path):
    """The evaluation is about 460 of this run's 540 episodes and had no resume at
    all: a hang anywhere in it sent the next process back to its first episode. At
    the hang rate exps measured, a 460-episode phase that restarts from zero
    finishes about once in a hundred attempts."""
    from conftest import load_script

    module = load_script("browser_pmpo")

    path = tmp_path / "eval-never.jsonl"
    seen = []
    with pytest.raises(SystemExit):
        module.play(path, lambda: Toy(once={1}, seen=seen, die_at=3), ToyScorer(),
                    None, price=0.0, episodes=4, budget=4, max_steps=5, seed=0,
                    max_attempts=3, escalate_prob=0.0)
    assert seen == [0, 1, 2], "1 was dropped and 3 took the process"

    again = []
    played = module.play(path, lambda: Toy(seen=again), ToyScorer(), None,
                         price=0.0, episodes=4, budget=4, max_steps=5, seed=0,
                         max_attempts=3, escalate_prob=0.0)
    assert again == [1, 3], "the dropped one and the one never reached"
    assert sorted(played) == [0, 1, 2, 3], "so every task is pairable again"
    assert played[0].reward == 1.0 and played[0].escalations == 0
    assert module.summarise(list(played.values()), 0.1)["episodes"] == 4


def test_a_file_written_under_other_settings_is_not_resumed_onto(tmp_path):
    """One --output and one changed flag is the operator error this catches. The
    cells are keyed by policy, price and seed, so a changed *split* leaves the
    filenames identical and the episodes inside answering a different question --
    and a 48-hour run under a launcher loop is exactly where that happens."""
    from slime_htm.resume import append_row, checked_rows, read_rows

    path = tmp_path / "eval-never.jsonl"
    rows = checked_rows(path, {"group": "click", "seed": 0})
    assert rows == [], "a new file starts empty and keeps its signature"
    append_row(path, {"index": 0, "terminal_reward": 1.0, "episode_steps": []})
    assert len(checked_rows(path, {"group": "click", "seed": 0})) == 1, "and resumes"

    rows = checked_rows(path, {"group": "click", "seed": 1})
    assert rows == [], "a different seed is a different question"
    assert [p.name for p in tmp_path.glob("*.stale-*")], "the old rows are kept aside"
    assert len(read_rows(path)) == 1, "and the new file holds only its signature"

    # A file from before this check exists: what it was written under is unknown,
    # and unknown is the case this is for.
    plain = tmp_path / "eval-always.jsonl"
    append_row(plain, {"index": 0, "terminal_reward": 1.0, "episode_steps": []})
    assert checked_rows(plain, {"group": "click"}) == []


def test_the_run_signs_what_it_collected_and_what_it_evaluated():
    """Including a digest of the corpus the heads were fitted to: two invocations
    whose collections differ must not fill one column each."""
    from conftest import script_source

    source = script_source("browser_pmpo")

    assert "collection_signature = {" in source
    assert "signature=collection_signature" in source, "the collection is checked"
    assert '"corpus": digest(' in source or "corpus=digest(" in source, \
        "and the evaluation carries the corpus it was played against"
    assert "signature=evaluation_signature" in source


def test_the_signatures_carry_the_commit_and_the_fast_modes_settings():
    """code-reviewer's general guard, after exps found the specific gap.

    `collection_signature` carried the venue, the seed and the splits but not the
    fast mode, so `c0783e7` changed the policy -- scroll became a default -- and
    left the dict byte-identical. A resume would have read 28845's 58 no-scroll
    episodes into a corpus the fixed fast mode was extending, and printed
    nothing, because nothing differed.

    Enumerating the settings that matter is how it was missed once: the next
    change would be a field nobody thought to add. The **commit** changes
    whenever the code does, which is why it is in there as well as the fields."""
    from conftest import script_source

    source = script_source("browser_pmpo")

    assert '"commit": commit_now()' in source, "the general guard"
    assert "from litjev.run.provenance import commit_now" in source
    for field in ('"offered_actions": environment(', '"state_chars": STATE_CHARS',
                  '"step_timeout": args.step_timeout'):
        assert field in source, f"the signature is blind to {field}"
    # The evaluation signature is built from the collection one, so it inherits
    # all of this rather than needing its own copy.
    assert "evaluation_signature = dict(\n        collection_signature" in source
    # And the state budget is named once, so the signature cannot claim a number
    # the environment was not built with.
    assert "state_chars=6000" not in source, "the budget is STATE_CHARS, not a literal"


def test_two_fast_modes_differing_only_in_scroll_sign_differently(monkeypatch):
    """The test code-reviewer asked for: configs differing only in `content_free`
    must not share a signature, or a resume mixes the two policies silently."""
    import sys
    import types

    from slime_htm.envs.browser_environment import BrowserEnvironment

    gym = types.ModuleType("gymnasium")
    gym.registry = {"t": object()}
    monkeypatch.setitem(sys.modules, "gymnasium", gym)

    scrolling = BrowserEnvironment(["t"]).offered_actions
    sealed = BrowserEnvironment(["t"], content_free=False).offered_actions
    assert scrolling != sealed

    base = {"venue": "miniwob", "group": "click", "seed": 0}
    assert dict(base, offered_actions=scrolling) != dict(base, offered_actions=sealed)


def test_both_paths_say_why_an_episode_stopped_in_the_same_word():
    """exps audited the two episode loops and found this: the same event -- the
    slow mode writing something unexecutable -- was `ended = "unexecutable_action"`
    on the gate's path and `metadata["unexecutable_action"]` on this one.
    Behaviour identical, field different, so a single count of bad actions across
    a gate run and a CDPO run would have missed this path's silently. It was the
    one thing in its table it had to remember, so the field is shared."""
    import numpy as np

    from slime_htm.environment import EpisodeResult, StepRecord
    from slime_htm.rollout import rollout_episodes

    class Writes:
        """One step, and the slow mode writes something the page cannot take."""

        def __init__(self):
            self.slow_instructions = lambda: "write an action"

        def reset(self, episode):
            return "page", ["click('a')"]

        def act_freely(self, text):
            return None, "not an action this page takes"

        def step(self, action):
            raise AssertionError("not reached: the episode ends on the bad action")

    class Scorer:
        def fast(self, context, actions):
            return np.array([1.0]), np.zeros(4, dtype=np.float32)

        def solve(self, context, instructions, budget):
            return "nonsense", 7, None

    assert EpisodeResult(episode=0, reward=0.0).ended is None, "unset until it stops"

    results = rollout_episodes(Writes(), Scorer(), head=None, lambda_=0.0, episodes=1,
                               budget=4, escalate_prob=1.0)
    assert results[0].ended == "unexecutable_action", "the gate's word, not a new one"
    # The detail stays where it was, so nothing that read it breaks.
    assert results[0].metadata["unexecutable_action"]

    # And the other endings carry the gate's vocabulary too.
    assert StepRecord is not None
    done = EpisodeResult(episode=0, reward=1.0)
    done.ended = "done"
    assert done.ended == "done"


def test_the_summary_counts_why_episodes_stopped():
    """So the count is in `policies.json` rather than in whoever read the log."""
    from conftest import load_script

    summarise = load_script("browser_pmpo").summarise
    rows = [FakeResult(1.0, 1, 3), FakeResult(0.0, 0, 2, ended="max_steps"),
            FakeResult(0.0, 1, 1, ended="unexecutable_action")]
    counted = summarise(rows, 0.1)["ended"]
    assert counted == {"done": 1, "max_steps": 1, "unexecutable_action": 1}


def test_a_killed_browser_is_not_labelled_done(monkeypatch, tmp_path):
    """code-reviewer: after a step timeout `BrowserEnvironment.step` returns
    `done=True` with no actions -- that is how it ends the episode -- so a rollout
    reading `done` alone labels a killed browser `done`. No stored row was wrong,
    because both callers check `environment.complete` and drop it first, but it
    was right only by compensation. The environment's own word wins now."""
    import time

    from slime_htm.envs.browser_environment import BrowserEnvironment
    from slime_htm.rollout import rollout_episodes

    class Hangs:
        def reset(self, seed=None):
            return {"goal": "g",
                    "extra_element_properties": {"a": {"clickable": True}}}, {}

        def step(self, action):
            time.sleep(3)
            raise AssertionError("not reached")

        def close(self):
            pass

    import sys
    import types

    gym = types.ModuleType("gymnasium")
    gym.make = lambda task, **kwargs: Hangs()
    gym.registry = {"t": object()}
    monkeypatch.setitem(sys.modules, "gymnasium", gym)
    from slime_htm.envs import browser_environment as module

    monkeypatch.setattr(module, "candidates",
                        lambda env, observation, limit, extra=(): (
                            [("click('a')", "click('a')")]
                            + [(a, a) for a in extra], 0))
    monkeypatch.setattr(module, "state_text",
                        lambda observation, goal, trail, history=6, chars=4000: "s")

    env = BrowserEnvironment(["t"], step_timeout=0.4,
                             stack_file=str(tmp_path / "stacks.log"))

    class Scorer:
        def fast(self, context, actions):
            return np.array([1.0] * len(actions)), np.zeros(4, dtype=np.float32)

        def slow(self, context, actions, budget):
            return np.array([1.0] * len(actions)), budget, "reasoning"

    results = rollout_episodes(env, Scorer(), head=None, lambda_=0.0, episodes=1,
                               budget=4, escalate_prob=0.0)

    assert env.ended == "step_timeout"
    assert results[0].ended == "step_timeout", \
        "not 'done', which is what the protocol alone would have said"
    assert env.complete is False, "and the collector still drops it"


def test_an_environment_that_offers_nothing_aborts_instead_of_draining(tmp_path):
    """The gate's `blind_run` guard, on this path. The gate *scores* a
    `no_candidates` episode and aborts when the first few are all that -- a run
    that saw no actions has measured nothing. This path *drops* such an episode,
    which is right for training (no decisions, no gradient, and not the policy's
    failure) but made the same breakage silent: 28845 lost 13.8% of its episodes
    to a missing scroll action and it took exps reading the log to notice.

    Aborting is not a verdict and cannot bias one."""
    from slime_htm.loop import (
        BLIND_EPISODES,
        ENVIRONMENT_BROKEN,
        LoopConfig,
        run,
    )

    assert BLIND_EPISODES == 3, "the gate's number"

    config = LoopConfig(rounds=1, episodes_per_round=12, explore_prob=1.0,
                        head=HeadRLConfig(epochs=1), output=str(tmp_path / "out"))
    seen = []

    # **Exit, not an exception.** A launcher loops this script to survive the
    # hang guard's process exit, so a RuntimeError here is another crash to
    # relaunch: three more dropped, abort again, and every index's attempt cap
    # spent -- a guard that exists to stop becoming a slow drain.
    # code-reviewer found that.
    with pytest.raises(SystemExit) as stopped:
        run(metadata(), lambda i, e: Toy(incomplete=set(range(12)), seen=seen),
            ToyScorer(), config, seed=0)

    assert stopped.value.code == ENVIRONMENT_BROKEN == 3, "a code a launcher can read"
    assert len(seen) == BLIND_EPISODES, \
        "it stops at the guard rather than draining all 12"


def test_the_round_row_reports_what_it_dropped_and_why(tmp_path):
    """So the next 13.8% is visible in the artefact instead of in whoever was
    watching the log."""
    import json

    from slime_htm.loop import LoopConfig, run

    config = LoopConfig(rounds=1, episodes_per_round=6, explore_prob=1.0,
                        head=HeadRLConfig(epochs=1), max_attempts=1,
                        output=str(tmp_path / "out"))
    run(metadata(), lambda i, e: Toy(incomplete={3}), ToyScorer(), config, seed=0)

    rounds = json.loads((tmp_path / "out" / "rounds.json").read_text())["rounds"]
    assert rounds[0]["dropped"] == 1
    assert rounds[0]["dropped_reasons"] == {"step_timeout": 1}
    assert rounds[0]["episodes"] == 5 and rounds[0]["given_up"] == 1



def test_the_ess_bands_match_the_bound_rather_than_intuition():
    """My first three bands were backwards, in the direction that flatters the
    result, and code-reviewer derived why before the run landed.

    Round 0's behaviour policy is a 0.5 coin, so `w = pi(taken) / 0.5` is in
    [0, 2], `w^2 <= 2w`, and `fraction = m^2/s >= m/2`. A near-deterministic head
    therefore lands near **0.5 by construction** -- and a fraction near **1** is
    the head that never moved off the coin, which is the *bad* case I had written
    as the good one. Derived here as well as stated in the README, so the table
    cannot drift from the arithmetic it rests on."""
    from slime_htm.head_trainer import effective_sample_size

    rng = np.random.default_rng(0)
    n = 500

    def read(pi_taken):
        w = np.asarray(pi_taken) / 0.5
        return effective_sample_size(w) / n, w.mean()

    unmoved, unmoved_mean = read(np.full(n, 0.5))
    assert unmoved == pytest.approx(1.0), "a head that learned nothing reads 1.0"
    assert unmoved_mean == pytest.approx(1.0)

    deterministic, _ = read((rng.random(n) < 0.5).astype(float))
    assert 0.40 < deterministic < 0.60, "a deterministic head reads about 0.5"
    # **And this draw reads 0.446 -- on the far side of the 0.5 my first table
    # used as a boundary.** So the same head could have landed in either of my
    # two top bands on sampling noise alone, which is code-reviewer's point about
    # the calibration and not a detail of this seed.
    assert deterministic < 0.5, "noise alone decided which band my old table gave"

    # The bound itself, over a spread of heads: the fraction never falls far
    # below half the ratio mean, which is why "below 0.2" needs a head that
    # systematically prefers the action that was not logged.
    for share in (0.15, 0.5, 0.85):
        agree = rng.random(n) < share
        fraction, mean = read(np.where(agree, 0.95, 0.05))
        assert fraction >= mean / 2 - 1e-9, "fraction >= ratio_mean / 2"
    low, low_mean = read(np.where(rng.random(n) < 0.15, 0.95, 0.05))
    assert low < 0.35 and low_mean < 0.5, "and that is the only route below 0.2"


def test_what_the_run_cannot_claim_is_recorded_before_its_table_exists():
    """28853 seed 0 clips 95.6% of its decisions with `ratio_max` at the
    boundary, so the clip constant weights those updates and not the importance
    ratio. The licensed sentence narrows, and it narrows before the table arrives
    rather than after somebody reads it."""
    from conftest import repo_path

    plan = repo_path("experiments/browser-gate/README.md").read_text()

    assert "How the ESS is read, and why my first three bands were backwards" in plan
    assert "a high fraction is not good news" in plan
    assert "was wrong" in plan, "the sentence it replaces is named as wrong"
    assert "what this run can still show" in plan
    assert "what it cannot show" in plan
    assert "is not a licensed sentence from this run" in plan
    # The design tension is named rather than presented as an oversight.
    assert "a genuine tension in the design rather than an oversight" in plan
    assert "pays with a saturated trust region" in plan
    # And the mechanism sentence is the corrected one: the clip constant weights
    # nothing, the narrowed claim rests on there being no ablation.
    assert "my first version of this sentence" in plan
    assert "contribute **zero gradient**" in plan
    assert "there is no ablation" in plan
    assert "28853 does not carry them" in plan
    # And the estimate I got wrong is recorded as mine.
    assert "where I estimated 400-650" in plan
    # The governing rule and the remedy survive the rewrite.
    assert "minimum over the three PMPO seeds still governs" in plan
    assert "no ESS at all" in plan
    assert "more seeds, not more episodes" in plan.lower(), \
        "a thin fraction from a bad fit is answered with seeds, not episodes"
    assert "**One collection, one table.**" in plan


def test_the_surrogate_reports_where_its_gradient_vanishes():
    """code-reviewer: `clipped_fraction` counts ratios outside the band, which is
    **not** the clip binding. With the loss `-min(rA, clip(r)A)`:

      - favourable side (r high with A > 0, r low with A < 0): `min` takes the
        bounded branch, which is constant in the parameters, so the decision
        contributes **zero** gradient -- the clip constant weights nothing
      - unfavourable side: `min` takes `rA` and the ratio **does** weight it

    I had written that a high `clipped_fraction` meant "the clip constant is
    weighting those updates". It weights nothing, and half the out-of-band
    decisions are still ratio-weighted. So the honest diagnostic is how much of
    the batch the surrogate silently dropped, and it is now reported."""
    import torch

    clip = 0.2

    def branch(ratio_value, advantage):
        r = torch.tensor([ratio_value], requires_grad=True)
        a = torch.tensor([advantage])
        loss = -torch.min(r * a, r.clamp(1 - clip, 1 + clip) * a).mean()
        loss.backward()
        return float(r.grad.item())

    # Outside the band and favourable: no gradient at all.
    assert branch(1.9, +1.0) == 0.0
    assert branch(0.1, -1.0) == 0.0
    # Outside the band and unfavourable: the ratio still weights the update.
    assert branch(1.9, -1.0) != 0.0
    assert branch(0.1, +1.0) != 0.0

    # **Pinned to autograd, not to a range.** My first test for this asserted
    # only `0.0 <= value <= 1.0`, which no proportion can fail, and the field was
    # wrong by 2.5x underneath it: `bounded <= unclipped` is also true *inside*
    # the band, where `bounded == unclipped` and the gradient is the full `rA`.
    # At epoch 0 with every ratio near 1 it read about 1.0 -- the whole batch
    # dropped -- exactly when nothing was. A test that cannot fail is not a test.
    ratio = torch.tensor([1.0, 1.1, 0.9, 1.9, 0.1, 1.9, 0.1], requires_grad=True)
    a = torch.tensor([1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0])
    unclipped, bounded = ratio * a, ratio.clamp(1 - clip, 1 + clip) * a
    (-torch.min(unclipped, bounded).mean()).backward()

    # **Against the trainer's own function, not a restatement of it.**
    # code-reviewer's second catch: my first version recomputed
    # `outside & (bounded <= unclipped)` inline, so it verified the formula and
    # left the trainer guarded only by the invariant below. A test of a copy
    # guards the copy.
    from slime_htm.head_trainer import dropped_decisions

    truth = float((ratio.grad == 0).float().mean())
    counted = float(dropped_decisions(ratio, a, clip).float().mean())
    naive = float((bounded <= unclipped).float().mean())
    assert counted == pytest.approx(truth), "the trainer's rule matches autograd"
    assert naive == pytest.approx(0.714, abs=0.01), "and the version without "
    assert truth == pytest.approx(0.286, abs=0.01), "`outside &` was 2.5x high"

    episodes = corpus(count=12, steps=4)
    _, report = train_head_on_episodes(metadata(), episodes, HeadRLConfig(epochs=3),
                                       seed=0)
    for field in ("zero_gradient_fraction", "zero_gradient_fraction_mean",
                  "clipped_fraction", "clipped_fraction_mean"):
        assert field in report, field
        assert 0.0 <= report[field] <= 1.0
    # A dropped decision is necessarily one whose ratio left the band, so the
    # dropped share can never exceed the clipped share. That ordering is the
    # invariant the bug violated.
    assert report["zero_gradient_fraction"] <= report["clipped_fraction"] + 1e-9
    assert (report["zero_gradient_fraction_mean"]
            <= report["clipped_fraction_mean"] + 1e-9)


def test_a_dead_environment_aborts_the_evaluation_too(tmp_path):
    """code-reviewer verified the collection aborts at three dropped episodes;
    the evaluation had only the per-index attempt cap. A browser that never came
    back would have burned 27 tasks x --max-attempts in **every** cell -- about
    1200 launches across fifteen -- before the empty intersection stopped it.
    Terminating, but hours of waste, and the asymmetry was mine: I told exps
    where the bound was on each path and only then noticed one was far away."""
    from conftest import load_script

    from slime_htm.loop import BLIND_EPISODES, ENVIRONMENT_BROKEN

    module = load_script("browser_pmpo")
    seen = []
    with pytest.raises(SystemExit) as stopped:
        module.play(tmp_path / "eval-never.jsonl",
                    lambda: Toy(incomplete=set(range(27)), seen=seen), ToyScorer(),
                    None, price=0.0, episodes=27, budget=4, max_steps=5, seed=0,
                    max_attempts=3, escalate_prob=0.0)

    assert stopped.value.code == ENVIRONMENT_BROKEN
    assert len(seen) == BLIND_EPISODES, "three, not 27 x 3"


def test_a_cell_that_has_collected_something_keeps_going(tmp_path):
    """The guard is for an environment that answers nothing, not for a hard task.
    One good episode means the browser works, so later drops are retried under
    the attempt cap rather than aborting the run."""
    from conftest import load_script

    module = load_script("browser_pmpo")
    seen = []
    played = module.play(tmp_path / "eval-never.jsonl",
                         lambda: Toy(incomplete={1, 2, 3}, seen=seen), ToyScorer(),
                         None, price=0.0, episodes=5, budget=4, max_steps=5, seed=0,
                         max_attempts=2, escalate_prob=0.0)

    assert sorted(played) == [0, 4], "the two that worked"
    assert seen.count(1) == 2, "and the failures used their attempts, not an abort"


def test_every_outcome_has_its_sentence_written_before_the_table():
    """The version that stood here named only the sentence a **win** would earn,
    which is the day's recurring mistake in its final form: a reading chosen
    after seeing the number. All five outcomes of the pre-registered primary test
    now have their licensed sentence, including the two that are not results."""
    from conftest import repo_path

    plan = repo_path("experiments/browser-gate/README.md").read_text()
    assert "The sentence for each way the table can come out" in plan

    # The outcomes that are not wins, which are the ones a later reading bends.
    assert "**not a result.**" in plan, "direction without significance"
    assert "not \"promising\", not \"a trend\"" in plan
    assert "PMPO did not beat the priced cross-entropy head." in plan
    assert "PMPO lost to cross-entropy on the same priced returns." in plan
    assert "most informative thing this run can produce" in plan, \
        "a negative result is stated as informative, not as a disappointment"
    # Beating the constants is explicitly not the paper's claim.
    assert "is **not** the paper's claim" in plan
    # And the twelve-test trap is closed in advance.
    assert "The secondary tests cannot be promoted." in plan
    # Wrapped in the file, so matched on the clause that carries the rule.
    assert "tie at the pre-registered 0.1 is **a tie**" in plan.replace("\n", " ")
    # The attribution narrowing holds in both directions.
    assert "already narrowed regardless of direction" in plan
    # Which seed the p-value is from, since one p over 27 tasks reads as the run.
    assert "The p-value is seed 0's." in plan
    # And that 28853 cannot supply these keys, with the command that recomputes
    # them -- the "from <sha> on" practice on the reading side.
    assert "28853's own `policies.json` will not have these fields." in plan
    assert "--verdict" in plan and "Not a second script" in plan
    assert "never against a mean across seeds" in plan.replace("\n", " ")
    # A row for the case where the significant seed is the only one agreeing.
    assert "ahead at **1** of 3" in plan
    # And "lost" has a threshold rather than a sign: overstating a negative
    # result is the same error as overstating a positive one.
    assert "A one-pair deficit is not a defeat" in plan
    assert "overstating a negative result is the same error" in plan


def test_matched_seeds_is_computed_right_not_merely_present():
    """code-reviewer flipped `difference` to `theirs - mine` in a clean copy and
    my source-string test still passed 53/53. These two fields decide which
    sentence the paper prints, and the third time in one day that I asserted a
    computation existed rather than that it was right.

    Hand-built episodes whose answer is known: CDPO ahead at two seeds and
    behind at one, with the arithmetic done on paper."""
    from conftest import load_script

    module = load_script("browser_pmpo")
    price = 0.1

    def cell(reward, escalations):
        """One episode, so the cell's mean value is reward - price*escalations."""
        return [module.Played(episode=0, reward=reward, finished=True, steps=[
            {"escalated": True, "generated_tokens": 0}] * escalations
            + [{"escalated": False, "generated_tokens": 0}])]

    # value = reward - 0.1 * escalations
    routed = {
        ("pmpo", price, 0): cell(1.0, 2),       # 1.0 - 0.2 = 0.8
        ("priced_ce", price, 0): cell(1.0, 5),  # 1.0 - 0.5 = 0.5   pmpo ahead +0.3
        ("pmpo", price, 1): cell(0.0, 1),       # 0.0 - 0.1 = -0.1
        ("priced_ce", price, 1): cell(1.0, 1),  # 1.0 - 0.1 =  0.9  pmpo behind -1.0
        ("pmpo", price, 2): cell(1.0, 0),       # 1.0
        ("priced_ce", price, 2): cell(1.0, 3),  # 0.7               pmpo ahead +0.3
    }
    found = module.matched_seeds(routed, [0, 1, 2], price)

    assert [r["difference"] for r in found["rows"]] == pytest.approx([0.3, -1.0, 0.3])
    assert found["pmpo_ahead_at"] == [0, 2]
    assert found["reversed_at"] == [1], "the sign is CDPO minus the baseline"
    assert "never against a mean" in found["definition"]
    # The flip code-reviewer made would invert both lists, so assert the order.
    assert found["rows"][0]["pmpo"] > found["rows"][0]["priced_ce"]


def test_the_reverse_test_is_the_other_direction_not_a_copy():
    """code-reviewer's second mutation: making `reverse` compare priced_ce with
    itself left the suite green. A self-comparison has no discordant pairs, so
    `reverse.sign_test_p` would be 1.0 for every result and "lost" would be
    unreachable -- the row would never fire."""
    from conftest import load_script

    module = load_script("browser_pmpo")
    price = 0.1

    def cell(rewards):
        return [module.Played(episode=i, reward=r, finished=True,
                              steps=[{"escalated": False, "generated_tokens": 0}])
                for i, r in enumerate(rewards)]

    # The baseline wins every one of six tasks, so the forward test must not be
    # significant and the reverse one must be.
    routed = {("pmpo", price): cell([0.0] * 6),
              ("priced_ce", price): cell([1.0] * 6)}
    found = module.primary_verdict(routed, price, seed=0)

    assert found["ahead"] == 0 and found["behind"] == 6
    assert found["sign_test_p"] == pytest.approx(1.0), "CDPO did not win"
    assert found["reverse"]["ahead"] == 6 and found["reverse"]["behind"] == 0
    assert found["reverse"]["sign_test_p"] == pytest.approx(1 / 64), \
        "and the reverse test is what makes 'lost' reachable"
    assert found["seed"] == 0 and "not in this" in found["seed_note"]

    # A self-comparison, which is what the mutation made it: no discordant pairs,
    # so p = 1.0 and the loss row could never fire.
    same = module.paired_test(routed[("priced_ce", price)],
                              routed[("priced_ce", price)], price)
    assert same["sign_test_p"] == 1.0 and same["ahead"] == same["behind"] == 0


def test_a_run_can_be_read_back_from_its_cell_files(tmp_path):
    """code-reviewer's second condition. 28853 runs from `c0783e7`, which computes
    no `primary_test.seed`, no `reverse` and no `matched_seeds` -- so the
    README's outcome table names keys its `policies.json` will not have. That is
    my own "from `<sha>` on" practice applied to the **reading** side, which I
    had written for the producing side only.

    `--verdict` recomputes them from the cells through the same functions a live
    run calls, rather than a second script with a second copy of the
    definitions -- which is how the resume arithmetic was wrong twice."""
    import json

    from conftest import load_script

    from slime_htm.resume import append_row

    module = load_script("browser_pmpo")

    def write(name, rewards, escalations):
        for index, (reward, spent) in enumerate(zip(rewards, escalations,
                                                    strict=True)):
            append_row(tmp_path / f"eval-{name}.jsonl", {
                "index": index, "attempt": 0, "terminal_reward": reward,
                "finished": True,
                "episode_steps": [{"escalated": True, "generated_tokens": 1}] * spent
                                 + [{"escalated": False, "generated_tokens": 0}]})

    # Four tasks. CDPO matches the baseline's reward on every one but escalates
    # less, so at price 0.1 it is ahead on all four by 0.1 -- p = 1/16.
    write("never", [0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0])
    write("always", [1.0, 1.0, 1.0, 1.0], [3, 3, 3, 3])
    for seed in (0, 1):
        write(f"pmpo-price0.1-seed{seed}", [1.0] * 4, [2, 2, 2, 2])
        write(f"priced_ce-price0.1-seed{seed}", [1.0] * 4, [3, 3, 3, 3])

    found = module.verdict(tmp_path)

    assert found["paired_tasks"] == 4
    assert found["lambdas"] == [0.1] and found["training_seeds"] == [0, 1]
    primary = found["primary_test"]
    assert primary["seed"] == 0, "and the file says which seed the p is from"
    assert (primary["ahead"], primary["behind"]) == (4, 0)
    assert primary["sign_test_p"] == pytest.approx(1 / 16)
    assert primary["reverse"]["sign_test_p"] == pytest.approx(1.0)
    assert found["matched_seeds"]["pmpo_ahead_at"] == [0, 1]
    assert found["matched_seeds"]["reversed_at"] == []
    # The constants are re-scored at the price rather than re-run.
    row = found["policies"]["price_0.1"]
    assert row["never"]["value"] == pytest.approx(0.0)
    assert row["always"]["value"] == pytest.approx(1.0 - 0.1 * 3)
    assert row["pmpo"]["value"] == pytest.approx(1.0 - 0.1 * 2)

    # And it writes nothing until asked, so reading a run cannot damage it.
    assert not (tmp_path / "verdict.json").exists()
    json.dumps(found, default=str)


def test_the_read_back_takes_the_middle_price_from_the_runs_own_order(tmp_path):
    """code-reviewer's nit, and it would have moved the primary test to a price
    the run never pre-registered. A live run takes `prices[len//2]` over the
    `--lambdas` order it was given; sorting the cell prices agrees only if that
    order was sorted. 28853's log reads `evaluated at [0.0, 0.1, 0.2]` so the two
    agree there -- the guard is for the next run."""
    import json

    from conftest import load_script

    from slime_htm.resume import append_row

    module = load_script("browser_pmpo")

    def write(name, price_label, reward, spent):
        append_row(tmp_path / f"eval-{name}.jsonl", {
            "index": 0, "attempt": 0, "terminal_reward": reward, "finished": True,
            "episode_steps": [{"escalated": True, "generated_tokens": 1}] * spent})
        return price_label

    for price in ("0.0", "0.1", "0.2"):
        for name in ("pmpo", "priced_ce"):
            write(f"{name}-price{price}-seed0", price, 1.0, 1)
    write("never", "-", 0.0, 0)
    write("always", "-", 1.0, 1)

    # No report yet: the sorted middle, with the limitation announced.
    assert module.recorded_lambdas(tmp_path) is None
    assert module.verdict(tmp_path)["primary_test"]["price"] == 0.1

    # A run launched unsorted pre-registered 0.2, and the read-back must use it.
    (tmp_path / "policies.json").write_text(json.dumps({"lambdas": [0.1, 0.2, 0.0]}))
    assert module.recorded_lambdas(tmp_path) == [0.1, 0.2, 0.0]
    assert module.verdict(tmp_path)["primary_test"]["price"] == 0.2, \
        "the middle of the order as given, not of the sorted list"

    # Cells and report from different runs: refused rather than guessed.
    (tmp_path / "policies.json").write_text(json.dumps({"lambdas": [0.3, 0.4]}))
    with pytest.raises(SystemExit, match="different runs") as refused:
        module.verdict(tmp_path)
    # **The refusal names both causes**, because the other one looks identical
    # from inside: cell names format prices with `:g`, so 1/3 is written
    # `price0.333333` and parses back as a different float -- measured, not
    # assumed: float("0.333333") != 1/3. code-reviewer found that this edge case
    # fails safe, and a refusal pointing at only the provenance cause would send
    # whoever hits the formatting one looking for a mismatched run.
    assert f"{1 / 3:g}" == "0.333333" and float(f"{1 / 3:g}") != 1 / 3
    assert "formatting" in str(refused.value)
    assert "naming limitation rather than a mismatched run" in str(refused.value)


def test_read_cells_parses_the_names_the_run_actually_writes(tmp_path):
    """**Observed names, not names I invented.** My other read-back tests build
    `price0.0`; 28853 writes `price0`, because the cell name formats the price
    with `:g` and `f"{0.0:g}"` is `"0"`. So those tests exercised a format
    production does not emit -- the same shape as the `candidates` stub that did
    not honour the real budget rule, and the playwright double that needed a real
    module name.

    These six are copied from `ls` on 28853's output directory."""
    from conftest import load_script

    from slime_htm.resume import append_row

    module = load_script("browser_pmpo")

    observed = ["never", "always", "smeared_ce-price0-seed0",
                "smeared_ce-price0.1-seed0", "priced_ce-price0.2-seed1",
                "pmpo-price0-seed2"]
    for name in observed:
        append_row(tmp_path / f"eval-{name}.jsonl", {
            "index": 0, "attempt": 0, "ended": "started"})
        append_row(tmp_path / f"eval-{name}.jsonl", {
            "index": 0, "attempt": 0, "terminal_reward": 1.0, "finished": True,
            "episode_steps": [{"escalated": False, "generated_tokens": 0}]})

    cells = module.read_cells(tmp_path)

    assert ("never",) in cells and ("always",) in cells, "constants keyed by name"
    assert ("smeared_ce", 0.0, 0) in cells, "`price0` is the price 0.0"
    assert ("smeared_ce", 0.1, 0) in cells
    assert ("priced_ce", 0.2, 1) in cells
    assert ("pmpo", 0.0, 2) in cells
    assert len(cells) == len(observed), "every observed name parsed to one cell"
    # The marker rows are not episodes, so each cell holds exactly one.
    assert all(len(played) == 1 for played in cells.values())
    # And the `:g` round-trip that makes `price0` safe, measured.
    assert f"{0.0:g}" == "0" and float("0") == 0.0


def test_the_result_is_recorded_with_the_pre_registered_sentence():
    """28853 landed negative, and the table written before it decides which
    sentence applies: the loss row keys on **reverse** p <= 0.05 and the reverse
    p is 0.133, so the licensed sentence is the tie row -- CDPO did not beat the
    priced cross-entropy head. Not "CDPO lost".

    Promoting it on the direction evidence afterwards would be the same act as
    softening it would be, and the direction evidence is reported in full
    instead."""
    from conftest import repo_path

    plan = repo_path("experiments/browser-gate/README.md").read_text()
    section = plan[plan.index("## 28853 的结果"):]

    assert "reverse p" in section or "reverse (priced_ce?)" in section
    assert "0.1334" in section, "the quantity the rule keys on"
    assert "没有超过带价交叉熵头" in section, "the pre-registered tie sentence"
    assert "我不把它升级成" in section, "and the refusal to promote it"
    # The direction evidence is present and labelled as not pre-registered.
    assert "三个种子全部反向" in section
    assert "都不是预注册的检验" in section
    # The asymmetry in my own table, recorded rather than filled after the fact.
    assert "有利于 PMPO" in section, "the gap favours the method, and is named"
    assert "事后改规则" in section


def test_the_mechanism_is_computed_from_the_table_not_asserted():
    """The informative part is not the verdict: CDPO earned the highest reward of
    any routing policy and paid 8.6x the escalations for it. That is a different
    failure from learning nothing, and the numbers are in the file."""
    import json

    from conftest import repo_path

    found = json.loads(repo_path(
        "experiments/browser-gate/results/browser-pmpo-click-c0783e7/"
        "verdict.json").read_text())
    row = found["policies"]["price_0.1"]

    # Highest reward among the policies that actually route.
    routing = {name: row[name]["reward"]
               for name in ("smeared_ce", "priced_ce", "pmpo")}
    assert max(routing, key=routing.get) == "pmpo", routing
    # And the escalation count that makes it lose anyway.
    assert row["pmpo"]["escalations"] / row["priced_ce"]["escalations"] > 8
    assert row["pmpo"]["value"] < row["priced_ce"]["value"]
    # The marginal trade the README quotes: more escalations, barely more reward.
    extra_cost = 0.1 * (row["pmpo"]["escalations"] - row["priced_ce"]["escalations"])
    extra_reward = row["pmpo"]["reward"] - row["priced_ce"]["reward"]
    assert extra_cost > 8 * extra_reward, "about eight times the cost of the gain"

    plan = repo_path("experiments/browser-gate/README.md").read_text()
    assert "学到了思考有用,没学到价格" in plan


def test_the_slow_only_comparison_is_recorded_as_a_second_question():
    """exps's user asked for the baseline to be the same model's slow-only. It
    selects a different winner than the pre-registered value metric, and the
    disagreement is the result -- so both are reported and neither replaces the
    other. Swapping the primary test to the framing that flatters the run, after
    seeing the numbers, is what pre-registration exists to stop."""
    from conftest import repo_path

    plan = repo_path("experiments/browser-gate/README.md").read_text()
    section = plan[plan.index("### 换一个基线"):]

    assert "不替换预注册的 primary test" in section
    assert "不换 primary test" in section
    # **Uncorrected and exploratory**, labelled as such: nine one-sided tests
    # asked after seeing the numbers, and Holm at 0.05 leaves only the two at
    # 0.0039 standing. So priced_ce's 0.0312 is not significant, and the word is
    # gone from that point. code-reviewer asked for it, after I pre-registered
    # "secondaries cannot be promoted" and then wrote "significant" in a post-hoc
    # framing with no correction.
    assert "都是未校正的" in section
    assert "Holm" in section and "0.00556" in section
    assert "只有两个结论能称为显著" in section
    assert "0.0312" in section
    assert "27 个任务上逐任务完全相同" in section
    assert "任何 λ 下都不显著低于" in section
    # And the two winners, named as two answers to two questions.
    assert "两个 metric 选出不同赢家" in section


def test_the_slow_only_findings_carry_their_three_limits():
    """Each limit tightens the reading rather than loosening it, and the first is
    the lesson from earlier today: a single-seed number is a measurement, not a
    property. Two have already been overturned by a second seed."""
    from conftest import repo_path

    plan = repo_path("experiments/browser-gate/README.md").read_text()
    section = plan[plan.index("### 换一个基线"):]

    # Single seed, with the cost of confirming it.
    assert "单一种子" in section
    assert "约一小时" in section, "and what confirming it would cost"
    # Not an equivalence test.
    assert "不是等价性检验" in section
    # Derived seconds, not measured.
    assert "推导的,不是测量的" in section
    assert "cell schema" in section


def test_only_the_middle_price_has_three_seeds():
    """The limit above is a fact about the run's own design -- the full curve is
    read at the first seed only -- so it is checkable rather than remembered."""
    from conftest import repo_path

    cells = sorted(p.stem[len("eval-"):] for p in repo_path(
        "experiments/browser-gate/results/browser-pmpo-click-c0783e7").glob(
            "eval-*.jsonl"))
    seeds_at = {}
    for name in cells:
        if "-price" not in name:
            continue
        objective, rest = name.split("-price", 1)
        price, seed = rest.split("-seed")
        seeds_at.setdefault((objective, price), []).append(int(seed))

    for (objective, price), seeds in seeds_at.items():
        expected = 3 if price == "0.1" else 1
        assert len(seeds) == expected, f"{objective} at {price}: {seeds}"
    assert ("smeared_ce", "0") in seeds_at, "the cell the finding rests on"
    assert seeds_at[("smeared_ce", "0")] == [0], "one seed, which is the limit"


def test_the_sign_tests_resolution_floor_is_stated():
    """The user asked how to read "CDPO is not significantly below slow-only".
    The answer is that it cannot be read as CDPO holding its score: a one-sided
    sign test with 4 discordant pairs has a minimum possible p of 1/2^4 = 0.0625,
    so on 27 tasks any effect touching 4 tasks or fewer is unreportable however
    large it is -- and CDPO's loss is 0.148 x 27 = exactly 4 tasks.

    Derived here rather than quoted, so the README cannot drift from it."""
    from math import comb

    from conftest import repo_path

    def one_sided(ahead, behind):
        d = ahead + behind
        return (sum(comb(d, k) for k in range(ahead, d + 1)) / 2 ** d
                if d else 1.0)

    # The floor: five discordant pairs is the minimum that can reach 0.05.
    assert one_sided(4, 0) == pytest.approx(0.0625)
    assert one_sided(5, 0) == pytest.approx(0.03125)
    assert one_sided(4, 0) > 0.05 >= one_sided(5, 0)

    # The two cells differ by one task and sit on opposite sides of the line.
    assert one_sided(5, 0) <= 0.05, "priced_ce, 5 below, significant"
    assert one_sided(4, 0) > 0.05, "pmpo, 4 below, not -- same direction"

    # And CDPO's score loss is exactly at the floor.
    assert round((0.444 - 0.296) * 27) == 4

    plan = repo_path("experiments/browser-gate/README.md").read_text()
    assert "是 n 不够,不是损失小" in plan
    assert "分辨率下限" in plan
    # The claim spans a line break AND markdown emphasis, so the half that sits
    # on one line is what is matched. Flattening newlines alone was not enough,
    # which the failing assertion is how I found out.
    assert "没有被证明低于" in plan, "the distinction the sentence turns on"
    assert "而这两件事不一样" in plan
    # A zero effect and an undetectable one are not the same, and were listed
    # side by side before.
    assert "零效应" in plan and "有效应,检不出" in plan


def test_the_primary_test_is_the_one_with_power():
    """13 discordant pairs against the slow-only comparisons' 3 to 6, so the
    pre-registered primary is the only one of these tests that is not pressed
    against the floor. Worth stating because the reason it was chosen was
    isolating the off-policy treatment, not power -- the power is luck."""
    import json

    from conftest import repo_path

    found = json.loads(repo_path(
        "experiments/browser-gate/results/browser-pmpo-click-c0783e7/"
        "verdict.json").read_text())
    primary = found["primary_test"]

    discordant = primary["ahead"] + primary["behind"]
    assert discordant == 13, "well above the five-pair floor"
    plan = repo_path("experiments/browser-gate/README.md").read_text()
    assert "13 个 discordant" in plan
    assert "而不是功效" in plan, "and that the power was not the reason"


def test_the_evaluations_step_cap_is_separate_from_the_collections():
    """28853 left 8 of 27 tasks unfinished under `always`, every one against the
    25-step cap. Asking whether more steps finishes them must not also discard
    the 81 collected episodes -- and it would, because `max_steps` is in the
    collection signature, so raising it sets the corpus aside and recollects.

    Two caps, so the head is the same head read at a longer horizon rather than
    a different head on a different corpus."""
    from conftest import script_source

    source = script_source("browser_pmpo")

    assert '"--eval-max-steps"' in source
    assert "eval_max_steps = args.eval_max_steps or args.max_steps" in source
    # The collection signature keeps the collection's cap...
    collection = source[source.index("collection_signature = {"):
                        source.index("# The rows the CDPO trainer sees")]
    assert '"max_steps": args.max_steps' in collection
    assert "eval_max_steps" not in collection, \
        "the evaluation's cap must not invalidate the corpus"
    # ...and the evaluation signature carries its own, so a longer read sets the
    # eval cells aside without touching the episodes.
    evaluation = source[source.index("evaluation_signature = dict("):
                        source.index('played = dict(')]
    assert "eval_max_steps=eval_max_steps" in evaluation


def test_the_degenerate_cells_are_recorded_as_never():
    """Four cells escalate on 0% of steps and are `never` position for position --
    `smeared_ce` and `priced_ce` collapse to not-thinking at two of three seeds
    on the primary price. So "both baselines positive at all three seeds" is, in
    two of them, "the baseline is the constant policy", and that changes what
    beating them means."""
    import json

    from conftest import repo_path

    d = repo_path("experiments/browser-gate/results/browser-pmpo-click-c0783e7")

    def stats(name):
        rows = [json.loads(line) for line in
                (d / f"eval-{name}.jsonl").read_text().splitlines() if line.strip()]
        eps = [r for r in rows if "episode_steps" in r]
        steps = sum(len(r["episode_steps"]) for r in eps)
        esc = sum(sum(bool(s["escalated"]) for s in r["episode_steps"]) for r in eps)
        return esc / steps, sum(r["terminal_reward"] for r in eps) / len(eps), steps

    never = stats("never")
    for name in ("smeared_ce-price0.1-seed1", "smeared_ce-price0.1-seed2",
                 "priced_ce-price0.1-seed1", "priced_ce-price0.1-seed2"):
        assert stats(name) == never, f"{name} is `never` exactly"

    # CDPO does not collapse at any seed on the primary price.
    for seed in (0, 1, 2):
        rate, _, _ = stats(f"pmpo-price0.1-seed{seed}")
        assert 0.25 < rate < 0.45, f"pmpo seed {seed} still routes: {rate:.2f}"

    plan = repo_path("experiments/browser-gate/README.md").read_text()
    assert "完全退化成了\n不思考" in plan or "完全退化成了" in plan
    assert "输给了不思考" in plan
    # And the design problem it exposes, owned rather than presented as a result.
    assert "那是我的设计问题而不是结果" in plan
    assert "现在不能改它" in plan


def test_the_carry_mechanism_is_gone_and_why_is_recorded():
    """code-reviewer's RED on c0f8a9f: the corpus was set aside anyway (81
    episodes read as 0 under the current signature), and `carry_finished` ran
    after `run`, stamping the new evaluation signature onto rows played by the
    old heads -- satisfying by construction the guard I had added that morning to
    keep two heads out of one cell.

    Fixing it needed three layers. Not fixing it costs 0.5h, because the finished
    episodes average 2.3 steps and rerunning them is nearly free. So it is
    removed rather than repaired."""
    from conftest import repo_path, script_source

    source = script_source("browser_pmpo")
    assert "carry_finished" not in source
    assert "--carry-finished-from" not in source

    plan = repo_path("experiments/browser-gate/README.md").read_text()
    assert "删掉了 `--carry-finished-from`" in plan
    assert "被构造性地满足了" in plan, "the guard's failure mode, named"
    assert "81 个 episode 读到 0 个" in plan, "and the measurement behind it"
    # The decision rule, which is the transferable part.
    assert "一个机制的正确性成本高于它节省的时间时,删掉它" in plan
    # And the separate eval cap survives, since that part was sound.
    assert "--eval-max-steps" in source


def _cell_rows(path, rewards, escalations):
    from slime_htm.resume import append_row

    for index, (reward, spent) in enumerate(zip(rewards, escalations, strict=True)):
        append_row(path, {
            "index": index, "attempt": 0, "terminal_reward": reward,
            "finished": True, "ended": "done",
            "episode_steps": [{"escalated": True, "generated_tokens": 7}] * spent
                             + [{"escalated": False, "generated_tokens": 0}]})


def test_eval_only_reports_per_cell_instead_of_raising_after_five_hours(tmp_path):
    """code-reviewer's RED on b759d68. Everything after the evaluation pairs
    policies -- the five-way table, the seed spread, `primary_verdict` against
    `priced_ce`, the paired tests -- so with a policy skipped they index a cell
    that was never played. The failure lands **after every episode has run**: exit
    1, no stacks.log growth, no report, the whole evaluation lost.

    My own `--eval-only` help text said the table "is a diagnostic, not a
    comparison". This is that sentence implemented."""
    from conftest import load_script

    module = load_script("browser_pmpo")

    _cell_rows(tmp_path / "eval-always.jsonl", [1.0, 1.0, 0.0], [3, 3, 3])
    _cell_rows(tmp_path / "eval-pmpo-price0.1-seed0.jsonl", [1.0, 0.0, 0.0], [1, 1, 1])

    found = module.verdict(tmp_path)

    assert found["kind"] == "diagnostic" and found["incomplete"] is True
    assert "priced_ce" in found["why"], "it names what is missing"
    assert set(found["cells"]) == {"always", "pmpo-price0.1-seed0"}
    # Per-cell numbers survive; the pairing does not exist.
    assert found["cells"]["always"]["reward"] == pytest.approx(2 / 3)
    assert found["cells"]["pmpo-price0.1-seed0"]["reward"] == pytest.approx(1 / 3)
    assert found["cells"]["pmpo-price0.1-seed0"]["escalations"] == pytest.approx(1.0)
    for absent in ("primary_test", "matched_seeds", "seed_spread", "paired"):
        assert absent not in found, f"{absent} cannot be computed and is absent"
    assert "任何对比都" in found["note"]


def test_the_live_path_stops_before_the_pairing_too():
    """Both paths, as asked. The live one cannot be driven without a model, so
    what is checked is that the early return exists above the pairing and uses
    the same function -- a second implementation of the diagnostic is how the
    resume arithmetic came to be wrong twice."""
    from conftest import script_source

    source = script_source("browser_pmpo")
    branch = source.index("if wanted is not None:\n        found = diagnostic_report(")
    pairing = source.index('report["primary_test"] = primary_verdict(')
    assert branch < pairing, "the early return is above everything that pairs"
    assert source.index("        return", branch) < pairing, "and it returns"
    # One definition, called by both paths.
    assert source.count("def diagnostic_report(") == 1
    assert source.count("diagnostic_report(") == 3, "defined once, called twice"


def test_a_full_run_still_pairs(tmp_path):
    """The guard must not swallow the complete case: with all five policies the
    read-back computes the primary test as before."""
    from conftest import load_script

    module = load_script("browser_pmpo")
    for name in ("never", "always"):
        _cell_rows(tmp_path / f"eval-{name}.jsonl", [1.0, 0.0, 0.0], [0, 0, 0])
    for name in ("smeared_ce", "priced_ce", "pmpo"):
        _cell_rows(tmp_path / f"eval-{name}-price0.1-seed0.jsonl",
                   [1.0, 1.0, 0.0], [1, 1, 1])

    found = module.verdict(tmp_path)
    assert found.get("kind") != "diagnostic"
    assert found["primary_test"]["against"] == "priced_ce"
    assert "matched_seeds" in found


def test_a_policy_block_says_which_seed_it_is_and_the_pooled_one_sits_beside_it(tmp_path):
    """29773's `policies` block was seed 0 and said nothing about it.

    `routed[(name, price)]` is aliased to `seeds[0]`, so every routed entry in
    `policies` holds the first seed's episodes. Its reward read as the run's --
    0.4444, where the three seeds pool to 0.4691 -- and its escalation rate and
    tokens went into a draft table whose caption promised a mean over seeds. The
    same file's `ceiling.seed_spread_warning` already required any single-seed
    figure from this venue to be labelled or not quoted, and this block was the
    one place that rule was not applied.

    The two seeds here differ in steps per episode as well as in escalations, so
    pooling the counts and averaging the rates give different answers -- which is
    why a cost column has to say which one it came from.
    """
    from conftest import load_script

    module = load_script("browser_pmpo")
    for name in ("never", "always"):
        _cell_rows(tmp_path / f"eval-{name}.jsonl", [1.0, 0.0, 0.0], [0, 0, 0])
    for name in ("smeared_ce", "priced_ce", "pmpo"):
        # seed 0: one escalation per episode, on short episodes.
        _cell_rows(tmp_path / f"eval-{name}-price0.1-seed0.jsonl",
                   [1.0, 0.0, 0.0], [1, 1, 1])
        # seed 1: three, on longer ones, so its rate is higher over more steps.
        _cell_rows(tmp_path / f"eval-{name}-price0.1-seed1.jsonl",
                   [1.0, 1.0, 1.0], [3, 3, 3])

    found = module.verdict(tmp_path)
    single = found["policies"]["price_0.1"]["pmpo"]
    pooled = found["policies_pooled"]["price_0.1"]["pmpo"]

    # The single-seed block names its seed, so it cannot be quoted silently.
    assert single["seed"] == 0
    assert "not a mean over seeds" in single["seed_note"]
    assert single["reward"] == 1 / 3, "seed 0's three episodes"

    # The pooled block names every seed it covers and covers all of them.
    assert pooled["seeds"] == [0, 1]
    assert pooled["episodes"] == 6
    assert pooled["reward"] == 4 / 6

    # **The stored rate has to reproduce the counts stored beside it.** That is
    # the property a mean of per-seed rates fails, and the reason 0.263 rather
    # than 0.315 went into the paper's Esc. cell.
    assert abs(pooled["escalation_rate"] * pooled["steps"]
               - pooled["escalations"]) < 1e-9
    averaged = (1 / 2 + 3 / 4) / 2
    assert abs(averaged * pooled["steps"] - pooled["escalations"]) > 1e-3, (
        "the two definitions must disagree here or this test proves nothing")


def test_the_cell_schema_carries_the_time_each_step_took():
    """Nothing before 2026-10-06 recorded time, so every figure in the gate
    README's cost table was derived from 28831's 2.29s / 10.95s -- a different
    run over a different task set. The paper's metric is score net of cost, and
    time cannot be recovered after the run, so it has to be written as it
    happens. exps raised it; the user approved it.

    The test measures a real elapsed time rather than asserting the field
    exists: a schema change that writes 0.0 everywhere would satisfy the field
    and answer nothing.
    """
    import time

    from slime_htm.environment import EpisodeResult, StepRecord
    from slime_htm.rollout import rollout_episodes

    class SlowEnvironment:
        """Two steps, the second of which takes measurably longer."""

        def __init__(self):
            self.taken = []

        def reset(self, episode):
            self.taken = []
            # Sleeps, so the episode clock can be told apart from the sum of its
            # steps. `reset` builds a browser on the real path and that cost is
            # the episode's; a per-step sum drops it silently.
            time.sleep(0.04)
            return "page", ["click('1')", "click('2')"]

        def step(self, action):
            self.taken.append(action)
            time.sleep(0.05 if len(self.taken) == 2 else 0.0)
            done = len(self.taken) >= 2
            return "page", ([] if done else ["click('1')", "click('2')"]), done, \
                (1.0 if done else 0.0)

    class Scorer:
        def fast(self, context, actions):
            return np.array([0.9, 0.1]), np.zeros(4, dtype=np.float32)

    class NeverEscalate:
        def predict(self, features):
            return np.zeros((1, 4), dtype=np.float32)

    results = list(rollout_episodes(
        SlowEnvironment(), Scorer(), NeverEscalate(), episodes=1, budget=16,
        max_steps=4, escalate_prob=0.0, lambda_=0.0, rng=np.random.default_rng(0)))
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, EpisodeResult) and len(result.steps) == 2
    assert all(isinstance(step, StepRecord) for step in result.steps)

    assert result.steps[1].seconds >= 0.05, \
        "the step that slept has to show it; a schema writing 0.0 answers nothing"
    assert result.steps[0].seconds < result.steps[1].seconds
    walked = sum(step.seconds for step in result.steps)
    assert result.seconds > walked + 0.03, (
        f"the episode clock must include reset: {result.seconds:.3f}s against "
        f"{walked:.3f}s of steps, and reset slept 0.04s. A sum over the steps "
        f"would satisfy >= and lose the environment build")


def test_the_marginal_value_of_an_escalation_is_recovered_and_never_pooled():
    """The number 28853 should have been priced from and was not.

    Its `lambda = 0.1` was the midpoint of the interval the head trains over,
    chosen without asking what an escalation is worth. Afterwards it came out at
    about 0.003 of a point, so 0.1 put every routing policy at a loss and the
    pre-registered test compared two losers. That was a design error of mine and
    this function is what prevents the next one.

    The test plants a known slope and asserts it is recovered, rather than
    asserting the output has the right keys: an estimator that returned zeros
    would satisfy a shape check and would pick the same bad price.
    """
    from conftest import load_script

    module = load_script("browser_pmpo")

    def episode(seed, escalations, reward, seconds, tokens_each=400):
        return {"seed": seed, "terminal_reward": reward, "seconds": seconds,
                "head_features": [
                    {"escalated": index < escalations,
                     "generated_tokens": tokens_each if index < escalations else 0}
                    for index in range(5)]}

    # Two seeds with deliberately *different* slopes, so pooling them would give
    # a number equal to neither. 0.20 and 0.05 reward per escalation.
    planted = []
    for count in range(5):
        planted.append(episode(0, count, 0.20 * count, 10.0 + 5.0 * count))
        planted.append(episode(1, count, 0.05 * count, 10.0 + 2.0 * count))

    report = module.marginal_value(planted)["per_seed"]
    assert set(report) == {"0", "1"}, "per seed, and only the seeds present"
    assert report["0"]["per_escalation_reward"] == pytest.approx(0.20, abs=1e-6)
    assert report["1"]["per_escalation_reward"] == pytest.approx(0.05, abs=1e-6)
    assert report["0"]["per_escalation_seconds"] == pytest.approx(5.0, abs=1e-6)
    assert report["1"]["per_escalation_seconds"] == pytest.approx(2.0, abs=1e-6)
    assert report["0"]["per_escalation_tokens"] == pytest.approx(400.0, abs=1e-6)

    # Pooling the two would give 0.125, which is neither. Asserted so that a
    # future "simplification" to one slope fails here: two single-seed numbers
    # have already been overturned on this venue in one day.
    assert report["0"]["per_escalation_reward"] != pytest.approx(0.125, abs=1e-3)

    # A corpus where nothing varied cannot answer, and says so rather than
    # dividing by zero or returning 0.0 -- which would read as "escalating is
    # worthless" and pick the highest price on offer.
    flat = [episode(0, 2, 0.5, 20.0) for _ in range(4)]
    assert module.marginal_value(flat)["per_seed"]["0"][
        "per_escalation_reward"] is None
    assert "slope undefined" in module.marginal_value(flat)["lines"][0]

    # And the caveat travels with the number, because the behaviour policy
    # escalated at random but harder episodes still drew more escalations.
    assert "not a causal" in module.marginal_value(planted)["caveat"]


def test_a_price_the_pre_registration_does_not_name_is_labelled_descriptive():
    """code-reviewer's condition, and this run is exactly the case.

    `middle = prices[len(prices) // 2]` makes a lone `--lambdas 0.0` look like
    an interval's midpoint, so `<- PRIMARY` would print over a comparison nobody
    registered. The registered price is `PRIMARY_PRICE`; changing it is changing
    the pre-registration, which is a thing to do before data and on purpose, not
    by passing a different `--lambdas`.
    """
    from conftest import load_script

    module = load_script("browser_pmpo")
    assert module.PRIMARY_PRICE == 0.1, "28853's price, which the README fixes"

    def cell(rewards, escalations):
        # A list, because `paired_test` zips the two cells and a dict would zip
        # its keys. Found by the test failing on `'int' object has no attribute
        # 'reward'`, which is the right way round.
        return [module.Played(
            episode=index, reward=reward, finished=True,
            steps=[{"escalated": True, "generated_tokens": 10, "seconds": 1.0}] * n)
            for index, (reward, n) in enumerate(zip(rewards, escalations,
                                                    strict=True))]

    routed = {}
    for price in (0.0, 0.1):
        routed[("pmpo", price)] = cell([1.0, 0.0, 1.0], [1, 1, 1])
        routed[("priced_ce", price)] = cell([0.0, 0.0, 1.0], [1, 1, 1])

    registered = module.primary_verdict(routed, 0.1, seed=0)
    assert registered["status"] == "registered"
    assert "isolates the off-policy treatment" in registered["reason"]

    descriptive = module.primary_verdict(routed, 0.0, seed=0)
    assert descriptive["status"] == "descriptive"
    assert "not the pre-registered test" in descriptive["reason"]
    assert descriptive["registered_price"] == 0.1
    # The numbers are still computed -- the label is what changes, because the
    # comparison is worth reading and is not worth calling primary.
    assert descriptive["ahead"] == registered["ahead"]


def test_time_that_was_never_recorded_does_not_read_as_zero_seconds():
    """code-reviewer's condition on 05e58bc, and the pattern 5ca7ed3 had just
    finished fixing on `used_outside_action`.

    Every reader used `.get("seconds", 0.0)`, so 28853's cells -- written before
    timing existed -- would summarise as `0.00s`, which reads as a measurement
    that thinking is instantaneous rather than as an absence.
    """
    from conftest import load_script

    module = load_script("browser_pmpo")

    timed = module.played_from({
        "index": 0, "terminal_reward": 1.0, "finished": True, "seconds": 7.5,
        "episode_steps": [{"escalated": True, "generated_tokens": 10, "seconds": 5.0},
                          {"escalated": False, "generated_tokens": 0, "seconds": 2.0}]})
    assert timed.timed is True
    assert timed.slow_seconds == pytest.approx(5.0)
    assert timed.fast_seconds == pytest.approx(2.0)
    assert timed.seconds == pytest.approx(7.5)

    archived = module.played_from({
        "index": 0, "terminal_reward": 1.0, "finished": True,
        "episode_steps": [{"escalated": True, "generated_tokens": 10},
                          {"escalated": False, "generated_tokens": 0}]})
    assert archived.timed is False
    assert archived.seconds is None, "absent, not zero"
    assert archived.slow_seconds is None and archived.fast_seconds is None

    # And the estimator keeps the two reasons apart: not recorded, versus
    # recorded and flat.
    def episode(count, seconds=None):
        row = {"seed": 0, "terminal_reward": 0.2 * count,
               "head_features": [{"escalated": i < count, "generated_tokens": 0}
                                 for i in range(5)]}
        if seconds is not None:
            row["seconds"] = seconds
        return row

    untimed = module.marginal_value([episode(n) for n in range(5)])
    assert untimed["per_seed"]["0"]["seconds_recorded"] is False
    assert untimed["per_seed"]["0"]["per_escalation_seconds"] is None
    assert "time not recorded" in untimed["lines"][0]

    with_time = module.marginal_value(
        [episode(n, seconds=10.0 + 3.0 * n) for n in range(5)])
    assert with_time["per_seed"]["0"]["seconds_recorded"] is True
    assert with_time["per_seed"]["0"]["per_escalation_seconds"] == pytest.approx(3.0)


def test_threads_guard_the_heads_and_not_the_corpus():
    """code-reviewer's condition on 6969f20, caught before the run started.

    Threads change the **fitted heads**, not the collected episodes: at
    `--rounds 1` the collection plays the explore coin and trains nothing. I had
    put them in `collection_signature`, so a resubmission landing on a node with
    a different `SLURM_CPUS_ON_NODE` would have discarded a valid 81-episode
    corpus -- about 1.2h -- while refitting the heads anyway.

    They belong in the collection's signature once `--rounds > 1`, where a
    mid-collection head steers the behaviour policy.
    """
    import os

    from conftest import load_script

    module = load_script("browser_pmpo")
    assert set(module.THREAD_VARIABLES) == {
        "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"}

    before = {name: os.environ.get(name) for name in module.THREAD_VARIABLES}
    try:
        os.environ["OMP_NUM_THREADS"] = "7"
        assert module.thread_settings()["OMP_NUM_THREADS"] == "7", \
            "read at call time, not frozen at import"
    finally:
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    # The placement is asserted on the source, because building both signatures
    # needs a live environment and a decision service.
    import inspect

    source = inspect.getsource(module.main)
    collection = source[source.index("collection_signature = {"):
                        source.index("evaluation_signature = dict(")]
    evaluation = source[source.index("evaluation_signature = dict("):]
    assert "thread_settings()" in evaluation, \
        "the evaluation signature covers the heads, so it carries them"
    assert "args.rounds > 1" in collection, (
        "the collection signature carries them only when a round can train a "
        "head that then steers collection")


def test_a_stored_verdict_with_no_status_is_not_called_primary():
    """The defaulting, which the placement test could not reach.

    `--verdict` reads a JSON that may predate the `status` field -- 28853's does
    -- and `primary.get("status", "registered")` would have printed "primary"
    over it. That is the same shape as three of today's other defects: an absent
    key taking the favourable reading.
    """
    from conftest import load_script

    module = load_script("browser_pmpo")

    status, heading, note = module.primary_heading(
        {"status": "registered", "price": 0.1})
    assert (status, heading, note) == ("registered", "primary", "")

    status, heading, note = module.primary_heading(
        {"status": "descriptive", "price": 0.0, "registered_price": 0.1})
    assert status == "descriptive" and "DESCRIPTIVE" in heading
    assert "registered price is 0.1" in heading
    assert note == "no pre-registered sentence is licensed by the line above"

    # The one that matters: no status at all, as in every pre-6969f20 file.
    status, heading, note = module.primary_heading({"price": 0.1})
    assert status == "unknown", "an absent status is not a registered one"
    assert "primary" not in heading.lower()
    assert "unrecorded" in note


def test_the_price_comes_from_a_between_policy_contrast_not_a_within_corpus_slope():
    """29074 showed the within-corpus slope flips sign, so it cannot price a run.

    `marginal_value` regresses terminal reward on escalation count inside one
    corpus. The behaviour policy escalated at random, but harder episodes still
    drew more escalations, and on 29074 that confound is larger than the effect:
    the slope is **-0.0739** while every between-policy contrast on the same run
    is non-negative (+0.0191 for always against never, +0.0000..+0.1250 across
    the nine routed cells). Nine of nine positive against one negative. I had
    named the slope in the pre-registration as the thing a price should come
    from; this test is what stops that happening again.
    """
    from conftest import load_script

    module = load_script("browser_pmpo")

    def cell(reward, escalations):
        return {index: module.Played(
            episode=index, reward=reward, finished=True,
            steps=[{"escalated": True, "generated_tokens": 100}] * escalations)
            for index in range(9)}

    cells = {"never": cell(0.3704, 0), "always": cell(0.5185, 8)}
    found = module.between_policy_value(cells)
    assert found["always"] == pytest.approx((0.5185 - 0.3704) / 8)
    assert found["always"] > 0, "the contrast is positive where the slope was not"

    # A policy that escalates as little as `never` cannot price an escalation,
    # and says so rather than dividing by ~0.
    cells["timid"] = cell(0.40, 0)
    assert module.between_policy_value(cells)["timid"] is None

    # **The demonstration, on planted data**: a corpus where escalating truly
    # helps but hard episodes draw more of it. The within-corpus slope reads
    # negative; the between-policy contrast reads positive.
    corpus = []
    for difficulty in range(9):
        # harder episodes escalate more and still score worse
        escalations = difficulty
        reward = max(0.0, 0.9 - 0.1 * difficulty)
        corpus.append({"seed": 0, "terminal_reward": reward, "seconds": 10.0,
                       "head_features": [{"escalated": i < escalations,
                                          "generated_tokens": 0}
                                         for i in range(9)]})
    slope = module.marginal_value(corpus)["per_seed"]["0"]["per_escalation_reward"]
    assert slope < 0, "the confound reverses the within-corpus slope"
    assert found["always"] > 0 > slope, (
        "the two estimators disagree in sign on confounded data, which is why "
        "only the contrast may set a price")
