"""The off-policy head update, on episodes built to have a known right answer."""
import math

import numpy as np
import pytest

from litjev._legacy.heads import HeadMetadata
from slime_htm.head_trainer import (
    HeadRLConfig,
    effective_sample_size,
    flatten_episodes,
    train_head_on_episodes,
)

WIDTH = 16


def metadata():
    return HeadMetadata(model_id="m", revision="r", hidden_size=WIDTH,
                        feature_layers=(0,), hidden_width=32, pca_dim=0)


def episode(reward, escalations, signal=1.0, steps=4, seed=0):
    """An episode whose feature carries `signal` in its first coordinate.

    Escalating is right exactly when the signal is positive, so a head that
    learns anything should end up escalating on positive signal and not on
    negative.
    """
    rng = np.random.default_rng(seed)
    features = []
    for step in range(steps):
        hidden = rng.normal(scale=0.1, size=WIDTH).astype(np.float32)
        hidden[0] = signal
        features.append({
            "step": step, "hidden": hidden,
            "probabilities": np.array([0.6, 0.4], dtype=np.float32),
            "candidates": 2, "escalated": step < escalations,
            "generated_tokens": 10 if step < escalations else 0, "gain": None,
        })
    return {"terminal_reward": reward, "head_features": features}


def test_effective_sample_size_counts_equal_weights_and_discounts_one_dominating():
    assert effective_sample_size([1.0] * 10) == pytest.approx(10.0)
    assert effective_sample_size([100.0] + [0.01] * 99) < 1.1
    assert effective_sample_size([0.0, 0.0]) == 0.0


def test_every_step_of_an_episode_carries_the_episode_reward_unpriced():
    """`flatten_episodes` hands back the reward and what the episode spent,
    separately. It used to hand back a return already priced at a fixed
    `cost_weight`, which is what made the price absent from the reward."""
    config = HeadRLConfig()
    _, _, _, rewards, spent, index = flatten_episodes(
        [episode(1.0, 2), episode(0.0, 1)], config)
    assert rewards.tolist() == [1.0] * 4 + [0.0] * 4
    assert spent.tolist() == [2.0] * 4 + [1.0] * 4
    assert index.tolist() == [0] * 4 + [1] * 4


def test_the_price_of_an_escalation_comes_off_the_reward_at_the_drawn_price():
    """The user's correction: the reward a decision is judged by has to be whether
    the slow mode's gain was worth it, and that depends on the price -- so the
    price belongs in the reward. It was in the policy term only, drawn per
    decision, while the reward was priced once at a separate fixed number.

    One price now, and the value moves with it, which is what lets one head serve
    every price."""
    from slime_htm.head_trainer import priced_value

    _, _, _, rewards, spent, _ = flatten_episodes(
        [episode(1.0, escalations=2)], HeadRLConfig())

    assert priced_value(rewards, spent, 0.0).tolist() == [1.0] * 4
    # The price discounts rather than subtracts: R * exp(-lambda * escalations),
    # two escalations here.
    assert priced_value(rewards, spent, 0.25) == pytest.approx([math.exp(-0.5)] * 4)
    assert priced_value(rewards, spent, 0.5) == pytest.approx([math.exp(-1.0)] * 4)
    # Monotone in the price, and never reaching zero: a correct answer that waited
    # is still a correct answer.
    assert (priced_value(rewards, spent, 0.75)
            < priced_value(rewards, spent, 0.25)).all()
    assert (priced_value(rewards, spent, 5.0) > 0).all()

    # **A failed episode is worth zero at every price**, because the discount is
    # multiplicative. So no price makes a success that thought throughout score
    # below a failure that did not -- the inversion the linear price allowed, and
    # the one the user asked to remove: cost alone should not make a right answer
    # worth less than nothing.
    #
    # The cost of that: this objective gives no gradient for the thinking spent on
    # an episode that failed anyway. The price shapes behaviour on successes only.
    cheap_failure = flatten_episodes([episode(0.0, escalations=0)], HeadRLConfig())
    dear_failure = flatten_episodes([episode(0.0, escalations=4)], HeadRLConfig())
    assert priced_value(cheap_failure[3], cheap_failure[4], 0.75).tolist() == [0.0] * 4
    assert priced_value(dear_failure[3], dear_failure[4], 0.75).tolist() == [0.0] * 4
    assert priced_value(rewards, spent, 0.75)[0] > priced_value(
        cheap_failure[3], cheap_failure[4], 0.75)[0]


def test_the_head_learns_to_escalate_where_escalating_paid():
    # Positive signal: escalating earned the reward. Negative signal: it did not.
    episodes = ([episode(1.0, escalations=4, signal=+1.0, seed=s) for s in range(20)]
                + [episode(0.0, escalations=4, signal=-1.0, seed=100 + s) for s in range(20)])
    head, report = train_head_on_episodes(metadata(), episodes,
                                          HeadRLConfig(epochs=40, behaviour_prob=1.0), seed=0)
    positive = head.predict(np.stack([
        f for f, *_ in [flatten_episodes([episode(1.0, 4, +1.0, seed=7)], HeadRLConfig())]][0]))
    negative = head.predict(flatten_episodes(
        [episode(0.0, 4, -1.0, seed=8)], HeadRLConfig())[0])
    gain_positive = float(np.mean(positive[:, 2] - positive[:, 1]))
    gain_negative = float(np.mean(negative[:, 2] - negative[:, 1]))
    assert gain_positive > gain_negative, (gain_positive, gain_negative, report["mean_return"])


def test_equal_rewards_leave_no_advantage_to_learn_from():
    episodes = [episode(1.0, escalations=2, signal=float(s), seed=s) for s in range(8)]
    _, report = train_head_on_episodes(metadata(), episodes, HeadRLConfig(epochs=5), seed=0)
    assert report["advantage_spread"] == pytest.approx(0.0, abs=1e-9)


def test_the_report_leads_with_how_much_of_the_sample_survived_reweighting():
    episodes = [episode(float(s % 2), escalations=2, signal=float(s), seed=s) for s in range(12)]
    _, report = train_head_on_episodes(metadata(), episodes, HeadRLConfig(epochs=5), seed=0)
    for key in ("effective_sample_size", "effective_fraction", "ratio_max", "clipped_fraction",
                "decisions", "decisions_per_episode", "escalation_rate_behaviour"):
        assert key in report, key
    assert 0.0 <= report["effective_fraction"] <= 1.0
    assert report["decisions"] == 48 and report["decisions_per_episode"] == pytest.approx(4.0)


def test_episodes_with_no_steps_are_skipped_and_an_empty_batch_is_an_error():
    config = HeadRLConfig()
    empty = {"terminal_reward": 1.0, "head_features": []}
    features, *_ = flatten_episodes([empty, episode(1.0, 1)], config)
    assert len(features) == 4
    with pytest.raises(ValueError, match="no decisions"):
        flatten_episodes([empty], config)


def test_the_behaviour_probability_of_the_acting_head_is_recorded_not_assumed():
    """After round one the acting policy is a head, so its rate varies by step."""
    import numpy as np

    from slime_htm.loop import behaviour_probability

    class Head:
        def predict(self, features):
            # two steps: one well above the threshold, one well below
            return np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    probabilities = behaviour_probability(Head(), np.zeros((2, 3)), lambda_=0.0, temperature=0.05)
    assert probabilities[0] > 0.99 and probabilities[1] < 0.01
    assert behaviour_probability(None, np.zeros((2, 3)), 0.0, 0.05) is None


def test_a_drawn_price_lands_in_the_range_and_moves_between_epochs():
    """The range is what makes the gain ordered across prices rather than at one."""
    import torch

    from slime_htm.head_trainer import HeadRLConfig, draw_prices

    config = HeadRLConfig(lambda_range=(-0.2, 0.5))
    generator = torch.Generator().manual_seed(0)
    first = draw_prices(256, config, generator)
    second = draw_prices(256, config, generator)
    assert float(first.min()) >= -0.2 and float(first.max()) <= 0.5
    assert not torch.allclose(first, second)


def test_without_a_range_every_decision_sees_the_same_price():
    import torch

    from slime_htm.head_trainer import HeadRLConfig, draw_prices

    prices = draw_prices(16, HeadRLConfig(lambda_=0.1), torch.Generator().manual_seed(0))
    assert torch.allclose(prices, torch.full((16,), 0.1))


def test_an_empty_range_is_refused_rather_than_silently_fixed():
    import pytest
    import torch

    from slime_htm.head_trainer import HeadRLConfig, draw_prices

    with pytest.raises(ValueError, match="interval"):
        draw_prices(4, HeadRLConfig(lambda_range=(0.5, 0.5)), torch.Generator())


def test_the_price_range_reaches_the_alfworld_launcher():
    """A price range nobody can set leaves the objective at a single price, where
    the discount factor is exactly 1 and the policy term only cares which side of
    one threshold each gain falls on.

    `htm_rl_loop.py` had `--lambda` and no range, so every ALFWorld run trained
    at a fixed price -- which is why a rerun at the defaults would have reproduced
    27608 round 0 numerically rather than testing the discounted reward. exps
    caught it. `browser_pmpo.py` has had the range all along, so this is the
    fourth wiring gap of the same shape in four days: `probe_prob` absent from its
    launcher, the probe absent from the corpus writer, and now this.
    """
    import pathlib

    launcher = (pathlib.Path(__file__).resolve().parents[1]
                / "scripts" / "htm_rl_loop.py").read_text()
    assert '"--price-range"' in launcher, "no flag means nothing can set it"
    # And it must reach the head's config, not merely be parsed.
    assert "lambda_range=" in launcher, (
        "the flag has to become HeadRLConfig(lambda_range=...), or the objective "
        "stays at one price however the flag is set")


def test_a_single_price_leaves_the_discount_inert():
    """Why the range matters, as arithmetic rather than as an assertion. At one
    fixed price of zero the discounted return is the raw return, so the
    exponential reward and the linear one are numerically identical and a run
    cannot tell them apart."""
    import numpy as np

    from slime_htm.head_trainer import HeadRLConfig, draw_prices, priced_value

    rewards = np.array([1.0, 1.0, 0.0, 1.0])
    spent = np.array([3.0, 0.0, 5.0, 11.0])

    fixed = draw_prices(len(rewards), HeadRLConfig(lambda_=0.0), None).numpy()
    assert (fixed == 0.0).all(), "no range means one price"
    assert priced_value(rewards, spent, fixed).tolist() == rewards.tolist(), (
        "at price zero the discount is the identity, whatever the escalations")

    # With a range the price varies per decision, so the same episodes are
    # valued differently and the head has to order gains across the whole range.
    import torch

    drawn = draw_prices(len(rewards), HeadRLConfig(lambda_range=(0.0, 0.3)),
                        torch.Generator().manual_seed(0)).numpy()
    assert len(set(drawn.tolist())) > 1, "a range must actually vary"
    valued = priced_value(rewards, spent, drawn)
    assert not np.allclose(valued, rewards), "and must change what the episodes are worth"
    assert (valued[rewards == 0.0] == 0.0).all(), "a failure stays worth zero at every price"


def test_the_alfworld_loop_signs_the_corpus_it_writes(tmp_path, monkeypatch):
    """A corpus with no signature row cannot say what it was collected under, and
    `checked_rows` returns unchecked when the signature is None -- so a reused
    `--output` silently resumes onto the earlier corpus.

    `htm_rl_loop.py` called `run(...)` with no `signature=`, so every ALFWorld
    corpus ever written carried no provenance: not the split, the offset, the
    episode count, the model, the code, lambda, or the price range. That was the
    fifth wiring gap of this shape, and it landed on the one requirement I had
    called non-negotiable -- that the corpus be the durable asset. exps found it.

    This **calls** the signature rather than grepping the launcher. The previous
    version asserted `'"commit"' in launcher` and went red when that field was
    renamed to `code` -- red for the rename, not for anything being wrong, which
    is what a source assertion is worth.
    """
    import importlib.util
    import inspect
    import pathlib
    import types

    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "htm_rl_loop_signature", root / "scripts" / "htm_rl_loop.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert "signature=" in inspect.getsource(module.main), \
        "run() must be given the signature, not merely handed one to print"

    # Every value distinctive and none of them a default, so a field wired to a
    # constant fails here. Checked: with `feature_layers` hardcoded to `[-1]`
    # this test went green while the fixture also said `(-1,)`, which is why the
    # fixture now says `(16, 24)`.
    args = types.SimpleNamespace(train_offset=7, episodes=137, model="qwen38-27b",
                                 budget=333, max_steps=41, explore_prob=0.37,
                                 lambda_=0.07)
    metadata = types.SimpleNamespace(feature_layers=(16, 24))
    (tmp_path / "REVISION").write_text("commit abc\ntree-id abc-d1-u2\n")
    monkeypatch.chdir(tmp_path)
    signature = module.collection_signature(args, metadata, (0.11, 0.29),
                                            ["game-a", "game-b"], None)

    # The fields that make two corpora incomparable, each checked by the value it
    # carries rather than by its presence: a field wired to a constant is the
    # defect this is for. **`feature_layers` above all**, which exps found
    # missing and which is the only one here that corrupts silently --
    # `build_features` flattens the hidden state, so the width depends on how
    # many layers were read and not on which, and two single-layer corpora pool
    # with no error at all.
    assert signature["feature_layers"] == [16, 24]
    assert signature["offset"] == 7 and signature["episodes"] == 137
    assert signature["model"] == "qwen38-27b"
    assert signature["budget"] == 333 and signature["max_steps"] == 41
    assert signature["explore_prob"] == 0.37 and signature["lambda"] == 0.07
    assert signature["price_range"] == [0.11, 0.29]
    assert signature["train_games"] == ["game-a", "game-b"]
    assert signature["venue"] == "alfworld" and signature["split"] == "train"
    # And the code, read from the record because the directory a job runs in is
    # not a checkout. `None` here is the bug this field existed through.
    assert signature["code"] == "abc-d1-u2"

    # It must **not** record what only changes the process after collecting.
    # `checked_rows` compares the whole dict, so a field that alters no row
    # renames a 15-hour corpus to `.stale-<timestamp>` and collects from zero.
    # Each round streams to its own `episodes-round{index}.jsonl`, so `rounds`
    # protects nothing: extending 2 rounds to 3 would discard round 0's rows,
    # which mean exactly what they meant before.
    for field in ("rounds", "eval_split", "eval_episodes"):
        assert field not in signature, (
            f"{field} changes no collection row, so it must not partition the "
            f"corpus")


def test_alfworld_can_be_given_an_explicit_game_list():
    """Two runs that both ask for `valid_unseen` from offset 0 can get different
    games, and ours did: the fast run and the slow shards record the same game
    indices while their task text differs on 89 of 134, and 45 episodes' task
    appears only in one of them. `load` slices `env.game_files[offset:offset +
    games]`, which is deterministic given the list -- so the list is not stable
    across jobs, and `AlfworldEnvironment.game_files` exists as a property whose
    docstring says it is "for the run's record" while no record writes it.

    That voided the 30-spoils/24-rescues pairing, the oracle, and the comparison
    the venue was chosen on. So a shard must be nameable rather than sliced.
    """
    import inspect

    from slime_htm.envs.alfworld.environment import AlfworldEnvironment

    parameters = inspect.signature(AlfworldEnvironment.load).parameters
    assert "game_files" in parameters, (
        "load must accept an explicit list, or 'the same games' rests on an "
        "ordering that has already differed between two of our own runs")
    assert parameters["game_files"].default is None, "slicing stays the default"

    # And the list must be readable back off a loaded shard, so a run can record
    # what it played rather than what it asked for.
    assert isinstance(AlfworldEnvironment.game_files, property)


def test_a_round_records_which_games_it_played():
    """What was asked for is in the signature; what was played has to be in the
    row. ALFWorld's two endpoints both recorded `offset: 0` and ran different
    games, and that was only recoverable by comparing task strings after the
    fact."""
    import inspect

    from slime_htm.loop import run

    source = inspect.getsource(run)
    assert '"games": (list(collecting.game_files)' in source, (
        "the round row must carry the played game list")
    assert 'hasattr(collecting, "game_files")' in source, (
        "an environment that cannot say must record None rather than raise")


def test_both_alfworld_call_sites_can_name_their_games():
    """The loop and the rollout script selected games by two different mechanisms.
    The loop slices `env.game_files[offset:offset+games]`; the script set
    `config["dataset"]["num_eval_games"] = offset + games` and burned resets to
    reach the offset -- so **the registered set depended on how many games were
    asked for**. That is how this venue's endpoints diverged: the fast run asked
    for 134 and the slow run asked for 45 at a time, and 45 of 134 task texts
    appear in only one of them.

    So both call sites must be able to name the shard, and the record must say
    which games were played.
    """
    import inspect
    import pathlib

    from slime_htm.envs.alfworld.environment import AlfworldEnvironment

    assert "game_files" in inspect.signature(AlfworldEnvironment.load).parameters
    assert isinstance(AlfworldEnvironment.engine, property), (
        "the rollout script shares the selection without the step loop, so the "
        "engine needs a public name rather than a reach for `_env`")

    scripts = pathlib.Path(__file__).resolve().parents[1] / "scripts"
    rollout = (scripts / "alf_rollout.py").read_text()
    assert '"--game-list"' in rollout, "the rollout script must accept a named shard"
    assert "AlfworldEnvironment.load(split=args.split, game_files=names" in rollout, (
        "and must select through the shared loader, not its own count mechanism")
    assert '"game_files": played' in rollout, "and must record what it played"

    loop = (scripts / "htm_rl_loop.py").read_text()
    assert '"--train-games"' in loop and '"--eval-games"' in loop


def test_the_pinned_game_lists_are_regenerable_from_a_documented_command():
    """exps produced the committed lists with an ad-hoc call to the loader, which
    is reproducible only because they said what it was. A documented mode makes
    them regenerable without that, and it has to go through the same loader --
    enumerating the split any other way is the two-mechanism problem the lists
    exist to close."""
    import inspect
    import pathlib

    from slime_htm.envs.alfworld.environment import AlfworldEnvironment

    assert hasattr(AlfworldEnvironment, "every_game"), (
        "the whole split must be readable without a second load at a different "
        "count, since differing counts is what produced two different sets")

    rollout = (pathlib.Path(__file__).resolve().parents[1]
               / "scripts" / "alf_rollout.py").read_text()
    assert '"--list-games"' in rollout
    assert "every.every_game()" in rollout, "and it must come from the loader"

    # `every` is captured before the shard is taken, or it would be the shard.
    source = inspect.getsource(AlfworldEnvironment.load)
    assert "every = list(env.game_files)" in source
    assert source.index("every = list(env.game_files)") < source.index(
        "if game_files is not None:"), "captured before the shard is sliced"
