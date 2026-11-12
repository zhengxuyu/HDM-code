"""The ALFWorld adapter, against a stand-in for the text engine.

alfworld is not installed here and does not need to be: what can be wrong
without it is the ordering, the state text and the reward, and all three are.
"""
import pytest

from slime_htm.envs.alfworld.environment import SPLITS, AlfworldEnvironment, load_config


class FakeEngine:
    """Hands out games in order, exactly as the text engine does."""

    def __init__(self, games):
        self.games = games
        self.served = -1
        self.position = 0

    def reset(self):
        self.served += 1
        self.position = 0
        game = self.games[self.served % len(self.games)]
        observation = f"-= Welcome =-\n\nYou are in a room.\n\nYour task is to: {game['task']}"
        return [observation], {"admissible_commands": [list(game["actions"])]}

    def step(self, actions):
        game = self.games[self.served % len(self.games)]
        self.position += 1
        done = self.position >= game["length"]
        score = game["score"] if done else 0
        nxt = [] if done else list(game["actions"])
        return [f"You did {actions[0]}."], [score], [done], {"admissible_commands": [nxt]}


def games(count=4):
    return [{"task": f"task {i}", "actions": ["go north", "go south"], "length": 2,
             "score": 1 if i % 2 == 0 else 0} for i in range(count)]


def environment(count=4, first=0, episodes=None):
    """`first` stands in for the slice load() takes out of the engine's game list."""
    engine = FakeEngine(games(count)[first:])
    return AlfworldEnvironment(engine, episodes=episodes or count - first), engine


def test_the_state_carries_the_rules_the_task_and_the_recent_trail():
    env, _ = environment()
    state, actions = env.reset(0)
    assert "Task: task 0" in state
    assert "go to the receptacle" in state, "the world's conventions are not guessable"
    assert "Recent steps:" not in state
    assert actions == ["go north", "go south"]
    state, actions, done, reward = env.step("go north")
    assert "Recent steps:" in state and "- go north ->" in state
    assert actions == ["go north", "go south"] and not done and reward == 0.0


def test_the_reward_arrives_only_when_the_episode_ends():
    env, _ = environment()
    env.reset(0)
    _, _, done, reward = env.step("go north")
    assert not done and reward == 0.0
    _, actions, done, reward = env.step("go south")
    assert done and reward == 1.0 and actions == []


def test_a_failed_episode_ends_with_no_reward():
    env, _ = environment()
    env.reset(1)                       # game 1 scores 0
    env.step("go north")
    _, _, done, reward = env.step("go south")
    assert done and reward == 0.0


def test_a_shard_starts_at_its_own_first_game_with_no_resets_spent():
    env, engine = environment(count=6, first=2, episodes=3)
    state, _ = env.reset(0)
    assert "Task: task 2" in state, "the shard's games are the only ones registered"
    assert engine.served == 0, "reaching the shard costs no reset"
    state, _ = env.reset(1)
    assert "Task: task 3" in state


def test_skipping_ahead_within_a_shard_is_allowed_and_going_back_is_refused():
    env, _ = environment(count=6, episodes=6)
    env.reset(3)
    with pytest.raises(ValueError, match="cannot return to"):
        env.reset(1)


def test_the_train_split_reads_the_train_directory_and_its_own_count_key():
    """Naming the wrong mode reads a validation split a twenty-fifth the size."""
    assert SPLITS["train"] == ("train", "num_train_games")
    assert SPLITS["valid_seen"] == ("eval_in_distribution", "num_eval_games")
    assert SPLITS["valid_unseen"] == ("eval_out_of_distribution", "num_eval_games")


def test_the_trail_is_bounded_so_the_prompt_does_not_grow_without_limit():
    env, engine = environment()
    engine.games[0]["length"] = 99
    env.reset(0)
    for turn in range(10):
        state, *_ = env.step(f"move {turn}")
    assert state.count("- move") == 6, "history is capped at the configured depth"


def test_an_unknown_split_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown split"):
        load_config("test", 1)


def test_every_split_maps_to_a_mode_the_engine_knows_and_the_count_key_it_reads():
    # The engine reads num_train_games only in "train" mode and num_eval_games in
    # the other two, so a mismatch truncates by a key nothing sets.
    for split, (train_eval, count_key) in SPLITS.items():
        assert train_eval in ("train", "eval_in_distribution", "eval_out_of_distribution"), split
        expected = "num_train_games" if train_eval == "train" else "num_eval_games"
        assert count_key == expected, split
