"""The ALFWorld adapter.

Two facts about ALFWorld shape this.

**A shard is chosen by the game list, not by skipping.** `reset` takes no
arguments and serves the registered games in order, so reaching the nth game
looks like it needs n resets. It does not: `AlfredTWEnv` collects `game_files`
before `init_env` registers them, so slicing that list is how a shard is taken,
and the environment then holds only the shard's games. The files played are
recorded, because "episodes 100 to 199 of the train split" is a claim about an
ordering that a future version of the dataset could change.

Within a shard the order is still the engine's, one game per reset, so this
refuses to go backwards rather than quietly serving a different game than the
caller named.

**Its conventions are not guessable.** An object has to be taken from the
receptacle holding it, and a receptacle has to be reached before anything on it
can be used. A model that has never seen the format spends its steps discovering
that, which is why the benchmark is normally run with in-context examples. The
rules are stated in the state text instead, which costs nothing and is honest
about being a prompt choice rather than a property of the method.
"""

import os
import re
from pathlib import Path

ALFWORLD_COMMIT = "aaba6870f86c5be6a08a491f32a50b906227bc3e"
CONFIG = Path(__file__).resolve().parents[4] / "configs" / "alfworld_base.yaml"
# train_eval decides which directory the games come from, and each mode reads its
# own count key. Naming the wrong mode for a split is silent: asking for "train"
# with eval_in_distribution reads valid_seen, which is a validation split a
# twenty-fifth the size, and the run looks fine.
SPLITS = {
    "train": ("train", "num_train_games"),
    "valid_seen": ("eval_in_distribution", "num_eval_games"),
    "valid_unseen": ("eval_out_of_distribution", "num_eval_games"),
}
RULES = (
    "You are an agent in a household, acting through text commands. To use or take "
    "an object you must first go to the receptacle holding it; opening a closed "
    "receptacle reveals what is inside; 'take X from Y' picks up X, and "
    "'put X in/on Y' places it. Finish the task in as few steps as you can."
)


def expand(node, data):
    if isinstance(node, dict):
        return {key: expand(value, data) for key, value in node.items()}
    if isinstance(node, list):
        return [expand(value, data) for value in node]
    return node.replace("$ALFWORLD_DATA", data) if isinstance(node, str) else node


def load_config(split, games, data=None, config_path=CONFIG):
    """The pinned upstream config, with the split's game count set.

    The config travels with the repository and is pinned to an upstream commit,
    so a run is reproducible without the environment's packaging being involved.
    """
    import yaml

    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; choose from {sorted(SPLITS)}")
    data = data or os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))
    os.environ["ALFWORLD_DATA"] = data
    with open(config_path) as handle:
        config = expand(yaml.safe_load(handle), data)
    train_eval, count_key = SPLITS[split]
    # The shard is taken from game_files below, so the count is left open here.
    config["dataset"][count_key] = -1
    # dqn rather than dagger: the dagger path wraps the train split in an expert
    # that computes a plan for every episode, which nothing here reads, and it
    # takes the step limit from a different block. dqn reads rl.training, which is
    # where the 50-step limit this benchmark is run at already lives.
    config["general"]["training_method"] = "dqn"
    return config, train_eval


class AlfworldEnvironment:
    """One ALFWorld episode at a time, as `slime_htm.Environment`."""

    def __init__(self, env, episodes, history=6, rules=RULES, game_files=(),
                 every=()):
        self._env = env
        self._episodes = episodes
        self._history = history
        self._rules = rules
        self._game_files = list(game_files)
        self._every = list(every) or list(game_files)
        # How many of the shard's games the engine has served.
        self._position = 0
        self._task = ""
        self._trail = []

    @classmethod
    def load(cls, split="valid_unseen", offset=0, games=20, history=6, data=None,
             game_files=None):
        """Register exactly the shard's games, so no reset is spent reaching it.

        `game_files` names the shard outright. **Prefer it over `offset`/`games`
        for anything that has to be compared across runs**, because slicing
        `env.game_files` is deterministic only given that list, and the list is
        not stable across jobs. Ours was not: `alfworld-fast-0-27309` and the
        three slow shards all record `offset: 0` and game indices 0..133, and
        their task text differs on 89 of 134 -- 45 episodes' task appears only in
        the fast run and 45 only in the slow one, which is exactly the shard
        size. So the fast and slow endpoints of this venue were measured on
        different games, and every paired figure built on them -- 30 spoils, 24
        rescues, the per-episode oracle -- paired two different populations.

        The list is returned by `game_files` so a run can record what it played
        rather than what it asked for. Nothing recorded it before, which is why
        the above had to be inferred from task strings.
        """
        import alfworld.agents.environment as environment

        config, train_eval = load_config(split, games, data=data)
        env = environment.get_environment("AlfredTWEnv")(config, train_eval=train_eval)
        # The whole split, kept before the shard is taken, so `every_game` can
        # hand it back without a second load at a different count.
        every = list(env.game_files)
        if game_files is not None:
            wanted = list(game_files)
            if not wanted:
                raise ValueError("game_files is empty; omit it to slice by offset")
            have = set(env.game_files)
            missing = [name for name in wanted if name not in have]
            if missing:
                raise ValueError(
                    f"{len(missing)} of {len(wanted)} named games are not in split "
                    f"{split!r}; first missing {missing[0]!r}. A pinned list must be "
                    f"pinned against the same data directory it was written from")
            env.game_files = wanted
        else:
            available = len(env.game_files)
            if offset + games > available:
                raise ValueError(
                    f"split {split!r} has {available} games; asked for {games} from {offset}")
            env.game_files = env.game_files[offset : offset + games]
        env.num_games = len(env.game_files)
        return cls(env.init_env(batch_size=1), episodes=len(env.game_files),
                   history=history, game_files=env.game_files, every=every)

    @property
    def episodes(self):
        return self._episodes

    def every_game(self):
        """Every game in this shard's split, before the shard was taken.

        For writing the pinned lists: a caller that wants the whole split asks
        for it here rather than loading the split twice with different counts,
        which is the thing that produced two different sets in the first place.
        """
        return list(self._every)

    @property
    def engine(self):
        """The underlying alfworld env, for a caller that drives resets itself.

        Exposed so `scripts/alf_rollout.py --backend transformers` can share this
        class's game selection without sharing its step loop -- that path has its
        own, which reads logits directly. Reaching for `_env` from a script would
        work and would make the one thing that must stay identical between the two
        paths depend on a private name.

        `--backend sglang` needs none of this: it drives `rollout_episodes` over
        this class, so it shares the step loop as well as the selection, which is
        what puts the loop's head on its own training distribution.
        """
        return self._env

    @property
    def game_files(self):
        """Exactly which games this shard holds, for the run's record."""
        return list(self._game_files)

    @property
    def task(self):
        """The current episode's task text, empty before the first reset.

        For a caller's per-episode record. It has to be read while the episode is
        the current one, since the next `reset` replaces it, and it is worth
        recording: the task strings are what showed that this venue's fast and
        slow endpoints had been measured on different games.
        """
        return self._task

    def reset(self, episode):
        """Start the shard's episode `episode`. The engine serves them in order."""
        if episode < self._position:
            raise ValueError(
                f"ALFWorld serves its games in order and cannot return to {episode} from "
                f"{self._position}; play a shard through, or register another")
        while self._position < episode:
            self._env.reset()
            self._position += 1
        observation, info = self._env.reset()
        self._position += 1
        self._task = re.split(r"Your task is to:", observation[0])[-1].strip().split("\n")[0]
        self._trail = []
        return self._state(), list(info["admissible_commands"][0])

    def step(self, action):
        observation, score, done, info = self._env.step([action])
        text = observation[0].strip()
        self._trail.append(f"- {action} -> {text[:120]}")
        finished = bool(done[0])
        reward = float(score[0]) if finished else 0.0
        actions = [] if finished else list(info["admissible_commands"][0])
        return self._state(), actions, finished, reward

    def _state(self):
        lines = [self._rules, "", f"Task: {self._task}"]
        if self._trail:
            lines += ["", "Recent steps:", *self._trail[-self._history :]]
        return "\n".join(lines)


def build(args, evaluation=False):
    """The factory `--htm-environment-path slime_htm.envs.alfworld:build` names.

    Evaluation uses the unseen split and training the train split, so the two are
    one setting rather than two that can disagree.
    """
    split = (getattr(args, "htm_eval_split", "valid_unseen") if evaluation
             else getattr(args, "htm_train_split", "train"))
    return AlfworldEnvironment.load(
        split=split,
        offset=getattr(args, "htm_eval_offset", 0) if evaluation
        else getattr(args, "htm_offset", 0),
        games=getattr(args, "htm_eval_games", 134) if evaluation
        else getattr(args, "htm_games", 100),
        history=getattr(args, "htm_history", 6),
    )
