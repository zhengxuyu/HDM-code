"""Optional Gymnasium environments; importing HDE itself does not require games."""

import gymnasium as gym
from gymnasium.envs.registration import register, registry

for name, entry in {
    "HDE/ChessKeys-v0": "hde.games.chess:ChessKeysEnv",
    "HDE/DoomButtons-v0": "hde.games.doom:DoomButtonsEnv",
}.items():
    if name not in registry:
        register(id=name, entry_point=entry)


def make_env(name, **kwargs):
    names = {"chess": "HDE/ChessKeys-v0", "doom": "HDE/DoomButtons-v0"}
    if name not in names:
        raise ValueError(f"Unknown environment {name!r}; choose chess or doom")
    return gym.make(names[name], **kwargs)
