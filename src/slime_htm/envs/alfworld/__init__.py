"""ALFWorld as a hybrid-thinking environment.

ALFWorld suits this method because its text engine publishes the admissible
commands at every step. The fast mode answers by scoring an enumerated candidate
set, so an environment that says what the actions are needs nothing constructed
for it.

Use it through `--htm-environment-path slime_htm.envs.alfworld:build`.
"""

from slime_htm.envs.alfworld.environment import ALFWORLD_COMMIT, AlfworldEnvironment, build

__all__ = ["ALFWORLD_COMMIT", "AlfworldEnvironment", "build"]
