"""Check that ALFWorld gives the method what it needs, before anything is run on a GPU.

The fast mode answers by scoring an enumerated candidate set with one forward
pass. ALFWorld's text engine publishes the admissible commands at every step,
so the candidate set is native here and nothing has to be constructed for it.
This confirms that, and prints one episode's first two steps.
"""
import glob
import os

import yaml

CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "alfworld_base.yaml")
ALFWORLD_COMMIT = "aaba6870f86c5be6a08a491f32a50b906227bc3e"


def expand(node, data):
    if isinstance(node, dict):
        return {k: expand(v, data) for k, v in node.items()}
    if isinstance(node, list):
        return [expand(v, data) for v in node]
    if isinstance(node, str):
        return node.replace("$ALFWORLD_DATA", data)
    return node


data = os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))
os.environ["ALFWORLD_DATA"] = data
for split in ("train", "valid_seen", "valid_unseen"):
    games = glob.glob(os.path.join(data, "json_2.1.1", split, "**", "*.tw-pddl"), recursive=True)
    print(f"{split:14} {len(games):5d} games")
print(f"config pinned at alfworld {ALFWORLD_COMMIT[:8]}")

with open(CONFIG) as handle:
    config = expand(yaml.safe_load(handle), data)
config["dataset"]["num_eval_games"] = 2
print("action_space:", config["rl"]["action_space"],
      " max steps:", config["rl"]["training"]["max_nb_steps_per_episode"])

import alfworld.agents.environment as environment

env = environment.get_environment("AlfredTWEnv")(config, train_eval="eval_out_of_distribution")
env = env.init_env(batch_size=1)
obs, info = env.reset()
print("\n--- observation ---")
print(obs[0][:600])
commands = info["admissible_commands"][0]
print(f"\n--- {len(commands)} admissible commands ---")
print(commands[:15])
obs, score, done, info = env.step([commands[0]])
print(f"\nafter '{commands[0]}':\n{obs[0][:200]}")
print(f"\nnext step: {len(info['admissible_commands'][0])} commands, done={done[0]}, score={score[0]}")
print("SMOKE_OK")
