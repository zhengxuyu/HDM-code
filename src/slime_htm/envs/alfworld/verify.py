"""Check the split mapping and the shard slicing against the real engine.

Both were wrong once and the unit tests were green, because they asserted the
semantics I believed rather than the engine's. This asks the engine. It needs
alfworld installed and the games downloaded, and no model: it only counts games
and checks which ones a shard holds.
"""
import argparse
import json

from slime_htm.envs.alfworld.environment import SPLITS, AlfworldEnvironment, load_config

EXPECTED = {"train": 3553, "valid_seen": 140, "valid_unseen": 134}


def collected(split):
    """How many games the engine finds for a split, before any truncation."""
    import alfworld.agents.environment as environment

    config, train_eval = load_config(split, games=-1)
    env = environment.get_environment("AlfredTWEnv")(config, train_eval=train_eval)
    return env.game_files, train_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report, ok = {}, True
    for split in SPLITS:
        files, train_eval = collected(split)
        directory = sorted({f.split("/json_2.1.1/")[-1].split("/")[0] for f in files})
        expected = EXPECTED[split]
        agrees = len(files) == expected and directory == [split]
        ok = ok and agrees
        report[split] = {"train_eval": train_eval, "games": len(files),
                         "expected": expected, "directory": directory, "agrees": agrees}
        print(f"{split:14} mode {train_eval:26} {len(files):5d} games "
              f"(expected {expected}), under {directory}  "
              f"{'ok' if agrees else 'WRONG SPLIT'}", flush=True)

    # A shard must hold exactly its slice, and its first episode must be its own
    # first game rather than the split's.
    files, _ = collected("train")
    shard = AlfworldEnvironment.load(split="train", offset=7, games=3)
    held = shard.game_files
    sliced = held == files[7:10]
    report["shard"] = {"offset": 7, "games": 3, "held": len(held),
                       "matches_slice": sliced,
                       "first": held[0].split("/")[-2] if held else None}
    print(f"\nshard offset 7 games 3: holds {len(held)}, "
          f"matches game_files[7:10] {sliced}")
    ok = ok and sliced and len(held) == 3

    with open(args.output, "w") as handle:
        json.dump({"ok": ok, **report}, handle, indent=1)
    if not ok:
        raise SystemExit("the engine disagrees with SPLITS or with the shard slicing")
    print("\nthe engine agrees")


if __name__ == "__main__":
    main()
