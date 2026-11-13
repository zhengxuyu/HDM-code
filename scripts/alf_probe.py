"""Go/no-go for ALFWorld: can this backbone solve anything, and does thinking help?

Nothing is trained here. The fast mode reads an action off the admissible
commands with one forward pass and no generated token; the slow mode reasons
first. If both succeed at near zero, or if they succeed equally often, there is
no routing decision to learn and the experiment stops here.
"""
import argparse
import os
import re
import time

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from hde.run.streaming import write_json_atomic

CODES = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
BOUNDARY = "\nAction:"


def expand(node, data):
    if isinstance(node, dict):
        return {k: expand(v, data) for k, v in node.items()}
    if isinstance(node, list):
        return [expand(v, data) for v in node]
    return node.replace("$ALFWORLD_DATA", data) if isinstance(node, str) else node


def single_token_codes(tokenizer, wanted):
    """Codes that are one token when they follow the boundary, in order.

    The fast mode reads its answer from one position, so a code that tokenises
    into two pieces cannot be read there. Checking this against the actual
    tokenizer is the difference between a candidate set that works and one that
    silently truncates.
    """
    usable = []
    for code in CODES:
        pieces = tokenizer.encode(" " + code, add_special_tokens=False)
        if len(pieces) == 1:
            usable.append((code, pieces[0]))
        if len(usable) == wanted:
            break
    return usable


def build_prompt(task, history, commands, codes, think):
    lines = [
        "You are an agent in a household, acting through text commands.",
        "Rules of this world: to use or take an object you must first 'go to' the",
        "receptacle holding it; opening a closed receptacle reveals what is inside;",
        "'take X from Y' picks up X, and 'put X in/on Y' places it. Finish the task",
        "in as few steps as you can.",
        "",
        f"Task: {task}",
        "",
    ]
    if history:
        lines.append("Recent steps:")
        lines.extend(history[-6:])
        lines.append("")
    lines.append("Available actions:")
    lines.extend(f"{code} {command}" for (code, _), command in zip(codes, commands))
    lines.append("")
    lines.append("Reason step by step, then give the code." if think
                 else "Reply with the code of the best action.")
    # ALFWorld's conventions are not guessable: an object has to be taken from
    # the receptacle it sits on, and a receptacle has to be reached first. A
    # model that has never seen the format spends its steps discovering it,
    # which is why the benchmark is normally run with in-context examples.
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0,
                        help="skip this many games first, so a run can be sharded")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--mode", choices=["fast", "slow"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))
    os.environ["ALFWORLD_DATA"] = data
    with open(os.path.join(os.path.dirname(__file__), "..", "configs",
                           "alfworld_base.yaml")) as handle:
        config = expand(yaml.safe_load(handle), data)
    # The environment hands out games in a fixed order, so a shard is a prefix
    # skipped and then the next `games` taken. Shards therefore tile the split
    # exactly, with no game run twice and none left out.
    config["dataset"]["num_eval_games"] = args.offset + args.games

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    codes = single_token_codes(tokenizer, len(CODES))
    print(f"single-token codes available: {len(codes)} of {len(CODES)}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    code_ids = torch.tensor([i for _, i in codes], device=model.device)

    import alfworld.agents.environment as environment
    env = environment.get_environment("AlfredTWEnv")(config, train_eval="eval_out_of_distribution")
    env = env.init_env(batch_size=1)

    episodes, generated_total, start = [], 0, time.time()
    for _ in range(args.offset):
        env.reset()
    for game in range(args.offset, args.offset + args.games):
        obs, info = env.reset()
        task = re.split(r"Your task is to:", obs[0])[-1].strip().split("\n")[0]
        history, success, steps, generated = [], False, 0, 0
        for step in range(args.max_steps):
            commands = info["admissible_commands"][0]
            usable = codes[: len(commands)]
            commands = commands[: len(usable)]
            prompt = build_prompt(task, history, commands, usable, args.mode == "slow")
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=args.mode == "slow")
            if args.mode == "slow":
                ids = tokenizer(text, return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    out = model.generate(**ids, max_new_tokens=args.budget, do_sample=False,
                                         pad_token_id=tokenizer.eos_token_id)
                generated += int(out.shape[1] - ids["input_ids"].shape[1])
                text = tokenizer.decode(out[0], skip_special_tokens=True)
            text = text + BOUNDARY
            ids = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                logits = model(**ids).logits[0, -1]
            choice = int(torch.argmax(logits[code_ids[: len(commands)]]).item())
            action = commands[choice]
            obs, score, done, info = env.step([action])
            history.append(f"- {action} -> {obs[0].strip()[:120]}")
            steps += 1
            if done[0]:
                success = bool(score[0] > 0)
                break
        generated_total += generated
        episodes.append({"game": game, "task": task, "success": success,
                         "steps": steps, "generated": generated})
        done_n = sum(e["success"] for e in episodes)
        print(f"[{game + 1}/{args.offset + args.games}] success={success} steps={steps} gen={generated} "
              f"running={done_n}/{len(episodes)}", flush=True)

    rate = sum(e["success"] for e in episodes) / len(episodes)
    summary = {"mode": args.mode, "games": len(episodes), "success_rate": rate,
               "mean_steps": sum(e["steps"] for e in episodes) / len(episodes),
               "generated_per_episode": generated_total / len(episodes),
               "seconds": time.time() - start, "model": args.model,
               "max_steps": args.max_steps, "budget": args.budget,
               "offset": args.offset, "episodes": episodes}
    write_json_atomic(args.output, summary)
    print(f"\n{args.mode}: success {rate:.3f} over {len(episodes)} games, "
          f"{summary['mean_steps']:.1f} steps, {summary['generated_per_episode']:.0f} tokens/episode")


if __name__ == "__main__":
    main()
