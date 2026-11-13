"""Does thinking pay on ScienceWorld? Run this before training anything.

ALFWorld cost a day of GPU before the answer arrived from the other direction:
always thinking scores 0.575 there against never thinking's 0.619, so the
benchmark was asking a router to learn to abstain, and the RL loop trained on it
produced a policy worse than both constant policies. The check that would have
said so costs two hours and is this script.

It runs the two endpoints and nothing else. No head, no lambda, no training:
fast only, slow only, the same variations, the same step limit. If the slow mode
is not worth more than it costs here, this benchmark cannot show a router
working and the right move is to say so rather than to tune.

One decision per step is not enough for ScienceWorld. Its admissible set is
templates crossed with objects in scope, which runs to 417 entries in an empty
hallway and 3249 in a kitchen, while the fast mode reads its answer from
single-token codes and this tokenizer yields 52. Filtering and truncating to 52
puts the gold action out of reach on most steps -- measured at 4 of 32 on
`melt` -- so a run on that would be measuring the truncation.

A step is therefore asked as up to three questions: which template, which first
object, which second object. Measured over five tasks, templates reach 26 and
objects reach 40, so each question fits. Every option is filtered from the
engine's own valid-combination list, so whatever is chosen composes into a legal
action. The three questions share one state, differing only in the suffix that
poses them, which is the shared prefill the paper describes; here SGLang's
prefix cache supplies it.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Stated because the conventions are not guessable and the cost of not knowing
# them is total. `focus on` is scored: the first episode run here chose
# `focus on air` as its opening move and the engine ended the episode at -100.
# An agent that has not been told this loses on step one, and a benchmark where
# both modes lose on step one compares nothing. This is a prompt choice, said
# out loud, not a property of the method.
RULES = (
    "You are an agent in a science laboratory, acting through text commands. "
    "Rules that matter:\n"
    "- 'focus on X' is scored. Focusing on the wrong object ends the task "
    "immediately as a failure. Focus only on the exact object the task names, "
    "and only once you have found it and are holding or standing at it.\n"
    "- Move between rooms with 'go <door>' or by opening a door first; you can "
    "only act on objects in the room you are in.\n"
    "- 'look around' reports what is in the room. 'inventory' reports what you "
    "are carrying.\n"
    "- Prefer looking and moving until you have found what the task names. "
    "Finish the task in as few steps as you can."
)


def stems(env):
    """Each declared template as the literal text before its first object."""
    found = []
    for template in env.get_possible_actions():
        stem = template.split(" OBJ")[0].strip()
        arity = template.count("OBJ")
        if stem and (stem, arity) not in found:
            found.append((stem, arity))
    # Longest first, so `look at` is not read as `look`.
    return sorted(found, key=lambda pair: -len(pair[0]))


def staged(env):
    """valid combos -> {(stem, arity): [combo, ...]}, every entry legal to send."""
    grouped = {}
    table = stems(env)
    for combo in env.get_valid_action_object_combinations():
        for stem, arity in table:
            if combo == stem or combo.startswith(stem + " "):
                grouped.setdefault((stem, arity), []).append(combo)
                break
    return grouped


def referents(grouped):
    """Every object name the engine used this step, longest first.

    Taken from the one-object combinations, where the name is whatever follows
    the template and needs no splitting. That matters because the names are
    compound: `thermometer in inventory`, `apple seed in seed jar`. Matching a
    two-object tail against the short list from `get_possible_objects` cuts those
    in half, which is how `focus on thermometer in inventory` became
    unreachable.
    """
    names = set()
    for (stem, arity), combos in grouped.items():
        if arity != 1:
            continue
        for combo in combos:
            rest = combo[len(stem):].strip()
            if rest:
                names.add(rest)
    return sorted(names, key=len, reverse=True)


def split_objects(rest, names):
    """A two-object tail as the pair of names it holds."""
    if not rest:
        return None, None
    first = next((n for n in names if rest == n or rest.startswith(n + " ")), None)
    if first is None:
        return rest, None
    tail = rest[len(first):].strip()
    if not tail:
        return first, None
    second = next((n for n in names if tail == n or tail.endswith(" " + n)), None)
    return first, second or tail


def options_for(stem, arity, combos, names, limit, chosen=None):
    """The distinct objects offered at this stage, in the engine's own order.

    A one-object template needs no splitting at all: the object is the rest of
    the combination. Only a two-object template is split, and only against the
    names the engine itself used.
    """
    seen, out = set(), []
    for combo in combos:
        rest = combo[len(stem):].strip()
        if arity == 1:
            first, second = rest, None
        else:
            first, second = split_objects(rest, names)
        value = second if chosen is not None else first
        if chosen is not None and first != chosen:
            continue
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out[:limit]


def state_text(rules, goal, here, trail, history=6):
    lines = [rules, "", f"Task: {goal}"]
    if here:
        lines += ["", f"Here: {here.strip()[:700]}"]
    if trail:
        lines += ["", "Recent steps:", *trail[-history:]]
    return "\n".join(lines)


def choose(decider, state, question, options, think, budget):
    """One readout, fast or slow, over an enumerated option list."""
    import numpy as np

    labelled = [f"{question}: {option}" for option in options]
    answer = (decider.slow(state, labelled, budget) if think
              else decider.fast(state, labelled))
    return options[int(np.argmax(answer["probabilities"]))], answer


def act(decider, env, state, think, budget, limit):
    """Compose a legal action out of up to three enumerated choices."""
    grouped = staged(env)
    if not grouped:
        return None, 0, 0.0
    names = referents(grouped)
    keys = sorted(grouped, key=lambda pair: (-len(grouped[pair]), pair[0]))[:limit]
    tokens, seconds = 0, 0.0

    stem, answer = choose(decider, state, "action", [k[0] for k in keys], think, budget)
    tokens += answer.get("generated_tokens", 0)
    seconds += answer["seconds"]
    arity = next(a for s, a in keys if s == stem)
    combos = grouped[(stem, arity)]
    if arity == 0:
        return stem, tokens, seconds

    firsts = options_for(stem, arity, combos, names, limit)
    if not firsts:
        return stem, tokens, seconds
    first, answer = choose(decider, state, f"object for '{stem}'", firsts, think, budget)
    tokens += answer.get("generated_tokens", 0)
    seconds += answer["seconds"]

    seconds_list = options_for(stem, arity, combos, names, limit, chosen=first)
    if arity == 1 or not seconds_list:
        return f"{stem} {first}".strip(), tokens, seconds

    second, answer = choose(decider, state, f"target for '{stem} {first}'",
                            seconds_list, think, budget)
    tokens += answer.get("generated_tokens", 0)
    seconds += answer["seconds"]
    # The connector is whatever the engine used between the two objects, taken
    # from a combination it offered rather than from the declared template.
    match = next((c for c in combos
                  if split_objects(c[len(stem):].strip(), names) == (first, second)), None)
    return (match or f"{stem} {first} {second}"), tokens, seconds


def episode(decider, env, task, variation, *, think, budget, limit, max_steps):
    env.load(task, variation)
    here, info = env.reset()
    goal = info["taskDesc"].strip()
    trail, tokens, seconds = [], 0, 0.0
    started = time.perf_counter()
    score, steps = 0.0, 0
    for steps in range(1, max_steps + 1):
        state = state_text(RULES, goal, here, trail)
        action, spent, served = act(decider, env, state, think, budget, limit)
        tokens += spent
        seconds += served
        if action is None:
            break
        here, _, done, info = env.step(action)
        score = float(info["score"])
        trail.append(f"- {action} -> {str(here).strip()[:110]}")
        if done or score >= 100:
            break
    return {"task": task, "variation": variation, "score": max(score, 0.0) / 100.0,
            "succeeded": score >= 100, "steps": steps, "generated_tokens": tokens,
            "served_seconds": round(seconds, 3),
            "wall_seconds": round(time.perf_counter() - started, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--decision-url", required=True)
    parser.add_argument("--task", default="measure-melting-point-known-substance")
    parser.add_argument("--split", default="test", choices=("train", "dev", "test"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--candidate-limit", type=int, default=52)
    parser.add_argument("--policies", default="fast,slow")
    parser.add_argument("--workers", type=int, default=1,
                        help="episodes in flight at once. The engine reads 54 GB of weights "
                             "for every decode step it takes whether that step serves one "
                             "sequence or many, so one request at a time is the expensive "
                             "way to use it. Each worker holds its own environment, and "
                             "ScienceWorld starts a JVM per environment")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from scienceworld import ScienceWorldEnv

    from inference import DecisionClient

    decider = DecisionClient(args.decision_url)
    print(f"decider {args.decision_url}: {decider.health()}", flush=True)

    env = ScienceWorldEnv("", envStepLimit=args.max_steps + 5)
    env.load(args.task, 0)
    variations = {"train": env.get_variations_train, "dev": env.get_variations_dev,
                  "test": env.get_variations_test}[args.split]()
    chosen = list(variations)[args.offset : args.offset + args.episodes]
    print(f"task {args.task} | {args.split} split | {len(chosen)} variations", flush=True)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"task": args.task, "split": args.split, "variations": chosen,
              "max_steps": args.max_steps, "budget": args.budget,
              "candidate_limit": args.candidate_limit, "workers": args.workers,
              "policies": {}}

    def run_one(variation, policy, own):
        """One episode on this worker's own environment."""
        return episode(decider, own, args.task, variation,
                       think=(policy == "slow"), budget=args.budget,
                       limit=args.candidate_limit, max_steps=args.max_steps)

    for policy in [p.strip() for p in args.policies.split(",") if p.strip()]:
        started = time.perf_counter()
        if args.workers <= 1:
            rows = []
            for index, variation in enumerate(chosen):
                row = run_one(variation, policy, env)
                rows.append(row)
                print(f"  {policy:5} {index + 1:3d}/{len(chosen)}  var {variation:4d}  "
                      f"score {row['score']:.2f}  steps {row['steps']:3d}  "
                      f"tok {row['generated_tokens']:5d}", flush=True)
        else:
            # A worker keeps one environment for the whole policy rather than one
            # per episode: a ScienceWorld environment is a JVM, and starting one
            # per episode would cost more than the episode.
            pool = [ScienceWorldEnv("", envStepLimit=args.max_steps + 5)
                    for _ in range(args.workers)]
            try:
                slices = [chosen[i::args.workers] for i in range(args.workers)]

                def drain(worker):
                    own = pool[worker]
                    return [run_one(variation, policy, own) for variation in slices[worker]]

                with ThreadPoolExecutor(max_workers=args.workers) as running:
                    rows = [row for part in running.map(drain, range(args.workers))
                            for row in part]
                print(f"  {policy:5} {len(rows)} episodes on {args.workers} workers", flush=True)
            finally:
                for own in pool:
                    own.close()
        report["policies"][policy] = {
            "episodes": len(rows),
            "success_rate": sum(r["succeeded"] for r in rows) / max(len(rows), 1),
            "mean_score": sum(r["score"] for r in rows) / max(len(rows), 1),
            "mean_steps": sum(r["steps"] for r in rows) / max(len(rows), 1),
            "tokens_per_episode": sum(r["generated_tokens"] for r in rows) / max(len(rows), 1),
            "seconds_per_episode": sum(r["wall_seconds"] for r in rows) / max(len(rows), 1),
            "wall_seconds": round(time.perf_counter() - started, 1),
            "rows": rows,
        }
        summary = report["policies"][policy]
        print(f"{policy:5} success {summary['success_rate']:.3f}  "
              f"score {summary['mean_score']:.3f}  "
              f"tokens/ep {summary['tokens_per_episode']:.0f}  "
              f"steps {summary['mean_steps']:.1f}  "
              f"wall {summary['wall_seconds']:.0f}s", flush=True)
        (output / "gate.json").write_text(json.dumps(report, indent=1) + "\n")
    env.close()


if __name__ == "__main__":
    main()
