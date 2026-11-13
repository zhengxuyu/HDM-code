"""ALFWorld rollouts that record what a router would need, and can be routed.

The probe measured the endpoints and threw the evidence away. This keeps it.

At every step the fast mode runs first: one forward pass over the prompt, the
action read off the admissible commands, and the hidden state at that readout
position kept. Only then is the escalation decided, which is what makes the
decision free — the state it reads was computed on the way to the fast answer
whether or not the step escalates.

Policies:
  fast    never think
  slow    always think
  random  think with probability --escalate-prob, for collecting training data
          that covers both actions at every state
  head    think when the decision head's predicted gain exceeds --lambda

The features are built by hde._legacy.heads.build_features, the same function the
MMLU-Pro heads were trained through, so a head trained there can be pointed at
this environment without conversion.

**Two backends, and they are not interchangeable.** `--backend transformers` is
the step loop below: its own prompt, built by `build_prompt`, and the candidate
distribution read from `model(...)` logits. `--backend sglang` does not use that
loop at all -- it runs `slime_hdm.rollout.rollout_episodes`, which is the same
step function `hdm_rl_loop.py` collects through, over `AlfworldEnvironment` and
`StepScorer`.

That is not a choice of engine. A head reads the hidden state at a readout
position *in a particular prompt*, and the two paths render different prompts, so
the loop's head is off its own training distribution under the transformers loop
however well the engine behaves. 29636 collected 150 episodes through the loop
and then asked this script for the arms, which would have crossed both the prompt
and the backend; it failed first for an unrelated reason (`hdm-env`'s transformers
has no `qwen3_5`), so the crossing never produced a number. Routing through the
loop's own step function is what makes "the same prompt" structural rather than
asserted, in the same way `--game-list` made "the same games" structural.
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from hde._legacy.heads import DecisionHead, build_features
from hde._legacy.routing import escalation_gain
from hde.run.streaming import write_json_atomic

CODES = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
BOUNDARY = "\nAction:"
# One definition, because `arm_policy` maps a name to a decision rule and
# argparse accepts a set of names: a name in one and not the other is a routed
# run reporting another policy's label.
POLICIES = ("fast", "slow", "random", "confidence", "head")
SPLITS = {"train": "train", "valid_seen": "eval_in_distribution",
          "valid_unseen": "eval_out_of_distribution"}


def expand(node, data):
    if isinstance(node, dict):
        return {k: expand(v, data) for k, v in node.items()}
    if isinstance(node, list):
        return [expand(v, data) for v in node]
    return node.replace("$ALFWORLD_DATA", data) if isinstance(node, str) else node


def single_token_codes(tokenizer, wanted):
    """Codes that are one token after the boundary; the readout reads one position."""
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
        lines += ["Recent steps:", *history[-6:], ""]
    lines += ["Available actions:",
              *(f"{code} {command}" for (code, _), command in zip(codes, commands)), ""]
    lines.append("Reason step by step, then give the code." if think
                 else "Reply with the code of the best action.")
    return "\n".join(lines)


def read_codes(model, tokenizer, text, code_ids, layer):
    """One forward pass: the candidate distribution and the hidden state beside it."""
    ids = tokenizer(text + BOUNDARY, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model(**ids, output_hidden_states=layer is not None)
    logits = out.logits[0, -1]
    probabilities = torch.softmax(logits[code_ids].float(), dim=-1).cpu().numpy()
    hidden = None
    if layer is not None:
        hidden = out.hidden_states[layer][0, -1].float().cpu().numpy().astype(np.float16)
    return probabilities, hidden


def arm_policy(policy, escalate_prob):
    """The decision rule each arm is, spelled as `rollout_episodes` takes it.

    `None` is what tells the rollout to let the head decide. The endpoints are
    probabilities of 0 and 1 rather than special cases, so every arm runs the one
    step function and they differ in the decision rule and in nothing else -- the
    fast and slow endpoints of this venue were once measured by separate code and
    turned out to have played different games.
    """
    return {"fast": 0.0, "slow": 1.0, "random": escalate_prob}.get(policy)


def confidence_rule(policy, cut):
    """The confidence arm's threshold, and `None` for every other arm.

    A threshold on the fast mode's top probability cannot be written as a
    coin probability, so it cannot join `decision_rule`'s table; and the
    `None` that table returns already means "let the head decide", so the
    two have to be separate arguments all the way down.
    """
    return cut if policy == "confidence" else None


def play_through_the_loop(args):
    """The arms, run by the step function the head's training data was collected with.

    `rollout_episodes` is what `hdm_rl_loop.py` collects through, so an arm here is
    that same function under a fixed decision rule: `escalate_prob` 0 never thinks,
    1 always thinks, and a head with no `escalate_prob` is the routed arm. The
    prompt, the candidate readout and the engine are the loop's, not reimplemented
    here, which is the whole reason this path exists -- a second readout would
    agree with the loop's by argument, and this one agrees by construction.

    **Every arm asks for the hidden state, including the ones that do not read
    it.** The readout is most of the fast pass's cost on this backbone, so an arm
    that skips it is cheaper for a reason that has nothing to do with its policy,
    and the arms' seconds would not be comparable -- the confound 29655 exists to
    isolate on JevBench. A fast baseline that pays no readout is a different
    measurement and has to be asked for deliberately.
    """
    from inference.sglang_backend import SGLangScorer, SGLangSettings
    from slime_hdm import StepScorer
    from slime_hdm.envs.alfworld.environment import AlfworldEnvironment
    from slime_hdm.rollout import rollout_episodes

    data = os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))
    os.environ["ALFWORLD_DATA"] = data

    head, layers = None, (-1,)
    if args.decision_head:
        # On CPU, where `load` leaves it: `predict` moves the features to the
        # head's own device, and the engine owns the GPU. The transformers path
        # puts it beside the backbone, which has no meaning when there is no
        # transformers model.
        head = DecisionHead.load(args.decision_head)
        layers = tuple(head.metadata.feature_layers)
        print(f"head reads layers {list(layers)}, lambda {args.lambda_}", flush=True)
    # Before the backbone is loaded: `_check_layers` refuses any layer but the
    # last, which is a second's refusal instead of an hour's.
    scorer = StepScorer(SGLangScorer.load(SGLangSettings(
        model_id=args.model, feature_layers=layers, batch_questions=False)))

    if args.game_list:
        names = [line.strip() for line in Path(args.game_list).read_text().splitlines()
                 if line.strip()]
        if not names:
            raise SystemExit(f"{args.game_list} names no games")
        environment = AlfworldEnvironment.load(split=args.split, game_files=names,
                                               data=data)
        print(f"playing {environment.episodes} named games from {args.game_list}",
              flush=True)
    else:
        environment = AlfworldEnvironment.load(split=args.split, offset=args.offset,
                                               games=args.games, data=data)
    played = environment.game_files

    escalate_prob = arm_policy(args.policy, args.escalate_prob)
    rows, started = [], time.time()

    def report(result):
        """One row per episode, while the environment still knows its task text."""
        rows.append({
            "game": result.episode,
            # Read here rather than after the loop: the next `reset` replaces it,
            # and the task text is what revealed that two runs of this venue had
            # played different games.
            "task": environment.task,
            "success": bool(result.reward > 0),
            "steps": len(result.steps),
            "generated": sum(step.generated_tokens for step in result.steps),
            "escalations": sum(1 for step in result.steps if step.escalated),
            # The rollout's own word for why it stopped, which the transformers
            # loop has no equivalent of.
            "ended": result.ended,
            "seconds": result.seconds,
        })
        done_n = sum(row["success"] for row in rows)
        print(f"[{len(rows)}/{environment.episodes}] success={rows[-1]['success']} "
              f"steps={rows[-1]['steps']} esc={rows[-1]['escalations']} "
              f"gen={rows[-1]['generated']} running={done_n}/{len(rows)}", flush=True)

    rollout_episodes(
        environment, scorer, head,
        lambda_=args.lambda_, episodes=environment.episodes, budget=args.budget,
        escalate_prob=escalate_prob,
        confidence_cut=confidence_rule(args.policy, getattr(args, "confidence", 0.8)),
        rng=np.random.default_rng(args.seed),
        max_steps=args.max_steps, question_type="choice", on_episode=report)

    rate = sum(row["success"] for row in rows) / len(rows)
    summary = {
        "policy": args.policy, "split": args.split, "lambda": args.lambda_,
        "escalate_prob": args.escalate_prob if args.policy == "random" else None,
        # getattr, because a caller that builds its own Namespace -- the arms test
        # does -- sets the fields its arm needs and not the others.
        "confidence_cut": (getattr(args, "confidence", None)
                           if args.policy == "confidence" else None),
        "decision_head": args.decision_head, "model": args.model,
        # **Which path played these episodes**, because it decides what they may
        # be compared with: the two render different prompts, so the same policy
        # on the same games is a different measurement under each.
        "backend": "sglang",
        "layer": list(layers),
        # Said in the record, since it is what makes the arms' seconds comparable
        # with one another and not with a headless run's.
        "hidden_states_requested": True,
        "games": len(rows), "offset": args.offset, "max_steps": args.max_steps,
        "game_files": played,
        "budget": args.budget, "seed": args.seed, "success_rate": rate,
        "mean_steps": sum(row["steps"] for row in rows) / len(rows),
        "generated_per_episode": sum(row["generated"] for row in rows) / len(rows),
        "escalations_per_episode": sum(row["escalations"] for row in rows) / len(rows),
        "seconds": time.time() - started, "episodes": rows,
    }
    write_json_atomic(args.output, summary)
    print(f"\n{args.policy}: success {rate:.3f} over {len(rows)} games, "
          f"{summary['mean_steps']:.1f} steps, "
          f"{summary['escalations_per_episode']:.1f} escalations, "
          f"{summary['generated_per_episode']:.0f} tokens/episode")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", choices=sorted(SPLITS), default="valid_unseen")
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--confidence", type=float, default=0.8,
                        help="escalate when the fast mode's top probability falls "
                             "below this; used by --policy confidence only")
    parser.add_argument(
        "--backend", choices=("transformers", "sglang"), default=None,
        help="which path plays the episodes. **Required, with no default**: a "
             "default is how an arm takes a path nobody chose for it. 29700's "
             "fast arm would have run the transformers loop while its head arm "
             "ran sglang, so the two arms would have differed by backend as well "
             "as by policy, and only the head arm refuses a missing flag. "
             "It has to be the path the head was "
             "fitted through**, and on this venue that is sglang: heads here come "
             "from `hdm_rl_loop.py`, which scores every step with `StepScorer` over "
             "an `SGLangScorer`. transformers runs this file's own step loop, with "
             "its own prompt, and is kept because it is what measured the endpoints "
             "in 27309 and 27343. The two are not comparable and no ALFWorld figure "
             "may mix them")
    parser.add_argument("--escalate-prob", type=float, default=0.5)
    parser.add_argument("--decision-head")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.0)
    parser.add_argument("--layer", type=int, help="hidden_states index to record; omit to skip")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--list-games", action="store_true",
                        help="print the split's games, one per line, and exit. "
                             "This is how the committed lists in "
                             "experiments/alfworld/games/ are produced, so they "
                             "are regenerable from a documented command rather "
                             "than from an ad-hoc call someone has to remember")
    parser.add_argument("--game-list", default=None,
                        help="a file naming the games, one per line. **Use it for "
                             "anything compared across runs.** Without it the "
                             "shard is whatever alfworld registers for a count of "
                             "`offset + games`, and that set depends on the count: "
                             "this venue's fast endpoint asked for 134 and its slow "
                             "endpoint asked for 45 at a time, and they got "
                             "different games -- 45 of 134 task texts appear in only "
                             "one of them. Every paired figure built on those two "
                             "runs paired different populations")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record", help="npz for the per-step features")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.policy == "head" and not args.decision_head:
        parser.error("--policy head needs --decision-head")
    if args.record and args.layer is None:
        parser.error("--record needs --layer, or there is nothing to record")
    if args.backend is None and not args.list_games:
        # Not defaulted, and not required by argparse either, because
        # `--list-games` loads no model and choosing a path for it is noise.
        parser.error("--backend is required: pass sglang for anything quotable on "
                     "this venue, or transformers to reproduce 27309 and 27343. "
                     "Every arm of one comparison must pass the same one")
    if args.backend == "sglang":
        if args.record:
            # The loop is this venue's recorder and writes a corpus with the probe
            # fields, the rewards and the signature that says what it was collected
            # under. A second recorder here would write a thinner row under the
            # same name.
            parser.error("--record is the transformers loop's own recorder; collect "
                         "through scripts/hdm_rl_loop.py --output, which writes the "
                         "corpus this venue trains on")
        if args.list_games:
            # It needs no engine and the transformers path already does it through
            # the same loader, so there is one implementation rather than two.
            parser.error("--list-games does not need a backend; run it without "
                         "--backend sglang")
        play_through_the_loop(args)
        return
    if args.policy == "head":
        # Refused rather than warned about. Every ALFWorld head so far comes from
        # `hdm_rl_loop.py`, which scores through `StepScorer` over an
        # `SGLangScorer`, so its features are that prompt's readout under that
        # engine -- and this loop renders a different prompt. The run would
        # complete and report a routed success rate that is the head reading
        # states it was never fitted on, which is the shape of error that costs a
        # conclusion rather than a job. If a head ever is fitted from this file's
        # own `--record` output, this is the line to change, and the head's own
        # metadata does not yet say which path produced it.
        parser.error("--policy head needs --backend sglang: ALFWorld heads are "
                     "fitted through the loop's StepScorer over SGLang, and this "
                     "loop renders a different prompt, so the head would be served "
                     "off its training distribution")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    data = os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))
    os.environ["ALFWORLD_DATA"] = data
    with open(os.path.join(os.path.dirname(__file__), "..", "configs",
                           "alfworld_base.yaml")) as handle:
        config = expand(yaml.safe_load(handle), data)
    key = "num_train_games" if args.split == "train" else "num_eval_games"
    # **Only when no list is named.** Setting the count makes the registered set
    # depend on how many games were asked for, which is how this venue's two
    # endpoints came to be measured on different games. With a list, the shared
    # loader registers exactly those.
    config["dataset"][key] = args.offset + args.games

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    codes = single_token_codes(tokenizer, len(CODES))
    print(f"single-token codes: {len(codes)} of {len(CODES)}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0").eval()
    code_ids = torch.tensor([i for _, i in codes], device=model.device)

    head = None
    if args.decision_head:
        head = DecisionHead.load(args.decision_head)
        head.to(model.get_input_embeddings().weight.device)
        layers = head.metadata.feature_layers
        if args.layer is not None and (len(layers) != 1 or layers[0] != args.layer):
            parser.error(f"head reads layers {layers}; --layer {args.layer} disagrees")
        args.layer = layers[0]
        print(f"head reads layer {args.layer}, lambda {args.lambda_}", flush=True)

    import alfworld.agents.environment as environment

    if args.list_games:
        # Through the shared loader, so the list is exactly what a later
        # `--game-list` run will register. Enumerating it any other way would
        # reintroduce the two-mechanism problem these lists exist to close.
        from slime_hdm.envs.alfworld.environment import AlfworldEnvironment

        every = AlfworldEnvironment.load(split=args.split, offset=0,
                                         games=1, data=data)
        print("\n".join(every.every_game()), flush=True)
        return

    played = None
    if args.game_list:
        # One selection mechanism shared with the loop, so "the same games" is
        # structural rather than an alignment of two different mechanisms. The
        # loop slices `env.game_files`; this script set a count and burned resets
        # to reach an offset, and the two need not agree on anything.
        from slime_hdm.envs.alfworld.environment import AlfworldEnvironment

        names = [line.strip() for line in Path(args.game_list).read_text().splitlines()
                 if line.strip()]
        if not names:
            parser.error(f"{args.game_list} names no games")
        shard = AlfworldEnvironment.load(split=args.split, game_files=names,
                                         data=data)
        played = shard.game_files
        env = shard.engine
        first, last = 0, len(played)
        print(f"playing {len(played)} named games from {args.game_list}", flush=True)
    else:
        env = environment.get_environment("AlfredTWEnv")(
            config, train_eval=SPLITS[args.split])
        env = env.init_env(batch_size=1)
        first, last = args.offset, args.offset + args.games
        for _ in range(args.offset):
            env.reset()

    rng = np.random.default_rng(args.seed)
    episodes, records, started = [], [], time.time()
    for game in range(first, last):
        obs, info = env.reset()
        task = re.split(r"Your task is to:", obs[0])[-1].strip().split("\n")[0]
        history, success, steps, generated, escalations = [], False, 0, 0, 0
        for step in range(args.max_steps):
            commands = info["admissible_commands"][0][: len(codes)]
            usable = codes[: len(commands)]
            prompt = build_prompt(task, history, commands, usable, think=False)
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            probabilities, hidden = read_codes(
                model, tokenizer, text, code_ids[: len(commands)], args.layer)

            gain = None
            if args.policy == "fast":
                escalate = False
            elif args.policy == "slow":
                escalate = True
            elif args.policy == "random":
                escalate = bool(rng.random() < args.escalate_prob)
            else:
                features = build_features(hidden, probabilities, "choice")
                outcome = head.predict(features[None, :])[0]
                gain = escalation_gain(outcome)
                escalate = gain > args.lambda_

            if escalate:
                slow_prompt = build_prompt(task, history, commands, usable, think=True)
                slow_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": slow_prompt}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=True)
                ids = tokenizer(slow_text, return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    out = model.generate(**ids, max_new_tokens=args.budget, do_sample=False,
                                         pad_token_id=tokenizer.eos_token_id)
                generated += int(out.shape[1] - ids["input_ids"].shape[1])
                escalations += 1
                decided, _ = read_codes(model, tokenizer,
                                        tokenizer.decode(out[0], skip_special_tokens=True),
                                        code_ids[: len(commands)], None)
            else:
                decided = probabilities

            if args.record:
                records.append({
                    "game": game, "step": step, "hidden": hidden,
                    "probabilities": np.pad(probabilities, (0, len(CODES) - len(probabilities))),
                    "candidates": len(commands), "escalated": escalate,
                })
            action = commands[int(np.argmax(decided))]
            obs, score, done, info = env.step([action])
            history.append(f"- {action} -> {obs[0].strip()[:120]}")
            steps += 1
            if done[0]:
                success = bool(score[0] > 0)
                break

        episodes.append({"game": game, "task": task, "success": success, "steps": steps,
                         "generated": generated, "escalations": escalations})
        done_n = sum(e["success"] for e in episodes)
        print(f"[{game + 1}/{args.offset + args.games}] success={success} steps={steps} "
              f"esc={escalations} gen={generated} running={done_n}/{len(episodes)}", flush=True)

    rate = sum(e["success"] for e in episodes) / len(episodes)
    summary = {
        "policy": args.policy, "split": args.split, "lambda": args.lambda_,
        "escalate_prob": args.escalate_prob if args.policy == "random" else None,
        "decision_head": args.decision_head, "model": args.model, "layer": args.layer,
        "games": len(episodes), "offset": args.offset, "max_steps": args.max_steps,
        # **Which games**, so two runs can be checked against each other instead
        # of assumed aligned. None when the shard was selected by count.
        "game_files": played,
        "budget": args.budget, "seed": args.seed, "success_rate": rate,
        "mean_steps": sum(e["steps"] for e in episodes) / len(episodes),
        "generated_per_episode": sum(e["generated"] for e in episodes) / len(episodes),
        "escalations_per_episode": sum(e["escalations"] for e in episodes) / len(episodes),
        "seconds": time.time() - started, "episodes": episodes,
    }
    write_json_atomic(args.output, summary)
    if args.record:
        success_of = {e["game"]: e["success"] for e in episodes}
        np.savez_compressed(
            args.record,
            hidden=np.stack([r["hidden"] for r in records]),
            probabilities=np.stack([r["probabilities"] for r in records]).astype(np.float32),
            candidates=np.array([r["candidates"] for r in records], dtype=np.int32),
            escalated=np.array([r["escalated"] for r in records], dtype=bool),
            game=np.array([r["game"] for r in records], dtype=np.int32),
            step=np.array([r["step"] for r in records], dtype=np.int32),
            episode_success=np.array([success_of[r["game"]] for r in records], dtype=bool),
            metadata=json.dumps({k: v for k, v in summary.items() if k != "episodes"}),
        )
        print(f"recorded {len(records)} steps over {len(episodes)} episodes", flush=True)
    print(f"\n{args.policy}: success {rate:.3f} over {len(episodes)} games, "
          f"{summary['mean_steps']:.1f} steps, {summary['escalations_per_episode']:.1f} escalations, "
          f"{summary['generated_per_episode']:.0f} tokens/episode")


if __name__ == "__main__":
    main()
