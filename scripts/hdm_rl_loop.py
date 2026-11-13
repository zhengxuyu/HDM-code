"""Run the routing RL loop: collect, update the head, evaluate, repeat.

The head is the only thing trained. The backbone is frozen and served by SGLang,
which is also what collects, so everything here is one implementation: the gain
the head learns from and the gain it later routes on come from the same kernels.
"""
import argparse
import json
from pathlib import Path

from litjev._legacy.heads import DecisionHead
from slime_hdm import HeadRLConfig, LoopConfig, StepScorer, run
from slime_hdm.envs.alfworld import AlfworldEnvironment


def _code():
    """What code the rollouts were collected by, as `REVISION` names it.

    Recorded because what a row *means* has changed under this corpus before: the
    gain became a log ratio and the reward became a discount, both after 27608
    was collected, and nothing in that corpus says which side of either it is on.

    This was `git rev-parse HEAD` in the job's cwd, which is a tree `sync.sh`
    rsynced and so is not a checkout: it recorded `None` in every signature a
    cluster run ever wrote, and a relaunch from any other code would have matched
    and appended. `code_identity` reads the tree id instead, which also names the
    working changes -- 29636 ran a dirty tree.
    """
    from litjev.run.provenance import code_identity

    return code_identity()


def collection_signature(args, metadata, price_range, train_games, eval_games):
    """**What the episodes on disk were collected under.**

    Without this `checked_rows` returns unchecked and writes no signature row, so
    a reused `--output` resumes onto an earlier corpus and the run answers two
    questions at once. The price range is in it because 384fcba made it settable:
    a corpus collected under one range must not be extended by a run using
    another, and that difference is invisible in the rows themselves. The model is
    in it because the hidden states are its, and the code because the readout and
    the gain definition have both changed under this corpus's feet already.

    A function rather than a dict inside `main` so that a test can call it. The
    test that guarded these fields asserted on the launcher's *source text* and
    went red when `commit` was renamed to `code` -- for the rename, not for
    anything being wrong, which is the failure mode of a source assertion.
    """
    return {
        "venue": "alfworld",
        "split": "train",
        "offset": args.train_offset,
        "episodes": args.episodes,
        "model": args.model,
        # **Which layer the hidden states are.** Missing from the first version,
        # and the only field here whose absence corrupts rather than confuses:
        # `build_features` flattens the hidden state, so the width depends on how
        # many layers were read and not on which, and two single-layer corpora
        # pool with no error at all. `hidden` is the whole input to every refit
        # this corpus is kept for. exps found it.
        "feature_layers": list(metadata.feature_layers),
        # **Which games**, not how many from where. Two runs that both say
        # `valid_unseen` from offset 0 have already got different games here.
        "train_games": train_games,
        "eval_games": eval_games,
        "budget": args.budget,
        "max_steps": args.max_steps,
        "explore_prob": args.explore_prob,
        "lambda": args.lambda_,
        "price_range": list(price_range) if price_range else None,
        # What code wrote these rows, read from the snapshot's own record: see
        # `_code`, which replaced a `git rev-parse` that answered None here.
        "code": _code(),
        # Deliberately absent: `rounds`, and anything about the evaluation. A
        # field belongs here when two corpora differing in it hold rows that mean
        # different things. `checked_rows` compares the whole dict, so a field
        # that changes only what the process does *after* collecting renames a
        # fifteen-hour corpus and collects from zero -- and since each round
        # streams to its own file, recording `rounds` would discard round 0 on a
        # decision to add a round 2. exps's point, and the weaker of their two.
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--head", required=True, help="a trained head, for its metadata only")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=134)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.0,
                        help="the price the rollout acts at: a step escalates when "
                             "the head's gain exceeds it")
    parser.add_argument("--price-range", default=None,
                        help="low,high -- the interval the training price is drawn "
                             "from, per decision and redrawn every epoch. **Without "
                             "it the objective trains at the single price --lambda "
                             "gives**, where the discount factor is exactly 1 at 0 "
                             "and the policy term only cares which side of one "
                             "threshold each gain falls on, so the gain is free to "
                             "drift everywhere else. Drawing it instead makes the "
                             "objective an average over the range and the head has "
                             "to order gains across all of it, which is what lets "
                             "one head serve every price. The range must bracket "
                             "the prices a deployment will use")
    parser.add_argument("--explore-prob", type=float, default=0.5)
    parser.add_argument("--cost-weight", type=float, default=0.0)
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--train-games", default=None,
                        help="a file naming the training games, one per line. "
                             "**Prefer it over --train-offset for anything "
                             "compared across runs**: slicing is deterministic "
                             "only given alfworld's own game_files order, and "
                             "that order has already differed between two of our "
                             "runs -- the fast and slow endpoints of this venue "
                             "were measured on different games")
    parser.add_argument("--eval-games", default=None,
                        help="the same for the evaluation shard. Pin it when a "
                             "baseline is measured in another job, or the two are "
                             "aligned only by assumption")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from inference import SGLangScorer, SGLangSettings

    # The metadata says which layer and what width; the weights are not reused,
    # since each round trains a head from scratch on that round's returns.
    metadata = DecisionHead.load(args.head).metadata
    print(f"head reads layers {metadata.feature_layers}, input width {metadata.input_dim}",
          flush=True)
    scorer = StepScorer(SGLangScorer.load(SGLangSettings(
        model_id=args.model, feature_layers=metadata.feature_layers,
        batch_questions=False)))

    price_range = None
    if args.price_range:
        low, high = (float(value) for value in args.price_range.split(","))
        if not high > low:
            raise SystemExit(f"--price-range must be low,high with high > low, "
                             f"got {args.price_range!r}")
        price_range = (low, high)
        print(f"training price drawn per decision from {price_range}", flush=True)
    else:
        # Said out loud, because it is the degenerate case and it is the default:
        # a run that does not pass this trains at one price and cannot distinguish
        # the discounted reward from the linear one at lambda 0.
        print(f"no --price-range: training at the single price {args.lambda_}, "
              f"where the discount is inert at 0", flush=True)

    def _named(path):
        if not path:
            return None
        names = [line.strip() for line in Path(path).read_text().splitlines()
                 if line.strip()]
        if not names:
            raise SystemExit(f"{path} names no games")
        return names

    train_games, eval_games = _named(args.train_games), _named(args.eval_games)

    def make_environment(index, evaluation):
        """A fresh slice of train per round, and the whole unseen split to evaluate.

        A named list wins over the offset: see `--train-games`.
        """
        if evaluation:
            return AlfworldEnvironment.load(split="valid_unseen", offset=0,
                                            games=args.eval_episodes,
                                            game_files=eval_games)
        if train_games is not None:
            # One round's slice of the named list, so rounds stay disjoint.
            start = index * args.episodes
            shard = train_games[start : start + args.episodes]
            if len(shard) < args.episodes:
                raise SystemExit(
                    f"--train-games names {len(train_games)} games; round {index} "
                    f"needs {args.episodes} from {start}")
            return AlfworldEnvironment.load(split="train", game_files=shard)
        offset = args.train_offset + index * args.episodes
        return AlfworldEnvironment.load(split="train", offset=offset, games=args.episodes)

    signature = collection_signature(args, metadata, price_range,
                                     train_games, eval_games)
    print(f"collection signature: {json.dumps(signature)}", flush=True)

    head, rounds = run(
        metadata, make_environment, scorer,
        LoopConfig(rounds=args.rounds, episodes_per_round=args.episodes,
                   eval_episodes=args.eval_episodes, budget=args.budget,
                   max_steps=args.max_steps, lambda_=args.lambda_,
                   explore_prob=args.explore_prob,
                   head=HeadRLConfig(lambda_=args.lambda_,
                                     lambda_range=price_range,
                                     cost_weight=args.cost_weight,
                                     behaviour_prob=args.explore_prob),
                   output=args.output),
        seed=0, signature=signature)
    Path(args.output, "summary.json").write_text(json.dumps(rounds, indent=1) + "\n")
    print(json.dumps(rounds[-1], indent=1))


if __name__ == "__main__":
    main()
