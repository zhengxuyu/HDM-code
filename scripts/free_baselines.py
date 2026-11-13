"""What can route without a learned block at all?

The paper's floor is a head trained on the fast distribution's statistics, which
is still a trained block. A reader is entitled to a harder floor: the model's own
numbers, thresholded, with nothing fitted. Each of these is produced by the fast
pass and costs nothing beyond it, so any of them could be shipped tomorrow.

All of them rank a question by how unsure the fast mode looks. That is the
comparison the paper wants, because looking unsure and being worth thinking about
are different: a question both modes get wrong looks maximally unsure and is the
last one worth escalating. Where these baselines lose, they lose on that.

    uv run python scripts/free_baselines.py results/records-4b.npz
"""
import argparse

import numpy as np

from hde._legacy.heads import HeadMetadata, auroc, outcome_labels
from hde._legacy.routing import escalation_gain
from hde.data.collect import open_records
from hde.run.streaming import write_json_atomic
from hde.train.training import features_for, holdout_mask, train_head

# The order distribution_stats writes them in.
STATS = {"max probability": 0, "normalised entropy": 1, "top-two margin": 2,
         "concentration": 3, "log K": 4}
# Higher should mean "escalate": confidence-like statistics are negated.
SIGN = {"max probability": -1, "normalised entropy": +1, "top-two margin": -1,
        "concentration": -1, "log K": +1}


def escalated_accuracy(score, fast, slow, rate):
    """Accuracy when the top `rate` by score escalate -- and whether that set is determined.

    `np.argsort` is not stable by default, and inside a block of equal scores its
    order is the implementation's. So when the cut falls inside such a block the
    number depends on the machine, and `tab:free`'s `log K` row is exactly that:
    eight distinct values over the 346 held-out questions, 291 of them sharing the
    top one, and the 20% cut at rank 69 landing inside that block. Two machines
    printed two different rows from the same file. `block_utils.auroc` has handled
    ties with mid-ranks since it was written, which is why that row's AUROC
    reproduces while its acc@ columns do not.

    So the order is made deterministic, and the tie is reported rather than hidden:
    `low` and `high` are the best and worst accuracy any ordering of the tied
    questions could produce. Escalating a question swaps `fast` for `slow`, so each
    one moves the mean by `slow - fast` in {-1, 0, +1}; the bracket takes the
    `inside` tied questions with the largest and smallest of those. **A row whose
    bracket is wide is not a quantity**, however stable the sort makes it look.
    """
    n = len(fast)
    k = int(rate * n)
    order = np.argsort(-score, kind="stable")
    pick = np.zeros(n, dtype=bool)
    pick[order[:k]] = True
    accuracy = float(np.where(pick, slow, fast).mean())
    rescued = int((~fast & slow)[pick].sum())
    wasted = int((~fast & ~slow)[pick].sum())
    if k == 0 or k == n:
        return accuracy, rescued, wasted, accuracy, accuracy, 0
    # The tie the cut sits in, if any: everything at the threshold score.
    threshold = score[order[k - 1]]
    tied = score == threshold
    inside = int(tied[pick].sum())
    straddling = int(tied.sum()) - inside
    if straddling == 0:
        return accuracy, rescued, wasted, accuracy, accuracy, 0
    # Outside the tie the set is forced: those questions contribute `slow` where
    # escalated and `fast` otherwise, whatever happens inside. Summed over the
    # non-tied questions only -- summing over all of them and then adding the
    # tied ones' `fast` counts the tie twice, which is how the first version of
    # this returned accuracies above 1.
    others = ~tied
    fixed = float(np.where(pick & others, slow, fast)[others].sum())
    # Inside the tie every question contributes `fast` unless it is one of the
    # `inside` that escalate, and escalating moves it by `slow - fast`.
    delta = np.sort((slow.astype(np.int8) - fast.astype(np.int8))[tied])
    floor = fixed + float(fast[tied].sum())
    low = (floor + float(delta[:inside].sum())) / n
    high = (floor + float(delta[len(delta) - inside:].sum())) / n
    return accuracy, rescued, wasted, low, high, int(tied.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records")
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--rates", default="0.2,0.3,0.4,0.5")
    parser.add_argument("--seeds", type=int, default=20,
                        help="the head is refit per seed; the split is fixed, as in the paper")
    parser.add_argument("--output")
    arguments = parser.parse_args()

    records, metadata = open_records(arguments.records)
    position = list(metadata["feature_layers"]).index(arguments.layer)
    fast = np.asarray(records["fast_correct"], dtype=bool)
    slow = np.asarray(records["slow_correct"], dtype=bool)
    labels = outcome_labels(fast, slow)
    # The split is held fixed across seeds, as in the paper: what varies is the
    # head's initialisation and its feature fit, not which questions it is tested on.
    test = np.asarray(holdout_mask(records["category"], 0.25, 0)[0], dtype=bool)
    stats = np.asarray(records["stats"], dtype=np.float64)
    features = features_for(records, (position,))
    rates = [float(value) for value in arguments.rates.split(",")]
    ft, st = fast[test], slow[test]
    helps = ~ft & st

    head_metadata = HeadMetadata(
        model_id=metadata["model_id"], revision=metadata["revision"],
        hidden_size=int(metadata["hidden_size"]), feature_layers=(arguments.layer,),
        hidden_width=64, pca_dim=64)

    # Parameter-free routers do not vary with a seed; only the head is refit.
    fixed = {name: SIGN[name] * stats[test, index] for name, index in STATS.items()}
    trials = {name: [] for name in fixed}
    trials["random"] = []
    trials["value of thinking (ours)"] = []
    for seed in range(arguments.seeds):
        head, _ = train_head(head_metadata, features[~test], labels[~test],
                             epochs=60, seed=seed)
        gains = np.array([escalation_gain(row) for row in head.predict(features[test])])
        trials["value of thinking (ours)"].append(
            (float(auroc(gains, helps)),
             [escalated_accuracy(gains, ft, st, rate) for rate in rates]))
        chance = np.random.default_rng(seed).random(int(test.sum()))
        trials["random"].append(
            (float(auroc(chance, helps)),
             [escalated_accuracy(chance, ft, st, rate) for rate in rates]))
        for name, score in fixed.items():
            if seed == 0:
                trials[name].append(
                    (float(auroc(score, helps)),
                     [escalated_accuracy(score, ft, st, rate) for rate in rates]))
        print(f"  seed {seed}", end="\r", flush=True)

    print(f"{int(test.sum())} held-out questions: fast {ft.mean():.3f}, slow {st.mean():.3f}, "
          f"thinking helps {helps.mean():.3f}, both wrong {(~ft & ~st).mean():.3f}")
    print(f"{arguments.seeds} seeds for anything refit; the statistics do not vary\n")

    def summarise(values):
        array = np.asarray(values, dtype=float)
        return array.mean(), (array.std(ddof=1) if len(array) > 1 else 0.0)

    print(f"{'router':28s} AUROC            " +
          "  ".join(f"acc@{rate:.0%}" for rate in rates))
    rows = {}
    for name, runs in trials.items():
        aurocs = [run[0] for run in runs]
        mean, spread = summarise(aurocs)
        cells = []
        detail = []
        for index, rate in enumerate(rates):
            accuracy, _ = summarise([run[1][index][0] for run in runs])
            low, _ = summarise([run[1][index][3] for run in runs])
            high, _ = summarise([run[1][index][4] for run in runs])
            tied = float(np.mean([run[1][index][5] for run in runs]))
            # A cell the tie leaves undetermined is marked where it is printed,
            # not only in the JSON. The printed `log K` row of tab:free is one,
            # and it read as a measurement for as long as it has been in the PDF.
            cells.append(f"{accuracy:.3f}{'*' if high - low > 0.005 else ' '}")
            detail.append({"rate": rate, "accuracy": accuracy,
                           "rescued": float(np.mean([run[1][index][1] for run in runs])),
                           "wasted": float(np.mean([run[1][index][2] for run in runs])),
                           # What any other ordering of the tied questions could
                           # have given, and how many were tied at the cut.
                           "tie_low": low, "tie_high": high, "tied_at_cut": tied})
        rows[name] = {"auroc": mean, "auroc_sd": spread, "seeds": len(runs),
                      "by_rate": detail}
        tail = f"{mean:.4f}" if len(runs) == 1 else f"{mean:.4f} +- {spread:.4f}"
        print(f"{name:28s} {tail:16s} {'  '.join(cells)}")
    print("* the cut falls inside a block of equal scores, so the escalation set is "
          "not determined by the score and this cell depends on sort order")
    for name, row in rows.items():
        worst = max(row["by_rate"], key=lambda cell: cell["tie_high"] - cell["tie_low"])
        if worst["tie_high"] - worst["tie_low"] > 0.005:
            print(f"  {name:26s} at {worst['rate']:.0%}: "
                  f"{worst['tie_low']:.3f}..{worst['tie_high']:.3f} over "
                  f"{worst['tied_at_cut']:.0f} tied questions")

    print("\nrescued / wasted among the escalated, at "
          f"{rates[0]:.0%} (mean over seeds):")
    for name in trials:
        first = rows[name]["by_rate"][0]
        print(f"  {name:28s} {first['rescued']:5.1f} rescued, {first['wasted']:5.1f} wasted")

    if arguments.output:
        write_json_atomic(arguments.output,
                          {"records": arguments.records, "layer": arguments.layer,
                           "held_out": int(test.sum()), "seeds": arguments.seeds,
                           "routers": rows})


if __name__ == "__main__":
    main()
