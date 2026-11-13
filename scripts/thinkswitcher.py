"""ThinkSwitcher as its paper describes it, on our records.

Liang et al., arXiv 2505.14183. Their switch regresses the pass rate of each
mode from the query embedding and escalates when the predicted difference clears
a threshold, so sweeping that threshold traces an accuracy-against-cost curve.
This fits their switch on our collections and sweeps it, which puts a method
that chooses *how long to reason* on the same axes as one that chooses *whether
to reason at all*.

Their appendix gives search ranges, not chosen values -- learning rate log-uniform
over [1e-5, 1e-2], batch size from {16, 32, 64, 128}, dropout uniform over [0, 0.5],
AdamW, at most 50 epochs with early stopping on a validation set and the best epoch
kept, all searched with Ray Tune -- and never says which layer or position the query
embedding comes from. So an exact replication is not available from the paper. What
is available is their search, run over their ranges, which `fit` below does: fitting
one guessed configuration and taking the last epoch would report a baseline we had
weakened ourselves.

Two further deviations, both stated rather than quietly taken.

**k=1, not k=8.** They estimate each mode's pass rate from eight samples a
question. Eight long chains over MMLU-Pro's training split is about eighty-eight
GPU-hours on the 27B here, against four for one greedy pass, and their own
ablation sweeps k from 1 to 16. So the targets are a single greedy outcome,
which is a pass rate in {0, 1}, and the regression is fitted to that.

**Their switch over our two modes, not their two modes.** On coded multiple
choice there is no length axis to trade along: the answer is one token, so
"reason first" and "answer directly" are the only points that exist and their
short mode coincides with our fast mode by construction. Collecting a separate
short arm measured that coincidence and worse -- probes 31196 and 31197 found it
writing two tokens an item, agreeing with the thinking arm on all 30 of the 27B's
JevBench questions, and scoring 13 points *below* the fast mode on the 4B's
MMLU-Pro, because `_generate_then_read` conditions the candidate read on the
model's own hasty answer. So what is compared here is the routing rule with the
modes held fixed: their 5-layer MLP on the query embedding against our head on
the hidden state, both choosing between the same fast and slow modes, both fitted
on the same records.

    uv run python scripts/thinkswitcher.py \
        --short results/records-...-short.npz --long results/records-...npz \
        --eval-short ... --eval-long ... --output results/thinkswitcher.json
"""
import argparse
import json

import numpy as np
import torch
from torch import nn

from hde.data.collect import open_records
from hde.train.training import features_for, holdout_mask


class Switcher(nn.Module):
    """Their 5-layer MLP: 1024, 768, 512, 256, 2, with ReLU, batch norm, dropout."""

    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        widths = (1024, 768, 512, 256)
        layers, previous = [], input_dim
        for width in widths:
            layers += [nn.Linear(previous, width), nn.BatchNorm1d(width), nn.ReLU(),
                       nn.Dropout(dropout)]
            previous = width
        layers.append(nn.Linear(previous, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def switch_loss(predicted, target, margin_weight=1.0):
    """Their L_MSE + lambda * L_margin, with lambda = 1 as their ablation prefers.

    The margin term is the squared error on the *difference* between the modes,
    which is the quantity the decision rule thresholds; without it the two heads
    can both be close and their difference still wrong in sign.
    """
    mse = nn.functional.mse_loss(predicted, target)
    gap = nn.functional.mse_loss(predicted[:, 1] - predicted[:, 0],
                                 target[:, 1] - target[:, 0])
    return mse + margin_weight * gap


def one_file(path, layer=-1):
    """Both arms from one collection: its fast columns and its slow columns.

    The modes are held fixed and only the rule varies, so the switch is fitted on
    the same records our head is. `short_tokens` is zero because the fast mode
    generates nothing -- that is the axis the two rules are compared on.
    """
    rec, meta = open_records(path)
    if meta.get("slow_mode", "think") != "think":
        raise SystemExit(f"{path} is not a thinking-mode collection")
    position = list(meta["feature_layers"]).index(layer)
    fast = np.asarray(rec["fast_correct"], bool)
    slow = np.asarray(rec["slow_correct"], bool)
    return {
        "features": features_for(rec, (position,)),
        "short_correct": fast,
        "long_correct": slow,
        "short_tokens": np.zeros(len(fast)),
        "long_tokens": np.asarray(rec["thinking_tokens"], float),
        "category": np.asarray(rec["category"]),
        "model": meta["model_id"].split("/")[-1],
        "n": len(fast),
    }


def join(short_path, long_path, layer=-1):
    """Short and long outcomes for the questions both passes scored.

    Joined on `question_id`, because the two are separate collections and a
    positional join would silently pair different questions if either dropped
    one. The hidden state is the fast pass's and identical in both, so it is
    taken from the long file.
    """
    short, short_meta = open_records(short_path)
    long_, long_meta = open_records(long_path)
    if short_meta.get("slow_mode") != "short":
        raise SystemExit(f"{short_path} was collected with slow_mode="
                         f"{short_meta.get('slow_mode', 'think')!r}, not 'short'")
    if long_meta.get("slow_mode", "think") != "think":
        raise SystemExit(f"{long_path} is not a thinking-mode collection")
    if short_meta["model_id"] != long_meta["model_id"]:
        raise SystemExit("the two collections are of different backbones")

    index = {qid: i for i, qid in enumerate(short["question_id"])}
    rows = [(j, index[qid]) for j, qid in enumerate(long_["question_id"]) if qid in index]
    if not rows:
        raise SystemExit("the two collections share no question")
    keep_long = np.array([j for j, _ in rows])
    keep_short = np.array([i for _, i in rows])
    position = list(long_meta["feature_layers"]).index(layer)
    return {
        "features": features_for(long_, (position,))[keep_long],
        "short_correct": np.asarray(short["slow_correct"], bool)[keep_short],
        "long_correct": np.asarray(long_["slow_correct"], bool)[keep_long],
        "short_tokens": np.asarray(short["thinking_tokens"], float)[keep_short],
        "long_tokens": np.asarray(long_["thinking_tokens"], float)[keep_long],
        "category": np.asarray(long_["category"])[keep_long],
        "model": long_meta["model_id"].split("/")[-1],
        "n": len(rows),
    }


def _train_once(x, y, split, lr, batch, dropout, epochs, seed, device):
    """One configuration, returning the best epoch's weights by validation loss.

    Their appendix trains for at most 50 epochs with early stopping on a
    validation set and keeps the best epoch, so a run that trains a fixed number
    and takes the last is a weaker switch than the one they describe.
    """
    torch.manual_seed(seed)
    train, valid = ~split, split
    model = Switcher(x.shape[1], dropout=dropout).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    best, best_state, stale = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        order = torch.randperm(int(train.sum()))
        xt, yt = x[train], y[train]
        for start in range(0, len(order), batch):
            index = order[start : start + batch]
            if len(index) < 2:      # batch norm needs more than one row
                continue
            optimiser.zero_grad()
            switch_loss(model(xt[index].to(device)), yt[index].to(device)).backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            loss = float(switch_loss(model(x[valid].to(device)), y[valid].to(device)))
        if loss < best - 1e-6:
            best, stale = loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 8:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best


def fit(train, epochs=50, seed=0, device="cpu", trials=24, verbose=False):
    """Their search, over the ranges their appendix gives, not one guessed point.

    Learning rate log-uniform over [1e-5, 1e-2], batch from {16, 32, 64, 128},
    dropout uniform over [0, 0.5], AdamW, at most 50 epochs, early stopping on a
    validation split with the best epoch kept. They search these with Ray Tune;
    `trials` random draws is the same search done smaller, and doing none at all
    would report a baseline we had weakened ourselves.
    """
    x = torch.as_tensor(np.asarray(train["features"], np.float32))
    mean, std = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp_min(1e-6)
    x = (x - mean) / std
    y = torch.as_tensor(np.stack([train["short_correct"], train["long_correct"]], 1)
                        .astype(np.float32))
    rng = np.random.default_rng(seed)
    split = torch.as_tensor(rng.random(len(x)) < 0.2)       # their validation set
    best = (float("inf"), None, None)
    for trial in range(trials):
        lr = float(10 ** rng.uniform(-5, -2))
        batch = int(rng.choice([16, 32, 64, 128]))
        dropout = float(rng.uniform(0.0, 0.5))
        model, loss = _train_once(x, y, split, lr, batch, dropout, epochs,
                                  seed + trial, device)
        if verbose:
            print(f"  trial {trial:2d}: lr {lr:.2e} batch {batch:3d} "
                  f"dropout {dropout:.2f} -> {loss:.4f}")
        if loss < best[0]:
            best = (loss, model, (lr, batch, dropout))
    lr, batch, dropout = best[2]
    if verbose:
        print(f"  chosen: lr {lr:.2e} batch {batch} dropout {dropout:.2f} "
              f"at validation {best[0]:.4f}")
    return best[1], (mean, std), {"lr": lr, "batch": batch, "dropout": dropout,
                                  "validation_loss": best[0]}


def sweep(model, norm, data, taus=None, device="cpu", points=201):
    """The curve, over a grid of tau that reaches both endpoints.

    The grid is taken from the predicted gaps rather than fixed, because a fixed
    range that the predictions do not span leaves the curve short of escalating
    none or escalating all. Read at a budget outside what it covers, such a curve
    reports the accuracy of its nearest end -- which at zero tokens is an
    escalating policy's accuracy filed under the fast mode's cost.
    """
    mean, std = norm
    x = (torch.as_tensor(np.asarray(data["features"], np.float32)) - mean) / std
    with torch.no_grad():
        predicted = model(x.to(device)).cpu().numpy()
    gap = predicted[:, 1] - predicted[:, 0]
    if taus is None:
        span = float(gap.max() - gap.min()) or 1.0
        taus = np.linspace(gap.min() - 0.02 * span, gap.max() + 0.02 * span, points)
    out = []
    for tau in taus:
        long_ = gap >= tau
        out.append({
            "tau": float(tau),
            "long_share": float(long_.mean()),
            "tokens": float(np.where(long_, data["long_tokens"], data["short_tokens"]).mean()),
            "accuracy": float(np.where(long_, data["long_correct"],
                                       data["short_correct"]).mean()),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records", help="one thinking-mode collection, both arms "
                                          "from its fast and slow columns")
    parser.add_argument("--short", help="training set, slow_mode=short")
    parser.add_argument("--long", help="training set, slow_mode=think")
    parser.add_argument("--evaluate", help="a second collection to score on")
    parser.add_argument("--eval-short"), parser.add_argument("--eval-long")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--held-out", action="store_true",
                        help="evaluate on the held-out categories of the training files")
    parser.add_argument("--train-categories", action="store_true",
                        help="with --evaluate, fit on the training categories only. The "
                             "served head this is compared against was fitted that way "
                             "(`split_level: category`), so without this flag the baseline "
                             "trains on a third more questions than the head did and the "
                             "curves differ by their training sets as well as their rules.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--trials", type=int, default=24,
                        help="random draws over the ranges their appendix gives")
    parser.add_argument("--output")
    arguments = parser.parse_args()

    if arguments.records:
        data = one_file(arguments.records, arguments.layer)
    elif arguments.short and arguments.long:
        data = join(arguments.short, arguments.long, arguments.layer)
    else:
        raise SystemExit("give --records, or --short and --long")
    take = lambda d, m: {k: (v[m] if isinstance(v, np.ndarray) else v)
                         for k, v in d.items()}
    if arguments.evaluate:
        train, test = data, one_file(arguments.evaluate, arguments.layer)
    elif arguments.eval_short and arguments.eval_long:
        train, test = data, join(arguments.eval_short, arguments.eval_long, arguments.layer)
    elif arguments.held_out:
        held = np.asarray(holdout_mask(data["category"], 0.25, 0)[0], bool)
        train, test = take(data, ~held), take(data, held)
    else:
        raise SystemExit("give --eval-short/--eval-long, or --held-out")
    if arguments.train_categories and (arguments.evaluate or arguments.eval_long):
        held = np.asarray(holdout_mask(data["category"], 0.25, 0)[0], bool)
        train = take(data, ~held)

    # len(features), not data["n"]: `n` is a scalar the subset mask leaves alone, so
    # it reports the whole collection even when the fit is on a subset of it.
    print(f"{data['model']}: {len(train['features'])} train, "
          f"{len(test['features'])} test")
    print(f"  short {test['short_correct'].mean():.4f} at "
          f"{test['short_tokens'].mean():.0f} tok, "
          f"long {test['long_correct'].mean():.4f} at {test['long_tokens'].mean():.0f} tok")

    model, norm, chosen = fit(train, epochs=arguments.epochs, trials=arguments.trials,
                              verbose=True)
    curve = sweep(model, norm, test)
    print(f"\n{'tau':>6s} {'long share':>11s} {'tokens':>8s} {'accuracy':>9s}")
    for row in curve[::20]:
        print(f"{row['tau']:6.2f} {row['long_share']:11.1%} {row['tokens']:8.0f} "
              f"{row['accuracy']:9.4f}")
    if arguments.output:
        from hde.run.streaming import write_json_atomic
        write_json_atomic(arguments.output, {"model": data["model"], "curve": curve,
                                             "k": 1, "test_items": len(test["features"]),
                                             "trials": arguments.trials, "chosen": chosen})
        print(f"\nwrote {arguments.output}")


if __name__ == "__main__":
    main()
