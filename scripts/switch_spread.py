"""Does validation loss pick a ThinkSwitcher that can actually be tuned?

The scatter in Figure~\\ref{fig:curves} showed the reproduced switch offering only
six distinct operating points on JevBench, with 176 of 201 thresholds giving the
identical escalation set: its predicted gap is nearly constant across items, so
there is no cut between 2.6% and 60.2% escalation. A router that cannot be moved to
a budget is not a fair stand-in for the published method, and the cause might be
ours -- `fit` selects on validation loss, and a model that predicts the mean gap
everywhere scores well on mean-squared error while being useless to threshold.

This runs the same search and reports, per trial, the validation loss *and* the
number of distinct operating points the model yields on the evaluation set. If the
low-loss trials are the degenerate ones and some higher-loss trial is tunable, then
the criterion is at fault and the baseline is ours to fix. If every trial collapses,
the collapse belongs to the method as we can reconstruct it, and the paper says so.

    uv run python scripts/switch_spread.py --records R.npz --evaluate E.npz \
        --train-categories --trials 24
"""
import argparse

import numpy as np
import torch

from hde.train.training import holdout_mask
from thinkswitcher import Switcher, _train_once, one_file, sweep


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records", required=True)
    parser.add_argument("--evaluate", required=True)
    parser.add_argument("--train-categories", action="store_true")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    data = one_file(arguments.records, arguments.layer)
    test = one_file(arguments.evaluate, arguments.layer)
    train = data
    if arguments.train_categories:
        held = np.asarray(holdout_mask(data["category"], 0.25, 0)[0], bool)
        train = {k: (v[~held] if isinstance(v, np.ndarray) else v)
                 for k, v in data.items()}

    x = torch.as_tensor(np.asarray(train["features"], np.float32))
    mean, std = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp_min(1e-6)
    x = (x - mean) / std
    y = torch.as_tensor(np.stack([train["short_correct"], train["long_correct"]], 1)
                        .astype(np.float32))
    rng = np.random.default_rng(arguments.seed)
    split = torch.as_tensor(rng.random(len(x)) < 0.2)

    print(f"{data['model']}: {len(train['features'])} train, {len(test['features'])} test")
    print(f"\n{'trial':>5s} {'lr':>9s} {'batch':>6s} {'drop':>5s} {'val loss':>9s} "
          f"{'gap sd':>8s} {'points':>7s} {'area':>7s}")
    rows = []
    for trial in range(arguments.trials):
        lr = float(10 ** rng.uniform(-5, -2))
        batch = int(rng.choice([16, 32, 64, 128]))
        dropout = float(rng.uniform(0.0, 0.5))
        model, loss = _train_once(x, y, split, lr, batch, dropout, arguments.epochs,
                                  arguments.seed + trial, "cpu")
        curve = sweep(model, (mean, std), test)
        points = len({(round(r["tokens"], 3), round(r["accuracy"], 6)) for r in curve})
        # The predicted gap's spread is what a threshold has to work with.
        xt = (torch.as_tensor(np.asarray(test["features"], np.float32)) - mean) / std
        with torch.no_grad():
            predicted = model(xt).numpy()
        spread = float((predicted[:, 1] - predicted[:, 0]).std())
        tok = np.array([r["tokens"] for r in curve])
        acc = np.array([r["accuracy"] for r in curve])
        order = np.argsort(tok)
        area = float(np.trapezoid(np.maximum.accumulate(acc[order]), tok[order])
                     / (tok.max() - tok.min()))
        rows.append((loss, spread, points, area, lr, batch, dropout, trial))
        print(f"{trial:5d} {lr:9.2e} {batch:6d} {dropout:5.2f} {loss:9.4f} "
              f"{spread:8.4f} {points:7d} {area:7.4f}")

    by_loss = sorted(rows)[0]
    by_area = sorted(rows, key=lambda r: -r[3])[0]
    print(f"\nselected on validation loss : trial {by_loss[7]}, loss {by_loss[0]:.4f}, "
          f"{by_loss[2]} operating points, area {by_loss[3]:.4f}")
    print(f"best area available in search: trial {by_area[7]}, loss {by_area[0]:.4f}, "
          f"{by_area[2]} operating points, area {by_area[3]:.4f}")
    print(f"\ncorrelation of validation loss with operating points: "
          f"{np.corrcoef([r[0] for r in rows], [r[2] for r in rows])[0, 1]:+.3f}")
    print(f"correlation of validation loss with area              : "
          f"{np.corrcoef([r[0] for r in rows], [r[3] for r in rows])[0, 1]:+.3f}")


if __name__ == "__main__":
    main()
