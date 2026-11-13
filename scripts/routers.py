"""Two published routing baselines over our two modes: KNN-Router and RouteLLM.

Both are the versions \\citet{pan2025rtr} run for exactly this comparison --
binary routing between a backbone's thinking and non-thinking modes, their
Table 4 -- and both are specified completely enough to implement, which the
switch of \\citet{liang2025thinkswitcher} is not: that paper's appendix gives
Ray Tune search ranges rather than chosen values, so reproducing it means
reproducing its search and the number depends on how well we ran it.

**KNN-Router** (their k=10) carries no weights and nothing to tune. For a test
query it takes the 10 nearest training queries in feature space and scores each
mode by `mean(correct) - lambda * mean(tokens)` over those neighbours, taking the
higher. Sweeping `lambda` traces the cost-accuracy curve. Two things make this
the fairest learned baseline available to us. It cannot be weakened by our
implementation, since its only choices are k, which they state, and `lambda`,
which is the axis of the comparison. And its score is the same functional form
as our own price of thinking (Eq.~\\ref{eq:gain}, Eq.~\\ref{eq:reward}): accuracy
minus a per-token price. So the comparison isolates the estimator -- a local
average over neighbours against a head fitted on the hidden state -- with the
objective, the features and the modes all held fixed.

**RouteLLM**, as they implement it: label each training query by which mode is
more accurate on it, fit a binary classifier on the query features, and sweep the
decision threshold. Logistic regression is that classifier here, with its one
regularisation strength chosen on a validation split, so the sweep is over the
threshold alone. The published RouteLLM \\citep{ong2024routellm} offers four
routers over a *model* pair and a preference dataset; this is the binary-
classifier reduction to a *mode* pair that their Table 4 reports, and it is
labelled as such.

EmbedLLM, the third of their learned baselines, is omitted and this is why: it
factorises a prompts-by-candidates correctness matrix to learn a vector per
candidate. With two candidates that matrix has two columns, and a rank-d
factorisation of it recovers the two columns rather than generalising across a
pool. It is a method for a model pool and there is no pool here.

    uv run python scripts/routers.py --records results/records-...npz --held-out \
        --output results/routers-27b-mmlupro.json
"""
import argparse

import numpy as np

from hde.train.training import holdout_mask
from thinkswitcher import join, one_file


def curve_from_margin(margin, data, points=201):
    """Every cut of one routing score, as the shared (tokens, accuracy) curve.

    The grid spans the score's own range, so the curve reaches both escalating
    none and escalating all. A fixed grid that the scores do not span stops short
    of an endpoint, and reading a budget beyond it reports the nearest end's
    accuracy at the wrong cost.
    """
    span = float(margin.max() - margin.min()) or 1.0
    taus = np.linspace(margin.min() - 0.02 * span, margin.max() + 0.02 * span, points)
    out = []
    for tau in taus:
        long_ = margin >= tau
        out.append({
            "tau": float(tau),
            "long_share": float(long_.mean()),
            "tokens": float(np.where(long_, data["long_tokens"],
                                     data["short_tokens"]).mean()),
            "accuracy": float(np.where(long_, data["long_correct"],
                                       data["short_correct"]).mean()),
        })
    return out


def knn_router(train, test, k=10):
    """Their KNN-Router: a local average over the k nearest training queries.

    Their score for a candidate is `mean accuracy - lambda * mean length` over the
    neighbours, and the router takes the higher of the two. Here the fast mode
    costs nothing, so the two scores differ by
    `mean(slow correct) - mean(fast correct) - lambda * mean(slow tokens)`, and a
    sweep of `lambda` is a sweep of that margin. Rather than sweep `lambda` on a
    guessed grid, the two neighbourhood averages are returned and the margin is
    swept directly, which covers every `lambda` including the ones no grid lands
    on -- the same curve their tuning of `lambda` searches over, read exactly.

    Distances are cosine, on features standardised by the training set, because
    the readout state's coordinates are on very different scales and a Euclidean
    neighbourhood would be chosen by whichever few of the 5120 happen to be large.
    """
    x = np.asarray(train["features"], np.float64)
    mean, std = x.mean(0), x.std(0).clip(1e-6)
    x = (x - mean) / std
    q = (np.asarray(test["features"], np.float64) - mean) / std
    x /= np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)
    q /= np.linalg.norm(q, axis=1, keepdims=True).clip(1e-12)

    neighbours = np.argsort(-(q @ x.T), axis=1, kind="stable")[:, :k]
    gain = (train["long_correct"].astype(float)[neighbours].mean(1)
            - train["short_correct"].astype(float)[neighbours].mean(1))
    price = np.asarray(train["long_tokens"], float)[neighbours].mean(1)
    # margin > 0 at lambda = gain/price, so gain/price orders the questions the
    # way a sweep of lambda escalates them: the best value per token goes first.
    return gain / price.clip(1e-9)


def _logistic(x, y, strength):
    """L2-penalised logistic regression, full batch, to convergence.

    `scikit-learn` is not installed on the cluster and this is not worth a
    dependency: the objective is the same one its lbfgs solver minimises --
    mean cross-entropy plus `||w||^2 / (2 C n)` -- and LBFGS on the whole batch
    reaches it deterministically, with no learning rate or epoch count to pick.
    """
    import torch

    xt = torch.as_tensor(x, dtype=torch.float64)
    yt = torch.as_tensor(np.asarray(y, float), dtype=torch.float64)
    weight = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimiser = torch.optim.LBFGS([weight, bias], max_iter=500,
                                  tolerance_grad=1e-9, line_search_fn="strong_wolfe")

    def closure():
        optimiser.zero_grad()
        logits = xt @ weight + bias
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, yt)
                + weight.square().sum() / (2.0 * strength * len(xt)))
        loss.backward()
        return loss

    optimiser.step(closure)
    return weight.detach().numpy(), float(bias.detach())


def route_llm(train, test, seed=0):
    """Their RouteLLM: a binary classifier on the query, thresholded.

    The label is which mode is more accurate on that query, which for a single
    greedy pass is `slow_correct > fast_correct`; the score swept is the
    classifier's probability. Regularisation strength is chosen on a 20% split so
    the one free parameter is not a guess, and nothing else is tuned.
    """
    x = np.asarray(train["features"], np.float64)
    mean, std = x.mean(0), x.std(0).clip(1e-6)
    x = (x - mean) / std
    q = (np.asarray(test["features"], np.float64) - mean) / std
    y = (train["long_correct"].astype(int) > train["short_correct"].astype(int))

    rng = np.random.default_rng(seed)
    held = rng.random(len(x)) < 0.2
    if y[~held].min() == y[~held].max():
        raise SystemExit("every training query has the same better mode; "
                         "a binary classifier has nothing to separate")
    best = (-np.inf, 1.0)
    for strength in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
        weight, bias = _logistic(x[~held], y[~held], strength)
        predicted = (x[held] @ weight + bias) > 0
        score = float((predicted == y[held]).mean())
        print(f"  C {strength:<6g} validation accuracy {score:.4f}")
        if score > best[0]:
            best = (score, strength)
    print(f"  chosen: C {best[1]:g} at validation {best[0]:.4f}")
    weight, bias = _logistic(x, y, best[1])
    return 1.0 / (1.0 + np.exp(-(q @ weight + bias))), {
        "C": best[1], "validation_accuracy": best[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records", help="one thinking-mode collection")
    parser.add_argument("--short"), parser.add_argument("--long")
    parser.add_argument("--evaluate", help="a second collection to score on")
    parser.add_argument("--held-out", action="store_true")
    parser.add_argument("--train-categories", action="store_true",
                        help="with --evaluate, fit on the training categories only. The "
                             "served head this is compared against was fitted that way "
                             "(`split_level: category`), so without this flag the baseline "
                             "trains on a third more questions than the head did and the "
                             "curves differ by their training sets as well as their rules.")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--k", type=int, default=10, help="their value")
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
        if arguments.train_categories:
            held = np.asarray(holdout_mask(data["category"], 0.25, 0)[0], bool)
            train = take(data, ~held)
    elif arguments.held_out:
        held = np.asarray(holdout_mask(data["category"], 0.25, 0)[0], bool)
        train, test = take(data, ~held), take(data, held)
    else:
        raise SystemExit("give --evaluate or --held-out")

    print(f"{data['model']}: {len(train['features'])} train, "
          f"{len(test['features'])} test")
    print(f"  fast {test['short_correct'].mean():.4f}, "
          f"slow {test['long_correct'].mean():.4f} at "
          f"{test['long_tokens'].mean():.0f} tok")

    print(f"\nKNN-Router, k={arguments.k} (no weights, nothing tuned)")
    knn = curve_from_margin(knn_router(train, test, arguments.k), test)
    print("\nRouteLLM (binary classifier on the query)")
    scores, chosen = route_llm(train, test)
    route = curve_from_margin(scores, test)

    for name, rows in (("KNN-Router", knn), ("RouteLLM", route)):
        print(f"\n{name}")
        print(f"{'escalated':>10s} {'tokens':>8s} {'accuracy':>9s}")
        for row in rows[::40]:
            print(f"{row['long_share']:10.1%} {row['tokens']:8.0f} "
                  f"{row['accuracy']:9.4f}")

    if arguments.output:
        from hde.run.streaming import write_json_atomic
        write_json_atomic(arguments.output, {
            "model": data["model"], "test_items": len(test["features"]),
            "k": arguments.k, "route_llm_chosen": chosen,
            "curves": {"KNN-Router": knn, "RouteLLM": route}})
        print(f"\nwrote {arguments.output}")


if __name__ == "__main__":
    main()
