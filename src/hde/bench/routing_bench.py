"""Measure the three quantities a routing policy trades off: time, tokens, accuracy.

Every policy is actually executed on the same questions, in the same ten-question
batches, so timings reflect real batching rather than a per-question estimate.
Gold labels are compared locally after each call and never enter a prompt.
"""

from __future__ import annotations

import socket
from datetime import datetime
from time import perf_counter

import math

import numpy as np

from hde.core.block import GAIN_EPS, MarginPolicy, MarginRouter, RoutingPolicy

# The price at which every question escalates, which is what makes the `slow_only`
# row slow-only. **Derived from the gain's floor rather than chosen**, because the
# floor moved: this was `-2.0` and justified as "below any gain in [-1, 1]", which
# was true while the gain was the difference `P(slow right) - P(fast right)` and
# false from 6ad45a1 onward, where it became `log P(slow right) - log P(fast
# right)`. A log ratio is bounded only by the epsilon the logs are floored at, so
# its minimum is `log(GAIN_EPS)`, about -13.82, and `-2.0` sat well inside the
# attainable range.
#
# Nothing measured was wrong: on `head-4b-sglang-29662` the gain bottoms out at
# -0.36 on the held-out 346 and -0.76 over all 1500, so every question still
# escalated and `slow_only` was slow-only. But it held by luck, and the failure it
# risked is silent and expensive -- a head confident that thinking would *hurt*
# some question would have left that question unescalated in the row every bracket
# in Table 1 is measured against, and the caption's "a row escalating 100% is the
# slow-only policy" would have been false with nothing raising.
ALL = math.log(GAIN_EPS) - 1.0


def _serving(field, key):
    """`cached_tokens` and `hidden_positions` from one mode's provenance.

    None when the mode did not run or the engine does not report them, so a
    reader can tell "not escalated" from "escalated and not reported" rather
    than seeing a zero for both.
    """
    provenance = field.get(key) or {}
    if not provenance:
        return None
    return {"cached_tokens": provenance.get("cached_tokens"),
            "hidden_positions": provenance.get("hidden_positions"),
            # From the provenance, not from the field: the field dict a policy
            # writes does not carry `input_tokens`, so reading it there returned
            # null for every row of 29654 and left the cache count without a
            # denominator. exps found it in the first rows on disk.
            "prompt_tokens": provenance.get("prompt_tokens")}


def _positions(schema, answers, diagnostics, labels, scored_ids):
    """Where each answer sat among the options, and how close the call was.

    A readout with a position prior and a readout reading the question produce the
    same accuracy column and different histograms. The rows carried the verdict and
    not the position, so a question in the main table could not be audited after
    the fact -- and with a prior as strong as the constructed sets show, "how much
    of this column is position" had no answer that did not need a new run.

    `score` and `noul` questions have no option order, so they are left out rather
    than given a meaningless index.
    """
    found = {}
    for key in scored_ids:
        question = schema[key]
        if question.type != "choice":
            continue
        options = list(question.choices)
        chosen = _answer_key(answers[key])
        distribution = diagnostics["fields"][key].get("probabilities") or {}
        found[key] = {
            "options": len(options),
            "chosen_index": options.index(chosen) if chosen in options else -1,
            "gold_index": (options.index(labels[key])
                           if labels.get(key) in options else -1),
            # In option order, so a histogram over positions can be taken without
            # the record having to be joined back to the schema.
            "probabilities": [round(float(distribution.get(option, float("nan"))), 5)
                              for option in options],
        }
    return found


def _answer_key(answer):
    """The answer in the form a gold label is written in, for all three types.

    Scoring only `choice` is why the paper's evidence was from `choice` alone.
    A `score` question answers with a level, so its key is the most probable
    level rather than the probability-weighted index the typed answer reports:
    the weighted index is the product's output, but accuracy asks which level
    was chosen. A `noul` question answers with a probability of true, so its
    key is that probability thresholded at a half.
    """
    if answer.type == "choice":
        return answer.choice
    if answer.type == "score":
        if not answer.probabilities:
            return None
        return int(max(answer.probabilities, key=answer.probabilities.get))
    if answer.type == "noul":
        return bool(answer.noul >= 0.5)
    return None


def batch_key(scored_ids):
    """A batch's identity, from the questions in it rather than its position.

    Resume keys on this, not on a count. `hde.streaming` learned that the hard
    way for a per-question file: a run that skipped a question resumes at the wrong
    place and scores one question against another's gold answer, and nothing says
    so. The same hazard is worse here, because a row covers ten questions and seven
    policies, so a misaligned resume would mispair seventy outcomes rather than one.

    Order matters and is kept: the engine sees the batch in this order, and two
    batches holding the same questions in a different order are not the same work.
    """
    return "+".join(str(key) for key in scored_ids)


def bench_policies(
    engine, batches, labels, lambdas, budget, synchronize=lambda: None, progress=1,
    label="batch", rows_file=None, done=None, repeat=0, node=None, margin_taus=(),
):
    """Run fast-only, slow-only and routed policies on the same batches.

    `batches` is a list of (SystemOneRequest, scored_ids). Returns a report with
    per-policy accuracy, generated tokens per question, seconds per question and
    escalation rate, plus per-batch rows for auditing.

    `rows_file` is a `streaming.Rows` written as the run goes, and `done` the rows
    a previous run already produced, keyed by `repeat|policy|batch_key`. The unit
    is one `evaluate` call, so a kill can lose at most that one call: a row is
    written only once its call has returned, and a truncated final line is dropped
    by `completed` rather than read.

    This existed as a list in memory and a single write at the end, which is how
    27862 lost four hours and ten minutes of 27B with nothing on disk -- not even a
    progress line. These runs are longer.

    `node` goes on every row, because a resume on Slurm rarely lands on the machine
    the first attempt ran on and seconds per question is one of the three quantities
    this reports. Refusing a cross-node resume would make resume useless here, so
    the rows carry where each was measured and `_summarise` says when a policy's
    seconds come from more than one machine. Accuracy and tokens do not depend on
    it; the clock does.

    `progress` is how many batches pass between lines, and every batch by default.
    It used to be every tenth and nothing at all during warmup, which at 27B with a
    budget of 8192 and seven policies to a batch meant a run could go ten hours
    without a line -- long enough that there was no way to tell a slow job from a
    hung one, which is the only question anyone asks of a log while waiting.
    """
    policies = [("fast_only", None), ("slow_only", RoutingPolicy(ALL, budget))]
    policies += [(f"routed_lambda_{lam:g}", RoutingPolicy(float(lam), budget)) for lam in lambdas]
    # **AnyJev's published rule, as a floor.** It fits nothing and reads only the
    # distribution the answering pass already produced, so it is the comparison a
    # trained head has to clear. We have that on MMLU-Pro and not on JevBench.
    #
    # One router per policy, created here and not per batch, because the cap is
    # over *recent* decisions and so depends on the order they are served. At 231
    # items the default cap (20% of a 1000-decision window, so 200 escalations)
    # cannot bind, which is worth knowing: at this scale the rule reduces to the
    # tau threshold, and so did theirs at a 15.2% rate.
    policies += [(f"margin_tau_{tau:g}", MarginPolicy(tau=float(tau), budget=budget))
                 for tau in margin_taus]
    routers = {name: MarginRouter(policy) for name, policy in policies
               if isinstance(policy, MarginPolicy)}
    rows = []
    done = done or {}
    node = node or socket.gethostname()
    began = perf_counter()
    for index, (request, scored_ids) in enumerate(batches, start=1):
        schema = request.to_schema()
        key = batch_key(scored_ids)
        for name, policy in policies:
            row_id = f"{repeat}|{name}|{key}"
            if row_id in done:
                # Taken as it was measured, including its seconds: re-timing it in a
                # second process would report this machine's hour rather than the one
                # the work was done in. Nothing else happens here, because the summary
                # is computed from the rows at the end -- so a resumed run and an
                # uninterrupted one agree by construction rather than by two
                # accumulations being written the same way.
                rows.append(done[row_id])
                continue
            synchronize()
            started = perf_counter()
            if policy is None:
                evaluation = engine.evaluate(request.state, schema)
            elif name in routers:
                evaluation = engine.evaluate(request.state, schema,
                                             router=routers[name])
            else:
                evaluation = engine.evaluate(request.state, schema, policy)
            synchronize()
            elapsed = perf_counter() - started
            answers = evaluation.result.answers
            # Keep the per-question verdict, not just the count. A benchmark with
            # more than one question type cannot be read from a single accuracy,
            # and a wrong answer cannot be traced back to its question without it.
            verdicts = {
                key: bool(labels.get(key) is not None and _answer_key(answers[key]) == labels[key])
                for key in scored_ids
            }
            correct = sum(verdicts.values())
            escalated = [
                key for key in evaluation.diagnostics["routing"]["escalated"] if key in scored_ids
            ]
            # Tokens for scored questions only; padding questions are excluded.
            tokens = sum(
                evaluation.diagnostics["fields"][key]["generated_tokens"] for key in escalated
            )
            row = {
                "id": row_id,
                "repeat": repeat,
                "batch": index,
                "question_ids": list(scored_ids),
                "correct_by_question": verdicts,
                # The primitive each question was asked as. Everything else here
                # is per-question, and this was the one quantity that was not, so
                # a per-type escalation rate on JevBench -- the only venue with
                # more than one type, and the venue whose type behaviour the paper
                # makes a claim about -- had to be recovered by joining the rows
                # back through the loader on `selected_question_ids`. That join
                # re-runs the selection to describe a run that already happened,
                # and the id prefix cannot stand in for it: those are task names
                # (`sol`, `opus`, `intent`), not primitives. One field here makes
                # the table derivable from the rows alone, as every other column
                # of it already is.
                "type_by_question": {key: schema[key].type for key in scored_ids},
                "position_by_question": _positions(
                    schema, answers, evaluation.diagnostics, labels, scored_ids),
                # The ids, the token counts and the text, not only their totals. A
                # killed run should be analysable from its rows and not merely
                # resumable: 27866 could be scraped for fast and slow correctness from
                # its log, and what it wrote existed only in the JSON that was never
                # written, which is why re-scoring it is impossible rather than
                # pending.
                #
                # There is no fast text to keep beside it: the fast mode generates
                # nothing, which is the property the whole method rests on. And no
                # stored quantity would let a readout that read the wrong token ids be
                # re-scored, because the logits over the right ids were never
                # computed -- which is why those ids are now checked in code rather
                # than left as a recoverable mistake.
                "escalated_by_question": list(escalated),
                "tokens_by_question": {
                    key: int(evaluation.diagnostics["fields"][key]["generated_tokens"])
                    for key in escalated
                },
                # **What the engine reused, per question and per mode.** The
                # escalation design turns on the slow prompt being one the fast
                # pass already computed; whether the engine served it from its
                # radix cache was argued from wall-clock and asserted in
                # docstrings, because the two numbers that answer it were
                # discarded. `cached_tokens` is the engine's own count of prompt
                # tokens it did not recompute, `hidden_positions` how many states
                # came back where the head reads one, and `slow` carries the same
                # pair for the escalated read.
                "serving_by_question": {
                    key: {
                        "fast": _serving(evaluation.diagnostics["fields"][key],
                                         "provenance"),
                        "slow": _serving(evaluation.diagnostics["fields"][key],
                                         "slow_provenance"),
                    }
                    for key in scored_ids
                },
                "reasoning_by_question": {
                    key: evaluation.diagnostics["fields"][key].get("reasoning")
                    for key in escalated
                },
                "policy": name,
                "node": node,
                "scored": len(scored_ids),
                # The denominator the seconds are apportioned by, so a resumed row
                # folds back exactly as it was counted the first time.
                "in_batch": len(schema),
                "correct": correct,
                "tokens": tokens,
                "seconds": elapsed,
                "escalated": len(escalated),
            }
            rows.append(row)
            if rows_file is not None:
                rows_file.append(row)
        if progress and index % progress == 0:
            spent = perf_counter() - began
            # Wall clock as well as elapsed: a log read the next morning has to be
            # placed against when the node was allocated and when a sync landed.
            print(f"{datetime.now():%H:%M:%S} {label} {index}/{len(batches)}  "
                  f"{spent / 60:.1f} min in, {spent / index:.0f}s per batch over "
                  f"{len(policies)} policies, "
                  f"{spent / index * (len(batches) - index) / 60:.0f} min left",
                  flush=True)
    summary = _summarise(rows, [name for name, _ in policies])
    slow, fast = summary["slow_only"], summary["fast_only"]
    for name, item in summary.items():
        item["tokens_vs_slow"] = _ratio(item["tokens_per_question"], slow["tokens_per_question"])
        item["seconds_vs_slow"] = _ratio(item["seconds_per_question"], slow["seconds_per_question"])
        item["median_seconds_vs_slow"] = _ratio(
            item["median_batch_seconds"], slow["median_batch_seconds"]
        )
        item["accuracy_delta_vs_slow"] = item["accuracy"] - slow["accuracy"]
        gain = slow["accuracy"] - fast["accuracy"]
        item["gain_kept"] = (item["accuracy"] - fast["accuracy"]) / gain if gain > 0 else None
    return {"policies": summary, "batches": rows, "budget": budget, "lambdas": list(lambdas)}


def _summarise(rows, names):
    """The per-policy figures, from the rows and from nothing else.

    Computed here rather than accumulated as the run goes, so that a resumed run
    and an uninterrupted one produce the same summary by construction. Two
    accumulations that have to be written the same way are the shape of fault this
    repository spent a day on.
    """
    summary = {}
    for name in names:
        mine = [row for row in rows if row["policy"] == name]
        bucket = {
            "correct": sum(row["correct"] for row in mine),
            "tokens": sum(row["tokens"] for row in mine),
            # Apportioned by the share of the batch that was scored, as it was when
            # the row was written: a padded batch costs time its padding should not
            # be charged for.
            "seconds": sum(row["seconds"] * row["scored"] / max(row["in_batch"], 1)
                           for row in mine),
            "escalated": sum(row["escalated"] for row in mine),
            "count": sum(row["scored"] for row in mine),
        }
        n = max(bucket["count"], 1)
        per_batch = [row["seconds"] for row in mine]
        nodes = sorted({row["node"] for row in mine if row.get("node")})
        summary[name] = {
            "accuracy": bucket["correct"] / n,
            "tokens_per_question": bucket["tokens"] / n,
            "seconds_per_question": bucket["seconds"] / n,
            # Median per-batch wall time is robust to co-tenant contention on shared nodes.
            "median_batch_seconds": float(np.median(per_batch)) if per_batch else 0.0,
            "escalation_rate": bucket["escalated"] / n,
            "questions": bucket["count"],
            "nodes": nodes,
            # Said rather than prevented. A resumed run on Slurm usually lands
            # somewhere else, so the choice is between a resume that is refused and
            # a seconds column that says where it came from.
            "seconds_mixed_nodes": len(nodes) > 1,
        }
    mixed = sorted({node for item in summary.values() for node in item["nodes"]})
    if any(item["seconds_mixed_nodes"] for item in summary.values()):
        print(f"seconds come from more than one machine ({', '.join(mixed)}): accuracy and "
              f"tokens are unaffected, the seconds and median columns are not comparable "
              f"with a single-machine run", flush=True)
    return summary


def _ratio(value, reference):
    return float(value / reference) if reference else None


def format_table(summary):
    """Plain-text table of the per-policy metrics."""
    header = (
        f"{'policy':22s} {'acc':>6s} {'tok/q':>8s} {'s/q':>8s} {'med s/b':>8s} "
        f"{'esc':>6s} {'tok%':>6s} {'time%':>6s} {'med%':>6s}"
    )
    lines = [header]
    for name, item in summary.items():
        lines.append(
            f"{name:22s} {item['accuracy']:6.3f} {item['tokens_per_question']:8.1f} "
            f"{item['seconds_per_question']:8.3f} {item.get('median_batch_seconds', 0):8.2f} "
            f"{item['escalation_rate']:6.2f} "
            f"{100 * (item['tokens_vs_slow'] or 0):5.0f}% {100 * (item['seconds_vs_slow'] or 0):5.0f}% "
            f"{100 * (item.get('median_seconds_vs_slow') or 0):5.0f}%"
        )
    return "\n".join(lines)


def mean_over_repeats(reports):
    """Average per-policy metrics over repeated runs; std on seconds for the caption."""
    names = list(reports[0]["policies"])
    merged = {}
    for name in names:
        rows = [r["policies"][name] for r in reports]
        merged[name] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in (
                "accuracy",
                "tokens_per_question",
                "seconds_per_question",
                "median_batch_seconds",
                "escalation_rate",
            )
        }
        merged[name]["seconds_std"] = float(np.std([row["seconds_per_question"] for row in rows]))
        merged[name]["questions"] = rows[0]["questions"]
        # Carried rather than rebuilt from a key list. The merged summary is what the
        # tables are read from, so dropping these here would put the seconds back to
        # saying nothing about which machine produced them -- the whole point of
        # recording them one line up the call stack.
        merged[name]["nodes"] = sorted({node for row in rows for node in row.get("nodes", [])})
        merged[name]["seconds_mixed_nodes"] = (
            any(row.get("seconds_mixed_nodes") for row in rows)
            or len(merged[name]["nodes"]) > 1
        )
    return merged
