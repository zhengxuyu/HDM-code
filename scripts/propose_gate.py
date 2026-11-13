"""Can the fast mode build the candidate set it needs?

The schema calls a candidate set optional and nothing made the optional case
work, so every benchmark in the paper is multiple choice. `litjev.construct`
lets the fast mode propose its own answers with a beam; whether that is worth
building on comes down to three numbers, and this measures them before anything
is built.

    coverage      how often the gold answer is among the beams. The fast mode
                  cannot read out an answer that is not there, and the slow mode
                  has no such cap, so this is the ceiling on the fast path.
    rank          where the gold answer sits when it is there. If it is beam one
                  almost always, the readout has nothing to decide.
    disagreement  how often the readout departs from beam one, and who is right
                  when it does. If it never departs, the second forward pass is
                  dead weight and the fast answer should simply be beam one.

Thresholds are in the experiment board and were written down before this ran.
No slow mode, no head, no training: this is one beam pass and one readout.

    uv run python scripts/propose_gate.py --model ... --limit 300
"""
import argparse
from collections import Counter

import numpy as np


def main():
    from litjev.candidates.construct import (
        FREQUENCY,
        LIKELIHOOD,
        correct,
        equivalent,
        schema_from_beams,
        strip_enumerator,
    )
    from litjev.candidates.proposing import (
        add_proposal_arguments,
        check_proposal_arguments,
        load_proposer,
        looks_like_codes,
        probe_sampler,
        propose,
        watch_proposals,
    )
    from litjev.data.datasets import LOADERS, load
    from litjev.data.schema import DecisionSchema
    from litjev.run.provenance import check_imported_tree, commit_now, describe_environment
    from litjev.run.streaming import Rows, completed, write_json_atomic

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(LOADERS), default="gsm8k")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--split", default=None,
        help="what this dataset calls it, and left to the dataset when unset: test "
             "or validation for the multiple-choice sets, public or a tier for "
             "jevbench, and one fixed name each for agieval and gpqa-diamond, which "
             "refuse any other")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--stride", type=int, default=1)
    add_proposal_arguments(parser)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", default="propose-gate.json")
    parser.add_argument("--rows", help="append a row per question here as it is produced")
    parser.add_argument("--resume", action="store_true",
                        help="skip questions already in --rows, matched by id")
    arguments = parser.parse_args()
    check_proposal_arguments(parser, arguments)
    # launch.sh submits from a directory named after the tree, which nothing
    # writes over. This checks the other half: that the process is importing from
    # that directory rather than through the shared environment's editable
    # install, which points at the tree every sync replaces.
    check_imported_tree()

    loaded = load(arguments.dataset, split=arguments.split)
    questions = loaded.questions[:: arguments.stride][: arguments.limit]
    labels = loaded.labels
    if not questions:
        parser.error("no questions selected")

    symbolic = loaded.provenance.get("answer_form") == "expression"
    scorer = load_proposer(arguments.model, backend=arguments.backend,
                           revision=arguments.revision, dtype=arguments.dtype,
                           device_map=arguments.device_map)

    ranks = (LIKELIHOOD, FREQUENCY) if arguments.rank == "both" else (arguments.rank,)
    done = completed(arguments.rows) if arguments.resume else {}
    if done:
        print(f"resuming: {len(done)} questions already done", flush=True)
    rows = [done[question.question_id] for question in questions
            if question.question_id in done]
    reasons = Counter()
    for row in rows:
        for rank, entry in row["by_rank"].items():
            if entry["reason"]:
                reasons[f"{rank}:{entry['reason']}"] += 1
    # Asked once, before anything is recorded: whether this engine's seed fixes
    # its draw or replaces it. The running watch below only covers temperatures
    # at which a total collapse cannot be the temperature's doing, and the sweep
    # goes below them.
    probe = {}
    if arguments.seed is not None:
        stop, probe = probe_sampler(scorer, "Answer the question.", [q.prompt for q in questions],
                                    arguments.width, arguments.max_new_tokens, arguments.seed)
        print(f"    sampler probe: {probe}", flush=True)
        if stop:
            raise SystemExit(f"stopping before the first question: {stop}")
    constraint = None
    if arguments.verified:
        kind, name = scorer.proposal_constraint()
        constraint = {"kind": kind, "parameter": name}
        print(f"    proposals are constrained by {kind} ({name})", flush=True)
    stream = Rows(arguments.rows, resume=arguments.resume, config={
        "commit": commit_now(), "environment": describe_environment(), **probe,
        "dataset": arguments.dataset, "model": arguments.model,
        "backend": arguments.backend, "temperature": arguments.temperature,
        "seed": arguments.seed, "constraint": constraint,
        "width": arguments.width, "rank": arguments.rank, "max_new_tokens": arguments.max_new_tokens})
    enumerated = 0
    seen_distinct, coded, warned = [], 0, False
    for index, question in enumerate(questions):
        name = question.question_id
        if name in done:
            continue
        if name not in labels:
            raise KeyError(f"{name} has no gold answer; scoring it as uncovered would "
                           f"understate the very number this gate decides")
        gold = str(labels[name])
        # One proposal pass, read out once per rank. Two passes would draw two
        # samples, and the difference between the ranks would then be partly the
        # difference between the draws -- which is exactly what a ranking by how
        # often an answer came up is a statement about.
        proposals = propose(scorer, "Answer the question.", question.prompt,
                            arguments.width, arguments.max_new_tokens, arguments.temperature,
                            seed=arguments.seed)
        # Printed, not summarised: the failure this guards against was invisible
        # in every number the run produced and obvious in one line of its output.
        if len(rows) < 5:
            print(f"    proposals for {name}: "
                  f"{[text for text, _ in proposals][:10]}", flush=True)
        seen_distinct.append(len({text.strip() for text, _ in proposals}))
        enumerated += sum(strip_enumerator(text)[1] for text, _ in proposals)
        coded += looks_like_codes(proposals)
        abort, warning = watch_proposals(len(seen_distinct), seen_distinct, coded,
                                         seeded=arguments.seed is not None,
                                         temperature=arguments.temperature)
        if abort:
            stream.close()
            raise SystemExit(f"stopping: {abort}")
        if warning and not warned:
            print(f"    note: {warning}", flush=True)
            warned = True
        row = {"id": name, "gold": gold, "by_rank": {}}
        for rank in ranks:
            built, texts, reason = schema_from_beams(proposals, instructions=question.prompt,
                                                     rank=rank)
            covered = [i for i, text in enumerate(texts)
                       if equivalent(text, gold, symbolic=symbolic)]
            entry = {"reason": reason, "candidates": len(texts),
                     "first": texts[0] if texts else None,
                     "gold_rank": covered[0] if covered else None,
                     "gold_in_proposals": bool(covered)}
            if reason:
                reasons[f"{rank}:{reason}"] += 1
            else:
                (field,) = scorer.score("Answer the question.",
                                        DecisionSchema({name: built}))
                chosen = int(np.argmax(np.asarray(field.logits, dtype=np.float64)))
                entry.update({"readout": texts[chosen], "readout_index": chosen,
                              "first_right": correct(texts[0], gold, symbolic=symbolic),
                              "readout_right": correct(texts[chosen], gold,
                                                       symbolic=symbolic)})
            row["by_rank"][rank] = entry
        rows.append(row)
        stream.append(row)
        if index % 25 == 0:
            print(f"  {index + 1}/{len(questions)}", flush=True)

    stream.close()

    n = len(rows)
    print(f"\n{n} questions, {arguments.backend} width {arguments.width}, "
          f"temperature {arguments.temperature}")
    # Coverage does not depend on the rank: reordering a set does not change what
    # is in it. It is reported once, from the first rank, and asserted equal.
    coverage = {rank: sum(row["by_rank"][rank]["gold_in_proposals"] for row in rows)
                for rank in ranks}
    if len(set(coverage.values())) != 1:
        raise AssertionError(f"coverage differs between ranks {coverage}, which cannot "
                             f"happen if they order one proposal set")
    covered = next(iter(coverage.values()))
    print(f"  coverage over all questions      {covered}/{n} = {covered / n:.3f}")

    summary = {}
    for rank in ranks:
        entries = [row["by_rank"][rank] for row in rows]
        scored = [entry for entry in entries if entry["reason"] is None]
        in_set = [entry for entry in scored if entry["gold_rank"] is not None]
        departed = [entry for entry in scored if entry["readout_index"] != 0]
        at_first = sum(entry["gold_rank"] == 0 for entry in in_set)
        better = sum(entry["readout_right"] and not entry["first_right"]
                     for entry in departed)
        worse = sum(entry["first_right"] and not entry["readout_right"]
                    for entry in departed)
        summary[rank] = {
            "with_a_choice": len(scored), "degenerate": len(entries) - len(scored),
            "gold_at_first": at_first, "gold_in_set": len(in_set),
            "readout_departed": len(departed),
            "departed_readout_better": better, "departed_readout_worse": worse,
            "first_accuracy": float(np.mean([e["first_right"] for e in scored])) if scored else 0.0,
            "readout_accuracy": float(np.mean([e["readout_right"] for e in scored])) if scored else 0.0,
        }
        print(f"\n  rank {rank}")
        print(f"    degenerate                     {summary[rank]['degenerate']}/{n}")
        print(f"    gold at candidate one          {at_first}/{len(in_set)} = "
              f"{at_first / max(len(in_set), 1):.3f}")
        print(f"    readout departs from candidate one  {len(departed)}/{len(scored)} = "
              f"{len(departed) / max(len(scored), 1):.3f}")
        print(f"      readout right, first wrong   {better}")
        print(f"      first right, readout wrong   {worse}")
        print(f"    accuracy: candidate one {summary[rank]['first_accuracy']:.3f}, "
              f"readout {summary[rank]['readout_accuracy']:.3f}")

    write_json_atomic(arguments.output,
                      {"commit": commit_now(),
                       "environment": describe_environment(),
                       "dataset": arguments.dataset, "model": arguments.model,
                       "backend": arguments.backend, "ranks": list(ranks),
                       "coverage": covered / n, "summary": summary,
                       "temperature": arguments.temperature,
                       "seed": arguments.seed,
                       "constraint": constraint, "enumerated_proposals": enumerated,

                       "width": arguments.width, "questions": n,
                       "reasons": dict(reasons),
                       "rows": rows})
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
