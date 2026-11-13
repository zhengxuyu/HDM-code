"""Fast, slow and routed on a benchmark whose answers are not enumerated.

Every number in the paper comes from a question that arrived with its options
written down. This runs the three policies where none did: the fast mode
proposes its own candidates with a beam and reads one out, the slow mode thinks
and answers in free form, and the router decides between them from the state the
fast pass produced.

The slow mode is given the question and nothing else. It never sees the beams,
so it stays the policy the always-thinking column measures, and the price of
that is an asymmetry worth stating plainly: the fast mode cannot answer outside
its own beam and the slow mode can. Coverage is therefore a ceiling on the fast
column and on nothing else.

Records land one row per question with both modes' answers, the beams and the
readout.

Under `--readout coded` the row also carries the hidden state, so a head can be
fitted afterwards without running the benchmark again. Under the content readout
it does not: `score_options` returns no hidden state, and a row without one
cannot train anything. The fast, slow and oracle columns are unaffected. Giving
it the prefix-final state -- option-independent, and in the same role the coded
readout's is, the last prefix token before the answer -- is what a content-trained
head would need, and is not written yet. The two prompts differ, so it is not the
same position.

    uv run python scripts/open_three_policies.py --dataset aime-2025 --model ...
"""
import argparse
import json

import numpy as np


def scores_as_correct(answer, gold, symbolic):
    """Strict, because this becomes an accuracy column."""
    from hde.candidates.construct import correct

    return correct(str(answer), str(gold), symbolic=symbolic)


def covered_by(answer, gold, symbolic):
    """Lenient, because this becomes a ceiling and a missed match understates it."""
    from hde.candidates.construct import equivalent

    return equivalent(str(answer), str(gold), symbolic=symbolic)


def main():
    from hde._legacy.decision import DEFAULT_READOUT, READOUTS, SchemaDecisionEngine
    from hde._legacy.heads import parse_feature_layers
    from hde.candidates.construct import extract_answer, strip_enumerator
    from hde.candidates.construction_settings import ConstructionSettings
    from hde.candidates.proposing import (
        add_proposal_arguments,
        check_proposal_arguments,
        load_proposer,
        looks_like_codes,
        probe_sampler,
        propose,
        watch_proposals,
    )
    from hde.data.datasets import LOADERS, load
    from hde.data.schema import Choice, DecisionSchema
    from hde.run.provenance import check_imported_tree, commit_now, describe_environment
    from hde.run.streaming import Rows, completed, write_json_atomic

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(LOADERS), default="aime-2025")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    # This default was "train", which asked every dataset for a split named train
    # whatever it called its own. Unset now, so each follows its own -- which makes an
    # unflagged GSM8K run test where it was train. No recorded run used the default:
    # the MATH-500 runs all passed --split test. code-reviewer found the change.
    parser.add_argument(
        "--split", default=None,
        help="what this dataset calls it, and left to the dataset when unset: test "
             "or validation for the multiple-choice sets, public or a tier for "
             "jevbench, and one fixed name each for agieval and gpqa-diamond, which "
             "refuse any other")
    parser.add_argument("--limit", type=int, default=30)
    add_proposal_arguments(parser)
    parser.add_argument("--propose-tokens", type=int, default=32)
    parser.add_argument("--budget", type=int, default=8192, help="slow-mode thinking cap")
    parser.add_argument(
        "--readout", choices=sorted(READOUTS), default=DEFAULT_READOUT,
        help="how the fast mode reads its answer from the constructed candidates. The "
             "default scores each candidate's own text; 'coded' scores single-token "
             "letters, which is what this script's earlier numbers measure, so it is a "
             "parameter of those runs rather than a property of the script")
    parser.add_argument("--layers", default="-1,16,24")
    parser.add_argument("--output", default="open-three-4b.json")
    parser.add_argument("--rows", help="append a row per question here as it is produced, "
                                       "so a job killed at hour twenty keeps its generation")
    parser.add_argument("--resume", action="store_true",
                        help="skip questions already in --rows, matched by id")
    parser.add_argument("--keep-slow-from",
                        help="a record whose slow column is sound: its slow_* fields are "
                             "joined on by id and the slow mode is not run again. For a run "
                             "whose fast half was wrong and whose thinking was not")
    arguments = parser.parse_args()
    check_proposal_arguments(parser, arguments)
    # launch.sh submits from a directory named after the tree, which nothing
    # writes over. This checks the other half: that the process is importing from
    # that directory rather than through the shared environment's editable
    # install, which points at the tree every sync replaces.
    check_imported_tree()

    loaded = load(arguments.dataset, split=arguments.split)
    questions = loaded.questions[: arguments.limit]
    labels = loaded.labels
    scorer = load_proposer(arguments.model, backend=arguments.backend,
                           revision=arguments.revision, dtype=arguments.dtype,
                           device_map=arguments.device_map,
                           feature_layers=parse_feature_layers(arguments.layers))

    # Expression answers are not integers and cannot be compared as strings.
    symbolic = loaded.provenance.get("answer_form") == "expression"
    state = ("Answer the question." if symbolic
             else "Answer the question with a single integer.")
    kept = {}
    if arguments.keep_slow_from:
        with open(arguments.keep_slow_from) as handle:
            source = json.load(handle)
        # Joining by id alone would glue another model's thinking onto this
        # fast pass, and nothing in the output would say so.
        differing = {key: (source.get(key), getattr(arguments, key))
                     for key in ("dataset", "model", "budget")
                     if source.get(key) is not None
                     and source.get(key) != getattr(arguments, key)}
        if differing:
            raise SystemExit(
                f"--keep-slow-from was produced by a different run and its slow column "
                f"does not belong to this one. Differing: {differing}")
        kept = {row["id"]: row for row in source["rows"] if row.get("slow_text") is not None}
        uncovered = [q.question_id for q in questions if q.question_id not in kept]
        if uncovered:
            print(f"  {len(uncovered)} of {len(questions)} questions are not in "
                  f"{arguments.keep_slow_from} and will have their slow mode run: "
                  f"{uncovered[:5]}{' ...' if len(uncovered) > 5 else ''}", flush=True)
        missing = len(source["rows"]) - len(kept)
        print(f"keeping the slow column of {len(kept)} questions from "
              f"{arguments.keep_slow_from}"
              + (f"; {missing} have no saved response and will be re-run" if missing else ""),
              flush=True)

    done = completed(arguments.rows) if arguments.resume else {}
    if done:
        print(f"resuming: {len(done)} questions already done", flush=True)
    rows = [done[question.question_id] for question in questions
            if question.question_id in done]
    enumerated = 0
    seen_distinct, coded, warned = [], 0, False
    # Asked once, before anything is recorded: whether this engine's seed fixes
    # its draw or replaces it. The running watch below only covers temperatures
    # at which a total collapse cannot be the temperature's doing, and the sweep
    # goes below them.
    probe = {}
    if arguments.seed is not None:
        stop, probe = probe_sampler(scorer, state, [q.prompt for q in questions],
                                    arguments.width, arguments.propose_tokens, arguments.seed)
        print(f"    sampler probe: {probe}", flush=True)
        if stop:
            raise SystemExit(f"stopping before the first question: {stop}")
    constraint = None
    if arguments.verified:
        kind, name = scorer.proposal_constraint()
        constraint = {"kind": kind, "parameter": name}
        print(f"    proposals are constrained by {kind} ({name})", flush=True)
    # The construction is the engine's, not this script's. It used to call
    # schema_from_beams itself, which is why the two paths could only be reached
    # through whichever script remembered to construct, and why the proposal's
    # state, the shuffle and the constrained shape had to be repeated here.
    settings = ConstructionSettings(
        state=state, width=arguments.width, max_new_tokens=arguments.propose_tokens,
        temperature=arguments.temperature, rank=arguments.rank, seed=arguments.seed,
        shuffle=arguments.shuffle, verified=arguments.verified)
    engine = SchemaDecisionEngine(scorer, model_id=arguments.model, construction=settings,
                                  readout=arguments.readout)
    stream = Rows(arguments.rows, resume=arguments.resume, config={
        "commit": commit_now(), "environment": describe_environment(), **probe,
        "dataset": arguments.dataset, "model": arguments.model,
        "backend": arguments.backend, "temperature": arguments.temperature,
        "seed": arguments.seed, "constraint": constraint,
        "readout": arguments.readout,
        "normalisation": engine.normalisation if arguments.readout == "content" else None,
        "width": arguments.width, "rank": arguments.rank, "budget": arguments.budget})
    for index, question in enumerate(questions):
        name = question.question_id
        if name in done:
            continue
        if name not in labels:
            raise KeyError(f"{name} has no gold answer; a missing one must not be "
                           f"scored as an ordinary miss")
        gold = labels[name]
        proposals = propose(scorer, state, question.prompt, arguments.width,
                            arguments.propose_tokens, arguments.temperature,
                            seed=arguments.seed)
        # The same watch the gate keeps: this path proposes too, and a rerun of
        # a fast half is exactly where an unguarded proposal would land.
        if len(rows) < 5:
            print(f"    proposals for {name}: {[text for text, _ in proposals][:10]}",
                  flush=True)
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
        constructed, notes = engine.constructed(
            state, DecisionSchema({name: Choice(instructions=question.prompt,
                                                criteria=None)}),
            proposals={name: proposals})
        note = notes[name]
        texts, reason = note["candidates"], note["reason"]
        built = constructed[name] if reason is None else None

        # The slow mode runs on every question, degenerate or not: it is a whole
        # column of the table and skipping it where the fast mode had no choice
        # would take those questions out of its denominator and report it low.
        if name in kept:
            # The thinking was sound; only the fast half was wrong. Re-scored
            # here rather than copied, so the answer comes from this commit's
            # extractor and not from whichever one wrote the record.
            written = kept[name]["slow_text"]
            thinking_tokens = kept[name]["slow_tokens"]
            truncated = kept[name].get("slow_truncated", False)
        else:
            written, thinking_tokens, truncated = scorer.solve(state, question.prompt,
                                                               arguments.budget)
        slow_answer, closed = extract_answer(written)

        row = {"id": name, "gold": gold, "beams": texts, "reason": reason,
               "gold_in_beams": any(covered_by(text, gold, symbolic) for text in texts),
               "slow_answer": slow_answer, "slow_tokens": int(thinking_tokens),
               # The whole response, not a tail: a mis-scored slow column can
               # only be found by reading what the model actually wrote.
               "slow_text": written, "slow_truncated": bool(truncated),
               "slow_reused": name in kept,
               "slow_left_its_thinking": bool(closed)}
        if reason is None:
            schema = DecisionSchema({name: built})
            # Through the engine, so the readout the run records is the readout
            # it used. Calling scorer.score here recorded content and measured
            # codes.
            (fast,) = engine.readout_scores(state, schema)
            chosen = int(np.argmax(np.asarray(fast.logits, dtype=np.float64)))
            row["fast_answer"] = texts[chosen]
            row["hidden"] = np.asarray(fast.hidden).tolist() if fast.hidden is not None else None
            row["fast_logits"] = np.asarray(fast.logits, dtype=float).tolist()
        else:
            # Nothing to read out, so the fast mode answers with the proposal it
            # would have given on its own.
            row["fast_answer"] = texts[0] if texts else None
        row["fast_correct"] = bool(row["fast_answer"] is not None
                                   and scores_as_correct(row["fast_answer"], gold, symbolic))
        row["slow_correct"] = bool(slow_answer
                                   and scores_as_correct(slow_answer, gold, symbolic))
        rows.append(row)
        stream.append(row)
        print(f"  {index + 1}/{len(questions)} {name} "
              f"fast={row['fast_correct']} slow={row['slow_correct']}"
              f"{' [' + reason + ']' if reason else ''}", flush=True)

    stream.close()

    # Every column is over all questions. Degenerate ones stay in: they are
    # questions the proposal method could not turn into a choice, which is a
    # property of the method and belongs in its numbers.
    scored = [row for row in rows if row["reason"] is None]
    fast = np.mean([row["fast_correct"] for row in rows])
    slow = np.mean([row["slow_correct"] for row in rows])
    covered = np.mean([row["gold_in_beams"] for row in rows])
    tokens = np.mean([row["slow_tokens"] for row in rows])
    print(f"\n{len(rows)} questions, {len(scored)} with a real choice, "
          f"{len(rows) - len(scored)} degenerate")
    print(f"  coverage over all questions   {covered:.3f}   <- ceiling on the fast column")
    print(f"  fast only              {fast:.3f}   0 generated beyond the beam")
    forced = sum(row["slow_truncated"] for row in rows)
    print(f"  slow only              {slow:.3f}   {tokens:.0f} thinking tokens per question")
    print(f"    made to answer at the cap   {forced}/{len(rows)} = {forced / len(rows):.3f}")
    print(f"  oracle per question    "
          f"{np.mean([r['fast_correct'] or r['slow_correct'] for r in rows]):.3f}")

    write_json_atomic(arguments.output,
                      {"commit": commit_now(),
                       "environment": describe_environment(),
                       "dataset": arguments.dataset, "model": arguments.model,
                       "answer_form": loaded.provenance.get("answer_form"),
                       "backend": arguments.backend, "rank": arguments.rank,
                       "temperature": arguments.temperature,
                       "seed": arguments.seed,
                       "readout": arguments.readout,
                       "normalisation": (engine.normalisation
                                         if arguments.readout == "content" else None),
                       "constraint": constraint, "enumerated_proposals": enumerated,

                       "width": arguments.width, "budget": arguments.budget,
                       "coverage": float(covered), "fast": float(fast),
                       "slow": float(slow),
                       "degenerate": len(rows) - len(scored),
                       "denominator": "all questions, degenerate included",
                       "rows": rows})
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
