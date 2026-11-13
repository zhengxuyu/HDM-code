"""The coded readout against the content one, on the questions already measured.

The coded readout scores single-token letters, and on MMLU-Pro it puts its
answers 0.171 in total variation away from where the answers are, on both
backbones and at different positions. The content readout has no label for a
prior to attach to: each option is scored by its own text, teacher-forced after
the same prompt, nothing generated.

What this measures, on the same stride-8 1500 the position passes used so that
every number here joins onto those:

    accuracy      coded against content, under each of the three normalisations
    agreement     where the four readouts pick the same option, per question,
                  which is the part an accuracy table cannot show
    position      the chosen-index histogram against the gold one, in every
                  mode, because listing order still exists without labels
    cost          seconds per question for each, since content scoring is one
                  prefill plus k continuations rather than one pass

The three normalisations fail differently -- sum prefers short options, mean
prefers long ones, PMI is the only one that is not a length artefact -- so all
three are recorded per question and none is chosen here.

    uv run python scripts/content_vs_coded.py --limit 1500 --stride 8 --model ...

`--backend` picks which implementation reads the options out. The two are not
one measurement: their readout states agree in direction to cosine 0.9995 and
differ about 3% in magnitude, and a content score is a sum over several such
positions, so a transformers run and an SGLang run belong in separate rows of a
table saying which is which. The backend is recorded in the rows file and in the
summary for that reason. 28031 (4B) and 28037 (27B) are transformers runs.
"""
import argparse
import math
import time
from collections import Counter

# The same state string the published MC numbers were produced under, used for
# both readouts, so the coded column here is comparable with Table 1 rather than
# being a fourth thing to explain.
STATE = "Answer each question using its listed options."


def main():
    import numpy as np

    from hde.candidates.proposing import load_proposer
    from hde.data.datasets import LOADERS, describe_selection, load, select_questions
    from hde.data.schema import Choice, DecisionSchema
    from hde.readout.content_readout import as_record, chosen_index
    from hde.run.provenance import check_imported_tree, commit_now, describe_environment
    from hde.run.streaming import Rows, completed, write_json_atomic

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(LOADERS), default="mmlu-pro")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--backend", choices=["sglang", "transformers"],
                        default="transformers",
                        help="which implementation reads the options out. transformers is "
                             "the default here, and only here, because it produced every "
                             "content number so far (28031, 28037) and a run that changes "
                             "both the backend and the question set compares nothing. "
                             "sglang is the serving default and the one the free-form path "
                             "needs; its numbers form their own group")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"],
                        default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--split", default=None,
        help="what this dataset calls it, and left to the dataset when unset: test "
             "or validation for the multiple-choice sets, public or a tier for "
             "jevbench, and one fixed name each for agieval and gpqa-diamond, which "
             "refuse any other")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--stride", type=int, default=8,
                        help="8 with --limit 1500 is the subset the position passes used")
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--no-content-free", dest="content_free", action="store_false",
                        help="skip the PMI pass, which halves the forwards and drops a column")
    parser.add_argument("--rows", help="append a row per question as it is produced")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-coded", dest="coded", action="store_false",
                        help="skip the coded column, which is roughly half the work. For a "
                             "run asking only which content normalisation is better: sum "
                             "and mean are computed from the same pass, so the coded "
                             "baseline adds cost and no information to that question")
    parser.add_argument("--output", default="content-vs-coded.json")
    arguments = parser.parse_args()
    check_imported_tree()

    loaded = load(arguments.dataset, split=arguments.split)
    available = len(loaded.questions)
    questions = select_questions(loaded.questions, arguments.offset, arguments.stride,
                                 arguments.limit)
    if not questions:
        parser.error("No questions selected")
    print(describe_selection(available, len(questions), arguments.offset, arguments.stride,
                             arguments.limit, extra="content readout against coded"),
          flush=True)

    # Counted up front rather than refused. When this script only compared two
    # readouts on a subset, refusing was right: a benchmark whose options cannot
    # be listed one per line is one not to compare on. It now picks the default
    # normalisation for the unified path, so it has to run under that path's own
    # policy, which rewrites the break to a space in the listing and the scoring
    # alike. Twelve of MMLU-Pro's 12032 test questions carry one and none are in
    # the stride-8 1500, which is why no comparison has met them.
    rewritten_up_front = [q.question_id for q in questions
                          if any("\n" in text or "\r" in text for text in q.options.values())]
    if rewritten_up_front:
        print(f"    {len(rewritten_up_front)} of {len(questions)} questions carry a line "
              f"break in an option; it is rewritten to a space in the prompt and in the "
              f"score alike: {rewritten_up_front[:5]}", flush=True)

    # Through the shared loader, so this script cannot drift from the two that
    # already take a backend. It refuses dtype and device_map under sglang rather
    # than dropping a setting the caller thought they had made.
    scorer = load_proposer(arguments.model, backend=arguments.backend,
                           revision=arguments.revision, dtype=arguments.dtype,
                           device_map=arguments.device_map)
    labels = loaded.labels
    ways = ("sum", "mean") + (("pmi",) if arguments.content_free else ())
    if arguments.coded:
        ways = ("coded", *ways)

    done = completed(arguments.rows) if arguments.resume else {}
    rows = [done[q.question_id] for q in questions if q.question_id in done]
    stream = Rows(arguments.rows, resume=arguments.resume, config={
        "commit": commit_now(), "environment": describe_environment(),
        "dataset": arguments.dataset, "model": arguments.model,
        "backend": arguments.backend,
        "stride": arguments.stride, "limit": arguments.limit,
        "content_free": arguments.content_free})
    began = time.perf_counter()
    for index, question in enumerate(questions, start=1):
        name = question.question_id
        if name in done:
            continue
        if name not in labels:
            raise KeyError(f"{name} has no gold answer; scoring it as a miss would "
                           f"understate every column here at once")
        # Built the way `hde.mmlu` builds it, rather than asked of the record:
        # a question that arrives with options keeps them, in source order, and
        # nothing is constructed.
        body = Choice(instructions=question.prompt, criteria=question.options)
        options = list(body.choices)
        gold = labels[name]

        coded_seconds, coded = 0.0, None
        if arguments.coded:
            started = time.perf_counter()
            (field,) = scorer.score(STATE, DecisionSchema({name: body}))
            coded_seconds = time.perf_counter() - started
            coded = int(np.argmax(np.asarray(field.logits, dtype=np.float64)))

        started = time.perf_counter()
        scores, rewritten, _ = scorer.score_options(STATE, body,
                                                    content_free=arguments.content_free)
        content_seconds = time.perf_counter() - started

        picked = {"coded": coded} if arguments.coded else {}
        for way in ways:
            if way != "coded":
                picked[way] = chosen_index(scores, way)
        row = {
            "id": name, "options": len(options),
            "gold_index": options.index(gold) if gold in options else -1,
            "chosen": picked,
            "correct": {way: options[choice] == gold for way, choice in picked.items()},
            "scores": as_record(scores),
            # A prompt quietly unlike the benchmark is worth knowing about even
            # when putting the option on one line is the right thing to do.
            "rewritten_options": rewritten,
            "seconds": {"coded": coded_seconds, "content": content_seconds},
        }
        rows.append(row)
        stream.append(row)
        if index % 25 == 0 or index == len(questions):
            spent = time.perf_counter() - began
            print(f"  {index}/{len(questions)}  {spent / 60:.1f} min in, "
                  f"{spent / index * (len(questions) - index) / 60:.0f} min left", flush=True)
    stream.close()

    print(f"\n{len(rows)} questions\n")
    print(f"{'readout':>8}  {'accuracy':>8}  {'TV(chosen, gold)':>17}")
    summary, histograms = {}, {}
    width = max(row["options"] for row in rows)
    gold_share = Counter(row["gold_index"] for row in rows)
    for way in ways:
        accuracy = float(np.mean([row["correct"][way] for row in rows]))
        chosen = Counter(row["chosen"][way] for row in rows)
        drift = 0.5 * sum(abs(chosen[p] - gold_share[p]) for p in range(width)) / len(rows)
        summary[way] = {"accuracy": accuracy, "chosen_vs_gold_total_variation": drift}
        histograms[way] = [chosen[p] / len(rows) for p in range(width)]
        print(f"{way:>8}  {accuracy:>8.4f}  {drift:>17.4f}")

    # Paired, because the readouts answer the same questions: the ones they agree
    # on carry no information about which is better, and a difference of 35 in
    # 1500 looks like 2.3 points until you notice it rests on 313 disagreements.
    paired = {}
    if arguments.coded:
        print("\nagainst the coded readout, paired (McNemar):")
    baseline = "coded" if arguments.coded else "sum"
    for way in [w for w in ways if w != baseline]:
        wins = sum(row["correct"][way] and not row["correct"][baseline] for row in rows)
        losses = sum(row["correct"][baseline] and not row["correct"][way] for row in rows)
        total = wins + losses
        z = (abs(wins - losses) - 1) / math.sqrt(total) if total else float("nan")
        significance = math.erfc(z / math.sqrt(2)) if total else float("nan")
        paired[way] = {"wins": wins, "losses": losses, "net": wins - losses,
                       "p": significance}
        print(f"  {way:>6}: wins {wins:>4}, losses {losses:>4}, net {wins - losses:+5d}, "
              f"p = {significance:.4f}")

    print("\nwhere they disagree, which an accuracy table cannot show:")
    agreement = {}
    for first in ways:
        for second in ways:
            if first < second:
                same = np.mean([row["chosen"][first] == row["chosen"][second] for row in rows])
                agreement[f"{first}|{second}"] = float(same)
                print(f"  {first:>6} and {second:<6} pick the same option {same:.3f}")

    rewrote = sum(row["rewritten_options"] for row in rows)
    if rewrote:
        print(f"\n{rewrote} options across {len(rows)} questions were put on one line")
    if not arguments.coded:
        print(f"\nagainst {baseline}, paired (McNemar):")
        for way, figures in paired.items():
            print(f"  {way:>6}: wins {figures['wins']:>4}, losses {figures['losses']:>4}, "
                  f"net {figures['net']:+5d}, p = {figures['p']:.4f}")
    seconds = {"coded": float(np.mean([row["seconds"]["coded"] for row in rows])),
               "content": float(np.mean([row["seconds"]["content"] for row in rows]))}
    print(f"\nseconds per question: coded {seconds['coded']:.3f}, "
          f"content {seconds['content']:.3f} "
          f"({seconds['content'] / max(seconds['coded'], 1e-9):.1f}x)")

    write_json_atomic(arguments.output, {
        "dataset": arguments.dataset, "model": arguments.model,
        "backend": arguments.backend,
        "commit": commit_now(), "environment": describe_environment(),
        "stride": arguments.stride, "limit": arguments.limit, "questions": len(rows),
        "content_free": arguments.content_free,
        "summary": summary, "chosen_share": histograms,
        "gold_share": [gold_share[p] / len(rows) for p in range(width)],
        "agreement": agreement, "paired_against": {"baseline": baseline, **paired},
        "coded": arguments.coded,
        "seconds_per_question": seconds,
        "rewritten_options": rewrote,
        "questions_with_a_line_break": rewritten_up_front, "rows": rows})
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
