"""AnyJev's own model and own adaptive rule, measured rather than modelled.

**Why this exists.** We first put AnyJev's rule beside ours by simulating it
offline from our recorded distributions, which gave exact tokens and no
seconds, and I then estimated the seconds from our own two endpoints. Two
defensible models of that estimate disagreed by 33% (0.695 against 0.926 s per
question), because the escalated items generate 507 tokens each against the
slow mode's 188 average and a count-based interpolation cannot see that. The
user's objection was correct and simpler than the modelling: their weights are
public and ungated, so the latency is measurable and estimating it was
invention.

So this runs `morriszjm/Tacit-9B` through `anyjev`'s own `Tacit.from_pretrained`
with their published defaults -- `tau=0.5`, `max_cot_share=0.2`,
`cot_window=1000` -- on the same 300 items our own run used, and records
accuracy, generated tokens and wall seconds **per item**, which is the
granularity our own harness does not have.

What this still cannot be: a controlled comparison. Their checkpoint is
self-distilled and 9B, ours is a frozen 27B with a trained head, and the two
differ in the backbone, its size and its training. The point is not to rank
them but to stop quoting a number I made up for the cost of their method.
"""
import argparse
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="morriszjm/Tacit-9B")
    parser.add_argument("--adaptive", action="store_true", default=True)
    parser.add_argument("--no-adaptive", dest="adaptive", action="store_false")
    # Their published defaults, named rather than left implicit so a run that
    # changed one is visible in its own header.
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--max-cot-share", type=float, default=0.2)
    parser.add_argument("--cot-window", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--revision",
        help="the full 40-character weights sha. **Required unless --model is a "
             "local path**, and resolved here rather than by the launcher: "
             "`Tacit.from_pretrained` takes no revision, and downloading by hash "
             "does not update the cache's `refs/main` that a bare repo name "
             "resolves through, so a launcher-side pin does not reach their "
             "loader. This script calls `snapshot_download` itself and hands "
             "over the resulting path. The repo was last modified 2026-10-06")
    parser.add_argument(
        "--resume", action="store_true",
        help="keep the rows already in --output and run only the items missing "
             "from it. At ~985s an escalated decision this is the difference "
             "between a resubmission costing minutes and costing hours")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # **Checked before the imports, because `anyjev` pulls torch.** A bad
    # invocation should cost milliseconds, not a model load -- the same ordering
    # `bench_routing` already uses for its dataset ("the dataset before the
    # model, so a split this dataset does not have costs seconds instead of a
    # 27B load"). My first version put these after `from anyjev import Tacit`,
    # so neither guard could even be exercised without the package installed.
    if args.revision and len(args.revision) != 40:
        parser.error(
            f"--revision must be a full 40-character sha, got "
            f"{len(args.revision)} characters. A short form may not resolve as a "
            f"revision, and a pin that silently does not pin is worse than none")
    if not args.revision and not Path(args.model).exists():
        parser.error(
            f"--model {args.model!r} is a repository name with no --revision, so "
            f"the weights this run used would not be recorded and could move "
            f"under it. Pass the full sha, or a local path if the weights are "
            f"already resolved")

    from anyjev import Tacit

    from hde.data.datasets import load
    from hde.run.provenance import commit_now, describe_environment

    # **The pin is resolved here, not in the launcher.** code-reviewer found the
    # RED: a launcher that calls `snapshot_download(name, revision=sha)` and then
    # passes the repo *name* to `from_pretrained` does not get pinned weights.
    # Downloading by commit hash writes `snapshots/<sha>/` and **does not update
    # `refs/main`**, which is what `from_pretrained(name)` resolves through -- so
    # offline the run either fails for a missing cache entry or silently loads
    # whatever `main` that machine last cached. That is the drift the pin exists
    # to prevent, and the repo was last modified 2026-10-06.
    #
    # `Tacit.from_pretrained` takes no `revision`, so the only way to pin it is to
    # hand it a resolved local path. Doing that here rather than in the sbatch
    # means no launcher can get it wrong.
    model = args.model
    if args.revision:
        from huggingface_hub import snapshot_download

        model = snapshot_download(args.model, revision=args.revision)
        print(f"pinned {args.model} @ {args.revision} -> {model}", flush=True)

    # The same 300 items, from our loader, so the comparison is item-matched
    # even though the backbone is not.
    found = load("banking77", seed=args.seed, limit=args.limit)
    options = found.questions[0].options
    order = sorted(options)

    tacit = Tacit.from_pretrained(
        model, adaptive=args.adaptive, tau=args.tau,
        max_cot_share=args.max_cot_share, cot_window=args.cot_window)

    header = {
        "commit": commit_now(),
        "environment": describe_environment(extra=("anyjev",)),
        "model": args.model,
        "revision": args.revision,
        # The path the weights were actually loaded from, so the artefact says
        # what ran rather than what was asked for.
        "resolved_path": model,
        "adaptive": args.adaptive,
        "tau": args.tau,
        "max_cot_share": args.max_cot_share,
        "cot_window": args.cot_window,
        "dataset": "banking77",
        "items": len(found.questions),
        "provenance": found.provenance,
        # Said in the artefact, not only in the paper: their rule and their
        # checkpoint, our items. The backbone differs and so does its training.
        "comparability": "item-matched to our run; backbone is theirs (9B "
                         "self-distilled) against ours (27B frozen + head), so "
                         "accuracy and latency are not controlled",
    }
    # **Streamed, not accumulated.** exps caught that this wrote its output only
    # after the loop -- and quoted `routing_bench.py:106-108`, which is my own
    # record of 27862 losing "four hours and ten minutes of 27B with nothing on
    # disk". I wrote the same shape again, in the slowest loop in the repo: an
    # escalated decision here takes ~985s, so this is the worst place in the
    # project for an all-or-nothing write.
    #
    # `append_row` rather than a new mechanism, because it is the idiom this tree
    # already has and it closes a truncated final line before appending -- which
    # is exactly what a wall-clock kill leaves behind.
    from slime_hdm.resume import append_row, read_rows

    out = Path(args.output)
    done = {}
    if args.resume and out.exists():
        done = {row["id"]: row for row in read_rows(out) if "id" in row}
        print(f"resuming: {len(done)} item(s) already in {out}", flush=True)
    else:
        if out.exists():
            out.unlink()
        append_row(out, {"header": header})
    # **One warm-up, discarded.** The first `decide` carries lazy
    # initialisation inside the clock, which would be charged to item one and
    # inflate the mean -- code-reviewer's condition. The median is reported
    # beside the mean for the same reason, since one slow item moves one and not
    # the other.
    warm = found.questions[0]
    began = time.perf_counter()
    tacit.decide(state=warm.prompt, question="What does the customer want?",
                 options=[options[key] for key in order])
    print(f"  warm-up {time.perf_counter() - began:.3f}s (discarded)", flush=True)

    seen = escalated_so_far = 0
    for question in found.questions:
        if question.question_id in done:
            continue
        began = time.perf_counter()
        answer = tacit.decide(state=question.prompt,
                              question="What does the customer want?",
                              options=[options[key] for key in order])
        elapsed = time.perf_counter() - began
        # **Their return is a dict**: `d["answer"]`, `d["probs"]`, `d["route"]`
        # in {"one_forward", "cot"}. I had written `answer.option` /
        # `.escalated` / `.generated_tokens` from memory and all three were
        # invented, the same fault as inventing the latency.
        #
        # **`reasoning_tokens` does exist**, on escalated results --
        # `tacit.py:386` builds `_result(..., "cot", ..., reasoning_tokens=n)`.
        # I had written that they expose no token count, from their README
        # rather than their source, and code-reviewer read the wheel and found
        # it. That is the docs-over-source fault on the line next to the one it
        # replaced. So tokens *are* measurable here, on cot rows; the key is
        # absent on `one_forward` rows, where the count is zero by
        # construction.
        append_row(out, {
            "id": question.question_id,
            "gold": options[found.labels[question.question_id]],
            "answer": answer["answer"],
            "route": answer["route"],
            "escalated": answer["route"] == "cot",
            "seconds": elapsed,
            "reasoning_tokens": int(answer.get("reasoning_tokens") or 0),
            "probs": answer.get("probs"),
        })
        seen += 1
        escalated_so_far += answer["route"] == "cot"
        # The running share, because it is what predicts whether the run fits
        # the wall clock: at ~985s an escalation and ~1.3s otherwise, the
        # schedule is set by this number and nothing else. exps asked for it
        # after projecting a timeout from it by hand.
        print(f"  {question.question_id} {elapsed:.3f}s "
              f"{'escalated' if answer['route'] == 'cot' else ''} "
              f"[{escalated_so_far}/{seen} = {escalated_so_far/seen:.1%}]",
              flush=True)

    scored = [row for row in read_rows(out) if "id" in row]
    hit = sum(1 for r in scored if r["answer"] == r["gold"])
    esc = sum(1 for r in scored if r["escalated"])
    from statistics import median

    secs = sorted(r["seconds"] for r in scored)
    toks = sum(r["reasoning_tokens"] for r in scored)
    print(f"\n{len(scored)} items  acc {hit/len(scored):.4f}  "
          f"escalated {esc/len(scored):.1%}  "
          f"{toks/len(scored):.1f} reasoning tok/q  "
          f"{sum(secs)/len(secs):.3f} s/q mean  {median(secs):.3f} median")
    # Their own counter, as a cross-check on ours rather than a substitute: if
    # the two escalation shares disagree, one of us is reading the rule wrong.
    print(f"their tacit.stats: {getattr(tacit, 'stats', None)}")
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
