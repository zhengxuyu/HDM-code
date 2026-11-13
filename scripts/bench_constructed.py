"""Fast-only, slow-only and routed policies on a typed-request dataset.

hde-bench-routing reads a multiple-choice benchmark and batches ten questions
into a synthetic request. This reads requests as they are actually written: a
shared state, several questions of mixed type, and the candidate sets the caller
declared. Each request is one batch, because its questions genuinely arrive
together and share one prefill.

That also makes it the only evaluation here in which a request's questions can
be correlated: they are about the same state, so a state the backbone reads
badly makes all of them hard at once.
"""
import argparse
import json
import time

import torch

from hde._legacy.decision import SchemaDecisionEngine
from hde._legacy.heads import DecisionHead
from hde.backends.backend import ModelSettings, TransformersScorer
from hde.bench.routing_bench import bench_policies, format_table
from hde.data.schema import SystemOneRequest
from hde.run.streaming import write_json_atomic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--decision-head", required=True)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--lambdas", default="0.1,0.05,0,-0.05,-0.1")
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.dataset) as handle:
        payload = json.load(handle)
    requests = [SystemOneRequest.model_validate(r) for r in payload["requests"]]
    labels = payload["labels"]
    batches = [(request, list(request.questions)) for request in requests]
    types = {}
    for request in requests:
        for qid, question in request.questions.items():
            types.setdefault(question.type, []).append(qid)
    print(f"{len(requests)} requests, {sum(len(b[1]) for b in batches)} questions: "
          + ", ".join(f"{t} {len(ids)}" for t, ids in sorted(types.items())), flush=True)
    missing = [q for _, ids in batches for q in ids if q not in labels]
    if missing:
        parser.error(f"no gold label for {missing}")

    head = DecisionHead.load(args.decision_head)
    head.metadata.check_serving(args.model, args.revision)
    settings = ModelSettings(args.model, args.revision, args.device_map, args.dtype,
                             feature_layers=head.metadata.feature_layers)
    scorer = TransformersScorer.load(settings)
    head.to(scorer.model.get_input_embeddings().weight.device)
    engine = SchemaDecisionEngine(scorer, model_id=args.model, head=head,
                                  # Table 1 and everything joined to it were read out from
                                  # letter codes; they do not reproduce under content.
                                  readout="coded")

    lambdas = [float(x) for x in args.lambdas.split(",")]
    synchronize = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    bench_policies(engine, batches, labels, lambdas, args.budget, synchronize)  # warmup, untimed
    started = time.perf_counter()
    report = bench_policies(engine, batches, labels, lambdas, args.budget, synchronize)
    report["seconds_total"] = time.perf_counter() - started
    report["dataset"] = args.dataset
    report["questions_by_type"] = {t: len(ids) for t, ids in types.items()}
    report["model"] = args.model
    report["decision_head"] = args.decision_head
    report["head_training"] = head.metadata.training
    print(format_table(report["policies"]))

    # Per type as well as overall: ten questions cannot separate a type effect
    # from a request effect, but the per-type rows say which types were answered
    # at all, which is what this run is for.
    # Per type as well as overall: ten questions cannot separate a type effect
    # from a request effect, but the per-type rows say which types were answered
    # at all, which is what this run is for.
    kind_of = {qid: kind for kind, ids in types.items() for qid in ids}
    by_type = {}
    for row in report["batches"]:
        for qid, correct in row["correct_by_question"].items():
            bucket = by_type.setdefault(row["policy"], {}).setdefault(kind_of[qid], [0, 0])
            bucket[0] += int(correct)
            bucket[1] += 1
    if by_type:
        kinds = sorted(types)
        print("\nper type (correct / asked)")
        print(f"{'policy':22}" + "".join(f"{k:>14}" for k in kinds))
        for name, counts in by_type.items():
            cells = []
            for kind in kinds:
                hit, total = counts.get(kind, (0, 0))
                cells.append(f"{hit}/{total}" if total else "-")
            print(f"{name:22}" + "".join(f"{c:>14}" for c in cells))
        report["by_type"] = by_type
    write_json_atomic(args.output, report)


if __name__ == "__main__":
    main()
