import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from hde.backends.slots import SLOT_FORMAT
from hde.readout.calibration import CalibrationProfile, TemperatureCalibrator


def serve():
    import uvicorn

    from hde.api import create_app
    from hde.backends.backend import ModelSettings, TransformersScorer
    from hde.core.pipeline import SchemaDecisionEngine
    from hde.core.pipeline_utils import DEFAULT_READOUT, READOUTS

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--calibration")
    parser.add_argument("--decision-head", help="Trained head (.safetensors) for /v1/systemtwo")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.0)
    parser.add_argument(
        "--readout", choices=sorted(READOUTS), default=DEFAULT_READOUT,
        help="how the fast mode reads its answer. The default scores each option's own "
             "text; 'coded' scores single-token letter codes and is what every published "
             "number was produced under, so a head trained before this has to name it")
    parser.add_argument("--think-budget", type=int, default=0, help="0 disables default routing")
    parser.add_argument("--port", type=int, default=8000)
    # Loopback by default, because the playground has no authentication. A Slurm
    # node needs 0.0.0.0: the login node is where a tunnel starts and it cannot
    # ssh into a compute node here, so it has to reach the port over the cluster
    # network, which is how scripts/serve_sglang.sbatch has always worked.
    parser.add_argument("--host", default="127.0.0.1")
    # An in-process SGLang engine installs signal handlers and so has to be built
    # on the main thread; lazily it is built inside a request, on a worker.
    parser.add_argument("--preload", action="store_true")
    parser.add_argument(
        "--backend", choices=("transformers", "sglang"), default="transformers",
        help="sglang is far faster at the slow mode; verify it first with inference.verify")
    parser.add_argument("--sglang-batch-questions", action="store_true",
                        help="send a request's questions to the engine together")
    args = parser.parse_args()

    def factory():
        from hde.core.block import RoutingPolicy
        from hde.serving import attach_head, feature_layers_for, load_head

        profile = CalibrationProfile.load(args.calibration) if args.calibration else None
        if profile and profile.model_id != args.model:
            raise ValueError("Calibration profile model does not match serving model")
        head = load_head(args.decision_head)
        layers = feature_layers_for(head)
        if args.backend == "sglang":
            from inference import SGLangScorer, SGLangSettings

            scorer = SGLangScorer.load(SGLangSettings(
                model_id=args.model, revision=args.revision, feature_layers=layers,
                batch_questions=args.sglang_batch_questions))
            device = "cpu"
        else:
            settings = ModelSettings(
                args.model, args.revision, args.device_map, args.dtype, feature_layers=layers
            )
            scorer = TransformersScorer.load(settings)
            device = scorer.model.get_input_embeddings().weight.device
        head = attach_head(head, scorer, args.model, args.revision, device=device)
        return SchemaDecisionEngine(
            scorer,
            profile.temperature if profile else 1.0,
            args.model,
            profile is not None,
            head=head,
            readout=args.readout,
            routing=RoutingPolicy(args.lambda_, args.think_budget),
        )

    # One process owns one model; concurrent forwards are serialized in the backend.
    uvicorn.run(create_app(factory, preload=args.preload),
                host=args.host, port=args.port, workers=1)


def mmlu():
    from datasets import load_dataset

    from hde.data.mmlu import DATASET_ID, convert_rows, ten_question_batches

    parser = argparse.ArgumentParser()
    # This one keeps a default and its choices: it is MMLU-Pro alone, read straight
    # from `load_dataset`, where `None` returns every split rather than one.
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/systemone/debug")
    parser.add_argument("--output", default="mmlu-results.json")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--export-logits", help="Validation-only calibration JSONL")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.export_logits and (args.split != "validation" or args.prepare_only):
        parser.error("--export-logits requires validation inference")
    dataset = load_dataset(DATASET_ID, revision=args.revision, split=args.split)
    questions, labels, skipped = convert_rows(dataset)
    questions = questions[: args.limit]
    runs = []
    calibration_records = []
    for batch, valid_ids in ten_question_batches(questions):
        request = batch.model_dump()
        run = {"request": request, "scored_ids": valid_ids}
        if not args.prepare_only:
            wire = urllib.request.Request(
                args.url,
                data=json.dumps(request).encode(),
                headers={"Content-Type": "application/json"},
            )
            started = time.perf_counter()
            with urllib.request.urlopen(wire, timeout=1800) as response:
                envelope = json.load(response)
                result = envelope["result"]
            run.update(response=result, elapsed_seconds=time.perf_counter() - started)
            run["correct"] = sum(
                result["answers"][key]["choice"] == labels[key] for key in valid_ids
            )
        runs.append(run)
        if args.export_logits:
            for key in valid_ids:
                answer = envelope["diagnostics"]["fields"][key]
                choices = list(batch.questions[key].criteria)
                calibration_records.append(
                    {
                        "split": "validation",
                        "slot_format": SLOT_FORMAT,
                        "question_id": key,
                        "logits": answer["logits"],
                        "label": choices.index(labels[key]),
                    }
                )
    report = {
        "dataset": DATASET_ID,
        "revision": args.revision,
        "split": args.split,
        "count": len(questions),
        "skipped": skipped,
        "runs": runs,
        "labels": {q.question_id: labels[q.question_id] for q in questions},
    }
    if runs and not args.prepare_only:
        report["accuracy"] = sum(r["correct"] for r in runs) / len(questions)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    if args.export_logits:
        Path(args.export_logits).write_text(
            "".join(json.dumps(row) + "\n" for row in calibration_records)
        )
    print(f"Saved {len(questions)} questions in {len(runs)} ten-question batches: {args.output}")


def calibrate():
    parser = argparse.ArgumentParser(description="Fit temperature on held-out logits JSONL")
    parser.add_argument("input")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="calibration.json")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    if not rows:
        parser.error("Calibration input is empty")
    if any(row.get("split") != "validation" for row in rows):
        parser.error("Every calibration record must declare split=validation")
    if any(row.get("slot_format") != SLOT_FORMAT for row in rows):
        parser.error(
            "Calibration logits must use the current slot_format; re-export validation logits"
        )
    max_choices = max(len(row["logits"]) for row in rows)
    logits = np.full((len(rows), max_choices), -1e30)
    for i, row in enumerate(rows):
        if not 0 <= row["label"] < len(row["logits"]):
            parser.error("Label outside candidate range")
        logits[i, : len(row["logits"])] = row["logits"]
    profile = TemperatureCalibrator.fit(logits, np.array([r["label"] for r in rows]), args.model)
    profile.save(args.output)
    print(json.dumps({"temperature": profile.temperature, "nll": profile.nll_after}))


def collect():
    from hde.backends.backend import ModelSettings, TransformersScorer
    from hde.core.block_utils import parse_feature_layers
    from hde.data.collect import collect_batch, save_records
    from hde.data.datasets import LOADERS, describe_selection, load, select_questions
    from hde.data.mmlu import own_state_requests, ten_question_batches

    parser = argparse.ArgumentParser(description="Collect fast/slow records for the decision head")
    parser.add_argument("--dataset", choices=sorted(LOADERS), default="mmlu-pro")
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--split", default=None,
        help="what this dataset calls it, and left to the dataset when unset: test "
             "or validation for the multiple-choice sets, public or a tier for "
             "jevbench, and one fixed name each for agieval and gpqa-diamond, which "
             "refuse any other")
    parser.add_argument(
        "--slow-mode", choices=("think", "short"), default="think",
        help="what fills the slow columns. 'think' is this paper's slow mode, reasoning "
             "to the budget; 'short' is ThinkSwitcher's short mode, an empty thinking "
             "block and then an answer, which still decodes a token at a time. The two "
             "are different policies on the same questions, so the file records which ran")
    parser.add_argument(
        "--readout", choices=("coded", "content"), default="coded",
        help="how the fast pass scores candidates. 'coded' reads the distribution over "
             "single-token letter codes and is what every published number was collected "
             "under; 'content' scores each option's own text. The two are different "
             "quantities on different scales, so a head fitted on one must not be served "
             "under the other -- the records file records which was used")
    parser.add_argument(
        "--content-score", choices=("sum", "mean", "pmi"), default="sum",
        help="which normalisation the content readout's distribution is built from. "
             "Ignored under --readout coded")
    parser.add_argument("--dataset-revision", help="HF revision; AGIEval is pinned and rejects it")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Take every k-th question so a small sample spans all categories",
    )
    parser.add_argument(
        "--categories",
        help="Comma-separated groups to keep (MMLU-Pro category, AGIEval task, GPQA "
             "domain), applied after --stride/--limit so the selection matches "
             "hde-bench-routing's identical flag. **It is a cost flag as much as a "
             "selection one.** Collecting more held-out questions in the three "
             "categories the split holds out costs 1.3 h an offset with it and 5.6 h "
             "without, because the slow pass runs on every question collected and only "
             "a quarter of them are in those categories",
    )
    parser.add_argument(
        "--phase",
        type=int, default=0,
        help="which residue class the stride lands on, `questions[phase::stride]`. "
             "**This is how a disjoint second sample is asked for.** `--offset` "
             "moves inside the strided sample so that shards tile it, which means "
             "two collections differing only in offset overlap almost entirely; "
             "phases 0..stride-1 partition the split instead",
    )
    parser.add_argument(
        "--layers",
        default="-1",
        help="hidden_states indices; write --layers=-1,40,48 (the value starts with '-')",
    )
    parser.add_argument("--budget", type=int, default=512, help="Thinking tokens per question")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Questions sharing one synthetic state. Ten was hardcoded here until a 27B "
        "at budget 4096 over four layers ran 530 MiB short on an 80 GiB card. The cause "
        "is the post-generation re-forward over prompt plus thinking, which holds every "
        "layer's hidden states and full-sequence logits at once and is linear in batch "
        "times sequence length -- not fragmentation, which expandable_segments cut from "
        "4 GiB of reserved-unallocated to 40 MiB without saving a single shard. The bench "
        "path has had the flag all along.",
    )
    parser.add_argument(
        "--code-offset", type=int, default=0,
        help="rotate the answer letters against the option slots, so slot i is labelled "
             "with code i+offset. A diagnostic: the readout prefers one option over the "
             "others, and with the letters in slot order no number says whether it reads "
             "the letter or the slot. Recorded beside whatever the run produces",
    )
    parser.add_argument(
        "--unified-prompt", action="store_true",
        help="on --backend transformers, render both modes from one string, as the "
             "sglang path does, instead of from this path's own two renders. The "
             "layer ablation needs it: the unified prompt is what served heads are "
             "read under, and sglang returns only the final layer, so a sweep over "
             "layers has to run on transformers. It changes what the fast pass "
             "reads, so records collected with and without it are not comparable; "
             "the provenance of each record says which render produced it. No "
             "effect under --backend sglang, which always renders this way",
    )
    parser.add_argument(
        "--backend", choices=("transformers", "sglang"), default="transformers",
        help="which path collects the features. **It has to be the path that will "
             "serve them**: a head reads the hidden state at a readout position in "
             "a particular prompt, and the two paths render different prompts, so a "
             "head fitted on one and served on the other is off its own training "
             "distribution. sglang serves the final layer only and refuses any "
             "other, which is checked before the backbone is loaded")
    parser.add_argument(
        "--fast-only",
        action="store_true",
        help="skip the slow path. For a pass asking about the fast readout alone -- where "
             "its answers sit among the options -- which is most of the questions and none "
             "of the thinking. The records are marked and cannot be trained on",
    )
    parser.add_argument("--output", default="head-records.npz")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Rewrite the output file every k batches so a timeout keeps what was collected",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.limit <= 0 or args.offset < 0 or args.budget <= 0 or args.stride <= 0:
        parser.error("--limit, --budget and --stride must be positive, --offset non-negative")
    layers = parse_feature_layers(args.layers)
    loaded = load(args.dataset, split=args.split, revision=args.dataset_revision)
    questions, labels, categories = loaded.questions, loaded.labels, loaded.groups
    available = len(questions)
    questions = select_questions(questions, args.offset, args.stride, args.limit,
                                 phase=args.phase)
    if args.categories:
        keep = {c.strip() for c in args.categories.split(",")}
        questions = [q for q in questions if categories.get(q.question_id) in keep]
    if not questions:
        parser.error("No questions selected")
    print(describe_selection(available, len(questions), args.offset, args.stride, args.limit,
                             extra=f"fast-only={bool(args.fast_only)}"
                                   + (f", categories {args.categories}"
                                      if args.categories else "")), flush=True)
    settings = ModelSettings(
        args.model, args.revision, args.device_map, args.dtype, feature_layers=layers,
        code_offset=args.code_offset, unified_prompt=args.unified_prompt,
    )
    if args.backend == "sglang":
        from inference.sglang_backend import SGLangScorer, SGLangSettings

        # `_check_layers` refuses anything but the final layer here, before the
        # backbone is loaded, so a request for layer 24 costs a second rather
        # than an hour.
        if args.code_offset:
            # Refused rather than ignored: this engine builds its own candidate
            # codes, so a shifted offset would reach nothing and the records
            # would claim a code scheme they were not collected under.
            parser.error("--code-offset is not supported by --backend sglang")
        scorer = SGLangScorer.load(SGLangSettings(
            model_id=args.model, revision=args.revision, feature_layers=layers))
    else:
        scorer = TransformersScorer.load(settings)
    metadata = {
        "model_id": args.model,
        "revision": args.revision,
        # **Which path produced these features**, because it decides which path
        # can serve a head fitted to them: the two render different prompts, so
        # the readout position's state is a different vector for the same
        # question. Collection used to be transformers only and recorded the name
        # as a constant.
        "backend": args.backend,
        "hidden_size": scorer.hidden_size,
        "feature_layers": list(layers),
        "budget": args.budget,
        # Batch composition is not cosmetic: batched greedy generation is not
        # bit-reproducible across compositions, which is the open candidate for
        # the gap between a collection and a bench over the same questions. A
        # records file that does not say how it was batched cannot be compared
        # with one that was batched differently. Overwritten below with the
        # effective batch once the batching is known.
        "batch_size": args.batch_size,
        "offset": args.offset,
        "stride": args.stride,
        "limit": args.limit,
        # **Which groups were kept**, so a short file reads as filtered rather
        # than as truncated. Without it a 347-row collection of three categories
        # is indistinguishable from a 1,500-row run that died early, and this
        # repository has already spent an evening on a number whose population
        # nobody could recover.
        "categories": args.categories,
        "phase": args.phase,
        "available": available,
        "selected": len(questions),
        "fast_only": bool(args.fast_only),
        "code_offset": args.code_offset,
        # Which of the transformers path's two renders produced these features.
        # In the file's own metadata and not only in each field's provenance,
        # because `run_training` reads the metadata and a sweep's rows have to
        # be able to say what prompt they describe. `records-4b-27063` -- which
        # `tab:features` is measured on -- predates the flag and carries no such
        # key, which is itself the answer for it: absent means the two-render
        # path. Always true under sglang, which has only the one render.
        "unified_prompt": bool(args.unified_prompt) or args.backend == "sglang",
        **{f"dataset_{key}": value for key, value in loaded.manifest().items()},
    }
    records = []
    started = time.perf_counter()
    # A dataset whose items carry their own context cannot be batched against one
    # state: ten of them in a request would be ten unrelated documents in one
    # prompt. bench_routing has branched on this since JevBench arrived and this
    # path did not, so every JevBench collection asked both modes about documents
    # they were never shown -- 0.16 fast against the bench path's 0.76 on the same
    # items and model, which is a measurement of the bug and not of the benchmark.
    own_state = any(question.state is not None for question in questions)
    if own_state:
        batching = list(own_state_requests(questions))
    else:
        batching = list(ten_question_batches(questions, args.batch_size))
    metadata["batching"] = "own_state" if own_state else "shared_state"
    # The effective batch, not the flag. Under own_state one question goes per
    # request, so --batch-size cannot have any effect and recording the CLI
    # default here is faithful and misleading at once: two own_state collections
    # would appear to differ in a field neither of them could act on.
    metadata["batch_size"] = 1 if own_state else args.batch_size
    # Which readout produced the features, so a head fitted here can refuse to be
    # served under the other one and a table can say what its rows read.
    metadata["readout"] = args.readout
    if args.readout == "content":
        metadata["content_score"] = args.content_score
    # What filled the slow columns. A short-mode file and a thinking-mode file
    # describe different policies on the same questions, and joining them without
    # noticing is how a length baseline would be reported as ours.
    metadata["slow_mode"] = args.slow_mode
    for batches, (batch, valid_ids) in enumerate(batching, start=1):
        records.extend(collect_batch(scorer, batch, valid_ids, labels, args.budget,
                                     categories, fast_only=args.fast_only,
                                     readout=args.readout,
                                     content_score=args.content_score,
                                     slow_mode=args.slow_mode))
        print(f"{len(records)}/{len(questions)} records, {time.perf_counter() - started:.0f}s")
        if args.checkpoint_every > 0 and batches % args.checkpoint_every == 0:
            save_records(args.output, records, {**metadata, "partial": True})
    save_records(args.output, records, metadata)
    fast = sum(r.fast_correct for r in records) / len(records)
    summary = {"records": len(records), "fast_accuracy": fast}
    if args.fast_only:
        summary["slow_accuracy"] = None  # not zero: nothing was asked
        summary["fast_only"] = True
    else:
        summary["slow_accuracy"] = sum(r.slow_correct for r in records) / len(records)
    print(json.dumps(summary))


def train_head():
    from hde.data.collect import load_records
    from hde.train.training import DEFAULT_LAMBDAS, run_training

    parser = argparse.ArgumentParser(description="Train the decision head on collected records")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", default="decision-head.safetensors")
    parser.add_argument("--report", default="decision-head-report.json")
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--select-layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--probe-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0, help="Controls the category split")
    parser.add_argument("--train-seed", type=int, help="Head initialisation; defaults to --seed")
    parser.add_argument(
        "--candidate", help="Skip selection and train this candidate, e.g. layer_24_pca64"
    )
    parser.add_argument("--lambdas", default=",".join(map(str, DEFAULT_LAMBDAS)))
    parser.add_argument(
        "--objective",
        choices=["supervised", "cdpo", "pmpo"],
        default="supervised",
        help="Final-head objective; both are trained and reported, this one is saved",
    )
    args = parser.parse_args()
    if not 0 < args.holdout_fraction < 1:
        parser.error("--holdout-fraction must be in (0, 1)")
    records, metadata = load_records(args.inputs)
    head, report = run_training(
        records,
        metadata,
        args.holdout_fraction,
        args.select_layers,
        args.epochs,
        args.probe_epochs,
        args.seed,
        tuple(float(x) for x in args.lambdas.split(",")),
        objective=args.objective,
        train_seed=args.train_seed,
        force_candidate=args.candidate,
    )
    head.save(args.output)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "chosen": report["chosen"],
                "chosen_layers": report["chosen_layers"],
                "objective": report["objective"],
                "auroc_head": report["head"]["auroc_fast_correct"],
                "routing_auroc_by_objective": {
                    k: v["auroc_gain_vs_helps"] for k, v in report["by_objective"].items()
                },
                "ece_by_objective": {
                    k: v["ece_fast_correct"] for k, v in report["by_objective"].items()
                },
                "auroc_stats_probe": report["baseline_stats_probe"]["auroc_fast_correct"],
                "auroc_concentration": report["baseline_auroc_concentration"],
                "auroc_max_probability": report["baseline_auroc_max_probability"],
                "split_level": report["split_level"],
                "fast_accuracy": report["fast_accuracy"],
                "slow_accuracy": report["slow_accuracy"],
                "output": args.output,
            }
        )
    )


def datasets_manifest():
    """Print what each benchmark resolves to, so a reviewer can check a run without a GPU."""
    from hde.data.datasets import LOADERS, load

    parser = argparse.ArgumentParser(
        description="Resolve benchmarks and print their pinned sources and content digests"
    )
    parser.add_argument("--dataset", action="append", choices=sorted(LOADERS))
    parser.add_argument(
        "--split", default=None,
        help="what this dataset calls it, and left to the dataset when unset: test "
             "or validation for the multiple-choice sets, public or a tier for "
             "jevbench, and one fixed name each for agieval and gpqa-diamond, which "
             "refuse any other")
    parser.add_argument("--dataset-revision", help="HF revision; AGIEval is pinned and rejects it")
    parser.add_argument("--output")
    args = parser.parse_args()
    manifests = []
    for name in args.dataset or sorted(LOADERS):
        loaded = load(name, split=args.split, revision=args.dataset_revision)
        manifests.append(loaded.manifest())
    report = json.dumps(manifests, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(report)
    print(report, end="")


def bench_header(args, readout, commit, environment, lambdas, split=None):
    """What a rows file carries about the run that opened it.

    Named rather than left inline so that the round trip can be tested without a
    GPU: what has to hold is that the dict one process writes is accepted by the
    next and refused for a different run, and a header built inside a function that
    loads a 27B model is a header nothing checks until the cluster checks it.

    `split` is what the dataset says it loaded, which is not always what was asked
    for: two of these loaders have one split each and record their own name for it.

    `environment` is the interpreter and the package versions -- no hostname and no
    GPU -- so a resume onto another machine is **not** refused here. Refusing on the
    node would make resume useless on Slurm, where a resubmission rarely lands on
    the same machine, so every row carries the node it was measured on instead and
    the summary marks a policy whose seconds come from more than one.
    """
    return {
        "commit": commit,
        "environment": environment,
        "model": args.model, "backend": args.backend, "decision_head": args.decision_head,
        # Constant here -- this benchmark names `coded` at the engine rather than
        # inheriting a default -- and recorded anyway, so the file says which readout
        # produced it without anyone reading this source at its commit.
        "readout": readout,
        "dataset": args.dataset, "split": args.split if split is None else split,
        "budget": args.budget,
        # The parsed grid, not the string it was typed as: "0.1, 0.05" and
        # "0.1,0.05" are the same run and should resume into each other.
        "lambdas": [float(value) for value in lambdas],
        "batch_size": args.batch_size, "repeats": args.repeats,
        "offset": args.offset, "stride": args.stride, "limit": args.limit,
    }


def bench_routing():
    """Time, tokens and accuracy of fast-only, slow-only and routed policies on real batches."""
    import platform
    import socket

    import torch

    from hde.backends.backend import ModelSettings, TransformersScorer
    from hde.bench.routing_bench import bench_policies, format_table, mean_over_repeats
    from hde.core.block import DecisionHead
    from hde.core.pipeline import SchemaDecisionEngine
    from hde.data.datasets import LOADERS, describe_selection, load, select_questions
    from hde.data.mmlu import own_state_requests, ten_question_batches
    from hde.run.provenance import commit_now, describe_environment
    from hde.run.streaming import Rows, completed

    parser = argparse.ArgumentParser(description="Benchmark routing policies end to end")
    parser.add_argument("--dataset", choices=sorted(LOADERS), default="mmlu-pro")
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--decision-head",
        help="trained head (.safetensors). Without one the run measures fast-only and "
             "slow-only and no routed policy, which is how a readout with no head "
             "trained for it yet can still have its endpoints measured")
    parser.add_argument(
        "--split", default=None,
        help="what the dataset calls it, and left to the dataset when unset: test or "
             "validation for the multiple-choice sets, public or a tier name "
             "(original, easy, hard) for jevbench, and one fixed name each for agieval "
             "and gpqa-diamond, which refuse any other. The dataset is loaded before "
             "the model, so a name it does not have costs seconds rather than a 27B load")
    parser.add_argument("--dataset-revision", help="HF revision; AGIEval is pinned and rejects it")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument(
        "--categories",
        help="Comma-separated groups to keep (MMLU-Pro category, AGIEval task, GPQA domain), "
        "applied after --stride/--limit so the selection matches hde-collect",
    )
    parser.add_argument("--batch-size", type=int, default=10, help="Questions per request")
    parser.add_argument(
        "--readout", choices=("coded", "content"), default="coded",
        help="coded is the default here whatever the engine's default is, because "
             "Table 1 and everything joined to it were read out from letter codes. "
             "content scores each option's own text and needs a head trained under it")
    parser.add_argument(
        "--normalisation", choices=("sum", "mean", "pmi"), default="sum",
        help="how an option's content score is ranked; ignored under --readout coded")
    parser.add_argument("--lambdas", default="0.1,0,-0.05,-0.1")
    parser.add_argument(
        "--margin-taus", default="",
        help="comma-separated thresholds on the top-two log-probability gap, each "
             "adding one policy that escalates below it. This is AnyJev's published "
             "adaptive rule (their default is 0.5) and it fits nothing: it reads only "
             "the distribution the answering pass already produced, so it is the floor "
             "a trained head has to clear. Empty adds no such policy")
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1, help="Untimed batches to run first")
    parser.add_argument(
        "--backend", choices=("transformers", "sglang"), default="transformers",
        help="sglang is far faster at the slow mode, and routes slightly differently: "
             "the two disagree on questions whose gain sits near lambda, so never "
             "compare a number from one against a number from the other")
    parser.add_argument("--rows",
                        help="where the per-call rows go; defaults beside --output. A row "
                             "is written when its evaluate returns, so a kill loses that "
                             "one call rather than the run")
    parser.add_argument("--resume", action="store_true",
                        help="fold in the rows already in that file instead of re-running "
                             "them. Refused if the header says another run wrote it")
    parser.add_argument("--output", default="routing-bench.json")
    args = parser.parse_args()
    if args.limit <= 0 or args.stride <= 0 or args.budget <= 0 or args.repeats <= 0:
        parser.error("--limit, --stride, --budget and --repeats must be positive")
    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
    margin_taus = [float(x) for x in args.margin_taus.split(",") if x.strip()]
    if any(tau < 0 for tau in margin_taus):
        parser.error("--margin-taus are log-probability gaps and cannot be negative")
    # The dataset before the model, so that a split this dataset does not have, or a
    # selection that keeps nothing, costs seconds instead of a 27B load. 28170 paid
    # the other order's bill once already, on a different guard.
    loaded = load(args.dataset, split=args.split, revision=args.dataset_revision)
    questions, labels = loaded.questions, loaded.labels
    available = len(questions)
    questions = select_questions(questions, args.offset, args.stride, args.limit)
    if args.categories:
        keep = {c.strip() for c in args.categories.split(",")}
        questions = [q for q in questions if loaded.groups.get(q.question_id) in keep]
    if not questions:
        parser.error("No questions selected")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    # Without a head there is nothing to route with, so the run measures the two
    # endpoints and no policy between them. That is the only way to get a fast-only
    # or slow-only column under a readout no head has been trained for yet, and it
    # is honest about what it is: `lambdas` is emptied rather than ignored, so the
    # report cannot carry a routed row that was never run.
    head = DecisionHead.load(args.decision_head) if args.decision_head else None
    if head is None:
        lambdas = []
        print("no --decision-head: fast-only and slow-only only, no routed policies",
              flush=True)
    else:
        head.metadata.check_serving(args.model, args.revision)
        # **Say which prices are being run, because omitting the flag asks for
        # four.** `--lambdas` defaults to a grid, so a run meant to measure the
        # two endpoints with the readout paid quietly measured six policies;
        # 29655 was that run, and the default was only visible afterwards in the
        # report's header. Printing it costs a line and puts it in the log the
        # launcher captures.
        print(f"routed prices: {lambdas if lambdas else 'none'}"
              + (" (the --lambdas default, not asked for)"
                 if args.lambdas == parser.get_default("lambdas") else ""),
              flush=True)
    layers = head.metadata.feature_layers if head is not None else ()
    if args.backend == "sglang":
        from inference import SGLangScorer, SGLangSettings

        # SGLang returns only the last hidden layer, so a head that reads deeper
        # cannot be served here. Saying so now beats reading another layer's vector
        # as though it were the trained one.
        if head is not None and tuple(layers) != (-1,):
            parser.error(
                f"--backend sglang serves heads that read layer -1; this one reads "
                f"{list(layers)}. Train one with hde-train-head --candidate layer_-1.")
        scorer = SGLangScorer.load(SGLangSettings(
            model_id=args.model, revision=args.revision,
            feature_layers=tuple(layers), batch_questions=False))
        if head is not None:
            head.to("cpu")
    else:
        settings = ModelSettings(
            args.model,
            args.revision,
            args.device_map,
            args.dtype,
            feature_layers=tuple(layers),
        )
        scorer = TransformersScorer.load(settings)
        if head is not None:
            head.to(scorer.model.get_input_embeddings().weight.device)
    # Table 1 and everything joined to it were read out from letter codes and do not
    # reproduce under content, so `coded` is the default here whatever the engine's
    # default is. A run that wants the other readout names it, and the file records
    # which was used.
    engine = SchemaDecisionEngine(scorer, model_id=args.model, head=head,
                                  readout=args.readout, normalisation=args.normalisation)
    # A dataset whose items carry their own context cannot be batched against one
    # state: ten of them in a request would be ten unrelated documents in one prompt.
    own_state = any(question.state is not None for question in questions)
    if own_state:
        batches = list(own_state_requests(questions))
        shape = f"{len(batches)} requests of one question, each with its own state"
    else:
        batches = list(ten_question_batches(questions, args.batch_size))
        shape = f"{len(batches)} batches of up to {args.batch_size}"
    print(describe_selection(available, len(questions), args.offset, args.stride, args.limit,
                             extra=shape + (f", categories {args.categories}"
                                            if args.categories else "")),
          flush=True)
    # Always on, derived from --output rather than asked for, because the runs this
    # drives are measured in hours and a flag that can be forgotten will be. Pass
    # --rows to put it elsewhere.
    rows_path = args.rows or str(Path(args.output).with_suffix(".rows.jsonl"))
    # The split the data actually is, not the one asked for: a loader with one split
    # records its own, and a header saying `test` over `v1_1` is a header nothing
    # downstream can tell is wrong.
    header = bench_header(args, engine.readout, commit_now(), describe_environment(), lambdas,
                          split=loaded.provenance.get("split", args.split))
    node_name = (f"{socket.gethostname()}/{torch.cuda.get_device_name(0)}"
                 if torch.cuda.is_available() else socket.gethostname())
    done = completed(rows_path) if args.resume else {}
    if done:
        print(f"resuming: {len(done)} rows already written in {rows_path}", flush=True)
    rows_file = Rows(rows_path, config=header, resume=args.resume)

    synchronize = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    if args.warmup:
        # Announced too: a warmup batch costs the same as a timed one, and a run
        # that printed nothing until the first timed batch looked hung for as long
        # as the warmup took.
        # The warmup is not written: it is the same work on the same batches and a
        # resumed run would otherwise skip the first timed batches as though they
        # were done, scoring them from a pass taken before the caches were warm.
        bench_policies(engine, batches[: args.warmup], labels, lambdas, args.budget,
                       synchronize, progress=1, label="warmup", margin_taus=margin_taus)
    reports = []
    started = time.perf_counter()
    for repeat in range(args.repeats):
        reports.append(
            bench_policies(engine, batches, labels, lambdas, args.budget, synchronize,
                           progress=1, rows_file=rows_file, done=done, repeat=repeat,
                           node=node_name, margin_taus=margin_taus)
        )
        print(f"repeat {repeat + 1}/{args.repeats} done, {time.perf_counter() - started:.0f}s")
    summary = mean_over_repeats(reports) if args.repeats > 1 else reports[0]["policies"]
    result = {
        "model": args.model,
        "decision_head": args.decision_head,
        "head_metadata": None if head is None else {
            "feature_layers": list(head.metadata.feature_layers),
            "pca_dim": head.metadata.pca_dim,
            "training": head.metadata.training,
            "readout": head.metadata.readout,
        },
        "questions": len(questions),
        "backend": args.backend,
        "readout": engine.readout,
        "normalisation": engine.normalisation,
        "available": available,
        "offset": args.offset,
        "stride": args.stride,
        "limit": args.limit,
        "batch_size": 1 if own_state else args.batch_size,
        "batching": "own_state" if own_state else "shared_state",
        "categories": args.categories,
        "dataset": loaded.manifest(),
        "selected_question_ids": [q.question_id for q in questions],
        "budget": args.budget,
        "lambdas": lambdas,
        "repeats": args.repeats,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.node(),
        "summary": summary,
        "repeats_detail": reports,
    }
    rows_file.close()
    result["rows"] = rows_path
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(format_table(summary))
