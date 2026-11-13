"""Benchmark sources, pinned so a run can be reproduced and checked.

Every loader returns the same shape and records what it read. The digest covers
the converted questions, not the raw download, so it changes if either the
source or our conversion changes. Reruns that print a different digest are not
comparable to the numbers in the paper.
"""

import hashlib
import json
import urllib.request
from dataclasses import dataclass, field

from hde.data.mmlu import LABELS, McqQuestion, convert_rows

MMLU_PRO = "TIGER-Lab/MMLU-Pro"
JEVBENCH_REPO = "fstandhartinger/jevbench"
# jevbench::v1.2. Pinned for the same reason AGIEval is: the published numbers are
# the numbers for this content, and the repository is under active development.
JEVBENCH_COMMIT = "bb05a335bc809e61b20c0f745d25499a82b326fc"
# The three tiers that are published. The manifest also lists heldout, router and
# judge splits, which are not in `datasets/public/` and are not fetched.
JEVBENCH_TIERS = ("original", "easy", "hard")

AGIEVAL_REPO = "ruixiangcui/AGIEval"
AGIEVAL_COMMIT = "84ab72d94318290aad2e4ec820d535a95a1f7552"
GPQA_DATASET = "Idavidrein/gpqa"
# The one split each of these has. Named here because the loader refuses anything
# else and the provenance records it, and those two have to agree.
GPQA_SPLIT = "gpqa_diamond/train"
AGIEVAL_SPLIT = "v1_1"

# The English multiple-choice tasks. gaokao-mathcloze and math are cloze, and the
# Chinese tasks are a different language transfer question; both are out of scope.
AGIEVAL_EN_MCQ = (
    "aqua-rat",
    "logiqa-en",
    "lsat-ar",
    "lsat-lr",
    "lsat-rc",
    "sat-en",
    "sat-math",
)


def _canonical(value):
    """A state as one deterministic string, whether it arrived as text or as JSON.

    Only for the digest. The request keeps the structure, because a benchmark whose
    context is an object is describing a prompt this repository does not get to
    flatten.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class LoadedDataset:
    """Questions plus everything needed to say which questions these were."""

    name: str
    questions: list
    labels: dict
    groups: dict
    provenance: dict = field(default_factory=dict)

    def digest(self):
        """Stable over the converted content, independent of load order."""
        payload = sorted(
            [
                question.question_id,
                question.prompt,
                *(f"{key}={question.options[key]}" for key in sorted(question.options)),
                # The state when a question carries one, and nothing when it does not,
                # so every digest recorded before typed datasets existed is unchanged.
                *([f"state={_canonical(question.state)}"]
                  if getattr(question, "state", None) else []),
                # Stringified because a typed benchmark's gold is a key, an int level or
                # a boolean, and sorting rows that mix them compares an int with a str.
                # Every earlier label is already a string, so no recorded digest moves.
                str(self.labels[question.question_id]),
                self.groups.get(question.question_id, ""),
            ]
            for question in self.questions
        )
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()

    def manifest(self):
        return {
            **self.provenance,
            "name": self.name,
            "questions": len(self.questions),
            "groups": sorted({group for group in self.groups.values() if group}),
            "digest_sha256": self.digest(),
        }


def _strip_option_prefix(text, label):
    """AGIEval ships options as "(A)text"; the prefix is redundant with the key."""
    for prefix in (f"({label})", f"{label})", f"({label}) ", f"{label}. "):
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text.strip()


def select_questions(questions, offset=0, stride=1, limit=None, phase=0):
    """The subset a run takes: every `stride`-th question, then `limit` from `offset`.

    Stride first, then offset. `questions[offset::stride]` counts the offset in
    unstrided space, so an array task's offset lands in a different residue class
    from task zero's: three tasks at 500 questions each and stride 8 covered 625
    of the 1500 a single task takes, duplicated 375 of them, and reported nothing
    -- a collection that looks complete and is a different, smaller sample.
    Counting the offset inside the strided sample makes the shards tile it.

    Identical at stride 1, and identical at offset 0, which between them cover
    every record collected so far.

    `phase` is the one thing `offset` deliberately stopped expressing: which
    residue class the stride lands on, `questions[phase::stride]`. Shards tile one
    sample and therefore must not change it, so `offset` moves inside the strided
    sample -- which means **two collections differing only in `offset` overlap
    almost entirely**, 1,499 of 1,500 at stride 8. A second sample disjoint from
    the first needs a different residue, and nothing could ask for one. Phases
    `0..stride-1` partition the split, so `phase=1` at stride 8 is 1,504 questions
    none of which the `phase=0` collection holds. Default 0, so every existing
    call is unchanged.
    """
    strided = questions[phase::stride]
    return strided[offset:] if limit is None else strided[offset:offset + limit]


def describe_selection(available, selected, offset, stride, limit, extra=""):
    """One line saying what a run is about to work on, printed before it starts.

    A run that quietly took a different subset from the one it was asked for said
    nothing about it: `--limit 300` against a 198-question set produced 198, and
    the log never mentioned either number. Both belong in the first seconds of it,
    where a wrong subset costs a cancelled job rather than a table.
    """
    asked = f"offset {offset}, stride {stride}, limit {limit}"
    # "the set is smaller" is one reason among several -- bench_routing filters by
    # category after selecting -- so the line says the count is short and leaves
    # the cause to the `extra`, which names the filter when there is one.
    short = (" -- fewer than asked for" if limit is not None and selected < limit else "")
    return (f"selected {selected} of {available} questions ({asked}){short}"
            + (f", {extra}" if extra else ""))


def load_mmlu_pro(split="test", revision=None, **_):
    revision = revision or "main"
    from datasets import load_dataset

    rows = load_dataset(MMLU_PRO, revision=revision, split=split)
    questions, labels, skipped = convert_rows(rows)
    groups = {str(row.get("question_id", "")): str(row.get("category", "")) for row in rows}
    return LoadedDataset(
        name="mmlu-pro",
        questions=questions,
        labels=labels,
        groups={q.question_id: groups.get(q.question_id, "") for q in questions},
        provenance={
            "source": MMLU_PRO,
            "revision": revision,
            "split": split,
            "skipped": len(skipped),
        },
    )


def _fetch(url):
    with urllib.request.urlopen(url) as response:
        return response.read()


def load_jevbench(split="public", commit=JEVBENCH_COMMIT, fetch=_fetch, revision=None, **_):
    """JevBench v1 public items: 231 typed decisions, all three question types.

    The first benchmark here whose items are typed as the schema types them --
    139 `choice`, 74 `noul`, 18 `score` -- and whose questions carry their own
    context. Both matter. The paper's evidence is from `choice` alone, which is a
    limitation it states; and every other dataset here is multiple choice with no
    context, so its state is a sentence this repository wrote rather than anything
    the benchmark provided.

    `split` is `public` for all three tiers, or one tier by name. The tiers are not
    difficulty labels in the usual sense: `easy` items name their answer in the
    text, `hard` items are long policy and multi-hop documents authored with a
    rationale, and `original` is the authored v1 set.

    Each file is checked against the sha256 in the repository's own manifest, and a
    mismatch is refused rather than loaded: the point of a pinned commit is that
    the content is fixed, and a file that does not hash to what the manifest says
    is not the content the commit names.

    35 of the 231 items carry a JSON object as their context rather than a string,
    which the request schema already allows and which is kept structured: a
    benchmark whose state is an object is describing a prompt, and flattening it to
    text here would be editing the benchmark.

    The gold label is normalised to the form the benchmark compares against, which
    differs per type: a key for `choice`, an integer level for `score`, and a
    boolean for `noul` -- where the file writes "yes"/"no" and the schema's
    candidates are "true"/"false".
    """
    if revision is not None and revision != commit:
        raise ValueError(
            f"JevBench is pinned to commit {commit}; --dataset-revision cannot move it. "
            "Edit JEVBENCH_COMMIT if you mean to re-pin it, and expect a new digest.")
    tiers = JEVBENCH_TIERS if split in ("public", "all", None) else (split,)
    unknown = [tier for tier in tiers if tier not in JEVBENCH_TIERS]
    if unknown:
        raise ValueError(
            f"JevBench has no published tier {unknown[0]!r}; use one of "
            f"{list(JEVBENCH_TIERS)} or 'public' for all three")

    base = f"https://raw.githubusercontent.com/{JEVBENCH_REPO}/{commit}"
    manifest = json.loads(fetch(f"{base}/datasets/manifest.json").decode("utf-8"))
    expected = {entry["name"]: entry for entry in manifest.get("splits", [])}

    questions, labels, groups, skipped, digests = [], {}, {}, [], {}
    for tier in tiers:
        raw = fetch(f"{base}/datasets/public/{tier}.jsonl")
        found = hashlib.sha256(raw).hexdigest()
        digests[tier] = found
        wanted = (expected.get(tier) or {}).get("sha256")
        if wanted and wanted != found:
            raise ValueError(
                f"JevBench {tier}.jsonl hashes to {found}, and the manifest at this commit "
                f"says {wanted}. Refusing rather than measuring something the manifest does "
                f"not describe")
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            question = row.get("question") or {}
            kind, criteria = question.get("type"), question.get("criteria")
            question_id = str(row.get("id", ""))
            gold, options = None, None
            if kind == "choice" and isinstance(criteria, dict):
                options = {str(key): str(value) for key, value in criteria.items()}
                gold = row.get("expected") if row.get("expected") in options else None
            elif kind == "noul" and isinstance(criteria, dict):
                options = {key: str(criteria[key]) for key in ("false", "true") if key in criteria}
                # The file says yes or no; the schema's candidates are true and false,
                # and `_answer_key` returns a boolean for this type.
                gold = {"yes": True, "no": False}.get(row.get("expected"))
            elif kind == "score" and isinstance(criteria, list):
                options = {str(index): str(level) for index, level in enumerate(criteria)}
                # An ordered scale, so the gold is a level index and the comparison is
                # against the most probable level rather than a key.
                try:
                    level = int(row.get("expected"))
                except (TypeError, ValueError):
                    level = None
                gold = level if level is not None and 0 <= level < len(criteria) else None
            state = row.get("state")
            if not question_id or not options or len(options) < 2 or gold is None \
                    or not isinstance(question.get("instructions"), str) \
                    or not isinstance(state, str | dict | list):
                skipped.append({"id": question_id, "reason": f"unusable {kind!r} item"})
                continue
            questions.append(
                McqQuestion(question_id=question_id, prompt=question["instructions"],
                            options=options, state=state, question_type=kind))
            labels[question_id] = gold
            groups[question_id] = str(row.get("family") or tier)

    return LoadedDataset(
        name="jevbench", questions=questions, labels=labels, groups=groups,
        provenance={
            "source": f"github.com/{JEVBENCH_REPO}", "revision": commit,
            "protocol": manifest.get("protocol"), "split": split,
            "tiers": list(tiers), "tier_sha256": digests, "license": "MIT",
            "skipped": len(skipped), "skipped_detail": skipped[:20],
            "types": {kind: sum(1 for q in questions if q.question_type == kind)
                      for kind in ("choice", "score", "noul")},
        },
    )


def load_agieval(tasks=AGIEVAL_EN_MCQ, commit=AGIEVAL_COMMIT, fetch=_fetch, revision=None,
                 split=None, **_):
    """AGIEval v1.1, read from the repository at a pinned commit.

    The commit fixes both the content and the line order, so question ids built
    from the line index are stable across runs and machines. It is not a knob:
    the reported numbers are the numbers for this commit, so a caller asking for
    a different one is told rather than quietly given this one.
    """
    if revision is not None and revision != commit:
        raise ValueError(
            f"AGIEval is pinned to commit {commit}; --dataset-revision cannot move it. "
            "Edit AGIEVAL_COMMIT if you mean to re-pin the benchmark, and expect a new digest."
        )
    # It used to swallow `split` in `**_`, so `--split test` was accepted here and
    # recorded in the run's header while the data was v1_1. The data was right and the
    # header was not, which is worse than an error: nothing downstream could tell.
    if split not in (None, AGIEVAL_SPLIT):
        raise ValueError(
            f"AGIEval has one split here, {AGIEVAL_SPLIT!r}; it was asked for {split!r}")
    questions, labels, groups, skipped = [], {}, {}, []
    for task in tasks:
        url = f"https://raw.githubusercontent.com/{AGIEVAL_REPO}/{commit}/data/v1_1/{task}.jsonl"
        lines = fetch(url).decode("utf-8").splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            row = json.loads(line)
            options, label = row.get("options"), row.get("label")
            question_id = f"{task}-{index:04d}"
            body = " ".join(part for part in (row.get("passage"), row.get("question")) if part)
            valid = (
                isinstance(options, list)
                and 2 <= len(options) <= len(LABELS)
                and all(isinstance(text, str) and text.strip() for text in options)
                and isinstance(label, str)
                and label in LABELS[: len(options)]
                and body.strip()
            )
            if not valid:
                skipped.append({"id": question_id, "reason": "invalid MCQ or missing label"})
                continue
            keys = LABELS[: len(options)]
            questions.append(
                McqQuestion(
                    question_id=question_id,
                    prompt=body.strip(),
                    options={
                        key: _strip_option_prefix(text, key) for key, text in zip(keys, options)
                    },
                )
            )
            labels[question_id] = label
            groups[question_id] = task
    return LoadedDataset(
        name="agieval",
        questions=questions,
        labels=labels,
        groups=groups,
        provenance={
            "source": f"github.com/{AGIEVAL_REPO}",
            "revision": commit,
            "split": AGIEVAL_SPLIT,
            "tasks": list(tasks),
            "skipped": len(skipped),
        },
    )


def load_gpqa_diamond(revision=None, seed=0, split=None, **_):
    """GPQA Diamond.

    The four options are shuffled per question with a fixed seed, because the
    source stores the correct answer in its own column and an unshuffled order
    would put it first every time.

    One split, and asking for another is refused rather than ignored: `split` used
    to be swallowed, so a run could record `test` in its header while reading
    `gpqa_diamond/train`.
    """
    if split not in (None, GPQA_SPLIT):
        raise ValueError(
            f"GPQA-Diamond has one split here, {GPQA_SPLIT!r}; it was asked for {split!r}")
    import random

    from datasets import load_dataset

    revision = revision or "main"
    try:
        rows = load_dataset(GPQA_DATASET, "gpqa_diamond", revision=revision, split="train")
    except Exception as error:  # gated on the Hub; say so instead of a stack trace
        raise RuntimeError(
            f"{GPQA_DATASET} is gated. Accept the terms at "
            f"https://huggingface.co/datasets/{GPQA_DATASET} and run `huggingface-cli login`. "
            "The other benchmarks need no account."
        ) from error
    questions, labels, groups, skipped = [], {}, {}, []
    for index, row in enumerate(rows):
        question_id = f"gpqa-diamond-{index:04d}"
        body = row.get("Question")
        correct = row.get("Correct Answer")
        wrong = [row.get(f"Incorrect Answer {n}") for n in (1, 2, 3)]
        texts = [correct, *wrong]
        if not (
            isinstance(body, str)
            and body.strip()
            and all(isinstance(text, str) and text.strip() for text in texts)
        ):
            skipped.append({"id": question_id, "reason": "missing question or options"})
            continue
        order = list(range(len(texts)))
        random.Random(f"{seed}:{question_id}").shuffle(order)
        keys = LABELS[: len(texts)]
        questions.append(
            McqQuestion(
                question_id=question_id,
                prompt=body.strip(),
                options={key: texts[pos].strip() for key, pos in zip(keys, order)},
            )
        )
        labels[question_id] = keys[order.index(0)]
        groups[question_id] = str(row.get("High-level domain") or "")
    return LoadedDataset(
        name="gpqa-diamond",
        questions=questions,
        labels=labels,
        groups=groups,
        provenance={
            "source": GPQA_DATASET,
            "revision": revision,
            "split": GPQA_SPLIT,
            "option_shuffle_seed": seed,
            "skipped": len(skipped),
        },
    )


GSM8K_DATASET = "openai/gsm8k"
MATH_500_DATASET = "HuggingFaceH4/MATH-500"
AIME_2025_DATASET = "yentinglin/aime_2025"


@dataclass(frozen=True)
class OpenQuestion:
    """A question whose answer is not drawn from a set the caller can write down.

    The record every other loader returns carries its options, because the fast
    mode reads its answer from them. There are none here: an AIME answer is an
    integer between 0 and 999, and enumerating a thousand of them exceeds the
    single-token codes a tokenizer supplies. The fast mode reaches this question
    only by proposing candidates itself (`hde.construct`), which is what the
    dataset is loaded to test.
    """

    question_id: str
    prompt: str


def load_aime_2025(revision=None, split="train", **_):
    """AIME 2025, both papers: 30 problems, integer answers, no options.

    Answers are compared after normalisation rather than by an exact string
    match, because a model writes 042, 42 and $42$ for the same integer.
    """
    from datasets import load_dataset

    revision = revision or "main"
    rows = load_dataset(AIME_2025_DATASET, revision=revision, split=split)
    questions, labels, groups, skipped = [], {}, {}, []
    for index, row in enumerate(rows):
        question_id = f"aime-2025-{index:03d}"
        body, answer = row.get("problem"), row.get("answer")
        if not (isinstance(body, str) and body.strip()) or answer is None:
            skipped.append({"id": question_id, "reason": "missing problem or answer"})
            continue
        questions.append(OpenQuestion(question_id=question_id, prompt=body.strip()))
        labels[question_id] = str(answer).strip()
        groups[question_id] = str(row.get("year") or "2025")
    return LoadedDataset(
        name="aime-2025",
        questions=questions,
        labels=labels,
        groups=groups,
        provenance={"source": AIME_2025_DATASET, "revision": revision, "split": split,
                    "answer_form": "integer", "skipped": len(skipped)},
    )


def load_gsm8k(revision=None, split="test", limit=None, **_):
    """GSM8K, grade-school word problems with a single integer answer.

    The answer sits after a `####` marker at the end of the worked solution, and
    the commas a writer puts in four-digit numbers are removed so that 1,000 and
    1000 do not count as different answers.

    It is here for the same reason AIME is: to measure a proposal method's
    coverage on questions whose answers were never enumerated. Thirty AIME
    problems put a 95% interval of roughly fourteen points on a coverage near
    80%, which straddles any threshold worth setting; a few hundred of these
    bring that to about four.
    """
    from datasets import load_dataset

    revision = revision or "main"
    rows = load_dataset(GSM8K_DATASET, "main", revision=revision, split=split)
    questions, labels, groups, skipped = [], {}, {}, []
    for index, row in enumerate(rows):
        if limit is not None and len(questions) >= limit:
            break
        question_id = f"gsm8k-{index:05d}"
        body, worked = row.get("question"), row.get("answer")
        if not (isinstance(body, str) and body.strip()
                and isinstance(worked, str) and "####" in worked):
            skipped.append({"id": question_id, "reason": "missing question or answer marker"})
            continue
        questions.append(OpenQuestion(question_id=question_id, prompt=body.strip()))
        labels[question_id] = worked.rsplit("####", 1)[1].strip().replace(",", "")
        groups[question_id] = "gsm8k"
    return LoadedDataset(
        name="gsm8k",
        questions=questions,
        labels=labels,
        groups=groups,
        provenance={"source": GSM8K_DATASET, "revision": revision, "split": split,
                    "answer_form": "integer", "skipped": len(skipped)},
    )


def load_math_500(revision=None, split="test", limit=None, **_):
    """MATH-500, the 500-problem subset of MATH, answers as expressions.

    Levels one to five across seven subjects, which is the span that makes it
    useful here: a benchmark everything is right on and one everything is wrong
    on both leave the head nothing to separate.

    Its answers are not integers. `\\frac{3}{4}`, `3/4` and `0.75` are one
    answer, so anything measuring correctness or coverage on this set has to
    compare them with `hde.candidates.construct.equivalent` and not as strings.
    """
    from datasets import load_dataset

    revision = revision or "main"
    rows = load_dataset(MATH_500_DATASET, revision=revision, split=split)
    questions, labels, groups, skipped = [], {}, {}, []
    for index, row in enumerate(rows):
        if limit is not None and len(questions) >= limit:
            break
        question_id = str(row.get("unique_id") or f"math500-{index:04d}")
        body, answer = row.get("problem"), row.get("answer")
        if not (isinstance(body, str) and body.strip()) or answer in (None, ""):
            skipped.append({"id": question_id, "reason": "missing problem or answer"})
            continue
        questions.append(OpenQuestion(question_id=question_id, prompt=body.strip()))
        labels[question_id] = str(answer).strip()
        groups[question_id] = f"{row.get('subject', '')}|level{row.get('level', '')}"
    return LoadedDataset(
        name="math-500",
        questions=questions,
        labels=labels,
        groups=groups,
        provenance={"source": MATH_500_DATASET, "revision": revision, "split": split,
                    "answer_form": "expression", "skipped": len(skipped)},
    )


# `PolyAI/banking77` is script-based and the installed `datasets` refuses those
# ("Dataset scripts are no longer supported"). The legacy mirror has the Hub's
# own parquet conversion, which is why both the name and the revision are pinned
# here rather than left to default.
BANKING77 = "legacy-datasets/banking77"
# The sha that `refs/convert/parquet` resolved to on 2026-10-07, not the branch
# name: that branch is the Hub's own conversion and the Hub can regenerate it, so
# a run recording the branch would record something that can change under it.
# code-reviewer's condition on d886f60.
BANKING77_REVISION = "54c053c5487708560dc60df4670185a2a4b27cd3"
BANKING20_CLASSES = 20
BANKING20_ITEMS = 300


def load_banking77(revision=None, seed=0, split="test", classes=BANKING20_CLASSES,
                   limit=BANKING20_ITEMS, **_):
    """BANKING77 reduced to a 20-way task, matching AnyJev's published parameters.

    Added 2026-10-07 to put our numbers beside theirs. Their results file gives
    `task: banking20`, `n: 300`, `seed: 0`, `k: 20` rotations, and reports
    accuracy, ECE and an order-flip rate on Qwen3-8B.

    **Their subset is not reproducible from their repository, and that bounds
    the comparison.** The published parameters say 20 classes and 300 items at
    seed 0; they do not say *which* 20 of the 77 intents, nor which 300 items,
    nor over what ordering the seed draws. We searched the README, the results
    file and `bench/` and the selection code is not there. So the draw below is
    ours: the 20 most frequent intents in the split, then 300 items stratified
    across them at `seed`. Both halves are recorded in the provenance so a
    reader knows which numbers are and are not item-matched.

    At 300 items and an accuracy near 0.8 the standard error of a proportion is
    `sqrt(0.8 * 0.2 / 300) = 0.023`, so two different draws of this construction
    differ by about 0.05 at two standard errors --- which is the size of the
    improvement they report (0.747 to 0.803). **A difference of that order
    between our number and theirs is therefore not interpretable**, and the
    write-up has to say so rather than rank the two.

    The 20 classes are the candidate set of every question, so this is a
    20-option `choice` with the same text for each item: the readout reads one
    token per intent name.

    **This docstring predicted that would be the coded readout's weakest shape
    -- 20 single-token codes against 20 intents whose names are what
    distinguishes them -- and gave that as a reason to run the task. 29122
    falsified it: the fast mode scores 0.917 in one forward pass, which makes it
    one of the readout's strongest shapes, not its weakest.** The prediction is
    left here rather than deleted, because a pre-registered expectation that
    comes out backwards is a result and the paper reports it as one.

    What the task is worth running for is that **thinking buys nothing here**:
    slow-only scores 0.910 against fast-only's 0.917, recovering 10 items and
    damaging 12 of 300, with overlapping Wilson intervals and a paired sign test
    at p=0.42. A wash, not a loss.

    **An earlier version of this paragraph said "thinking is a net loss, 0.980
    against 0.900, the first task in this project where that is true", from
    n=50.** Every figure in it was a prefix artefact of the index sort described
    below, and the claim did not survive the full run. A number read off a
    partial run does not belong in a docstring, where it is read as a property of
    the task.
    """
    import random
    from collections import Counter
    from string import ascii_uppercase

    from datasets import load_dataset

    revision = revision or BANKING77_REVISION
    rows = load_dataset(BANKING77, revision=revision, split=split)
    names = rows.features["label"].names

    # **A seeded draw, not "the most frequent", which selected nothing.**
    # code-reviewer caught that every one of the 77 intents has exactly 40 test
    # items, so a frequency criterion with a label-order tie-break *is* label
    # order, and I had named it for a criterion that does no work.
    #
    # Taking the first twenty label ids is not a neutral fallback either. Those
    # twenty contain **all ten** `card_*` intents in the dataset -- about to
    # expire, acceptance, arrival, delivery estimate, linking, not working,
    # payment fee charged, payment not recognised, payment wrong exchange rate,
    # swallowed. A random draw of twenty expects 2.6 of them, and over 2000
    # random draws the largest same-prefix cluster reached 7 at most and never
    # 10. So the alphabetical block is the single most confusable twenty by that
    # measure, and it would have depressed our accuracy for a reason that has
    # nothing to do with the method.
    counts = Counter(int(row["label"]) for row in rows)
    if len(counts) < classes:
        raise RuntimeError(f"{BANKING77}/{split} has {len(counts)} intents, not {classes}")
    kept = sorted(random.Random(f"banking77:intents:{seed}").sample(
        sorted(counts), classes))
    # **Not `LABELS`**, which is MMLU-Pro's ten letters: zipping twenty intents
    # against it silently produced a ten-way task, and the first draw of this
    # loader was that bug. Twenty keys stay inside A--Z, so the readout's
    # single-token letter codes cover them without reaching AA.
    #
    # Twenty options is itself past `collect.MAX_OPTIONS` (16), which pads the
    # stored probability row for *training*. Evaluation is unaffected:
    # `heads.distribution_stats` is scale-free in the option count by
    # construction -- its docstring says so and it reduces any K to five scalars
    # and a type one-hot -- so an existing head reads a twenty-way question
    # without truncation. Collecting training records here would truncate, which
    # is why this is an evaluation-only dataset until that constant moves.
    keys = ascii_uppercase[:classes]
    if len(keys) < classes:
        raise RuntimeError(f"{classes} options need more keys than A-Z supplies")
    options = {key: names[label].replace("_", " ") for key, label in zip(keys, kept)}
    key_of = {label: key for key, label in zip(keys, kept)}

    # Stratified across the kept intents so a 20-way accuracy is not dominated by
    # whichever intent the split happens to hold most of.
    by_intent = {}
    for index, row in enumerate(rows):
        label = int(row["label"])
        if label in key_of and isinstance(row.get("text"), str) and row["text"].strip():
            by_intent.setdefault(label, []).append((index, row["text"].strip()))
    rng = random.Random(f"banking77:items:{seed}")
    for pool in by_intent.values():
        rng.shuffle(pool)
    drawn = []
    while len(drawn) < limit and any(by_intent.values()):
        for label in kept:
            if by_intent.get(label) and len(drawn) < limit:
                drawn.append((label, *by_intent[label].pop()))
    # **Shuffled, not sorted by index -- sorting was a real defect.** Every
    # intent occupies a contiguous block of indices in BANKING77, so ordering
    # the draw by index put the twenty intents in twenty consecutive runs: the
    # first 160 items covered 11 intents, the last 140 covered 10, and they
    # overlapped by **one**. Two consequences, and the second is the serious one.
    #
    # Reading a partial run gave a near-disjoint subset of the task rather than a
    # sample of it -- `helps` read 0, then 1, then 10 across the prefixes of
    # 29122, and the gap read 0.080, 0.056, 0.044, 0.007. Those were four
    # different populations, not a sequence converging on one.
    #
    # And `--limit` below the full count silently selects a biased intent subset,
    # which is a defect in the loader and not only a hazard in reading it.
    #
    # Shuffling at the item seed fixes both: any prefix is now a random sample of
    # the 300, so a partial read is a small sample of the right task.
    rng.shuffle(drawn)

    questions, labels, groups = [], {}, {}
    for label, index, text in drawn:
        question_id = f"banking77-{index:05d}"
        questions.append(McqQuestion(question_id=question_id, prompt=text,
                                     options=dict(options)))
        labels[question_id] = key_of[label]
        groups[question_id] = names[label]
    return LoadedDataset(
        name="banking77",
        questions=questions,
        labels=labels,
        groups=groups,
        provenance={
            "source": BANKING77,
            "revision": revision,
            "split": split,
            "classes": classes,
            "intents": [names[label] for label in kept],
            "selection": "twenty intents drawn at seed, then a stratified draw of "
                         "items across them",
            # Theirs says `calib: 200` and we do not know what it calibrates.
            # Ours is 300 test items and nothing else: no calibration split is
            # held out, because nothing here is calibrated.
            "calibration_items": 0,
            "selection_seed": seed,
            "items": len(questions),
            # The part of their setting we could not match, recorded next to the
            # part we could, so neither is mistaken for the other.
            "matches_anyjev": ["classes", "items", "seed_value"],
            "not_matched": ["which intents", "which items", "their draw order"],
            # One draw's standard error. **The comparison against their number
            # needs the standard error of a *difference* of two independent
            # draws**, which is sqrt(2) times this, so two of those is 0.065 --
            # and that counts item sampling only, with no term for our twenty
            # intents not being theirs. code-reviewer's condition on d886f60;
            # my first version used one draw's 0.023 and drew bands too narrow
            # at both ends.
            "standard_error_at_0.8": 0.023,
            # Named for what it covers, because the manifest is what reaches the
            # paper and a bare `difference_standard_error` there would read as
            # the whole uncertainty. exps raised it via code-reviewer.
            "item_sampling_difference_se": 0.033,
            "uncertainty_not_included": "variance from our twenty intents not "
                                        "being theirs",
        },
    )


LOADERS = {
    "banking77": load_banking77,
    "gsm8k": load_gsm8k,
    "math-500": load_math_500,
    "mmlu-pro": load_mmlu_pro,
    "agieval": load_agieval,
    "gpqa-diamond": load_gpqa_diamond,
    "aime-2025": load_aime_2025,
    "jevbench": load_jevbench,
}


def load(name, **kwargs):
    """Load a dataset, leaving `split=None` to mean "whatever this one calls its own".

    Dropped rather than forwarded, because the loaders' splits are not one
    vocabulary: MMLU-Pro has validation and test, JevBench has public and three
    tiers, AGIEval and GPQA-Diamond have exactly one each and now refuse any other
    name. Every entry point used to default to `"test"` and send it to all of them,
    which is how refusing an unknown split broke the Table 2 launcher the moment it
    landed. A caller that means a split says it; a caller that does not, does not.
    """
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(LOADERS)}")
    if kwargs.get("split") is None:
        kwargs.pop("split", None)
    return LOADERS[name](**kwargs)
