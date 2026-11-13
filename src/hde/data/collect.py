"""Collect decision-head training data: fast readout features plus slow-path outcomes.

Gold labels are compared locally after both passes; they never enter any prompt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from hde.backends.slots import SLOT_FORMAT
from hde.core.block_utils import distribution_stats
from hde.formats import STATS_DIM
from hde.readout.scoring import calibrated_distribution

RECORD_FORMAT = "head_records_v2"
# v1 held no position columns, so a question's verdict in the main table could not
# be audited afterwards: which option the readout chose, which one was gold, and
# how close the call was were all gone by the time the record was written. v1 files
# still load -- the columns come back as absent rather than as a re-collection --
# because the 1500-question MMLU-Pro set the paper's head is trained on is one of
# them.
READABLE_FORMATS = (RECORD_FORMAT, "head_records_v1")
# **v2 also renamed the dataset provenance**, which is the half a reader notices
# first and the comment above did not mention. v1 wrote flat `dataset` and
# `split`; v2 writes the loader's whole manifest under a `dataset_` prefix --
# `dataset_source`, `dataset_split`, `dataset_name`, `dataset_revision`,
# `dataset_questions`, `dataset_groups` and a `dataset_digest_sha256` that v1 had
# no equivalent of. A reader keying on the v1 names against a v2 file gets None
# for both and reads it as a blank provenance field, which happened once;
# `record_format` is in the file and says which set to expect.
# Wider than any benchmark here; distributions are padded to it with NaN so a
# ragged set of option counts can still be one array.
MAX_OPTIONS = 16
# Always present after a load, whatever the file's format, so that a reader asks
# "is this absent" rather than "does this file have the column". -1 and NaN say
# absent; no real chosen index is negative and no real probability is NaN.
POSITION_COLUMNS = ("chosen_index", "gold_index", "option_count", "fast_probabilities")
# A slow answer that was never asked for. Distinct from a wrong one, which is the
# whole point: `slow_correct` on a missing answer is unknown and not False, and a
# head trained on the difference would be trained on a slow mode that never got
# anything right.
ABSENT = "\x00absent"


@dataclass(frozen=True)
class Record:
    question_id: str
    category: str
    question_type: str
    hidden: np.ndarray  # [layers, hidden_size]
    stats: np.ndarray  # [STATS_DIM]
    # str | int | bool, per type: an option key for `choice`, a level for
    # `score`, a boolean for `noul`. Annotating these `str` is what let
    # `fast_choice == label` compare a string against an int and a bool for a
    # year without anyone reading it twice.
    fast_choice: str | int | bool
    slow_choice: str | int | bool
    label: str | int | bool
    fast_probability: float
    slow_probability: float
    thinking_tokens: int
    # Where the answer sat, not just what it was. A readout with a position prior
    # and a readout that is reading the question produce the same accuracy column
    # and different histograms, and without these the two cannot be told apart
    # after the fact.
    chosen_index: int = -1
    gold_index: int = -1
    fast_probabilities: tuple = ()
    # The content readout's raw per-option scores, so `sum`, `mean` and `pmi`
    # are all recomputable from one collection and a wrong choice of
    # normalisation does not cost another pass over the benchmark. Empty under
    # `coded`, where there is no such thing.
    content_total: tuple = ()
    content_tokens: tuple = ()
    content_prior: tuple = ()

    @property
    def fast_correct(self):
        return self.fast_choice == self.label

    @property
    def slow_correct(self):
        if self.slow_choice == ABSENT:
            raise ValueError(
                "this record has no slow answer, so whether thinking was right is not "
                "False, it is unknown. It came from a fast-only pass, which is a "
                "diagnostic and not training data")
        return self.slow_choice == self.label


def gold_form(choice, question_type):
    """A chosen option in the form `datasets.py` writes the gold label in.

    `question.choices` is strings for all three types -- an option key, a level
    index as a string, "false"/"true" -- while the gold label is normalised per
    type: a key for `choice`, an `int` for `score`, a `bool` for `noul`. So
    `fast_choice == label` was `'3' == 3` and `'true' == True`, which are False
    forever. On JevBench that is 74 noul plus 18 score, exactly the 92 of 231
    items that scored wrong on both paths in both models and read as difficulty.
    `routing_bench._answer_key` has always done this correctly; this is the same
    rule on the collect path.
    """
    if question_type == "score":
        return int(choice)
    if question_type == "noul":
        return choice == "true"
    return choice


def choice_form(label, question_type):
    """The inverse, for looking a gold label up in `question.choices`."""
    if question_type == "score":
        return str(label)
    if question_type == "noul":
        return "true" if label else "false"
    return label


def _content_distribution(scores, by):
    """A candidate distribution from the content readout's per-option scores.

    The coded readout hands the head a softmax over candidate *logits*. The
    content readout produces a summed log-probability per option, which is a
    different quantity on a different scale, so the eight statistics the head
    reads are not comparable across the two and a head fitted under one must not
    be served under the other. `HeadMetadata.readout` is what says which, and the
    records file records it beside the features.

    `by` picks the normalisation: `sum` ranks on the summed log-probability,
    which is what the paper's readout comparison selects, `mean` divides by the
    option's token count, and `pmi` subtracts the content-free pass.
    """
    values = []
    for score in scores:
        value = {"sum": score.total, "mean": score.mean, "pmi": score.pmi}[by]
        if value is None:
            raise ValueError("pmi needs the content-free pass; collect with it enabled")
        values.append(value)
    return calibrated_distribution(np.asarray(values, dtype=np.float64),
                                   list(range(len(values))), 1.0)


def _winner(row, question):
    distribution = calibrated_distribution(row.logits, list(range(len(question.choices))), 1.0)
    return question.choices[distribution.winner_index], distribution


def _padded(values):
    """One row of a ragged set of distributions. NaN is absence, not zero."""
    row = np.full(MAX_OPTIONS, np.nan, dtype=np.float32)
    row[: min(len(values), MAX_OPTIONS)] = np.asarray(
        values[:MAX_OPTIONS], dtype=np.float32)
    return row


def collect_batch(provider, request, valid_ids, labels, budget, categories=None,
                  fast_only=False, readout="coded", content_score="sum",
                  slow_mode="think"):
    """Run the fast path and the slow path on one request; one record per scored ID.

    `slow_mode` picks what fills the slow columns. ``think`` is this paper's slow
    mode, reasoning to the budget. ``short`` is ThinkSwitcher's short mode, an
    empty thinking block and then an answer, which still decodes a token at a time
    and is therefore a length baseline rather than our fast mode. A file collected
    under one is not comparable with a file collected under the other, so the mode
    is written into the metadata and the readers check it.

    `fast_only` skips the slow path, for a pass that is asking about the fast
    readout alone -- where its answers sit among the options, which is most of the
    cost of the question and none of the thinking. The slow columns are then
    absent rather than wrong, and `save_records` marks the file so that nothing
    trains on it: a head fitted to records whose slow answer is missing would be
    fitted to a slow mode that is always wrong.
    """
    schema = request.to_schema()
    if readout == "content":
        # One call per question, because the content readout scores an option's
        # own tokens and a schema's questions do not share a candidate set. The
        # hidden state it returns is the one at the readout position, as on the
        # coded path, so the head's input is the same vector either way.
        content = {}
        for name in valid_ids:
            option_scores, _, hidden = provider.score_options(
                request.state, schema[name], content_free=(content_score == "pmi"))
            content[name] = (option_scores, hidden)
        fast = None
    else:
        fast = {row.name: row for row in provider.score(request.state, schema)}
    if slow_mode not in ("think", "short"):
        raise ValueError(f"slow_mode must be think or short; got {slow_mode!r}")
    run_slow = (provider.think if slow_mode == "think" else provider.short_cot)
    slow = ({} if fast_only else
            {row.name: row
             for row in run_slow(request.state, schema, tuple(valid_ids), budget)})
    records = []
    for name in valid_ids:
        question = schema[name]
        if readout == "content":
            option_scores, fast_hidden = content[name]
            fast_distribution = _content_distribution(option_scores, content_score)
            fast_choice = question.choices[fast_distribution.winner_index]
            raw = ([score.total for score in option_scores],
                   [float(score.tokens) for score in option_scores],
                   [float("nan") if score.prior is None else score.prior
                    for score in option_scores])
        else:
            fast_row = fast[name]
            fast_hidden = fast_row.hidden
            fast_choice, fast_distribution = _winner(fast_row, question)
            raw = ((), (), ())
        if fast_hidden is None:
            raise ValueError("Collection requires a scorer configured with feature_layers")
        if fast_only:
            slow_choice, slow_probability, thinking_tokens = ABSENT, float("nan"), -1
        else:
            slow_row = slow[name]
            slow_choice, slow_distribution = _winner(slow_row, question)
            slow_probability = slow_distribution.winner_probability
            thinking_tokens = slow_row.generated_tokens
        records.append(
            Record(
                question_id=name,
                category=(categories or {}).get(name, ""),
                question_type=question.type,
                hidden=np.asarray(fast_hidden, dtype=np.float16),
                stats=distribution_stats(fast_distribution.probabilities, question.type),
                fast_choice=gold_form(fast_choice, question.type),
                slow_choice=(slow_choice if slow_choice == ABSENT
                             else gold_form(slow_choice, question.type)),
                label=labels[name],
                fast_probability=fast_distribution.winner_probability,
                slow_probability=slow_probability,
                thinking_tokens=thinking_tokens,
                chosen_index=int(fast_distribution.winner_index),
                gold_index=(question.choices.index(choice_form(labels[name], question.type))
                            if choice_form(labels[name], question.type)
                            in question.choices else -1),
                fast_probabilities=tuple(
                    float(value) for value in fast_distribution.probabilities),
                content_total=tuple(raw[0]),
                content_tokens=tuple(raw[1]),
                content_prior=tuple(raw[2]),
            )
        )
    return records


def save_records(path, records, metadata):
    if not records:
        raise ValueError("No records to save")
    fast_only = any(record.slow_choice == ABSENT for record in records)
    if fast_only and not all(record.slow_choice == ABSENT for record in records):
        raise ValueError(
            "some of these records have a slow answer and some do not, so nothing "
            "downstream can tell which half a slow column describes")
    metadata = {**metadata, "fast_only": fast_only}
    hidden = np.stack([record.hidden for record in records])
    if hidden.ndim != 3:
        raise ValueError("hidden must be [N, layers, hidden_size]")
    payload = {
        "hidden": hidden,
        "stats": np.stack([record.stats for record in records]).astype(np.float32),
        "fast_correct": np.array([record.fast_correct for record in records]),
        "has_slow": np.full(len(records), not fast_only, dtype=bool),
        # `slow_correct` only when there is one. Seven scripts np.load these files
        # directly rather than through load_records, and read this column without
        # asking whether it means anything: a column of False would tell each of
        # them that thinking was never right, and one of them would fit to it. A
        # missing column stops them at the line that reads it.
        **({} if fast_only else
           {"slow_correct": np.array([record.slow_correct for record in records])}),
        "fast_probability": np.array([record.fast_probability for record in records]),
        "thinking_tokens": np.array([record.thinking_tokens for record in records]),
        "chosen_index": np.array([record.chosen_index for record in records]),
        "gold_index": np.array([record.gold_index for record in records]),
        "option_count": np.array([len(record.fast_probabilities) for record in records]),
        "fast_probabilities": np.stack(
            [_padded(record.fast_probabilities) for record in records]),
        # Absent under the coded readout, where an option has no such score.
        "content_total": np.stack(
            [_padded(record.content_total) for record in records]),
        "content_tokens": np.stack(
            [_padded(record.content_tokens) for record in records]),
        "content_prior": np.stack(
            [_padded(record.content_prior) for record in records]),
        "question_id": np.array([record.question_id for record in records]),
        "category": np.array([record.category for record in records]),
        "question_type": np.array([record.question_type for record in records]),
        "metadata": np.array(
            json.dumps(
                {
                    **metadata,
                    "record_format": RECORD_FORMAT,
                    "slot_format": SLOT_FORMAT,
                    "stats_dim": STATS_DIM,
                    "count": len(records),
                }
            )
        ),
    }
    np.savez_compressed(path, **payload)


def refuse_partial(path, metadata, allow_partial=False):
    """Raise if `metadata` came from a checkpoint rather than a finished collection.

    **The guard in `load_records` protected one reader of ten.** Nine scripts open
    a records file with `np.load` directly, because each wants one file's columns
    rather than a merge of several, and each then reads `metadata` on the next
    line -- `free_baselines.py`, which produces the fitted-nothing table, and
    `table3.py` among them. So the check has to sit where the reads happen.

    `hde-collect` rewrites its output every `--checkpoint-every` batches, five
    by default, marking it `partial: True` so a timeout keeps what was collected.
    A reader that ignores the flag fits on the questions the job reached and
    reports as though it held the whole selection.

    **And it can be a biased selection rather than merely a smaller one**, which is
    the part worth refusing over. `select_questions` keeps the dataset's order and
    these datasets are grouped -- MMLU-Pro by category, JevBench by task -- so a
    stopped collection holds a prefix of that grouping rather than a sample of it,
    and **it is biased by whichever groups it does not reach**.

    Measured on `records-sglang-29662.npz`, the complete 1,500 at `B=4096`: its
    first 900 give an at-cap share of 0.347 against 0.361 over the whole file, with
    0.383 in the remainder, so that prefix *understates*. The cause is where the
    expensive categories sit, which is both ends -- law is second at 0.761 and
    engineering is last at 0.718, with the cheap ones between them. Over the file
    as a whole, tokens do not trend with position at all (Spearman $+0.037$). So
    the sign is not a property of the dataset either, only of which groups a given
    prefix omits.

    **The magnitude is the budget's.** The same questions in the same order at
    `B=1024` are flat -- at-cap 0.902, 0.902, 0.904 by thirds -- because a cap that
    binds on 90.3% leaves almost no variance to vary. A prefix is therefore
    unbiased on some collections and badly biased on others, and neither its
    direction nor its size can be predicted from the dataset and the budget without
    measuring that file. Which is the argument for refusing the flag rather than
    reporting alongside it: there is no correction to apply.
    """
    if metadata.get("partial") and not allow_partial:
        raise ValueError(
            f"{path} is a checkpoint, not a finished collection: it holds "
            f"{metadata.get('count')} records where the run asked for "
            f"{metadata.get('limit')}, and the job it came from did not reach the "
            f"end. Fitting on it is a smaller selection reported as the whole one. "
            f"Pass allow_partial=True to use the salvage deliberately, or re-run "
            f"the collection")
    return metadata


def open_records(path, allow_partial=False):
    """`np.load` and the metadata, with the checkpoint flag checked between them.

    The pair every direct reader builds by hand on the next two lines, so that
    adding a guard does not mean adding it nine times.
    """
    data = np.load(path, allow_pickle=True)
    metadata = json.loads(str(data["metadata"]))
    refuse_partial(path, metadata, allow_partial=allow_partial)
    return data, metadata


def load_records(paths, allow_fast_only=False, allow_partial=False):
    """Concatenate record files; metadata must agree on model, layers and formats.

    A fast-only file is refused unless the caller asks for it, and carries no
    `slow_correct` column at all, so a reader that goes straight to `np.load`
    stops at the line that wants it. Refusing here as well means a diagnostic
    collection cannot become a head by being handed to the wrong command.

    Fast-only and ordinary files are not merged even when both are asked for:
    concatenating them would produce one slow column describing half the rows.

    **A checkpoint is refused on the same grounds.** `hde-collect` rewrites its
    output every `--checkpoint-every` batches, five by default, marking it
    `partial: True` so that a timeout keeps what was collected. Nothing read that
    key. A collection killed at 900 of 1500 therefore left a file that loaded,
    trained and reported as though it held the whole selection, and the only
    trace was a `count` that nobody compares with the `limit` recorded beside it.
    A caller wanting the salvage asks for it, and gets to say so in writing.
    """
    parts, metadata = [], None
    for path in paths:
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["metadata"]))
        if meta.get("record_format") not in READABLE_FORMATS or meta.get("slot_format") != SLOT_FORMAT:
            raise ValueError(f"{path}: records use an incompatible format; re-collect")
        if meta.get("fast_only") and not allow_fast_only:
            raise ValueError(
                f"{path} came from a fast-only pass: its slow column is absent, not "
                f"False, and training or scoring against it would fit a slow mode that "
                f"never answered. Use it for the position diagnostic, or collect with "
                f"the slow path on")
        refuse_partial(path, meta, allow_partial=allow_partial)
        if metadata is not None and bool(meta.get("fast_only")) != bool(
                metadata.get("fast_only")):
            raise ValueError(
                f"{path} and the files before it disagree about whether the slow path "
                f"ran, and concatenating them would give one slow column describing "
                f"half the rows")
        keys = ("model_id", "revision", "feature_layers", "hidden_size")
        if metadata is None:
            metadata = meta
        elif any(meta.get(key) != metadata.get(key) for key in keys):
            raise ValueError(f"{path}: records were collected with a different model or layers")
        parts.append({key: data[key] for key in data.files if key != "metadata"})
    if not parts:
        raise ValueError("At least one record file is required")
    # A v1 file has no position columns. They come back as absent -- -1 and NaN --
    # rather than forcing a re-collection of records a trained head depends on, and
    # rather than dropping the columns for everyone when one old file is in the mix.
    keys = {key for part in parts for key in part} | set(POSITION_COLUMNS) | {"has_slow"}
    for part in parts:
        count = len(part["question_id"])
        for key in keys - set(part):
            if key == "fast_probabilities":
                part[key] = np.full((count, MAX_OPTIONS), np.nan, dtype=np.float32)
            elif key == "has_slow":
                # Not absent: a file written before this column existed is one the
                # slow path ran for, and -1 here would be a truthy int standing in
                # for a boolean nobody would think to check the dtype of.
                part[key] = np.ones(count, dtype=bool)
            else:
                part[key] = np.full(count, -1, dtype=np.int64)
    merged = {key: np.concatenate([part[key] for part in parts]) for key in sorted(keys)}
    return merged, metadata
