"""Build a candidate set for a question that did not arrive with one.

The request schema makes a candidate set optional, and until now nothing made
the optional case work: the fast-thinking mode reads its answer from enumerated
codes, so a question with nothing to enumerate had nothing to read and every
benchmark in the paper was multiple choice.

A model that cannot be given candidates can still propose them. With thinking
switched off, a beam of width `width` over a short answer returns that many
distinct completions in one decode, and those become the candidate set the
readout scores. What follows is unchanged: the codes are assigned the same way,
the readout sits at the same boundary, the head reads the same two inputs, and
the routing rule is the same rule. Only the origin of the candidates differs.

Two properties of the construction are the whole design and are easy to lose.

**The slow-thinking mode never sees the constructed candidates.** It renders its
own prompt with thinking switched on and answers in free form, exactly as it
does when no construction has happened, so the labels, the deployed policy and
the always-thinking baseline remain one configuration. Anything that puts the
beams into the slow prompt turns the slow mode into a different policy and the
comparison with it into a different comparison.

**The fast mode is capped by its own beam and the slow mode is not.** A gold
answer absent from the beams cannot be read out, whatever the head decides. That
makes coverage the number the approach stands on, which is why it is measured
before anything is built on it.

The cost is a real one: the fast path is a short bounded generation rather than
no generation at all. It is bounded by `max_new_tokens` and the beams run in
parallel, so it stays far below a reasoning trace, but the claim for this path
is that the decision is free, not that the answer is.
"""

import re
import unicodedata

from hde.data.schema import Choice

DEGENERATE = "degenerate_single_candidate"
EMPTY = "no_candidates_generated"


# A proposal that arrives already numbered. The readout labels every candidate by
# slot, so one that carries its own index is labelled twice -- "A. 42" is written
# into the prompt as "B: A. 42" -- and the two indices disagree as soon as the
# candidates are shuffled. The whitespace is required, so 1.5 is a number and not
# an enumerated 5.
ENUMERATOR = re.compile(r"^\(?(?:[A-Ja-j]|\d{1,2})\)?[.:)]\s+")


def strip_enumerator(text):
    """`("A. 42", True)` becomes `("42", True)`; anything else is returned as is."""
    stripped = ENUMERATOR.sub("", text, count=1)
    return stripped, stripped != text


def normalise(text):
    """Strip the differences beams produce that are not differences in answer.

    Beams separate on whitespace, on a trailing full stop, on unicode minus
    against hyphen. Left alone these fill a candidate set with one answer written
    several ways, which reads to the head as a question with several live options
    and to the readout as a choice that is not one.

    A leading enumerator goes too, for a different reason: the readout adds its
    own label by slot, so a proposal that brought one arrives carrying two, and
    they point at different candidates the moment the order is shuffled.
    """
    text = unicodedata.normalize("NFKC", text).strip()
    text = text.strip("`$ \t\n")
    text = ENUMERATOR.sub("", text, count=1)
    text = re.sub(r"[\s]+", " ", text)
    text = text.rstrip(".")
    text = text.replace("−", "-")
    if re.fullmatch(r"-?\d+\.0+", text):  # 42.0 and 42 are one answer
        text = text.split(".")[0]
    if re.fullmatch(r"-?0\d*", text):  # and so are 042 and 42, which AIME invites
        digits = text.lstrip("-").lstrip("0") or "0"
        text = ("-" if text.startswith("-") and digits != "0" else "") + digits
    return text.strip()


LIKELIHOOD = "likelihood"
FREQUENCY = "frequency"


def candidates_from_beams(proposals, limit=52, rank=LIKELIHOOD):
    """Distinct answers, ranked, with the rank the caller asked for.

    `proposals` is an iterable of (text, score) where higher scores are better and
    duplicates are kept: a sampler returns the same answer several times and how
    often it does is itself a ranking.

    Which rank is used decides what the first candidate means, and the first
    candidate is what the readout is compared against.

        likelihood  the model's own score, summed token logprobs for a sample or
                    the sequence score for a beam. This is what a proposer that
                    returned one answer would have returned.
        frequency   how many of the proposals agree on this answer, ties broken
                    by likelihood. Only a sampler produces this, and it is the
                    self-consistency signal a beam cannot give.

    Neither rank reaches the decision head, which reads the readout's own
    distribution over the constructed codes. What the rank fixes is the order the
    options are written in, and which candidate counts as the one the proposer
    would have answered on its own.
    """
    if rank not in (LIKELIHOOD, FREQUENCY):
        raise ValueError(f"unknown rank {rank!r}; use {LIKELIHOOD!r} or {FREQUENCY!r}")
    best, counts = {}, {}
    for text, score in proposals:
        key = normalise(text)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
        if key not in best or score > best[key]:
            best[key] = float(score)
    if rank == LIKELIHOOD:
        order = sorted(best, key=lambda key: -best[key])
    else:
        order = sorted(best, key=lambda key: (-counts[key], -best[key]))
    return [(key, best[key]) for key in order[:limit]]


def schema_from_beams(proposals, instructions, limit=52, rank=LIKELIHOOD, shuffle=None,
                      question_id=""):
    """A `Choice` over the distinct beams, or a reason there is no choice to make.

    Returns `(question, texts, reason)`, `texts` in the order the options are
    written into the prompt. `reason` is set when the construction produced
    nothing to choose between, and the caller answers with candidate one rather
    than reading anything out: a single candidate gives the readout no decision, a
    point-mass distribution, and statistics that describe nothing. Those records
    are marked rather than dropped silently, because a benchmark where the beams
    collapse is a fact about the benchmark.

    `shuffle` is the run's seed and `question_id` the question's; together they
    fix a permutation of the ranked order. Both are needed: a seed without the
    question gives every question of a width the same permutation, which relabels
    the slots rather than shuffling them. Without it the options are written in rank order, so the readout's
    preference for one slot is also a preference for one rank: it took the
    runner-up on all but one of 478 questions, and no number could say whether
    that was a judgement about the answers or about the slot. Permuting breaks
    that: the slot a biased readout favours now holds a candidate of no particular
    rank, so a fast column built this way is not systematically the second-best
    answer, and what is left of the preference shows up as noise rather than as a
    result. Seeded per question by the caller, so a run is repeatable.

    Use `ranked_positions` to get back which presented slot holds which rank.
    """
    kept = candidates_from_beams(proposals, limit=limit, rank=rank)
    if not kept:
        return None, [], EMPTY
    if len(kept) == 1:
        return None, [text for text, _ in kept], DEGENERATE
    texts = [text for text, _ in kept]
    if shuffle is not None:
        texts = shuffled(texts, shuffle, question_id)
    # Keyed by the answer itself, not by its position. The key is what comes back
    # as the answer, so an index key would make the system's output "2" where the
    # answer is "4" -- an index standing for an answer, which is the one thing the
    # no-thinking output must never be. The candidates are distinct by
    # construction, so they are usable as keys.
    return (Choice(instructions=instructions, criteria={text: text for text in texts}),
            texts, None)


def shuffled(texts, seed, question_id):
    """The same permutation every time for the same run, question and candidates.

    The question id is in the seed, not only the run's. Without it every question
    of a given width gets the same permutation, which is a constant relabelling
    of the slots rather than a shuffle: a readout that favours slot 1 would still
    be favouring one rank, just a different one from before.

    Seeded from the question rather than from a counter, so that resuming a run
    presents a question the way it was presented before it died.
    """
    import random

    order = list(texts)
    random.Random(f"{seed}:{question_id}:{len(order)}").shuffle(order)
    return order


def ranked_positions(presented, ranked):
    """Which presented slot holds which rank, so a shuffled run can still be read.

    `first_right` has to stay a statement about the top-ranked candidate and not
    about whichever one was written first, or shuffling would quietly redefine
    the baseline the readout is compared against.
    """
    where = {}
    for rank, text in enumerate(ranked):
        for slot, shown in enumerate(presented):
            if shown == text and slot not in where.values():
                where[rank] = slot
                break
    return where


def correct(candidate, gold, symbolic=True):
    """Strict: does this answer match the gold one, for a score that goes in a table.

    `equivalent` tries the symbolic comparison two ways and accepts either, which
    leans towards calling answers equal. That lean is safe for coverage, where a
    missed match understates a ceiling; it is not safe for accuracy, where it
    would inflate every column. This takes the stricter reading: the answer has
    to survive the comparison with maths delimiters, which is the form the gold
    answers are written in.
    """
    if normalise(candidate) == normalise(gold):
        return True
    if not symbolic:
        return False
    parse, verify = _math_verify()
    try:
        return bool(verify(parse(f"$${gold}$$"), parse(f"$${candidate}$$")))
    except Exception:  # noqa: BLE001  an unparseable answer is not a match
        return False


def _math_verify():
    try:
        from math_verify import parse, verify
    except ImportError as error:
        raise RuntimeError(
            "this benchmark's answers are expressions and comparing them needs "
            "math_verify: uv pip install math-verify. Comparing them as strings "
            "would undercount coverage and fail a gate that should have passed."
        ) from error
    return parse, verify


def equivalent(candidate, gold, symbolic=True):
    """Whether two written answers are the same answer.

    An integer benchmark needs nothing more than `normalise`: 042, 42 and 42.0
    are one answer and a string comparison settles it. A benchmark whose answers
    are expressions does not, because \\frac{3}{4}, 3/4 and 0.75 are one answer
    and no amount of whitespace stripping says so. `symbolic` routes to
    `math_verify` for those.

    Two calls, not one, and either is accepted. The parser reads `\\frac34`
    only inside maths delimiters and reads `2*pi` against `2\\pi` only outside
    them, so one call alone is wrong in one direction or the other. Taking either
    biases towards calling answers equal, which is the safe direction here: this
    decides coverage, and undercounting coverage would fail a gate that should
    have passed.
    """
    if normalise(candidate) == normalise(gold):
        return True
    if not symbolic:
        return False
    parse, verify = _math_verify()
    for wrap in ("{}", "$${}$$"):
        try:
            if verify(parse(wrap.format(gold)), parse(wrap.format(candidate))):
                return True
        except Exception:  # noqa: BLE001  an unparseable answer is not a match
            continue
    return False


THINK_CLOSE = "</think>"
# The format the model was asked for, at the start of a line. Anchored because
# "This answer uses the formula" is a sentence about an answer and not one.
ANSWER_LINE = re.compile(r"^\s*(?:final\s+)?answer\s*(?:is)?\s*[:=]\s*(.+?)\s*$",
                         re.IGNORECASE | re.MULTILINE)
# A model that ignored the format still tends to say it in words.
ANSWER_PHRASE = re.compile(r"\b(?:final\s+)?answer\s+is\s+(.+?)\s*$",
                           re.IGNORECASE | re.MULTILINE)


def after_thinking(text):
    """What the model wrote after it stopped reasoning, and whether it stopped.

    A trace contains answers the model then abandoned -- "maybe 5, no wait" --
    and reading one out of the whole response picks an abandoned one. Everything
    before the final close tag is working, not answer.

    A response with no close tag never stopped reasoning, which is what the cap
    looks like from here, and the caller is told rather than handed the last
    line of a thought.
    """
    if THINK_CLOSE in text:
        return text.rsplit(THINK_CLOSE, 1)[1], True
    return text, False


def _boxed(text):
    """The first \\boxed{...}, matched by counting braces rather than by a pattern.

    A regular expression can only allow a fixed depth of nesting, and one level
    was not enough: `\\boxed{\\frac{\\sqrt{3}}{2}}` is three deep and fell
    through to the last line of the response.
    """
    marker = "\\boxed{"
    index = text.find(marker)
    while index != -1:
        depth, position = 1, index + len(marker)
        while position < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[position], 0)
            position += 1
        if not depth:
            return text[index + len(marker):position - 1]
        index = text.find(marker, index + len(marker))
    return None


# Where a sentence resumes after the answer it stated. A model told to end with
# `Answer: x` writes `Answer: 12. This answer uses the formula` on one line as
# readily as on two, and a newline was doing the work in the first version of
# this.
#
# Two patterns rather than one with IGNORECASE, because the flag would have
# applied to the capital as well and "before a capital" would have meant
# "before any letter", cutting `Answer: 3. x` at the decimal point.
SENTENCE_BREAK = re.compile(r"\.\s+(?=[A-Z])")
CLAUSE_BREAK = re.compile(
    r"[,]?\s+(?:because|since|so|which|where|therefore|hence|as well)\b", re.IGNORECASE)


def _depths(text):
    """Brace depth before each position, so a cut can be refused inside one."""
    depth, out = 0, []
    for character in text:
        out.append(depth)
        depth += {"{": 1, "}": -1}.get(character, 0)
    out.append(depth)
    return out


def _cut_point(answer):
    """Where the sentence continues, at brace depth zero, or None.

    Depth matters because MATH-500 answers are often `\\text{...}` and the words
    inside are prose: `\\text{Sunday as well}` was being cut at " as well" and
    came out as `\\text{Sunday`, which is not an answer and does not parse.
    """
    depths = _depths(answer)
    candidates = [match.start() for pattern in (SENTENCE_BREAK, CLAUSE_BREAK)
                  for match in pattern.finditer(answer)]
    outside = [start for start in candidates if start > 0 and depths[start] == 0]
    return min(outside) if outside else None


def _tidy(answer):
    """The answer, without the sentence it was written inside.

    Two things get stripped. The delimiters and punctuation around it, from both
    ends: `rstrip` alone left the opening dollar of `$\\frac{1}{2}$.`, which the
    comparison then wrapped again into `$$$...$$`. And whatever continues after
    it, cut only at brace depth zero.

    The cut is deliberately conservative. It fires on a sentence boundary or on
    a word that starts a clause, and leaves anything else alone, because an
    answer can legitimately contain a full stop (a decimal) or a comma (a pair).
    """
    answer = answer.strip()
    cut = _cut_point(answer)
    if cut is not None:
        answer = answer[:cut]
    return answer.strip().strip("$ .").strip()


def balanced(answer):
    """Whether the braces in an answer close. A cut in the wrong place shows up here."""
    depth = 0
    for character in answer:
        depth += {"{": 1, "}": -1}.get(character, 0)
        if depth < 0:
            return False
    return depth == 0


def extract_answer(text):
    """The answer a free-form response gave, and whether it ever stopped reasoning.

    Returns `(answer, closed)`. `closed` is False when the response never left
    its reasoning, which happens when the budget runs out mid-thought; what is
    returned then comes from the trace and is worth what a trace is.

    Everything is read from the first statement of an answer rather than the
    last. A model told to end with `Answer: x` often writes a sentence after it
    -- "Answer: 12. This answer uses the formula" -- and the last match is then
    the sentence. The forced continuation after a truncated thought invites
    exactly that, because it appends the marker and lets the model keep writing.
    """
    if not text:
        return "", False
    body, closed = after_thinking(text)
    boxed = _boxed(body)
    if boxed is not None:
        return _tidy(boxed), closed
    for pattern in (ANSWER_LINE, ANSWER_PHRASE):
        found = pattern.search(body)
        if found:
            return _tidy(found.group(1)), closed
    lines = [line.strip() for line in body.strip().splitlines() if line.strip()]
    return (_tidy(lines[-1]) if lines else ""), closed
