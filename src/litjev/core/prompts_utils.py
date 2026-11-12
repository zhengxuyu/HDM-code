"""What goes inside a prompt: state text, question bodies, options, the boundary.

Split from :mod:`litjev.core.prompts` so that file holds the message builders
alone. Both readout paths read the answer at :data:`ANSWER_BOUNDARY`, which is why
it lives here rather than in either builder.
"""

import json
import re


def state_text(state):
    return (
        state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, allow_nan=False)
    )

ANSWER_BOUNDARY = "\nAnswer:"

def question_body(question, codes):
    """The question text shared by the fast readout and the slow thinking path."""
    options = [
        {"code": codes[i], "option": key, "description": description}
        for i, (key, description) in enumerate(
            zip(question.choices, question.descriptions, strict=True)
        )
    ]
    body = {"type": question.type, "instructions": question.instructions, "options": options}
    return "Question: " + json.dumps(body, ensure_ascii=False, allow_nan=False)

LINE_BREAK = re.compile(r"[\r\n]+")

def listable_options(question):
    """`(texts, rewritten)`: the option texts as one line each, and how many moved.

    A line break inside an option would be read as the start of the next one, so
    the prompt would show more options than are being scored. Refusing was the
    first answer and is the wrong one here: once every question goes through this
    readout, a refusal is a question that cannot be answered at all, and the
    MMLU-Pro converter keeps a source newline wherever the benchmark has one.

    So the break becomes a space -- in the listing and in the scoring alike,
    which is the whole point, since the two disagreeing is the fault being
    avoided -- and the count goes in the run's record. A prompt quietly unlike
    the benchmark is worth knowing about even when it is the right thing to do.
    """
    texts, rewritten = [], 0
    for text in question.descriptions:
        flat = LINE_BREAK.sub(" ", text) if isinstance(text, str) else text
        rewritten += flat != text
        texts.append(flat)
    return texts, rewritten

def unlabelled_body(question):
    """The question and its options, with nothing standing in for an option.

    The labelled form gives every option a single-token code and the readout
    scores those codes, which is why the readout has a preference over positions
    at all: a code is a token, a token has a prior, and the prior is attached to
    whatever sits in that slot. Here the options are a list of their own content
    and the readout scores the content instead, so there is no label to prefer.

    The order still exists, so listing order can still bias the answer. That is
    measured rather than assumed away.

    Plain lines rather than JSON, because the readout scores the option's own
    text and the prompt has to show that same text. Through `json.dumps` the
    model reads `"\\frac{1}{2}"`, escaped and in quotes, while the score is for
    `\frac{1}{2}` -- so the system prompt asks for one of the options exactly as
    written and the thing being scored is not what was written.

    One line per option, so an option carrying a newline would be read as two and
    the listing would disagree with the scoring about how many options there are.
    `listable_options` rewrites those breaks to spaces, identically here and in
    the scoring, and counts how many it rewrote.
    """
    listed, _ = listable_options(question)
    return f"Question: {question.instructions}\nOptions:\n" + "\n".join(listed)

def unlabelled_suffix(question):
    """Ends at the colon, with no trailing space.

    A space at the end of a prefix is its own token, and BPE attaches the same
    space to the word that follows -- " Oxygen" is one token -- so the joint
    encoding of prompt and option does not begin with the prompt's own tokens and
    the option cannot be separated from it. Measured on gpt2: with the trailing
    space, none of seven ordinary options is separable; without it, all seven.
    The space belongs to the option, and `score_options` puts it there.
    """
    return unlabelled_body(question) + ANSWER_BOUNDARY

CONTENT_FREE_INSTRUCTIONS = "N/A"

def question_suffix(question, codes):
    return question_body(question, codes) + ANSWER_BOUNDARY

# The Qwen3 template, asked for thinking, ends the prompt with an OPEN block --
# and prepends a reasoning-effort preamble to the system turn. Both matter.
OPEN_THINKING = "<think>\n"
# Closing it reproduces exactly what the template emits with thinking off.
CLOSE_THINKING = "\n\n</think>\n\n"


def shared_prefix(rendered):
    """The part of a thinking render both modes share, ending at ``<think>``.

    The cut drops only the newline after ``<think>``, so the slow mode's prompt is
    the render itself and the fast mode's is the same tokens with the block
    closed. Cutting here rather than one character later is what makes them a
    token prefix of each other: the newline run after ``<think>`` tokenises
    differently depending on what follows it.

    **Render with thinking on, not off.** Asked for thinking off the template
    emits an already-closed empty ``<think></think>`` and no preamble, and a slow
    mode continued from a bare ``<think>`` cut out of *that* does not reason: it
    emits ``\\n\\n</think>`` and an answer, five tokens, because a bare
    ``<think>`` is what the no-think template always follows that way. Measured on
    the sixteen-arrow walk at k=2, where it answered 0 of 4 having written
    nothing. The preamble is load-bearing, and it is only in the thinking render.
    """
    if not rendered.endswith(OPEN_THINKING):
        raise ValueError(
            "this chat template does not end with an open thinking block, so the "
            "fast and slow prompts cannot be cut from one render; check the "
            "template, and that it was rendered with thinking enabled")
    return rendered[: -1]


def open_thinking(prefix):
    """The slow mode's prompt: the shared prefix with the block still open."""
    return prefix + "\n"


def close_thinking(prefix):
    """The fast mode's prompt: the shared prefix with the block closed and empty."""
    return prefix + CLOSE_THINKING
