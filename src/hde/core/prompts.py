"""Every prompt \\method{} sends, in one place.

Each builder returns chat messages; what goes inside them -- the state text, the
question body, the option list, the answer boundary -- is in
:mod:`hde.core.prompts_utils`.

**One prompt, two cuts.** `build_messages` is what both modes are rendered from:
one system prompt that takes no position on reasoning, and the question in the
user turn. The chat template is asked for thinking *on*, which on Qwen3 prepends
a reasoning-effort preamble to the system turn and ends the prompt with an open
`<think>`; the slow mode gets that render as it stands, and the fast mode closes
the block and reads at the boundary. They are a token prefix of each
other, so an escalation continues from the fast pass's cache rather than
prefilling the prompt again (`tests/test_shared_prefix.py` checks this against
the real template).

Rendering with thinking *off* and cutting the closed block open does not work,
though the strings line up: a slow mode continued from a bare `<think>` taken out
of the no-think render writes `\n\n</think>` and an answer -- five tokens, no
reasoning -- because that is what the no-think template always does after
`<think>`. Measured on the sixteen-arrow walk at k=2: 0 of 4 correct, nothing
written. The preamble is load-bearing and it exists only in the thinking render.
88c1a83 hit the same wall from the other side, appending `<think>` to the
thinking-off prefix and watching the model close it after one token.
"""


from hde.core.prompts_utils import (  # noqa: F401
    ANSWER_BOUNDARY,
    CONTENT_FREE_INSTRUCTIONS,
    LINE_BREAK,
    listable_options,
    question_body,
    question_suffix,
    state_text,
    unlabelled_body,
    unlabelled_suffix,
)

UNIFIED_SYSTEM = (
    "Evaluate the state using the question and labeled options. "
    "Answer with only the option code."
)


def build_messages(state, body):
    """The one prompt both modes are rendered from.

    The system text takes no position on whether to reason: saying "do not explain
    or reason aloud" or "think it through first" is what used to force two
    renders, and the thinking block already carries that distinction.
    """
    return [
        {"role": "system", "content": UNIFIED_SYSTEM},
        {"role": "user", "content": state_text(state) + "\n\n" + body},
    ]


def build_decision_messages(state, schema=None):
    """Only shared state is prefetched; question ids never affect inference."""
    return [
        {
            "role": "system",
            "content": (
                "Evaluate the state using the question and labeled options that follow. "
                "Return only the option code. Do not explain or reason aloud."
            ),
        },
        {"role": "user", "content": state_text(state)},
    ]

def build_thinking_messages(state, body):
    """Slow path: the question joins the state in the user turn and reasoning is allowed."""
    return [
        {
            "role": "system",
            "content": (
                "Evaluate the state using the question and labeled options. "
                "Think it through first, then answer with only the option code."
            ),
        },
        {"role": "user", "content": state_text(state) + "\n\n" + body},
    ]

def build_content_messages(state, schema=None):
    """The system prompt for a readout that scores option content, not codes."""
    return [
        {
            "role": "system",
            "content": (
                "Evaluate the state using the question and options that follow. "
                "Answer with one of the options exactly as it is written. Do not "
                "explain or reason aloud."
            ),
        },
        {"role": "user", "content": state_text(state)},
    ]

def build_open_thinking_messages(state, question):
    """Slow path for a question with no options: think, then write the answer out.

    `build_thinking_messages` renders a question through `question_body`, which
    writes every option into the prompt. That is right when the options came from
    the caller and both modes are answering the same enumerated question. It is
    wrong when the options were constructed by the fast mode, because the slow
    mode would then be reading the fast mode's proposals and answering from
    inside them, which is a different policy from the one the always-thinking
    column is supposed to measure.

    So this renders the question alone. What comes back is free text and has to
    be graded against the gold answer rather than read off a code.
    """
    return [
        {
            "role": "system",
            "content": (
                "Think it through, then give the final answer on its own line "
                "as: Answer: <answer>"
            ),
        },
        {"role": "user", "content": state_text(state) + "\n\n" + state_text(question)},
    ]

def build_proposal_messages(state, question, verified=False):
    """Ask for the answer itself, from a model that has no options to pick from.

    `build_decision_messages` says "Return only the option code", which is right
    when the readout is about to score enumerated codes and wrong when the point
    is to obtain the answers that would be enumerated. Rendered with it, a beam
    over a maths question returned the letters B, C and A, because that is what
    it was asked for, and coverage came out at 1.4%.

    Nothing here mentions an option or a code. What comes back is an answer, to
    be normalised and put into a candidate set by `hde.construct`.

    `verified` asks for the object the engine's schema enforces. The schema fixes
    the shape and cannot say what belongs in it, so the prompt still has to: a
    single letter is the answer this path once returned on 369 of 500 questions,
    and a schema would have accepted every one of them as a string.
    """
    if verified:
        return [
            {
                "role": "system",
                "content": (
                    'Reply with a JSON object of the form {"answer": "..."} and '
                    "nothing else. The answer is the final value itself, with no "
                    "variable name, no equals sign and no units. It is never a "
                    "letter or a label standing for an answer. Do not explain, do "
                    "not show working."
                ),
            },
            {"role": "user", "content": state_text(state) + "\n\n" + state_text(question)},
        ]
    return [
        {
            "role": "system",
            "content": (
                "Give only the final answer itself: the value, with no variable "
                "name, no equals sign and no units. Do not explain, do not show "
                "working."
            ),
        },
        {"role": "user", "content": state_text(state) + "\n\n" + state_text(question)},
    ]
