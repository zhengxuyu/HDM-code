"""Kept so existing imports resolve; the code moved into :mod:`litjev.core`.

The message builders are in :mod:`litjev.core.prompts` and what goes inside them
in :mod:`litjev.core.prompts_utils`.
"""

from litjev.core.prompts import (  # noqa: F401
    build_content_messages,
    build_decision_messages,
    build_open_thinking_messages,
    build_proposal_messages,
    build_thinking_messages,
)
from litjev.core.prompts_utils import (  # noqa: F401
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
