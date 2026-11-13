"""The version strings two modules must agree on, in a module that imports nothing.

A format constant is a contract between a producer and a consumer -- the slot
compiler and the head that records which slot format it was trained under, the
feature builder and the head that refuses a head built on another one. Keeping
them here is what lets the self-routing block stop importing the prompt stack for
one string, and what makes a mismatch a loud failure rather than a silent one.

Nothing in this module may import from hde.
"""

# How a question's candidate slots are compiled into a prompt.
SLOT_FORMAT = "isolated_question_codes_v2"

# What the self-routing block reads.
FEATURE_FORMAT = "readout_hidden_plus_stats_v1"
QUESTION_TYPES = ("choice", "score", "noul")
STATS_DIM = 5 + len(QUESTION_TYPES)
OUTCOME_DIM = 4

# How a reported confidence was computed.
CONFIDENCE_METHOD = "normalized_gini_concentration_v1"
HEAD_CONFIDENCE_METHOD = "decision_head_outcome_v1"
