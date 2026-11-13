"""Kept so existing imports resolve; the code moved into :mod:`hde.core.block`.

The routing policy sits beside the head it reads, because the two are one
mechanism: the head produces the joint outcome and the policy thresholds the gain
it implies.
"""

from hde.core.block import (  # noqa: F401
    OUTCOME_CLASSES,
    MarginPolicy,
    MarginRouter,
    RoutingPolicy,
    escalation_gain,
    fast_confidence,
    should_escalate,
    top_two_margin,
)
