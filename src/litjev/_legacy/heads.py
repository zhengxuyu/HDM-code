"""Kept so existing imports resolve; the code moved into :mod:`litjev.core`.

The self-routing block is in :mod:`litjev.core.block`, its feature and metric
helpers in :mod:`litjev.core.block_utils`, and the objective that trains it in
:mod:`litjev.core.objective`. New code should import from there. This module
forwards because ninety-odd files, some of them untracked launchers belonging to
other sessions on this checkout, import `litjev.heads` by name.
"""

from litjev.core.block import (  # noqa: F401
    FEATURE_FORMAT,
    OUTCOME_DIM,
    QUESTION_TYPES,
    STATS_DIM,
    DecisionHead,
    HeadMetadata,
    build_features,
    concentration,
    distribution_stats,
    parse_feature_layers,
)
from litjev.core.block_utils import auroc, coverage_accuracy_curve  # noqa: F401
from litjev.core.objective import (  # noqa: F401
    FORMER_NAMES,
    OBJECTIVES,
    CDPOConfig,
    PMPOConfig,
    cdpo_loss,
    normalise_objective,
    outcome_labels,
    pmpo_loss,
    train_head,
)
