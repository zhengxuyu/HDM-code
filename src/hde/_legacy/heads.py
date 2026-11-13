"""Kept so existing imports resolve; the code moved into :mod:`hde.core`.

The self-routing block is in :mod:`hde.core.block`, its feature and metric
helpers in :mod:`hde.core.block_utils`, and the objective that trains it in
:mod:`hde.core.objective`. New code should import from there. This module
forwards because ninety-odd files, some of them untracked launchers belonging to
other sessions on this checkout, import `hde.heads` by name.
"""

from hde.core.block import (  # noqa: F401
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
from hde.core.block_utils import auroc, coverage_accuracy_curve  # noqa: F401
from hde.core.objective import (  # noqa: F401
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
