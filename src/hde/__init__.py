"""System-1 calibrated decisions."""

import os as _os

from hde.data.schema import Choice, DecisionSchema, Noul, Score, SystemOneRequest

__all__ = ["Choice", "DecisionSchema", "Noul", "Score", "SystemOneRequest"]

# On the cluster, `scripts/launch.sh` sets this. Every entry point imports this
# package, and only two of them called the check by hand -- 27862 ran through
# `python -c ... bench_routing`, which would not have. Checking on import covers
# whatever is run next without anyone remembering to add a line, and the stderr
# note puts the directory the code came from in the first lines of every log,
# where it can be read rather than reasoned about.
if _os.environ.get("HDE_REQUIRE_TREE"):
    from hde.run.provenance import check_imported_tree as _check_imported_tree

    _check_imported_tree(announce=True)
