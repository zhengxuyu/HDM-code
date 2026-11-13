"""Table 3 under the paper's own protocol, with a baseline that is not handicapped.

The published baseline is class-weighted cross-entropy. The loss ablation shows
the weighting costs it about seven points of routing AUROC, because inverse
frequency multiplies up the rarest outcome and the gain g = o3 - o2 subtracts
exactly that class. Whatever the paper ends up claiming, the comparison has to
be against cross-entropy trained properly.

Unlike scripts/ablate.py this runs the full protocol: grouped cross-validation
selects the feature set, and the held-out categories are the test set.
"""
import os
import sys

import numpy as np

from litjev._legacy.heads import CDPOConfig, normalise_objective
from litjev.data.collect import open_records
from litjev.run.streaming import write_json_atomic
from litjev.train.training import run_training

records, meta = open_records(sys.argv[1])
seeds = int(os.environ.get("SEEDS", "5"))

PLAIN_CE = CDPOConfig(policy_weight=0.0, calibration_weight=0.0,
                      anchor_weight=1.0, weighted_anchor=False)
VARIANTS = {
    "cross-entropy (weighted, as published)": ("supervised", None),
    "cross-entropy (unweighted)": ("cdpo", PLAIN_CE),
    "CDPO": ("cdpo", CDPOConfig()),
}

rows = {}
for name, (objective, config) in VARIANTS.items():
    out = []
    for seed in range(seeds):
        _, report = run_training(records, meta, objective=objective, cdpo=config, train_seed=seed)
        # `by_objective` is keyed by the canonical name, so a caller passing the
        # former one must be normalised here or this raises KeyError -- which it
        # did, silently, from the first rename until 2026-10-09.
        final = report["by_objective"][normalise_objective(objective)]
        out.append((final["auroc_gain_vs_helps"], final["auroc_fast_correct"],
                    final["ece_fast_correct"], final["brier_fast_correct"],
                    final["gain_calibration_error"]))
    rows[name] = out
    a = np.array(out)
    print(f"{name:40} routing {a[:,0].mean():.4f}±{a[:,0].std(ddof=1):.4f}  "
          f"fast {a[:,1].mean():.4f}±{a[:,1].std(ddof=1):.4f}  "
          f"ECE {a[:,2].mean():.4f}±{a[:,2].std(ddof=1):.4f}  "
          f"Brier {a[:,3].mean():.4f}±{a[:,3].std(ddof=1):.4f}  "
          f"gainECE {a[:,4].mean():.4f}±{a[:,4].std(ddof=1):.4f}", flush=True)

write_json_atomic(sys.argv[2] if len(sys.argv) > 2 else "results/table3.json",
                  {"columns": ["auroc_routing", "auroc_fast", "ece", "brier", "gain_ece"],
                   "seeds": seeds, "variants": rows})
