"""Can PMPO be repaired by keeping a proper scoring rule underneath it?

The ablation says an unweighted cross-entropy head beats PMPO on routing
AUROC, on the calibration of the reported confidence and on the calibration of
the gain. Cross-entropy on the four-way outcome is a proper scoring rule, so
every functional of the predicted distribution inherits its calibration. PMPO
keeps that term at weight 0.25 and class-weights it, which gives up the
property, and then spends two more terms buying parts of it back.

This sweeps the obvious repair: full unweighted cross-entropy as the base, and
the policy term added on top at a range of weights. If the policy term carries
anything the likelihood does not, some weight above zero beats weight zero.
"""
import os
import sys

import numpy as np

from hde._legacy.heads import CDPOConfig, normalise_objective
from hde.data.collect import open_records
from hde.run.streaming import write_json_atomic
from hde.train.training import run_training

records, meta = open_records(sys.argv[1])
base = dict(pca_dim=64, selection_folds=4, holdout_fraction=0.25,
            force_candidate="layer_24_pca64", seed=0,
            lambdas=np.round(np.linspace(-0.8, 0.8, 321), 4).tolist())

def cfg(policy, calibration):
    """Full unweighted cross-entropy underneath; vary what is stacked on it."""
    return CDPOConfig(policy_weight=policy, calibration_weight=calibration,
                      anchor_weight=1.0, weighted_anchor=False)

VARIANTS = {f"policy {p:<4} + Brier {c}": ("cdpo", cfg(p, c))
            for c in (0.0, 1.0) for p in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)}

rows, seeds = [], int(os.environ.get("SEEDS", "20"))
for name, (objective, config) in VARIANTS.items():
    out = []
    for seed in range(seeds):
        _, rep = run_training(records, meta, objective=objective, cdpo=config,
                              train_seed=seed, **base)
        f = rep["by_objective"][normalise_objective(objective)]
        curve = sorted(((p["escalation_rate"], p["accuracy"]) for p in f["curve"]),
                       key=lambda row: row[0])
        rates, accs = [r for r, _ in curve], [a for _, a in curve]
        out.append((f["auroc_gain_vs_helps"], f["ece_fast_correct"],
                    f["gain_calibration_error"], f["gain_calibration_slope"],
                    float(np.interp(0.5, rates, accs)), float(np.interp(0.7, rates, accs))))
    a = np.array(out)
    rows.append((name, a.tolist()))
    print(f"{name:26} routing {a[:,0].mean():.4f}±{a[:,0].std(ddof=1):.4f}  "
          f"ECE {a[:,1].mean():.4f}  gainECE {a[:,2].mean():.4f}±{a[:,2].std(ddof=1):.4f}  "
          f"slope {a[:,3].mean():.4f}  acc@50% {a[:,4].mean():.4f}±{a[:,4].std(ddof=1):.4f}  "
          f"acc@70% {a[:,5].mean():.4f}±{a[:,5].std(ddof=1):.4f}", flush=True)

write_json_atomic(sys.argv[2] if len(sys.argv) > 2 else "results/repair.json",
                  {"columns": ["auroc_routing", "ece", "gain_ece", "gain_slope",
                               "acc50", "acc70"],
                   "variants": dict(rows)})
