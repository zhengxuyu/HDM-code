"""Isolate what each CDPO term carries. Head only; the backbone is never touched.

The last three variants fix the price instead of drawing it. CDPO's default range
is (-0.2, 0.5), and routing AUROC is a ranking of the gain, so it does not depend
on where the threshold sits: if marginalising only moved the threshold, AUROC
would be unchanged across these four and only the curve would differ. That is the
claim the name "price-marginalised" used to rest on, and it has never been run.
"""
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

VARIANTS = {
  "cross-entropy (weighted)": ("supervised", None),
  "cross-entropy (unweighted)": ("cdpo", CDPOConfig(policy_weight=0.0, calibration_weight=0.0,
                                                    anchor_weight=1.0, weighted_anchor=False)),
  "CE + Brier, no policy":     ("cdpo", CDPOConfig(policy_weight=0.0, calibration_weight=1.0,
                                                    anchor_weight=1.0)),
  "policy + Brier, no anchor": ("cdpo", CDPOConfig(anchor_weight=0.0)),
  "policy + anchor, no Brier": ("cdpo", CDPOConfig(calibration_weight=0.0)),
  "CDPO (full)":               ("cdpo", CDPOConfig()),
  # Price fixed rather than drawn, at three points of the default range.
  "policy at fixed lam=-0.1": ("cdpo", CDPOConfig(lambda_range=(-0.1, -0.1))),
  "policy at fixed lam=0.15": ("cdpo", CDPOConfig(lambda_range=(0.15, 0.15))),
  "policy at fixed lam=0.4":  ("cdpo", CDPOConfig(lambda_range=(0.4, 0.4))),
}
rows = []
for name, (objective, cfg) in VARIANTS.items():
    out = []
    for seed in range(int(__import__('os').environ.get('SEEDS','5'))):
        _, rep = run_training(records, meta, objective=objective, cdpo=cfg,
                              train_seed=seed, **base)
        f = rep["by_objective"][normalise_objective(objective)]
        # gain > lambda is exactly top-k by gain, so a dense lambda grid traces
        # the whole coverage-accuracy curve and lets us read accuracy at a
        # matched escalation rate. AUROC ranks; this is what the paper reports.
        curve = sorted(((p["escalation_rate"], p["accuracy"]) for p in f["curve"]),
                       key=lambda row: row[0])
        rates = [row[0] for row in curve]
        accs = [row[1] for row in curve]
        acc30 = float(np.interp(0.3, rates, accs))
        acc50 = float(np.interp(0.5, rates, accs))
        acc70 = float(np.interp(0.7, rates, accs))
        out.append((f["auroc_gain_vs_helps"], f["auroc_fast_correct"],
                    f["ece_fast_correct"], f["gain_calibration_error"],
                    f["gain_calibration_slope"], acc30, acc50, acc70))
    a = np.array(out)
    rows.append((name, a.tolist()))
    print(f"{name:28} routing {a[:,0].mean():.4f}  ECE {a[:,2].mean():.4f}  "
          f"gainECE {a[:,3].mean():.4f}\u00b1{a[:,3].std():.4f}  "
          f"slope {a[:,4].mean():.4f}\u00b1{a[:,4].std():.4f}  "
          f"acc@30% {a[:,5].mean():.4f}  acc@50% {a[:,6].mean():.4f}  "
          f"acc@70% {a[:,7].mean():.4f}", flush=True)

# Seeds are shared across variants, so the comparison that matters is paired.
# Dump per-seed values and let the analysis be done off the cluster.
write_json_atomic(sys.argv[2] if len(sys.argv) > 2 else "results/ablation.json",
                  {"columns": ["auroc_routing", "auroc_fast", "ece", "gain_ece", "gain_slope",
                               "acc30", "acc50", "acc70"],
                   "seeds": list(range(len(rows[0][1]))),
                   "variants": dict(rows)})
