"""Token-accuracy curves for each router, on one record set."""
import json, sys
import numpy as np
from litjev.core.block import escalation_gain
from litjev.data.collect import open_records
from litjev.serving import load_head
from litjev.train.training import features_for, holdout_mask

def sweep(score, higher, fast, slow, tok, grid):
    xs, ys = [], []
    for v in grid:
        e = (score > v) if higher else (score < v)
        xs.append(float(np.where(e, tok, 0).mean()))
        ys.append(float(np.where(e, slow, fast).mean()))
    order = np.argsort(xs)
    return list(np.asarray(xs)[order]), list(np.asarray(ys)[order])

out = {}
for i in range(1, len(sys.argv), 4):
    path, head_path, label, held = sys.argv[i:i+4]
    rec, meta = open_records(path)
    head = load_head(head_path)
    pos = list(meta["feature_layers"]).index(-1)
    keep = (np.asarray(holdout_mask(rec["category"], 0.25, 0)[0], bool)
            if held == "heldout" else np.ones(len(rec["fast_correct"]), bool))
    fast = np.asarray(rec["fast_correct"], bool)[keep]
    slow = np.asarray(rec["slow_correct"], bool)[keep]
    tok = np.asarray(rec["thinking_tokens"], float)[keep]
    top = np.asarray(rec["stats"], float)[keep, 0]
    gain = np.array([escalation_gain(r) for r in head.predict(features_for(rec, (pos,))[keep])])
    hx, hy = sweep(gain, True, fast, slow, tok,
                   np.concatenate([[-np.inf], np.linspace(gain.min()-1e-9, gain.max()+1e-9, 400)]))
    cx, cy = sweep(top, False, fast, slow, tok,
                   np.concatenate([np.linspace(0, 1, 400), [1 + 1e-9]]))
    out[label] = {"n": int(keep.sum()), "fast": float(fast.mean()), "slow": float(slow.mean()),
                  "slow_tokens": float(tok.mean()),
                  "htm": [hx, hy], "confidence": [cx, cy]}
print(json.dumps(out))
