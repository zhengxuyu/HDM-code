"""Check that the 27B shards reassemble into the question set the 4B run used.

The shards were cut by arithmetic on the CLI's slice, questions[offset::stride][:limit].
If that arithmetic is wrong the shards still load and still train; they simply
describe a different benchmark, and the two backbones stop being comparable.
Compare the question ids, not the counts.
"""
import glob

import numpy as np

from hde.data.collect import open_records

shards = sorted(glob.glob("results/records-27b-shard*.npz"))
ids, fast, slow, cats = [], [], [], []
for path in shards:
    # A checkpoint among the shards would pass this check and defeat its purpose:
    # the ids it holds are a prefix of that shard's, so the reassembled set would
    # be short of questions while every id in it matched, which is exactly the
    # "still loads, still trains, describes a different benchmark" case above.
    r, _ = open_records(path)
    ids.append(r["question_id"])
    fast.append(r["fast_correct"])
    slow.append(r["slow_correct"])
    cats.append(r["category"])
    print(f"{path.split('/')[-1]:44} {len(r['question_id']):5d}")
ids = np.concatenate(ids)
fast = np.concatenate(fast).astype(bool)
slow = np.concatenate(slow).astype(bool)
cats = np.concatenate(cats)

print(f"\ntotal {len(ids)}   unique {len(set(ids.tolist()))}")
reference = open_records("results/records-4b-27063.npz")[0]["question_id"]
same = set(ids.tolist()) == set(reference.tolist())
print(f"same question set as the 4B run: {same}")
if not same:
    only27 = set(ids.tolist()) - set(reference.tolist())
    only4 = set(reference.tolist()) - set(ids.tolist())
    print(f"  only in 27B: {len(only27)}   only in 4B: {len(only4)}")

print(f"\nfast {fast.mean():.4f}   slow {slow.mean():.4f}   gap {slow.mean() - fast.mean():.4f}")
print(f"helps {int((~fast & slow).sum())}  hurts {int((fast & ~slow).sum())}  "
      f"both right {int((fast & slow).sum())}  both wrong {int((~fast & ~slow).sum())}")
print(f"categories {len(set(cats.tolist()))}")
