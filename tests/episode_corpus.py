"""A synthetic episode corpus, shared by the tests that fit a block from returns.

It lived in the browser test file, which this distribution does not carry: that
file drives a browser environment and reads a run record from the experiments
tree. The corpus itself is neither, so it sits on its own here.
"""

import numpy as np


def corpus(count=24, steps=4, seed=0):
    rng = np.random.default_rng(seed)
    episodes = []
    for index in range(count):
        episodes.append({
            "terminal_reward": 1.0 if index % 3 == 0 else 0.0,
            "head_features": [
                {"step": step, "hidden": rng.normal(size=8),
                 "probabilities": rng.random(4), "candidates": 4,
                 "escalated": bool(step % 2), "generated_tokens": 0,
                 "behaviour_prob": 0.5}
                for step in range(steps)],
        })
    return episodes
