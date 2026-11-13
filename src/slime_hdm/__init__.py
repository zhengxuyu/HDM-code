"""The hybrid-thinking model as a slime rollout.

slime pairs Megatron for training with SGLang for generation, joined by a data
buffer, and lets a custom function replace the generation half:

    generate_rollout(args, rollout_id, data_buffer, evaluation=False)

That is the half this package fills. An episode here is an agentic trajectory in
which the escalation decision is taken at every step by the decision head, and
the reward is whatever the environment returns at the end.

What this package does not do is train through Megatron. The thing being learned
is a 5128 by 256 by 4 head, about 700 KB, beside a frozen backbone. That is the
method's premise rather than an optimisation: the fast and slow modes are the
backbone's own two modes, and every recorded label depends on them not moving.
Megatron updating the backbone would invalidate them.

So the samples this returns are marked so that the training half leaves them
alone, and `head_trainer` updates the head from the same samples. If the
backbone is ever to be trained too, that is a different experiment and it should
say so out loud rather than arrive through a default.

The exports are resolved on first use rather than on import. An environment runs
where its own dependencies are, which for ScienceWorld is a container with a JVM
and no GPU and therefore no torch; importing the head trainer to reach an
environment adapter would put torch in that image for nothing.
"""

import importlib

# name -> the module that defines it, imported when the name is first read.
_EXPORTS = {
    "Environment": "slime_hdm.environment",
    "EpisodeResult": "slime_hdm.environment",
    "StepRecord": "slime_hdm.environment",
    "ClientScorer": "slime_hdm.factory",
    "StepScorer": "slime_hdm.factory",
    "build_head": "slime_hdm.factory",
    "build_scorer": "slime_hdm.factory",
    "load_environment": "slime_hdm.factory",
    "HeadRLConfig": "slime_hdm.head_trainer",
    "effective_sample_size": "slime_hdm.head_trainer",
    "train_head_on_episodes": "slime_hdm.head_trainer",
    "LoopConfig": "slime_hdm.loop",
    "run": "slime_hdm.loop",
    "generate_rollout": "slime_hdm.rollout",
    "rollout_episodes": "slime_hdm.rollout",
    "samples_from": "slime_hdm.rollout",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = [
    "Environment", "EpisodeResult", "StepRecord",
    "ClientScorer", "StepScorer", "build_head", "build_scorer", "load_environment",
    "HeadRLConfig", "effective_sample_size", "train_head_on_episodes",
    "LoopConfig", "run",
    "generate_rollout", "rollout_episodes", "samples_from",
]
