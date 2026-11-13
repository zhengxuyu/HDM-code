"""Unified measured trace: pre-action RGB frame, decision, then transition outcome."""

import time
from dataclasses import asdict

from hde.vision import encode_image

# The tag keeps the engine's old spelling, as do the compare and reward-offline
# tags elsewhere. A schema version identifies traces that are already written to
# disk; a reader that stopped recognising them would reject every trace recorded
# before the rename. The next change to the format takes a new version number.
TRACE_VERSION = "litjev.trace.v1"


def record_episode(env, policy, *, seed=0, episode=1):
    observation, _ = env.reset(seed=seed)
    decisions = []
    running_reward = 0.0
    started = time.perf_counter()
    while True:
        picture = encode_image(observation)
        decision = policy.decide(observation)
        next_observation, reward, terminated, truncated, info = env.step(decision.action)
        running_reward += reward
        row = asdict(decision)
        row.update(
            episode=episode,
            step=len(decisions),
            frame=picture,
            action_index=decision.action,
            action=policy.action_names[decision.action],
            latency_ms=decision.latency_seconds * 1000,
            timestamp=time.perf_counter() - started,
            reward=reward,
            running_reward=running_reward,
            terminated=terminated,
            truncated=truncated,
            forward_calls=decision.usage.get("forward_calls"),
            output_tokens=decision.usage.get("output_tokens"),
            info=info,
        )
        decisions.append(row)
        observation = next_observation
        if terminated or truncated:
            break
    return {
        "schema_version": TRACE_VERSION,
        "environment": env.spec.id if env.spec else type(env).__name__,
        "model": policy.client.model_id,
        "training": False,
        "observation": "rgb_pixels_only",
        "seed": seed,
        "actions": list(policy.action_names),
        "decisions": decisions,
        "episode_reward": running_reward,
        "terminated": terminated,
        "truncated": truncated,
        "wall_seconds": time.perf_counter() - started,
    }
