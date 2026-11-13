"""The hybrid-thinking rollout, with slime's generate_rollout signature.

Each step: the fast mode runs one forward pass over the prompt and reads its
action from the candidate set; the head decides from that same forward pass
whether to escalate; an escalated step reasons first and then reads the same
candidates. Nothing is parsed from generated text.

The reward arrives once, at the end. That is recorded on the sample and on every
step of it, because which step deserved the credit is exactly what is not known.
"""

import time
from dataclasses import replace

import numpy as np

from hde.core.block import escalation_gain, fast_confidence
from hde.core.block_utils import build_features
from slime_hdm.environment import EpisodeResult, StepRecord


def _canonical(environment, text):
    """`(action, problem)` for written text, through the environment's own parse.

    The environment owns the grammar, so the rollout asks rather than reimplements
    -- and it asks for the parse alone, never `act_freely`, so nothing is executed
    and the trajectory is untouched. A venue without the hook gets the raw last
    line, which is what every venue got before.
    """
    if not text:
        return None, "nothing was written"
    if hasattr(environment, "canonical_action"):
        return environment.canonical_action(text)
    return text.strip().splitlines()[-1].strip(), None


def rollout_episodes(environment, scorer, head, *, lambda_, episodes, budget,
                     escalate_prob=None, confidence_cut=None, rng=None,
                     max_steps=50, question_type="choice",
                     on_episode=None, on_start=None, indices=None, probe_prob=0.0,
                     clock=time.perf_counter):
    """Run `episodes` trajectories and return an EpisodeResult for each.

    `head` decides, unless `escalate_prob` is given, in which case the decision is
    a coin toss at that probability. A random policy is how the first batch of
    training data gets both actions at every state; a head that has not been
    trained yet cannot explore.

    `probe_prob` runs the mode **not** taken at a step as well, and records what it
    would have done. It never changes which action is taken, so the trajectory
    distribution is identical with and without it and the episodes stay valid for
    the off-policy update -- `tests/test_slime_hdm_rollout.py` pins that by
    replaying the same seed both ways. It is how the two asymmetric outcome
    classes get any observation at all; see `StepRecord.probe_probabilities`.

    An escalated step is probed unconditionally, because the fast distribution was
    already computed to make the decision and was being discarded, so that half is
    free. A step that did not escalate needs a generation to probe, so `probe_prob`
    rations it, and the probe runs whichever slow path this venue's escalations
    use -- `solve` where they write their action, `slow` where they read it off the
    candidates -- because a probe against a mode the policy never takes measures
    nothing. Probe tokens and seconds are recorded separately and are not the
    policy's cost.

    `clock` is the time source, injectable because the one property worth
    asserting about it -- that a probe's duration is excluded from the policy's
    `seconds` -- was being checked by comparing two real rollouts within 20 ms.
    exps measured that tolerance at about the 97th percentile of the noise, so the
    guard for the first defect in this sequence failed roughly one suite run in
    sixteen, and a guard that flaky gets quarantined rather than fixed. With a
    fake clock the assertion is exact.

    `indices` names which episodes to run, for resuming: the caller knows which
    ones it already has, which is not the same as how many. A count would be
    wrong the moment one episode is dropped -- nine rows after episode five was
    dropped would restart at nine, re-running episode eight and never retrying
    five. `on_start(index)` is called before each episode begins, so a caller
    can record the attempt **before** a hang that takes the process.
    """
    if escalate_prob is not None and confidence_cut is not None:
        raise ValueError(
            "escalate_prob and confidence_cut are two decision rules; passing both "
            "would run one arm and let the record name the other")
    if head is None and escalate_prob is None and confidence_cut is None:
        raise ValueError("give a head to decide with, an escalate_prob to explore "
                         "with, or a confidence_cut to threshold with")
    rng = rng if rng is not None else np.random.default_rng(0)
    # **A separate stream for the probe**, or the invariance above is false: the
    # escalation decision draws from `rng` too, so a probe draw would shift every
    # later escalation draw and probing would silently change the trajectory --
    # the one thing it must not do. `spawn` advances the seed sequence without
    # consuming the parent's stream, so `rng`'s own outputs are identical whether
    # probing is on or off.
    if not hasattr(rng, "spawn"):
        # A fixed-seed fallback here would give every episode the identical probe
        # stream, so `probe_prob` would stop being independent across episodes and
        # the labels would quietly correlate -- it would not raise. exps caught it.
        # Deriving from the parent's own draw keeps independence without consuming
        # the stream the escalation decision reads, since this happens once before
        # any episode.
        raise TypeError(
            "probing needs a numpy Generator with `spawn` (numpy >= 1.25), not "
            f"{type(rng).__name__}: a legacy RandomState would give every episode "
            "the same probe stream")
    probe_rng = rng.spawn(1)[0]
    results = []
    for episode in (range(episodes) if indices is None else indices):
        if on_start is not None:
            on_start(episode)
        began = clock()
        context, actions = environment.reset(episode)
        result, total = EpisodeResult(episode=episode, reward=0.0), 0.0
        for step in range(max_steps):
            if not actions:
                break
            started = clock()
            probabilities, hidden = scorer.fast(context, actions)
            gain = confidence = None
            if escalate_prob is not None:
                escalate = bool(rng.random() < escalate_prob)
            elif confidence_cut is not None:
                # The fast mode's own top probability, thresholded. No head and
                # nothing trained: the rule a practitioner reaches for first, and
                # the one the single-question venues are compared against.
                # `confidence` stays None here on purpose -- that field records
                # the head's calibrated P(fast ok), which is a different quantity
                # from this maximum and must not be written under its name.
                escalate = bool(float(np.max(probabilities)) < confidence_cut)
            else:
                outcome = head.predict(build_features(hidden, probabilities, question_type)[None])[0]
                gain, confidence = escalation_gain(outcome), fast_confidence(outcome)
                escalate = gain > lambda_
            # An escalated step writes its action when the environment can take a
            # written one, and picks from the enumerated set otherwise. Reading the
            # slow answer off the same codes gives the two modes one action space,
            # so a step needing text is unwinnable at every price -- which is a
            # fact about the readout and not about the task.
            can_write = (hasattr(environment, "act_freely") and hasattr(scorer, "solve"))
            written = can_write and escalate
            outcome = problem = None
            if written:
                text, generated, _ = scorer.solve(
                    context, environment.slow_instructions(), budget)
                outcome, problem = environment.act_freely(text)
                # **What it executes as**, not the raw last line. The two differ
                # whenever the model backticks the call, writes a sentence after
                # it, or quotes a bid differently, and the raw line was being
                # recorded as the taken action on every written step -- so the
                # corpus misdescribed 44% of its own steps. `canonical_action`
                # runs the same parse `act_freely` just ran and executes nothing.
                action, taken_trouble = _canonical(environment, text)
                if action is None:
                    action = text.strip().splitlines()[-1].strip() if text else ""
                decided = probabilities
            elif escalate:
                # Three values, not two. Both real scorers have returned the
                # reasoning alongside the distribution and the token count since
                # they were written; this line unpacked two and therefore raised
                # `too many values to unpack` on the first escalation of any
                # environment without `act_freely`. The browser venue has one, so
                # its escalations take the `written` branch above and never reach
                # here -- which is why no run ever hit it.
                decided, generated, _reasoning = scorer.slow(context, actions, budget)
                taken_trouble = None
            else:
                decided, generated, taken_trouble = probabilities, 0, None
            if not written:
                action = actions[int(np.argmax(decided))]
            # **The probe.** After the action is decided and before it is taken,
            # so nothing below depends on it. On an escalated step the fast
            # distribution is already in hand; otherwise a slow pass is run, at
            # `probe_prob`, purely to record what it would have chosen.
            # Was what was taken something the fast mode could have picked? On a
            # written step it need not be, and that is the reach/reasoning split.
            trouble_taken = taken_trouble if written else None
            taken = {"action_problem": trouble_taken,
                     # `None`, not `False`, when the text named no action: on such
                     # a step `action` is the raw-line fallback, so `in actions` is
                     # false and would assert the slow mode reached outside the
                     # offered set -- a reach claim about a text that named nothing
                     # at all. The probe side already did this; the taken side did
                     # not, and nothing pinned it. exps found it in my own test's
                     # record dump.
                     "action_offered": (None if trouble_taken is not None
                                        else action in actions)}
            probe = {}
            if escalate:
                # **Free, and it does not depend on which slow path ran.** The
                # fast distribution was computed to make the decision and is in
                # hand either way. The first version of this required
                # `not written`, which in a venue with `act_freely` is false on
                # every escalation -- so the half advertised as unconditional
                # never fired at all there, and 44% of a browser corpus's steps
                # got no probe. exps caught it before any collection ran.
                probe = {"probe_probabilities": probabilities,
                         "probe_action": actions[int(np.argmax(probabilities))],
                         "probe_mode": "fast", "probe_tokens": 0,
                         # By construction, since it is an argmax over them.
                         "probe_action_offered": True}
            elif probe_prob > 0 and probe_rng.random() < probe_prob:
                # **The probe has to run the slow path this venue would have
                # run**, or it records what a mode the policy never uses would
                # have chosen. Where escalating writes its action, the probe
                # writes too and parses the text the same way; it just does not
                # execute it, which is what keeps the trajectory untouched.
                probing = clock()
                if can_write:
                    text, spent, _truncated = scorer.solve(
                        context, environment.slow_instructions(), budget)
                    thought = probabilities
                    # Canonical on both sides, or a formatting difference reads as
                    # the two modes disagreeing. That error has a direction: it
                    # can only invent disagreement, never agreement, so it pushes
                    # the one number this collection exists for toward 100%.
                    chosen, trouble = _canonical(environment, text)
                    mode = "slow_written"
                else:
                    thought, spent, _reasoning = scorer.slow(context, actions, budget)
                    chosen, trouble = actions[int(np.argmax(thought))], None
                    mode = "slow"
                probe = {"probe_probabilities": thought, "probe_action": chosen,
                         "probe_mode": mode, "probe_tokens": int(spent),
                         "probe_problem": trouble,
                         # The reach/reasoning split, recorded now because the
                         # offered list is not stored and cannot be recovered.
                         "probe_action_offered": (None if trouble is not None
                                                  else chosen in actions),
                         "probe_seconds": clock() - probing}
            # Appended before the environment moves, so the row exists even if
            # `step` raises -- and the time is written afterwards by *replacing*
            # the entry, because `StepRecord` is `frozen=True` and that is
            # deliberate. `dataclasses.replace` keeps both properties: the row is
            # in the list before the move, and its `seconds` covers the move as
            # well as the decision.
            record = StepRecord(
                step=step, hidden=hidden, probabilities=probabilities, candidates=len(actions),
                escalated=escalate, generated_tokens=generated, action=action,
                gain=gain, confidence=confidence, **taken, **probe)
            result.steps.append(record)
            if written and problem is not None:
                result.steps[-1] = replace(
                    record,
                    seconds=clock() - started - record.probe_seconds)
                # The model wrote something unexecutable. Not a click, and not a
                # silent skip: the step is recorded and the episode ends, because
                # continuing would attribute the rest of it to a decision nobody
                # made.
                result.ended = "unexecutable_action"
                result.metadata["unexecutable_action"] = problem
                result.metadata["unexecutable_text"] = action
                break
            context, actions, done, reward = (
                outcome if written else environment.step(action))
            result.steps[-1] = replace(
                record,
                # Net of the probe: what this step cost the policy, so a probed
                # run's latency column compares with an unprobed one's.
                seconds=clock() - started - record.probe_seconds,
                reward=float(reward))
            total += float(reward)
            if done:
                result.finished = True
                break
        # **Why it stopped, with the environment's own word winning.** It knows
        # what the browser did and the rollout can only see the protocol: after a
        # step timeout `BrowserEnvironment.step` returns `done=True` with no
        # actions, so reading `done` alone labels a killed browser `done`.
        # code-reviewer found that. No stored row was wrong, because both callers
        # check `environment.complete` and drop the episode first -- but it was
        # right by compensation, which is not a property to rely on. The
        # rollout's own word comes first only where the environment cannot have
        # one: an action the model wrote that the page could not take.
        # Net of the probes, for the same reason each step's is.
        result.seconds = clock() - began - result.probe_seconds
        result.ended = (result.ended
                        or getattr(environment, "ended", None)
                        or ("done" if result.finished
                            else "no_candidates" if not result.steps
                            else "max_steps"))
        result.reward = total
        results.append(result)
        if on_episode is not None:
            on_episode(result)
    return results


def samples_from(results, sample_cls, *, rollout_id, trainable):
    """Turn episodes into slime samples.

    `trainable` says whether the training half should touch these. It is false
    for the hybrid-thinking model, whose backbone is frozen: the samples exist to
    carry the reward and the per-step features to the head trainer, and a zero
    loss mask keeps Megatron from taking a gradient through the backbone on them.
    Passing true trains the backbone as well, which invalidates every recorded
    label, so nothing sets it by accident.
    """
    samples = []
    for result in results:
        sample = sample_cls()
        sample.index = result.episode
        sample.rollout_id = rollout_id
        sample.reward = float(result.reward)
        sample.response_length = len(result.steps)
        sample.tokens = []
        sample.loss_mask = [1] * len(result.steps) if trainable else [0] * len(result.steps)
        sample.status = (sample_cls.Status.COMPLETED if result.finished
                         else sample_cls.Status.TRUNCATED)
        # One definition of what a row holds, shared with `loop.built`: see
        # `EpisodeResult.row`. The features live under `head_features` rather
        # than in `tokens`, because they are not tokens and the training half
        # must not read them as any.
        sample.metadata = result.row(
            steps=len(result.steps),
            head_features=[s.row() for s in result.steps])

        samples.append([sample])
    return samples


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """slime's hook. Builds the environment and scorer from `args`, then rolls out.

    The environment and the scorer are constructed through the paths named on
    `args` rather than imported here, so this function does not know which
    environment it is running and cannot quietly become specific to one.
    """
    from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
    from slime.utils.types import Sample

    from slime_hdm.factory import build_head, build_scorer, load_environment

    environment = load_environment(args, evaluation=evaluation)
    scorer = build_scorer(args)
    head = build_head(args)
    episodes = min(getattr(args, "rollout_batch_size", environment.episodes),
                   environment.episodes)
    results = rollout_episodes(
        environment, scorer, head,
        lambda_=getattr(args, "hdm_lambda", 0.0),
        episodes=episodes,
        budget=getattr(args, "hdm_think_budget", 256),
        escalate_prob=None if head is not None else getattr(args, "hdm_escalate_prob", 0.5),
        rng=np.random.default_rng(rollout_id),
        max_steps=getattr(args, "hdm_max_steps", 50),
    )
    metrics = {
        "hdm/reward": float(np.mean([r.reward for r in results])),
        "hdm/escalations": float(np.mean([r.escalations for r in results])),
        "hdm/generated_tokens": float(np.mean([r.generated_tokens for r in results])),
        "hdm/finished": float(np.mean([r.finished for r in results])),
        "hdm/steps": float(np.mean([len(r.steps) for r in results])),
    }
    if evaluation:
        return RolloutFnEvalOutput(data={"hdm": metrics}, metrics=metrics)
    samples = samples_from(results, Sample, rollout_id=rollout_id,
                           trainable=bool(getattr(args, "hdm_train_backbone", False)))
    return RolloutFnTrainOutput(samples=samples, metrics=metrics)
