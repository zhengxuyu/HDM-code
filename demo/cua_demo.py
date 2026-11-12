"""Run one browser task under three policies and record what each one cost.

The point the demo has to make is on the clock: always thinking is slow, never
thinking is fast and often fails, and routing spends the thinking only where the
head says it buys something. So every step is timed, and the timings are what the
demo replays.

Policies, all on the same task and the same seed:

    fast    never think            the floor
    slow    think at every step    the ceiling, and the cost
    routed  think when g > lambda  the claim

lambda is not a constant carried over from another benchmark. The gain a head
predicts is calibrated to the domain it was trained on, so a head from elsewhere
escalates everywhere or nowhere here. It is set instead from this domain's own
traffic: run the fast mode over a sample of steps, take the quantile of g that
gives the escalation rate asked for. Section 3.1.2 of the paper describes this as
the way a deployment picks lambda when it has a budget rather than a price.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from slime_htm.envs.browser import candidates, state_text

PANEL = """
(info) => {
  const put = (id, css) => {
    let box = document.getElementById(id);
    if (!box) {
      box = document.createElement('div');
      box.id = id;
      box.style.cssText = css;
      document.body.appendChild(box);
    }
    return box;
  };
  const shell = 'position:fixed;right:14px;z-index:2147483647;color:#f5f5f7;' +
    'font:600 13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;' +
    'background:rgba(17,17,20,.94);border-radius:10px;padding:11px 14px;' +
    'box-shadow:0 2px 16px rgba(0,0,0,.4);pointer-events:none';
  const clock = put('htm-clock', shell + ';top:14px;width:214px');
  clock.style.display = info.clock ? 'block' : 'none';
  clock.innerHTML =
    '<div style="font-size:11px;letter-spacing:.09em;text-transform:uppercase;opacity:.6">'
    + info.policy + '</div>'
    + '<div style="font-size:24px;margin:3px 0 1px">' + info.elapsed.toFixed(1) + 's</div>'
    + '<div style="opacity:.7">step ' + info.step + '</div>'
    + '<div style="color:' + (info.thinking ? '#ffd166' : '#8ce99a') + '">'
    + (info.thinking ? 'thinking  +' + info.think_seconds.toFixed(1) + 's' : 'instant') + '</div>';

  const think = put('htm-thinking', shell + ';top:' + (info.clock ? 130 : 14) + 'px;' +
    'width:330px;max-height:62vh;' +
    'overflow:hidden;font-weight:400;font-size:12px');
  if (info.reasoning) {
    think.style.display = 'block';
    think.innerHTML =
      '<div style="font-size:11px;letter-spacing:.09em;text-transform:uppercase;' +
      'color:#ffd166;margin-bottom:6px">thinking</div>'
      + '<div style="white-space:pre-wrap;opacity:.93">'
      + info.reasoning.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))
      + '</div>';
  } else {
    think.style.display = 'none';
  }
  return true;
}
"""


def show_clock(env, policy, step, thinking, elapsed, think_seconds, reasoning=None,
               clock=True):
    """Draw the clock, and the reasoning when there was any, into the page.

    An overlay added by the recorder would show the same numbers, but only to
    whoever set the recorder up. In the page it travels with the screenshots and
    with anyone else's capture.

    The reasoning is shown because a yellow label saying `thinking` only asserts
    that the step cost more. The text is what the extra seconds bought, and it
    is also what makes a viewer able to tell the two modes apart on sight.

    `clock` draws the elapsed time into the page as well. That is right when the
    browser itself is being recorded and wrong when `replay.py` assembles the
    video: a screenshot is taken once a step, so a clock inside it is frozen at
    that step's reading while the replay's own clock runs on, and the two
    disagree on screen.

    Playwright passes one argument to the function it is given, so this takes an
    object. Written as `(...arguments)` it raised on every step and the frames
    came out with no clock at all, which is why the failure is returned rather
    than swallowed.
    """
    try:
        env.unwrapped.page.evaluate(PANEL, {
            "policy": policy, "step": step, "thinking": bool(thinking),
            "elapsed": float(elapsed), "think_seconds": float(think_seconds),
            # The reasoning goes into the page only when the page is what is
            # being recorded. Otherwise replay.py draws it, and can draw it
            # when it was actually produced.
            "reasoning": reasoning if clock else None, "clock": bool(clock)})
        return None
    except Exception as problem:  # noqa: BLE001  a missing clock must not end a run
        return repr(problem)


def reached(env, done):
    """Whether the task is finished, for a site that scores nothing itself.

    MiniWoB ends an episode and rewards it. An open-ended task on a real site
    does neither, so the demo says what finishing looks like: a substring of the
    address, or of the text on the page.
    """
    if not done:
        return False
    page = env.unwrapped.page
    if done.get("url") and done["url"] not in page.url:
        return False
    if done.get("text"):
        try:
            text = page.inner_text("body")
        except Exception:  # noqa: BLE001  a page that cannot be read is not done
            return False
        if done["text"].lower() not in text.lower():
            return False
    return True


def run_episode(env, decider, goal, policy, *, lambda_=0.0, max_steps=15,
                limit=52, budget=256, frames=None, seed=0, done=None,
                selector=None, extra=(), state_chars=4000, page_clock=True):
    """One episode. Returns the per-step record, with a clock on every step."""
    observation, _ = env.reset(seed=seed)
    trail, steps, started = [], [], time.perf_counter()
    for index in range(max_steps):
        offered, dropped = candidates(env, observation, limit, selector=selector,
                                      extra=extra)
        if not offered:
            break
        actions = [shown for _, shown in offered]
        state = state_text(observation, goal, trail, chars=state_chars)

        step_started = time.perf_counter()
        fast = decider.fast(state, actions)
        fast_seconds = time.perf_counter() - step_started

        gain, confidence = fast["gain"], fast["confidence"]
        if policy == "slow":
            escalate = True
        elif policy == "fast":
            escalate = False
        else:
            if gain is None:
                raise RuntimeError("the service is serving no head, so nothing can route")
            escalate = gain > lambda_

        decided, generated, think_seconds, reasoning = fast["probabilities"], 0, 0.0, None
        if escalate:
            think_started = time.perf_counter()
            slow = decider.slow(state, actions, budget)
            think_seconds = time.perf_counter() - think_started
            decided, generated = slow["probabilities"], slow["generated_tokens"]
            reasoning = slow.get("reasoning")

        action = offered[int(np.argmax(decided))][0]
        clock_failure = show_clock(env, policy, index, escalate,
                                   time.perf_counter() - started, think_seconds,
                                   reasoning=reasoning, clock=page_clock)
        if clock_failure and index == 0:
            print(f"  the clock did not draw: {clock_failure}", flush=True)
        if frames is not None:
            path = Path(frames) / f"{policy}-{index:02d}.png"
            try:
                path.write_bytes(env.unwrapped.page.screenshot())
            except Exception:  # noqa: BLE001  a missing frame must not end the run
                path = None
        observation, reward, terminated, truncated, _ = env.step(action)
        trail.append(f"- {action}")
        if done and reached(env, done):
            reward, terminated = 1.0, True
        steps.append({
            "step": index, "action": action, "candidates": len(actions),
            "escalated": bool(escalate), "gain": gain, "confidence": confidence,
            "generated_tokens": generated, "reasoning": reasoning,
            "fast_seconds": round(fast_seconds, 4),
            "think_seconds": round(think_seconds, 4),
            # What the serving side measured, so the record separates the model's
            # time from the network's without anyone having to trust the total.
            "fast_seconds_served": fast["seconds"],
            "think_seconds_served": slow["seconds"] if escalate else 0.0,
            "step_seconds": round(fast_seconds + think_seconds, 4),
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "frame": path.name if frames and path else None,
            "reward": float(reward),
        })
        if terminated or truncated:
            break

    # Every frame above is the page a step was decided on, taken before the
    # action runs, so without this one the last action's result is never seen:
    # the video stopped on the shelf and never showed the book it opened.
    final = None
    if frames is not None and steps:
        show_clock(env, policy, steps[-1]["step"], False,
                   time.perf_counter() - started, 0.0, clock=page_clock)
        final = Path(frames) / f"{policy}-end.png"
        try:
            final.write_bytes(env.unwrapped.page.screenshot())
        except Exception:  # noqa: BLE001  as above: the run's numbers still stand
            final = None
    return {
        "policy": policy, "lambda": lambda_ if policy == "routed" else None,
        "steps": steps, "final_frame": final.name if final else None,
        "total_seconds": round(time.perf_counter() - started, 3),
        "thinking_seconds": round(sum(s["think_seconds"] for s in steps), 3),
        "served_seconds": round(sum(s["fast_seconds_served"] + s["think_seconds_served"]
                                    for s in steps), 3),
        "escalations": sum(s["escalated"] for s in steps),
        "generated_tokens": sum(s["generated_tokens"] for s in steps),
        "succeeded": bool(steps and steps[-1]["reward"] > 0),
    }


def calibrate_lambda(env, decider, goal_of, rate, *, episodes=8, limit=52, max_steps=10,
                     selector=None, extra=(), state_chars=4000, page_clock=True):
    """Pick lambda from this domain's own gains, for the escalation rate asked for.

    A threshold carried over from the benchmark the head was trained on means
    something different here, because the gain is calibrated to that domain. The
    quantile is what the paper prescribes when a deployment names a rate instead
    of a price, and it needs no labels: the gain is produced by the fast mode on
    every step whether or not the step escalates.

    It is calibrated across episodes rather than within the one about to be run.
    Within one, a short task yields two or three gains, and a quantile of two
    numbers is not a threshold: widening the browser window was enough to move
    the gains slightly, flip a step, and turn a routed run that succeeded in 42%
    of always-thinking's time into one that failed in 13% of it. The episodes
    used here are other seeds of the same task, so the demo's own episode is not
    among them.
    """
    gains = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=1000 + episode)
        goal, trail = goal_of(observation), []
        for _ in range(max_steps):
            offered, dropped = candidates(env, observation, limit, selector=selector,
                                      extra=extra)
            if not offered:
                break
            actions = [shown for _, shown in offered]
            fast = decider.fast(state_text(observation, goal, trail, chars=state_chars), actions)
            if fast["gain"] is None:
                raise RuntimeError("the service is serving no head, so nothing can route")
            gains.append(fast["gain"])
            chosen = offered[int(np.argmax(fast["probabilities"]))][0]
            observation, _, terminated, truncated, _ = env.step(chosen)
            trail.append(f"- {chosen}")
            if terminated or truncated:
                break
    if not gains:
        return 0.0, []
    return float(np.quantile(gains, 1.0 - rate)), [round(g, 4) for g in gains]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-url",
                        help="a decision service: the engine and the head in one job, "
                             "answering like any other model API")
    parser.add_argument("--model",
                        help="run the engine in this process instead, for a machine with the GPU")
    parser.add_argument("--head", help="swap this for a browser-trained head")
    parser.add_argument("--headless", action="store_true",
                        help="off by default: the point of the demo is watching it")
    parser.add_argument("--task", default="browsergym/miniwob.click-button")
    parser.add_argument("--start-url",
                        help="a real site instead of a MiniWoB page. The task becomes "
                             "browsergym/openended, and --goal says what to do there")
    parser.add_argument("--goal", help="the instruction, when --start-url is given")
    parser.add_argument("--success-url", help="the episode has succeeded once the address "
                                              "contains this")
    parser.add_argument("--success-text", help="...or the page's text does")
    parser.add_argument("--content-selector",
                        help="restrict candidates to this part of the page, e.g. #bodyContent "
                             "for a Wikipedia article. Ignored on a page it does not match")
    parser.add_argument("--also", action="append", default=[],
                        help="an action offered at every step that belongs to no element, "
                             "e.g. goto('https://books.toscrape.com/'). Repeatable")
    parser.add_argument("--state-chars", type=int, default=4000,
                        help="how much of the page's accessibility tree the model is shown")
    parser.add_argument("--no-page-clock", dest="page_clock", action="store_false",
                        help="leave the elapsed time and the reasoning out of the page, for "
                             "runs that replay.py will assemble. A step is photographed once, "
                             "after it has been decided, so an overlay baked into that shot is "
                             "shown for the whole step: the clock stands still at the step's "
                             "final reading, and the reasoning is on screen from the first "
                             "frame although it was not written until the last. replay.py "
                             "draws both from the record, where the timings are")
    parser.add_argument("--video", action="store_true",
                        help="record the browser itself, which is what a real site is for")
    parser.add_argument("--escalation-rate", type=float, default=0.3,
                        help="lambda is set to the quantile of g that gives this rate")
    parser.add_argument("--calibration-episodes", type=int, default=8,
                        help="other seeds of the same task, sampled to set lambda. Two or "
                             "three gains from the demo's own episode is not a threshold")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0,
                        help="the episode all three policies run. MiniWoB draws a different "
                             "instance of the task per seed")
    parser.add_argument("--viewport", default="1024x640",
                        help="MiniWoB's own frame is 320 wide, so the default viewport puts "
                             "the clock on top of the task. WIDTHxHEIGHT")
    parser.add_argument("--policies", default="fast,slow,routed",
                        help="which of the three to run. A sweep looking for tasks where "
                             "thinking changes the outcome needs only fast,slow")
    parser.add_argument("--budget", type=int, default=256, help="the slow mode's token budget")
    parser.add_argument("--candidate-limit", type=int, default=52,
                        help="single-token codes available; crowded pages are truncated to this")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import browsergym.miniwob  # noqa: F401  registers the environments
    import gymnasium as gym

    output = Path(args.output)
    (output / "frames").mkdir(parents=True, exist_ok=True)

    if args.decision_url:
        from inference import DecisionClient

        decider = DecisionClient(args.decision_url)
        health = decider.health()
        if not health.get("head"):
            raise SystemExit(f"the service at {args.decision_url} has no head loaded: {health}")
    else:
        from inference import build_service

        if not (args.model and args.head):
            raise SystemExit("without --decision-url this process holds the model, so it "
                             "needs --model and --head")
        decider = build_service(args.model, args.head)
        health = decider.health()
    print(f"decider {args.decision_url or 'in-process'}: {health}", flush=True)

    width, _, height = args.viewport.partition("x")
    settings = dict(headless=args.headless, wait_for_user_message=False,
                    viewport={"width": int(width), "height": int(height)})
    if args.video:
        settings["record_video_dir"] = str(output / "video")
    done = {"url": args.success_url, "text": args.success_text}
    done = done if (args.success_url or args.success_text) else None
    if args.start_url:
        if not args.goal:
            raise SystemExit("--start-url needs --goal: the site scores nothing by itself")
        if done is None:
            raise SystemExit("--start-url needs --success-url or --success-text, or nothing "
                             "can say whether the episode finished")
        env = gym.make("browsergym/openended",
                       task_kwargs={"start_url": args.start_url, "goal": args.goal},
                       **settings)
    else:
        env = gym.make(args.task, **settings)

    def goal_of(observation):
        return observation.get("goal") or " ".join(
            part.get("text", "") for part in observation.get("goal_object", [])
            if isinstance(part, dict))

    observation, _ = env.reset(seed=args.seed)
    goal = args.goal or goal_of(observation)
    print(f"task {args.start_url or args.task}\ngoal: {goal}", flush=True)

    policies = [name.strip() for name in args.policies.split(",") if name.strip()]
    unknown = set(policies) - {"fast", "slow", "routed"}
    if unknown:
        raise SystemExit(f"unknown policies {sorted(unknown)}")

    lambda_, sampled = (0.0, [])
    if "routed" in policies:
        lambda_, sampled = calibrate_lambda(env, decider, goal_of, args.escalation_rate,
                                            episodes=args.calibration_episodes,
                                            limit=args.candidate_limit,
                                            max_steps=args.max_steps,
                                            selector=args.content_selector,
                                            extra=tuple(args.also),
                                            state_chars=args.state_chars)
        print(f"lambda {lambda_:.4f} for a {args.escalation_rate:.0%} escalation rate, from "
              f"{len(sampled)} gains over {args.calibration_episodes} other seeds", flush=True)

    runs = []
    for policy in policies:
        run = run_episode(env, decider, goal, policy, lambda_=lambda_,
                          max_steps=args.max_steps, limit=args.candidate_limit,
                          budget=args.budget, frames=output / "frames", seed=args.seed,
                          done=done, selector=args.content_selector, extra=tuple(args.also),
                          state_chars=args.state_chars, page_clock=args.page_clock)
        runs.append(run)
        print(f"{policy:7} {run['total_seconds']:7.2f}s  "
              f"served {run['served_seconds']:7.2f}s  "
              f"thinking {run['thinking_seconds']:6.2f}s  "
              f"{run['escalations']}/{len(run['steps'])} steps escalated  "
              f"{run['generated_tokens']:5d} tokens  "
              f"{'succeeded' if run['succeeded'] else 'did not succeed'}", flush=True)
        (output / "trajectory.json").write_text(json.dumps({
            "task": args.start_url or args.task, "goal": goal, "seed": args.seed,
            "success": done,
            "decision_url": args.decision_url,
            "model": args.model, "head": args.head, "health": health,
            "escalation_rate_requested": args.escalation_rate,
            "lambda": lambda_, "sampled_gains": sampled,
            "budget": args.budget, "candidate_limit": args.candidate_limit,
            "runs": runs,
        }, indent=1) + "\n")
    env.close()

    slow = next((r for r in runs if r["policy"] == "slow"), None)
    routed = next((r for r in runs if r["policy"] == "routed"), None)
    if slow and routed and slow["total_seconds"]:
        print(f"\nrouted took {routed['total_seconds'] / slow['total_seconds']:.0%} of "
              f"always-thinking's wall clock")


if __name__ == "__main__":
    main()
