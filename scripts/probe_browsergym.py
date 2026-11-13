"""Can BrowserGym run here, and how many candidates does a step offer?

The fast mode reads its action off an enumerated candidate set, one single-token
code per candidate, and the tokenizer yields 52 such codes. ALFWorld offers 28 to
50 actions a step and fits. A web page's accessibility tree may offer far more,
and if it does the readout needs a different encoding before any of this works.

So this counts, before anything is built on top.

It counts through `slime_htm.envs.browser.candidates`, the same function the gate
offers its actions from, because the previous version of this script counted
`[bid]` occurrences in `observation["axtree_txt"]` -- **a key browsergym does not
have.** The observation carries `axtree_object`, `dom_object` and
`extra_element_properties`; there is no `axtree_txt` anywhere in
browsergym-core 0.14.3. So the regex ran over an empty string, every step counted
zero, nothing exceeded the 52 codes, and the script printed "the readout fits as
it stands" -- a pass drawn from an empty set. Measuring the thing the gate
actually uses is the only version of this that can be wrong in a visible way.
"""
import argparse
from collections import Counter

from litjev.run.streaming import write_json_atomic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--limit", type=int, default=52,
                        help="the candidate budget the readout has, which is what "
                             "`dropped` is measured against")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import browsergym.miniwob  # noqa: F401  registers the environments
    import gymnasium as gym

    from slime_htm.envs.browser import candidates

    names = [name for name in gym.registry if name.startswith("browsergym/miniwob")]
    print(f"{len(names)} miniwob tasks registered", flush=True)
    if not names:
        raise SystemExit("no browsergym tasks registered; is browsergym-miniwob installed "
                         "and MINIWOB_URL set?")

    counts, dropped_counts, per_task, failures = [], [], {}, {}
    for name in sorted(names)[: args.tasks]:
        try:
            env = gym.make(name, headless=True, wait_for_user_message=False)
            observation, _ = env.reset(seed=0)
            seen, lost = [], []
            for _ in range(args.steps):
                offered, dropped = candidates(env, observation, args.limit)
                seen.append(len(offered))
                lost.append(dropped)
                if not offered:
                    break
                observation, _, terminated, truncated, _ = env.step(offered[0][0])
                if terminated or truncated:
                    break
            env.close()
            per_task[name] = {"offered": seen, "dropped": lost}
            counts.extend(seen)
            dropped_counts.extend(lost)
            print(f"  {name.split('.')[-1]:28} offered {seen}  dropped {lost}",
                  flush=True)
        except Exception as error:  # noqa: BLE001  one task failing is a result
            failures[name] = f"{type(error).__name__}: {error}"
            print(f"  {name.split('.')[-1]:28} FAILED {type(error).__name__}: "
                  f"{str(error)[:90]}", flush=True)

    report = {"tasks_registered": len(names), "tasks_tried": len(per_task),
              "limit": args.limit, "per_task": per_task, "failures": failures}
    measured = sorted(counts)
    if measured:
        report.update({
            "steps_measured": len(measured),
            "min": measured[0], "median": measured[len(measured) // 2],
            "max": measured[-1],
            # What `candidates` could not offer because the budget ran out. The
            # offered count is capped at the limit by construction, so counting
            # `offered > limit` would always be zero: `dropped` is the quantity
            # that says whether the page had more than the readout can encode.
            "steps_with_dropped": sum(d > 0 for d in dropped_counts),
            "dropped_max": max(dropped_counts),
            "histogram": dict(sorted(Counter(measured).items())),
        })
        print(f"\ncandidates offered per step over {len(measured)} steps: "
              f"min {measured[0]}, median {measured[len(measured) // 2]}, "
              f"max {measured[-1]} (budget {args.limit})")
        print(f"{report['steps_with_dropped']} of {len(measured)} steps had more "
              f"elements than the budget, dropping up to {max(dropped_counts)}")
        print("the readout fits as it stands" if report["dropped_max"] == 0 else
              "the readout needs a different encoding for the crowded steps")
    write_json_atomic(args.output, report)
    if not measured:
        # The previous version concluded "fits as it stands" from this case, which
        # is what a probe must never do: no observation is not a passing result.
        raise SystemExit(
            f"no candidate was offered on any of {len(per_task)} task(s), so nothing "
            f"was measured. {len(failures)} task(s) raised. This is a failure of the "
            f"probe or the environment, not a result about the readout"
        )


if __name__ == "__main__":
    main()
