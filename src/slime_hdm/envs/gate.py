"""Never thinking against always thinking in a browser, before any router exists.

This is a gate and nothing else. It builds no head, trains nothing, and routes
nothing: it runs the two constant policies on the same tasks and says whether
routing is worth evaluating in this venue at all. ALFWorld is why it exists ---
there, always thinking scored 0.575 against never thinking's 0.619, so the best
policy was approximately a constant one and two rounds of RL on top learned
something worse than both. That was a fact about the venue, and it cost GPU to
find out after the router had been built rather than before.

Three venues, one instrument. **WebArena** is the venue, chosen on 2026-10-05 for
the reason that decided it: it has external numbers, 812 tasks and several
independent leaderboards, where WorkArena has almost none and JevBench cost us a
section for exactly that. **MiniWoB** is the precondition check -- pinned local
pages, no server, no container, no API key -- and answers the one question both
real venues depend on: whether the fast mode can finish a multi-step browser
episode at all. **WorkArena** is kept because the gate is the same instrument and
it comes back if a ServiceNow instance ever does.

The thresholds are in `experiments/webarena/README.md` and are fixed before the
run. This script prints its verdict against them rather than leaving a reader to
decide, because a gate whose pass condition is settled afterwards is not a gate.

Three ways this venue can fail, and they are not the same failure:

    always <= never     thinking is a net liability, as in ALFWorld
    both near zero      the group may be beyond the fast mode entirely, so there
                        is no gap to route rather than no value in it. This is the
                        likely failure on WebArena, whose published results are
                        20-40% for frontier models that do think
    both near one       the group may be answerable without thinking at all,
                        which is the same conclusion from the other end

So the verdict is per group and never pooled.

What it records, per step, is what a later collection would need and what this
run needs to be read at all: the candidate set as offered, how many candidates
were dropped to fit the single-token codes, the chosen index, the readout's
distribution, and the seconds. A step whose right element was truncated away is
not a step the model got wrong, and `dropped` is what tells them apart.
"""

import argparse
import json
import time
from statistics import mean

# The only module-level import outside the standard library: provenance pulls in
# nothing heavy, and the label it defines is checked before the browser starts.
from litjev.run.provenance import UNKNOWN

PASS = {
    # Fixed before the run; see experiments/workarena/README.md for why each.
    "always_over_never": 0.05,   # thinking must be worth something
    "both_right_min": 0.20,      # and there must be somewhere to decline
    # Distance from either end: both modes near zero is no gap to route rather than
    # no value in routing, and both near one is a level answerable without thinking.
    "spread_min": 0.05,
    # And the effect must be reliable as well as large, which the margin above
    # cannot say: `helps - hurts` is `(always - never) * n` exactly, so counting the
    # rescued tasks restates the margin rather than testing it. The first version of
    # this rule made that mistake and called the count a sign test; code-reviewer
    # worked out the algebra. A sign test uses the *split* of the discordant pairs,
    # not their difference, and it is independent of how many tasks both modes get
    # right -- which is the other thing the margin cannot separate.
    #
    # One-sided exact binomial on (helps, hurts) at one half. 4 against 0 is 0.0625
    # and passes; 3 against 0 is 0.125 and does not; 23 against 20 is 0.38 and does
    # not, which is the case a difference-of-three rule would have waved through.
    "sign_test_p": 0.10,
}


def chosen_tasks(found, count, stride):
    """Which tasks a level is measured on, and why not simply the first `count`.

    `group_tasks` returns them sorted, and a venue's names group by family, so
    the first twenty are a few families rather than a sample of the group: a
    verdict from them is a verdict about whichever families sort first. A stride
    walks the list instead, deterministically, and the names chosen go in the
    header so the run says which tasks it measured.
    """
    if count <= 0 or count >= len(found):
        return list(found)
    if stride <= 0:
        stride = max(1, len(found) // count)
    picked = found[::stride][:count]
    # A stride that lands short of the count fills from what it skipped, in order,
    # rather than returning fewer tasks than asked for without saying so.
    if len(picked) < count:
        picked += [name for name in found if name not in picked][: count - len(picked)]
    return picked


def require_commit(commit):
    """Refuse to run when nothing can say which code this is.

    A gate's whole output is a verdict against pre-registered thresholds, and a
    verdict that cannot be tied to a flagged commit is not evidence of anything.
    This is not hypothetical: the environment image copies `pyproject.toml`,
    `README.md` and `src/` and nothing else, so without a `REVISION` written into
    it there is no `.git` and no record, and every row would say `unknown`.
    """
    if commit == UNKNOWN:
        raise SystemExit(
            "refusing to run: no git checkout and no REVISION file, so every row "
            "would be labelled 'unknown' and the verdict could not be tied to a "
            "reviewed commit. In the container, build with "
            "--build-arg GIT_COMMIT=$(git rev-parse HEAD)."
        )
    return commit


def _registered(ids, venue, group):
    """The ids, after checking the registry really holds them.

    Every venue builds its ids from a package's own task list and a prefix, and
    the prefix is an assumption about how that package registers things. When it
    is wrong the run must die here rather than at the first `gym.make`, an hour
    in and holding a GPU.
    """
    import gymnasium

    ids = sorted(ids)
    missing = [task for task in ids if task not in gymnasium.registry]
    if missing:
        raise RuntimeError(
            f"{len(missing)} {venue} {group} task(s) are named by the package but not "
            f"registered under that id, e.g. {missing[0]}; the id scheme has changed"
        )
    return ids


def workarena_tasks(group):
    """One WorkArena level, as the installed package defines it.

    The level is a *suffix*, not a prefix: every id looks like
    `browsergym/workarena.servicenow.<task>` and the compositional ones end in
    `-l2` or `-l3`. Counted against `browsergym-workarena==0.5.3`: 33 atomic,
    341 L2, 341 L3, and the registry is exactly `workarena_tasks_all`.

    L1 comes from the package's own `workarena_tasks_l1` rather than from "has no
    level suffix", because those two differ: `update-private` is registered and
    unsuffixed but is not an L1 benchmark task.

    Kept after the venue moved to WebArena, because the gate is the same
    instrument and WorkArena comes back if a ServiceNow instance ever does.
    """
    import browsergym.workarena as workarena

    if group == "l1":
        names = list(workarena.workarena_tasks_l1)
    else:
        names = [name for name in workarena.workarena_tasks_all
                 if name.endswith(f"-{group}")]
    return _registered((f"browsergym/{name}" for name in names), "workarena", group)


def _webarena_configs():
    """WebArena's own task file, which is the only thing that knows the sites.

    The ids are `browsergym/webarena.0` to `.811` -- opaque integers. Nothing in a
    name says the site, the validator or the difficulty, so unlike WorkArena the
    grouping cannot come from the registry and has to be joined against this file.
    """
    from importlib.util import find_spec
    from pathlib import Path

    # `webarena` ships without an __init__, so it is a namespace package: its spec
    # has no origin and the directory is in submodule_search_locations. Reading
    # `spec.origin` alone said "not installed" about a package that was installed.
    spec = find_spec("webarena")
    roots = list(getattr(spec, "submodule_search_locations", None) or [])
    if spec is not None and spec.origin:
        roots.insert(0, str(Path(spec.origin).parent))
    if spec is None or not roots:
        raise RuntimeError("libwebarena is not installed, so the task file that "
                           "names the sites cannot be read")
    path = next((Path(root) / "test.raw.json" for root in roots
                 if (Path(root) / "test.raw.json").exists()),
                Path(roots[0]) / "test.raw.json")
    if not path.exists():
        raise RuntimeError(f"{path} is missing; libwebarena's layout has changed and "
                           f"the site grouping cannot be rebuilt from the registry")
    return json.loads(path.read_text())


def judged_by_a_model(config):
    """Whether this task's answer is graded by an LLM rather than by a program.

    118 of the 812, 14.5%, carry a `fuzzy_match` reference answer, and grading
    those needs `OPENAI_API_KEY`. They are excluded: a routing result whose score
    depends on a third party's model version cannot be reproduced once that
    version is deprecated, and 694 programmatically graded tasks are plenty. The
    exclusion is pre-registered rather than decided after seeing any score.
    """
    return "fuzzy_match" in ((config.get("eval") or {}).get("reference_answers") or {})


def webarena_tasks(group):
    """One WebArena site, excluding the model-judged tasks.

    A task belongs to a site group only when that site is the *only* one it
    needs, because each site is a container to stand up and a task spanning two
    needs both. The 48 multi-site tasks are reachable as `multi` and are not
    silently folded into a single-site group.

    Counted against `browsergym-webarena==0.14.3`: shopping 187, shopping_admin
    182, gitlab 180, map 109, reddit 106, before the model-judged exclusion.
    """
    configs = _webarena_configs()
    if group == "multi":
        wanted = [c for c in configs if len(set(c.get("sites") or [])) > 1]
    else:
        wanted = [c for c in configs if set(c.get("sites") or []) == {group}]
    kept = [c for c in wanted if not judged_by_a_model(c)]
    if wanted and not kept:
        raise RuntimeError(f"every {group} task is model-judged, so the programmatic "
                           f"subset is empty")
    return _registered((f"browsergym/webarena.{c['task_id']}" for c in kept),
                       "webarena", group)


def miniwob_split_path():
    """Where the committed split lives, resolved from this file and not from the cwd.

    It sits beside the code on purpose. It was under `experiments/` and read by a
    cwd-relative path, and neither place a run happens has `experiments/`:
    `scripts/sync.sh` rsyncs `src scripts tests configs`, and the image copies
    `pyproject.toml`, `README.md` and `src/`. So `--list` died in `miniwob_tasks`
    before anything ran, and the error's advice to regenerate it would have
    regenerated it into the same unreachable directory.
    """
    from pathlib import Path

    return Path(__file__).resolve().parent / "miniwob-action-split.json"


def miniwob_tasks(group, split_path=None):
    """One side of MiniWoB's action-space split, as the ceiling measurement.

    MiniWoB answers the question both real venues depend on and neither answers
    cheaply: can the fast mode finish a multi-step browser episode at all. The
    pages are a pinned local checkout, so there is no instance, no container and
    no API key -- `scripts/build_browsergym.sbatch` already installs it.

    There is deliberately **no pooled group**. Over 125 tasks that mix clicking
    and typing, a single "both near zero" would restate our action space and say
    nothing about anything else, so the only groups are the two sides of the split
    and every verdict is per side. `click` is the measurement; `beyond-click` is
    the control that shows the ceiling is the action space, and is expected to be
    near zero for both policies.
    """
    from pathlib import Path

    # Read before importing the venue, so a missing pre-registration is reported as
    # one on any machine rather than as a ModuleNotFoundError on the ones without
    # chromium. The same ordering mistake cost `--groups l4` its exit code once.
    path = Path(split_path) if split_path else miniwob_split_path()
    if not path.exists():
        raise RuntimeError(
            f"{path} is missing. It is the pre-registered split of MiniWoB by what "
            f"each task needs beyond clicking, and it ships beside this module so "
            f"that a synced tree and a container both have it. Regenerate it with "
            f"scripts/miniwob_action_split.py -o {path} against the pinned pages"
        )
    record = json.loads(path.read_text())
    names = record["groups"].get(group)
    if names is None:
        raise ValueError(f"{path} has groups {sorted(record['groups'])}, not {group!r}")

    import browsergym.miniwob  # noqa: F401  registers the tasks

    return _registered((f"browsergym/miniwob.{name}" for name in names),
                       "miniwob", group)


# What each venue is, in one place: the groups a verdict may be given over, the
# packages whose versions decide what a task name means, and the module whose
# import registers the tasks.
VENUES = {
    "webarena": {
        "groups": ("shopping", "shopping_admin", "gitlab", "map", "reddit",
                   "wikipedia", "multi"),
        "tasks": webarena_tasks,
        "packages": ("browsergym-core", "browsergym-webarena", "libwebarena",
                     "gymnasium", "playwright"),
        "registers": "browsergym.webarena",
        # WA_FULL_RESET is optional, so it is not required here.
        "requires_env": ("WA_SHOPPING", "WA_SHOPPING_ADMIN", "WA_REDDIT", "WA_GITLAB",
                         "WA_WIKIPEDIA", "WA_MAP", "WA_HOMEPAGE"),
    },
    "workarena": {
        "groups": ("l1", "l2", "l3"),
        "tasks": workarena_tasks,
        "packages": ("browsergym-core", "browsergym-workarena", "gymnasium",
                     "playwright"),
        "registers": "browsergym.workarena",
        "requires_env": ("SNOW_INSTANCE_URL", "SNOW_INSTANCE_UNAME",
                         "SNOW_INSTANCE_PWD"),
    },
    "miniwob": {
        "groups": ("click", "pointer", "beyond-click"),
        "tasks": miniwob_tasks,
        "packages": ("browsergym-core", "browsergym-miniwob", "gymnasium",
                     "playwright"),
        "registers": "browsergym.miniwob",
        "requires_env": ("MINIWOB_URL",),
    },
}


def missing_environment(venue, environ=None):
    """The variables this venue needs and does not have.

    Each venue reaches its tasks through the environment and fails differently
    without it: miniwob's pages are a `MINIWOB_URL` away, webarena's sites are
    seven URLs, workarena's instance is three. browsergym checks none of them
    until a task is constructed, so an unset variable costs a queue wait and a
    GPU allocation before it says so.

    `--list` does not need any of them, which is why this is not in the selector.
    """
    import os

    if venue not in VENUES:
        raise ValueError(f"venue must be one of {tuple(VENUES)}, not {venue!r}")
    environ = os.environ if environ is None else environ
    return [name for name in VENUES[venue]["requires_env"] if not environ.get(name)]


def group_tasks(venue, group):
    """The task ids of one group of one venue, validated before anything runs."""
    if venue not in VENUES:
        raise ValueError(f"venue must be one of {tuple(VENUES)}, not {venue!r}")
    known = VENUES[venue]["groups"]
    if group not in known:
        raise ValueError(f"{venue} has groups {known}, not {group!r}")
    tasks = VENUES[venue]["tasks"](group)
    if not tasks:
        # WebArena's `wikipedia` is this case today: all of its tasks also need
        # another site, so the single-site group is empty. An empty group would
        # otherwise reach `four_way([])` and be reported as a verdict over nothing.
        raise RuntimeError(
            f"{venue}'s {group} group resolves to no tasks. For webarena this means "
            f"every task naming that site also needs another one, so it exists only "
            f"inside the `multi` group"
        )
    return tasks


def step_record(task, policy, index, offered, dropped, chosen, readout, seconds,
                generated, escalated, attempt=0):
    return {
        # The kind is in the id. Without it a step row at index 0 and an episode row
        # at seed 0 are both `task|policy|0`, and `completed` keys on the id, so one
        # would have silently shadowed the other in every resumed run.
        "id": f"step:{task}|{policy}|{attempt}|{index}",
        "kind": "step",
        "attempt": attempt,
        "task": task,
        "policy": policy,
        "step": index,
        "candidates": len(offered),
        "dropped": dropped,
        # None for a written action: there was no index into the candidate set,
        # and a row that claims one names a candidate the model never chose.
        "chosen_index": chosen,
        "chosen_action": (offered[chosen][0] if offered and chosen is not None
                          else None),
        # The whole distribution, because a coded readout's own weakness is a prior
        # over slots: at 27B its off-target bias is the worst of the readouts, and
        # over twenty steps a preference for slot 0 compounds, so the distribution
        # is what would show it.
        #
        # **Nothing reads this yet.** The comment used to say `position_bias.py`
        # reads it; that script takes `litjev-collect` records from a .npz through
        # `load_records` and has never seen a gate row. code-reviewer caught that
        # after I had told exps the uniform-placeholder defect "protected
        # position_bias.py" -- it did not, because no position statistic was ever
        # exposed to it. An analysis that would use these rows has to be written.
        "probabilities": [float(value) for value in readout],
        "escalated": bool(escalated),
        "generated_tokens": int(generated),
        "seconds": float(seconds),
    }


def routing_fit(steps):
    """How well each step's action matched the mode that chose it.

    A function rather than five expressions inline in `run_episode`, because
    inline they could only be tested by running an episode -- and a mutation
    that swapped the predicate back to the narrow one passed every test.

    `unfit_fast_steps` is **the user's metric**, 0 being good, and the predicate
    is theirs: "the fast mode cannot express this", not "this needs generated
    text". `drag_and_drop` is why -- no text, two bids already on the page, and
    still impossible to offer, because enumerating it is `|bids|^2`.

    It is an invariant while the fast mode picks from the candidate set, so the
    three `escalated_*` counters are the measurable part: why each escalation was
    needed. Over 28831/28832's 709 `always` steps they are 118 / 454 / 137.
    `escalated_redundant` is the one a router should drive down -- those
    escalations bought nothing the fast mode did not already have on its menu.
    """
    fast = [row for row in steps if not row["escalated"]]
    slow = [row for row in steps if row["escalated"]]
    return {
        "unfit_fast_steps": sum(1 for row in fast if not row["fast_expressible"]),
        "fast_steps": len(fast),
        "escalated_type": sum(1 for row in slow if not row["fast_expressible"]),
        "escalated_element": sum(1 for row in slow if row["fast_expressible"]
                                 and row.get("outside_fast_set")),
        "escalated_redundant": sum(1 for row in slow if row["fast_expressible"]
                                   and not row.get("outside_fast_set")),
    }


def episode_record(run):
    """The outcome, as its own row, written when the episode ends.

    The step rows were streamed and the outcome was not, so a kill lost every
    result the gate exists for and kept the traces of how they were reached. The
    commit that added streaming said "rows stream with --resume" and meant the
    steps; code-reviewer caught that the sentence covered the half that does not
    matter on its own.
    """
    return {
        "id": f"episode:{run['task']}|{run['policy']}|{run['seed']}",
        "kind": "episode",
        # **Every key `four_way` reads has to be here**, because `rebuild_pairs`
        # builds a resumed run's pairs out of these rows rather than out of
        # memory. `used_outside_action` was missing from this list, so on a
        # resumed run `a.get("used_outside_action")` was `None` for every pair
        # that came back from the file -- and `not None` is true, so each of
        # those helps was counted as `helps_by_reasoning`.
        #
        # **That is where 28832's reasoning effect came from.** It hung
        # (`hang-stack-28832.txt`), was resumed, and reported 12 reach / 2
        # reasoning where its own step rows say 14 / 0. Recomputed from
        # `outside_fast_set` across both seeds: **28 reach, 0 reasoning**, against
        # the 26 / 2 in the JSONs. The 0.0185 reasoning rate was an artefact of
        # this line, not a measurement.
        **{key: run[key] for key in ("task", "policy", "seed", "reward", "success",
                                     "steps", "done", "truncated", "tokens", "seconds",
                                     "dropped_total", "ended", "used_outside_action",
                                     "unexecutable", "complete")},
        # Written out for the same reason, though nothing reads them yet: a metric
        # that only exists in memory cannot be checked after the run.
        **{key: run[key] for key in ("unfit_fast_steps", "fast_steps",
                                      "escalated_type", "escalated_element",
                                      "escalated_redundant") if key in run},
    }


def combine_results(results):
    """One verdict over several runs of the same level at different seeds.

    L1 has 33 tasks and no more, so the only way to buy power there is to run the
    level again at another seed. The rows file refuses to mix seeds -- its header
    records the seed and a resume against a different one is rejected -- so each
    seed is its own run and its own output, and this is what reads them back
    together. Without it, two runs means two verdicts and somebody averaging them,
    which is the pooling the gate is written to avoid.

    A (task, seed) pair is the unit, which is what the pairing already is: both
    policies saw the same instance. Two seeds of one task are **not** two
    independent tasks -- they are two instances of the same script, so 66 units
    from 33 tasks are worth somewhat less than 66 tasks would be. The count is
    reported as both so the reader can see which they have.

    Everything in the headers except the seed must agree, because a combined
    verdict over two different budgets or two different step limits is not a
    verdict about anything.
    """
    if not results:
        raise ValueError("nothing to combine")
    seeds = [r["header"]["seed"] for r in results]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"the same seed appears twice: {sorted(seeds)}; those are "
                         f"the same episodes, not more of them")
    ignored = {"seed", "environment"}
    first = results[0]["header"]
    for other in results[1:]:
        differing = {key: (first.get(key), other["header"].get(key))
                     for key in set(first) | set(other["header"])
                     if key not in ignored and first.get(key) != other["header"].get(key)}
        if differing:
            raise ValueError(f"these runs are not the same gate: {differing}")

    groups, tasks_per_group = {}, {}
    for group in first["groups"]:
        pairs, names = [], set()
        for result in results:
            seed = result["header"]["seed"]
            by_task = {}
            for episode in result["episodes"]:
                if episode["task"] in result["tasks"][group]:
                    by_task.setdefault(episode["task"], {})[episode["policy"]] = episode
            for task, both in sorted(by_task.items()):
                if len(both) == 2:
                    pairs.append((both["never"], both["always"]))
                    names.add((task, seed))
        if not pairs:
            raise ValueError(f"no complete pair for {group} in any of these runs")
        groups[group] = {**four_way(pairs), "verdict": verdict(four_way(pairs)),
                         "pairs": len(pairs),
                         "tasks": len({task for task, _ in names}),
                         "seeds": sorted({seed for _, seed in names})}
        tasks_per_group[group] = sorted({task for task, _ in names})
    return {"combined_from": seeds, "groups": groups, "tasks": tasks_per_group,
            "thresholds": PASS}


BLIND_EPISODES = 3


def blind_run(outcomes, limit=BLIND_EPISODES):
    """Whether this run is seeing no actions at all, rather than failing tasks.

    An episode that ends `no_candidates` offered the model nothing to choose from,
    so it ended at step 0. One is a fact about a page. The first few in a row are
    a broken observation: the page did not load, or `candidates` is reading a key
    the environment does not have -- which is exactly what
    `scripts/probe_browsergym.py` did for weeks while reporting that the readout
    fitted, because it counted `[bid]` in an `axtree_txt` that browsergym has
    never had.

    Aborting is not a verdict and cannot bias one: a run that saw no actions has
    measured nothing, and the alternative is a full run reporting `both near zero`
    about an environment rather than about thinking.
    """
    first = outcomes[:limit]
    if len(first) >= limit and all(ended == "no_candidates" for ended in first):
        return (f"the first {limit} episodes all ended with no candidates offered, "
                f"so nothing was measured. The page is not loading, or the candidate "
                f"rule is reading something this environment does not provide -- "
                f"check one episode by hand before spending the rest of the run")
    return None


def rebuild_pairs(done, tasks):
    """The pairs a resumed run already has, from the episode rows rather than memory.

    A resume that recomputed the split from the episodes it ran in this process
    would report a split over the tail of the task list and call it the level.
    """
    pairs, missing = [], []
    for task in tasks:
        both = {}
        for policy in ("never", "always"):
            for row in done.values():
                if row.get("kind") == "episode" and row.get("task") == task \
                        and row.get("policy") == policy:
                    both[policy] = row
        if len(both) == 2:
            pairs.append((both["never"], both["always"]))
        else:
            missing.append((task, sorted(set(("never", "always")) - set(both))))
    return pairs, missing


def attempts_so_far(done, task, policy):
    """How many times this episode has been started, so a re-run does not collide."""
    seen = {row.get("attempt", 0) for row in done.values()
            if row.get("kind") == "step" and row.get("task") == task
            and row.get("policy") == policy}
    return (max(seen) + 1) if seen else 0


def run_episode(env, decider, task, policy, *, budget, max_steps, limit, seed,
                state_chars, rows=None, attempt=0, step_timeout=120.0,
                stack_file=None):
    """One episode under one constant policy. Returns its steps and its outcome.

    `policy` is `never` or `always` and nothing else: this script cannot route,
    and a policy name it does not know is a mistake rather than a default.
    """
    from slime_hdm.envs.browser import (
        StepTimeout,
        candidates,
        content_free_actions,
        fast_expressible,
        needs_content,
        parse_action,
        state_text,
        step_deadline,
    )
    from slime_hdm.envs.browser_environment import SLOW_INSTRUCTIONS

    if policy not in ("never", "always"):
        raise ValueError(f"the gate runs never and always, not {policy!r}")
    try:
        with step_deadline(step_timeout, stack_file=stack_file):
            observation, _ = env.reset(seed=seed)
    except StepTimeout as problem:
        # A page that will not load is the same kind of non-outcome as a step that
        # will not return: recorded, incomplete, and not scored against either
        # policy. `reset` sits outside the episode clock, so without this it was
        # invisible in the rows as well as unbounded.
        return {"task": task, "policy": policy, "seed": seed, "reward": 0.0,
                "success": False, "steps": 0, "done": False, "truncated": False,
                "tokens": 0, "seconds": 0.0, "dropped_total": 0, "unexecutable": 0,
                "used_outside_action": False, "complete": False,
                "timed_out": str(problem), "ended": "reset_timeout", "step_rows": []}
    goal = observation.get("goal", "")
    trail, steps = [], []
    reward, done, truncated, ended = 0.0, False, False, "max_steps"
    timed_out = None
    began = time.perf_counter()
    scrolled = False
    for index in range(max_steps):
        try:
            with step_deadline(step_timeout, stack_file=stack_file):
                # The content-free half of browsergym's action set, which the fast mode can
                # have without becoming a generator: `candidates` offered 1 of the 13
                # actions, and the window filter exists only because scroll was missing.
                offered, dropped = candidates(env, observation, limit,
                                              extra=content_free_actions(scrolled))
                if not offered:
                    # Not a plain failure: the page offered nothing this step could choose
                    # from, which is a fact about the filtering and the page rather than
                    # about the model, and the two read the same in a success column.
                    ended = "no_candidates"
                    break
                actions = [shown for _, shown in offered]
                state = state_text(observation, goal, trail, chars=state_chars)
                started = time.perf_counter()
                # Always-think writes its action; never-think reads one off the codes. That
                # asymmetry is the point and it was missing: `decider.slow` reasons and
                # then picks from the same click-only list, so always-think had exactly
                # never-think's action space and a step needing text was unwinnable at any
                # price. The gate would then have reported "both near zero" about a venue
                # when it was reporting a restriction of ours.
                written = problem = None
                if policy == "always":
                    answer = decider.solve(state, SLOW_INSTRUCTIONS, budget)
                    allowed = set(observation.get("extra_element_properties", {}))
                    written, problem = parse_action(answer.get("text", ""), allowed)
                    # No index and no distribution: always-think did not choose
                    # from the candidate set, it wrote its action. The first
                    # version put `chosen = 0` and a uniform distribution here, so
                    # every one of 390 `always` rows claimed `chosen_action` was
                    # whatever happened to be offered first -- a candidate the
                    # model never saw the inside of. exps found it only because
                    # one row's parse failed and the two fields disagreed:
                    # `chosen_action: click('13')` beside `'et' is not an
                    # element`. A row that cannot say what the model wrote is
                    # worse than one that says nothing.
                    readout = []
                    chosen = None
                else:
                    answer = decider.fast(state, actions)
                    readout = answer["probabilities"]
                    chosen = max(range(len(readout)), key=lambda i: readout[i])
                seconds = time.perf_counter() - started
                row = step_record(task, policy, index, offered, dropped, chosen, readout,
                                  seconds, answer.get("generated_tokens", 0),
                                  policy == "always", attempt=attempt)
                # **The action that will actually be taken, and which mode produced
                # it.** `needs_content` is true when taking it required having
                # generated text, so a step with `needs_content and not escalated`
                # means the two modes' action sets have come apart: the fast mode
                # generates nothing, so it cannot express such an action at all.
                #
                # Today that combination is **structurally impossible** -- the
                # candidate set is `click` plus `scroll`, both content-free -- so
                # the count is an invariant rather than a learned quantity, and
                # that is worth recording rather than assuming. It becomes a real
                # measurement the moment the fast mode is offered anything from
                # `text_actions`, and then it answers "did the router learn to send
                # the typing steps to the slow mode" with 0 as the good value.
                attempted = (written if written is not None
                             else (offered[chosen][0] if offered and chosen is not None
                                   else None))
                # Both, because they answer different questions and the user chose
                # the wider one for the metric on 2026-10-06. `needs_content` is
                # "this required generated text"; `fast_expressible` is "the fast
                # mode could have produced this type at all". `drag_and_drop` is
                # the difference: no text, still impossible for the fast mode,
                # because enumerating it is `|bids|^2`.
                row["needs_content"] = needs_content(attempted)
                row["fast_expressible"] = fast_expressible(attempted)
                if written is not None:
                    row["written_action"] = written
                    # Whether the slow mode reached outside the fast mode's candidate set.
                    # This is what separates "reasoning won" from "a bigger action space
                    # won", and it is the separation the gate was said not to be able to
                    # make: the two policies differ in *both*, so a win has to say which.
                    row["outside_fast_set"] = written not in [action for action, _ in offered]
                if problem is not None:
                    # Recorded, never substituted by a click: an unexecutable action is a
                    # fact about the step, and the nearest element would credit thinking
                    # with an outcome it did not choose.
                    row["unexecutable"] = problem
                steps.append(row)
                if rows is not None:
                    rows.append(row)
                if problem is not None:
                    ended = "unexecutable_action"
                    break
                acted = written if written is not None else offered[chosen][0]
                scrolled = scrolled or acted.startswith("scroll(")
                observation, step_reward, done, truncated, _ = env.step(acted)
                reward = max(reward, float(step_reward or 0.0))
                trail.append(acted)
                if done or truncated:
                    ended = "done" if done else "truncated"
                    break
        except StepTimeout as problem:
            # Recorded, and the episode is marked incomplete rather than
            # failed: a browser that stopped answering is not a decision the
            # model made, and scoring it as a loss would bias whichever policy
            # happened to hit it. The pair is dropped from the split and the
            # count is reported, which is the same discipline as `dropped`.
            ended = "step_timeout"
            timed_out = str(problem)
            break
    return {
        "task": task,
        "policy": policy,
        "seed": seed,
        # How many written actions this episode could not execute. Reported per run
        # so that a parser too strict for how the model phrases things reads as a
        # parser problem rather than as the model being incapable.
        "unexecutable": sum(1 for row in steps if row.get("unexecutable")),
        # Whether this episode produced an outcome at all. A step that never
        # returned leaves the task neither done nor failed by the model.
        "complete": ended != "step_timeout",
        "timed_out": timed_out,
        # Did this episode act outside what the fast mode was offered? Only
        # always-think can, and it is what tells a win by reasoning from a win by
        # reach.
        "used_outside_action": any(row.get("outside_fast_set") for row in steps),
        # **Steps the fast mode decided at which the action taken needed generated
        # text.** Zero is good, and zero is what the current action set forces:
        # see `needs_content` on the step rows for why this is an invariant today
        # and a learning signal once the fast mode is widened. Written out rather
        # than recomputed, because `used_outside_action` above was computed the
        # same way and `episode_record` never serialised it -- the key is absent
        # from every archived episode row, so no analysis can check it afterwards.
        **routing_fit(steps),
        # The environment's own verdict, not ours. WorkArena validates by querying
        # the instance's database and the open page, so this is the only thing in
        # the record that says whether the episode succeeded -- and it arrives once,
        # at the end, which is the property the whole venue was chosen for.
        "reward": reward,
        "success": bool(reward >= 1.0),
        "steps": len(steps),
        "done": bool(done),
        "truncated": bool(truncated),
        "tokens": sum(row["generated_tokens"] for row in steps),
        "seconds": time.perf_counter() - began,
        "dropped_total": sum(row["dropped"] for row in steps),
        "ended": ended,
        "attempt": attempt,
        "step_rows": steps,
    }


def four_way(pairs):
    """The joint outcome of the two modes on the same task, and the oracle.

    Paired, because the four-way split is the whole point and an unpaired pair of
    accuracies cannot produce it: two runs at 0.5 can mean every question is
    answered by exactly one mode or by both on half of them, and a router has
    everything to do in the first case and nothing in the second.
    """
    # A pair is scorable only if both episodes produced an outcome. A step that
    # never returned is the browser stopping, not a decision: scoring it as a loss
    # would bias whichever policy happened to hit the hang -- in 28816 that was
    # `never`, on the fifteenth task. The count is reported so a run with many of
    # them is not read as a run with a verdict.
    incomplete = [(n, a) for n, a in pairs
                  if not n.get("complete", True) or not a.get("complete", True)]
    pairs = [pair for pair in pairs if pair not in incomplete]
    both_right = sum(1 for n, a in pairs if n["success"] and a["success"])
    helps = sum(1 for n, a in pairs if not n["success"] and a["success"])
    hurts = sum(1 for n, a in pairs if n["success"] and not a["success"])
    both_wrong = sum(1 for n, a in pairs if not n["success"] and not a["success"])
    total = max(len(pairs), 1)
    return {
        "n": len(pairs),
        "incomplete_pairs": len(incomplete),
        "never": sum(1 for n, _ in pairs if n["success"]) / total,
        "always": sum(1 for _, a in pairs if a["success"]) / total,
        "oracle": (both_right + helps + hurts) / total,
        "both_right": both_right / total,
        "helps": helps / total,
        # `helps` split by *why*. A task thinking rescued where the slow mode
        # acted outside the fast mode's candidate set was rescued by reach; one
        # where it stayed inside was rescued by reasoning. The two support
        # different claims and the gate used to report only their sum, which is
        # the confound exps raised on 28816.
        # `[...]` not `.get(...)`: a pair whose row predates the key would be
        # silently counted as `reasoning`, which is exactly how 28832 reported a
        # reasoning effect it had not measured. A rows file too old to answer
        # this must fail loudly rather than answer wrongly.
        "helps_by_reach": sum(1 for n, a in pairs if not n["success"] and a["success"]
                              and a["used_outside_action"]) / total,
        "helps_by_reasoning": sum(1 for n, a in pairs if not n["success"] and a["success"]
                                  and not a["used_outside_action"]) / total,
        "hurts": hurts / total,
        "both_wrong": both_wrong / total,
        "tokens_never": mean([n["tokens"] for n, _ in pairs] or [0.0]),
        "tokens_always": mean([a["tokens"] for _, a in pairs] or [0.0]),
        "seconds_never": mean([n["seconds"] for n, _ in pairs] or [0.0]),
        "seconds_always": mean([a["seconds"] for _, a in pairs] or [0.0]),
        "dropped_steps": sum(n["dropped_total"] + a["dropped_total"] for n, a in pairs),
        # Written actions that could not be executed, over the whole group. A high
        # count means the parser or the instructions are wrong, not the model.
        "unexecutable_steps": sum(n.get("unexecutable", 0) + a.get("unexecutable", 0)
                                  for n, a in pairs),
    }


def sign_test(helps, hurts):
    """One-sided exact binomial on the discordant pairs, at one half.

    The question a gate asks of a paired run is whether thinking helps more often
    than it hurts, and that is a statement about the *split* of the tasks the two
    modes disagree on. The tasks they agree on carry no information about it, which
    is why they are not in the denominator -- and why this says something the
    accuracy margin cannot, the margin being `(helps - hurts) / n` for the same two
    counts.

    Exact rather than normal-approximated: at these sizes the counts are single
    digits, where the approximation is worst and where a gate is most tempted to
    believe it.
    """
    from math import comb

    trials = helps + hurts
    if trials == 0:
        return 1.0
    return sum(comb(trials, k) for k in range(helps, trials + 1)) / 2 ** trials


# The fewest pairs a verdict may be given over. Not a threshold -- it can only
# turn a verdict into "no verdict", never flip pass into fail -- and it is here
# because exps's two-task probe printed a full FAIL with all four threshold lines,
# in a file indistinguishable from a real run's. 20 is the smallest row of the
# power table in experiments/browser-gate/README.md, where it is already 46%
# against a venue that has an effect; below it a FAIL says nothing and should not
# be quotable from the JSON by someone who did not read the launch command.
MIN_PAIRS = 20


def verdict(split, thresholds=None, min_pairs=MIN_PAIRS):
    """Pass or fail against the thresholds fixed before the run, with the reason.

    Three failures and they mean different things, so each is named rather than
    folded into one boolean. `oracle - always` is reported but is not a condition:
    ScienceWorld passed with 0.034 there, and what made it worth running was that
    69% of its episodes were unchanged by thinking -- the room to decline, which
    `both_right` measures.

    Below `min_pairs` there is **no verdict**: `pass` is None and the reason says
    so. The split is still returned and printed, because a probe's numbers are
    worth reading; what is refused is the word FAIL on two tasks.
    """
    rule = {**PASS, **(thresholds or {})}
    if split["n"] < min_pairs:
        return {"pass": None, "n": split["n"],
                "reasons": [f"no verdict: {split['n']} pair(s) is below the "
                            f"{min_pairs} the power analysis assumes, so neither "
                            f"PASS nor FAIL means anything here"]}
    gain = split["always"] - split["never"]
    helps = round(split["helps"] * split["n"])
    hurts = round(split["hurts"] * split["n"])
    observed = sign_test(helps, hurts)
    reasons = []
    if observed > rule["sign_test_p"]:
        reasons.append(
            f"thinking rescues {helps} task(s) and spoils {hurts}, which an exact "
            f"one-sided sign test puts at p = {observed:.3f}, over "
            f"{rule['sign_test_p']}: the accuracy margin cannot say this, since it is "
            f"the same quantity divided by n")
    if gain < rule["always_over_never"]:
        reasons.append(
            f"thinking is worth {gain:+.3f}, under {rule['always_over_never']}: this is "
            f"ALFWorld's shape, where the best policy is approximately a constant one")
    if split["both_right"] < rule["both_right_min"]:
        reasons.append(
            f"only {split['both_right']:.3f} of tasks are answered by both modes, under "
            f"{rule['both_right_min']}: there is almost nowhere that declining to think "
            f"costs nothing, so a router has little to save")
    if max(split["never"], split["always"]) < rule["spread_min"]:
        reasons.append(
            f"both modes are near zero ({split['never']:.3f}, {split['always']:.3f}): the "
            f"level is beyond the fast mode, so there is no gap to route rather than no "
            f"value in routing")
    if min(split["never"], split["always"]) > 1 - rule["spread_min"]:
        reasons.append(
            f"both modes are near one ({split['never']:.3f}, {split['always']:.3f}): the "
            f"level is answerable without thinking")
    return {"pass": not reasons, "reasons": reasons,
            "sign_test_p": observed, "helps_tasks": helps, "hurts_tasks": hurts,
            "oracle_over_always": split["oracle"] - split["always"]}


def format_groups(groups):
    lines = [f"{'group':14}{'n':>4}{'never':>8}{'always':>8}{'oracle':>8}"
             f"{'both':>7}{'helps':>7}{'hurts':>7}{'tok/ep':>9}{'bad':>5}{'verdict':>9}"]
    for name, split in groups.items():
        call = verdict(split)
        lines.append(
            f"{name:14}{split['n']:>4}{split['never']:>8.3f}{split['always']:>8.3f}"
            f"{split['oracle']:>8.3f}{split['both_right']:>7.3f}{split['helps']:>7.3f}"
            f"{split['hurts']:>7.3f}{split['tokens_always']:>9.0f}"
            f"{split.get('unexecutable_steps', 0):>5}"
            f"{'PASS' if call['pass'] else ('FAIL' if call['pass'] is False else '--'):>9}")
        for reason in call["reasons"]:
            lines.append(f"{'':14}  {reason}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--decision-url",
                        help="a decision service. The gate needs no head and does not "
                             "use one if the service has it loaded. Required unless "
                             "--list, --check-env or --combine, none of which run "
                             "anything")
    parser.add_argument("--venue", required=True, choices=sorted(VENUES),
                        help="webarena is the venue; miniwob is the precondition "
                             "check, which needs no server; workarena is kept and "
                             "needs a ServiceNow instance")
    parser.add_argument("--groups",
                        help="comma-separated groups to run, which for webarena are "
                             "its sites. Each group's tasks come from the installed "
                             "package, never from a name pattern. Required for a run "
                             "and for --list; --check-env and --combine do not use it")
    parser.add_argument("--list", action="store_true",
                        help="print the tasks each requested group holds, and stop")
    parser.add_argument("--check-env", action="store_true",
                        help="report whether this venue's environment variables are "
                             "set, and stop. Runs on a login node, so a launcher can "
                             "fail before sbatch rather than after the queue wait")
    parser.add_argument("--tasks-per-group", type=int, default=0,
                        dest="tasks_per_group",
                        help="0 is every task in the group, which is the default "
                             "because a subset is a verdict about a subset. A positive "
                             "value takes a deterministic stride through the sorted "
                             "names, not the first n, which would be a few families")
    parser.add_argument("--task-stride", type=int, default=0,
                        help="0 spreads --tasks-per-group evenly over the group")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--candidates", type=int, default=52,
                        help="the single-token codes this tokenizer supplies")
    parser.add_argument("--state-chars", type=int, default=6000)
    parser.add_argument("--tags-to-mark", default="all",
                        choices=("all", "standard_html"),
                        help="which elements browsergym gives a bid. The default "
                             "matches BrowserEnvironment's; browsergym's own is "
                             "standard_html, which leaves SVG elements with no bid "
                             "and so unreachable by either mode")
    parser.add_argument("--recycle-driver", type=int, default=0,
                        help="drop and restart browsergym's process-global "
                             "Playwright every N episodes. **0, off, is the "
                             "default on purpose.** It exists for an unevidenced "
                             "hypothesis -- 28816 hung at episode 29, after 28 "
                             "make/close cycles against that one global, which is "
                             "a shape and not a finding -- and defaulting it to 25 "
                             "turned that shape into a crash at episode 26 in "
                             "every run. An unevidenced mitigation is opt-in")
    parser.add_argument("--max-attempts", type=int, default=3,
                        help="how many times one episode may be started before it "
                             "is written off as `process_hang`. The hang exps "
                             "captured in 28832 takes the process -- a greenlet "
                             "blocked in Playwright cannot be unwound, only exited "
                             "-- so a launcher loops --resume, and without a cap a "
                             "task that hangs every time would loop forever")
    parser.add_argument("--step-timeout", type=float, default=120.0,
                        help="seconds a single step may take before it is recorded "
                             "as not having returned. 0 disables it. The free-form "
                             "slow mode costs about 11 s per step, so this is an "
                             "order of magnitude above normal rather than a tight "
                             "bound -- 28816 blocked for 3h35m inside one step")
    parser.add_argument("--no-headless", dest="headless", action="store_false",
                        default=True, help="show the browser, for watching one task")
    parser.add_argument("--rows", help="per-step rows as the run goes; defaults beside "
                                       "--output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--combine",
                        help="comma-separated gate outputs to read back as one verdict, "
                             "instead of running anything. For the same group at "
                             "several seeds, which is the only way to grow a small one")
    parser.add_argument("--output", default="workarena-gate.json")
    args = parser.parse_args()

    from pathlib import Path

    # Combining reads JSON written by earlier runs: no browser, no venue package, no
    # decision service. So it happens before the imports that need all three, and
    # can be run anywhere the outputs are.
    if args.combine:
        # No decision service, no venue package, no browser: it reads JSON.
        paths = [p.strip() for p in args.combine.split(",") if p.strip()]
        combined = combine_results([json.loads(Path(p).read_text()) for p in paths])
        for group, split in combined["groups"].items():
            print(f"\n=== {group}: {split['pairs']} pairs from {split['tasks']} tasks "
                  f"at seeds {split['seeds']}")
            print(format_groups({group: split}))
        Path(args.output).write_text(json.dumps(combined, indent=2) + "\n")
        print(f"\nwrote {args.output}", flush=True)
        return

    venue = VENUES[args.venue]

    # Before the venue imports, so a launcher can run it on a login node where
    # browsergym is not installed and nothing is allocated. It asks about the
    # venue's environment and nothing about groups, so it does not want --groups.
    if args.check_env:
        absent = missing_environment(args.venue)
        for name in venue["requires_env"]:
            print(f"{'unset' if name in absent else 'set':>5}  {name}")
        if absent:
            raise SystemExit(f"{args.venue} needs {', '.join(absent)}")
        print(f"{args.venue}: every required variable is set")
        return

    if not args.groups:
        parser.error("--groups is required for a run and for --list")
    # Before the venue imports too, so a typo in --groups is an argument error on
    # any machine rather than a ModuleNotFoundError on the ones without chromium.
    names = [n.strip() for n in args.groups.split(",") if n.strip()]
    unknown = [n for n in names if n not in venue["groups"]]
    if unknown:
        parser.error(f"--groups for {args.venue} takes "
                     f"{','.join(venue['groups'])}; got {unknown[0]!r}")
    if not args.decision_url and not args.list:
        parser.error("--decision-url is required unless --list, --check-env or "
                     "--combine")

    from importlib import import_module

    import gymnasium as gym

    import_module(venue["registers"])  # registers the tasks

    from inference import DecisionClient
    from litjev.run.provenance import commit_now, describe_environment
    from litjev.run.streaming import Rows, completed
    from slime_hdm.envs.browser import close_env
    if args.list:
        for group in names:
            found = group_tasks(args.venue, group)
            print(f"{group}: {len(found)} tasks")
            for task in found[:40]:
                print(f"  {task}")
        return

    absent = missing_environment(args.venue)
    if absent:
        # Exit 1, not argparse's 2: the contract in the README reserves 2 for a bad
        # argument and puts a broken environment at 1, and a launcher branches on it.
        raise SystemExit(
            f"{args.venue} needs {', '.join(absent)} in the environment. browsergym "
            f"does not check until a task is built, which is well into an allocated "
            f"job -- run --check-env on the login node before sbatch"
        )

    decider = DecisionClient(args.decision_url)
    health = decider.health()
    print(f"decision service: {health}", flush=True)

    rows_path = args.rows or str(Path(args.output).with_suffix(".rows.jsonl"))
    header = {"commit": require_commit(commit_now()),
              "environment": describe_environment(extra=venue["packages"]),
              "venue": args.venue, "groups": names, "budget": args.budget,
              "max_steps": args.max_steps, "candidates": args.candidates,
              "tasks_per_group": args.tasks_per_group, "task_stride": args.task_stride,
              "step_timeout": args.step_timeout,
              # code-reviewer's condition on 9da34cf: the header is both the run's
              # recorded config and what `Rows(..., config=header, resume=...)`
              # checks, so an argument that reaches `gym.make` and not this dict
              # leaves the output unable to say which environment produced it --
              # and lets `--resume` mix `all` and `standard_html` rows at one
              # commit. A test asserts every `gym.make` keyword appears here.
              "tags_to_mark": args.tags_to_mark,
              "recycle_driver": args.recycle_driver, "seed": args.seed}
    done = completed(rows_path) if args.resume else {}
    rows = Rows(rows_path, config=header, resume=args.resume)

    episodes, groups, chosen, ran = [], {}, {}, []
    played = 0
    for name in names:
        found = group_tasks(args.venue, name)
        tasks = chosen_tasks(found, args.tasks_per_group, args.task_stride)
        chosen[name] = tasks
        print(f"\n=== {name}: {len(tasks)} of {len(found)} tasks", flush=True)
        # What the rows file already holds, so a resumed run reports a split over the
        # whole level rather than over the tasks this process happened to reach.
        #
        # The episodes go in with the pairs, in one place, for two reasons. The
        # output JSON is what `--combine` reads, so a resumed pair missing from it
        # was a pair missing from the combined verdict. And the per-task version of
        # this -- `dict(rebuild_pairs(done, [task])[0])` -- was `dict()` of a list of
        # (never, always) tuples, which takes a row as a key and raises
        # `unhashable type: 'dict'`: `--resume` died on the first task it had already
        # finished, the one case it exists for.
        kept, _ = rebuild_pairs(done, tasks)
        if kept:
            print(f"  resuming: {len(kept)} task(s) already paired in the rows file",
                  flush=True)
        pairs = list(kept)
        for never, always in kept:
            episodes.extend([never, always])
        paired_already = {row["task"] for pair in kept for row in pair}
        for task in tasks:
            if task in paired_already:
                continue
            run = {}
            for policy in ("never", "always"):
                finished = [row for row in done.values()
                            if row.get("kind") == "episode" and row.get("task") == task
                            and row.get("policy") == policy]
                if finished:
                    run[policy] = finished[0]
                    print(f"  {task:58} {policy:6} from the rows file", flush=True)
                    continue
                attempt = attempts_so_far(done, task, policy)
                if args.max_attempts and attempt >= args.max_attempts:
                    # Written off, not retried and not scored: a task that takes
                    # the process every time is a fact about the environment, and
                    # the pair is dropped from the split like any other incomplete
                    # episode. Without this a --resume loop spins on it.
                    run[policy] = {
                        "task": task, "policy": policy, "seed": args.seed,
                        "reward": 0.0, "success": False, "steps": 0, "done": False,
                        "truncated": False, "tokens": 0, "seconds": 0.0,
                        "dropped_total": 0, "unexecutable": 0,
                        "used_outside_action": False, "complete": False,
                        "timed_out": f"{attempt} attempt(s) took the process",
                        "ended": "process_hang", "step_rows": []}
                    rows.append(episode_record(run[policy]))
                    print(f"  {task:58} {policy:6} written off after {attempt} "
                          f"attempt(s): process_hang", flush=True)
                    continue
                # **Threaded, not defaulted.** `BrowserEnvironment` defaults
                # `tags_to_mark="all"` and this call site did not, so the gate
                # would have run on `standard_html` while the CDPO path ran on
                # `all` -- four SVG tasks visible to one and invisible to the
                # other, and the gate is the CDPO run's precondition. That is the
                # same shape as `content_free`: a guarantee living on one path and
                # not the other, found by code-reviewer rather than by me, and the
                # third time on this pair.
                env = gym.make(task, headless=args.headless,
                               wait_for_user_message=False,
                               tags_to_mark=args.tags_to_mark)
                try:
                    run[policy] = run_episode(
                        env, decider, task, policy, budget=args.budget,
                        max_steps=args.max_steps, limit=args.candidates,
                        seed=args.seed, state_chars=args.state_chars, rows=rows,
                        attempt=attempt, step_timeout=args.step_timeout,
                        stack_file=str(Path(args.output).with_suffix(".stacks.log")))
                finally:
                    trouble = close_env(env, args.step_timeout)
                    played += 1
                    if args.recycle_driver and played % args.recycle_driver == 0:
                        # The env is closed above; dropping the global here makes
                        # the next `gym.make` build a driver rather than reuse one
                        # that has served `--recycle-driver` episodes. Episode
                        # count is the only variable, which is what makes this a
                        # bound on the hypothesis rather than a guess at a cause.
                        from slime_hdm.envs.browser import forget_global_playwright

                        dropped = forget_global_playwright()
                        print(f"  recycling the browser driver after {played} "
                              f"episode(s): {'dropped' if dropped else 'failed'}",
                              flush=True)
                    if trouble:
                        # Printed, not raised: after a hang this is the second
                        # thing already wrong, and 28816 ended as a dead run
                        # because closing a wedged browser waits forever.
                        print(f"  {task:58} {policy:6} {trouble}", flush=True)
                # The outcome is written when it exists, not at the end of the run.
                rows.append(episode_record(run[policy]))
                print(f"  {task:58} {policy:6} reward {run[policy]['reward']:.2f} "
                      f"steps {run[policy]['steps']:3d} tok {run[policy]['tokens']:5d} "
                      f"{run[policy]['seconds']:6.1f}s  {run[policy]['ended']}",
                      flush=True)
                # Only episodes this process ran: a resumed one was already judged.
                ran.append(run[policy]["ended"])
                blind = blind_run(ran)
                if blind:
                    rows.close()
                    raise SystemExit(blind)
            pairs.append((run["never"], run["always"]))
            episodes.extend([run["never"], run["always"]])
        groups[name] = four_way(pairs)
    rows.close()

    print("\n" + format_groups(groups), flush=True)
    result = {
        "header": header,
        # The names, so the verdict says which tasks it is a verdict about.
        "tasks": chosen,
        "groups": {name: {**split, "verdict": verdict(split)}
                   for name, split in groups.items()},
        "thresholds": PASS,
        "episodes": [{k: v for k, v in e.items() if k != "step_rows"} for e in episodes],
        "rows": rows_path,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {args.output} and {rows_path}", flush=True)


if __name__ == "__main__":
    main()
