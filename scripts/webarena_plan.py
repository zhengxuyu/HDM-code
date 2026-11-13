"""Pre-register which WebArena tasks a run reads, before any of them has a score.

This exists because of a specific mistake. On 2026-10-05 I regrouped a MiniWoB
task set *after* seeing the first 28 episodes of a run and described the result as
"a pre-registered subgroup"; review caught it, and the correction is recorded in
the gate's README. Naming a subset in advance is not pre-registering an analysis
of it. So the WebArena selection is computed here, written to a committed file,
and the file is what a run reads -- the same shape as
`src/slime_htm/envs/miniwob-action-split.json`.

**The site choice is the design decision, and it is made on the evaluator mix.**
MiniWoB's click gate PASSED and the gain is almost all **reach**: combined over
seeds 0 and 1 (108 pairs), `helps_by_reach` is 0.241 against `helps_by_reasoning`
0.019 -- about 13 to 1. Reasoning inside the same action space wins something, but
little. (The 0.000 that an earlier version of this file quoted was seed 0 alone;
exps's seed 1 gives 0.037, and a single-seed zero is not a property.) So the
licensed claim there is "the router detects a step the cheap mode cannot reach",
which is weaker than the paper wants.
WebArena's evaluators split exactly along that line:

    string_match                 the agent must SEND an answer. With no
                                 content-bearing action in the fast mode (the
                                 user's decision of 2026-10-05), these are
                                 unreachable at every price -- the `reach` shape
                                 again, and more of MiniWoB's problem
    program_html / url_match     scored on the page or the URL, which clicking and
                                 navigating can satisfy. **Both modes can do
                                 these**, so a difference between them is a
                                 difference in quality, which is the thing
                                 MiniWoB could not measure

Counted over the 812 tasks (every number below is reproduced by this script):

    site             usable  needs text  page/url only  image
    shopping_admin      156          62             94  1.2 GB
    shopping            153          54             99  5.0 GB
    reddit              104           9             95  4.3 GB
    gitlab              172          45            127  20.5 GB
    map                  64          44             20  1.1 GB

`reddit` is the venue this picks: 95 page/url tasks against only 9 that need text,
so a `never` baseline is not pinned at zero by the action space and the router's
gain has to come from somewhere other than reach. `gitlab` has more such tasks
but a 20.5 GB image, and `map` is mostly text-answer tasks. The choice is on the
evaluator mix and the image size, both known before any task was run.

    python scripts/webarena_plan.py --configs path/to/test.raw.json \
      -o src/slime_htm/envs/webarena-plan.json
"""

import argparse
import hashlib
import json
from pathlib import Path

# What `environment_docker`'s images publish, verified on 2026-10-06 by pulling
# `am1n3e/webarena-verified-shopping_admin` and reading its process table: the
# original Magento/MySQL/nginx/redis/elasticsearch stack, plus a control plane on
# 8877. The official archive.org tars are gone -- the items return
# `"is_dark": true` -- so these images are the only working source found.
SITE_IMAGES = {
    # Pinned by **digest**, not by tag: these come from an individual Docker Hub
    # account and a tag can be repushed under a run. Read 2026-10-06; the
    # preflight refuses a mismatch, so a repush stops the run rather than
    # silently changing the sites underneath a result.
    "shopping": ("am1n3e/webarena-verified-shopping", 7770, 5.0,
                 "sha256:3e8cb9b945ea9b1c94ab26dba53e8d12dd0406abbf4bf686fd3bb2b6a5908feb"),
    "shopping_admin": ("am1n3e/webarena-verified-shopping_admin", 7780, 1.2,
                       "sha256:d0531dd27ed98d0c459ff9e88118bf2ed8b660b0ed99c38837db46c065a5be13"),
    "reddit": ("am1n3e/webarena-verified-reddit", 9999, 4.3,
               "sha256:0594908059a03e5f610005689440a95c05e776e734006197c7b1eba12734cb46"),
    "gitlab": ("am1n3e/webarena-verified-gitlab", 8023, 20.5,
               "sha256:673ba94d757c5b3f4bb55751ad8b745a33bad2b987349322403cbc41035a92e0"),
    "wikipedia": ("am1n3e/webarena-verified-wikipedia", 8888, 0.0,
                  "sha256:8c52e6cc8b7cc6f77daaf279929517399b1889bd02b09a257ebd465364ad387d"),
    "map": ("am1n3e/webarena-verified-map", 3000, 1.1,
            "sha256:c769aa312c979d68e696dace7049d58eeeb016ce608f0f8779a181bf3714206f"),
}

# The gate pairs `never` against `always` per task and refuses a verdict under
# this many pairs. It is the gate's own constant, restated so this script's
# verdict can be read without opening `gate.py`.
MIN_PAIRS = 20


def judged_by_a_model(config):
    """The same rule as `gate.judged_by_a_model`, restated so this script stands alone."""
    return "fuzzy_match" in ((config.get("eval") or {}).get("reference_answers") or {})


def site_tasks(configs, site):
    """The programmatically graded tasks that need **only** this site.

    Only-this-site because each site is a container to stand up, and a task
    spanning two needs both of them running.
    """
    want = [c for c in configs if set(c.get("sites") or []) == {site}]
    return [c for c in want if not judged_by_a_model(c)]


def required(config):
    """Every value the task's validators require, with the channel it arrives by.

    Reuses `webarena_span_coverage.required_values`, which was corrected three
    times to get this boundary right -- `program_html`'s `must_include` is text
    that must end up **on the page**, `reference_answers` is the answer the agent
    **sends**, and a `reference_url` query parameter is something typed into a
    search box. Those are different channels and cannot share one rule.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from webarena_span_coverage import required_values

    return required_values(config)


def classify(config):
    """Which channel this task's score depends on. One of four.

    `none` is the **upper bound** on click-only reachability, not a count of it:
    an evaluator that requires no typed value may still only be satisfiable by a
    path a click-only policy cannot walk. It is an upper bound in the honest
    direction, which is what makes a verdict read off it safe.
    """
    values = required(config)
    if not values:
        return "none"
    channels = {row[2] for row in values}
    if "page" in channels:
        return "page"
    if "url" in channels:
        return "url_query"
    return "answer"


def intents(configs):
    """How many **independent** questions a set of tasks is.

    WebArena instantiates each intent template many times -- shopping ids 158 to
    162 are five fillings of one template -- and the gate's sign test pairs per
    task. Five variants of one intent are not five independent pairs, so a sample
    counted in tasks overstates the power by whatever the duplication factor is.
    Spreading the draw over the file, which is what an earlier version of this
    script did, does not fix that: it changes which variants are drawn, not how
    many distinct questions they are.
    """
    return {c.get("intent_template_id", c.get("intent")) for c in configs}


def build(configs):
    """Can a click-only fast mode be gated on WebArena? Per site, then a verdict.

    **The answer this returns is no**, and it is a finding rather than a
    configuration problem. The fast mode has no content-bearing action -- the
    user's decision of 2026-10-05 -- so a task whose score depends on a typed
    value is unreachable at every price, which is the `helps_by_reach` shape that
    dominates MiniWoB 13 to 1. Counting only the tasks whose
    evaluators require no typed value, and then counting distinct intents rather
    than task ids, the best site has 8. The gate needs 20.

    An earlier version of this script picked `reddit` on the grounds that it had
    "95 page/url tasks against 9 needing text, so `never` is not pinned at zero".
    That was wrong: those 95 are scored on content that has to appear on the page,
    which on a forum means a post body somebody typed. code-reviewer caught it.
    Counted properly, reddit has **zero**.
    """
    sites, best = {}, None
    for name, (image, port, size, digest) in SITE_IMAGES.items():
        kept = site_tasks(configs, name)
        buckets = {}
        for config in kept:
            buckets.setdefault(classify(config), []).append(config)
        clickable = buckets.get("none", [])
        row = {
            "usable": len(kept),
            "no_typed_value_required": len(clickable),
            "distinct_intents_there": len(intents(clickable)),
            # Concrete ids, so the preflight can ask whether libwebarena's
            # numbering exists in the registry these images are run under. Not a
            # sample: the verdict is that there are too few to gate on.
            "no_typed_value_task_ids": sorted(int(c["task_id"]) for c in clickable),
            "needs_sent_answer": len(buckets.get("answer", [])),
            "needs_page_content": len(buckets.get("page", [])),
            "needs_url_query": len(buckets.get("url_query", [])),
            "image": image,
            "digest": digest,
            "port": port,
            "image_gb_compressed": size,
        }
        sites[name] = row
        if best is None or row["distinct_intents_there"] > sites[best]["distinct_intents_there"]:
            best = name

    reachable = sites[best]["distinct_intents_there"]
    return {
        "venue": "webarena",
        "min_pairs": MIN_PAIRS,
        "gateable_with_a_click_only_fast_mode": reachable >= MIN_PAIRS,
        "verdict": (
            f"NO. The best site ({best}) has {reachable} distinct intents whose "
            f"score requires no typed value, against MIN_PAIRS = {MIN_PAIRS}. "
            f"This is not a site-selection problem: every site is short, and the "
            f"shortfall is the action space meeting a venue where almost every "
            f"task is scored on something typed."
            if reachable < MIN_PAIRS else
            f"yes, on {best}: {reachable} distinct intents"),
        "best_site": best,
        "sites": sites,
        "channels": {
            "none": "the evaluator requires no typed value. The UPPER BOUND on "
                    "click-only reachability -- such a task may still need a path "
                    "a click-only policy cannot walk",
            "answer": "the agent must send the answer, so the fast mode cannot "
                      "reach it at any price",
            "page": "text must appear on the page. On a forum or an issue tracker "
                    "that means a body somebody typed; this is the bucket the "
                    "earlier version of this script wrongly counted as reachable",
            "url_query": "the reference URL carries a search query, which is typed "
                         "into a box",
        },
        "what_would_change_this": [
            "a content-bearing fast action -- option (b), `fill`/`select_option` "
            "over spans of the goal, implemented in `browser.text_actions` and "
            "DECLINED by the user on 2026-10-05. `webarena_span_coverage.py` is "
            "the script that measures what it would buy",
            "a venue whose tasks are scored on clicks and navigation, which is "
            "what MiniWoB's click group is and why it was the precondition",
            "accepting a reach-dominated claim on WebArena, which is the same "
            "claim MiniWoB's two seeds already support (reach 0.241 against "
            "reasoning 0.019) and does not need 32 GB of containers",
        ],
        "login": {
            "require_login": "all 812 tasks",
            "note": "every task needs an authenticated session, so `storage_state` "
                    "must be produced before any episode. This is what the verified "
                    "images' auto-login headers are for and it is a hard dependency "
                    "that no earlier note recorded",
        },
        "state_reset": {
            "task_file_says": "require_reset is False for every task in the sites "
                              "examined, so WebArena itself does not ask for a full "
                              "reset per task",
            "but": "tasks still mutate the sites, so state carries between them in "
                   "whatever order they run. The control plane's /init is the "
                   "candidate mechanism and its semantics are UNVERIFIED",
            "mechanism": "POST to the container's control plane on 8877",
            "routes": ["/status", "/is_ready", "/init", "/start", "/stop", "/restart"],
            "verified": "2026-10-06, shopping_admin: /status returns per-service "
                        "health; POST /init did not return inside 10 s",
            "open": "whether /init restores the initial state is NOT established. "
                    "`scripts/webarena_preflight.py` settles it destructively",
        },
        "excluded": {
            "model_judged": "fuzzy_match reference answers need OPENAI_API_KEY; a "
                            "score depending on a third party's model version is "
                            "not reproducible once that version is deprecated",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", help="webarena/test.raw.json. Defaults to the "
                                          "copy inside the installed libwebarena")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    if args.configs:
        raw = Path(args.configs).read_bytes()
    else:
        from slime_htm.envs.gate import _webarena_configs
        raw = json.dumps(_webarena_configs()).encode()
    configs = json.loads(raw)

    plan = build(configs)
    # The digest is the point: the verdict is only pre-registered against *this*
    # task file, and libwebarena shipping a different one has to invalidate it
    # rather than leave a stale count standing.
    plan["task_file"] = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "tasks": len(configs),
        "source": args.configs or "installed libwebarena",
    }
    Path(args.output).write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")

    print(f"{'site':16}{'usable':>7}{'no typed':>10}{'intents':>9}"
          f"{'answer':>8}{'page':>6}{'query':>7}{'GB':>6}")
    for name, row in plan["sites"].items():
        print(f"{name:16}{row['usable']:>7}{row['no_typed_value_required']:>10}"
              f"{row['distinct_intents_there']:>9}{row['needs_sent_answer']:>8}"
              f"{row['needs_page_content']:>6}{row['needs_url_query']:>7}"
              f"{row['image_gb_compressed']:>6.1f}")
    print(f"\ngateable with a click-only fast mode: "
          f"{plan['gateable_with_a_click_only_fast_mode']}")
    print(plan["verdict"])
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
