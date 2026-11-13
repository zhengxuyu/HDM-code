"""Before any WebArena run: establish the two things that are currently unknown.

Standing the sites up is solved -- rootless Docker runs the images and the control
plane reports health. What is **not** established, and what this measures, is:

  1. **Does `POST /init` restore the initial state?** Per-task isolation depends
     on it. The test is destructive on purpose: take a fingerprint, change the
     site's database, confirm the fingerprint moved, call `/init`, and see whether
     it moved back. Without the change in the middle, a fingerprint that is equal
     before and after proves nothing -- `/init` doing nothing looks identical to
     `/init` restoring a state that was already initial. That is the trap this
     script exists to avoid.
  2. **Do the original task definitions score against these images?** The images
     come from `webarena-verified`, which rewrote the task definitions and the
     evaluators, while `gate.py` reads `libwebarena`'s original `test.raw.json`.
     The site stack looking original is evidence, not proof. This registers the
     pre-registered sample through browsergym and reports how many of the ids
     actually exist.

Every check reports `pass`, `fail` or **`unverified`**, and `unverified` is not a
pass. A run may not claim per-task isolation until check 1 says `pass` here.

    # inside the job that started the containers
    python scripts/webarena_preflight.py --site reddit -o results/wa-preflight.json
"""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

from slime_htm.envs import webarena_env as wa

# How each site's data is counted. The credentials are not secrets -- WebArena
# publishes them, and `shopping_admin`'s own /status output prints
# `mysqladmin ping -u magentouser -pMyPassword` -- but they are per-image, so a
# site with no entry here reports `unverified` rather than guessing.
FINGERPRINT = {
    "reddit": ("psql -U postgres -d postmill -tAc "
               "'select count(*) from submissions'"),
    "shopping": ("mysql -u magentouser -pMyPassword magentodb -N -s -e "
                 "'select count(*) from sales_order'"),
    "shopping_admin": ("mysql -u magentouser -pMyPassword magentodb -N -s -e "
                       "'select count(*) from sales_order'"),
}
# The change in the middle: reversible in principle, and irrelevant to any task's
# score, but enough to move the count. A delete of the newest row is used rather
# than an insert, because an insert can be rejected by a constraint and then the
# fingerprint would not move for a reason that is not about `/init`.
DISTURB = {
    "reddit": ("psql -U postgres -d postmill -tAc "
               "'delete from submissions where id in "
               "(select id from submissions order by id desc limit 1)'"),
    "shopping": ("mysql -u magentouser -pMyPassword magentodb -N -s -e "
                 "'delete from sales_order order by entity_id desc limit 1'"),
    "shopping_admin": ("mysql -u magentouser -pMyPassword magentodb -N -s -e "
                       "'delete from sales_order order by entity_id desc limit 1'"),
}


def container(site):
    return f"webarena-{site}"


def inside(site, command, timeout=120):
    """Run a command in the site's container. Returns (ok, output)."""
    try:
        done = subprocess.run(
            ["docker", "exec", container(site), "sh", "-c", command],
            capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as problem:
        return False, str(problem)
    if done.returncode != 0:
        return False, (done.stderr or done.stdout).strip()[:300]
    return True, done.stdout.strip()


def count(site):
    """The site's fingerprint, or None when it cannot be taken."""
    query = FINGERPRINT.get(site)
    if not query:
        return None
    ok, output = inside(site, query)
    if not ok:
        return None
    found = re.search(r"-?\d+", output)
    return int(found.group()) if found else None


def check_ready(host, site, timeout=900.0):
    sites_up, problem = wa.wait_ready(host, site, timeout=timeout)
    return {"check": "every service healthy",
            "result": "pass" if sites_up else "fail",
            "unhealthy": problem}


def check_variables(host, site, output_dir):
    """All seven set, and the one no image serves actually written."""
    from slime_htm.envs.gate import VENUES, missing_environment

    values = wa.environment(host)
    page = wa.write_homepage(Path(output_dir) / "homepage" / "index.html", host)
    missing = missing_environment("webarena", values)
    return {"check": "seven variables and a homepage",
            "result": "pass" if not missing else "fail",
            "missing": missing,
            "required": list(VENUES["webarena"]["requires_env"]),
            "homepage_file": str(page),
            "note": "WA_HOMEPAGE has no container; this file is what must be "
                    "served on its port"}


def check_digest(site, plan):
    """Is the running container the image the plan was written against?

    code-reviewer's condition: these images live under an individual Docker Hub
    account and a tag can be repushed under a run. A result whose sites changed
    underneath it and said nothing is worse than a run that stops, so a mismatch
    is a `fail` and the preflight's verdict is not `ready`.
    """
    expected = (plan.get("sites", {}).get(site) or {}).get("digest")
    row = {"check": "the image is the digest the plan was written against",
           "expected": expected}
    if not expected:
        row["result"] = "unverified"
        row["why"] = f"the plan records no digest for {site!r}"
        return row
    ok, output = inside(site, "true")  # is the container even there
    if not ok:
        row["result"] = "unverified"
        row["why"] = f"container {container(site)} is not running: {output}"
        return row
    try:
        done = subprocess.run(
            ["docker", "inspect", "--format", "{{index .RepoDigests 0}}",
             container(site)],
            capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as problem:
        row["result"] = "unverified"
        row["why"] = str(problem)
        return row
    found = done.stdout.strip()
    row["found"] = found
    row["result"] = "pass" if expected in found else "fail"
    if row["result"] == "fail":
        row["why"] = ("the tag was repushed, or this container was started from "
                      "another image. The plan's counts describe a different "
                      "build and the run must not continue on them")
    return row


def check_tasks(site, plan):
    """Do the pre-registered ids exist in the registry browsergym builds?"""
    ids = (plan.get("sites", {}).get(site) or {}).get("no_typed_value_task_ids") or []
    wanted = [f"browsergym/webarena.{index}" for index in ids]
    if not wanted:
        return {"check": "libwebarena's task ids exist in this registry",
                "result": "unverified",
                "why": f"the plan lists no click-only ids for {site!r}, so there "
                       f"is nothing to look up -- which is itself the finding "
                       f"for that site"}
    try:
        import browsergym.webarena  # noqa: F401
        import gymnasium as gym
    except ImportError as problem:
        return {"check": "libwebarena's task ids exist in this registry",
                "result": "unverified", "why": f"not importable here: {problem}"}
    known = set(gym.registry)
    found = [name for name in wanted if name in known]
    return {"check": "the pre-registered tasks are registered",
            "result": "pass" if len(found) == len(wanted) else "fail",
            "wanted": len(wanted), "found": len(found),
            "absent": [name for name in wanted if name not in known][:10],
            "note": "the ids come from libwebarena's original test.raw.json while "
                    "the images come from webarena-verified, which rewrote the "
                    "task definitions -- an absent id means the two are not one "
                    "benchmark and the plan has to be rebuilt against whichever "
                    "is being run"}


def check_reset(host, site):
    """The destructive test. A fingerprint equal on both sides proves nothing."""
    row = {"check": "POST /init restores the initial state", "result": "unverified"}
    before = count(site)
    if before is None:
        row["why"] = (f"no fingerprint for {site!r}, or the query failed, so "
                      f"whether /init restores state is still unknown")
        return row
    row["before"] = before

    ok, output = inside(site, DISTURB[site])
    if not ok:
        row["why"] = f"could not change the site's data: {output}"
        return row
    disturbed = count(site)
    row["after_change"] = disturbed
    if disturbed == before:
        # Without a moved fingerprint the rest of the test is uninformative, and
        # reporting it as a pass is how this check would lie.
        row["why"] = ("the change did not move the fingerprint, so this run "
                      "cannot tell a restoring /init from an inert one")
        return row

    code, body, seconds = wa.reinitialise(host, site)
    row["init_status"] = code
    row["init_seconds"] = round(seconds, 1)
    row["init_body"] = body if isinstance(body, dict) else str(body)[:200]
    ready, problem = wa.wait_ready(host, site, timeout=900.0)
    row["healthy_after_init"] = ready
    row["unhealthy_after_init"] = problem
    restored = count(site)
    row["after_init"] = restored
    row["result"] = "pass" if restored == before else "fail"
    row["note"] = (f"a reset costs {row['init_seconds']}s, so per-task reset over "
                   f"40 tasks costs {round(row['init_seconds'] * 40 / 60, 1)} "
                   f"minutes of the run" if row["result"] == "pass" else
                   "/init did not restore the count, so tasks are NOT independent "
                   "and a run must either reset by recreating the container or "
                   "report that state carries between tasks")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--site", default=None,
                        help="defaults to the plan's best_site -- the one with the "
                             "most click-only-reachable intents, which the plan "
                             "finds is still too few to gate on")
    parser.add_argument("--plan",
                        default="src/slime_htm/envs/webarena-plan.json")
    parser.add_argument("--ready-timeout", type=float, default=900.0,
                        help="seconds to wait for every service. The largest "
                             "images take minutes; a short value is for checking "
                             "this script against nothing running")
    parser.add_argument("--skip-reset", action="store_true",
                        help="the reset test changes the site's data. Skipping it "
                             "leaves check 1 `unverified`, which is not a pass")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    plan = json.loads(Path(args.plan).read_text())
    site = args.site or plan["best_site"]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"preflight on {site} at {args.host}", flush=True)
    # Said up front, because the preflight passing does not make the gate
    # runnable: the plan's verdict is that a click-only fast mode cannot be gated
    # on this venue at all. What the preflight establishes is the environment --
    # which any WebArena use needs, including one with a typing fast mode.
    if not plan.get("gateable_with_a_click_only_fast_mode", True):
        print(f"  note: {plan['verdict']}", flush=True)
        print("  this preflight establishes the environment, not the gate",
              flush=True)
    report = {"site": site, "host": args.host, "plan": args.plan,
              "task_file_sha256": plan["task_file"]["sha256"],
              "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "checks": []}

    for step in (check_ready(args.host, site, args.ready_timeout),
                 check_digest(site, plan),
                 check_variables(args.host, site, out.parent),
                 check_tasks(site, plan)):
        report["checks"].append(step)
        print(f"  {step['result']:<11} {step['check']}", flush=True)

    if args.skip_reset:
        report["checks"].append({
            "check": "POST /init restores the initial state",
            "result": "unverified", "why": "--skip-reset was passed"})
        print("  unverified  POST /init restores the initial state (skipped)",
              flush=True)
    else:
        step = check_reset(args.host, site)
        report["checks"].append(step)
        print(f"  {step['result']:<11} {step['check']}"
              f"{'  ' + str(step.get('init_seconds')) + 's' if step.get('init_seconds') else ''}",
              flush=True)
        if step.get("why"):
            print(f"              {step['why']}", flush=True)

    results = [step["result"] for step in report["checks"]]
    report["verdict"] = ("ready" if all(r == "pass" for r in results)
                         else "not ready")
    report["may_claim_task_independence"] = any(
        step["result"] == "pass"
        and step["check"].startswith("POST /init") for step in report["checks"])
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\n{report['verdict']}: {results.count('pass')}/{len(results)} passed; "
          f"task independence claimable: {report['may_claim_task_independence']}",
          flush=True)
    print(f"wrote {out}", flush=True)
    return 0 if report["verdict"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
