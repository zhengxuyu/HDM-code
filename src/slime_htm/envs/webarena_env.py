"""Standing up WebArena's sites, and the seven URLs browsergym needs to find them.

`gate.VENUES["webarena"]` requires seven environment variables and refuses to run
without them. Six name site containers; the seventh, `WA_HOMEPAGE`, names a page
that **no image provides** -- in the original repository it is a small static page
listing the sites, and browsergym opens it as the starting point. So this module
both describes the containers and writes that page, because a missing seventh
variable is a run that cannot start.

What is verified and what is not, kept apart on purpose -- a sentence here
asserting a property the code lacks is the defect shape this repository keeps
paying for:

**Verified on 2026-10-06**, by pulling `am1n3e/webarena-verified-shopping_admin`
on a compute node and reading it:

  - the container runs the original stack -- mysqld, php-fpm 8.1, nginx, redis,
    elasticsearch, cron, mailcatcher under supervisord -- so the site data is
    WebArena's own and not a reimplementation
  - it publishes the site on 80 and a **control plane** on 8877, whose routes are
    `/status`, `/is_ready`, `/init`, `/start`, `/stop`, `/restart`
  - `/status` returns `{"success", "message", "details": {"value": {"services":
    {...}}}}` with a state per service, `RUNNING` or `HEALTHY`
  - rootless Docker on the cluster's compute nodes runs these images, and a
    published port answers on the node's routable address, so the sites and the
    agent may live in different jobs

**Not verified, and therefore not relied on anywhere below:**

  - what `POST /init` does. It did not return inside 10 s, so it is a long
    operation; whether it *restores the initial state* or merely initialises a
    fresh container is unknown. `reinitialise` calls it and says in its own
    docstring that the semantics are unestablished. **No run may claim per-task
    state isolation until `scripts/webarena_preflight.py` has measured it.**
  - whether the original task definitions in `libwebarena` score correctly
    against these images. The images come from `webarena-verified`, which
    rewrote the task definitions and evaluators; the site data looking original
    is evidence, not proof. The preflight checks one task end to end.
"""

import json
import time
from urllib import error, request

# site -> (image, the port the site is published on, the control-plane port,
#          the environment variable browsergym reads)
#
# The ports are WebArena's own conventions, which its task definitions hard-code
# into expected URLs: changing one makes `url_match` evaluators fail in a way that
# looks like an agent error. shopping_admin's site lives under /admin.
SITES = {
    "shopping": ("am1n3e/webarena-verified-shopping", 7770, 7771, "WA_SHOPPING"),
    "shopping_admin": ("am1n3e/webarena-verified-shopping_admin", 7780, 7781,
                       "WA_SHOPPING_ADMIN"),
    "reddit": ("am1n3e/webarena-verified-reddit", 9999, 9998, "WA_REDDIT"),
    "gitlab": ("am1n3e/webarena-verified-gitlab", 8023, 8024, "WA_GITLAB"),
    "wikipedia": ("am1n3e/webarena-verified-wikipedia", 8888, 8889, "WA_WIKIPEDIA"),
    "map": ("am1n3e/webarena-verified-map", 3000, 3001, "WA_MAP"),
}

# The path each site is entered at. shopping_admin's tasks start in the admin
# panel, and wikipedia's are written against kiwix's search entry point.
SITE_PATH = {
    "shopping_admin": "/admin",
    "wikipedia": "/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing",
}

HEALTHY = ("RUNNING", "HEALTHY")


def site_url(host, site):
    """Where browsergym should point for one site."""
    _, port, _, _ = SITES[site]
    return f"http://{host}:{port}{SITE_PATH.get(site, '')}"


def control_url(host, site):
    """The container's control plane, which is not the site."""
    _, _, control, _ = SITES[site]
    return f"http://{host}:{control}"


def environment(host, homepage_port=4399, sites=None):
    """The seven variables `gate.VENUES["webarena"]` requires.

    All seven, including `WA_HOMEPAGE`, which no image serves -- `write_homepage`
    produces the file and the caller serves it. `sites` restricts which hosts are
    named, for a run that stands up one site: the other variables still have to
    be set, because `missing_environment` refuses on any unset one, and a URL
    pointing at a container that is not running fails at the task rather than at
    startup, which is the honest order. A task needing an absent site is not in
    the pre-registered sample.
    """
    chosen = set(sites) if sites is not None else set(SITES)
    unknown = chosen - set(SITES)
    if unknown:
        raise ValueError(f"not WebArena sites: {sorted(unknown)}")
    values = {variable: site_url(host, name)
              for name, (_, _, _, variable) in SITES.items()}
    values["WA_HOMEPAGE"] = f"http://{host}:{homepage_port}"
    return values


def write_homepage(path, host, homepage_port=4399):
    """The seventh URL's content, which the images do not ship.

    WebArena's own homepage is a list of links to the sites; browsergym opens it
    as the start page and some intents refer to it. Written rather than fetched so
    the run does not depend on a page that is not in any image.
    """
    links = "\n".join(
        f'    <li><a href="{site_url(host, name)}">{name}</a></li>'
        for name in SITES)
    html = ("<!DOCTYPE html>\n<html>\n<head><meta charset=\"utf-8\">\n"
            "<title>WebArena Homepage</title></head>\n<body>\n"
            "  <h1>WebArena</h1>\n  <ul>\n"
            f"{links}\n  </ul>\n</body>\n</html>\n")
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html)
    return target


def _get(url, timeout, method="GET"):
    """One request, returning (status, parsed-or-text). Never raises on HTTP status."""
    req = request.Request(url, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
            code = response.status
    except error.HTTPError as problem:
        body, code = problem.read().decode("utf-8", "replace"), problem.code
    except (error.URLError, TimeoutError, OSError) as problem:
        return None, str(problem)
    try:
        return code, json.loads(body)
    except json.JSONDecodeError:
        return code, body


def status(host, site, timeout=20.0):
    """Per-service health from the control plane, or None if it cannot be reached."""
    code, body = _get(f"{control_url(host, site)}/status", timeout)
    if code != 200 or not isinstance(body, dict):
        return None
    value = ((body.get("details") or {}).get("value") or {})
    services = value.get("services")
    return services if isinstance(services, dict) else None


def unhealthy(services):
    """Which services are not in a good state. Empty means every one is.

    A separate function because "the container answered" and "the site works" are
    different claims: the probe found `/status` answering 200 while elasticsearch
    was still starting, and a task run then would have failed as an agent error.
    """
    if not services:
        return ["(no status)"]
    return sorted(name for name, state in services.items()
                  if str(state).upper() not in HEALTHY)


def wait_ready(host, site, timeout=600.0, interval=5.0, now=time.monotonic,
               sleep=time.sleep):
    """Block until every service reports healthy. Returns (ready, last problem).

    Ten minutes by default: the probe's shopping_admin took tens of seconds to
    bring elasticsearch and mysqld up, and a larger site is slower. Returning
    rather than raising, so a launcher can report which service never came up.
    """
    deadline = now() + timeout
    problem = ["(not probed)"]
    while now() < deadline:
        problem = unhealthy(status(host, site, timeout=min(interval * 2, 20.0)))
        if not problem:
            return True, []
        sleep(interval)
    return False, problem


def reinitialise(host, site, timeout=900.0):
    """POST /init, which is the control plane's heaviest route.

    **The semantics are not established.** The container runs
    `environment_control.cli init` at startup, and `/init` is the same operation
    over HTTP; it did not return inside 10 s in the probe, so it is slow. Whether
    it *restores the initial state* -- which is what per-task isolation would
    need -- or merely initialises a container that is already initialised, is
    exactly what `scripts/webarena_preflight.py` measures. Until it has, a run
    using this between tasks must not describe the tasks as independent.

    Returns (status code or None, body or error text, seconds).
    """
    started = time.monotonic()
    code, body = _get(f"{control_url(host, site)}/init", timeout, method="POST")
    return code, body, time.monotonic() - started


def docker_run(site, host_bind="0.0.0.0"):
    """The command that starts one site, as a list, for a launcher to run.

    Here rather than in a `.sbatch` file so the ports and the image stay in one
    place with the variables that point at them; the launcher is a wrapper. Both
    ports are published: the site, and the control plane the reset goes through.
    """
    image, port, control, _ = SITES[site]
    return ["docker", "run", "-d", "--name", f"webarena-{site}",
            "-p", f"{host_bind}:{port}:80",
            "-p", f"{host_bind}:{control}:8877", image]
