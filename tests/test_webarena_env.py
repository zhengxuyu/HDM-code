"""Standing WebArena up, without Docker and without a site.

The parts that can be wrong with no container running are the ones that would
cost a 5-day job: a missing seventh variable, a port that does not match what the
task definitions expect, and "the container answered" being mistaken for "the
site works".
"""
import json

import pytest

from slime_htm.envs import webarena_env as wa


def test_all_seven_variables_are_produced_including_the_one_no_image_serves():
    """`gate.VENUES["webarena"]` refuses on any unset variable, and `WA_HOMEPAGE`
    has no container behind it -- in the original repo it is a static page. Six
    URLs and a missing seventh is a run that cannot start."""
    from slime_htm.envs.gate import VENUES

    values = wa.environment("node-7")
    assert set(values) == set(VENUES["webarena"]["requires_env"])
    assert len(values) == 7
    assert values["WA_HOMEPAGE"] == "http://node-7:4399"


def test_the_ports_are_webarenas_own_because_the_tasks_hard_code_them():
    """A `url_match` evaluator compares against an expected URL that carries the
    port. Serving shopping on anything but 7770 makes those tasks fail in a way
    that reads as an agent error."""
    values = wa.environment("h")
    assert values["WA_SHOPPING"] == "http://h:7770"
    assert values["WA_REDDIT"] == "http://h:9999"
    assert values["WA_GITLAB"] == "http://h:8023"
    # shopping_admin's tasks start inside the admin panel, not at the storefront.
    assert values["WA_SHOPPING_ADMIN"] == "http://h:7780/admin"


def test_an_unknown_site_is_refused_rather_than_silently_dropped():
    with pytest.raises(ValueError, match="not WebArena sites"):
        wa.environment("h", sites=["shoping"])


def test_the_control_plane_is_not_the_site():
    """Two published ports per container: 80 is the site, 8877 is the control
    plane the reset goes through. Pointing browsergym at the second gives it a
    status page instead of a store."""
    assert wa.site_url("h", "reddit") == "http://h:9999"
    assert wa.control_url("h", "reddit") == "http://h:9998"
    run = wa.docker_run("reddit")
    assert "0.0.0.0:9999:80" in run and "0.0.0.0:9998:8877" in run
    assert run[-1] == "am1n3e/webarena-verified-reddit"


def test_answering_is_not_the_same_as_being_ready(monkeypatch):
    """The probe found `/status` answering 200 while elasticsearch was still
    coming up. A task run then fails, and it looks like the agent's fault."""
    body = {"success": True, "details": {"value": {"services": {
        "nginx": "RUNNING", "mysqld": "HEALTHY", "elasticsearch": "STARTING"}}}}
    monkeypatch.setattr(wa, "_get", lambda *a, **k: (200, body))

    assert wa.unhealthy(wa.status("h", "reddit")) == ["elasticsearch"]
    assert wa.unhealthy({"nginx": "RUNNING", "redis-server": "HEALTHY"}) == []
    assert wa.unhealthy(None) == ["(no status)"], "no status is not healthy"


def test_an_unreachable_or_broken_control_plane_reads_as_not_ready(monkeypatch):
    monkeypatch.setattr(wa, "_get", lambda *a, **k: (None, "connection refused"))
    assert wa.status("h", "reddit") is None
    monkeypatch.setattr(wa, "_get", lambda *a, **k: (200, "<html>not json</html>"))
    assert wa.status("h", "reddit") is None, "a page is not a status"
    monkeypatch.setattr(wa, "_get", lambda *a, **k: (500, {"success": False}))
    assert wa.status("h", "reddit") is None


def test_wait_ready_gives_up_and_names_what_never_came_up(monkeypatch):
    """Returning rather than raising, so a launcher can report the service. A
    5-day job that dies with a traceback and no service name costs a day to
    diagnose."""
    body = {"details": {"value": {"services": {"mysqld": "FATAL"}}}}
    monkeypatch.setattr(wa, "_get", lambda *a, **k: (200, body))
    clock = iter([0.0, 10.0, 20.0, 30.0, 40.0])
    slept = []

    ready, problem = wa.wait_ready("h", "reddit", timeout=25.0, interval=10.0,
                                   now=lambda: next(clock), sleep=slept.append)
    assert ready is False and problem == ["mysqld"]
    assert slept, "it waited between probes rather than spinning"


def test_wait_ready_returns_as_soon_as_every_service_is_up(monkeypatch):
    body = {"details": {"value": {"services": {"nginx": "RUNNING"}}}}
    monkeypatch.setattr(wa, "_get", lambda *a, **k: (200, body))
    slept = []
    ready, problem = wa.wait_ready("h", "reddit", timeout=100.0,
                                   now=lambda: 0.0, sleep=slept.append)
    assert (ready, problem, slept) == (True, [], [])


def test_the_reset_reports_its_cost_and_does_not_claim_to_isolate(monkeypatch):
    """`POST /init` did not return inside 10 s in the probe, and whether it
    *restores* state or merely initialises is unestablished. The function returns
    the elapsed time because that number is what decides whether per-task reset
    fits in a run at all, and its docstring must not promise isolation."""
    monkeypatch.setattr(wa, "_get", lambda *a, **k: (200, {"success": True}))
    code, body, seconds = wa.reinitialise("h", "reddit")
    assert code == 200 and seconds >= 0.0

    doc = wa.reinitialise.__doc__
    assert "not established" in doc, "the unknown is stated where it is used"
    assert "preflight" in doc, "and it names what would establish it"


def test_the_homepage_is_written_because_no_image_ships_it(tmp_path):
    path = wa.write_homepage(tmp_path / "wa" / "index.html", "node-7")
    html = path.read_text()
    for site in wa.SITES:
        assert site in html, f"{site} is linked from the start page"
    assert "http://node-7:7780/admin" in html


def test_the_plan_says_webarena_cannot_gate_a_click_only_fast_mode():
    """The finding, in the artefact a reader will have. An earlier version of the
    plan picked `reddit` because it had "95 page/url tasks against 9 needing text,
    so never is not pinned at zero". Those 95 are scored on content that has to
    appear on the page, which on a forum is a post body somebody typed:
    code-reviewer caught it, and counted properly reddit has **zero**. Every site
    is short, so it is the action space and not the site choice."""
    from pathlib import Path

    plan = json.loads(Path(__file__).resolve().parent.parent.joinpath(
        "src/slime_htm/envs/webarena-plan.json").read_text())

    assert plan["gateable_with_a_click_only_fast_mode"] is False
    assert plan["verdict"].startswith("NO.")
    assert plan["min_pairs"] == 20
    assert plan["sites"]["reddit"]["no_typed_value_required"] == 0, \
        "the claim that made reddit look right"
    # Distinct intents, not task ids: shopping's 32 are 8 fillings of templates,
    # and the sign test pairs per task, so 32 would overstate the power by 4x.
    assert plan["sites"]["shopping"]["no_typed_value_required"] == 32
    assert plan["sites"]["shopping"]["distinct_intents_there"] == 8
    assert plan["best_site"] == "shopping"
    assert max(row["distinct_intents_there"]
               for row in plan["sites"].values()) < plan["min_pairs"]
    # And what would change it, including the option the user declined, named.
    assert any("option (b)" in item for item in plan["what_would_change_this"])
    assert plan["task_file"]["tasks"] == 812
    assert len(plan["task_file"]["sha256"]) == 64


def test_every_image_is_pinned_by_digest_not_by_tag():
    """code-reviewer: these come from an individual Docker Hub account and a tag
    can be repushed under a run. A result whose sites changed underneath it and
    said nothing is worse than a run that stops."""
    from pathlib import Path

    plan = json.loads(Path(__file__).resolve().parent.parent.joinpath(
        "src/slime_htm/envs/webarena-plan.json").read_text())

    for site, row in plan["sites"].items():
        assert row["digest"].startswith("sha256:"), site
        assert len(row["digest"]) == 71, site
    assert len({row["digest"] for row in plan["sites"].values()}) == 6, "all distinct"


def test_the_login_dependency_is_recorded_because_nothing_earlier_recorded_it():
    """All 812 tasks set `require_login`, so an authenticated `storage_state` has
    to exist before any episode. Found while checking the reset question."""
    from pathlib import Path

    plan = json.loads(Path(__file__).resolve().parent.parent.joinpath(
        "src/slime_htm/envs/webarena-plan.json").read_text())
    assert "812" in plan["login"]["require_login"]
    assert "storage_state" in plan["login"]["note"]
    # The reset question stays open rather than being answered by the task file.
    assert "UNVERIFIED" in plan["state_reset"]["but"]
    assert "NOT established" in plan["state_reset"]["open"]
