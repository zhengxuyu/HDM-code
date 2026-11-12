"""Recycling browsergym's global Playwright, against the real driver.

These need `playwright` installed and skip without it. They are separate from the
rest of the browser tests on purpose: the defect they cover was **hidden by a
stub**. `forget_global_playwright` first only set browsergym's global to None, and
the gate test that was supposed to prove a wedged close still reaches a verdict
passed because its fake global had no `stop` to forget to call. Against the real
library, the next `.start()` then raises "It looks like you are using Playwright
Sync API inside the asyncio loop", which with recycling on by default crashed
every run at episode 26.

No browser is launched here -- only the driver process -- so these cost about a
second and need no chromium.
"""
import pytest

playwright_api = pytest.importorskip("playwright.sync_api",
                                     reason="playwright is not installed here")


@pytest.fixture
def browsergym_global(monkeypatch):
    """A stand-in for `browsergym.core`'s module global, with its real API."""
    import sys
    import types

    core = types.ModuleType("browsergym.core")
    core._PLAYWRIGHT = None
    core._set_global_playwright = lambda value: setattr(core, "_PLAYWRIGHT", value)
    package = types.ModuleType("browsergym")
    package.core = core
    monkeypatch.setitem(sys.modules, "browsergym", package)
    monkeypatch.setitem(sys.modules, "browsergym.core", core)
    yield core
    if core._PLAYWRIGHT is not None:
        try:
            core._PLAYWRIGHT.stop()
        except Exception:  # noqa: BLE001  the test's own cleanup
            pass


def test_two_starts_in_one_thread_is_the_defect_being_guarded_against():
    """The thing that made dropping the reference insufficient, stated as a fact
    about the library rather than as a claim in a comment."""
    first = playwright_api.sync_playwright().start()
    try:
        with pytest.raises(Exception, match="asyncio loop"):
            playwright_api.sync_playwright().start()
    finally:
        first.stop()
    # And a start after a stop is fine, which is why stopping is the fix.
    second = playwright_api.sync_playwright().start()
    second.stop()


def test_forgetting_the_global_stops_it_so_the_next_start_works(browsergym_global):
    """What every recycle and every close-timeout recovery depends on."""
    from slime_htm.envs.browser import forget_global_playwright

    browsergym_global._set_global_playwright(playwright_api.sync_playwright().start())
    assert forget_global_playwright() is True
    assert browsergym_global._PLAYWRIGHT is None

    fresh = playwright_api.sync_playwright().start()
    try:
        assert fresh is not None, "a driver can be started again"
    finally:
        fresh.stop()


def test_it_works_after_the_driver_process_has_been_killed(browsergym_global):
    """The recovery path's real case: `kill_browser_children` has just killed the
    driver, and the cached object is talking to a corpse. `stop()` returns at once
    and a new start succeeds."""
    import os
    import signal
    import subprocess

    from slime_htm.envs.browser import forget_global_playwright

    browsergym_global._set_global_playwright(playwright_api.sync_playwright().start())
    # Only the driver, by name. `pgrep -P` alone would kill every child of the
    # test process, which in somebody else's CI is not only the driver -- a test
    # that can kill its runner's other children is not a test to ship.
    listing = subprocess.run(["pgrep", "-lP", str(os.getpid())],
                             capture_output=True, text=True).stdout.splitlines()
    drivers = [line.split()[0] for line in listing
               if "node" in line.lower() or "playwright" in line.lower()]
    if not drivers:
        pytest.skip("the driver process is not a child here, so it cannot be killed")
    for pid in drivers:
        os.kill(int(pid), signal.SIGKILL)

    assert forget_global_playwright(stop_timeout=15) is True
    fresh = playwright_api.sync_playwright().start()
    try:
        assert fresh is not None, "a killed driver does not poison the process"
    finally:
        fresh.stop()


def test_a_missing_browsergym_is_not_an_error(monkeypatch):
    """It runs during a recovery, so it reports rather than raising."""
    import sys

    from slime_htm.envs.browser import forget_global_playwright

    monkeypatch.setitem(sys.modules, "browsergym", None)
    monkeypatch.setitem(sys.modules, "browsergym.core", None)
    assert forget_global_playwright() is False
