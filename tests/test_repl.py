"""Unit tests for the REPL's server-startup decision logic.

The REPL must never spawn its own uvicorn while the systemd unit owns the port:
a spawned process that outlives the wait orphans and squats 8123, sending the
managed unit into an unbindable restart loop. `_startup_action` encodes that
policy purely so it can be tested without touching the network or systemd.
"""

import pytest

from hosaka.cli.repl import _startup_action


def test_attach_when_server_already_up():
    # Health passes -> attach, regardless of unit state.
    assert _startup_action(True, "loaded", "active") == "attach"
    assert _startup_action(True, "not-found", None) == "attach"


def test_spawn_only_when_no_managed_unit():
    # No systemd unit (or systemctl absent) -> REPL owns the server.
    assert _startup_action(False, "not-found", None) == "spawn"
    assert _startup_action(False, None, None) == "spawn"
    assert _startup_action(False, "masked", "inactive") == "spawn"


@pytest.mark.parametrize("active", ["active", "activating", "reloading", "deactivating"])
def test_wait_when_unit_is_providing(active):
    # Unit installed and up / coming up -> wait, never compete for the port.
    # The regression: "activating" (systemd auto-restart) previously spawned an
    # orphan that squatted the port and broke the managed unit.
    assert _startup_action(False, "loaded", active) == "wait"


@pytest.mark.parametrize("active", ["failed", "inactive"])
def test_start_unit_when_installed_but_down(active):
    # Unit installed but stopped/failed -> ask systemd to start it, don't spawn.
    assert _startup_action(False, "loaded", active) == "start_unit"
