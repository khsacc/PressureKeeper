"""build_app() wiring not already covered by test_controller_unit.py's
logging-failure test: the single-instance guard against a second real
(non-simulator) process on the same PACE5000 endpoint. Genuinely distinct
*process* semantics (a foreign live vs. stale pid) are already covered in
detail by test_instance_lock.py; this only checks build_app() wires the
right lock path from config and actually propagates/releases it."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from pressurekeeper.app import build_app
from pressurekeeper.clock import FakeClock
from pressurekeeper.instance_lock import InstanceAlreadyRunning, lock_path_for

from .helpers import make_config


def test_rejected_while_a_foreign_live_process_holds_the_lock(tmp_path):
    config = make_config(tmp_path, dry_run=True)
    lock_path = lock_path_for(config.pace5000_api.base_url, tmp_path)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps({"pid": proc.pid, "acquired_at_wall": 0.0}))
        with pytest.raises(InstanceAlreadyRunning):
            build_app(config, use_simulator=False, dry_run=True, clock=FakeClock(0.0), lock_dir=tmp_path)
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_released_on_close_so_a_later_instance_can_start(tmp_path):
    config = make_config(tmp_path, dry_run=True)
    ctx1 = build_app(config, use_simulator=False, dry_run=True, clock=FakeClock(0.0), lock_dir=tmp_path)
    lock_path = lock_path_for(config.pace5000_api.base_url, tmp_path)
    assert lock_path.exists()
    ctx1.close()
    assert not lock_path.exists(), "close() must release the instance lock"


def test_simulator_instances_are_exempt_from_the_lock(tmp_path):
    config = make_config(tmp_path, dry_run=False)
    ctx1 = build_app(config, use_simulator=True, dry_run=False, clock=FakeClock(0.0), lock_dir=tmp_path)
    ctx2 = build_app(config, use_simulator=True, dry_run=False, clock=FakeClock(0.0), lock_dir=tmp_path)
    ctx1.close()
    ctx2.close()


def test_lock_path_is_independent_of_logging_directory(tmp_path):
    # Regression: the lock used to live under config.logging.directory, so
    # two processes launched with different --config files (different log
    # directories) but the same PACE5000 endpoint never collided at all --
    # see instance_lock.py's default_lock_dir() docstring.
    config_a = make_config(tmp_path / "run_a", dry_run=True)
    config_b = make_config(tmp_path / "run_b", dry_run=True)
    assert config_a.logging.directory != config_b.logging.directory
    assert (
        lock_path_for(config_a.pace5000_api.base_url, tmp_path)
        == lock_path_for(config_b.pace5000_api.base_url, tmp_path)
    )
