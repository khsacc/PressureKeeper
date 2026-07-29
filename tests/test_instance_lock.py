from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pressurekeeper.instance_lock import InstanceAlreadyRunning, SingleInstanceLock, default_lock_dir, lock_path_for


def test_lock_path_for_derives_from_host_and_port(tmp_path):
    p = lock_path_for("http://192.168.1.5:8765", tmp_path)
    assert p.parent == tmp_path
    assert "192.168.1.5" in p.name
    assert "8765" in p.name


def test_lock_path_for_defaults_to_a_fixed_per_user_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    p = lock_path_for("http://10.0.0.5:8765")
    assert p.parent == default_lock_dir() == tmp_path / ".pressurekeeper" / "locks"


def test_acquire_succeeds_over_an_unreadable_or_corrupt_lock_file(tmp_path):
    lock_path = tmp_path / "lock"
    lock_path.write_text("not valid json {{{")
    lock = SingleInstanceLock(lock_path)
    lock.acquire()  # must not raise -- an unreadable file is treated as stale
    recorded = json.loads(lock_path.read_text())
    assert recorded["pid"] == os.getpid()


def test_acquire_writes_pid_and_release_removes_it(tmp_path):
    lock = SingleInstanceLock(tmp_path / "lock")
    lock.acquire()
    assert lock.path.exists()
    recorded = json.loads(lock.path.read_text())
    assert recorded["pid"] == os.getpid()
    lock.release()
    assert not lock.path.exists()


def test_acquire_twice_in_the_same_process_does_not_raise(tmp_path):
    # A caller re-acquiring its own lock (e.g. re-entrant construction in a
    # single process) must not be treated as a foreign instance.
    lock_path = tmp_path / "lock"
    SingleInstanceLock(lock_path).acquire()
    SingleInstanceLock(lock_path).acquire()  # same pid -- must not raise


def test_acquire_raises_while_a_live_foreign_pid_holds_it(tmp_path):
    lock_path = tmp_path / "lock"
    # A real, currently-alive PID that is not this test process.
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        lock_path.write_text(json.dumps({"pid": proc.pid, "acquired_at_wall": 0.0}))
        with pytest.raises(InstanceAlreadyRunning):
            SingleInstanceLock(lock_path).acquire()
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_acquire_succeeds_over_a_stale_lock_from_a_dead_pid(tmp_path):
    lock_path = tmp_path / "lock"
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    dead_pid = proc.wait(timeout=5)
    assert dead_pid is not None  # process has exited
    lock_path.write_text(json.dumps({"pid": proc.pid, "acquired_at_wall": 0.0}))

    lock = SingleInstanceLock(lock_path)
    lock.acquire()  # must not raise -- the previous holder is gone
    recorded = json.loads(lock_path.read_text())
    assert recorded["pid"] == os.getpid()


def _race_acquire(lock_path_str: str, barrier, result_queue) -> None:
    # Runs in its own spawned process (see test below): a fresh import here,
    # not anything inherited from the parent, so os.getpid() genuinely
    # differs from every other racer and from the parent test process.
    from pressurekeeper.instance_lock import InstanceAlreadyRunning, SingleInstanceLock

    barrier.wait()
    try:
        SingleInstanceLock(Path(lock_path_str)).acquire()
        result_queue.put("ACQUIRED")
    except InstanceAlreadyRunning:
        result_queue.put("REJECTED")


def test_acquire_is_atomic_across_concurrently_racing_processes(tmp_path):
    """Regression for the exists()-then-write_text() TOCTOU: the old acquire()
    let two processes that both observed "no lock file yet" go on to both
    write, so both came away believing they held it. A `multiprocessing.Barrier`
    holds every racer immediately before its `acquire()` call so they all fire
    within a sub-millisecond window of each other (ordinary process-launch
    jitter alone is far too coarse to reliably reproduce this TOCTOU), then
    checks that exactly one of them ends up believing it holds a lock path
    that did not exist beforehand.
    """
    import multiprocessing

    lock_path = tmp_path / "race_lock"
    ctx = multiprocessing.get_context("spawn")
    n = 6
    barrier = ctx.Barrier(n)
    result_queue = ctx.Queue()
    procs = [ctx.Process(target=_race_acquire, args=(str(lock_path), barrier, result_queue)) for _ in range(n)]
    for p in procs:
        p.start()
    try:
        results = [result_queue.get(timeout=15) for _ in range(n)]
    finally:
        for p in procs:
            p.join(timeout=15)

    assert results.count("ACQUIRED") == 1, f"expected exactly one winner, got {results}"
    assert results.count("REJECTED") == n - 1, results


def test_release_does_not_delete_a_lock_taken_over_by_someone_else(tmp_path):
    lock_path = tmp_path / "lock"
    lock = SingleInstanceLock(lock_path)
    lock.acquire()
    # Simulate another process having since taken over the (now-stale) lock.
    lock_path.write_text(json.dumps({"pid": os.getpid() + 1, "acquired_at_wall": 0.0}))
    lock.release()
    assert lock_path.exists(), "release() must not remove a lock it no longer owns"
