"""SingleInstanceLock: a best-effort, cross-platform guard against starting a
second PressureKeeper process against the same PACE5000 control app.

Not a substitute for real exclusivity enforced by the PACE5000 control app
itself -- that would need an owner lease on the API side, which lives in a
separate repository (PaceMaker) and is out of this repo's scope (see
CLAUDE.md's repo-boundary notes). This only catches an operator accidentally
starting a second instance *on the same machine*, which is exactly the
failure mode that left several completely empty log directories after a real
mDAC session: four short-lived process starts overlapped a still-running
instance and produced zero ticks each, with nothing on disk to explain why
(see logging_sink.py's manifest.json for the other half of that fix).

Deliberately PID-liveness-based rather than an OS advisory lock
(fcntl.flock/msvcrt.locking): the real hardware in this project's own
deployment runs on Windows, and a liveness check is what lets a lock left
behind by a killed/crashed process be told apart from one still legitimately
held, without platform-specific lock APIs on both sides.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


class InstanceAlreadyRunning(RuntimeError):
    pass


def default_lock_dir() -> Path:
    """Fixed, per-user location for instance lock files.

    Deliberately independent of any run's `logging.directory`: keying the
    lock file's location off the log directory (as this used to) meant two
    PressureKeeper processes launched with different `--config` files (and
    thus different log directories) but the *same* PACE5000 endpoint held
    separate locks and never collided -- exactly the failure mode this
    module's own docstring describes (several empty log directories from
    overlapping instances). `lock_path_for()` below still keys the actual
    filename only on the normalized endpoint, so any two processes pointed
    at the same PACE5000 always land on the same lock file regardless of
    where either one logs to.
    """
    return Path.home() / ".pressurekeeper" / "locks"


def lock_path_for(base_url: str, lock_dir: str | Path | None = None) -> Path:
    parts = urlsplit(base_url)
    host = (parts.hostname or "unknown").replace(":", "_")
    port = parts.port or 0
    directory = Path(lock_dir) if lock_dir is not None else default_lock_dir()
    return directory / f".pk_lock_{host}_{port}"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(process_query_limited_information, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just owned by someone else
    except OSError:
        return False
    return True


@dataclass
class SingleInstanceLock:
    path: Path

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"pid": os.getpid(), "acquired_at_wall": time.time()}).encode("utf-8")
        # O_CREAT | O_EXCL is atomic on both POSIX and Windows: at most one of
        # two processes racing to create the same path can win this call, no
        # matter how closely their calls overlap. The previous
        # exists()-then-write_text() sequence was two separate syscalls with
        # an unbounded window between them, in which two processes launched
        # together could both observe "does not exist" and both go on to
        # write -- exactly the double-acquire this lock exists to prevent.
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            self._take_over_if_stale(payload)
            return
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)

    def _take_over_if_stale(self, payload: bytes) -> None:
        """Reached only when the exclusive create above lost the race to an
        already-existing lock file -- decide whether that file represents a
        live foreign process (refuse) or something safe to overwrite (a
        stale lock from a dead pid, our own pid re-acquiring, or a
        genuinely unreadable/corrupt file). This overwrite itself is not
        exclusive, but by this point the file already exists, so it can no
        longer be a *different* live process's exclusive-create winning at
        the same moment -- only a genuinely dead/foreign/corrupt lock
        reaches here, once `_read_settled()` has ruled out reading the
        eventual winner of that create mid-write.
        """
        existing_pid = -1
        raw = self._read_settled()
        if raw:
            try:
                existing = json.loads(raw)
                existing_pid = int(existing.get("pid", -1))
            except (ValueError, TypeError):
                pass  # non-empty but unparseable -- genuinely corrupt, safe to take over
        if existing_pid > 0 and existing_pid != os.getpid() and _pid_alive(existing_pid):
            raise InstanceAlreadyRunning(
                f"another PressureKeeper process (pid {existing_pid}) appears to already "
                f"be running against this PACE5000 endpoint (lock file: {self.path}). "
                "Close that instance first. If it has actually already exited without "
                "cleaning up, delete this file and retry."
            )
        self.path.write_bytes(payload)

    def _read_settled(self) -> bytes:
        """Read the lock file's content, tolerating the brief window between
        another process's O_CREAT|O_EXCL winning the create in `acquire()`
        and that process's payload write actually landing.

        Without this, a loser that loses the exclusive-create race can read
        the file while it is still empty (0 bytes, between the winner's
        `os.open()` and `os.write()`), conclude "unreadable/corrupt, safe to
        take over", and overwrite the winner's just-created lock with its
        own -- reproducibly observed under real multi-process contention
        (see test_instance_lock.py's
        test_acquire_is_atomic_across_concurrently_racing_processes), not
        just a theoretical concern. A single small `os.write()` call is
        atomic with respect to concurrent readers on POSIX (no torn reads),
        so any *non-empty* read here is always the winner's complete,
        parseable payload -- only a read that stays empty for the whole
        retry budget is treated as genuinely stale/corrupt.
        """
        for _ in range(20):
            try:
                data = self.path.read_bytes()
            except OSError:
                return b""
            if data:
                return data
            time.sleep(0.005)
        return b""

    def release(self) -> None:
        try:
            existing = json.loads(self.path.read_text(encoding="utf-8"))
            if int(existing.get("pid", -1)) != os.getpid():
                return  # not ours (e.g. already taken over by a later process) -- leave it alone
        except (OSError, ValueError, TypeError):
            return
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass

    def __enter__(self) -> "SingleInstanceLock":
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()
