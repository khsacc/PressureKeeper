"""Interactive CLI: runs the control loop in a background thread at the
ruby-API polling cadence, and lets an operator watch state/key values and
issue pause/resume/abort/target commands from the terminal while it runs.
"""
from __future__ import annotations

import argparse
import select
import sys
import threading
import time
from pathlib import Path

from .app import build_app
from .config import load_config
from .models import ControllerSnapshot
from .sim import DACPhysicsConfig

COMMANDS_HELP = """\
Commands:
  target <GPa>   set a new target sample pressure (GPa)
  pause          request manual pause (blocks new pressurization)
  resume         clear a manual pause
  abort          request abort (sticky; requires 'reset' to clear)
  reset          recover from ABORT; stays Measure until a safe setpoint is confirmed
  status         print an immediate full status line
  help           show this message
  quit           stop the control loop and exit
"""


def _fmt(x: float | None, spec: str = "{:.4f}") -> str:
    return spec.format(x) if x is not None else "n/a"


def _format_snapshot(snap: ControllerSnapshot) -> str:
    reasons = ",".join(snap.safety_reasons)
    return (
        f"[{snap.state.value:<8}] "
        f"target={_fmt(snap.user_target_gpa, '{:.3f}')}GPa "
        f"filtered={_fmt(snap.filtered_pressure_gpa)}GPa "
        f"slope={_fmt(snap.pressure_slope_gpa_s, '{:.5f}')}GPa/s "
        f"pred={_fmt(snap.predicted_pressure_gpa)}GPa "
        f"membrane={_fmt(snap.membrane_setpoint_mpa, '{:.4f}')}/{_fmt(snap.membrane_actual_mpa, '{:.4f}')}MPa(set/act) "
        f"supply={_fmt(snap.source_pressure_positive_mpa, '{:.4f}')}MPa "
        f"safe_gain={_fmt(snap.safe_gain)} "
        f"valid={snap.estimator_valid} "
        f"safety={snap.safety_level}{f'[{reasons}]' if reasons else ''}"
        f"{' MANUAL-PAUSE' if snap.manual_pause else ''}"
        f"{f' LOGGING-ERROR[{snap.logging_error}]' if snap.logging_error else ''}"
    )


class _ControllerRunner(threading.Thread):
    def __init__(self, controller, poll_interval_s: float, stop_event: threading.Event) -> None:
        super().__init__(daemon=True)
        self._controller = controller
        self._poll_interval_s = poll_interval_s
        self._stop_event = stop_event
        self._lock = threading.Lock()
        self.latest_snapshot: ControllerSnapshot | None = None
        self.last_error: BaseException | None = None
        self.crashed = False

    def snapshot(self) -> ControllerSnapshot | None:
        with self._lock:
            return self.latest_snapshot

    def run(self) -> None:
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                snap = self._controller.step()
            except Exception as e:  # fail safe: latch abort, log, stop the loop
                self.last_error = e
                self.crashed = True
                try:
                    self._controller.abort(f"controller loop crashed: {e!r}")
                    snap = self._controller.step()
                except Exception:
                    snap = None
                if snap is not None:
                    with self._lock:
                        self.latest_snapshot = snap
                self._stop_event.set()
                break
            with self._lock:
                self.latest_snapshot = snap
            elapsed = time.monotonic() - t0
            remaining = self._poll_interval_s - elapsed
            if remaining > 0:
                self._stop_event.wait(remaining)


def _handle_command(line: str, controller, runner: "_ControllerRunner", stop_event: threading.Event) -> None:
    line = line.strip()
    if not line:
        return
    parts = line.split()
    cmd = parts[0].lower()
    try:
        if cmd == "target" and len(parts) == 2:
            value = float(parts[1])
            controller.set_target(value)
            print(f"\ntarget set to {value:.4f} GPa")
        elif cmd == "pause":
            controller.pause("operator CLI pause")
            print("\npause requested")
        elif cmd == "resume":
            controller.resume()
            print("\nresume requested")
        elif cmd == "abort":
            controller.abort("operator CLI abort")
            print("\nABORT requested")
        elif cmd == "reset":
            controller.reset()
            print("\ncontroller reset (-> APPROACH in Measure; safe setpoint will be recalculated before Control)")
        elif cmd == "status":
            snap = runner.snapshot()
            if snap is None:
                print("\nno status yet (control loop hasn't completed a tick)")
            else:
                print("\n" + _format_snapshot(snap))
        elif cmd == "help":
            print(COMMANDS_HELP)
        elif cmd in ("quit", "exit"):
            stop_event.set()
        else:
            print(f"\nunrecognized command: {line!r} (type 'help')")
    except ValueError as e:
        print(f"\ninvalid command {line!r}: {e}")


def _interactive_loop(controller, runner: _ControllerRunner, stop_event: threading.Event) -> None:
    last_draw = 0.0
    while not stop_event.is_set():
        ready, _, _ = select.select([sys.stdin], [], [], 0.5)
        if ready:
            line = sys.stdin.readline()
            if line == "":  # EOF
                break
            _handle_command(line, controller, runner, stop_event)
        snap = runner.snapshot()
        now = time.monotonic()
        if snap is not None and now - last_draw >= 0.5:
            print("\r" + _format_snapshot(snap) + "    ", end="", flush=True)
            last_draw = now
        if runner.crashed:
            print(f"\n[FATAL] control loop crashed: {runner.last_error!r}")
            break
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pressurekeeper",
        description="One-sided predictive pressure controller for a membrane-driven DAC.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML or TOML configuration file")
    parser.add_argument("--target", type=float, default=None, help="Initial target sample pressure, in GPa")
    parser.add_argument("--sim", action="store_true", help="Use the built-in simulator instead of real devices")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for --sim")
    dry_run_group = parser.add_mutually_exclusive_group()
    dry_run_group.add_argument("--dry-run", dest="dry_run", action="store_const", const=True, default=None,
                                help="Force-disable real PACE5000 writes (overrides config)")
    dry_run_group.add_argument("--live", dest="dry_run", action="store_const", const=False,
                                help="Enable real PACE5000 writes — DANGEROUS: this moves the real membrane")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    sim_physics = DACPhysicsConfig(seed=args.seed) if args.sim else None
    ctx = build_app(config, use_simulator=args.sim, dry_run=args.dry_run, sim_physics=sim_physics)

    if args.target is not None:
        ctx.controller.set_target(args.target)

    effective_dry_run = args.dry_run if args.dry_run is not None else config.control.dry_run
    if not args.sim and not effective_dry_run:
        print("!!! LIVE MODE: PACE5000 writes are ENABLED. This will move the real membrane. !!!")
    mode = "SIMULATOR" if args.sim else ("DRY-RUN" if effective_dry_run else "LIVE")
    if ctx.logger is None:
        print(f"[WARNING] logging is disabled: {ctx.controller.logging_error}")
        log_location = "DISABLED"
    else:
        log_location = str(ctx.logger.directory)
    print(f"PressureKeeper starting in {mode} mode. Logs: {log_location}")
    print(COMMANDS_HELP)

    # loop_min_interval_s is a floor on how often the control loop iterates,
    # independent of (and possibly slower than) the ruby API's own polling
    # cadence.
    loop_interval_s = max(config.ruby_api.poll_interval_s, config.control.loop_min_interval_s)
    stop_event = threading.Event()
    runner = _ControllerRunner(ctx.controller, loop_interval_s, stop_event)
    runner.start()

    try:
        _interactive_loop(ctx.controller, runner, stop_event)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        # Stop the PACE5000 before closing anything: quit/Ctrl-C/a crashed
        # loop must not leave the membrane ramping toward its last setpoint
        # after this process exits.
        if not ctx.controller.stop_and_confirm():
            print("\n[WARNING] could not confirm the PACE5000 stopped before exiting -- "
                  "check it manually.")
        # A tick may still be completing bounded device timeouts.  Closing the
        # HTTP sessions/logger while that daemon thread is live creates a
        # use-after-close race and can lose the final safety records.
        runner.join()
        ctx.close()
    return 1 if runner.crashed else 0


if __name__ == "__main__":
    raise SystemExit(main())
